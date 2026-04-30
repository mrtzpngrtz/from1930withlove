"""Chat server — talkie-1930-13b-it, NF4 4-bit, SSE streaming."""
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import bitsandbytes as bnb

import talkie.generate as _talkie_generate
import talkie.model as _talkie_model
from talkie.model import GPTConfig, TalkieModel, resize_model_embeddings


def _trim_ram() -> None:
    import ctypes
    try:
        ctypes.windll.psapi.EmptyWorkingSet(
            ctypes.windll.kernel32.GetCurrentProcess()
        )
    except Exception:
        pass


def _apply_4bit(model: nn.Module) -> nn.Module:
    """Weight-steal bf16 → NF4 in place (model is already bf16 before this runs)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            weight_data = child.weight.data.contiguous()
            child.weight = None
            layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )
            layer.weight = bnb.nn.Params4bit(
                weight_data,
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )
            del weight_data
            if child.bias is not None:
                layer.bias = nn.Parameter(child.bias.data)
                child.bias = None
            setattr(model, name, layer)
        else:
            _apply_4bit(child)
    return model


def _load_checkpoint_4bit(checkpoint_path, device, target_vocab_size=None):
    import gc
    print("  loading checkpoint (mmap)...", file=sys.stderr, flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    ckpt_vocab_size = sd["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=ckpt_vocab_size)
    cpu = torch.device("cpu")

    # Stream mmap fp32 → bf16 one tensor at a time.
    # Peak: ~13 GB (half of fp32). Never allocates a 26 GB fp32 model in RAM.
    print("  streaming mmap → bf16 (peak ~13 GB)...", file=sys.stderr, flush=True)
    bf16_sd = {}
    for k, v in sd.items():
        bf16_sd[k] = v.to(dtype=torch.bfloat16)
    del ckpt, sd
    gc.collect()

    # Build model structure on meta device (zero RAM), then assign bf16 tensors
    # directly via assign=True — no copy, model takes ownership of bf16_sd tensors.
    print("  building model (meta + assign)...", file=sys.stderr, flush=True)
    try:
        with torch.device("meta"):
            model = TalkieModel(config, cpu)
        model.load_state_dict(bf16_sd, strict=True, assign=True)
    except Exception as exc:
        print(f"  meta device failed ({exc}), using fallback...", file=sys.stderr, flush=True)
        model = TalkieModel(config, cpu)
        model.load_state_dict(bf16_sd, strict=True, assign=True)
    del bf16_sd
    gc.collect()

    # Buffers computed in __init__ (e.g. RoPE freqs) aren't in the checkpoint,
    # so they remain as meta tensors after assign=True. Recompute them from a
    # fresh reference model (peak = 13 GB bf16 + 26 GB fp32 ref = 39 GB; ref freed immediately).
    meta_buf_names = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_buf_names:
        print(f"  recomputing {len(meta_buf_names)} non-checkpoint buffer(s)...", file=sys.stderr, flush=True)
        ref = TalkieModel(config, cpu)
        ref_bufs = dict(ref.named_buffers())
        for bname in meta_buf_names:
            *parents, leaf = bname.split(".")
            m = model
            for p in parents:
                m = getattr(m, p)
            if bname in ref_bufs:
                m.register_buffer(leaf, ref_bufs[bname].clone())
        del ref, ref_bufs
        gc.collect()

    if target_vocab_size is not None and ckpt_vocab_size < target_vocab_size:
        model = resize_model_embeddings(model, target_vocab_size, cpu)

    print("  NF4 quantization (in-place)...", file=sys.stderr, flush=True)
    model = _apply_4bit(model)
    gc.collect()

    print("  moving to GPU...", file=sys.stderr, flush=True)
    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    _trim_ram()

    model.device = device
    model.eval()
    return model


_talkie_generate.load_checkpoint = _load_checkpoint_4bit
_talkie_model.load_checkpoint = _load_checkpoint_4bit

from talkie.generate import Talkie  # noqa: E402

import asyncio
import threading
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

MAX_PROMPT_CHARS = 1500
MAX_NEW_TOKENS   = 512
MIN_TEMPERATURE  = 0.05
MAX_TEMPERATURE  = 2.0

app = FastAPI()
_talkie: Talkie | None = None

# --- queue state (mutated only under _q_lock) ---
_waiters: list[asyncio.Event] = []
_q_lock  = asyncio.Lock()


async def _enqueue() -> asyncio.Event:
    async with _q_lock:
        ev = asyncio.Event()
        _waiters.append(ev)
        if len(_waiters) == 1:
            ev.set()          # first in line — go immediately
        return ev


async def _dequeue(ev: asyncio.Event) -> None:
    async with _q_lock:
        try:
            _waiters.remove(ev)
        except ValueError:
            pass
        if _waiters:
            _waiters[0].set() # wake next waiter


def _queue_pos(ev: asyncio.Event) -> int:
    try:
        return _waiters.index(ev) + 1
    except ValueError:
        return 0


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return resp


@app.on_event("startup")
async def startup():
    import gc
    global _talkie
    print("Loading model...", file=sys.stderr, flush=True)
    _talkie = Talkie("talkie-1930-13b-it")
    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated()
    ram = torch.cuda.memory_reserved()
    print(f"VRAM: {vram / 1024**3:.2f} GB — ready.", file=sys.stderr, flush=True)


@app.get("/")
async def index():
    return HTMLResponse(Path("chat.html").read_text(encoding="utf-8"))


@app.get("/stream")
async def stream(
    prompt: str = Query(..., max_length=MAX_PROMPT_CHARS),
    max_tokens: int = Query(512),
    temperature: float = Query(0.7),
    top_p: float = Query(0.9),
):
    max_tokens  = max(1,    min(max_tokens,  MAX_NEW_TOKENS))
    temperature = max(MIN_TEMPERATURE, min(temperature, MAX_TEMPERATURE))
    top_p       = max(0.01, min(top_p, 1.0))
    prompt      = prompt[:MAX_PROMPT_CHARS]

    loop = asyncio.get_event_loop()
    ev   = await _enqueue()

    async def generate():
        try:
            # --- wait phase: send queue position every second ---
            while not ev.is_set():
                pos = _queue_pos(ev)
                yield f"data: {json.dumps({'queue': pos})}\n\n"
                await asyncio.sleep(1.0)

            # --- generation phase: run model in thread, stream chunks ---
            chunk_q: asyncio.Queue = asyncio.Queue()

            def run():
                try:
                    for chunk in _talkie.stream(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    ):
                        loop.call_soon_threadsafe(chunk_q.put_nowait, {"t": chunk})
                except Exception as exc:
                    loop.call_soon_threadsafe(chunk_q.put_nowait, {"error": str(exc)})
                finally:
                    loop.call_soon_threadsafe(chunk_q.put_nowait, None)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            while True:
                item = await chunk_q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

            thread.join()

        finally:
            await _dequeue(ev)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
