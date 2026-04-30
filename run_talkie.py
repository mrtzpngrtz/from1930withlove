import sys
import argparse

import torch
import torch.nn as nn
import bitsandbytes as bnb

# Import talkie modules BEFORE patching so their global namespaces exist.
import talkie.generate as _talkie_generate
import talkie.model as _talkie_model
from talkie.model import GPTConfig, TalkieModel, resize_model_embeddings

MODEL_ID = "talkie-1930-13b-it"
VRAM_LIMIT_BYTES = 14 * 1024**3


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
                weight_data, requires_grad=False, compress_statistics=True, quant_type="nf4",
            )
            del weight_data
            if child.bias is not None:
                layer.bias = nn.Parameter(child.bias.data)
                child.bias = None
            setattr(model, name, layer)
        else:
            _apply_4bit(child)
    return model


def _load_checkpoint_4bit(
    checkpoint_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
) -> TalkieModel:
    import gc
    print("  reading checkpoint (mmap)...", file=sys.stderr)
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

    # Stream mmap fp32 → bf16 one tensor at a time. Peak: ~13 GB.
    print("  streaming mmap → bf16 (peak ~13 GB)...", file=sys.stderr)
    bf16_sd = {}
    for k, v in sd.items():
        bf16_sd[k] = v.to(dtype=torch.bfloat16)
    del ckpt, sd
    gc.collect()

    print("  building model (meta + assign)...", file=sys.stderr)
    try:
        with torch.device("meta"):
            model = TalkieModel(config, cpu)
        model.load_state_dict(bf16_sd, strict=True, assign=True)
    except Exception as exc:
        print(f"  meta device failed ({exc}), fallback...", file=sys.stderr)
        model = TalkieModel(config, cpu)
        model.load_state_dict(bf16_sd, strict=True, assign=True)
    del bf16_sd
    gc.collect()

    meta_buf_names = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_buf_names:
        print(f"  recomputing {len(meta_buf_names)} non-checkpoint buffer(s)...", file=sys.stderr)
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

    print("  NF4 quantization (in-place)...", file=sys.stderr)
    model = _apply_4bit(model)
    gc.collect()

    print("  moving to GPU...", file=sys.stderr)
    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    _trim_ram()
    model.device = device
    model.eval()
    return model


# Patch the reference that Talkie.__init__ actually calls.
_talkie_generate.load_checkpoint = _load_checkpoint_4bit
_talkie_model.load_checkpoint = _load_checkpoint_4bit

from talkie.generate import Talkie  # noqa: E402 — must come after patching


def main():
    parser = argparse.ArgumentParser(
        description="Run talkie-1930-13b-it with NF4 4-bit quantization on RTX 4090"
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("Loading model...", file=sys.stderr)
    talkie = Talkie(MODEL_ID)

    vram = torch.cuda.memory_allocated()
    print(f"VRAM after load: {vram / 1024**3:.2f} GB", file=sys.stderr)
    if vram > VRAM_LIMIT_BYTES:
        raise RuntimeError(
            f"VRAM {vram / 1024**3:.2f} GB exceeds 14 GB limit — aborting rather than CPU-offloading."
        )

    if args.smoke_test:
        print("--- smoke test ---", file=sys.stderr)
        out = ""
        for chunk in talkie.stream(
            "Hello, who are you?", max_tokens=100, temperature=0.7, top_p=0.9
        ):
            print(chunk, end="", flush=True)
            out += chunk
        print()
        if not out.strip():
            raise RuntimeError("Smoke test failed: empty output")
        print("--- smoke test passed ---", file=sys.stderr)
        return

    if args.prompt is None:
        parser.error("--prompt required unless --smoke-test is set")

    for chunk in talkie.stream(
        args.prompt,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
