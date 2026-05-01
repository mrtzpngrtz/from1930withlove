---
title: From 1930, with Love
emoji: 📻
colorFrom: gray
colorTo: black
sdk: docker
pinned: false
---

# From 1930, with Love

A chat interface for [Talkie 1930-13B-IT](https://huggingface.co/talkie-lm) — a language model trained exclusively on pre-1931 text. Knowledge cutoff: 1930.

Runs locally on a single RTX 4090 with NF4 4-bit quantization (7.4 GB VRAM). Responses stream token-by-token over SSE.

## What this is

A FastAPI server that loads the Talkie 13B model with bitsandbytes NF4 4-bit quantization and serves it through a minimal chat interface. The frontend is a single HTML file with no build step.

## Requirements

- Windows with CUDA (driver ≥525 for CUDA 12.6)
- GPU with ≥10 GB VRAM (tested on RTX 4090)
- Python 3.11+

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Web server (http://localhost:8000)
start.bat

# Or directly
.venv\Scripts\python server.py

# CLI
.venv\Scripts\python run_talkie.py --prompt "Describe the moral character of the radio."
.venv\Scripts\python run_talkie.py --smoke-test
```

## Memory

| Mode | VRAM |
|------|------|
| bf16 (original) | ~28 GB |
| NF4 4-bit | ~7–8 GB |

Peak RAM during load: ~13 GB (fp32→bf16 streaming via mmap, then quantized in-place).

## Architecture

- `server.py` — FastAPI + SSE streaming, async queue for concurrent users
- `run_talkie.py` — CLI wrapper with same quantization path
- `chat.html` — single-file frontend, no dependencies

## Links

- [Talkie on HuggingFace](https://huggingface.co/talkie-lm)
- [Introducing Talkie](https://talkie-lm.com/introducing-talkie)
