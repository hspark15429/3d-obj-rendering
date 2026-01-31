---
created: 2026-01-31T04:15
title: Integrate real reconstruction models (ReconViaGen + nvdiffrec)
area: models
files:
  - app/models/reconviagen.py
  - app/models/nvdiffrec.py
  - Dockerfile
  - requirements.txt
---

## Problem

Current model wrappers use STUB implementations that produce placeholder meshes. Real model integration blocked by dependency conflicts:

**ReconViaGen (estheryang11/ReconViaGen):**
- Requires CUDA 12.0 (we have 11.8)
- Requires PyTorch 2.4.0 (we have 2.1.0)
- Heavy dependencies: spconv-cu120, xformers, flash_attn, MASt3R
- Uses HuggingFace models: esther11/trellis-vggt-v0-2, naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

**nvdiffrec (NVlabs/nvdiffrec):**
- Compatible with current setup (CUDA 11.3+, PyTorch 1.10+)
- Optimization-based: ~1000 iterations per input
- Could be integrated now, but keeping as STUB for consistency

## Solution

This is to be fixed by:

1. **Upgrade infrastructure to CUDA 12 + PyTorch 2.4** - enables ReconViaGen but requires testing all dependencies

Key integration steps when ready:
- Update Dockerfile base image to CUDA 12
- Update PyTorch to 2.4.0
- Add spconv, xformers, flash_attn dependencies
- Download model weights in Docker build
- Replace STUB inference code with real model calls
