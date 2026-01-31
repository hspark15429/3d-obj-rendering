# Phase 3: Model Integration - Research

**Researched:** 2026-01-31
**Domain:** 3D reconstruction model inference with PyTorch and CUDA
**Confidence:** MEDIUM

## Summary

Phase 3 integrates two neural 3D reconstruction models (ReconViaGen and nvdiffrec) into the existing Celery task pipeline. Both models use PyTorch with CUDA 11.8 and require careful VRAM management for sequential execution on a single RTX 3090 (24GB).

The standard approach uses PyTorch3D for mesh I/O (OBJ/PLY export with textures), torch.cuda memory management for sequential model runs, and tempfile module for intermediate file cleanup. Critical challenges include OOM prevention, proper VRAM cleanup between models, and handling model failures gracefully.

ReconViaGen's official code is not yet released (stuck in company review), so implementation must use either the Hugging Face Space demo API or unofficial implementations. Nvdiffrec is fully open source but designed for training workflows, requiring adaptation for inference-only use.

**Primary recommendation:** Use PyTorch3D for mesh export, implement explicit VRAM cleanup (del + gc.collect() + torch.cuda.empty_cache()) between sequential models, pre-download model weights in Docker image, and set conservative timeouts (30-60 minutes per model based on iteration counts).

## Standard Stack

The established libraries/tools for PyTorch-based 3D reconstruction:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 1.10+ | Deep learning framework | Required by both models, CUDA 11.8 compatibility |
| PyTorch3D | Latest | 3D deep learning primitives | Facebook's official library for mesh I/O, texture handling |
| nvdiffrast | Latest | Differentiable rasterizer | Required dependency for nvdiffrec |
| trimesh | Latest | Mesh processing | Fallback for mesh manipulation if PyTorch3D limitations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tempfile | stdlib | Temporary file management | Intermediate files during model processing |
| gc | stdlib | Garbage collection | VRAM cleanup between sequential runs |
| filetype | 1.2.0+ | File validation | Already in project, validate output meshes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch3D | trimesh + PyMeshLab | More mature mesh I/O but less integration with PyTorch tensors |
| Pre-download weights | Runtime download | Faster startup but larger Docker image (~5-10GB increase) |
| Sequential execution | Multi-GPU parallel | Better throughput but requires 2x GPUs, violates single RTX 3090 constraint |

**Installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch3d
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install trimesh
```

## Architecture Patterns

### Recommended Project Structure
```
app/
├── tasks/
│   └── reconstruction.py     # Main task with model selection logic
├── models/
│   ├── __init__.py          # Model registry/factory
│   ├── reconviagen.py       # ReconViaGen wrapper
│   ├── nvdiffrec.py         # nvdiffrec wrapper
│   └── base.py              # Abstract base class for models
├── services/
│   ├── file_handler.py      # Already exists, extend for mesh I/O
│   └── vram_manager.py      # VRAM cleanup utilities
└── outputs/
    └── {job_id}/
        ├── reconviagen/     # Model-specific output folder
        │   ├── mesh.obj
        │   ├── mesh.mtl
        │   ├── texture.png
        │   └── mesh.ply
        └── nvdiffrec/       # Model-specific output folder
            ├── mesh.obj
            ├── mesh.mtl
            ├── texture.png
            └── mesh.ply
```

### Pattern 1: Sequential Model Execution with VRAM Cleanup
**What:** Execute models sequentially with explicit GPU memory cleanup between runs
**When to use:** Running multiple models on same GPU (the locked decision for this phase)
**Example:**
```python
# Source: PyTorch forums, GeeksforGeeks CUDA memory management 2025-2026
import gc
import torch

def run_models_sequentially(job_id: str, model_selection: str):
    """Run one or both models with proper VRAM cleanup."""

    if model_selection in ['reconviagen', 'both']:
        # Run ReconViaGen
        model = load_reconviagen_model()
        result = model.inference(job_id)

        # CRITICAL: Explicit cleanup before next model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if not result.success:
            return result  # Stop on first failure (locked decision)

    if model_selection in ['nvdiffrec', 'both']:
        # Run nvdiffrec
        model = load_nvdiffrec_model()
        result = model.inference(job_id)

        # Cleanup after completion
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return result
```

### Pattern 2: Model Wrapper with Progress Reporting
**What:** Wrap each model with standardized interface for progress tracking
**When to use:** Integrating models with different APIs into unified task system
**Example:**
```python
# Source: Celery best practices for GPU tasks, Towards Data Science 2024-2026
from abc import ABC, abstractmethod

class BaseReconstructionModel(ABC):
    """Abstract base for reconstruction models."""

    def __init__(self, celery_task=None):
        self.celery_task = celery_task

    @abstractmethod
    def load_weights(self):
        """Load pre-downloaded model weights."""
        pass

    @abstractmethod
    def inference(self, input_path: str, output_path: str):
        """Run inference and return mesh paths."""
        pass

    def report_progress(self, progress: int, step: str):
        """Report progress to Celery."""
        if self.celery_task:
            self.celery_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'step': f'{self.model_name}: {step}'
                }
            )
```

### Pattern 3: Mesh Export with Textures
**What:** Export PyTorch tensors to OBJ/PLY with separate texture files
**When to use:** Converting model output to standard formats (locked requirement)
**Example:**
```python
# Source: PyTorch3D documentation - io module
from pytorch3d.io import save_obj, save_ply
import torch

def export_mesh_with_texture(verts, faces, texture_map, uvs, output_dir):
    """Export mesh in both OBJ and PLY formats."""

    # OBJ with texture support
    save_obj(
        f=output_dir / "mesh.obj",
        verts=verts,
        faces=faces,
        verts_uvs=uvs,
        faces_uvs=faces,  # Assuming 1:1 mapping
        texture_map=texture_map,  # FloatTensor (H, W, 3) in [0, 1]
        decimal_places=6
    )

    # PLY for compatibility
    save_ply(
        f=output_dir / "mesh.ply",
        verts=verts,
        faces=faces,
        ascii=False,  # Binary for smaller files
        decimal_places=6
    )

    # Texture saved separately as PNG by save_obj automatically
```

### Pattern 4: Intermediate File Cleanup
**What:** Use tempfile context managers for intermediate processing files
**When to use:** Model generates intermediate outputs before final mesh (locked decision: delete intermediates)
**Example:**
```python
# Source: Python tempfile module documentation 2026
from pathlib import Path
import tempfile
import shutil

def process_with_cleanup(job_id: str):
    """Process with automatic cleanup of intermediate files."""

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as temp_dir:
        temp_path = Path(temp_dir)

        # Model writes intermediate files here
        model_output = run_model(input_path, temp_path)

        # Extract only final outputs
        final_mesh = extract_final_mesh(temp_path)
        final_texture = extract_final_texture(temp_path)

        # Copy to permanent location
        shutil.copy(final_mesh, output_dir / "mesh.obj")
        shutil.copy(final_texture, output_dir / "texture.png")

        # temp_dir automatically deleted on context exit
```

### Anti-Patterns to Avoid
- **torch.cuda.empty_cache() without del**: Only frees unused cached memory, not tensors still referenced
- **No gc.collect() before empty_cache()**: Python garbage collector may not have run yet
- **Keeping model in memory**: Loading both models simultaneously will OOM on 24GB VRAM
- **Runtime model downloads**: Network failures during inference, unpredictable startup times
- **Generic exception handling**: OOM errors need specific handling, not generic try/except

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OBJ/PLY export with textures | Custom file writer | PyTorch3D save_obj/save_ply | Handles UV mapping, MTL files, vertex normals correctly |
| VRAM tracking | Manual nvidia-smi parsing | torch.cuda.memory_allocated() | Real-time, per-process, handles fragmentation |
| Temporary file management | Manual os.remove() calls | tempfile.TemporaryDirectory | Automatic cleanup, exception-safe, cross-platform |
| Mesh validation | Custom geometry checks | trimesh.is_watertight() | Handles edge cases (non-manifold, self-intersections) |
| Model weight downloading | wget/curl in Dockerfile | torch.hub / huggingface_hub | Handles checksums, retries, caching |

**Key insight:** 3D mesh I/O has many edge cases (texture coordinates, face winding, normal consistency). PyTorch3D and trimesh solve these correctly. Custom implementations will have bugs.

## Common Pitfalls

### Pitfall 1: Incomplete VRAM Cleanup Between Models
**What goes wrong:** Second model in sequence fails with OOM despite first model completing successfully
**Why it happens:** PyTorch caches memory aggressively; Python GC doesn't run immediately; model references persist
**How to avoid:** Three-step cleanup: `del model` → `gc.collect()` → `torch.cuda.empty_cache()`
**Warning signs:** torch.cuda.memory_allocated() shows high usage after model completion

### Pitfall 2: Assuming Training Code Works for Inference
**What goes wrong:** nvdiffrec expects training loop (2000 iterations), not single-shot inference
**Why it happens:** Both models designed for training/optimization, not direct inference
**How to avoid:** Adapt training scripts to run fixed iterations with pre-trained weights, or use demo code from Hugging Face Spaces
**Warning signs:** Code requires dataset loaders, optimizer configuration

### Pitfall 3: Missing Model Weight Pre-download Validation
**What goes wrong:** Container starts but model initialization fails at runtime with missing weights
**Why it happens:** Docker build succeeded but weight download failed silently, or wrong CUDA version
**How to avoid:** Add weight validation step in Dockerfile (load model, check output shape), fail build if invalid
**Warning signs:** Fast Docker build (weights not actually downloaded), runtime "checkpoint not found" errors

### Pitfall 4: OOM on First Model (Not Just Sequential)
**What goes wrong:** Even single model run causes OOM
**Why it happens:** Model default resolution too high for 24GB, batch size > 1, or memory fragmentation
**How to avoid:** Use model default resolutions (don't upscale), batch_size=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
**Warning signs:** OOM during model initialization (before inference loop starts)

### Pitfall 5: Texture File Not Referenced in MTL
**What goes wrong:** OBJ loads but shows white mesh (no texture)
**Why it happens:** PyTorch3D save_obj may not create MTL file correctly, or texture path wrong
**How to avoid:** Verify MTL file exists, contains "map_Kd texture.png", and texture.png in same directory
**Warning signs:** Mesh renders but appears untextured in Blender/MeshLab

### Pitfall 6: File Cleanup Before Result Retrieval
**What goes wrong:** Intermediate files deleted before final outputs extracted
**Why it happens:** tempfile.TemporaryDirectory deletes on __exit__, even if exception occurs
**How to avoid:** Copy final outputs to permanent location BEFORE exiting context, or use try/finally
**Warning signs:** Job succeeds but output directory empty

### Pitfall 7: No Timeout on Model Inference
**What goes wrong:** Model hangs indefinitely on bad input, blocking worker
**Why it happens:** Some inputs cause convergence issues, infinite loops, or deadlocks
**How to avoid:** Set task-level timeout in Celery (soft_time_limit=3600, time_limit=3900 for 1-hour timeout)
**Warning signs:** Worker stuck on single task, no progress updates for >30 minutes

## Code Examples

Verified patterns from official sources:

### VRAM Cleanup Pattern
```python
# Source: PyTorch forums, GeeksforGeeks 2025-2026
# Pattern verified by multiple sources for sequential model execution

import gc
import torch

def cleanup_gpu_memory():
    """
    Complete GPU memory cleanup pattern.

    Order matters:
    1. Delete Python references to GPU objects
    2. Run garbage collector to free Python objects
    3. Clear PyTorch's caching allocator

    Note: ~254MB baseline usage remains (PyTorch + CUDA overhead)
    """
    # Delete model/tensor references in calling scope
    # (caller must use: del model, del tensors, etc.)

    # Force Python garbage collection
    gc.collect()

    # Release unused cached memory
    torch.cuda.empty_cache()

    # Optional: Check remaining usage
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3     # GB
    print(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### PyTorch3D Mesh Export
```python
# Source: PyTorch3D documentation - io module
from pytorch3d.io import save_obj, save_ply
from pathlib import Path

def save_mesh_both_formats(verts, faces, texture_map, verts_uvs, output_dir: Path):
    """
    Save mesh in both OBJ and PLY formats with texture.

    Args:
        verts: FloatTensor (V, 3) - vertex positions
        faces: LongTensor (F, 3) - face indices
        texture_map: FloatTensor (H, W, 3) in range [0, 1] - RGB texture
        verts_uvs: FloatTensor (V, 2) - UV coordinates per vertex
        output_dir: Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # OBJ with texture and MTL file
    save_obj(
        f=str(output_dir / "mesh.obj"),
        verts=verts,
        faces=faces,
        verts_uvs=verts_uvs,
        faces_uvs=faces,  # Assumes 1:1 vertex-to-UV mapping
        texture_map=texture_map,
        decimal_places=6
    )
    # Creates: mesh.obj, mesh.mtl, texture.png automatically

    # PLY format (no texture in PLY via PyTorch3D)
    save_ply(
        f=str(output_dir / "mesh.ply"),
        verts=verts,
        faces=faces,
        ascii=False,  # Binary for smaller file size
        decimal_places=6
    )
```

### Model Timeout Configuration
```python
# Source: Celery documentation + GPU task best practices
from celery import shared_task

@shared_task(
    bind=True,
    name="reconstruction.process",
    soft_time_limit=3600,      # 1 hour soft limit (raises SoftTimeLimitExceeded)
    time_limit=3900,           # 1 hour 5 min hard limit (kills task)
    acks_late=True,            # Don't acknowledge until task completes
    reject_on_worker_lost=True # Requeue if worker crashes
)
def process_reconstruction(self, job_id: str, model_type: str):
    """
    Process with timeout protection.

    Soft limit: Task can catch exception and cleanup
    Hard limit: Task killed forcefully (5 min grace period)
    """
    try:
        # Model inference here
        pass
    except SoftTimeLimitExceeded:
        # Graceful cleanup
        cleanup_gpu_memory()
        delete_job_files(job_id)
        return {"status": "timeout", "job_id": job_id}
```

### Temporary Directory Pattern
```python
# Source: Python tempfile documentation 2026
import tempfile
from pathlib import Path
import shutil

def process_model_with_intermediates(job_id: str, input_path: Path, output_path: Path):
    """
    Process model with automatic intermediate file cleanup.

    Uses tempfile.TemporaryDirectory as context manager for automatic cleanup.
    Only final outputs are copied to permanent location.
    """
    with tempfile.TemporaryDirectory(prefix=f"recon_{job_id}_") as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Model writes intermediate files to temp_path
            # (e.g., per-iteration checkpoints, debug images, cached tensors)
            intermediate_results = run_model_inference(input_path, temp_path)

            # Extract only final mesh and texture
            final_mesh = temp_path / "final_mesh.obj"
            final_texture = temp_path / "final_texture.png"

            if not final_mesh.exists():
                raise ValueError("Model did not produce final mesh")

            # Copy to permanent output directory
            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(final_mesh, output_path / "mesh.obj")
            shutil.copy(final_texture, output_path / "texture.png")

            return {"status": "success", "output_path": str(output_path)}

        except Exception as e:
            # Even on exception, temp_dir will be cleaned up
            raise

        # temp_dir and all contents automatically deleted here
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NeRF implicit representations | Explicit mesh with DMTet | 2022 (nvdiffrec) | Direct mesh output, faster rendering |
| Single-view reconstruction | Multi-view with depth | 2024-2025 | Higher accuracy, complete geometry |
| Runtime weight download | Docker image pre-download | Best practice 2025+ | Reliable cold start, no network dependency |
| Manual memory cleanup | torch.cuda.empty_cache() | PyTorch 1.0+ (2018) | Still requires del + gc.collect() |
| Synchronous API | Async job queue | Industry standard | Required for >30s inference times |

**Deprecated/outdated:**
- **@app.on_event**: Deprecated FastAPI pattern, use lifespan context manager (already followed in Phase 1)
- **Pydantic Config class**: Use model_config instead (already followed in Phase 2)
- **torch.cuda.reset_max_memory_allocated()**: Doesn't free memory, only resets counters (confusion source)

## Open Questions

Things that couldn't be fully resolved:

1. **ReconViaGen Official Code Availability**
   - What we know: Official code stuck in company review, alpha demo on Hugging Face Spaces exists
   - What's unclear: Timeline for official release, whether unofficial implementation is production-ready
   - Recommendation: Start with Hugging Face Spaces API or unofficial implementation, plan to swap in official code when released. Document assumption in code.

2. **Exact Iteration Counts for Inference**
   - What we know: nvdiffrec uses 2000 iterations for training, "high quality after 1000 iterations"
   - What's unclear: Can iterations be reduced for inference-only? What's minimum acceptable quality?
   - Recommendation: Start with 1000 iterations (balance quality/speed), make configurable for tuning. Timeout at 60 minutes.

3. **VRAM Usage for Specific Models**
   - What we know: nvdiffrec designed for 32GB+ GPUs, can run on 24GB with reduced batch size. ReconViaGen based on TRELLIS (16-24GB requirement)
   - What's unclear: Exact VRAM usage with our specific input (6 views, 2048x2048 RGB, 1024x1024 depth)
   - Recommendation: Test empirically, add VRAM monitoring, fail fast with clear error if approaching 90% usage (21.6GB)

4. **Texture Quality vs Resolution Tradeoff**
   - What we know: nvdiffrec default is 2048x2048 texture, can be reduced for VRAM
   - What's unclear: Minimum acceptable texture resolution for visual quality
   - Recommendation: Use model defaults (locked decision: no user control), document in code comments

5. **Handling Partial Model Outputs**
   - What we know: Models might produce mesh but fail texture baking
   - What's unclear: Should we return untextured mesh or fail completely?
   - Recommendation: Fail completely (simpler error handling), log specifically "texture generation failed"

## Sources

### Primary (HIGH confidence)
- PyTorch3D io module documentation: https://pytorch3d.readthedocs.io/en/latest/modules/io.html - mesh I/O APIs verified
- Python tempfile documentation: https://docs.python.org/3/library/tempfile.html - temporary file best practices
- NVlabs/nvdiffrec GitHub: https://github.com/NVlabs/nvdiffrec - official implementation, requirements, configuration
- PyTorch CUDA memory management: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html - memory APIs

### Secondary (MEDIUM confidence)
- PyTorch forums VRAM cleanup: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530 - community-verified del + gc + empty_cache pattern
- GeeksforGeeks CUDA memory clearing: https://www.geeksforgeeks.org/deep-learning/clearing-gpu-memory-after-pytorch-training-without-kernel-restart/ - 2025 article confirming pattern
- Towards Data Science Celery GPU tasks: https://towardsdatascience.com/deploying-ml-models-in-production-with-fastapi-and-celery-7063e539a5db/ - FastAPI + Celery + ML pattern
- Medium CUDA memory management: https://medium.com/@soumensardarintmain/manage-cuda-cores-ultimate-memory-management-strategy-with-pytorch-2bed30cab1 - strategy guide

### Tertiary (LOW confidence)
- ReconViaGen GitHub: https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen - official repo but code not released, alpha demo status
- ReconViaGen HuggingFace Space: https://huggingface.co/spaces/Stable-X/ReconViaGen - demo available, API not documented
- TRELLIS VRAM requirements: https://github.com/microsoft/TRELLIS/issues/31 - community reports 12-24GB, ReconViaGen based on TRELLIS
- Best-of-web nvdiffrec overview: https://best-of-web.builder.io/library/NVlabs/nvdiffrec - 2025 summary, not official docs

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - PyTorch3D verified, but model-specific libraries unclear due to ReconViaGen code unavailability
- Architecture: HIGH - Patterns verified from official PyTorch docs and community best practices
- Pitfalls: MEDIUM - VRAM cleanup verified, but model-specific issues require empirical testing

**Research date:** 2026-01-31
**Valid until:** 2026-02-14 (14 days - fast-moving AI/ML domain)

**Critical dependencies on locked decisions:**
- Sequential execution (not parallel) → requires VRAM cleanup pattern
- Both OBJ and PLY output → PyTorch3D save_obj + save_ply
- Delete intermediate files → tempfile.TemporaryDirectory pattern
- Pre-downloaded weights → Dockerfile RUN commands, not runtime
- Model selection parameter → factory pattern with three modes

**What to validate during execution:**
1. VRAM usage under load (empirical test with actual models)
2. ReconViaGen integration approach (unofficial vs HF Space API vs wait for official)
3. Iteration counts for acceptable quality vs timeout balance
4. Texture export works correctly (MTL file references, file paths)
5. Sequential cleanup prevents OOM (monitor between models)
