# Phase 3: Model Integration - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Execute ReconViaGen and nvdiffrec reconstruction models to produce textured mesh outputs. Model weights are pre-downloaded in Docker image. Both models can run sequentially without VRAM overflow. Quality metrics and preview generation are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Model selection behavior
- Required parameter: user must explicitly choose 'reconviagen', 'nvdiffrec', or 'both'
- 'both' option runs ReconViaGen then nvdiffrec sequentially
- Per-model progress reporting: show which model is running and its individual progress
- Stop on first failure: if first model fails when running 'both', don't attempt second

### Output file structure
- Both OBJ and PLY formats generated for each model
- Model subfolders: `reconviagen/` and `nvdiffrec/` each with their outputs
- Textures as separate files (texture.png) referenced by MTL, not embedded
- Delete intermediate files after completion (only keep final outputs)

### Inference configuration
- Fixed resolution: use model defaults, no user control
- Fixed texture resolution: use model native output
- Hidden iteration count: use optimal settings internally
- Simple API: model + input images only, no other parameters

### Failure handling
- OOM: fail immediately with clear error message, no retry
- Corrupted/empty output: fail the job with 'invalid output' error, no retry
- Fixed timeout: use sensible default internally (not user-configurable)
- User-friendly error messages only: simple messages like 'Model failed to process images'

### Claude's Discretion
- Exact timeout values per model
- VRAM management between sequential model runs
- Internal retry logic for transient errors (file I/O, etc.)
- Log verbosity and format

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-model-integration*
*Context gathered: 2026-01-31*
