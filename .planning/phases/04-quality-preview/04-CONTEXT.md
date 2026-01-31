# Phase 4: Quality & Preview - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Compute reconstruction quality metrics (PSNR, SSIM) and generate preview images for completed jobs. Classify results by quality status. Does NOT include download endpoints or error handling (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Quality Thresholds
- Moderate thresholds: Normal ≥25dB, Warning 20-25dB, Failure <20dB (PSNR)
- Hardcoded defaults — no runtime configuration
- Always save mesh regardless of quality status — user decides what to do with low-quality results
- Overall metrics only — single PSNR/SSIM score averaged across all views, no per-view breakdown

### Preview Images
- Both textured and wireframe renderings — two sets of preview images
- Resolution matches input image resolution — consistent with user's data
- PNG format — lossless compression for sharp edges
- Claude's discretion on number of views and angles

### Quality Report Format
- Standalone quality.json file in job output directory
- Include human-readable summary text alongside raw metrics
- Standard metadata: job ID, model used, timestamps, input file info, processing duration
- No recommendations — just report status, user interprets

### Comparison Reference
- Compare rendered output against original input views
- Use exact same camera poses as input for precise comparison
- Include depth comparison — compare rendered depth to input depth for geometry validation
- Quality metrics are required — if rendering fails, job fails (no partial success)

### Claude's Discretion
- Number of preview angles (4, 6, or 8 views)
- Exact SSIM thresholds (should complement PSNR thresholds)
- Depth comparison metric choice (MAE, RMSE, or similar)
- Wireframe rendering style (line width, color)

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

*Phase: 04-quality-preview*
*Context gathered: 2026-01-31*
