# Phase 5: Results & Error Handling - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver complete result downloads and comprehensive error handling across the API. Users can download all job outputs as a single ZIP, and all failure scenarios return structured, actionable error responses with clear HTTP status codes.

</domain>

<decisions>
## Implementation Decisions

### Download Responses
- Single ZIP file packaging for all job results
- ZIP contains ALL outputs: OBJ, PLY, GLB, textures, all preview images, quality.json
- Quality report embedded in ZIP only (no separate endpoint)
- Partial success = job failure (no partial downloads allowed)

### Error Message Format
- Structured JSON format: `{"error": {"code": "...", "message": "...", "details": {...}}}`
- Human-readable string error codes (VALIDATION_FAILED, MODEL_OOM, FILE_TOO_LARGE)
- Include actionable suggestions for all errors ("suggestion" field with fix hints)
- No request ID or trace ID — keep responses simple

### Failure Granularity
- Detailed breakdown for model failures (include pipeline stage, model name)
- Distinguish GPU VRAM vs system RAM for OOM errors
- Per-file validation errors (list each failing file with specific reason)
- Quality threshold failures show actual vs expected values (e.g., "PSNR 18.5dB below threshold 20dB")

### Claude's Discretion
- HTTP status code selection (400 vs 422 for validation, 202 vs 409 for not-ready, 503 vs 507 for resources, 404 vs 410 for expired)
- Exact error code taxonomy
- Suggestion wording for each error type
- ZIP file internal structure/naming

</decisions>

<specifics>
## Specific Ideas

- Error messages should be specific enough for debugging without exposing internal system paths
- Quality failures should include both PSNR and SSIM values so users understand which metric failed
- OOM errors should mention the limit hit (e.g., "16GB VRAM limit") when possible

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-results-error-handling*
*Context gathered: 2026-01-31*
