---
phase: 04-quality-preview
plan: 01
subsystem: quality
tags: [scikit-image, psnr, ssim, metrics, depth-error]

# Dependency graph
requires:
  - phase: 03.1-cuda12-upgrade
    provides: Model integration with mesh output
provides:
  - compute_psnr() for PSNR quality metric
  - compute_ssim() for SSIM quality metric
  - compute_depth_error() for depth MAE/RMSE
  - classify_quality_status() for quality classification
affects: [04-02, 04-03, quality-report, preview-generation]

# Tech tracking
tech-stack:
  added: [scikit-image>=0.26.0]
  patterns: [quality-thresholds-hardcoded, float-image-normalization, valid-mask-depth-comparison]

key-files:
  created:
    - app/services/quality_metrics.py
  modified:
    - requirements.txt

key-decisions:
  - "PSNR thresholds: Normal>=25dB, Warning>=20dB, Failure<20dB"
  - "SSIM thresholds: Normal>=0.85, Warning>=0.75, Failure<0.75"
  - "Both metrics must pass for status level (AND logic)"
  - "data_range=1.0 always for float images in [0,1] range"
  - "channel_axis=-1 for RGB multi-channel SSIM"
  - "Depth comparison uses valid_mask excluding zero pixels"

patterns-established:
  - "Float image loading: PIL RGB convert, float32 / 255.0"
  - "Quality classification: hardcoded thresholds, no runtime config"
  - "Depth metrics: MAE + RMSE with valid_mask for non-zero pixels"

# Metrics
duration: 3min
completed: 2026-01-31
---

# Phase 4 Plan 01: Quality Metrics Service Summary

**PSNR/SSIM/depth-error quality metrics using scikit-image with hardcoded Normal/Warning/Failure thresholds**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-31T10:14:00Z
- **Completed:** 2026-01-31T10:17:05Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added scikit-image dependency for industry-standard PSNR/SSIM computation
- Created quality_metrics.py with 6 functions (285 lines)
- Implemented PSNR/SSIM with correct data_range and channel_axis parameters
- Implemented depth error metrics with valid_mask for background handling
- Hardcoded quality thresholds matching user decisions in CONTEXT.md

## Task Commits

Each task was committed atomically:

1. **Task 1: Add scikit-image dependency** - `16613fb` (chore)
2. **Task 2: Create quality_metrics.py service** - `ca2beb2` (feat)

## Files Created/Modified
- `requirements.txt` - Added scikit-image>=0.26.0 dependency
- `app/services/quality_metrics.py` - Quality metrics computation service with PSNR, SSIM, depth error, and status classification

## Decisions Made
- Used scikit-image over IQA-PyTorch (sufficient for PSNR/SSIM, simpler dependency)
- Hardcoded thresholds per user decision (no runtime configuration)
- SSIM thresholds set to 0.85/0.75 to complement PSNR thresholds (from research recommendations)
- Both metrics must meet threshold for status level (AND logic prevents high PSNR/low SSIM from being marked normal)
- Always specify data_range=1.0 for float images (scikit-image defaults incorrectly assume [-1,1] for floats)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - scikit-image installed successfully and all imports work.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- quality_metrics.py ready for integration with preview generation (04-02)
- Functions can be imported by other services: `from app.services.quality_metrics import compute_psnr, compute_ssim`
- Thresholds exported as constants for use in quality report generation

---
*Phase: 04-quality-preview*
*Completed: 2026-01-31*
