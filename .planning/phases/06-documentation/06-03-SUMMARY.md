---
phase: 06-documentation
plan: 03
subsystem: documentation
tags: [examples, samples, input-format, output-format, user-guide]

# Dependency graph
requires:
  - phase: 06-01
    provides: Core documentation (architecture.md, API.md)
provides:
  - examples/input/ with symlinked sample files
  - examples/input/README.md documenting 6-view input format
  - examples/output/README.md documenting output structure and quality metrics
affects: [users, testing, onboarding]

# Tech tracking
tech-stack:
  added: []
  patterns: [symlinks-for-sample-data]

key-files:
  created:
    - examples/input/README.md
    - examples/output/README.md
    - examples/input/multi_views (symlink)
    - examples/input/depth_renders (symlink)
  modified: []

key-decisions:
  - "Symlink sample files instead of copying to avoid duplicating ~20MB of PNG data"
  - "Document expected output format instead of generating real outputs (GPU required)"

patterns-established:
  - "Symlinks for sample data: keeps repo size small while providing accessible examples"
  - "Comprehensive format documentation: users understand requirements before first API call"

# Metrics
duration: 2min
completed: 2026-01-31
---

# Phase 06 Plan 03: Examples Directory Summary

**Sample input files with format documentation and comprehensive output structure reference**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-31T12:55:15Z
- **Completed:** 2026-01-31T12:57:09Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created examples/input/ directory with symlinks to existing sample files
- Documented 6-view input format requirements (camera setup, resolution, naming)
- Documented complete output structure (GLB/OBJ formats, previews, quality.json)
- Included usage examples for curl, Blender, Three.js, and Python

## Task Commits

Each task was committed atomically:

1. **Task 1: Create examples/input/ with sample files and README** - `cbb92d3` (docs)
2. **Task 2: Create examples/output/ with format documentation** - `c7be4ce` (docs)

**Plan metadata:** (to be committed after SUMMARY)

## Files Created/Modified

- `examples/input/README.md` - 63 lines documenting 6-view input format, camera setup, and API usage
- `examples/input/multi_views` - Symlink to `../../input/multi_views/` (6 PNG files at 2048×2048)
- `examples/input/depth_renders` - Symlink to `../../input/depth_renders/` (6 depth PNGs at 1024×1024)
- `examples/output/README.md` - 189 lines documenting output structure, quality metrics, file formats, and usage

## Decisions Made

**1. Symlink sample files instead of copying**
- Avoids duplicating ~20MB of PNG data in repository
- Keeps examples/ directory lightweight
- Sample files remain accessible via relative symlinks

**2. Document output format instead of generating samples**
- Real outputs require GPU runtime (not available in all environments)
- Documentation provides complete reference without dependency on execution
- Users understand expected output before submitting first job

**3. Comprehensive quality.json documentation**
- Explained PSNR/SSIM metrics with thresholds
- Documented status classification logic (AND logic for normal/warning/failure)
- Clarified that quality failure = job failure per requirement

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 06-04:** Main README creation
- Sample files accessible for testing examples
- Format documentation complete for reference links
- Input/output requirements clearly documented

**Blockers:** None

**Notes:**
- Users can immediately test with provided samples
- Documentation covers all output formats and quality assessment
- Examples directory provides self-contained testing resource

---
*Phase: 06-documentation*
*Completed: 2026-01-31*
