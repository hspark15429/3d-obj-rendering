# Phase 2: Job Pipeline - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Async job queue infrastructure for long-running inference. Users submit multi-view images + depth data, receive a job ID, check status via polling, and can cancel jobs. Jobs process asynchronously via Celery + Redis. Downloading results and error handling details are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Upload structure
- Single multipart POST request for all files (6 views + depth renders)
- Strict validation — reject invalid uploads upfront with clear errors
- File size limits: per-file AND total limits enforced
- PNG format only — no JPEG or other formats accepted

### Job status flow
- Polling only — no webhooks
- Four statuses: queued, processing, completed, failed (plus cancelled)
- Show percentage estimate during processing
- Completed jobs persist until manually deleted (no auto-expiry)

### Response format
- Short random job IDs (human-friendly, e.g., "abc123xy")
- Flat JSON responses (no wrapper objects)
- ISO 8601 timestamps (e.g., 2026-01-31T14:30:00Z)

### Cancellation behavior
- Cancelled jobs show distinct "cancelled" status (not failed)
- Cancellation requires confirmation (two-step: request → confirm)
- Files deleted immediately on cancellation (inputs and any partial outputs)

### Claude's Discretion
- Error response detail level (code + message, with field-level info as needed)
- Cancel window (queued only vs queued + processing) based on complexity
- Exact short ID format and length
- Percentage estimate granularity and update frequency

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for Celery/Redis queue patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-job-pipeline*
*Context gathered: 2026-01-31*
