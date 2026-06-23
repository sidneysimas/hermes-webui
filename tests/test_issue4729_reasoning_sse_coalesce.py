"""Regression test for #4729 — reasoning SSE coalescing throttle.

The bug: during the reasoning/thinking phase of models like DeepSeek the server
emitted one SSE `reasoning` event per token (tens of thousands per turn), each
triggering a full-text scan in the frontend renderer and freezing the JS main thread.

The fix throttles reasoning SSE events to ~10 Hz. The SUBTLE correctness requirement
(the reason the first attempt was bounced): reasoning deltas are INCREMENTAL and the
frontend APPENDS them (`reasoningText += text` in static/messages.js), so the throttle
must COALESCE — accumulate dropped deltas into a buffer and flush the buffer, NOT drop
deltas — otherwise live reasoning text is permanently lost. And the tail (the last
sub-100ms window) must be flushed when the reasoning phase ends, or it's lost too.

These are source-structure assertions on the on_reasoning closure in api/streaming.py
(the closure isn't unit-testable in isolation), pinning the three properties so the
coalescing contract can't silently regress to the drop-based version.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).parent.parent
STREAMING = (REPO / "api" / "streaming.py").read_text(encoding="utf-8")
MESSAGES = (REPO / "static" / "messages.js").read_text(encoding="utf-8")


def _on_reasoning_body() -> str:
    """Extract the body of the on_reasoning closure by brace/indent scanning."""
    start = STREAMING.index("def on_reasoning(text):")
    # grab a generous slice (the closure is short) bounded by the next top-level
    # `def on_` callback in the same scope.
    nxt = STREAMING.find("\n            def on_", start + 1)
    return STREAMING[start: nxt if nxt != -1 else start + 2500]


def test_reasoning_uses_coalescing_buffer_not_drop():
    body = _on_reasoning_body()
    # A coalescing buffer accumulates every delta...
    assert "_reasoning_buffer[0] += reasoning_delta" in body, (
        "on_reasoning must ACCUMULATE each delta into the coalescing buffer — dropping "
        "deltas would permanently lose live reasoning text (the frontend appends them)"
    )
    # ...and the throttled flush emits the WHOLE buffer, then CLEARS it (so the next
    # emit carries only the since-last-flush accumulation — coarse delta, not cumulative).
    assert "put('reasoning', {'text': _reasoning_buffer[0]})" in body, (
        "the throttled flush must emit the accumulated buffer, not a single delta"
    )
    # the clear must follow the flush so we don't re-send already-delivered text
    flush_idx = body.index("put('reasoning', {'text': _reasoning_buffer[0]})")
    clear_idx = body.index("_reasoning_buffer[0] = ''", flush_idx)
    assert clear_idx > flush_idx, "the buffer must be cleared right after each flush"


def test_reasoning_throttle_is_rate_limited():
    body = _on_reasoning_body()
    # ~10 Hz gate (0.1s) on the flush
    assert "_reasoning_last_put" in body and "0.1" in body, (
        "reasoning flush must be rate-limited to ~10 Hz (0.1s gate)"
    )


def test_reasoning_tail_flushed_on_phase_end():
    body = _on_reasoning_body()
    # When the reasoning phase ends (text is None), any remaining buffered text must be
    # flushed so the last partial (<100ms) window isn't lost.
    none_branch = body[body.index("if text is None:"): body.index("if text is None:") + 400]
    assert "_reasoning_buffer[0]" in none_branch and "put('reasoning'" in none_branch, (
        "on_reasoning(text=None) must flush any remaining coalesced buffer (the tail)"
    )


def test_frontend_appends_reasoning_deltas():
    # The whole coalesce requirement hinges on the frontend APPENDING (not replacing).
    # If this ever changes to assignment, the throttle design must change with it.
    assert "reasoningText += text" in MESSAGES, (
        "frontend reasoning handler must append deltas — if this changes, revisit the "
        "server-side coalescing throttle (#4729)"
    )
