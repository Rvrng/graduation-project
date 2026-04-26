import asyncio
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid
from typing import Any, Deque, Dict, Iterable, Optional

from fastapi import Request

from gpustack.runtime_state.types import RequestAppMeta


WINDOW_10_SECONDS = 10
WINDOW_30_SECONDS = 30


@dataclass
class _TrackedRequest:
    """
    Internal in-memory representation of one request.

    This structure is intentionally richer than RequestAppMeta because we need to
    capture lifecycle information after the request is routed to an instance and
    after the response finishes.
    """

    request_id: str
    model_name: str
    task_type: str
    slo_class: str
    estimated_prompt_tokens: int
    estimated_output_tokens: int
    stream: bool
    started_at: datetime
    model_id: Optional[int] = None
    worker_id: Optional[int] = None
    instance_id: Optional[int] = None
    ttft_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None
    finished_at: Optional[datetime] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    ttft_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    ok: Optional[bool] = None

    @property
    def inflight(self) -> bool:
        return self.finished_at is None

    @property
    def effective_prompt_tokens(self) -> int:
        """
        Prefer true usage when it is available and fall back to the request-time
        estimate while the request is still running.
        """

        if self.prompt_tokens is not None:
            return self.prompt_tokens
        return self.estimated_prompt_tokens


_lock = asyncio.Lock()
_requests_by_id: Dict[str, _TrackedRequest] = {}
_model_start_events: Deque[tuple[datetime, str]] = deque()
_instance_bound_events: Deque[tuple[datetime, int]] = deque()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_type_from_path(path: str) -> str:
    """
    Convert request path to the task type used by the lightweight scheduler state.
    """

    normalized = path.lower()
    if normalized.endswith("/chat/completions"):
        return "chat_completion"
    if normalized.endswith("/completions"):
        return "completion"
    if normalized.endswith("/embeddings"):
        return "embedding"
    if normalized.endswith("/rerank"):
        return "rerank"
    if normalized.endswith("/images/generations") or normalized.endswith("/images/edits"):
        return "image_generation"
    if normalized.endswith("/audio/speech"):
        return "audio_speech"
    if normalized.endswith("/audio/transcriptions"):
        return "audio_transcription"
    return "unknown"


def _default_slo_for_task(task_type: str) -> str:
    """
    Provide a conservative default when the client does not send explicit SLO hints.
    """

    if task_type in {"chat_completion", "completion"}:
        return "latency"
    if task_type in {"embedding", "rerank"}:
        return "throughput"
    return "standard"


def _estimate_tokens_from_text(text: str) -> int:
    """
    Fast heuristic token estimation used only for scheduling hints.

    We do not want to add heavy tokenizer dependencies in this lightweight module.
    A character-based heuristic is sufficient for relative comparisons between
    requests at scheduling time.
    """

    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_tokens_from_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _estimate_tokens_from_text(value)
    if isinstance(value, list):
        return sum(_estimate_tokens_from_value(item) for item in value)
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            if key in {"role", "name", "type"}:
                continue
            total += _estimate_tokens_from_value(item)
        return total
    return _estimate_tokens_from_text(str(value))


def _estimate_prompt_tokens(task_type: str, body_json: Optional[dict]) -> int:
    """
    Extract prompt/context size from the request body.

    The goal here is not exact billing-grade token counting. We only need a stable
    relative estimate that the scheduler can compare across requests.
    """

    if not body_json:
        return 0

    if task_type == "chat_completion":
        return _estimate_tokens_from_value(body_json.get("messages", []))
    if task_type == "completion":
        return _estimate_tokens_from_value(body_json.get("prompt"))
    if task_type == "embedding":
        return _estimate_tokens_from_value(body_json.get("input"))
    if task_type == "rerank":
        return _estimate_tokens_from_value(body_json.get("query")) + _estimate_tokens_from_value(
            body_json.get("documents", [])
        )
    return 0


def _estimate_output_tokens(body_json: Optional[dict]) -> int:
    if not body_json:
        return 0

    for key in ("max_completion_tokens", "max_tokens"):
        value = body_json.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _prune_old_events(now: datetime):
    """
    Keep only the recent events that are useful for 10s / 30s scheduling windows.
    """

    cutoff = now - timedelta(seconds=WINDOW_30_SECONDS)

    while _model_start_events and _model_start_events[0][0] < cutoff:
        _model_start_events.popleft()

    while _instance_bound_events and _instance_bound_events[0][0] < cutoff:
        _instance_bound_events.popleft()


def extract_request_meta(
    request: Request,
    model_name: str,
    body_json: Optional[dict] = None,
    form_data: Optional[Any] = None,
    stream: bool = False,
) -> RequestAppMeta:
    """
    Convert the incoming request into the minimal application-layer metadata
    needed by the scheduler.

    Notes:
    - body_json is used for text-heavy requests and yields the best context estimate.
    - multipart/form-data requests are usually image/audio related here; the first
      version leaves their context estimate as zero because it is rarely a useful
      routing signal compared with text generation requests.
    """

    del form_data  # Explicitly unused in the first lightweight implementation.

    task_type = _task_type_from_path(request.url.path)
    slo_class = request.headers.get("X-GPUStack-SLO-Class") or _default_slo_for_task(
        task_type
    )

    return RequestAppMeta(
        model_name=model_name,
        task_type=task_type,
        slo_class=slo_class,
        ttft_slo_ms=_parse_float(request.headers.get("X-GPUStack-TTFT-SLO-MS")),
        tpot_slo_ms=_parse_float(request.headers.get("X-GPUStack-TPOT-SLO-MS")),
        estimated_prompt_tokens=_estimate_prompt_tokens(task_type, body_json),
        estimated_output_tokens=_estimate_output_tokens(body_json),
        stream=bool(stream),
    )


async def request_started(meta: RequestAppMeta) -> str:
    """
    Register a new request as soon as it reaches the server-side scheduling point.
    """

    now = _now()
    request_id = meta.request_id or uuid.uuid4().hex
    meta.request_id = request_id
    meta.started_at = now

    tracked = _TrackedRequest(
        request_id=request_id,
        model_name=meta.model_name,
        task_type=meta.task_type,
        slo_class=meta.slo_class,
        estimated_prompt_tokens=meta.estimated_prompt_tokens,
        estimated_output_tokens=meta.estimated_output_tokens,
        stream=meta.stream,
        started_at=now,
        model_id=meta.model_id,
        ttft_slo_ms=meta.ttft_slo_ms,
        tpot_slo_ms=meta.tpot_slo_ms,
    )

    async with _lock:
        _requests_by_id[request_id] = tracked
        _model_start_events.append((now, meta.model_name))
        _prune_old_events(now)

    return request_id


async def request_bound(
    request_id: str,
    model_id: int,
    instance_id: int,
    worker_id: int,
) -> None:
    """
    Register the selected runtime target for a request.

    This is what enables instance-level QPS and inflight statistics.
    """

    now = _now()
    async with _lock:
        tracked = _requests_by_id.get(request_id)
        if tracked is None:
            return

        tracked.model_id = model_id
        tracked.instance_id = instance_id
        tracked.worker_id = worker_id
        _instance_bound_events.append((now, instance_id))
        _prune_old_events(now)


async def request_finished(
    request_id: Optional[str],
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    ttft_ms: Optional[float] = None,
    tokens_per_second: Optional[float] = None,
    ok: bool = True,
) -> None:
    """
    Mark a request as finished and backfill the best runtime information available.

    The function is intentionally idempotent so the middleware can safely call it
    more than once in streaming scenarios.
    """

    if not request_id:
        return

    async with _lock:
        tracked = _requests_by_id.get(request_id)
        if tracked is None or tracked.finished_at is not None:
            return

        tracked.prompt_tokens = prompt_tokens
        tracked.completion_tokens = completion_tokens
        tracked.ttft_ms = ttft_ms
        tracked.tokens_per_second = tokens_per_second
        tracked.ok = ok
        tracked.finished_at = _now()


def _recent_requests(
    seconds: int,
    *,
    model_name: Optional[str] = None,
    model_id: Optional[int] = None,
    instance_id: Optional[int] = None,
) -> Iterable[_TrackedRequest]:
    cutoff = _now() - timedelta(seconds=seconds)

    for tracked in _requests_by_id.values():
        if tracked.started_at < cutoff:
            continue
        if model_name is not None and tracked.model_name != model_name:
            continue
        if model_id is not None and tracked.model_id != model_id:
            continue
        if instance_id is not None and tracked.instance_id != instance_id:
            continue
        yield tracked


async def get_model_runtime_metrics(
    model_id: int,
    model_name: str,
) -> dict:
    """
    Return model-level short-window application metrics used by the scheduler.
    """

    async with _lock:
        _prune_old_events(_now())

        qps_10s = (
            sum(
                1
                for ts, event_model_name in _model_start_events
                if event_model_name == model_name
                and ts >= _now() - timedelta(seconds=WINDOW_10_SECONDS)
            )
            / WINDOW_10_SECONDS
        )

        inflight = sum(
            1
            for tracked in _requests_by_id.values()
            if tracked.model_id == model_id and tracked.inflight
        )

        recent = list(_recent_requests(WINDOW_10_SECONDS, model_name=model_name))
        avg_prompt_tokens = (
            sum(item.effective_prompt_tokens for item in recent) / len(recent)
            if recent
            else 0.0
        )

        task_type_mix = dict(Counter(item.task_type for item in recent))
        slo_mix = dict(Counter(item.slo_class for item in recent))

    return {
        "qps_10s": qps_10s,
        "inflight": inflight,
        "avg_prompt_tokens_10s": avg_prompt_tokens,
        "task_type_mix": task_type_mix,
        "slo_mix": slo_mix,
    }


async def get_instance_runtime_metrics(instance_id: int) -> dict:
    """
    Return instance-level short-window application metrics used by the scheduler.
    """

    async with _lock:
        _prune_old_events(_now())

        qps_10s = (
            sum(
                1
                for ts, event_instance_id in _instance_bound_events
                if event_instance_id == instance_id
                and ts >= _now() - timedelta(seconds=WINDOW_10_SECONDS)
            )
            / WINDOW_10_SECONDS
        )

        inflight = sum(
            1
            for tracked in _requests_by_id.values()
            if tracked.instance_id == instance_id and tracked.inflight
        )

        recent = list(_recent_requests(WINDOW_30_SECONDS, instance_id=instance_id))
        prompt_samples = [item.effective_prompt_tokens for item in recent]
        ttft_samples = [
            item.ttft_ms
            for item in recent
            if item.ttft_ms is not None and item.finished_at is not None
        ]

    return {
        "qps_10s": qps_10s,
        "inflight": inflight,
        "recent_avg_prompt_tokens": (
            sum(prompt_samples) / len(prompt_samples) if prompt_samples else 0.0
        ),
        "recent_avg_ttft_ms": (
            sum(ttft_samples) / len(ttft_samples) if ttft_samples else 0.0
        ),
    }
