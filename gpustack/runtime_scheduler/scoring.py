from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from gpustack.runtime_scheduler.deployer import (
    DeploymentHint,
    InstanceDeploymentMetrics,
    ModelDeploymentMetrics,
    WorkerDeploymentMetrics,
)
from gpustack.runtime_scheduler.pareto import candidate_vram_usage_rate

if TYPE_CHECKING:
    from gpustack.runtime_scheduler.request_scheduler import RequestScheduleCandidate
    from gpustack.runtime_state.types import RequestAppMeta


INTERACTIVE_GENERATION_TASKS = {"chat_completion", "completion"}
VECTOR_THROUGHPUT_TASKS = {"embedding", "rerank"}
HEAVY_MULTIMEDIA_TASKS = {
    "image_generation",
    "audio_speech",
    "audio_transcription",
}

SHORT_CONTEXT_TOKENS = 1024
MEDIUM_CONTEXT_TOKENS = 8192
TTFT_SLO_APPROACH_RATIO = 0.90
TTFT_OBSERVATION_COMPENSATION_FACTOR = 0.85
VRAM_SCALE_OUT_WATERMARK = 0.90


@dataclass(frozen=True)
class RequestScheduleScore:
    candidate: RequestScheduleCandidate
    score: float
    task_group: str
    context_bucket: str
    slo_class: str
    deployment_hint: Optional[DeploymentHint] = None


@dataclass(frozen=True)
class _CandidateFeatures:
    candidate: RequestScheduleCandidate
    vram_usage_rate: Optional[float]
    vram_balance_score: float
    vram_headroom_score: float
    heavy_vram_score: float
    qps_per_inflight: float
    qps_per_inflight_score: float
    inflight: int
    inflight_score: float
    ttft_ms: Optional[float]
    ttft_score: float
    resident_seconds: float
    residency_score: float


class RequestScoreSelector:
    def score_candidates(
        self,
        candidates: Sequence[RequestScheduleCandidate],
        current_request: RequestAppMeta,
    ) -> List[RequestScheduleScore]:
        features = [_build_features(candidate) for candidate in candidates]
        features = _normalize_relative_features(features)
        task_group = classify_task_group(current_request.task_type)
        context_bucket = classify_context_bucket(current_request)
        slo_class = current_request.slo_class or "standard"
        weights = _weights_for(task_group, context_bucket, slo_class)
        hint = _build_deployment_hint(features, current_request, task_group)

        scores = []
        for item in features:
            score = _weighted_score(item, weights)
            scores.append(
                RequestScheduleScore(
                    candidate=item.candidate,
                    score=score,
                    task_group=task_group,
                    context_bucket=context_bucket,
                    slo_class=slo_class,
                    deployment_hint=hint,
                )
            )
        return scores

    def select(
        self,
        candidates: Sequence[RequestScheduleCandidate],
        current_request: RequestAppMeta,
    ) -> Optional[RequestScheduleScore]:
        scores = self.score_candidates(candidates, current_request)
        if not scores:
            return None
        return max(scores, key=lambda item: item.score)


def classify_task_group(task_type: str) -> str:
    if task_type in INTERACTIVE_GENERATION_TASKS:
        return "interactive_generation"
    if task_type in VECTOR_THROUGHPUT_TASKS:
        return "vector_throughput"
    if task_type in HEAVY_MULTIMEDIA_TASKS:
        return "heavy_multimedia"
    return "standard"


def classify_context_bucket(current_request: RequestAppMeta) -> str:
    total_tokens = (
        current_request.estimated_prompt_tokens
        + current_request.estimated_output_tokens
    )
    if total_tokens <= SHORT_CONTEXT_TOKENS:
        return "short"
    if total_tokens <= MEDIUM_CONTEXT_TOKENS:
        return "medium"
    return "long"


def _build_features(candidate: RequestScheduleCandidate) -> _CandidateFeatures:
    instance_state = candidate.instance_state
    vram_usage_rate = candidate_vram_usage_rate(candidate)
    qps = instance_state.instance_qps_10s if instance_state is not None else 0.0
    inflight = instance_state.instance_inflight if instance_state is not None else 0
    qps_per_inflight = qps / max(inflight, 1)
    ttft_ms = (
        instance_state.recent_avg_ttft_ms
        if instance_state is not None and instance_state.recent_avg_ttft_ms > 0
        else None
    )
    resident_seconds = (
        instance_state.resident_seconds if instance_state is not None else 0.0
    )

    return _CandidateFeatures(
        candidate=candidate,
        vram_usage_rate=vram_usage_rate,
        vram_balance_score=_vram_balance_score(vram_usage_rate),
        vram_headroom_score=_vram_headroom_score(vram_usage_rate),
        heavy_vram_score=_heavy_vram_score(vram_usage_rate),
        qps_per_inflight=qps_per_inflight,
        qps_per_inflight_score=0.0,
        inflight=inflight,
        inflight_score=1.0 / (1.0 + inflight),
        ttft_ms=ttft_ms,
        ttft_score=0.5,
        resident_seconds=resident_seconds,
        residency_score=0.0,
    )


def _normalize_relative_features(
    features: List[_CandidateFeatures],
) -> List[_CandidateFeatures]:
    max_qps_per_inflight = max(
        (item.qps_per_inflight for item in features),
        default=0.0,
    )
    max_resident_seconds = max(
        (item.resident_seconds for item in features),
        default=0.0,
    )
    ttft_samples = [item.ttft_ms for item in features if item.ttft_ms is not None]
    max_ttft = max(ttft_samples) if ttft_samples else 0.0

    normalized = []
    for item in features:
        qps_score = (
            item.qps_per_inflight / max_qps_per_inflight
            if max_qps_per_inflight > 0
            else 0.0
        )
        residency_score = (
            item.resident_seconds / max_resident_seconds
            if max_resident_seconds > 0
            else 0.0
        )
        ttft_score = (
            1.0 - (item.ttft_ms / max_ttft)
            if item.ttft_ms is not None and max_ttft > 0
            else 0.5
        )

        normalized.append(
            _CandidateFeatures(
                candidate=item.candidate,
                vram_usage_rate=item.vram_usage_rate,
                vram_balance_score=item.vram_balance_score,
                vram_headroom_score=item.vram_headroom_score,
                heavy_vram_score=item.heavy_vram_score,
                qps_per_inflight=item.qps_per_inflight,
                qps_per_inflight_score=_clamp01(qps_score),
                inflight=item.inflight,
                inflight_score=item.inflight_score,
                ttft_ms=item.ttft_ms,
                ttft_score=_clamp01(ttft_score),
                resident_seconds=item.resident_seconds,
                residency_score=_clamp01(residency_score),
            )
        )
    return normalized


def _weighted_score(
    features: _CandidateFeatures,
    weights: Dict[str, float],
) -> float:
    return (
        weights["ttft"] * features.ttft_score
        + weights["inflight"] * features.inflight_score
        + weights["vram_balance"] * features.vram_balance_score
        + weights["vram_headroom"] * features.vram_headroom_score
        + weights["heavy_vram"] * features.heavy_vram_score
        + weights["qps"] * features.qps_per_inflight_score
        + weights["residency"] * features.residency_score
    )


def _weights_for(
    task_group: str,
    context_bucket: str,
    slo_class: str,
) -> Dict[str, float]:
    if task_group == "interactive_generation":
        weights = _interactive_weights(context_bucket)
    elif task_group == "vector_throughput":
        weights = _vector_weights(context_bucket)
    elif task_group == "heavy_multimedia":
        weights = _heavy_weights(context_bucket)
    else:
        weights = _standard_weights(context_bucket)

    return _normalize_weights(_adjust_weights_by_slo(weights, slo_class))


def _interactive_weights(context_bucket: str) -> Dict[str, float]:
    if context_bucket == "long":
        return _weight_map(0.30, 0.25, 0.15, 0.15, 0.15, 0.10, 0.05)
    if context_bucket == "short":
        return _weight_map(0.45, 0.25, 0.10, 0.05, 0.05, 0.10, 0.05)
    return _weight_map(0.40, 0.25, 0.15, 0.05, 0.05, 0.10, 0.05)


def _vector_weights(context_bucket: str) -> Dict[str, float]:
    if context_bucket == "long":
        return _weight_map(0.05, 0.15, 0.25, 0.10, 0.10, 0.35, 0.10)
    if context_bucket == "short":
        return _weight_map(0.05, 0.10, 0.25, 0.00, 0.00, 0.50, 0.10)
    return _weight_map(0.05, 0.15, 0.25, 0.00, 0.00, 0.45, 0.10)


def _heavy_weights(context_bucket: str) -> Dict[str, float]:
    if context_bucket == "long":
        return _weight_map(0.05, 0.30, 0.00, 0.20, 0.25, 0.10, 0.10)
    if context_bucket == "short":
        return _weight_map(0.10, 0.30, 0.00, 0.15, 0.15, 0.20, 0.10)
    return _weight_map(0.10, 0.30, 0.00, 0.15, 0.20, 0.15, 0.10)


def _standard_weights(context_bucket: str) -> Dict[str, float]:
    if context_bucket == "long":
        return _weight_map(0.20, 0.20, 0.10, 0.20, 0.10, 0.15, 0.05)
    return _weight_map(0.25, 0.20, 0.20, 0.05, 0.05, 0.20, 0.05)


def _weight_map(
    ttft: float,
    inflight: float,
    vram_balance: float,
    vram_headroom: float,
    heavy_vram: float,
    qps: float,
    residency: float,
) -> Dict[str, float]:
    return {
        "ttft": ttft,
        "inflight": inflight,
        "vram_balance": vram_balance,
        "vram_headroom": vram_headroom,
        "heavy_vram": heavy_vram,
        "qps": qps,
        "residency": residency,
    }


def _adjust_weights_by_slo(
    weights: Dict[str, float],
    slo_class: str,
) -> Dict[str, float]:
    adjusted = dict(weights)
    if slo_class == "latency":
        adjusted["ttft"] += 0.10
        adjusted["inflight"] += 0.05
        adjusted["qps"] -= 0.10
        adjusted["vram_balance"] -= 0.05
    elif slo_class == "throughput":
        adjusted["qps"] += 0.10
        adjusted["vram_balance"] += 0.05
        adjusted["ttft"] -= 0.10
        adjusted["inflight"] -= 0.05

    return {key: max(value, 0.0) for key, value in adjusted.items()}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return _standard_weights("medium")
    return {key: value / total for key, value in weights.items()}


def _vram_balance_score(vram_usage_rate: Optional[float]) -> float:
    if vram_usage_rate is None:
        return 0.5
    return _clamp01(1.0 - abs(vram_usage_rate - 0.75) / 0.75)


def _vram_headroom_score(vram_usage_rate: Optional[float]) -> float:
    if vram_usage_rate is None:
        return 0.5
    return _clamp01(1.0 - vram_usage_rate)


def _heavy_vram_score(vram_usage_rate: Optional[float]) -> float:
    if vram_usage_rate is None:
        return 0.5
    if vram_usage_rate <= 0.60:
        return 1.0
    if vram_usage_rate <= 0.75:
        return 0.80
    if vram_usage_rate <= 0.90:
        return _clamp01(0.80 - (vram_usage_rate - 0.75) / 0.15 * 0.60)
    return 0.05


def _build_deployment_hint(
    features: Sequence[_CandidateFeatures],
    current_request: RequestAppMeta,
    task_group: str,
) -> Optional[DeploymentHint]:
    if not features:
        return None

    model_id = current_request.model_id
    preferred_worker_id = _least_loaded_worker_id(features)

    ttft_slo_ms = current_request.ttft_slo_ms
    if ttft_slo_ms is not None:
        ttft_samples = [item.ttft_ms for item in features if item.ttft_ms is not None]
        if ttft_samples and all(
            _compensated_ttft_ms(ttft) >= ttft_slo_ms * TTFT_SLO_APPROACH_RATIO
            for ttft in ttft_samples
        ):
            return DeploymentHint(
                reason="latency_slo_risk",
                task_group=task_group,
                model_id=model_id,
                preferred_worker_id=preferred_worker_id,
                ttft_slo_ms=ttft_slo_ms,
                detail=(
                    "all compensated candidate TTFT samples approach or exceed "
                    f"{ttft_slo_ms} ms"
                ),
                instance_metrics=_instance_deployment_metrics(features),
                model_metrics=_model_deployment_metrics(features),
                worker_metrics=_worker_deployment_metrics(features),
            )

    if task_group == "vector_throughput":
        if all(item.inflight >= 4 for item in features):
            return DeploymentHint(
                reason="throughput_pressure",
                task_group=task_group,
                model_id=model_id,
                preferred_worker_id=preferred_worker_id,
                detail="all vector candidates have inflight >= 4",
                instance_metrics=_instance_deployment_metrics(features),
                model_metrics=_model_deployment_metrics(features),
                worker_metrics=_worker_deployment_metrics(features),
            )

    if all(item.inflight >= 4 for item in features):
        return DeploymentHint(
            reason="inflight_pressure",
            task_group=task_group,
            model_id=model_id,
            preferred_worker_id=preferred_worker_id,
            ttft_slo_ms=ttft_slo_ms,
            detail="all candidates have inflight >= 4",
            instance_metrics=_instance_deployment_metrics(features),
            model_metrics=_model_deployment_metrics(features),
            worker_metrics=_worker_deployment_metrics(features),
        )

    if all(
        item.vram_usage_rate is not None
        and item.vram_usage_rate >= VRAM_SCALE_OUT_WATERMARK
        for item in features
    ):
        return DeploymentHint(
            reason="vram_bottleneck",
            task_group=task_group,
            model_id=model_id,
            preferred_worker_id=preferred_worker_id,
            ttft_slo_ms=ttft_slo_ms,
            detail="all candidates exceed 90% VRAM usage",
            instance_metrics=_instance_deployment_metrics(features),
            model_metrics=_model_deployment_metrics(features),
            worker_metrics=_worker_deployment_metrics(features),
        )

    if task_group == "heavy_multimedia":
        if all(
            item.vram_usage_rate is not None and item.vram_usage_rate > 0.75
            for item in features
        ):
            return DeploymentHint(
                reason="vram_pressure",
                task_group=task_group,
                model_id=model_id,
                preferred_worker_id=preferred_worker_id,
                ttft_slo_ms=ttft_slo_ms,
                detail="all heavy candidates exceed 75% VRAM usage",
                instance_metrics=_instance_deployment_metrics(features),
                model_metrics=_model_deployment_metrics(features),
                worker_metrics=_worker_deployment_metrics(features),
            )

    return None


def _least_loaded_worker_id(
    features: Sequence[_CandidateFeatures],
) -> Optional[int]:
    selected = min(features, key=lambda item: item.inflight, default=None)
    if selected is None:
        return None
    return selected.candidate.instance.worker_id


def _instance_deployment_metrics(
    features: Sequence[_CandidateFeatures],
) -> List[InstanceDeploymentMetrics]:
    metrics = []
    for item in features:
        state = item.candidate.instance_state
        if state is None:
            continue
        metrics.append(
            InstanceDeploymentMetrics(
                instance_id=state.instance_id,
                model_id=state.model_id,
                worker_id=state.worker_id,
                state=state.state,
                gpu_indexes=list(state.gpu_indexes),
                claimed_ram=state.claimed_ram,
                claimed_vram=state.claimed_vram,
                resident_seconds=state.resident_seconds,
                instance_qps_10s=state.instance_qps_10s,
                instance_inflight=state.instance_inflight,
                recent_avg_prompt_tokens=state.recent_avg_prompt_tokens,
                recent_avg_ttft_ms=state.recent_avg_ttft_ms,
                vram_usage_rate=item.vram_usage_rate,
                qps_per_inflight=item.qps_per_inflight,
            )
        )
    return metrics


def _model_deployment_metrics(
    features: Sequence[_CandidateFeatures],
) -> Optional[ModelDeploymentMetrics]:
    for item in features:
        state = item.candidate.model_state
        if state is None:
            continue
        return ModelDeploymentMetrics(
            model_id=state.model_id,
            model_name=state.model_name,
            total_instances=state.total_instances,
            running_instances=state.running_instances,
            allocated_ram=state.allocated_ram,
            allocated_vram=state.allocated_vram,
            model_qps_10s=state.model_qps_10s,
            model_inflight=state.model_inflight,
            avg_prompt_tokens_10s=state.avg_prompt_tokens_10s,
            task_type_mix=dict(state.task_type_mix),
            slo_mix=dict(state.slo_mix),
        )
    return None


def _worker_deployment_metrics(
    features: Sequence[_CandidateFeatures],
) -> List[WorkerDeploymentMetrics]:
    metrics_by_id: Dict[int, WorkerDeploymentMetrics] = {}
    for item in features:
        state = item.candidate.worker_state
        if state is None:
            continue
        vram_usage_rate = (
            state.vram_used / state.vram_total if state.vram_total > 0 else None
        )
        metrics_by_id[state.worker_id] = WorkerDeploymentMetrics(
            worker_id=state.worker_id,
            worker_name=state.worker_name,
            state=state.state,
            cpu_utilization_rate=state.cpu_utilization_rate,
            ram_total=state.ram_total,
            ram_used=state.ram_used,
            ram_allocated=state.ram_allocated,
            ram_allocatable=state.ram_allocatable,
            gpu_count=state.gpu_count,
            gpu_utilization_rate_avg=state.gpu_utilization_rate_avg,
            vram_total=state.vram_total,
            vram_used=state.vram_used,
            vram_allocatable=state.vram_allocatable,
            vram_usage_rate=vram_usage_rate,
            power_watts=state.power_watts,
        )
    return list(metrics_by_id.values())


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _compensated_ttft_ms(ttft_ms: float) -> float:
    return ttft_ms * TTFT_OBSERVATION_COMPENSATION_FACTOR
