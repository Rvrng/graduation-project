from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, TYPE_CHECKING

from gpustack.runtime_state.types import GPUState

if TYPE_CHECKING:
    from gpustack.runtime_scheduler.request_scheduler import RequestScheduleCandidate


TARGET_VRAM_USAGE_RATE = 0.75
EPSILON = 1e-9


@dataclass(frozen=True)
class RequestParetoObjective:
    candidate: RequestScheduleCandidate
    vram_usage_distance: float
    qps_per_inflight: float
    ttft_ms: float


class ParetoFrontSolver(Protocol):
    def solve(
        self,
        objectives: Sequence[RequestParetoObjective],
    ) -> List[RequestParetoObjective]:
        ...


class ExactParetoFrontSolver:
    """
    Exact non-dominated front solver for the current finite candidate set.
    """

    def solve(
        self,
        objectives: Sequence[RequestParetoObjective],
    ) -> List[RequestParetoObjective]:
        front: List[RequestParetoObjective] = []

        for objective in objectives:
            if any(
                _dominates(other, objective)
                for other in objectives
                if other is not objective
            ):
                continue
            front.append(objective)

        return front


def build_request_pareto_objectives(
    candidates: Sequence[RequestScheduleCandidate],
) -> List[RequestParetoObjective]:
    return [_build_request_pareto_objective(candidate) for candidate in candidates]


def _build_request_pareto_objective(
    candidate: RequestScheduleCandidate,
) -> RequestParetoObjective:
    vram_usage_rate = candidate_vram_usage_rate(candidate)
    vram_usage_distance = (
        abs(vram_usage_rate - TARGET_VRAM_USAGE_RATE)
        if vram_usage_rate is not None
        else math.inf
    )

    instance_state = candidate.instance_state
    qps = instance_state.instance_qps_10s if instance_state is not None else 0.0
    inflight = instance_state.instance_inflight if instance_state is not None else 0
    qps_per_inflight = qps / max(inflight, 1)

    ttft_ms = (
        instance_state.recent_avg_ttft_ms
        if instance_state is not None and instance_state.recent_avg_ttft_ms > 0
        else math.inf
    )

    return RequestParetoObjective(
        candidate=candidate,
        vram_usage_distance=vram_usage_distance,
        qps_per_inflight=qps_per_inflight,
        ttft_ms=ttft_ms,
    )


def candidate_vram_usage_rate(
    candidate: RequestScheduleCandidate,
) -> Optional[float]:
    worker_state = candidate.worker_state
    if worker_state is None:
        return None

    instance_state = candidate.instance_state
    if instance_state is not None and instance_state.gpu_indexes:
        selected_gpus = _select_instance_gpus(
            worker_state.gpus,
            instance_state.gpu_indexes,
        )
        if selected_gpus:
            total = sum(gpu.vram_total for gpu in selected_gpus)
            if total > 0:
                used = sum(gpu.vram_used for gpu in selected_gpus)
                return _clamp_rate(used / total)

    if worker_state.vram_total > 0:
        return _clamp_rate(worker_state.vram_used / worker_state.vram_total)

    return None


def _select_instance_gpus(
    gpus: Sequence[GPUState],
    gpu_indexes: Sequence[int],
) -> List[GPUState]:
    index_set = set(gpu_indexes)
    return [
        gpu
        for gpu in gpus
        if gpu.gpu_index in index_set and gpu.vram_total > 0
    ]


def _clamp_rate(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _dominates(
    a: RequestParetoObjective,
    b: RequestParetoObjective,
) -> bool:
    not_worse = (
        a.vram_usage_distance <= b.vram_usage_distance + EPSILON
        and a.qps_per_inflight + EPSILON >= b.qps_per_inflight
        and a.ttft_ms <= b.ttft_ms + EPSILON
    )
    strictly_better = (
        a.vram_usage_distance + EPSILON < b.vram_usage_distance
        or a.qps_per_inflight > b.qps_per_inflight + EPSILON
        or a.ttft_ms + EPSILON < b.ttft_ms
    )
    return not_worse and strictly_better
