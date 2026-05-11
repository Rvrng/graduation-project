import logging
import math
import os
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.db import async_session
from gpustack.server.services import ModelInstanceService, ModelService
from gpustack.runtime_state import get_current_state
from gpustack.runtime_state.types import InstanceState, SchedulingState, WorkerState


logger = logging.getLogger(__name__)

SCALE_OUT_WINDOW_SECONDS = 60
SCALE_OUT_COOLDOWN_SECONDS = 180
SCALE_OUT_MIN_HINTS = 3
SCALE_OUT_DEFAULT_MAX_REPLICAS = 8
MODEL_SCALE_IN_COOLDOWN_SECONDS = 300
WORKER_COMPACTION_COOLDOWN_SECONDS = 300
SCALE_IN_INTERVAL_SECONDS = 60
SCALE_IN_HISTORY_CYCLES = 3
TTFT_SLO_APPROACH_RATIO = 0.90
TTFT_OBSERVATION_COMPENSATION_FACTOR = 0.85
INFLIGHT_PER_INSTANCE_WATERMARK = 1.5
QPS_RISING_RATIO = 1.5
QPS_RISING_MIN_DELTA = 0.5
VRAM_HIGH_WATERMARK = 0.90
MODEL_LOW_QPS_THRESHOLD = 0.2
INSTANCE_QPS_SAFE_LIMIT = 5.0
INSTANCE_TTFT_GOOD_MS = 500.0
GPU_LOW_UTILIZATION_RATE = 20.0
CLUSTER_LOW_GPU_UTILIZATION_RATE = 25.0
WORKER_LOW_VRAM_USAGE_RATE = 0.20
WORKER_DEST_GPU_DANGER_RATE = 70.0
DRAIN_GRACE_SECONDS = 10
PLACEMENT_PIN_TTL_SECONDS = 600
SCALE_OUT_NSGA_POPULATION_SIZE = 50
SCALE_OUT_NSGA_MAX_GENERATIONS = 50
SCALE_OUT_MUTATION_PROBABILITY = 0.10
SCALE_OUT_RESOURCE_VRAM_WEIGHT = 0.60
SCALE_OUT_RESOURCE_GPU_WEIGHT = 0.30
SCALE_OUT_RESOURCE_RAM_WEIGHT = 0.10
SCALE_OUT_ESTIMATED_INSTANCE_GPU_UTIL = 0.05
SCALE_OUT_CONTENTION_QPS_WEIGHT = 1.0
SCALE_OUT_TOPSIS_WEIGHTS = (0.40, 0.20, 0.40)
SCALE_OUT_TOPSIS_WEIGHTS_WITH_ENERGY = (0.36, 0.18, 0.36, 0.10)
SCALE_OUT_DECISION_STRATEGY_ENV = "GPUSTACK_RUNTIME_SCALE_OUT_DECISION_STRATEGY"
EPSILON = 1e-9

TRANSITIONAL_STATES = {
    ModelInstanceStateEnum.PENDING,
    ModelInstanceStateEnum.ANALYZING,
    ModelInstanceStateEnum.SCHEDULED,
    ModelInstanceStateEnum.INITIALIZING,
    ModelInstanceStateEnum.DOWNLOADING,
    ModelInstanceStateEnum.STARTING,
}


@dataclass(frozen=True)
class InstanceDeploymentMetrics:
    instance_id: int
    model_id: int
    worker_id: Optional[int]
    state: str
    gpu_indexes: List[int] = field(default_factory=list)
    claimed_ram: int = 0
    claimed_vram: int = 0
    resident_seconds: float = 0.0
    instance_qps_10s: float = 0.0
    instance_inflight: int = 0
    recent_avg_prompt_tokens: float = 0.0
    recent_avg_ttft_ms: float = 0.0
    vram_usage_rate: Optional[float] = None
    qps_per_inflight: float = 0.0


@dataclass(frozen=True)
class ModelDeploymentMetrics:
    model_id: int
    model_name: str
    total_instances: int = 0
    running_instances: int = 0
    allocated_ram: int = 0
    allocated_vram: int = 0
    model_qps_10s: float = 0.0
    model_inflight: int = 0
    avg_prompt_tokens_10s: float = 0.0
    task_type_mix: Dict[str, int] = field(default_factory=dict)
    slo_mix: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerDeploymentMetrics:
    worker_id: int
    worker_name: str
    state: str
    cpu_utilization_rate: Optional[float] = None
    ram_total: int = 0
    ram_used: int = 0
    ram_allocated: int = 0
    ram_allocatable: int = 0
    gpu_count: int = 0
    gpu_utilization_rate_avg: Optional[float] = None
    vram_total: int = 0
    vram_used: int = 0
    vram_allocatable: int = 0
    vram_usage_rate: Optional[float] = None
    power_watts: Optional[float] = None


@dataclass(frozen=True)
class DeploymentHint:
    reason: str
    task_group: str
    model_id: Optional[int]
    preferred_worker_id: Optional[int] = None
    ttft_slo_ms: Optional[float] = None
    detail: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    instance_metrics: List[InstanceDeploymentMetrics] = field(default_factory=list)
    model_metrics: Optional[ModelDeploymentMetrics] = None
    worker_metrics: List[WorkerDeploymentMetrics] = field(default_factory=list)


@dataclass(frozen=True)
class DrainPlan:
    source_instance_id: int
    model_id: int
    source_worker_id: Optional[int]
    destination_instance_id: Optional[int]
    destination_worker_id: Optional[int]
    reason: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _ModelHistorySample:
    generated_at: datetime
    model_qps_10s: float
    gpu_utilization_rate_avg: Optional[float]


@dataclass
class _WorkerCompactionPlan:
    source_worker_id: int
    destination_worker_id: int
    model_id: int
    source_instance_id: int
    source_claimed_vram: int = 0
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class PlacementPin:
    model_id: int
    target_worker_id: int
    source_instance_id: Optional[int]
    reason: str
    strict: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class _ScaleOutWorkerFeature:
    worker: WorkerState
    candidate_index: int
    ram_headroom: float
    vram_headroom: float
    ram_usage_rate: float
    vram_usage_rate: float
    gpu_utilization_rate: float
    power_watts: Optional[float]
    gpu_count: int
    same_model_instances: int
    worker_inflight: int
    worker_qps: float


@dataclass
class _ScaleOutPlan:
    assignments: Tuple[int, ...]
    objectives: Tuple[float, ...]
    constraint_violation: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0


class DeploymentHintStore:
    def __init__(self, maxlen: int = 1000):
        self._hints: Deque[DeploymentHint] = deque(maxlen=maxlen)

    def add(self, hint: DeploymentHint) -> None:
        self._hints.append(hint)

    def recent(self, seconds: int = 60) -> List[DeploymentHint]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        return [hint for hint in self._hints if hint.generated_at >= cutoff]

    def counts_by_model(self, seconds: int = 60) -> Dict[int, Counter]:
        counts: Dict[int, Counter] = {}
        for hint in self.recent(seconds):
            if hint.model_id is None:
                continue
            counts.setdefault(hint.model_id, Counter())[hint.reason] += 1
        return counts


deployment_hint_store = DeploymentHintStore()
drain_plan_store: Dict[int, DrainPlan] = {}
model_last_scale_up_at: Dict[int, datetime] = {}
worker_last_scale_up_at: Dict[int, datetime] = {}
placement_pin_store: Dict[int, Deque[PlacementPin]] = {}


def is_instance_draining(instance_id: Optional[int]) -> bool:
    return instance_id is not None and instance_id in drain_plan_store


def drain_destination_instance_ids(model_id: Optional[int]) -> set[int]:
    if model_id is None:
        return set()
    return {
        plan.destination_instance_id
        for plan in drain_plan_store.values()
        if plan.model_id == model_id and plan.destination_instance_id is not None
    }


def peek_model_placement_pin(model_id: Optional[int]) -> Optional[PlacementPin]:
    if model_id is None:
        return None
    _prune_expired_placement_pins()
    pins = placement_pin_store.get(model_id)
    if not pins:
        return None
    return pins[0]


def consume_model_placement_pin(
    model_id: Optional[int],
    worker_id: Optional[int],
) -> Optional[PlacementPin]:
    if model_id is None or worker_id is None:
        return None
    _prune_expired_placement_pins()
    pins = placement_pin_store.get(model_id)
    if not pins:
        return None
    pin = pins[0]
    if pin.target_worker_id != worker_id:
        return None
    pins.popleft()
    if not pins:
        placement_pin_store.pop(model_id, None)
    return pin


def _add_placement_pin(pin: PlacementPin) -> None:
    pins = placement_pin_store.setdefault(pin.model_id, deque())
    if any(
        item.source_instance_id == pin.source_instance_id
        and item.target_worker_id == pin.target_worker_id
        for item in pins
    ):
        return
    pins.append(pin)


def _prune_expired_placement_pins() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=PLACEMENT_PIN_TTL_SECONDS)
    for model_id, pins in list(placement_pin_store.items()):
        while pins and pins[0].created_at < cutoff:
            pins.popleft()
        if not pins:
            placement_pin_store.pop(model_id, None)


class RuntimeAwareDeployer:
    """
    Consume runtime deployment hints and request scale-out through Model.replicas.

    The model controller still creates the pending instance and the original
    scheduler still validates backend-specific resource fit. Runtime-aware scale
    out selects a preferred worker first and passes that preference through a
    non-strict placement pin.
    """

    def __init__(self, hint_store: Optional[DeploymentHintStore] = None):
        self._hint_store = hint_store or deployment_hint_store
        self._scale_lock = None
        self._model_history: Dict[int, Deque[_ModelHistorySample]] = {}
        self._worker_compaction_plans: Dict[int, _WorkerCompactionPlan] = {}
        self._worker_compaction_target_pins: Dict[int, int] = {}
        self._seen_running_instance_ids: set[int] = set()
        self._running_observation_initialized = False

    async def report_hint(self, hint: DeploymentHint) -> None:
        self._hint_store.add(hint)
        logger.info(
            "Runtime-aware deployment hint: reason=%s task_group=%s "
            "model_id=%s preferred_worker_id=%s ttft_slo_ms=%s "
            "instances=%s workers=%s detail=%s",
            hint.reason,
            hint.task_group,
            hint.model_id,
            hint.preferred_worker_id,
            hint.ttft_slo_ms,
            len(hint.instance_metrics),
            len(hint.worker_metrics),
            hint.detail,
        )
        await self._maybe_scale_out(hint)

    def recent_hints(self, seconds: int = 60) -> List[DeploymentHint]:
        return self._hint_store.recent(seconds)

    def hint_counts_by_model(self, seconds: int = 60) -> Dict[int, Counter]:
        return self._hint_store.counts_by_model(seconds)

    async def _maybe_scale_out(self, hint: DeploymentHint) -> None:
        if hint.model_id is None:
            return

        recent_hints = [
            item
            for item in self._hint_store.recent(SCALE_OUT_WINDOW_SECONDS)
            if item.model_id == hint.model_id
        ]
        should_scale, reason = self._should_scale_out(recent_hints)
        if not should_scale:
            logger.debug(
                "Runtime-aware scale-out skipped for model_id=%s: %s",
                hint.model_id,
                reason,
            )
            return

        if self._scale_lock is None:
            import asyncio

            self._scale_lock = asyncio.Lock()

        async with self._scale_lock:
            now = datetime.now(timezone.utc)
            last_scaled_at = model_last_scale_up_at.get(hint.model_id)
            if (
                last_scaled_at is not None
                and now - last_scaled_at
                < timedelta(seconds=SCALE_OUT_COOLDOWN_SECONDS)
            ):
                return

            async with async_session() as session:
                model = await Model.one_by_id(session, hint.model_id)
                if model is None or model.deleted_at is not None:
                    return

                instances = await ModelInstance.all_by_field(
                    session,
                    "model_id",
                    hint.model_id,
                )
                if len(instances) < model.replicas:
                    logger.info(
                        "Runtime-aware scale-out already pending for model_id=%s: "
                        "instances=%s replicas=%s",
                        hint.model_id,
                        len(instances),
                        model.replicas,
                    )
                    return

                if any(instance.state in TRANSITIONAL_STATES for instance in instances):
                    logger.info(
                        "Runtime-aware scale-out deferred for model_id=%s because a "
                        "model instance is already in deployment.",
                        hint.model_id,
                    )
                    return

                max_replicas = _model_max_replicas(model)
                if len(instances) >= max_replicas:
                    logger.info(
                        "Runtime-aware scale-out capped for model_id=%s at "
                        "max_replicas=%s.",
                        hint.model_id,
                        max_replicas,
                    )
                    return

                state = await get_current_state(session)
                target_worker = _select_scale_out_worker(
                    hint=hint,
                    reason=reason,
                    state=state,
                )
                if target_worker is not None:
                    _add_placement_pin(
                        PlacementPin(
                            model_id=hint.model_id,
                            target_worker_id=target_worker.worker_id,
                            source_instance_id=None,
                            reason=f"scale_out:{reason}",
                            strict=False,
                            created_at=now,
                        )
                    )
                    logger.info(
                        "Runtime-aware scale-out selected target worker_id=%s "
                        "worker=%s model_id=%s reason=%s.",
                        target_worker.worker_id,
                        target_worker.worker_name,
                        hint.model_id,
                        reason,
                    )
                else:
                    logger.info(
                        "Runtime-aware scale-out found no preferred target worker "
                        "for model_id=%s reason=%s; falling back to original "
                        "scheduler placement.",
                        hint.model_id,
                        reason,
                    )

                target_replicas = len(instances) + 1
                model.replicas = target_replicas
                await ModelService(session).update(model)
                self._record_model_scale_up(hint.model_id, now)

                logger.info(
                    "Runtime-aware scale-out requested for model_id=%s "
                    "model=%s replicas=%s reason=%s. Model controller will create "
                    "a PENDING instance and the scheduler will place it.",
                    hint.model_id,
                    model.name,
                    target_replicas,
                    reason,
                )

    def _should_scale_out(self, hints: List[DeploymentHint]) -> tuple[bool, str]:
        if len(hints) < SCALE_OUT_MIN_HINTS:
            return False, "not_enough_recent_hints"

        latest = hints[-1]
        model_metrics = latest.model_metrics
        if model_metrics is None or model_metrics.running_instances <= 0:
            return False, "missing_model_metrics"

        inflight_per_instance = (
            model_metrics.model_inflight / model_metrics.running_instances
        )
        inflight_pressure = inflight_per_instance >= INFLIGHT_PER_INSTANCE_WATERMARK
        qps_rising = _qps_is_rising(hints)
        ttft_pressure = _ttft_pressure_is_sustained(hints)
        vram_pressure = _vram_pressure_is_sustained(hints)

        if ttft_pressure:
            return True, "ttft_slo_pressure"
        if inflight_pressure and qps_rising:
            return True, "inflight_qps_pressure"
        if vram_pressure and (inflight_pressure or qps_rising):
            return True, "vram_pressure"

        return (
            False,
            "pressure_not_sustained "
            f"inflight_per_instance={inflight_per_instance:.2f} "
            f"qps_rising={qps_rising} ttft_pressure={ttft_pressure} "
            f"vram_pressure={vram_pressure}",
        )

    async def reconcile_scale_in(self) -> None:
        async with async_session() as session:
            state = await get_current_state(session)
            await self._observe_running_instances(state)
            await self._finalize_draining_instances(session, state)
            self._record_history(state)
            await self._reconcile_worker_compaction_plans(session, state)
            await self._reconcile_instance_consolidation(session, state)
            await self._reconcile_worker_compaction(session, state)

    def _record_model_scale_up(self, model_id: int, when: datetime) -> None:
        model_last_scale_up_at[model_id] = when

    async def _observe_running_instances(self, state: SchedulingState) -> None:
        running_ids = {
            instance_id
            for instance_id, instance_state in state.instances.items()
            if instance_state.state == str(ModelInstanceStateEnum.RUNNING)
        }
        if not self._running_observation_initialized:
            self._seen_running_instance_ids = running_ids
            self._running_observation_initialized = True
            return

        now = datetime.now(timezone.utc)
        new_running_ids = running_ids - self._seen_running_instance_ids
        for instance_id in new_running_ids:
            instance_state = state.instances.get(instance_id)
            if instance_state is None:
                continue
            if instance_state.worker_id is not None:
                worker_last_scale_up_at[instance_state.worker_id] = now
            self._record_model_scale_up(instance_state.model_id, now)
        self._seen_running_instance_ids = running_ids

    def _record_history(self, state: SchedulingState) -> None:
        for model_id, model_state in state.models.items():
            gpu_util = _model_gpu_utilization_rate(model_id, state)
            history = self._model_history.setdefault(
                model_id,
                deque(maxlen=SCALE_IN_HISTORY_CYCLES),
            )
            history.append(
                _ModelHistorySample(
                    generated_at=state.generated_at,
                    model_qps_10s=model_state.model_qps_10s,
                    gpu_utilization_rate_avg=gpu_util,
                )
            )

    async def _reconcile_instance_consolidation(
        self,
        session,
        state: SchedulingState,
    ) -> None:
        for model_id, model_state in state.models.items():
            if model_state.running_instances <= 1:
                continue
            if self._model_in_cooldown(model_id):
                continue
            if not self._model_is_low_load(model_id):
                continue

            running_instances = _running_model_instances(model_id, state)
            source = self._select_consolidation_source(running_instances)
            if source is None:
                continue
            destination = self._select_consolidation_destination(
                source,
                running_instances,
                state,
            )
            if destination is None:
                continue

            await self._mark_draining(
                source_instance_id=source.instance_id,
                model_id=model_id,
                source_worker_id=source.worker_id,
                destination_instance_id=destination.instance_id,
                destination_worker_id=destination.worker_id,
                reason="instance_traffic_consolidation",
            )

    def _model_is_low_load(self, model_id: int) -> bool:
        history = self._model_history.get(model_id)
        if history is None or len(history) < SCALE_IN_HISTORY_CYCLES:
            return False

        return all(
            item.model_qps_10s <= MODEL_LOW_QPS_THRESHOLD
            and item.gpu_utilization_rate_avg is not None
            and item.gpu_utilization_rate_avg <= GPU_LOW_UTILIZATION_RATE
            for item in history
        )

    def _select_consolidation_source(
        self,
        instances: List[InstanceState],
    ) -> Optional[InstanceState]:
        candidates = [
            instance
            for instance in instances
            if not is_instance_draining(instance.instance_id)
            and instance.instance_inflight == 0
        ]
        if len(candidates) <= 1:
            return None
        return min(candidates, key=lambda item: item.resident_seconds)

    def _select_consolidation_destination(
        self,
        source: InstanceState,
        instances: List[InstanceState],
        state: SchedulingState,
    ) -> Optional[InstanceState]:
        candidates = [
            instance
            for instance in instances
            if instance.instance_id != source.instance_id
            and instance.model_id == source.model_id
            and not is_instance_draining(instance.instance_id)
            and instance.resident_seconds > source.resident_seconds
            and _destination_has_capacity(source, instance, state)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.resident_seconds)

    async def _reconcile_worker_compaction(
        self,
        session,
        state: SchedulingState,
    ) -> None:
        if not _cluster_gpu_utilization_is_low(state):
            return

        source_worker = _select_source_worker_for_compaction(state, self)
        if source_worker is None:
            return

        source_instances = [
            instance
            for instance in state.instances.values()
            if instance.worker_id == source_worker.worker_id
            and instance.state == str(ModelInstanceStateEnum.RUNNING)
            and not is_instance_draining(instance.instance_id)
        ]
        if not source_instances:
            logger.info(
                "Runtime-aware worker compaction found empty source worker_id=%s.",
                source_worker.worker_id,
            )
            return

        for source_instance in sorted(
            source_instances,
            key=lambda item: item.resident_seconds,
        ):
            pinned_destination_worker_id = self._worker_compaction_target_pins.get(
                source_worker.worker_id
            )
            destination = _select_worker_compaction_destination(
                source_instance,
                state,
                self,
                required_worker_id=pinned_destination_worker_id,
            )
            if destination is not None:
                await self._mark_draining(
                    source_instance_id=source_instance.instance_id,
                    model_id=source_instance.model_id,
                    source_worker_id=source_instance.worker_id,
                    destination_instance_id=destination.instance_id,
                    destination_worker_id=destination.worker_id,
                    reason="worker_compaction_existing_destination",
                )
                continue

            destination_worker = _select_worker_compaction_replacement_worker(
                source_worker_id=source_worker.worker_id,
                source_instance=source_instance,
                state=state,
                deployer=self,
            )
            if destination_worker is None:
                continue

            await self._request_worker_compaction_replacement(
                session,
                source_worker_id=source_worker.worker_id,
                destination_worker_id=destination_worker.worker_id,
                source_instance=source_instance,
            )

    async def _request_worker_compaction_replacement(
        self,
        session,
        source_worker_id: int,
        destination_worker_id: int,
        source_instance: InstanceState,
    ) -> None:
        if source_instance.instance_id in self._worker_compaction_plans:
            return

        model = await Model.one_by_id(session, source_instance.model_id)
        if model is None or model.deleted_at is not None:
            return

        instances = await ModelInstance.all_by_field(
            session,
            "model_id",
            source_instance.model_id,
        )
        if len(instances) < model.replicas:
            return
        if any(instance.state in TRANSITIONAL_STATES for instance in instances):
            return

        now = datetime.now(timezone.utc)
        model.replicas = len(instances) + 1
        await ModelService(session).update(model)
        self._record_model_scale_up(source_instance.model_id, now)
        self._worker_compaction_target_pins[source_worker_id] = destination_worker_id
        _add_placement_pin(
            PlacementPin(
                model_id=source_instance.model_id,
                target_worker_id=destination_worker_id,
                source_instance_id=source_instance.instance_id,
                reason="worker_compaction_replacement",
                created_at=now,
            )
        )
        self._worker_compaction_plans[source_instance.instance_id] = (
            _WorkerCompactionPlan(
                source_worker_id=source_worker_id,
                destination_worker_id=destination_worker_id,
                model_id=source_instance.model_id,
                source_instance_id=source_instance.instance_id,
                source_claimed_vram=source_instance.claimed_vram,
                requested_at=now,
            )
        )
        logger.info(
            "Runtime-aware worker compaction requested replacement for "
            "model_id=%s source_instance_id=%s source_worker_id=%s "
            "destination_worker_id=%s.",
            source_instance.model_id,
            source_instance.instance_id,
            source_worker_id,
            destination_worker_id,
        )

    async def _reconcile_worker_compaction_plans(
        self,
        session,
        state: SchedulingState,
    ) -> None:
        del session
        for source_instance_id, plan in list(self._worker_compaction_plans.items()):
            source = state.instances.get(source_instance_id)
            if source is None or source.state != str(ModelInstanceStateEnum.RUNNING):
                self._worker_compaction_plans.pop(source_instance_id, None)
                continue

            destination = _select_worker_compaction_destination(
                source,
                state,
                self,
                required_worker_id=plan.destination_worker_id,
            )
            if destination is None:
                continue

            await self._mark_draining(
                source_instance_id=source.instance_id,
                model_id=source.model_id,
                source_worker_id=source.worker_id,
                destination_instance_id=destination.instance_id,
                destination_worker_id=destination.worker_id,
                reason="worker_compaction_replacement_ready",
            )
            self._worker_compaction_plans.pop(source_instance_id, None)

    async def _mark_draining(
        self,
        source_instance_id: int,
        model_id: int,
        source_worker_id: Optional[int],
        destination_instance_id: Optional[int],
        destination_worker_id: Optional[int],
        reason: str,
    ) -> None:
        if source_instance_id in drain_plan_store:
            return

        drain_plan_store[source_instance_id] = DrainPlan(
            source_instance_id=source_instance_id,
            model_id=model_id,
            source_worker_id=source_worker_id,
            destination_instance_id=destination_instance_id,
            destination_worker_id=destination_worker_id,
            reason=reason,
        )
        logger.info(
            "Runtime-aware draining started for instance_id=%s model_id=%s "
            "destination_instance_id=%s reason=%s.",
            source_instance_id,
            model_id,
            destination_instance_id,
            reason,
        )

    async def _finalize_draining_instances(self, session, state: SchedulingState) -> None:
        now = datetime.now(timezone.utc)
        for instance_id, plan in list(drain_plan_store.items()):
            instance_state = state.instances.get(instance_id)
            if instance_state is None:
                drain_plan_store.pop(instance_id, None)
                continue
            if instance_state.instance_inflight > 0:
                continue
            if now - plan.created_at < timedelta(seconds=DRAIN_GRACE_SECONDS):
                continue

            model = await Model.one_by_id(session, plan.model_id)
            model_instance = await ModelInstance.one_by_id(session, instance_id)
            if model is None or model_instance is None:
                drain_plan_store.pop(instance_id, None)
                continue

            model.replicas = max(model.replicas - 1, 0)
            await ModelService(session).update(model)
            await ModelInstanceService(session).delete(model_instance)
            drain_plan_store.pop(instance_id, None)
            self._worker_compaction_plans.pop(instance_id, None)
            if plan.source_worker_id is not None and not _worker_has_running_instances(
                plan.source_worker_id,
                state,
                exclude_instance_id=instance_id,
            ):
                self._worker_compaction_target_pins.pop(plan.source_worker_id, None)
            logger.info(
                "Runtime-aware draining finalized for instance_id=%s model_id=%s "
                "reason=%s.",
                instance_id,
                plan.model_id,
                plan.reason,
            )

    def _model_in_cooldown(self, model_id: int) -> bool:
        last_scaled_at = model_last_scale_up_at.get(model_id)
        if last_scaled_at is None:
            return False
        return (
            datetime.now(timezone.utc) - last_scaled_at
            < timedelta(seconds=MODEL_SCALE_IN_COOLDOWN_SECONDS)
        )

    def _worker_in_cooldown(self, worker_id: Optional[int]) -> bool:
        if worker_id is None:
            return True
        last_scaled_at = worker_last_scale_up_at.get(worker_id)
        if last_scaled_at is None:
            return False
        return (
            datetime.now(timezone.utc) - last_scaled_at
            < timedelta(seconds=WORKER_COMPACTION_COOLDOWN_SECONDS)
        )


def _select_scale_out_worker(
    hint: DeploymentHint,
    reason: str,
    state: SchedulingState,
) -> Optional[WorkerState]:
    if hint.model_id is None:
        return None

    required_ram, required_vram = _estimated_model_replica_claim(hint, state)
    features = _build_scale_out_worker_features(
        hint=hint,
        state=state,
        required_ram=required_ram,
        required_vram=required_vram,
    )
    if not features:
        return None

    decision_strategy = _scale_out_decision_strategy(hint)
    plan = _select_scale_out_plan(
        features=features,
        required_ram=required_ram,
        required_vram=required_vram,
        new_instance_count=1,
        decision_strategy=decision_strategy,
    )
    if plan is None or not plan.assignments:
        return None

    selected_feature = features[plan.assignments[0]]
    logger.info(
        "Runtime-aware NSGA-II scale-out selected worker_id=%s reason=%s "
        "strategy=%s objectives=%s candidates=%s.",
        selected_feature.worker.worker_id,
        reason,
        decision_strategy,
        [round(item, 6) for item in plan.objectives],
        len(features),
    )
    return selected_feature.worker


def _build_scale_out_worker_features(
    hint: DeploymentHint,
    state: SchedulingState,
    required_ram: int,
    required_vram: int,
) -> List[_ScaleOutWorkerFeature]:
    features: List[_ScaleOutWorkerFeature] = []
    for worker in state.workers.values():
        if not _worker_can_receive_scale_out(worker, required_ram, required_vram):
            continue

        worker_instances = [
            instance
            for instance in state.instances.values()
            if instance.worker_id == worker.worker_id
            and instance.state == str(ModelInstanceStateEnum.RUNNING)
            and not is_instance_draining(instance.instance_id)
        ]
        same_model_instances = [
            instance
            for instance in worker_instances
            if instance.model_id == hint.model_id
        ]
        worker_inflight = sum(instance.instance_inflight for instance in worker_instances)
        worker_qps = sum(instance.instance_qps_10s for instance in worker_instances)

        features.append(
            _ScaleOutWorkerFeature(
                worker=worker,
                candidate_index=len(features),
                ram_headroom=_resource_headroom_ratio(
                    worker.ram_allocatable,
                    required_ram,
                ),
                vram_headroom=_resource_headroom_ratio(
                    worker.vram_allocatable,
                    required_vram,
                ),
                ram_usage_rate=_safe_rate(worker.ram_used, worker.ram_total),
                vram_usage_rate=_safe_rate(worker.vram_used, worker.vram_total),
                gpu_utilization_rate=_rate_from_percent(
                    worker.gpu_utilization_rate_avg
                ),
                power_watts=worker.power_watts,
                gpu_count=worker.gpu_count,
                same_model_instances=len(same_model_instances),
                worker_inflight=worker_inflight,
                worker_qps=worker_qps,
            )
        )
    return features


def _select_scale_out_plan(
    features: Sequence[_ScaleOutWorkerFeature],
    required_ram: int,
    required_vram: int,
    new_instance_count: int,
    decision_strategy: str,
) -> Optional[_ScaleOutPlan]:
    if not features or new_instance_count <= 0:
        return None

    energy_enabled = all(feature.power_watts is not None for feature in features)
    if new_instance_count == 1:
        plans = [
            _evaluate_scale_out_plan(
                assignments=(feature.candidate_index,),
                features=features,
                required_ram=required_ram,
                required_vram=required_vram,
                energy_enabled=energy_enabled,
            )
            for feature in features
        ]
        fronts = _fast_non_dominated_sort(plans)
        front = fronts[0] if fronts else plans
        return _select_plan_from_front(
            front,
            features,
            required_ram,
            required_vram,
            energy_enabled,
            decision_strategy,
        )

    population = _initial_scale_out_population(
        candidate_count=len(features),
        new_instance_count=new_instance_count,
    )
    population = [
        _evaluate_scale_out_plan(
            assignments=assignments,
            features=features,
            required_ram=required_ram,
            required_vram=required_vram,
            energy_enabled=energy_enabled,
        )
        for assignments in population
    ]

    rng = random.Random(
        _scale_out_rng_seed(features, required_ram, required_vram, new_instance_count)
    )
    for _ in range(SCALE_OUT_NSGA_MAX_GENERATIONS):
        fronts = _fast_non_dominated_sort(population)
        for front in fronts:
            _assign_crowding_distance(front)

        children: List[_ScaleOutPlan] = []
        while len(children) < SCALE_OUT_NSGA_POPULATION_SIZE:
            parent_a = _tournament_select(population, rng)
            parent_b = _tournament_select(population, rng)
            child_a, child_b = _crossover(parent_a.assignments, parent_b.assignments, rng)
            for child in (child_a, child_b):
                child = _mutate(child, len(features), rng)
                children.append(
                    _evaluate_scale_out_plan(
                        assignments=child,
                        features=features,
                        required_ram=required_ram,
                        required_vram=required_vram,
                        energy_enabled=energy_enabled,
                    )
                )
                if len(children) >= SCALE_OUT_NSGA_POPULATION_SIZE:
                    break

        population = _next_generation(population + children)

    fronts = _fast_non_dominated_sort(population)
    front = fronts[0] if fronts else population
    return _select_plan_from_front(
        front,
        features,
        required_ram,
        required_vram,
        energy_enabled,
        decision_strategy,
    )


def _evaluate_scale_out_plan(
    assignments: Tuple[int, ...],
    features: Sequence[_ScaleOutWorkerFeature],
    required_ram: int,
    required_vram: int,
    energy_enabled: bool,
) -> _ScaleOutPlan:
    assigned_counts = Counter(assignments)
    resource_pressure = 0.0
    constraint_violation = 0.0

    for feature in features:
        assigned_count = assigned_counts.get(feature.candidate_index, 0)
        worker = feature.worker

        predicted_vram_usage = _safe_rate(
            worker.vram_used + assigned_count * required_vram,
            worker.vram_total,
            default=feature.vram_usage_rate,
        )
        predicted_ram_usage = _safe_rate(
            worker.ram_used + assigned_count * required_ram,
            worker.ram_total,
            default=feature.ram_usage_rate,
        )
        predicted_gpu_utilization = min(
            1.0,
            feature.gpu_utilization_rate
            + assigned_count * SCALE_OUT_ESTIMATED_INSTANCE_GPU_UTIL,
        )
        resource_pressure += (
            SCALE_OUT_RESOURCE_VRAM_WEIGHT * predicted_vram_usage
            + SCALE_OUT_RESOURCE_GPU_WEIGHT * predicted_gpu_utilization
            + SCALE_OUT_RESOURCE_RAM_WEIGHT * predicted_ram_usage
        )

        if required_ram > 0:
            constraint_violation += max(
                assigned_count * required_ram - worker.ram_allocatable,
                0,
            ) / required_ram
        if required_vram > 0:
            constraint_violation += max(
                assigned_count * required_vram - worker.vram_allocatable,
                0,
            ) / required_vram

    selected_features = [
        features[index]
        for index in sorted(assigned_counts)
        if assigned_counts[index] > 0
    ]
    contention = sum(
        feature.worker_inflight
        + SCALE_OUT_CONTENTION_QPS_WEIGHT * feature.worker_qps
        for feature in selected_features
    )
    anti_affinity = sum(
        (
            feature.same_model_instances
            + assigned_counts.get(feature.candidate_index, 0)
        )
        ** 2
        for feature in selected_features
    )

    objectives = (resource_pressure, contention, float(anti_affinity))
    if energy_enabled:
        energy = sum(float(feature.power_watts or 0.0) for feature in selected_features)
        objectives = objectives + (energy,)

    return _ScaleOutPlan(
        assignments=assignments,
        objectives=objectives,
        constraint_violation=constraint_violation,
    )


def _initial_scale_out_population(
    candidate_count: int,
    new_instance_count: int,
) -> List[Tuple[int, ...]]:
    rng = random.Random(candidate_count * 131 + new_instance_count * 17)
    population: List[Tuple[int, ...]] = []

    for index in range(candidate_count):
        population.append(tuple([index] * new_instance_count))

    if candidate_count > 1:
        for offset in range(candidate_count):
            population.append(
                tuple(
                    (offset + gene_index) % candidate_count
                    for gene_index in range(new_instance_count)
                )
            )

    while len(population) < SCALE_OUT_NSGA_POPULATION_SIZE:
        population.append(
            tuple(
                rng.randrange(candidate_count)
                for _ in range(new_instance_count)
            )
        )

    return population[:SCALE_OUT_NSGA_POPULATION_SIZE]


def _fast_non_dominated_sort(plans: Sequence[_ScaleOutPlan]) -> List[List[_ScaleOutPlan]]:
    plan_list = list(plans)
    domination_counts: Dict[int, int] = {}
    dominated: Dict[int, List[int]] = {}
    front_indexes: List[List[int]] = [[]]

    for p_index, p_plan in enumerate(plan_list):
        dominated[p_index] = []
        domination_counts[p_index] = 0
        for q_index, q_plan in enumerate(plan_list):
            if p_index == q_index:
                continue
            if _constrained_dominates(p_plan, q_plan):
                dominated[p_index].append(q_index)
            elif _constrained_dominates(q_plan, p_plan):
                domination_counts[p_index] += 1
        if domination_counts[p_index] == 0:
            p_plan.rank = 0
            front_indexes[0].append(p_index)

    front_index = 0
    while front_index < len(front_indexes) and front_indexes[front_index]:
        next_front: List[int] = []
        for p_index in front_indexes[front_index]:
            for q_index in dominated[p_index]:
                domination_counts[q_index] -= 1
                if domination_counts[q_index] == 0:
                    plan_list[q_index].rank = front_index + 1
                    next_front.append(q_index)
        front_index += 1
        if next_front:
            front_indexes.append(next_front)

    fronts = [
        [plan_list[index] for index in indexes]
        for indexes in front_indexes
        if indexes
    ]
    for front in fronts:
        _assign_crowding_distance(front)
    return fronts


def _constrained_dominates(a: _ScaleOutPlan, b: _ScaleOutPlan) -> bool:
    a_feasible = a.constraint_violation <= EPSILON
    b_feasible = b.constraint_violation <= EPSILON
    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    if not a_feasible and not b_feasible:
        return a.constraint_violation + EPSILON < b.constraint_violation

    not_worse = all(
        a_value <= b_value + EPSILON
        for a_value, b_value in zip(a.objectives, b.objectives)
    )
    strictly_better = any(
        a_value + EPSILON < b_value
        for a_value, b_value in zip(a.objectives, b.objectives)
    )
    return not_worse and strictly_better


def _assign_crowding_distance(front: Sequence[_ScaleOutPlan]) -> None:
    if not front:
        return
    for plan in front:
        plan.crowding_distance = 0.0
    if len(front) <= 2:
        for plan in front:
            plan.crowding_distance = math.inf
        return

    objective_count = len(front[0].objectives)
    for objective_index in range(objective_count):
        sorted_front = sorted(
            front,
            key=lambda plan: plan.objectives[objective_index],
        )
        sorted_front[0].crowding_distance = math.inf
        sorted_front[-1].crowding_distance = math.inf
        min_value = sorted_front[0].objectives[objective_index]
        max_value = sorted_front[-1].objectives[objective_index]
        if abs(max_value - min_value) <= EPSILON:
            continue
        for index in range(1, len(sorted_front) - 1):
            if math.isinf(sorted_front[index].crowding_distance):
                continue
            previous_value = sorted_front[index - 1].objectives[objective_index]
            next_value = sorted_front[index + 1].objectives[objective_index]
            sorted_front[index].crowding_distance += (
                next_value - previous_value
            ) / (max_value - min_value)


def _tournament_select(
    population: Sequence[_ScaleOutPlan],
    rng: random.Random,
) -> _ScaleOutPlan:
    first = rng.choice(population)
    second = rng.choice(population)
    if first.rank != second.rank:
        return first if first.rank < second.rank else second
    if first.crowding_distance != second.crowding_distance:
        return (
            first
            if first.crowding_distance > second.crowding_distance
            else second
        )
    return first if first.assignments <= second.assignments else second


def _crossover(
    first: Tuple[int, ...],
    second: Tuple[int, ...],
    rng: random.Random,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if len(first) <= 1 or rng.random() >= 0.90:
        return first, second

    point = rng.randrange(1, len(first))
    return (
        first[:point] + second[point:],
        second[:point] + first[point:],
    )


def _mutate(
    assignments: Tuple[int, ...],
    candidate_count: int,
    rng: random.Random,
) -> Tuple[int, ...]:
    mutated = list(assignments)
    probability = max(SCALE_OUT_MUTATION_PROBABILITY, 1.0 / max(len(mutated), 1))
    for index in range(len(mutated)):
        if rng.random() < probability:
            mutated[index] = rng.randrange(candidate_count)
    return tuple(mutated)


def _next_generation(population: Sequence[_ScaleOutPlan]) -> List[_ScaleOutPlan]:
    fronts = _fast_non_dominated_sort(population)
    next_population: List[_ScaleOutPlan] = []
    for front in fronts:
        if len(next_population) + len(front) <= SCALE_OUT_NSGA_POPULATION_SIZE:
            next_population.extend(front)
            continue

        sorted_front = sorted(
            front,
            key=lambda plan: (
                -plan.crowding_distance,
                plan.constraint_violation,
                plan.objectives,
                plan.assignments,
            ),
        )
        remaining = SCALE_OUT_NSGA_POPULATION_SIZE - len(next_population)
        next_population.extend(sorted_front[:remaining])
        break
    return next_population


def _select_plan_by_topsis(
    plans: Sequence[_ScaleOutPlan],
    energy_enabled: bool,
) -> Optional[_ScaleOutPlan]:
    feasible_plans = [
        plan for plan in plans if plan.constraint_violation <= EPSILON
    ]
    candidates = feasible_plans or list(plans)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    weights = (
        SCALE_OUT_TOPSIS_WEIGHTS_WITH_ENERGY
        if energy_enabled
        else SCALE_OUT_TOPSIS_WEIGHTS
    )
    objective_count = len(candidates[0].objectives)
    weights = weights[:objective_count]

    columns = [
        [plan.objectives[index] for plan in candidates]
        for index in range(objective_count)
    ]
    normalized_columns = []
    for column in columns:
        min_value = min(column)
        max_value = max(column)
        if abs(max_value - min_value) <= EPSILON:
            normalized_columns.append([0.0 for _ in column])
        else:
            normalized_columns.append(
                [
                    (value - min_value) / (max_value - min_value)
                    for value in column
                ]
            )

    scored: List[Tuple[float, Tuple[float, ...], Tuple[int, ...], _ScaleOutPlan]] = []
    for plan_index, plan in enumerate(candidates):
        weighted_values = tuple(
            normalized_columns[objective_index][plan_index] * weights[objective_index]
            for objective_index in range(objective_count)
        )
        distance_to_best = math.sqrt(sum(value * value for value in weighted_values))
        distance_to_worst = math.sqrt(
            sum(
                (weights[index] - weighted_values[index])
                * (weights[index] - weighted_values[index])
                for index in range(objective_count)
            )
        )
        closeness = distance_to_worst / (
            distance_to_best + distance_to_worst + EPSILON
        )
        scored.append(
            (
                -closeness,
                plan.objectives,
                plan.assignments,
                plan,
            )
        )

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


def _select_plan_from_front(
    plans: Sequence[_ScaleOutPlan],
    features: Sequence[_ScaleOutWorkerFeature],
    required_ram: int,
    required_vram: int,
    energy_enabled: bool,
    decision_strategy: str,
) -> Optional[_ScaleOutPlan]:
    if decision_strategy == "binpack":
        return _select_plan_by_binpack(
            plans,
            features,
            required_ram,
            required_vram,
        )
    if decision_strategy == "spread":
        return _select_plan_by_spread(plans)
    return _select_plan_by_topsis(plans, energy_enabled)


def _select_plan_by_binpack(
    plans: Sequence[_ScaleOutPlan],
    features: Sequence[_ScaleOutWorkerFeature],
    required_ram: int,
    required_vram: int,
) -> Optional[_ScaleOutPlan]:
    candidates = [
        plan for plan in plans if plan.constraint_violation <= EPSILON
    ] or list(plans)
    if not candidates:
        return None

    scored = []
    for plan in candidates:
        assigned_counts = Counter(plan.assignments)
        residual_score = 0.0
        for index, count in assigned_counts.items():
            feature = features[index]
            if required_vram > 0:
                residual_score += max(
                    feature.worker.vram_allocatable - count * required_vram,
                    0,
                ) / required_vram
            if required_ram > 0:
                residual_score += max(
                    feature.worker.ram_allocatable - count * required_ram,
                    0,
                ) / required_ram
        scored.append(
            (
                residual_score,
                plan.objectives[1],
                plan.objectives[2],
                plan.objectives[0],
                plan.assignments,
                plan,
            )
        )

    scored.sort(key=lambda item: item[:-1])
    return scored[0][-1]


def _select_plan_by_spread(
    plans: Sequence[_ScaleOutPlan],
) -> Optional[_ScaleOutPlan]:
    candidates = [
        plan for plan in plans if plan.constraint_violation <= EPSILON
    ] or list(plans)
    if not candidates:
        return None

    scored = [
        (
            plan.objectives[2],
            plan.objectives[1],
            plan.objectives[0],
            plan.objectives[3] if len(plan.objectives) > 3 else 0.0,
            plan.assignments,
            plan,
        )
        for plan in candidates
    ]
    scored.sort(key=lambda item: item[:-1])
    return scored[0][-1]


def _scale_out_decision_strategy(hint: DeploymentHint) -> str:
    configured = os.getenv(SCALE_OUT_DECISION_STRATEGY_ENV, "auto").strip().lower()
    if configured in {"binpack", "spread", "topsis"}:
        return configured

    if hint.task_group == "vector_throughput":
        return "binpack"
    return "spread"


def _scale_out_rng_seed(
    features: Sequence[_ScaleOutWorkerFeature],
    required_ram: int,
    required_vram: int,
    new_instance_count: int,
) -> int:
    seed = required_ram * 31 + required_vram * 17 + new_instance_count * 13
    for feature in features:
        seed = seed * 131 + int(feature.worker.worker_id or 0)
    return seed


def _resource_headroom_ratio(allocatable: int, required: int) -> float:
    if required <= 0:
        return math.inf
    return allocatable / required


def _safe_rate(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or denominator <= 0:
        return default
    return min(max(numerator / denominator, 0.0), 1.0)


def _estimated_model_replica_claim(
    hint: DeploymentHint,
    state: SchedulingState,
) -> tuple[int, int]:
    ram_samples = [
        item.claimed_ram
        for item in hint.instance_metrics
        if item.model_id == hint.model_id and item.claimed_ram > 0
    ]
    vram_samples = [
        item.claimed_vram
        for item in hint.instance_metrics
        if item.model_id == hint.model_id and item.claimed_vram > 0
    ]

    if not ram_samples or not vram_samples:
        for instance in state.instances.values():
            if instance.model_id != hint.model_id:
                continue
            if instance.claimed_ram > 0:
                ram_samples.append(instance.claimed_ram)
            if instance.claimed_vram > 0:
                vram_samples.append(instance.claimed_vram)

    required_ram = max(ram_samples) if ram_samples else 0
    required_vram = max(vram_samples) if vram_samples else 0
    return required_ram, required_vram


def _worker_can_receive_scale_out(
    worker: WorkerState,
    required_ram: int,
    required_vram: int,
) -> bool:
    if worker.state != "ready":
        return False
    if worker.gpu_count <= 0:
        return False
    gpu_util = worker.gpu_utilization_rate_avg
    if gpu_util is not None and gpu_util >= WORKER_DEST_GPU_DANGER_RATE:
        return False
    if required_ram > 0 and worker.ram_allocatable < required_ram:
        return False
    if required_vram > 0 and worker.vram_allocatable < required_vram:
        return False
    return True


def _scale_out_worker_score(
    reason: str,
    gpu_util_score: float,
    vram_headroom_score: float,
    ram_headroom_score: float,
    inflight_score: float,
    same_model_spread_score: float,
    same_model_inflight_score: float,
    qps_score: float,
    preferred_score: float,
) -> float:
    if reason == "vram_pressure":
        weights = (0.15, 0.35, 0.10, 0.10, 0.15, 0.05, 0.10, 0.10)
    elif reason == "ttft_slo_pressure":
        weights = (0.30, 0.15, 0.10, 0.20, 0.10, 0.10, 0.05, 0.10)
    else:
        weights = (0.20, 0.20, 0.10, 0.20, 0.10, 0.10, 0.10, 0.10)

    features = (
        gpu_util_score,
        vram_headroom_score,
        ram_headroom_score,
        inflight_score,
        same_model_spread_score,
        same_model_inflight_score,
        qps_score,
        preferred_score,
    )
    return sum(weight * feature for weight, feature in zip(weights, features))


def _resource_headroom_score(
    total: int,
    used: int,
    allocatable: int,
    required: int,
) -> float:
    if required > 0:
        return min(max(allocatable / required, 0.0), 2.0) / 2.0
    if total > 0:
        return 1.0 - min(max(used / total, 0.0), 1.0)
    return 0.5


def _rate_from_percent(value: Optional[float]) -> float:
    if value is None:
        return 0.5
    return min(max(value / 100.0, 0.0), 1.0)


def _model_max_replicas(model: Model) -> int:
    for key in (
        "runtime_scale_max_replicas",
        "runtime_max_replicas",
        "max_replicas",
    ):
        if not model.meta or key not in model.meta:
            continue
        try:
            return max(int(model.meta[key]), 1)
        except (TypeError, ValueError):
            continue

    env_value = os.getenv("GPUSTACK_RUNTIME_SCALE_MAX_REPLICAS")
    if env_value:
        try:
            return max(int(env_value), 1)
        except ValueError:
            pass
    return SCALE_OUT_DEFAULT_MAX_REPLICAS


def _qps_is_rising(hints: List[DeploymentHint]) -> bool:
    if len(hints) < 2:
        return False

    first = _avg_instance_qps(hints[0])
    last = _avg_instance_qps(hints[-1])
    return (
        last >= first * QPS_RISING_RATIO
        and last - first >= QPS_RISING_MIN_DELTA
    )


def _avg_instance_qps(hint: DeploymentHint) -> float:
    samples = [
        item.instance_qps_10s
        for item in hint.instance_metrics
        if item.instance_qps_10s >= 0
    ]
    return sum(samples) / len(samples) if samples else 0.0


def _ttft_pressure_is_sustained(hints: List[DeploymentHint]) -> bool:
    ttft_hints = [
        hint
        for hint in hints
        if hint.reason == "latency_slo_risk" and hint.ttft_slo_ms is not None
    ]
    if len(ttft_hints) < SCALE_OUT_MIN_HINTS:
        return False

    for hint in ttft_hints[-SCALE_OUT_MIN_HINTS:]:
        samples = [
            item.recent_avg_ttft_ms
            for item in hint.instance_metrics
            if item.recent_avg_ttft_ms > 0
        ]
        if not samples:
            return False
        compensated_p99_proxy = max(samples) * TTFT_OBSERVATION_COMPENSATION_FACTOR
        if compensated_p99_proxy < hint.ttft_slo_ms * TTFT_SLO_APPROACH_RATIO:
            return False
    return True


def _vram_pressure_is_sustained(hints: List[DeploymentHint]) -> bool:
    pressure_hints = [
        hint
        for hint in hints
        if any(
            item.vram_usage_rate is not None
            and item.vram_usage_rate >= VRAM_HIGH_WATERMARK
            for item in hint.instance_metrics
        )
    ]
    return len(pressure_hints) >= SCALE_OUT_MIN_HINTS


def _running_model_instances(
    model_id: int,
    state: SchedulingState,
) -> List[InstanceState]:
    return [
        instance
        for instance in state.instances.values()
        if instance.model_id == model_id
        and instance.state == str(ModelInstanceStateEnum.RUNNING)
    ]


def _model_gpu_utilization_rate(
    model_id: int,
    state: SchedulingState,
) -> Optional[float]:
    worker_ids = {
        instance.worker_id
        for instance in _running_model_instances(model_id, state)
        if instance.worker_id is not None
    }
    samples = [
        worker.gpu_utilization_rate_avg
        for worker_id, worker in state.workers.items()
        if worker_id in worker_ids and worker.gpu_utilization_rate_avg is not None
    ]
    return sum(samples) / len(samples) if samples else None


def _destination_has_capacity(
    source: InstanceState,
    destination: InstanceState,
    state: Optional[SchedulingState] = None,
) -> bool:
    planned_qps = (
        _planned_qps_to_destination(destination.instance_id, state)
        if state is not None
        else 0.0
    )
    combined_qps = source.instance_qps_10s + destination.instance_qps_10s + planned_qps
    if combined_qps >= INSTANCE_QPS_SAFE_LIMIT:
        return False
    if destination.recent_avg_ttft_ms > INSTANCE_TTFT_GOOD_MS:
        return False
    return True


def _planned_qps_to_destination(
    destination_instance_id: int,
    state: SchedulingState,
) -> float:
    total = 0.0
    for plan in drain_plan_store.values():
        if plan.destination_instance_id != destination_instance_id:
            continue
        source = state.instances.get(plan.source_instance_id)
        if source is not None:
            total += source.instance_qps_10s
    return total


def _cluster_gpu_utilization_is_low(state: SchedulingState) -> bool:
    samples = [
        worker.gpu_utilization_rate_avg
        for worker in state.workers.values()
        if worker.gpu_utilization_rate_avg is not None and worker.gpu_count > 0
    ]
    if not samples:
        return False
    return sum(samples) / len(samples) <= CLUSTER_LOW_GPU_UTILIZATION_RATE


def _select_source_worker_for_compaction(
    state: SchedulingState,
    deployer: RuntimeAwareDeployer,
) -> Optional[WorkerState]:
    candidates = []
    for worker in state.workers.values():
        if deployer._worker_in_cooldown(worker.worker_id):
            continue
        if worker.state != "ready":
            continue
        if worker.gpu_count <= 0:
            continue
        gpu_util = worker.gpu_utilization_rate_avg
        if gpu_util is None or gpu_util > GPU_LOW_UTILIZATION_RATE:
            continue
        vram_usage_rate = (
            worker.vram_used / worker.vram_total if worker.vram_total > 0 else 0.0
        )
        if vram_usage_rate > WORKER_LOW_VRAM_USAGE_RATE:
            continue
        candidates.append((gpu_util, vram_usage_rate, worker.worker_id, worker))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _select_worker_compaction_destination(
    source: InstanceState,
    state: SchedulingState,
    deployer: RuntimeAwareDeployer,
    required_worker_id: Optional[int] = None,
) -> Optional[InstanceState]:
    candidates = []
    for destination in _running_model_instances(source.model_id, state):
        if destination.instance_id == source.instance_id:
            continue
        if destination.worker_id == source.worker_id:
            continue
        if required_worker_id is not None and destination.worker_id != required_worker_id:
            continue
        if is_instance_draining(destination.instance_id):
            continue
        if not _destination_has_capacity(source, destination, state):
            continue
        worker = state.workers.get(destination.worker_id)
        if worker is None:
            continue
        if deployer._worker_in_cooldown(worker.worker_id):
            continue
        gpu_util = worker.gpu_utilization_rate_avg
        if gpu_util is not None and gpu_util >= WORKER_DEST_GPU_DANGER_RATE:
            continue
        candidates.append((destination.resident_seconds, worker.worker_id, destination))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _select_worker_compaction_replacement_worker(
    source_worker_id: int,
    source_instance: InstanceState,
    state: SchedulingState,
    deployer: RuntimeAwareDeployer,
) -> Optional[WorkerState]:
    pinned_worker_id = deployer._worker_compaction_target_pins.get(source_worker_id)
    if pinned_worker_id is not None:
        pinned_worker = state.workers.get(pinned_worker_id)
        if pinned_worker is not None and _worker_can_receive_compaction_instance(
            pinned_worker,
            source_instance,
            deployer,
        ):
            return pinned_worker
        deployer._worker_compaction_target_pins.pop(source_worker_id, None)

    candidates = []
    for worker in state.workers.values():
        if worker.worker_id == source_worker_id:
            continue
        if not _worker_can_receive_compaction_instance(worker, source_instance, deployer):
            continue
        gpu_util = worker.gpu_utilization_rate_avg
        vram_usage_rate = (
            worker.vram_used / worker.vram_total if worker.vram_total > 0 else 0.0
        )
        candidates.append(
            (
                gpu_util if gpu_util is not None else 0.0,
                vram_usage_rate,
                worker.worker_id,
                worker,
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _worker_can_receive_compaction_instance(
    worker: WorkerState,
    source_instance: InstanceState,
    deployer: RuntimeAwareDeployer,
) -> bool:
    if worker.gpu_count <= 0:
        return False
    if worker.state != "ready":
        return False
    if deployer._worker_in_cooldown(worker.worker_id):
        return False
    gpu_util = worker.gpu_utilization_rate_avg
    if gpu_util is not None and gpu_util >= WORKER_DEST_GPU_DANGER_RATE:
        return False
    planned_vram = _planned_replacement_vram_to_worker(
        worker.worker_id,
        source_instance,
        deployer,
    )
    return _worker_has_vram_headroom(worker, source_instance, planned_vram)


def _worker_has_vram_headroom(
    worker: WorkerState,
    source: InstanceState,
    planned_vram: int = 0,
) -> bool:
    if source.claimed_vram <= 0:
        return True
    if worker.vram_allocatable <= 0:
        return False
    return source.claimed_vram + planned_vram <= worker.vram_allocatable


def _planned_replacement_vram_to_worker(
    worker_id: int,
    source_instance: InstanceState,
    deployer: RuntimeAwareDeployer,
) -> int:
    total = 0
    for plan in deployer._worker_compaction_plans.values():
        if plan.destination_worker_id != worker_id:
            continue
        if plan.source_instance_id == source_instance.instance_id:
            continue
        total += plan.source_claimed_vram
    return total


def _worker_has_running_instances(
    worker_id: int,
    state: SchedulingState,
    exclude_instance_id: Optional[int] = None,
) -> bool:
    return any(
        instance.worker_id == worker_id
        and instance.instance_id != exclude_instance_id
        and instance.state == str(ModelInstanceStateEnum.RUNNING)
        and not is_instance_draining(instance.instance_id)
        for instance in state.instances.values()
    )
