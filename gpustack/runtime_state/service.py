from datetime import datetime, timezone
from statistics import mean
from typing import Dict, Optional

from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.policies.utils import get_worker_allocatable_resource
from gpustack.runtime_state.tracker import (
    get_instance_runtime_metrics,
    get_model_runtime_metrics,
)
from gpustack.runtime_state.types import (
    GPUState,
    InstanceState,
    ModelState,
    RequestAppMeta,
    SchedulingState,
    WorkerState,
)
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.workers import Worker


def _safe_int(value: Optional[int]) -> int:
    return int(value or 0)


def _state_value(value) -> str:
    return getattr(value, "value", str(value))


def _seconds_since(value: Optional[datetime], now: datetime) -> float:
    if value is None:
        return 0.0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max((now - value).total_seconds(), 0.0)


def _extract_power_watts(obj) -> Optional[float]:
    """
    Best-effort power extraction.

    Current GPUStack schemas do not guarantee power fields, but keeping the logic in
    one place makes the state object ready for future detector/exporter extensions.
    """

    for attr in ("power_watts", "power_used", "power_draw", "power_usage", "power"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _sum_instance_claims(instance: ModelInstance) -> tuple[int, int]:
    """
    Sum the main instance claim together with subordinate worker claims so the
    returned footprint reflects the full distributed deployment.
    """

    claimed_ram = _safe_int(getattr(instance.computed_resource_claim, "ram", 0))
    claimed_vram = sum(
        _safe_int(vram)
        for vram in (getattr(instance.computed_resource_claim, "vram", {}) or {}).values()
    )

    if (
        instance.distributed_servers
        and instance.distributed_servers.subordinate_workers
    ):
        for subordinate in instance.distributed_servers.subordinate_workers:
            claim = subordinate.computed_resource_claim
            claimed_ram += _safe_int(getattr(claim, "ram", 0))
            claimed_vram += sum(
                _safe_int(vram)
                for vram in (getattr(claim, "vram", {}) or {}).values()
            )

    return claimed_ram, claimed_vram


async def get_current_state(
    session: AsyncSession,
    current_request: Optional[RequestAppMeta] = None,
) -> SchedulingState:
    """
    Build a unified scheduling snapshot from:
    - current worker / instance / model resource state already tracked by GPUStack
    - recent application-layer load tracked at the request entry point

    The scheduler should treat this function as its single read entrypoint.
    """

    workers = await Worker.all(session)
    model_instances = await ModelInstance.all(
        session,
        options=[selectinload(ModelInstance.model)],
    )

    now = datetime.now(timezone.utc)
    state = SchedulingState(
        generated_at=now,
        current_request=current_request,
    )

    model_map: Dict[int, Model] = {}
    for instance in model_instances:
        if instance.model is not None and instance.model.id is not None:
            model_map[instance.model.id] = instance.model

    if (
        current_request is not None
        and current_request.model_id is not None
        and current_request.model_id not in model_map
    ):
        model = await Model.one_by_id(session, current_request.model_id)
        if model is not None and model.id is not None:
            model_map[model.id] = model

    for worker in workers:
        if worker.status is None:
            continue

        allocatable = get_worker_allocatable_resource(model_instances, worker)

        gpu_states = []
        gpu_util_samples = []
        total_vram = 0
        used_vram = 0
        allocatable_vram = 0

        for gpu in worker.status.gpu_devices or []:
            gpu_total = _safe_int(getattr(gpu.memory, "total", 0))
            gpu_used = _safe_int(getattr(gpu.memory, "used", 0))
            gpu_allocatable = _safe_int((allocatable.vram or {}).get(gpu.index, 0))
            gpu_allocated = _safe_int(getattr(gpu.memory, "allocated", 0))
            if gpu_allocated == 0 and gpu_total:
                # Fallback when allocated bytes are not explicitly injected.
                gpu_allocated = max(
                    gpu_total - gpu_allocatable - _safe_int(worker.system_reserved.vram),
                    0,
                )

            total_vram += gpu_total
            used_vram += gpu_used
            allocatable_vram += gpu_allocatable

            gpu_util = getattr(gpu.core, "utilization_rate", None)
            if gpu_util is not None:
                gpu_util_samples.append(gpu_util)

            gpu_power_watts = _extract_power_watts(gpu)
            gpu_states.append(
                GPUState(
                    gpu_index=int(gpu.index or 0),
                    gpu_name=gpu.name or "",
                    gpu_utilization_rate=gpu_util,
                    vram_total=gpu_total,
                    vram_used=gpu_used,
                    vram_allocated=gpu_allocated,
                    vram_allocatable=gpu_allocatable,
                    power_watts=gpu_power_watts,
                )
            )

        ram_total = _safe_int(getattr(worker.status.memory, "total", 0))
        ram_used = _safe_int(getattr(worker.status.memory, "used", 0))
        ram_allocated = _safe_int(getattr(worker.status.memory, "allocated", 0))
        if ram_allocated == 0 and ram_total:
            ram_allocated = max(
                ram_total - _safe_int(allocatable.ram) - _safe_int(worker.system_reserved.ram),
                0,
            )

        gpu_power_samples = [
            gpu.power_watts for gpu in gpu_states if gpu.power_watts is not None
        ]
        worker_power_watts = _extract_power_watts(worker.status)
        if worker_power_watts is None and gpu_power_samples:
            worker_power_watts = sum(gpu_power_samples)

        state.workers[worker.id] = WorkerState(
            worker_id=worker.id,
            worker_name=worker.name,
            state=_state_value(worker.state),
            cpu_utilization_rate=getattr(worker.status.cpu, "utilization_rate", None),
            ram_total=ram_total,
            ram_used=ram_used,
            ram_allocated=ram_allocated,
            ram_allocatable=_safe_int(allocatable.ram),
            gpu_count=len(gpu_states),
            gpu_utilization_rate_avg=(mean(gpu_util_samples) if gpu_util_samples else None),
            vram_total=total_vram,
            vram_used=used_vram,
            vram_allocatable=allocatable_vram,
            power_watts=worker_power_watts,
            gpus=gpu_states,
        )

    model_to_instances: Dict[int, list[ModelInstance]] = {}
    for instance in model_instances:
        model_to_instances.setdefault(instance.model_id, []).append(instance)

    for instance in model_instances:
        claimed_ram, claimed_vram = _sum_instance_claims(instance)
        app_metrics = await get_instance_runtime_metrics(instance.id)

        state.instances[instance.id] = InstanceState(
            instance_id=instance.id,
            model_id=instance.model_id,
            worker_id=instance.worker_id,
            state=_state_value(instance.state),
            gpu_indexes=list(instance.gpu_indexes or []),
            claimed_ram=claimed_ram,
            claimed_vram=claimed_vram,
            resident_seconds=_seconds_since(
                instance.last_restart_time or instance.created_at,
                now,
            ),
            instance_qps_10s=app_metrics["qps_10s"],
            instance_inflight=app_metrics["inflight"],
            recent_avg_prompt_tokens=app_metrics["recent_avg_prompt_tokens"],
            recent_avg_ttft_ms=app_metrics["recent_avg_ttft_ms"],
        )

    for model_id, model in model_map.items():
        instances = model_to_instances.get(model_id, [])
        allocated_ram = 0
        allocated_vram = 0
        for instance in instances:
            claimed_ram, claimed_vram = _sum_instance_claims(instance)
            allocated_ram += claimed_ram
            allocated_vram += claimed_vram

        app_metrics = await get_model_runtime_metrics(model_id, model.name)

        state.models[model_id] = ModelState(
            model_id=model_id,
            model_name=model.name,
            total_instances=len(instances),
            running_instances=sum(
                1
                for instance in instances
                if instance.state == ModelInstanceStateEnum.RUNNING
            ),
            allocated_ram=allocated_ram,
            allocated_vram=allocated_vram,
            model_qps_10s=app_metrics["qps_10s"],
            model_inflight=app_metrics["inflight"],
            avg_prompt_tokens_10s=app_metrics["avg_prompt_tokens_10s"],
            task_type_mix=app_metrics["task_type_mix"],
            slo_mix=app_metrics["slo_mix"],
        )

    return state
