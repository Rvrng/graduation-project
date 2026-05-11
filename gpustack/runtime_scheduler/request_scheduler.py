import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.runtime_state import get_current_state
from gpustack.runtime_state.types import (
    InstanceState,
    ModelState,
    RequestAppMeta,
    WorkerState,
)
from gpustack.runtime_scheduler.pareto import (
    ExactParetoFrontSolver,
    ParetoFrontSolver,
    build_request_pareto_objectives,
)
from gpustack.runtime_scheduler.deployer import RuntimeAwareDeployer
from gpustack.runtime_scheduler.deployer import is_instance_draining
from gpustack.runtime_scheduler.scoring import RequestScoreSelector
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.workers import Worker, WorkerStateEnum

logger = logging.getLogger(__name__)


@dataclass
class RequestScheduleCandidate:
    instance: ModelInstance
    instance_state: Optional[InstanceState]
    worker_state: Optional[WorkerState]
    model_state: Optional[ModelState]
    hard_filter_reasons: List[str] = field(default_factory=list)


class RuntimeAwareRequestScheduler:
    """
    Request-level scheduler for selecting a RUNNING model instance.
    """

    def __init__(
        self,
        pareto_solver: Optional[ParetoFrontSolver] = None,
        score_selector: Optional[RequestScoreSelector] = None,
        deployer: Optional[RuntimeAwareDeployer] = None,
    ):
        self._pareto_solver = pareto_solver or ExactParetoFrontSolver()
        self._score_selector = score_selector or RequestScoreSelector()
        self._deployer = deployer or RuntimeAwareDeployer()

    async def filter_candidates(
        self,
        session: AsyncSession,
        current_request: RequestAppMeta,
        running_instances: List[ModelInstance],
    ) -> List[RequestScheduleCandidate]:
        state = await get_current_state(session, current_request=current_request)
        workers = await Worker.all(session)
        workers_by_id: Dict[int, Worker] = {
            worker.id: worker for worker in workers if worker.id is not None
        }

        strict_candidates = self._filter_candidates(
            current_request=current_request,
            running_instances=running_instances,
            workers_by_id=workers_by_id,
            instance_states=state.instances,
            worker_states=state.workers,
            model_states=state.models,
            require_worker_ready=True,
        )
        if strict_candidates:
            return strict_candidates

        relaxed_candidates = self._filter_candidates(
            current_request=current_request,
            running_instances=running_instances,
            workers_by_id=workers_by_id,
            instance_states=state.instances,
            worker_states=state.workers,
            model_states=state.models,
            require_worker_ready=False,
        )
        if relaxed_candidates:
            logger.warning(
                "Runtime-aware request hard filtering found no READY-worker "
                "candidates for model_id=%s; using basic reachable candidates.",
                current_request.model_id,
            )
        return relaxed_candidates

    def pareto_front(
        self,
        candidates: List[RequestScheduleCandidate],
    ) -> List[RequestScheduleCandidate]:
        if len(candidates) <= 1:
            return candidates

        objectives = build_request_pareto_objectives(candidates)
        front = self._pareto_solver.solve(objectives)
        if not front:
            return candidates

        logger.debug(
            "Runtime-aware request Pareto front selected %s/%s candidates.",
            len(front),
            len(candidates),
        )
        return [objective.candidate for objective in front]

    async def select_candidate(
        self,
        candidates: List[RequestScheduleCandidate],
        current_request: RequestAppMeta,
    ) -> Optional[RequestScheduleCandidate]:
        selected = self._score_selector.select(candidates, current_request)
        if selected is None:
            return None

        if selected.deployment_hint is not None:
            await self._deployer.report_hint(selected.deployment_hint)

        logger.debug(
            "Runtime-aware request score selected instance_id=%s score=%.4f "
            "task_group=%s context_bucket=%s slo_class=%s.",
            selected.candidate.instance.id,
            selected.score,
            selected.task_group,
            selected.context_bucket,
            selected.slo_class,
        )
        return selected.candidate

    def _filter_candidates(
        self,
        current_request: RequestAppMeta,
        running_instances: List[ModelInstance],
        workers_by_id: Dict[int, Worker],
        instance_states: Dict[int, InstanceState],
        worker_states: Dict[int, WorkerState],
        model_states: Dict[int, ModelState],
        require_worker_ready: bool,
    ) -> List[RequestScheduleCandidate]:
        candidates: List[RequestScheduleCandidate] = []
        model_id = current_request.model_id

        for instance in running_instances:
            reasons = self._hard_filter_reasons(
                instance=instance,
                model_id=model_id,
                workers_by_id=workers_by_id,
                require_worker_ready=require_worker_ready,
            )
            if reasons:
                logger.debug(
                    "Filtered request candidate instance_id=%s model_id=%s reasons=%s",
                    instance.id,
                    instance.model_id,
                    reasons,
                )
                continue

            candidates.append(
                RequestScheduleCandidate(
                    instance=instance,
                    instance_state=instance_states.get(instance.id),
                    worker_state=worker_states.get(instance.worker_id),
                    model_state=model_states.get(instance.model_id),
                )
            )

        return candidates

    @staticmethod
    def _hard_filter_reasons(
        instance: ModelInstance,
        model_id: Optional[int],
        workers_by_id: Dict[int, Worker],
        require_worker_ready: bool,
    ) -> List[str]:
        reasons: List[str] = []

        if model_id is not None and instance.model_id != model_id:
            reasons.append("model_id_mismatch")

        if instance.state != ModelInstanceStateEnum.RUNNING:
            reasons.append("instance_not_running")

        if is_instance_draining(instance.id):
            reasons.append("instance_draining")

        if instance.worker_id is None:
            reasons.append("missing_worker_id")
            worker = None
        else:
            worker = workers_by_id.get(instance.worker_id)
            if worker is None:
                reasons.append("worker_not_found")

        if require_worker_ready and worker is not None:
            if worker.state != WorkerStateEnum.READY:
                reasons.append("worker_not_ready")

        if not instance.worker_ip:
            reasons.append("missing_instance_worker_ip")

        if not instance.port:
            reasons.append("missing_instance_port")

        if worker is not None and not worker.port:
            reasons.append("missing_worker_port")

        return reasons
