import asyncio
import logging

from gpustack.scheduler.scheduler import Scheduler
from gpustack.runtime_scheduler.deployer import (
    SCALE_IN_INTERVAL_SECONDS,
    RuntimeAwareDeployer,
)


logger = logging.getLogger(__name__)


class RuntimeAwareScheduler(Scheduler):
    """
    Entry point for the custom runtime-aware scheduler.

    The initial implementation intentionally preserves the original GPUStack
    scheduling behavior by subclassing Scheduler without overriding methods.
    Runtime-aware scheduling logic can be added here without editing the
    original scheduler module.
    """

    def __init__(self, *args, deployer: RuntimeAwareDeployer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._runtime_deployer = deployer or RuntimeAwareDeployer()

    async def start(self):
        asyncio.create_task(self._runtime_scale_in_cycle())
        await super().start()

    async def _runtime_scale_in_cycle(self):
        while True:
            try:
                await asyncio.sleep(SCALE_IN_INTERVAL_SECONDS)
                await self._runtime_deployer.reconcile_scale_in()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Runtime-aware scale-in cycle failed: %s", e)
