from gpustack.scheduler.scheduler import Scheduler


class RuntimeAwareScheduler(Scheduler):
    """
    Entry point for the custom runtime-aware scheduler.

    The initial implementation intentionally preserves the original GPUStack
    scheduling behavior by subclassing Scheduler without overriding methods.
    Runtime-aware scheduling logic can be added here without editing the
    original scheduler module.
    """

    pass
