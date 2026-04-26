"""
Manual test script for the lightweight scheduling-state module.

Purpose:
- Open a real GPUStack database session.
- Call `get_current_state(...)`.
- Print a compact summary of the current resource and application-layer state.

Why this file exists:
- The project does not currently have a formal pytest layout.
- For this first iteration, a small async script is the fastest way to validate
  that the resource aggregation layer can read live GPUStack state.

How to run:
1. Start from the repository parent directory so `gpustack.*` imports resolve and
   the local `gpustack/logging.py` file does not shadow the Python stdlib module.
2. Run:

   cd /home/aue3n/gpustack-main
   python3 -m gpustack.runtime_state.test_get_current_state

Optional:
- Add `--verbose` to print more model / instance details.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Iterable

from gpustack.runtime_state import get_current_state
from gpustack.server.db import async_session


def _print_worker_summary(state, verbose: bool = False) -> None:
    print(f"workers: {len(state.workers)}")
    for worker in state.workers.values():
        print(
            "  "
            f"worker_id={worker.worker_id} "
            f"name={worker.worker_name} "
            f"state={worker.state} "
            f"cpu={worker.cpu_utilization_rate} "
            f"ram_allocatable={worker.ram_allocatable} "
            f"gpu_count={worker.gpu_count} "
            f"gpu_util_avg={worker.gpu_utilization_rate_avg} "
            f"vram_allocatable={worker.vram_allocatable}"
        )
        if verbose:
            for gpu in worker.gpus:
                print(
                    "    "
                    f"gpu_index={gpu.gpu_index} "
                    f"name={gpu.gpu_name} "
                    f"util={gpu.gpu_utilization_rate} "
                    f"vram_total={gpu.vram_total} "
                    f"vram_used={gpu.vram_used} "
                    f"vram_allocated={gpu.vram_allocated} "
                    f"vram_allocatable={gpu.vram_allocatable}"
                )


def _iter_models(state, limit: int | None) -> Iterable:
    models = sorted(
        state.models.values(),
        key=lambda item: (item.model_qps_10s, item.model_inflight, item.model_name),
        reverse=True,
    )
    if limit is None:
        return models
    return models[:limit]


def _print_model_summary(state, limit: int | None, verbose: bool = False) -> None:
    print(f"models: {len(state.models)}")
    for model in _iter_models(state, limit):
        print(
            "  "
            f"model_id={model.model_id} "
            f"name={model.model_name} "
            f"running_instances={model.running_instances}/{model.total_instances} "
            f"allocated_ram={model.allocated_ram} "
            f"allocated_vram={model.allocated_vram} "
            f"qps_10s={model.model_qps_10s:.3f} "
            f"inflight={model.model_inflight} "
            f"avg_prompt_tokens_10s={model.avg_prompt_tokens_10s:.2f}"
        )
        if verbose:
            print(f"    task_type_mix={model.task_type_mix}")
            print(f"    slo_mix={model.slo_mix}")


def _print_instance_summary(state, limit: int | None) -> None:
    instances = sorted(
        state.instances.values(),
        key=lambda item: (item.instance_qps_10s, item.instance_inflight, item.instance_id),
        reverse=True,
    )
    if limit is not None:
        instances = instances[:limit]

    print(f"instances: {len(state.instances)}")
    for instance in instances:
        print(
            "  "
            f"instance_id={instance.instance_id} "
            f"model_id={instance.model_id} "
            f"worker_id={instance.worker_id} "
            f"state={instance.state} "
            f"gpu_indexes={instance.gpu_indexes} "
            f"claimed_ram={instance.claimed_ram} "
            f"claimed_vram={instance.claimed_vram} "
            f"qps_10s={instance.instance_qps_10s:.3f} "
            f"inflight={instance.instance_inflight} "
            f"recent_avg_prompt_tokens={instance.recent_avg_prompt_tokens:.2f} "
            f"recent_avg_ttft_ms={instance.recent_avg_ttft_ms:.2f}"
        )


async def _main(verbose: bool, model_limit: int | None, instance_limit: int | None) -> None:
    """
    Open a real DB session and fetch one scheduling snapshot.
    """

    async with async_session() as session:
        state = await get_current_state(session)

    print(f"generated_at={state.generated_at.isoformat()}")
    _print_worker_summary(state, verbose=verbose)
    _print_model_summary(state, limit=model_limit, verbose=verbose)
    _print_instance_summary(state, limit=instance_limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the current runtime scheduling state.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-GPU details and model mixes.",
    )
    parser.add_argument(
        "--model-limit",
        type=int,
        default=10,
        help="Maximum number of models to print.",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=20,
        help="Maximum number of instances to print.",
    )
    args = parser.parse_args()

    asyncio.run(
        _main(
            verbose=args.verbose,
            model_limit=args.model_limit,
            instance_limit=args.instance_limit,
        )
    )


if __name__ == "__main__":
    main()
