"""
Manual test script that simulates one user request flowing through the runtime-state
tracker and then inspects the resulting scheduling snapshot.

Purpose:
- Emulate the exact lifecycle used by the lightweight state module:
  1. extract request metadata
  2. register request start
  3. bind the request to a running model instance
  4. read scheduling state while the request is inflight
  5. finish the request and read scheduling state again

This does not send a real HTTP request to a model backend.
It tests the scheduling-state module in isolation using real database-backed
GPUStack model / instance records.

How to run:
1. Start from the repository parent directory:

   cd /home/aue3n/gpustack-main

2. Pick a model that already has at least one RUNNING instance.
3. Run:

   python3 -m gpustack.runtime_state.test_simulated_request --model-name <YOUR_MODEL_NAME>

Optional:
- `--path` lets you simulate a different task shape such as embeddings.
- `--slo-class` lets you inject a scheduling hint.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from gpustack.runtime_state import (
    extract_request_meta,
    get_current_state,
    request_bound,
    request_finished,
    request_started,
)
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.db import async_session


@dataclass
class _DummyURL:
    path: str


class _DummyRequest:
    """
    Tiny request-like object compatible with `extract_request_meta(...)`.

    The tracker only reads:
    - request.url.path
    - request.headers
    So we do not need a full FastAPI Request instance for this simulation.
    """

    def __init__(self, path: str, headers: dict[str, str]):
        self.url = _DummyURL(path=path)
        self.headers = headers


def _default_body_for_path(path: str, model_name: str) -> dict:
    """
    Build a small request body that looks like a real OpenAI-compatible request.
    """

    if path.endswith("/embeddings"):
        return {
            "model": model_name,
            "input": [
                "GPUStack scheduling test text.",
                "This request is used to verify the runtime_state tracker.",
            ],
        }

    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a scheduling test assistant."},
            {
                "role": "user",
                "content": "Explain how the scheduler should balance latency and utilization.",
            },
        ],
        "max_tokens": 128,
        "stream": False,
    }


async def _load_model_and_instance(session, model_name: str) -> tuple[Model, ModelInstance]:
    """
    Find the target model plus one RUNNING instance that can be used as the
    simulated binding target.
    """

    model = await Model.one_by_field(session, "name", model_name)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' not found.")

    instances = await ModelInstance.all_by_fields(
        session,
        fields={"model_id": model.id, "state": ModelInstanceStateEnum.RUNNING},
    )
    if not instances:
        raise RuntimeError(
            f"Model '{model_name}' has no RUNNING instances. "
            "Start at least one instance before running this simulation."
        )

    return model, instances[0]


def _print_model_view(state, model_id: int) -> None:
    model_state = state.models.get(model_id)
    if model_state is None:
        print(f"model_id={model_id} not present in scheduling snapshot")
        return

    print(
        "model_state "
        f"qps_10s={model_state.model_qps_10s:.3f} "
        f"inflight={model_state.model_inflight} "
        f"avg_prompt_tokens_10s={model_state.avg_prompt_tokens_10s:.2f} "
        f"task_type_mix={model_state.task_type_mix} "
        f"slo_mix={model_state.slo_mix}"
    )


def _print_instance_view(state, instance_id: int) -> None:
    instance_state = state.instances.get(instance_id)
    if instance_state is None:
        print(f"instance_id={instance_id} not present in scheduling snapshot")
        return

    print(
        "instance_state "
        f"qps_10s={instance_state.instance_qps_10s:.3f} "
        f"inflight={instance_state.instance_inflight} "
        f"recent_avg_prompt_tokens={instance_state.recent_avg_prompt_tokens:.2f} "
        f"recent_avg_ttft_ms={instance_state.recent_avg_ttft_ms:.2f}"
    )


async def _main(model_name: str, path: str, slo_class: str | None) -> None:
    async with async_session() as session:
        model, instance = await _load_model_and_instance(session, model_name)

        headers: dict[str, str] = {}
        if slo_class:
            headers["X-GPUStack-SLO-Class"] = slo_class

        dummy_request = _DummyRequest(path=path, headers=headers)
        body_json = _default_body_for_path(path, model.name)

        # Step 1: extract application-layer metadata from the simulated request.
        request_meta = extract_request_meta(
            dummy_request,
            model_name=model.name,
            body_json=body_json,
            stream=bool(body_json.get("stream", False)),
        )
        request_meta.model_id = model.id

        print("current_request")
        print(
            "  "
            f"task_type={request_meta.task_type} "
            f"slo_class={request_meta.slo_class} "
            f"estimated_prompt_tokens={request_meta.estimated_prompt_tokens} "
            f"estimated_output_tokens={request_meta.estimated_output_tokens}"
        )

        # Step 2: register the request lifecycle in the in-memory tracker.
        request_id = await request_started(request_meta)
        await request_bound(
            request_id,
            model_id=model.id,
            instance_id=instance.id,
            worker_id=instance.worker_id,
        )

        # Step 3: inspect scheduling state while the request is inflight.
        inflight_state = await get_current_state(session, current_request=request_meta)
        print("during_request")
        _print_model_view(inflight_state, model.id)
        _print_instance_view(inflight_state, instance.id)

        # Step 4: finish the request with fake runtime metrics.
        await request_finished(
            request_id,
            prompt_tokens=max(request_meta.estimated_prompt_tokens, 1),
            completion_tokens=max(request_meta.estimated_output_tokens, 16),
            ttft_ms=85.0,
            ok=True,
        )

        # Step 5: inspect the state again after the request has completed.
        finished_state = await get_current_state(session)
        print("after_request")
        _print_model_view(finished_state, model.id)
        _print_instance_view(finished_state, instance.id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate one request lifecycle against the runtime_state module."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of an existing GPUStack model that already has a RUNNING instance.",
    )
    parser.add_argument(
        "--path",
        default="/v1/chat/completions",
        help="OpenAI-compatible request path to simulate.",
    )
    parser.add_argument(
        "--slo-class",
        default=None,
        help="Optional SLO class injected through the same header path used in production.",
    )
    args = parser.parse_args()

    asyncio.run(_main(args.model_name, args.path, args.slo_class))


if __name__ == "__main__":
    main()
