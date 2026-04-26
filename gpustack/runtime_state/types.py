from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RequestAppMeta(BaseModel):
    """
    Application-layer metadata extracted from the incoming request.

    This object represents the "intent" of a request before any instance is picked.
    It is intentionally lightweight so the scheduler can evaluate a new request
    without touching the full request payload again.
    """

    request_id: Optional[str] = None
    model_name: str
    model_id: Optional[int] = None
    task_type: str
    slo_class: str
    ttft_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None
    estimated_prompt_tokens: int = 0
    estimated_output_tokens: int = 0
    stream: bool = False
    started_at: Optional[datetime] = None


class GPUState(BaseModel):
    """
    Aggregated state of one GPU visible to the scheduling layer.
    """

    gpu_index: int
    gpu_name: str = ""
    gpu_utilization_rate: Optional[float] = None
    vram_total: int = 0
    vram_used: int = 0
    vram_allocated: int = 0
    vram_allocatable: int = 0
    power_watts: Optional[float] = None


class WorkerState(BaseModel):
    """
    Scheduling-oriented snapshot of one worker.
    """

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
    power_watts: Optional[float] = None
    gpus: List[GPUState] = Field(default_factory=list)


class InstanceState(BaseModel):
    """
    Scheduling-oriented snapshot of one model instance.
    """

    instance_id: int
    model_id: int
    worker_id: Optional[int] = None
    state: str
    gpu_indexes: List[int] = Field(default_factory=list)
    claimed_ram: int = 0
    claimed_vram: int = 0
    instance_qps_10s: float = 0.0
    instance_inflight: int = 0
    recent_avg_prompt_tokens: float = 0.0
    recent_avg_ttft_ms: float = 0.0


class ModelState(BaseModel):
    """
    Scheduling-oriented snapshot of one model.
    """

    model_id: int
    model_name: str
    total_instances: int = 0
    running_instances: int = 0
    allocated_ram: int = 0
    allocated_vram: int = 0
    model_qps_10s: float = 0.0
    model_inflight: int = 0
    avg_prompt_tokens_10s: float = 0.0
    task_type_mix: Dict[str, int] = Field(default_factory=dict)
    slo_mix: Dict[str, int] = Field(default_factory=dict)


class SchedulingState(BaseModel):
    """
    Unified scheduling snapshot.

    The scheduler only needs one object that combines:
    - the current incoming request metadata
    - worker / instance / model resource state
    - recent application-side load indicators
    """

    generated_at: datetime
    current_request: Optional[RequestAppMeta] = None
    workers: Dict[int, WorkerState] = Field(default_factory=dict)
    models: Dict[int, ModelState] = Field(default_factory=dict)
    instances: Dict[int, InstanceState] = Field(default_factory=dict)
