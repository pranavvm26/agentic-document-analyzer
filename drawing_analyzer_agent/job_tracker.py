"""
Job tracking for the IEM Drawing Review Agent.

Provides a thread-safe in-memory job store that tracks execution state,
phase progress, token usage, and timing. Designed to be queried by the
REST API for real-time status updates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PhaseInfo:
    """Tracks a single workflow phase."""

    name: str
    description: str
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "pending"


@dataclass
class JobState:
    """Full state of a review job."""

    job_id: str
    mode: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    current_phase: str = ""
    current_step: int = 0
    total_steps_estimate: int = 0
    tool_calls: int = 0

    phases: list[PhaseInfo] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0

    report_path: str = ""
    report_s3_uri: str = ""
    report_summary: str = ""
    error: str = ""

    schematic_path: str = ""
    wiring_diagram_path: str = ""
    output_s3_path: str = ""

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return round(end - self.started_at, 1)

    @property
    def progress_pct(self) -> int:
        if self.total_steps_estimate <= 0:
            return 0
        return min(100, int(self.current_step / self.total_steps_estimate * 100))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "mode": self.mode,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "current_phase": self.current_phase,
            "current_step": self.current_step,
            "total_steps_estimate": self.total_steps_estimate,
            "progress_pct": self.progress_pct,
            "tool_calls": self.tool_calls,
            "phases": [
                {"name": p.name, "description": p.description,
                 "status": p.status, "started_at": p.started_at,
                 "completed_at": p.completed_at}
                for p in self.phases
            ],
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.input_tokens + self.output_tokens,
            },
            "report_path": self.report_path,
            "report_s3_uri": self.report_s3_uri,
            "report_summary": self.report_summary,
            "error": self.error,
            "log_count": len(self.log),
        }


STEP_ESTIMATES = {
    "bom": 40,
    "circuit": 60,
    "both": 90,
    "wdanalysis": 50,
}


class JobTracker:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, mode: str, **kwargs: Any) -> JobState:
        job = JobState(
            job_id=job_id,
            mode=mode,
            total_steps_estimate=STEP_ESTIMATES.get(mode, 50),
            **kwargs,
        )
        self._init_phases(job)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def start(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

    def update_phase(self, job_id: str, phase: str, step: int = 0) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if step > 0:
                job.current_step = step
            if phase and phase != job.current_phase:
                for p in job.phases:
                    if p.name == job.current_phase and p.status == "running":
                        p.status = "completed"
                        p.completed_at = time.time()
                for p in job.phases:
                    if p.name == phase:
                        p.status = "running"
                        p.started_at = time.time()
                        break
                job.current_phase = phase

    def add_tool_call(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.tool_calls += 1

    def add_tokens(self, job_id: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.input_tokens += input_tokens
                job.output_tokens += output_tokens

    def add_log(self, job_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                entry["timestamp"] = time.time()
                job.log.append(entry)

    def complete(self, job_id: str, report_path: str = "", report_s3_uri: str = "", summary: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.report_path = report_path
                job.report_s3_uri = report_s3_uri
                job.report_summary = summary
                for p in job.phases:
                    if p.status == "running":
                        p.status = "completed"
                        p.completed_at = time.time()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = error

    def _init_phases(self, job: JobState) -> None:
        mode = job.mode
        if mode in ("bom", "both"):
            job.phases.extend([
                PhaseInfo("A", "Extract schematic pages + Drawing Index"),
                PhaseInfo("B", "Extract WD pages + Drawing Index"),
                PhaseInfo("C", "Compare BOMs with vision"),
                PhaseInfo("D", "Generate BOM report"),
            ])
        if mode in ("circuit", "both"):
            job.phases.extend([
                PhaseInfo("E", "Extract schematic 3L diagrams"),
                PhaseInfo("F", "Extract WD 3L diagrams + continuations"),
                PhaseInfo("G", "Compare circuit pairs"),
            ])
        if mode == "wdanalysis":
            job.phases.extend([
                PhaseInfo("B", "Extract WD pages + Drawing Index"),
                PhaseInfo("H", "Analyze WD diagrams for errors"),
            ])


tracker = JobTracker()
