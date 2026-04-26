"""
FastAPI REST API for the IEM Drawing Review Agent.

Async job-based API with progress tracking, token counting, and S3 integration.

Usage:
    uvicorn iem_bom_agent.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import uuid
from enum import Enum
from pathlib import Path

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from iem_bom_agent.agent.graph import run_review
from iem_bom_agent.job_tracker import JobStatus, tracker

logger = logging.getLogger(__name__)

REPORT_DIR = Path(os.environ.get("REPORT_OUTPUT_DIR", "/tmp/iem_reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="IEM Drawing Review Agent",
    description="Async REST API for IEM drawing review with job tracking.",
    version="2.0.0",
)


class ReviewMode(str, Enum):
    bom = "bom"
    circuit = "circuit"
    both = "both"
    wdanalysis = "wdanalysis"


class ReviewRequest(BaseModel):
    schematic_path: str = Field(description="Local path or S3 URI to schematic PDF.")
    wiring_diagram_path: str = Field(description="Local path or S3 URI to WD PDF.")
    output_s3_path: str = Field(description="S3 URI where the report will be uploaded.")
    mode: ReviewMode = Field(default=ReviewMode.both)
    verbose: bool = Field(default=False)


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    status_url: str
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "iem-review-agent"


def _resolve_path(path: str) -> str:
    """Download S3 URI to temp file, or validate local path."""
    if not path.startswith("s3://"):
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"File not found: {path}")
        return path
    try:
        parts = path[5:].split("/", 1)
        bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
        s3 = boto3.client("s3")
        suffix = os.path.splitext(key)[1] or ".pdf"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, prefix="iem_s3_", delete=False)
        s3.download_file(bucket, key, tmp.name)
        return tmp.name
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"S3 download failed: {path} — {exc}") from exc


def _upload_to_s3(local_path: str, output_s3_path: str, job_id: str) -> str:
    """Upload report to user-specified S3 path."""
    s3_path = output_s3_path.rstrip("/")
    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    key = f"{prefix}/{job_id}/{os.path.basename(local_path)}".lstrip("/")
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as exc:
        logger.warning("S3 upload failed: %s", exc)
        return ""


def _run_job(job_id: str, schematic_local: str, wd_local: str,
             mode: str, verbose: bool, output_s3_path: str) -> None:
    """Background worker that runs the review and updates the job tracker."""
    tracker.start(job_id)
    job_dir = REPORT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        report_text = run_review(
            schematic_pdf=schematic_local,
            wiring_diagram_pdf=wd_local,
            mode=mode,
            verbose=verbose,
            job_id=job_id,
        )

        report_files = [
            "bom_comparison_report.html",
            "circuit_comparison_report.html",
            "wd_analysis_report.html",
        ]
        report_path = ""
        for rf in report_files:
            if os.path.exists(rf):
                dest = str(job_dir / rf)
                shutil.move(rf, dest)
                report_path = dest
                break

        if not report_path:
            report_path = str(job_dir / f"{mode}_report.txt")
            with open(report_path, "w") as f:
                f.write(report_text or "No report generated.")

        s3_uri = ""
        if output_s3_path:
            s3_uri = _upload_to_s3(report_path, output_s3_path, job_id)

        summary = "Review completed. See HTML report for details."
        if report_text and len(report_text) < 500:
            summary = report_text

        tracker.complete(job_id, report_path=report_path,
                         report_s3_uri=s3_uri, summary=summary)

    except Exception as exc:
        logger.exception("Job %s failed.", job_id)
        tracker.fail(job_id, str(exc))


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse()


@app.post("/review", response_model=JobSubmitResponse)
def submit_review(request: ReviewRequest) -> JobSubmitResponse:
    """Submit a review job. Returns immediately with a job_id for polling."""
    job_id = str(uuid.uuid4())[:8]

    schematic_local = _resolve_path(request.schematic_path)
    wd_local = _resolve_path(request.wiring_diagram_path)

    job = tracker.create(
        job_id=job_id,
        mode=request.mode.value,
        schematic_path=request.schematic_path,
        wiring_diagram_path=request.wiring_diagram_path,
        output_s3_path=request.output_s3_path,
    )

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, schematic_local, wd_local,
              request.mode.value, request.verbose, request.output_s3_path),
        daemon=True,
    )
    thread.start()

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        mode=request.mode.value,
        status_url=f"/jobs/{job_id}",
        message=f"Job submitted. Poll GET /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    """Get full job status including phase progress, tokens, and completion."""
    job = tracker.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.get("/jobs/{job_id}/log")
def get_job_log(job_id: str, last: int = 20) -> dict:
    """Get recent log entries for a job."""
    job = tracker.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"job_id": job_id, "log": job.log[-last:], "total": len(job.log)}


@app.get("/jobs/{job_id}/events")
def get_job_events(job_id: str, since_step: int = 0) -> dict:
    """Get structured events for building a UI timeline.

    Returns only key events (phase changes, tool calls, tool results,
    errors) — not raw reasoning text. Use ``since_step`` to get only
    new events since the last poll.
    """
    job = tracker.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    events = [
        e for e in job.log
        if e.get("step", 0) > since_step
        and e.get("type") in ("phase", "tool_call", "tool_result", "error", "self_correct")
    ]

    return {
        "job_id": job_id,
        "status": job.status.value,
        "progress_pct": job.progress_pct,
        "elapsed_seconds": job.elapsed_seconds,
        "events": events,
        "total_events": len(events),
        "tokens": {
            "input": job.input_tokens,
            "output": job.output_tokens,
            "total": job.input_tokens + job.output_tokens,
        },
    }


@app.get("/jobs")
def list_jobs() -> dict:
    """List all jobs with their current status."""
    jobs = []
    for jid, job in tracker._jobs.items():
        jobs.append({
            "job_id": jid,
            "mode": job.mode,
            "status": job.status.value,
            "progress_pct": job.progress_pct,
            "elapsed_seconds": job.elapsed_seconds,
        })
    return {"jobs": jobs}


@app.get("/reports/{job_id}/{filename}")
def get_report(job_id: str, filename: str) -> FileResponse:
    """Download a generated report."""
    path = REPORT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {job_id}/{filename}")
    media = "text/html" if filename.endswith(".html") else "text/plain"
    return FileResponse(str(path), media_type=media, filename=filename)


@app.get("/reports/{job_id}")
def list_job_reports(job_id: str) -> dict:
    """List all report files for a job."""
    job_dir = REPORT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    files = [f.name for f in job_dir.iterdir() if f.is_file()]
    return {"job_id": job_id, "files": files,
            "urls": [f"/reports/{job_id}/{f}" for f in files]}
