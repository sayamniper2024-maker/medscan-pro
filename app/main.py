"""
main.py

The FastAPI app - the "front door" that lets a website send a request
and get a MedScan Pro report back.

THIS IS THE BACKGROUND-JOB VERSION: starting a report returns immediately
with a job ID. The actual research runs in a background thread. The
client polls a separate endpoint to check progress and retrieve the
finished report once it's ready.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from research import generate_full_report
from jobs import start_job, get_job


app = FastAPI(title="MedScan Pro API")

# CORS: allows the frontend HTML page (served from the browser) to make
# requests to this API. Without this, browsers block the request as a
# security measure since the page and API could be seen as "different sites."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local development; tighten this before going public
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    device_concept: str
    indication: str


class StartJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    status: str          # "pending" | "running" | "done" | "failed"
    report: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
def root():
    """Serves the frontend HTML page."""
    return FileResponse("static/index.html")


@app.get("/api/health")
def health_check():
    """A simple health-check route, separate from the frontend page."""
    return {"status": "MedScan Pro API is running"}


@app.post("/generate-report", response_model=StartJobResponse)
def generate_report(request: ReportRequest):
    """
    Starts a report generation job in the background and returns
    immediately with a job_id. Does NOT wait for the report to finish.
    """
    job_id = start_job(
        generate_full_report,
        request.device_concept,
        request.indication,
    )
    return StartJobResponse(job_id=job_id, status="pending")


@app.get("/report-status/{job_id}", response_model=JobStatusResponse)
def report_status(job_id: str):
    """
    Checks on a job's progress. The client (website) calls this
    repeatedly ("polling") until status is "done" or "failed".
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        status=job["status"],
        report=job["report"],
        error=job["error"],
    )
