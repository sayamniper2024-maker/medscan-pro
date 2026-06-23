"""
jobs.py

A simple in-memory job tracker for background report generation.

WHY THIS EXISTS: generating a MedScan Pro report takes several minutes
(multiple searches, rate-limit pauses, multiple model calls). A web
request that just waits that whole time is bad practice - it can time
out, and it ties up server resources the entire time.

Instead: starting a report creates a "job" with a unique ID and returns
immediately. The actual work happens in a background thread. The client
(the website) then periodically checks back ("polls") using the job ID
to see if it's done yet.

HONEST LIMITATION: this job store lives in server memory (a plain Python
dict). It resets if the server restarts, and won't work correctly if you
ever run multiple server processes/copies at once. A real production
system would use a database or a tool like Redis instead. This is the
right level of complexity for a first working version, not the final
answer.
"""

import uuid
import threading
import traceback
from datetime import datetime, timezone


# job_id -> job info dict
_jobs = {}
_jobs_lock = threading.Lock()


def create_job():
    """Creates a new job entry in the 'pending' state and returns its ID."""
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",   # pending -> running -> done | failed
            "report": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    return job_id


def get_job(job_id):
    """Returns the job dict, or None if the job_id doesn't exist."""
    with _jobs_lock:
        return _jobs.get(job_id)


def _run_job(job_id, work_function, *args, **kwargs):
    """
    Internal: actually runs the slow work in a background thread,
    updating the job's status as it goes.
    """
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    try:
        result = work_function(*args, **kwargs)
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["report"] = result
    except Exception as e:
        # Catch anything that goes wrong so a crash in the background
        # thread doesn't just disappear silently - it shows up in the
        # job's status instead.
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"{e}\n{traceback.format_exc()}"


def start_job(work_function, *args, **kwargs):
    """
    Creates a job and starts the work in a background thread immediately.
    Returns the job_id right away - does NOT wait for the work to finish.
    """
    job_id = create_job()
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, work_function, *args),
        kwargs=kwargs,
        daemon=True,
    )
    thread.start()
    return job_id
