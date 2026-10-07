from contextlib import asynccontextmanager

import agent.tools  # noqa: F401  (register tool surface)
from agent.config import settings
from api.jobs import JOBS
from api.routers import router
from api.store import REGISTRY
from fastapi import FastAPI


APP_VERSION = "0.3.0"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version=APP_VERSION)
app.include_router(router)


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "datasets": len(REGISTRY),
        "jobs_running": sum(1 for job in JOBS.values() if job.get("status") == "running"),
        "jobs_total": len(JOBS),
    }


@app.get("/")
async def root():
    return {"app": settings.app_name, "version": APP_VERSION, "docs": "/docs"}
