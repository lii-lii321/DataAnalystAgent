from contextlib import asynccontextmanager

import agent.tools  # noqa: F401  (register tool surface)
from agent.config import settings
from api.routers import router
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version="0.3.0")
app.include_router(router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"app": settings.app_name, "version": "0.1.0", "docs": "/docs"}
