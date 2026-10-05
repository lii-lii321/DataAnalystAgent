from contextlib import asynccontextmanager

import agent.tools  # noqa: F401  (register tool surface)
from agent.config import settings
from api.routers import router
from fastapi import FastAPI


APP_VERSION = "0.3.0"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version=APP_VERSION)
app.include_router(router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"app": settings.app_name, "version": APP_VERSION, "docs": "/docs"}
