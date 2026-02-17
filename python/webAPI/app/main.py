from fastapi import FastAPI, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging
import time

from app.routes import users, files, llm, program_settings
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Аутентификация (в работе)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
oauth_deps = Annotated[str, Depends(oauth2_scheme)]

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOW_METHODS,
    allow_headers=settings.ALLOW_HEADERS,
)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        logging.info("Request: %s %s", request.method, request.url.path)
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logging.info("Response: %s %s status=%s duration_ms=%.1f", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


app.add_middleware(LoggingMiddleware)

api_router = APIRouter(prefix=settings.API_V1_STR)
api_router.include_router(users.router)
api_router.include_router(llm.router)
api_router.include_router(files.router)
api_router.include_router(program_settings.router)
app.include_router(api_router)


@app.get("/", tags=["root"])
async def read_root():
    return {"message": "Ping!"}
