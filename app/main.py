from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import archive


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["GET"],
    allow_headers=["*"],
)


app.include_router(archive.router, prefix="/archive")
