from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logger import configure_logger
from app.routes import archive, bookmarks


data_base_folder = Path(__file__).parent.parent / "data"
configure_logger(data_base_folder)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


app.include_router(archive.router, prefix="/archive", tags=["archive"])
app.include_router(bookmarks.router, prefix="/bookmarks", tags=["bookmarks"])
