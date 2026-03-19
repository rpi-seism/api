from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class BookmarkBase(SQLModel):
    label:    str
    channels: str          # comma-separated, e.g. "EHZ,EHN"
    start:    datetime
    end:      datetime
    units:    str


class Bookmark(BookmarkBase, table=True):
    id:       str = Field(default=None, primary_key=True)
    saved_at: datetime


class BookmarkPublic(BookmarkBase):
    id:        str
    saved_at:  datetime
    channels:  list[str]   # deserialized for the client

    model_config = {"from_attributes": True}

class BookmarkCreate(BookmarkBase):
    channels: list[str]    # client sends a list, we serialize to str


class BookmarkUpdate(SQLModel):
    label:    Optional[str] = None
    channels: Optional[list[str]] = None
    start:    Optional[datetime] = None
    end:      Optional[datetime] = None
    units:    Optional[str] = None
