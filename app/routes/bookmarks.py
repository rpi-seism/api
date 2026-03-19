from fastapi import APIRouter, HTTPException
from sqlmodel import select
from datetime import datetime, timezone
import uuid

from app.db import SessionDep
from app.entities.bookmark import Bookmark, BookmarkCreate, BookmarkPublic, BookmarkUpdate

router = APIRouter()


def to_public(bm: Bookmark) -> BookmarkPublic:
    try:
        print(bm)
        return BookmarkPublic(
            **bm.model_dump(exclude={"channels"}),
            channels=bm.channels.split(",") if bm.channels.find(",") != -1 else [bm.channels],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting bookmark: {e}") from e


@router.get("/", response_model=list[BookmarkPublic])
def list_bookmarks(session: SessionDep):
    bookmarks = session.exec(select(Bookmark).order_by(Bookmark.saved_at.desc())).all()
    return [to_public(bm) for bm in bookmarks]

@router.post("/", response_model=BookmarkPublic)
def create_bookmark(data: BookmarkCreate, session: SessionDep):
    print(data)
    bm = Bookmark(
        id       = str(uuid.uuid4()),
        label    = data.label.strip() or f"{data.start}",
        channels = ",".join(data.channels),
        start    = data.start,
        end      = data.end,
        units    = data.units,
        saved_at = datetime.now(timezone.utc),
    )
    session.add(bm)
    session.commit()
    session.refresh(bm)
    return to_public(bm)

@router.patch("/{bookmark_id}", response_model=BookmarkPublic)
def update_bookmark(bookmark_id: str, data: BookmarkUpdate, session: SessionDep):
    bm = session.get(Bookmark, bookmark_id)
    if not bm:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    update = data.model_dump(exclude_unset=True)
    if "channels" in update:
        update["channels"] = ",".join(update["channels"])
    bm.sqlmodel_update(update)
    session.add(bm)
    session.commit()
    session.refresh(bm)
    return to_public(bm)

@router.delete("/{bookmark_id}")
def delete_bookmark(bookmark_id: str, session: SessionDep):
    bm = session.get(Bookmark, bookmark_id)
    if not bm:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    session.delete(bm)
    session.commit()
    return {"deleted": bookmark_id}
