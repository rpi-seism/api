from typing import Annotated
from pathlib import Path

from fastapi import Depends
from sqlmodel import Session, create_engine


DB_PATH    = Path(__file__).parent.parent / "data" / "database.db"
engine     = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
