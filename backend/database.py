from typing import Optional, List
from sqlmodel import Field, SQLModel, create_engine, Relationship
from datetime import datetime
import os

# Database Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "db", "missing_persons.db")
sqlite_url = f"sqlite:///{DB_PATH}"

engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

class MissingPerson(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    description: Optional[str] = None
    image_path: str
    embedding: str # Store as JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    sightings: List["Sighting"] = Relationship(back_populates="person")

class Sighting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    person_id: Optional[int] = Field(default=None, foreign_key="missingperson.id")
    location: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_path: str # Path to the annotated frame/image
    
    person: Optional[MissingPerson] = Relationship(back_populates="sightings")

def create_db_and_tables():
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    SQLModel.metadata.create_all(engine)
