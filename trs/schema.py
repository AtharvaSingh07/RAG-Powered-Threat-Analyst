from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uuid



# ---------------------------------------------
# Raw Document Object
# ---------------------------------------------
class Document(BaseModel):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document ID"
    )
    text: str
    source: str
    metadata: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Optional metadata dict"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Time the document was ingested"
    )


# ---------------------------------------------
# Summary Object
# ---------------------------------------------
class Summary(BaseModel):
    summary: str
    source: str
    metadata: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Optional metadata dict"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Time the summary was generated"
    )


# ---------------------------------------------
# IOC Extraction Output
# ---------------------------------------------
class Indicators(BaseModel):
    ips: List[str] = Field(
        default_factory=list,
        description="List of extracted IP addresses"
    )
    urls: List[str] = Field(
        default_factory=list,
        description="List of extracted URLs"
    )
    hashes: List[str] = Field(
        default_factory=list,
        description="List of extracted file hashes (MD5/SHA)"
    )


# ---------------------------------------------
# Chat / QnA Structures
# ---------------------------------------------
class Message(BaseModel):
    role: str  # "user" / "assistant" / "system"
    content: str


class Conversation(BaseModel):
    history: List[Message] = Field(
        default_factory=list,
        description="Prior conversation turns"
    )
