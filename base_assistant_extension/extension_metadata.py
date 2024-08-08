from dataclasses import dataclass
from typing import TypedDict


@dataclass
class ExtensionMetadata(TypedDict):
    name: str
    description: str
