from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import md5
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from hope.common.config import hope_config
from hope.utils.hash import md5_hash


@dataclass
class ChunkData:
    text: str
    chunking_method: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.text, Path):
            assert self.text.is_file(), f"Path {self.text} is not a file"
            self.text = self.text.read_text()

    @property
    def fingerprint(self) -> str:
        return md5(self.text.encode()).hexdigest()

    def __str__(self) -> str:
        return self.text


@dataclass
class Question:
    question: str
    answer: str | None = None


@dataclass
class DataScapeMetadata:
    title: str
    url: str
    date: str
    source: str
    file_name: str

    def dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        self.file_name = self.file_name.strip().replace(" ", "_")


@dataclass
class DocumentMeta:
    num_chunks: int
    num_words: int
    num_characters: int
    chunk_length_mean: float | None = None
    chunk_length_std: float | None = None
    chunks_md5_ids: list[str] | None = None

    @classmethod
    def from_chunks_and_document(
        cls, chunks: list[str], document: str
    ) -> "DocumentMeta":
        return cls(
            num_chunks=len(chunks),
            num_words=len(document.split()),
            num_characters=len(document),
            chunk_length_mean=np.mean([len(chunk) for chunk in chunks]),
            chunk_length_std=np.std([len(chunk) for chunk in chunks]),
            chunks_md5_ids=[md5_hash(chunk) for chunk in chunks],
        )


@dataclass
class HopeScore:
    conceptualism: float | None = None
    information_preservation: float | None = None
    semantic_independence: float | None = None
    hope: float | None = None
    document_meta: DocumentMeta | None = None

    def __post_init__(self):
        self.hope = np.average(
            [
                self.conceptualism or 0.0,
                self.information_preservation or 0.0,
                self.semantic_independence or 0.0,
            ],
            weights=[
                (
                    hope_config.scoring.hope.conceptualism_weight
                    if self.conceptualism
                    else 0.000000001
                ),
                (
                    hope_config.scoring.hope.information_preservation_weight
                    if self.information_preservation
                    else 0.000000001
                ),
                (
                    hope_config.scoring.hope.semantic_independence_weight
                    if self.semantic_independence
                    else 0.000000001
                ),
            ],
        )

    def __str__(self) -> str:
        return f"{self.hope} (conceptualism={self.conceptualism}, information_preservation={self.information_preservation}, semantic_independence={self.semantic_independence})"

    __float__ = lambda self: float(self.hope)

    def __add__(self, other):
        if not isinstance(other, HopeScore):
            return NotImplemented
        return HopeScore(
            conceptualism=(self.conceptualism or 0.0) + (other.conceptualism or 0.0),
            information_preservation=(self.information_preservation or 0.0)
            + (other.information_preservation or 0.0),
            semantic_independence=(self.semantic_independence or 0.0)
            + (other.semantic_independence or 0.0),
        )

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return HopeScore(
            conceptualism=(self.conceptualism or 0.0) * other,
            information_preservation=(self.information_preservation or 0.0) * other,
            semantic_independence=(self.semantic_independence or 0.0) * other,
        )

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.__mul__(1 / other)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])
