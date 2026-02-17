import logging
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RagDebugConfig:
    enabled: bool = False
    sample_chunks: int = 3

    @classmethod
    def from_env(cls) -> "RagDebugConfig":
        return cls(
            enabled=_env_bool("RAG_DEBUG", False),
            sample_chunks=int(os.getenv("RAG_DEBUG_SAMPLE_CHUNKS", "3")),
        )


class RagDebugger:
    def __init__(self, logger: logging.Logger | None = None, config: RagDebugConfig | None = None):
        self.logger = logger or logging.getLogger("rag.debug")
        self.config = config or RagDebugConfig.from_env()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def log(self, message: str, *args: Any) -> None:
        if self.enabled:
            self.logger.info("[RAG DEBUG] " + message, *args)

    def log_chunks(self, chunks: Iterable[Any], source: str) -> None:
        if not self.enabled:
            return
        chunks = list(chunks)
        lengths = [len(getattr(chunk, "page_content", "") or "") for chunk in chunks]
        self.log("Chunking for %s: chunks=%s", source, len(chunks))
        if lengths:
            self.log(
                "Chunk stats for %s: min=%s max=%s avg=%.1f",
                source,
                min(lengths),
                max(lengths),
                mean(lengths),
            )
        for idx, chunk in enumerate(chunks[: self.config.sample_chunks]):
            content = (getattr(chunk, "page_content", "") or "").replace("\n", " ")
            metadata = getattr(chunk, "metadata", {}) or {}
            self.log("Chunk sample #%s metadata=%s text=%s", idx, metadata, content[:220])

    def log_retrieval(self, prompt: str, k: int, filters: dict | None, results: list[dict]) -> None:
        if not self.enabled:
            return
        self.log("Retrieval params: prompt=%s k=%s filters=%s", prompt[:200], k, filters)
        self.log("Retrieval results count=%s", len(results))
        for idx, item in enumerate(results[: self.config.sample_chunks]):
            snippet = (item.get("content") or "").replace("\n", " ")
            self.log(
                "Top result #%s score=%s metadata=%s text=%s",
                idx,
                item.get("score"),
                item.get("metadata"),
                snippet[:220],
            )

    def log_indexing(self, source: str, chunks_count: int, records_count: int | None = None) -> None:
        if not self.enabled:
            return
        self.log("Indexing source=%s chunks=%s", source, chunks_count)
        if records_count is not None:
            self.log("Vector store total records=%s", records_count)
