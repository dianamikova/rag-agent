"""
logger.py — Structured logging for each escalation attempt.

Records per iteration:
- Level and strategy used
- Wall clock time (ms)
- Tokens used (input + output)
- Confidence score before and after
- Whether this level produced the final answer
"""

import time
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class IterationLog:
    level: int
    strategy: str
    query_used: str
    chunks_retrieved: int
    confidence_before: float
    confidence_after: float
    tokens_used: int
    time_ms: float
    resolved: bool = False  # True if this level produced the final answer

@dataclass
class QueryLog:
    original_query: str
    final_answer: str = ""
    resolved_at_level: int = -1
    total_time_ms: float = 0.0
    total_tokens: int = 0
    iterations: list[IterationLog] = field(default_factory=list)

    def add_iteration(self, log: IterationLog):
        self.iterations.append(log)
        self.total_tokens += log.tokens_used
        self.total_time_ms += log.time_ms
        if log.resolved:
            self.resolved_at_level = log.level

    def summary_table(self) -> list[dict]:
        """Returns a list of dicts suitable for display as a table."""
        return [
            {
                "Level": i.level,
                "Strategy": i.strategy,
                "Query used": i.query_used[:60] + ("..." if len(i.query_used) > 60 else ""),
                "Chunks": i.chunks_retrieved,
                "Confidence": f"{i.confidence_after:.2f}",
                "Tokens": i.tokens_used,
                "Time (ms)": f"{i.time_ms:.0f}",
                "Resolved": "✓" if i.resolved else "",
            }
            for i in self.iterations
        ]

    def to_dict(self) -> dict:
        return asdict(self)


class Timer:
    """Simple context manager for timing blocks."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

def save_log(log: QueryLog, log_dir: str = "logs"):
    """Append query log to a JSONL file."""
    try:
        Path(log_dir).mkdir(exist_ok=True)
        path = Path(log_dir) / "queries.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")
    except Exception:
        pass  # no influence of logging failure for the agent