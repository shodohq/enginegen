from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional


@dataclass
class Diagnostic:
    code: str
    message: str
    severity: str = "ERROR"
    location: Optional[str] = None
    hints: List[str] = field(default_factory=list)
    data: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "location": self.location,
            "hints": self.hints,
            "data": self.data,
        }


class EngineGenError(Exception):
    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class Diagnostics:
    def __init__(self) -> None:
        self.items: List[Diagnostic] = []

    def add(self, diagnostic: Diagnostic) -> None:
        self.items.append(diagnostic)

    def extend(self, diagnostics: Iterable[Diagnostic] | "Diagnostics") -> None:
        if isinstance(diagnostics, Diagnostics):
            self.items.extend(diagnostics.items)
        else:
            self.items.extend(diagnostics)

    def has_errors(self) -> bool:
        return any(d.severity == "ERROR" for d in self.items)

    def raise_for_errors(self) -> None:
        if self.has_errors():
            # Raise the first error for now; callers can access the rest via diagnostics
            first = next(d for d in self.items if d.severity == "ERROR")
            raise EngineGenError(first)

    def to_list(self) -> List[dict]:
        return [d.to_dict() for d in self.items]
