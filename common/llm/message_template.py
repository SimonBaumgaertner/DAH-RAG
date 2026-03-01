from __future__ import annotations

"""Template interface for building LLM message payloads.

Sub‑classes must implement :meth:`build_from_template` and return a list of
``{"role": str, "content": str}`` dictionaries that can be passed directly to
an LLM chat completion endpoint.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MessageTemplate(ABC):
    """Abstract base class for chat‑prompt templates.

    The goal is to decouple *how* a prompt is built from *what* an LLM does with
    it. Concrete templates plug in whatever text fragments they need and always
    emit the same canonical structure: ``[{"role": ..., "content": ...}, …]``.
    """

    @abstractmethod
    def build_from_template(self, **kwargs: Any) -> List[Dict[str, str]]:
        raise NotImplementedError

    def get_message(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""

        formatted_parts = []
        for msg in messages:
            role = msg.get('role', 'Unknown').title()
            content = msg.get('content', '')
            formatted_parts.append(f"{role}: '{content}'")

        return " ".join(formatted_parts)