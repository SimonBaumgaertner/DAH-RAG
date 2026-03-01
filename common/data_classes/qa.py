from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Choice:
    label: str
    text: str


@dataclass(frozen=True)
class Proof:
    document_id: str
    context: str


@dataclass
class QuestionAnswerPair:
    question_id: str
    question: str
    choices: List[Choice]
    correct_answer: str
    proofs: List[Proof]

    @classmethod
    def from_dict(cls, data: dict) -> "QuestionAnswerPair":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            choices=[Choice(**c) for c in data.get("choices", [])],
            correct_answer=data["correct_answer"],
            proofs=[Proof(**p) for p in data.get("proofs", [])],
        )

    def get_correct_choice(self) -> Choice:
        return next(c for c in self.choices if c.label == self.correct_answer)
