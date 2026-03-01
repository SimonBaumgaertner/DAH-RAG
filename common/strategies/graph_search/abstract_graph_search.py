from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk
from common.neo4j.data_classes import Executor
from common.neo4j.standard_executor import StandardExecutor


class GraphSearch(ABC):
    """Abstract base class for search strategies over the graph."""

    @abstractmethod
    def search(self, query: str, k: int, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        """Search the graph and return the top ``k`` chunks."""
        raise NotImplementedError



