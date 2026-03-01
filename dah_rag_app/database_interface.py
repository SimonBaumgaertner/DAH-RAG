from abc import ABC, abstractmethod
from typing import List

from common.data_classes.documents import Document
from common.data_classes.rag_system import Chunk


class RAGDatabase(ABC):
    @abstractmethod
    def initialize_database(self, wipe_at_start: bool) -> bool:
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int) -> List[Chunk]:
        pass

    @abstractmethod
    def add_document(self, document: Document) -> bool:
        pass

    @abstractmethod
    def remove_document(self, document_id: str) -> bool:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        pass

    @abstractmethod
    def get_document_by_id(self, document_id: str) -> Document:
        pass

    @abstractmethod
    def get_all_documents_count(self) -> int:
        pass



