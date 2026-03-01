# create a class that stores the ranking of documents from Graph Search
# Per document there are 3 scores:
# 1. Lexical Score
# 2. Dense Score
# 3. Entity Score

from typing import Dict, List

class DocumentScore:
    def __init__(
        self, 
        document_id: str, 
        lexical_score: float, 
        dense_score: float, 
        entity_score: int
    ):
        self.document_id = document_id
        self.lexical_score = lexical_score
        self.dense_score = dense_score
        self.entity_score = entity_score
        self.filter_lexical_weight: float = 0.0
        self.filter_dense_weight: float = 0.0
        self.filter_entity_weight: float = 0.0
        self.score: float = 0.0
    
    def calculate_score(
        self, 
        filter_lexical_weight: float, 
        filter_dense_weight: float, 
        filter_entity_weight: float
    ) -> float:
        self.filter_lexical_weight = filter_lexical_weight
        self.filter_dense_weight = filter_dense_weight
        self.filter_entity_weight = filter_entity_weight
        self.score = (
            filter_lexical_weight * self.lexical_score 
            + filter_dense_weight * self.dense_score 
            + filter_entity_weight * self.entity_score
        )
        return self.score

    def get_json(self) -> Dict:
        return {
            "document_id": self.document_id,
            "filter_lexical_score": self.lexical_score,
            "filter_dense_score": self.dense_score,
            "filter_entity_score": self.entity_score,
            "score": self.score
        }


class DocumentRankings:
    def __init__(self, query: str):
        self.query = query
        self.document_rankings: List[DocumentScore] = []

    def add_document_ranking(
        self, 
        document_id: str, 
        lexical_score: float, 
        dense_score: float, 
        entity_score: int
    ) -> None:
        self.document_rankings.append(
            DocumentScore(document_id, lexical_score, dense_score, entity_score)
        )

    def get_document_rankings(self) -> List[DocumentScore]:
        return self.document_rankings
    
    def calculate_document_rankings(
        self, 
        filter_lexical_weight: float, 
        filter_dense_weight: float, 
        filter_entity_weight: float
    ) -> List[DocumentScore]:
        for document in self.document_rankings:
            document.calculate_score(
                filter_lexical_weight, 
                filter_dense_weight, 
                filter_entity_weight
            )
        self.document_rankings.sort(key=lambda x: x.score, reverse=True)
        return self.document_rankings