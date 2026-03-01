# create a class that stores the ranking of chunks from Graph Search
# Per chunk there are 3 scores:
# 1. PPR Score (PageRank)
# 2. Dense Score
# 3. Lexical Score

from typing import Dict, List


class ChunkScore:
    def __init__(
        self, 
        chunk_id: str, 
        ppr_score: float, 
        dense_score: float, 
        lexical_score: float
    ):
        self.chunk_id = chunk_id
        self.ppr_score = ppr_score
        self.dense_score = dense_score
        self.lexical_score = lexical_score
        self.rank_ppr_weight: float = 0.0
        self.rank_dense_weight: float = 0.0
        self.rank_lexical_weight: float = 0.0
        self.score: float = 0.0
    
    def calculate_score(
        self, 
        rank_ppr_weight: float, 
        rank_dense_weight: float, 
        rank_lexical_weight: float
    ) -> float:
        self.rank_ppr_weight = rank_ppr_weight
        self.rank_dense_weight = rank_dense_weight
        self.rank_lexical_weight = rank_lexical_weight
        self.score = (
            rank_ppr_weight * self.ppr_score 
            + rank_dense_weight * self.dense_score 
            + rank_lexical_weight * self.lexical_score
        )
        return self.score

    def get_json(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "ppr_score": self.ppr_score,
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "score": self.score
        }


class ChunkRankings:
    def __init__(self, query: str):
        self.query = query
        self.chunk_rankings: List[ChunkScore] = []

    def add_chunk_ranking(
        self, 
        chunk_id: str, 
        ppr_score: float, 
        dense_score: float, 
        lexical_score: float
    ) -> None:
        self.chunk_rankings.append(
            ChunkScore(chunk_id, ppr_score, dense_score, lexical_score)
        )

    def get_chunk_rankings(self) -> List[ChunkScore]:
        return self.chunk_rankings
    
    def calculate_chunk_rankings(
        self, 
        rank_ppr_weight: float, 
        rank_dense_weight: float, 
        rank_lexical_weight: float
    ) -> List[ChunkScore]:
        for chunk in self.chunk_rankings:
            chunk.calculate_score(
                rank_ppr_weight, 
                rank_dense_weight, 
                rank_lexical_weight
            )
        self.chunk_rankings.sort(key=lambda x: x.score, reverse=True)
        return self.chunk_rankings

