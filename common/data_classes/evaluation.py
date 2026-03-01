from enum import Enum


class LLMCallContext(Enum):
    """Context for LLM calls to determine appropriate tracking."""
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


class EntryType(Enum):
    # base indexing retrieval
    FULL_INDEXING_TRACK = "full_indexing_track"
    DOCUMENT_INDEXING_TRACK = "document_indexing_track"
    CHUNK_COUNT_TRACK = "chunk_count_track"
    FULL_RETRIEVAL_TRACK = "full_retrieval_track"
    CHUNK_RETRIEVAL_TRACK = "chunk_retrieval"
    # QA answering
    ANSWER_TRACK = "answer_track"
    CORRECTNESS_TRACK = "correctness_track"
    ROUGE_L_TRACK = "rouge_l_track"
    PROOF_TRACK = "proof_track"
    TOTAL_PROOFS_TRACK = "total_proofs_track"
    # LLM Stuff - Indexing
    LLM_INDEXING_GENERATION_TIME_TRACK = "llm_indexing_generation_time_track"
    LLM_INDEXING_INPUT_TOKENS_TRACK = "llm_indexing_input_tokens_track"
    LLM_INDEXING_OUTPUT_TOKENS_TRACK = "llm_indexing_output_tokens_track"
    # LLM Stuff - Retrieval
    LLM_RETRIEVAL_INPUT_TOKENS_TRACK = "llm_retrieval_input_tokens_track"
    LLM_RETRIEVAL_OUTPUT_TOKENS_TRACK = "llm_retrieval_output_tokens_track"
    LLM_RETRIEVAL_GENERATION_TIME_TRACK = "llm_retrieval_generation_time_track"
    # LLM Stuff - Generation (QA Answer Generation)
    LLM_QA_GENERATION_TIME_TRACK = "llm_qa_generation_time_track"
    LLM_QA_INPUT_TOKENS_TRACK = "llm_qa_input_tokens_track"
    LLM_QA_OUTPUT_TOKENS_TRACK = "llm_qa_output_tokens_track"
    # tracks scaling experiment
    SCALING_INDEXING_TRACK = "scaling_indexing_track"
    SCALING_RETRIEVAL_TRACK = "scaling_retrieval_track"
    # Cost Tracking
    LLM_INDEXING_COST_TRACK = "llm_indexing_cost_track"
    LLM_QA_COST_TRACK = "llm_qa_cost_track"



class CorrectnessType(Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    FORMATTING_ERROR = "formatting_error"
