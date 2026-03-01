from enum import Enum, auto

class RegisteredDataset(Enum):
    NovelQA = "NovelQA"
    HotpotQA_10 = "HotpotQA_10"
    HotpotQA_100 = "HotpotQA_100"
    HotpotQA_1k = "HotpotQA_1k"
    HotpotQA_10k = "HotpotQA_10k"
    HotpotQA_Dev = "HotpotQA_Dev"
    PubMedQA = "PubMedQA"
    PubMedQA_5k = "PubMedQA_5k"
    PubMedQA_10k = "PubMedQA_10k"
    MultiHopRAG = "MultiHopRAG"
    alexendriawiki = "alexendriawiki"
    Scaling = "Scaling"

class RAG(Enum):
    NaiveVectorDB = "NaiveVectorDB"
    LightRAG = "LightRAG"
    HippoRAG = "HippoRAG"
    RaptorRAG = "RaptorRAG"
    DocAwareHybridRAG = "DocAwareHybridRAG"
    FastGraphRAG = "FastGraphRAG"
    NoRAGGeneration = "NoRAGGeneration"
    BM25 = "BM25"

class GenerationConfiguration(Enum):
    NoGen = "NoGen"
    MCQA = "MCQA" # Multiple Choice Question Answering
    OpenAnswer = "OpenAnswer" # Open-ended question answering

class ChunkingStrategy(Enum):
    ContextualizedChunker = "ContextualizedChunker"
    FixedSizeChunker = "FixedSizeChunker"
    SemanticChunker = "SemanticChunker"

class Reranker(Enum):
    NoRerank = "NoRerank"
    Jina_Reranker_v3_600M = "Jina_Reranker_v3_600M"
    
class Encoder(Enum):
    Qwen3_600M = "Qwen3_600M"
    Qwen3_4B = "Qwen3_4B"
    Qwen3_4B_OpenRouter = "Qwen3_4B_OpenRouter"
    NV_Embed_v2 = "NV_Embed_v2"
    Jina_v3_600M = "Jina_v3_600M"
    KaLM_EMBEDDING_Gemma3_12B = "KaLM_EMBEDDING_Gemma3_12B"
    Qwen3_8B = "Qwen3_8B"



class LLMBackend(Enum):
    Dummy = "dummy"
    LocalInstruct = "local-instruct"
    LocalThinking = "local-thinking"
    GPTOss = "gpt-oss"
    OpenAI = "openai"
    OpenRouter = "openrouter"

class LLMName(Enum):
    Mistral_small = "mistralai/mistral-small-3.2-24b-instruct" # small-ish model
    Llama_3_3_70B = "meta-llama/llama-3.3-70b-instruct" # medium model
    Qwen3_235B = "qwen/qwen3-235b-a22b-2507" # larger model
    Deepseek_V3 = "deepseek/deepseek-v3.2" # larger reasoning model