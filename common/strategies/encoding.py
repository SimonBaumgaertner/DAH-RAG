# encoding_strategy.py
from __future__ import annotations


from abc import ABC, abstractmethod
import os
from typing import List, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from common.logging.run_logger import RunLogger

class EncodingStrategy(ABC):
    @abstractmethod
    def encode(self, text: str, query: bool = False) -> np.ndarray:
        ...

    @abstractmethod
    def get_embedding_dims(self) -> int:
        ...


class QwenEncoder(EncodingStrategy):
    """
    Qwen3-Embedding encoder for text relevance / RAG.

    Key behaviors:
      - Uses the model's instruction format for queries:
          "Instruct: <task>\\nQuery:<text>"
      - Last-token pooling (as recommended by the model card)
      - Left padding to make last-token pooling unambiguous
      - L2-normalized output (cosine-ready)
      - Optional chunking for very long inputs

    Args:
        model_name: HF ID, default "Qwen/Qwen3-Embedding-4B"
        task: "query" (instruction-formatted) or "document" (raw text)
        instruction: custom one-sentence instruction for queries;
                     if None, uses the model card’s retrieval default.
        device: "cuda", "cpu", etc. Autodetected if None.
        max_length: tokenizer max_length used per forward pass. Model supports long context;
                    8192 is a good practical default.
        long_strategy: currently "mean" — chunk > max_length and mean the chunk vectors.
        normalize: L2-normalize the embedding (recommended for cosine similarity).
        embedding_dims: If None, inferred from model.config.hidden_size (2560 for 4B).
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        embedding_dims: Optional[int] = None,
        task: Literal["query", "document"] = "document",
        instruction: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 8192,
        long_strategy: Literal["mean"] = "mean",
        normalize: bool = True,
        log: Optional["RunLogger"] = None,
    ):
        import os
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.task = task
        if self.task not in {"query", "document"}:
            raise ValueError("task must be 'query' or 'document'")

        # Default instruction recommended by Qwen for retrieval queries
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
            if self.task == "query"
            else None
        )

        # Choose device
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Help CUDA allocator avoid fragmentation on tight VRAM
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Tokenizer — left padding recommended for last-token pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            use_fast=True,
        )

        # Optional logging of device info
        if log:
            self.log = log
            device_info = self._get_device_info()
            log.info(f"QwenEncoder initializing on device: {device_info}")

        # Load model directly on the target device in half precision, streaming shards.
        # Try flash attention 2, fall back to default if not available
        model_kwargs = {
            "torch_dtype": torch.bfloat16,          # or torch.float16 if you prefer
            "low_cpu_mem_usage": True,
        }
        
        # Try to use flash attention 2 if available
        try:
            import flash_attn  # noqa: F401, F811
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if log:
                log.info("Using Flash Attention 2 for Encoder")
        except ImportError:
            if log:
                log.info("Flash Attention not available, using default attention for QwenEncoder")
        if str(self.device).startswith("cuda"):
            model_kwargs["device_map"] = {"": self.device}  # place modules directly on GPU

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # IMPORTANT: do NOT call self.model.to(self.device); we already placed it with device_map.

        # Sequence handling
        self.max_length = max_length
        if long_strategy != "mean":
            raise ValueError("Only long_strategy='mean' is implemented.")
        self.long_strategy = long_strategy

        # Normalization toggle
        self.normalize = normalize

        # Embedding dimension: infer from the model unless explicitly provided
        inferred_dim = getattr(self.model.config, "hidden_size", None)
        self.embedding_dims = embedding_dims or inferred_dim or 1024  # fallback to 1024 if unknown

    @torch.inference_mode()
    def encode(self, text: str, query: bool = False) -> np.ndarray:
        # Determine task mode based on query parameter
        current_task = "query" if query else "document"

        # Temporarily override task for this encoding operation
        original_task = self.task
        self.task = current_task

        try:
            formatted = self._format_input(text)

            # If it fits, do one pass
            token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
            if len(token_ids) <= self.max_length:
                vec = self._forward(formatted)
                return vec

            # Otherwise chunk and mean-pool the chunk vectors
            chunks = self._chunk_by_tokens(formatted, self.max_length)
            vecs = [self._forward(c) for c in chunks]
            return np.mean(vecs, axis=0)
        except Exception as e:
            if self.log:
                self.log.error(f"Error encoding text: {e}")
            raise e
        finally:
            # Restore original task
            self.task = original_task

    def get_embedding_dims(self) -> int:
        return int(self.embedding_dims)

    def _get_device_info(self) -> str:
        """Get detailed device information for logging."""
        import torch
        if self.device == "cpu":
            import platform
            return f"CPU ({platform.processor() or 'Unknown'})"
        elif str(self.device).startswith("cuda"):
            try:
                device_id = int(str(self.device).split(":")[1]) if ":" in str(self.device) else 0
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
                return f"CUDA:{device_id} ({gpu_name}, {gpu_memory:.1f}GB)"
            except Exception:
                return "CUDA (device info unavailable)"
        else:
            return str(self.device)


    # --- internals ---

    def _format_input(self, text: str) -> str:
        if self.task == "query":
            return f"Instruct: {self.instruction}\nQuery: {text}"
        return text

    def _chunk_by_tokens(self, text: str, window: int) -> list[str]:
        # Tokenize once, rebuild chunk strings with the tokenizer to avoid splits mid-token
        tokens = self.tokenizer.tokenize(text)
        out = []
        for i in range(0, len(tokens), window):
            toks = tokens[i : i + window]
            out.append(self.tokenizer.convert_tokens_to_string(toks))
        return out

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Works for either left or right padding; with left padding we can just take [:, -1]
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch, device=last_hidden_states.device), seq_lens
            ]

    def _forward(self, text: str) -> np.ndarray:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        pooled = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled[0].detach().cpu().float().numpy()


class MiniLMMeanPoolingEncoder(EncodingStrategy):
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        window_size: int = 512,
        device: str | None = None,
        long_strategy: str = "mean",
        log: Optional["RunLogger"] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Log device information
        if log:
            device_info = self._get_device_info()
            log.info(f"MiniLMMeanPoolingEncoder initializing on device: {device_info}")
        self.window_size = window_size
        if long_strategy not in {"mean"}:
            raise ValueError("Only long_strategy='mean' is implemented for now.")
        self.long_strategy = long_strategy

    @torch.inference_mode()
    def encode(self, text: str, query: bool = False) -> np.ndarray:
        return self._encode_one(text)

    def get_embedding_dims(self) -> int:
        return 384

    def _get_device_info(self) -> str:
        """Get detailed device information for logging."""
        if self.device == "cpu":
            import platform
            return f"CPU ({platform.processor() or 'Unknown'})"
        elif self.device.startswith("cuda"):
            try:
                device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
                return f"CUDA:{device_id} ({gpu_name}, {gpu_memory:.1f}GB)"
            except Exception:
                return f"CUDA (device info unavailable)"
        else:
            return self.device

    def _encode_one(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.window_size:
            return self._forward(text)

        windows = [
            self.tokenizer.convert_tokens_to_string(tokens[i: i + self.window_size])
            for i in range(0, len(tokens), self.window_size)
        ]
        vecs = [self._forward(chunk_text) for chunk_text in windows]
        return np.mean(vecs, axis=0)

    def _forward(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.window_size,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        out = self.model(**encoded).last_hidden_state  # [1, T, H]
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # [1, T, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled.cpu().numpy()[0]


class NVEmbedV2Encoder(EncodingStrategy):
    """
    NVIDIA NV-Embed-v2 encoder for text embeddings.

    Args:
        model_name: HuggingFace model ID, default "nvidia/NV-Embed-v2"
        embedding_dims: Dimension of the output embeddings, default 4096
        device: "cuda", "cpu", etc. Autodetected if None.
        max_length: Maximum sequence length for tokenization (supports up to 32768)
        normalize: Whether to L2-normalize the embeddings
    """
    def __init__(
        self,
        *,
        model_name: str = "nvidia/NV-Embed-v2",
        embedding_dims: int = 4096,
        device: Optional[str] = None,
        max_length: int = 32768,
        normalize: bool = True,
        log: Optional["RunLogger"] = None,
        task_instruction: str = "Given a question, retrieve passages that answer the question",
    ):
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = normalize
        self.task_instruction = task_instruction

        if log:
            device_info = self._get_device_info()
            log.info(f"{model_name} initializing on device: {device_info}")

        # Load tokenizer and model **with remote code**
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Optional but matches card recommendations for ST path:
        # self.tokenizer.padding_side = "right"

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: Union[str, List[str]], query: bool = False) -> np.ndarray:
        # Ensure list input for batch encode
        texts = [text] if isinstance(text, str) else text

        instruction = f"Instruct: {self.task_instruction}\nQuery: " if query else ""
        # Use model’s custom encoder (does latent-attention pooling internally)
        embeddings = self.model.encode(
            texts,
            instruction=instruction,
            max_length=self.max_length
        )

        if self.normalize:
            embeddings = F.normalize(torch.as_tensor(embeddings), p=2, dim=1).cpu().float().numpy()
        else:
            embeddings = torch.as_tensor(embeddings).cpu().float().numpy()

        return embeddings[0] if isinstance(text, str) else embeddings

    def get_embedding_dims(self) -> int:
        # Optionally read from config if available
        return getattr(getattr(self.model, "config", object()), "embedding_dim", self.embedding_dims)

    def _get_device_info(self) -> str:
        if self.device == "cpu":
            import platform
            return f"CPU ({platform.processor() or 'Unknown'})"
        elif str(self.device).startswith("cuda"):
            try:
                device_id = int(str(self.device).split(":")[1]) if ":" in str(self.device) else 0
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
                return f"CUDA:{device_id} ({gpu_name}, {gpu_memory:.1f}GB)"
            except Exception:
                return "CUDA (device info unavailable)"
        else:
            return str(self.device)


class JinaEncoder(EncodingStrategy):
    """
    Jina-Embeddings-v3 encoder.
    Supports task-specific LoRA adapters (retrieval.query, retrieval.passage, etc.)
    Default dimension: 1024.
    """
    def __init__(
        self,
        *,
        model_name: str = "jinaai/jina-embeddings-v3",
        embedding_dims: int = 1024,
        device: Optional[str] = None,
        max_length: int = 8192,
        normalize: bool = True,
        log: Optional["RunLogger"] = None,
    ):
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = normalize

        if log:
            self.log = log
            log.info(f"JinaEncoder initializing on device: {self.device}")

        # Loading with trust_remote_code=True as required for Jina v3
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str, query: bool = False) -> np.ndarray:
        task = "retrieval.query" if query else "retrieval.passage"
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Jina v3 uses the 'task' argument in forward pass
        outputs = self.model(**inputs, task=task)
        embeddings = outputs.last_hidden_state[:, 0] # Jina v3 typically uses [CLS] or similar pooling internally, but check model card
        
        # NOTE: Jina v3 often provides a dedicated encode method if loaded via sentence-transformers, 
        # but here we use the raw model. According to Jina docs, we should use the task adapter.
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings[0].cpu().float().numpy()

    def get_embedding_dims(self) -> int:
        return self.embedding_dims


class KaLMEncoder(EncodingStrategy):
    """
    tencent/KaLM-Embedding-Gemma3-12B-2511 encoder.
    Uses last-token pooling.
    Dimension: 3840.
    """
    def __init__(
        self,
        *,
        model_name: str = "tencent/KaLM-Embedding-Gemma3-12B-2511",
        embedding_dims: int = 3840,
        device: Optional[str] = None,
        max_length: int = 8192,
        normalize: bool = True,
        log: Optional["RunLogger"] = None,
    ):
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = normalize

        if log:
            self.log = log
            log.info(f"KaLMEncoder initializing on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left" # recommended for last-token pooling
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str, query: bool = False) -> np.ndarray:
        # KaLM Gemma3 12B typically uses last-token pooling
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        
        # Last token pooling
        attention_mask = inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        embeddings = last_hidden_state[torch.arange(batch_size, device=self.device), sequence_lengths]

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings[0].cpu().float().numpy()

    def get_embedding_dims(self) -> int:
        return self.embedding_dims


class OpenRouterEncoder(EncodingStrategy):
    """
    OpenRouter API-based encoder using Qwen 4B embedding model.
    Makes HTTP requests to OpenRouter's embeddings endpoint.
    Dimension: 2560 (for qwen/qwen3-embedding-4b).
    """
    def __init__(
        self,
        *,
        model_name: str = "qwen/qwen3-embedding-4b",
        embedding_dims: int = 2560,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
        log: Optional["RunLogger"] = None,
        run_on_cluster: bool = False,
    ):
        import requests
        from pathlib import Path
        
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self.max_retries = max_retries
        self.log = log
        self.run_on_cluster = run_on_cluster
        
        # Handle cluster vs. local configuration
        if run_on_cluster:
            cluster_config = self._load_cluster_config()
            self.base_url = cluster_config.get("CLUSTER_BASE_URL", base_url).rstrip("/")
            self.api_key = cluster_config.get("CLUSTER_API_KEY", api_key)
            
            # Auto-prepend 'openrouter/' if missing for cluster proxy
            if not self.model_name.startswith("openrouter/"):
                self.model_name = f"openrouter/{self.model_name}"
                
            if log:
                log.info(f"🚀 OpenRouterEncoder running on CLUSTER mode. URL: {self.base_url}, Model: {self.model_name}")
        else:
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or self._load_api_key_from_file()
        
        if not self.api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Provide via argument or set OPENROUTER_API_KEY."
            )
        
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add optional headers
        referer = os.getenv("OPENROUTER_REFERER")
        title = os.getenv("OPENROUTER_TITLE")
        if referer:
            self._headers["HTTP-Referer"] = referer
        if title:
            self._headers["X-Title"] = title
        
        if log:
            log.info(f"✅ OpenRouterEncoder ready – model {self.model_name}")
        
        # Create a simple tokenizer for chunking (using tiktoken as fallback)
        # This is needed for the chunker compatibility
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B", use_fast=True)
        except Exception:
            # Fallback: create a minimal tokenizer-like object
            if log:
                log.warning("⚠️ Could not load Qwen tokenizer for OpenRouterEncoder, using minimal fallback")
            self.tokenizer = None
    
    def encode(self, text: str, query: bool = False) -> np.ndarray:
        """Encode text using OpenRouter embeddings API."""
        import requests
        import time
        
        payload = {
            "model": self.model_name,
            "input": text,
        }
        
        url = f"{self.base_url}/embeddings"
        
        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self._headers,
                    timeout=120,
                )
                response.raise_for_status()
                body = response.json()
                
                # Extract embedding from response
                # OpenAI-compatible format: {"data": [{"embedding": [...]}]}
                embedding = body["data"][0]["embedding"]
                
                # Convert to numpy array
                return np.array(embedding, dtype=np.float32)
                
            except Exception as exc:
                if self.log:
                    self.log.warning(
                        f"⚠️ OpenRouter embedding call failed ({exc}) – retry {attempt}/{self.max_retries}"
                    )
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Failed to get embedding after {self.max_retries} retries: {exc}")
                time.sleep(2 ** attempt)
    
    def get_embedding_dims(self) -> int:
        return self.embedding_dims
    
    def _load_api_key_from_file(self) -> Optional[str]:
        """Attempt to load the OpenRouter API key from a config file."""
        from pathlib import Path
        
        candidate_paths = [
            Path(__file__).resolve().parents[1] / "llm" / "openrouter.txt", 
        ]
        
        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key:
                            if self.log:
                                self.log.info(f"🔐 Loaded OpenRouter API key from {path}")
                            return key
            except OSError as exc:
                if self.log:
                    self.log.warning(f"⚠️ Could not read {path} ({exc})")
        return None
    
    def _load_cluster_config(self) -> dict:
        """Attempt to load the Cluster config from a file."""
        from pathlib import Path
        
        config = {}
        candidate_paths = [
            Path(__file__).resolve().with_name("cluster.txt"),
            Path(__file__).resolve().parents[2] / "common/llm/cluster.txt",
        ]
        
        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        config[k.strip()] = v.strip()
                
                if config:
                    if self.log:
                        self.log.info(f"🔐 Loaded Cluster config from {path}")
                    return config
            except OSError as exc:
                if self.log:
                    self.log.warning(f"⚠️ Could not read {path} ({exc})")
        return config
