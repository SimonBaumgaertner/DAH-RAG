import os
import time
import re
import uuid
from pathlib import Path
from typing import *

from common.data_classes.evaluation import EntryType, LLMCallContext
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalInstructLLMRunner(BaseLLMRunner):
    """Minimal HF backend for Qwen3 Instruct models (singleton)."""

    _instance: Optional["LocalInstructLLMRunner"] = None
    _lock = object()

    def __new__(cls, *a, **kw):
        if cls._instance is None:
            # simple, thread-unsafe singleton is fine here unless you need multithreaded construction
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model: str | os.PathLike,
        *,
        log: RunLogger,
        torch_dtype: "torch.dtype | str" = "auto",
        device_map: str | Dict[str, int] | None = "auto",
        trust_remote_code: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        llm_short_name: str = "?"
    ) -> None:
        if getattr(self, "_initialised", False):
            return  # already built

        default_gen = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 4096}
        merged = {**default_gen, **(gen_kwargs or {})}
        super().__init__(log=log, gen_kwargs=merged, llm_short_name=llm_short_name)

        if torch is None:
            raise ImportError("Install `torch` and `transformers` for the local backend.")

        self.model_path = Path(model).expanduser()
        self.dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}

        self._load_model()
        self._initialised = True

        # Special token ids (robust to missing tokens)
        try:
            self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            self.im_end_id = None
        self.eos_id = self.tokenizer.eos_token_id

        self.log.info("✅ LocalInstructLLMRunner ready – model %s", self.model_path)

        # --- Log hardware info ---
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current = torch.cuda.current_device()
            self.log.info(
                "🖥️ Using CUDA (%d device(s)) – current: %s (%s)",
                device_count,
                current,
                torch.cuda.get_device_name(current),
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.log.info("🍎 Using Apple Metal Performance Shaders (MPS)")
        else:
            self.log.info("🖥️ Using CPU for inference")

    # ---------------- internal helpers ------------------------------------

    def _load_model(self) -> None:
        self.log.info("🔠 Loading tokenizer from %s", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path.as_posix(),
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            **self.tokenizer_kwargs,
        )
        self.log.info("🪄 Loading model weights (%s)", self.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path.as_posix(),
            torch_dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            low_cpu_mem_usage=True,
            **self.model_kwargs,
        ).eval()

    # ---------------- public API ------------------------------------------

    def generate_text(self, messages: List[Dict[str, str]], context: LLMCallContext = LLMCallContext.INDEXING, identifier: str = None) -> str:
        # Generate a random identifier if none provided
        if identifier is None:
            identifier = f"call_{uuid.uuid4().hex[:8]}"
        self.log.info("📝 Starting full-text generation")

        prompt = self._build_prompt_from_messages(messages)
        self.log.debug("🔧 Prompt built:\n%s", prompt)

        # Track input tokens
        input_tokens = 0
        if self.log:
            input_tokens = len(self.tokenize(prompt))
            if input_tokens:
                self.log.track_llm_tokens(context=context, identifier=identifier, input_tokens=input_tokens)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        cfg = self.gen_kwargs.copy()
        # Avoid passing None into eos_token_id / pad_token_id
        eos_list = [tid for tid in (self.eos_id, self.im_end_id) if tid is not None]
        if eos_list:
            cfg.setdefault("eos_token_id", eos_list)
            cfg.setdefault("pad_token_id", eos_list[0])

        start_time = time.perf_counter_ns()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **cfg)
        duration_ms = (time.perf_counter_ns() - start_time) / 1e6

        gen_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        clean_text = self._clean_output(text)
        
        # Track timing and output tokens
        if self.log:
            self.log.track_llm_call(context=context, identifier=identifier, duration_ms=duration_ms)
            
            output_tokens = 0
            if clean_text:
                output_tokens = len(self.tokenize(clean_text))
                if output_tokens:
                    self.log.track_llm_tokens(context=context, identifier=identifier, output_tokens=output_tokens)
            
            # Log generation completion
            self.log.info(
                "✅ Generation complete – %.0f ms, %d in tok, %d out tok",
                duration_ms,
                input_tokens,
                output_tokens,
            )
        
        return clean_text.strip()

    def tokenize(self, text: str, **kwargs: Any) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)

    def _build_prompt_from_messages(self, messages: Sequence[Dict[str, str]]) -> str:
        # Qwen3-4B-Instruct-2507 is non-thinking; no special flags needed
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


    def _clean_output(self, raw: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"</?think>", "", cleaned)
        return cleaned.strip()

    def dispose(self) -> None:
        if torch is None:
            return
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
        LocalInstructLLMRunner._instance = None
        self.log.info("🧹 Local runner disposed.")
