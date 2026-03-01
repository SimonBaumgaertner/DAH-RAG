import os
import re
import time
import uuid
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import aiohttp
import asyncio

from common.data_classes.evaluation import EntryType, LLMCallContext
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
import tiktoken
import re

class OpenRouterAPIRunner(BaseLLMRunner):
    """Runner that fetches chat completions via the OpenRouter API."""

    def __init__(
        self,
        model: str,
        *,
        log: RunLogger,
        api_key: Optional[str] = None,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        max_retries: int = 2,
        chat_kwargs: Optional[Dict[str, Any]] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        llm_short_name: str = "openrouter",
        max_concurrent_llm_executions: int = 10,
        run_on_cluster: bool = False,
    ) -> None:
        default_gen = {"temperature": 0.7, "max_tokens": 8192}
        merged = {**default_gen, **(gen_kwargs or {})}
        super().__init__(log=log, gen_kwargs=merged, llm_short_name=llm_short_name, max_concurrent_executions=max_concurrent_llm_executions)

        self.model = model
        self.run_on_cluster = run_on_cluster
        
        if run_on_cluster:
            cluster_config = self._load_cluster_config()
            self.base_url = cluster_config.get("CLUSTER_BASE_URL", base_url).rstrip("/")
            self.api_key = cluster_config.get("CLUSTER_API_KEY", api_key)
            
            # Auto-prepend 'openrouter/' if missing, as required by the cluster proxy
            if not self.model.startswith("openrouter/"):
                self.model = f"openrouter/{self.model}"
                
            self.log.info("🚀 Running on CLUSTER mode. URL: %s, Model: %s", self.base_url, self.model)
        else:
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or self._load_api_key_from_file()

        self.referer = referer or os.getenv("OPENROUTER_REFERER")
        self.title = title or os.getenv("OPENROUTER_TITLE")

        self.max_retries = max_retries
        self.chat_kwargs = chat_kwargs or {}

        if not self.api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Provide via argument or set OPENROUTER_API_KEY."
            )

        self._headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            self._headers["HTTP-Referer"] = self.referer
        if self.title:
            self._headers["X-Title"] = self.title

        self.log.info("✅ OpenRouterAPIRunner ready – model %s", self.model)
        
        # Initialize async session for true async HTTP calls
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------

    def _request_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        payload.update(self.chat_kwargs)
        payload.update(self.gen_kwargs)
        return payload

    def _completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        context: LLMCallContext = LLMCallContext.INDEXING,
        identifier: str = None,
    ) -> str:
        # Generate a random identifier if none provided
        if identifier is None:
            identifier = f"call_{uuid.uuid4().hex[:8]}"

        payload = self._request_payload(messages)

        input_tokens = 0
        # We don't track input tokens here anymore, we'll use the 'usage' from the response if it returns successfully
        # unless it fails, in which case we might want to know how many tokens we sent, but let's keep it simple.

        attempt = 0
        while True:
            attempt += 1
            start_time = time.perf_counter_ns()
            try:
                response = requests.post(
                    self._completions_url(),
                    json=payload,
                    headers=self._headers,
                    timeout=120,
                )
                response.raise_for_status()
                body = response.json()

                message = body["choices"][0]["message"]["content"]
                self.log.debug("📥 Raw OpenRouter output before cleaning:\n%s", message)

                clean_output = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL).strip()
                self.log.debug("📤 Cleaned output:\n%s", clean_output)
                
                # Track timing, tokens and cost
                if self.log:
                    duration_ms = (time.perf_counter_ns() - start_time) / 1e6
                    self.log.track_llm_call(context=context, identifier=identifier, duration_ms=duration_ms)

                    usage = body.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    total_cost = usage.get("cost", 0)
                    
                    if input_tokens or output_tokens:
                        self.log.track_llm_tokens(context=context, identifier=identifier, input_tokens=input_tokens, output_tokens=output_tokens)
                    
                    if total_cost:
                        self.log.track_llm_cost(context=context, identifier=identifier, cost=total_cost)

                    # Calculate cost per million tokens if details are available
                    cost_details = usage.get("cost_details", {})
                    prompt_cost = cost_details.get("upstream_inference_prompt_cost") or 0
                    completion_cost = cost_details.get("upstream_inference_completions_cost") or 0
                    
                    prompt_rate = (prompt_cost / input_tokens * 1_000_000) if input_tokens else 0
                    completion_rate = (completion_cost / output_tokens * 1_000_000) if output_tokens else 0
                    
                    # Log generation completion
                    self.log.info(
                        "✅ Generation complete – %.0f ms, %d in tok, %d out tok with %.2f$/M Input & %.2f$/M Output tokens.",
                        duration_ms,
                        input_tokens,
                        output_tokens,
                        prompt_rate,
                        completion_rate
                    )
                
                return clean_output
            except Exception as exc:  # pragma: no cover - network dependent
                if self.log:
                    duration_ms = (time.perf_counter_ns() - start_time) / 1e6
                    self.log.track_llm_call(context=context, identifier=identifier, duration_ms=duration_ms)
                
                self.log.warning(
                    "⚠️ OpenRouter call failed (%s) – retry %d/%d",
                    exc,
                    attempt,
                    self.max_retries,
                )
                if attempt >= self.max_retries:
                    raise
                time.sleep(2 ** attempt)

    async def generate_text_async(
        self,
        messages: List[Dict[str, str]],
        context: LLMCallContext = LLMCallContext.INDEXING,
        identifier: str = None,
    ) -> str:
        """True async implementation using aiohttp with semaphore control."""
        # Use the semaphore from the base class to limit concurrent executions
        async with self._semaphore:
            # Generate a random identifier if none provided
            if identifier is None:
                identifier = f"call_{uuid.uuid4().hex[:8]}"

            payload = self._request_payload(messages)

            input_tokens = 0
            # Token tracking is now done using response body usage stats

            attempt = 0
            while True:
                attempt += 1
                start_time = time.perf_counter_ns()
                try:
                    # Create a new session for each request to avoid context issues
                    async with aiohttp.ClientSession(headers=self._headers) as session:
                        async with session.post(
                            self._completions_url(),
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=120),
                        ) as response:
                            response.raise_for_status()
                            body = await response.json()

                            message = body["choices"][0]["message"]["content"]
                            self.log.debug("📥 Raw OpenRouter output before cleaning:\n%s", message)

                            clean_output = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL).strip()
                            self.log.debug("📤 Cleaned output:\n%s", clean_output)
                            
                            # Track timing, tokens and cost
                            if self.log:
                                duration_ms = (time.perf_counter_ns() - start_time) / 1e6
                                self.log.track_llm_call(context=context, identifier=identifier, duration_ms=duration_ms)

                                usage = body.get("usage", {})
                                input_tokens = usage.get("prompt_tokens", 0)
                                output_tokens = usage.get("completion_tokens", 0)
                                total_cost = usage.get("cost", 0)
                                
                                if input_tokens or output_tokens:
                                    self.log.track_llm_tokens(context=context, identifier=identifier, input_tokens=input_tokens, output_tokens=output_tokens)
                                
                                if total_cost:
                                    self.log.track_llm_cost(context=context, identifier=identifier, cost=total_cost)

                                # Calculate cost per million tokens if details are available
                                cost_details = usage.get("cost_details", {})
                                prompt_cost = cost_details.get("upstream_inference_prompt_cost") or 0
                                completion_cost = cost_details.get("upstream_inference_completions_cost") or 0
                                
                                prompt_rate = (prompt_cost / input_tokens * 1_000_000) if input_tokens else 0
                                completion_rate = (completion_cost / output_tokens * 1_000_000) if output_tokens else 0
                                
                                # Log generation completion
                                self.log.info(
                                    "✅ Generation complete – %.0f ms, %d in tok, %d out tok with %.2f$/M Input & %.2f$/M Output tokens.",
                                    duration_ms,
                                    input_tokens,
                                    output_tokens,
                                    prompt_rate,
                                    completion_rate
                                )
                            
                            return clean_output
                except Exception as exc:  # pragma: no cover - network dependent
                    if self.log:
                        duration_ms = (time.perf_counter_ns() - start_time) / 1e6
                        self.log.track_llm_call(context=context, identifier=identifier, duration_ms=duration_ms)
                    
                    self.log.warning(
                        "⚠️ OpenRouter call failed (%s) – retry %d/%d",
                        exc,
                        attempt,
                        self.max_retries,
                    )
                    if attempt >= self.max_retries:
                        raise
                    await asyncio.sleep(2 ** attempt)

    def tokenize(self, text: str, **kwargs: Any) -> List[int]:
        if tiktoken is None:
            raise ImportError("Install `tiktoken` for tokenisation support.")
        encoding_name = self.chat_kwargs.get("encoding", None)
        if encoding_name:
            enc = tiktoken.get_encoding(encoding_name)
        else:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text, **kwargs)

    def dispose(self) -> None:
        # No persistent session to clean up since we create new sessions per request
        self.log.info("🧹 OpenRouter runner disposed")

    # ------------------------------------------------------------------

    def _load_api_key_from_file(self) -> Optional[str]:
        """Attempt to load the OpenRouter API key from a config file."""
        candidate_paths = [
            Path(__file__).resolve().with_name("openrouter.txt"),
            Path(__file__).resolve().parents[2] / "openrouter.txt",
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
                            self.log.info("🔐 Loaded OpenRouter API key from %s", path)
                            return key
            except OSError as exc:
                self.log.warning("⚠️ Could not read %s (%s)", path, exc)
        return None


    def _load_cluster_config(self) -> Dict[str, str]:
        """Attempt to load the Cluster config from a file."""
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
                    self.log.info("🔐 Loaded Cluster config from %s", path)
                    return config
            except OSError as exc:
                self.log.warning("⚠️ Could not read %s (%s)", path, exc)
        return config
