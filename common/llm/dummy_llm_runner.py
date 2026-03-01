import time
import uuid
from typing import List, Dict, Any, Optional

from common.data_classes.evaluation import LLMCallContext
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger


class DummyLLMRunner(BaseLLMRunner):
    """Dummy LLM runner that always returns '42' for testing purposes."""

    def __init__(
        self,
        model: str,
        *,
        log: RunLogger,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        llm_short_name: str = "Dummy",
        max_concurrent_executions: int = 1
    ) -> None:
        super().__init__(log=log, gen_kwargs=gen_kwargs or {}, llm_short_name=llm_short_name, max_concurrent_executions=max_concurrent_executions)
        self.model = model
        self.log.info("✅ DummyLLMRunner ready – model %s", self.model)

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        context: LLMCallContext = LLMCallContext.INDEXING,
        identifier: Optional[str] = None,
    ) -> str:
        # Generate a random identifier if none provided
        if identifier is None:
            identifier = f"dummy_call_{uuid.uuid4().hex[:8]}"
        
        self.log.info("📝 Starting dummy text generation")
        
        # Simulate some processing time
        start_time = time.perf_counter_ns()
        time.sleep(0.1)  # Simulate 100ms processing time
        duration_ms = (time.perf_counter_ns() - start_time) / 1e6
        
        # Always return "42"
        response = "42"
        
        # Track tokens and timing for consistency with other runners
        if self.log:
            # Estimate input tokens (rough approximation)
            input_text = " ".join(msg.get("content", "") for msg in messages)
            input_tokens = len(input_text.split())  # Rough word count as token estimate
            
            # Track input tokens
            if input_tokens:
                self.log.track_llm_tokens(context=context, identifier=identifier, input_tokens=input_tokens)
            
            # Track timing
            self.log.track_llm_call(context=context, identifier=identifier, duration_ms=duration_ms)
            
            # Track output tokens (response is just "42", so 1 token)
            output_tokens = 1
            self.log.track_llm_tokens(context=context, identifier=identifier, output_tokens=output_tokens)
            
            # Log generation completion
            self.log.info(
                "✅ Dummy generation complete – %.0f ms, %d in tok, %d out tok",
                duration_ms,
                input_tokens,
                output_tokens,
            )
        
        return response

    def tokenize(self, text: str, **kwargs: Any) -> List[int]:
        """Simple tokenization that splits on whitespace and converts to integers."""
        # Simple word-based tokenization for dummy purposes
        words = text.split()
        # Convert each word to a hash-based token ID
        return [hash(word) % 10000 for word in words]

    def dispose(self) -> None:
        """Clean up resources (nothing to clean up for dummy runner)."""
        self.log.info("🧹 Dummy runner disposed.")
