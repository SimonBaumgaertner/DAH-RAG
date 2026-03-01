import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from common.data_classes.evaluation import LLMCallContext
from common.logging.run_logger import RunLogger

class BaseLLMRunner(ABC):
    def __init__(self, *, log: Optional[RunLogger] = None, llm_short_name: str="?", gen_kwargs: Optional[Dict[str, Any]] = None, max_concurrent_executions: int = 6):
        self.log: RunLogger = log
        # Default generation parameters applied to every call
        self.gen_kwargs: Dict[str, Any] = gen_kwargs or {}
        self.llm_short_name = llm_short_name
        self.max_concurrent_executions = max_concurrent_executions
        # Dictionary to store semaphores per event loop to support parallel indexing across loops
        self._loop_semaphores: Dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    @property
    def _semaphore(self) -> asyncio.Semaphore:
        """
        Returns an asyncio.Semaphore bound to the current event loop.
        Creates a new one if it doesn't exist for the current loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback for cases where no loop is running yet (should be rare in async contexts)
            loop = asyncio.get_event_loop()
            
        if loop not in self._loop_semaphores:
            self._loop_semaphores[loop] = asyncio.Semaphore(self.max_concurrent_executions)
        return self._loop_semaphores[loop]

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        context: LLMCallContext,
        identifier: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    def tokenize(self, text: str, **kwargs: Any) -> List[int]: ...

    @abstractmethod
    def dispose(self) -> None: ...

    async def generate_text_async(
        self,
        messages: List[Dict[str, str]],
        context: LLMCallContext,
        identifier: Optional[str] = None,
    ) -> str:
        """
        Async version of generate_text that respects the max_concurrent_executions limit.
        Subclasses should override this method to provide true async implementation.
        """
        async with self._semaphore:
            # Run the synchronous method in a thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.generate_text, messages, context, identifier)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()
