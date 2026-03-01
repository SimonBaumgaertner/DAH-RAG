import os
from typing import Any, Optional, Dict

from common.llm.base_llm_runner import BaseLLMRunner
from common.llm.local_instruct_llm_runner import LocalInstructLLMRunner
from common.llm.open_router_llm_runner import OpenRouterAPIRunner
from common.llm.dummy_llm_runner import DummyLLMRunner
from common.logging.run_logger import RunLogger


def get_llm_runner(
    *,
    backend: str,
    model: str,
    log: RunLogger,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    llm_short_name: str = "?",
    run_on_cluster: bool = False,
    max_concurrent_llm_executions: int = 10
) -> BaseLLMRunner:
    backend_lc = backend.lower()
    if backend_lc == "local-instruct":
        return LocalInstructLLMRunner(model=model, log=log, gen_kwargs=gen_kwargs, llm_short_name=llm_short_name, run_on_cluster=run_on_cluster, max_concurrent_executions=max_concurrent_llm_executions)
    if backend_lc in {"openrouter", "open-router", "router"}:
        return OpenRouterAPIRunner(model=str(model), log=log, gen_kwargs=gen_kwargs, llm_short_name=llm_short_name, run_on_cluster=run_on_cluster, max_concurrent_llm_executions=max_concurrent_llm_executions)
    if backend_lc == "dummy":
        return DummyLLMRunner(model=model, log=log, gen_kwargs=gen_kwargs, llm_short_name=llm_short_name, max_concurrent_executions=max_concurrent_llm_executions)
    raise ValueError(f"Unknown backend '{backend}'. Expected 'local', 'openai'.")
