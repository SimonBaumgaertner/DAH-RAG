import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        return BedrockLLM(config)
    
    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)
    
    # Check if we have a custom LLM runner available
    try:
        import sys
        from pathlib import Path
        # Add the project root to the path
        project_root = Path(__file__).parents[4]
        sys.path.insert(0, str(project_root))
        
        from common.llm.open_router_llm_runner import OpenRouterAPIRunner
        from common.logging.run_logger import RunLogger
        from common.strategies.encoding import QwenEncoder
        from rag_approaches.hippo_rag.hippo_rag import _RunnerBackedLLM
        
        # Use the logger from the config if available, otherwise create a new one
        if hasattr(config, 'experiment_logger') and config.experiment_logger:
            log = config.experiment_logger
        
        runner = OpenRouterAPIRunner(
            model='meta-llama/llama-3.3-70b-instruct', 
            log=log,
            run_on_cluster=getattr(config, 'run_on_cluster', False)
        )
        return _RunnerBackedLLM(runner=runner, global_config=config)
    except Exception as e:
        print(f"🔧 Failed to create custom LLM, falling back to CacheOpenAI: {e}")
        return CacheOpenAI.from_experiment_config(config)
    