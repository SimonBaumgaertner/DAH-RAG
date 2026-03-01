import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.data_classes.qa import Choice, QuestionAnswerPair
from common.data_classes.rag_system import Chunk
from common.data_classes.evaluation import LLMCallContext
from common.templates.answer_mc_question_template import AnswerMCQuestionTemplate
import time
from contextlib import ExitStack
from textwrap import dedent

from experiments.base_experiment import prepare_llm, prepare_log

RUN_ON_CLUSTER = False


def main() -> None:
    log = prepare_log("LLM-Test")
    qa_pair = QuestionAnswerPair(
        question_id="1",
        correct_answer="correct_answer",
        proofs=[],
        question=dedent(
            """
            In the context of interplanetary mission planning, consider a fictional research initiative called the Chronos Project. 
            This project requires identifying the most suitable celestial body for establishing a long-term autonomous research habitat that focuses 
            on studying paleoclimatic records, analyzing regolith chemistry, and deploying an array of atmospheric monitoring stations. 
            The selected location must exhibit prominent iron oxide deposits leading to a reddish appearance, possess geological evidence 
            of ancient fluvial networks, and have been historically nicknamed after a deity associated with war. Which option best satisfies all these constraints?
            """
        ).strip(),
        choices=[
            Choice(label="A", text="Ganymede"),
            Choice(label="B", text="Mars"),
            Choice(label="C", text="Europa"),
            Choice(label="D", text="Mercury"),
        ],
    )

    chunks = [
        Chunk(
            text=dedent(
                """
                Historical and spectroscopic analyses reveal that Mars exhibits a pervasive rusty hue due to abundant iron oxide in its regolith. 
                Geological surveys conducted by orbiters and rovers, including the Perseverance and Curiosity missions, have mapped extensive 
                evidence of paleo-river deltas, suggesting sustained fluvial activity billions of years ago. Mars has long been known as the "Red Planet" 
                and is named after the Roman god of war, reinforcing its cultural association with reddish coloration.
                """
            ).strip(),
            chunk_id="1",
        ),
        Chunk(
            text=dedent(
                """
                Contemporary mission design frameworks evaluate atmospheric variability and dust storm patterns on Mars to plan autonomous habitat deployments. 
                Advances in in-situ resource utilization have made it feasible to extract water from subsurface ice deposits, enabling long-duration scientific operations. 
                Comparative analyses demonstrate that other bodies such as Ganymede or Europa lack the same combination of iron oxide surface features and ancient river systems.
                """
            ).strip(),
            chunk_id="2",
        ),
    ]

    messages = AnswerMCQuestionTemplate().build_from_template(qa_pair, chunks)


    remote_llm = prepare_llm(
        run_on_cluster=RUN_ON_CLUSTER,
        log=log,
        backend="openrouter",
        gen_kwargs={"max_tokens": 1024, "temperature": 0.8, "top_p": 0.95},
        )


if __name__ == "__main__":
    main()
