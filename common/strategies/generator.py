# generation_strategy.py
from __future__ import annotations

import re
from typing import List, Optional

from common.data_classes.evaluation import EntryType
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, Generator
from common.llm.message_template import MessageTemplate
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.templates.answer_mc_question_template import AnswerMCQuestionTemplate
from common.templates.answer_open_question_template import AnswerOpenQuestionTemplate


class StandardMCAnswerGenerator(Generator):
    def __init__(
        self,
        *,
        llm: BaseLLMRunner,
        log: RunLogger,
        template: Optional[MessageTemplate] = None,
        max_chunks: int = 5,
    ):
        self.llm = llm
        self.log = log
        self._template = template or AnswerMCQuestionTemplate()
        self._max_chunks = max_chunks

    def generate(
            self,
            qa_pair: QuestionAnswerPair,
            context: List[Chunk],
    ) -> str:
        self.log.info("📜 Building prompt for MC question '%s' …", qa_pair.question[:60])
        messages = self._template.build_from_template(qa_pair, context, max_chunks=self._max_chunks)

        self.log.info("🏭 Start generating answer for question %s.", qa_pair.question_id)

        # Use context-aware LLM call for generation
        from common.data_classes.evaluation import LLMCallContext
        raw = self.llm.generate_text(messages=messages, context=LLMCallContext.GENERATION, identifier=qa_pair.question_id).strip()

        match = re.search(r"\b([A-Z])\b", raw)
        return match.group(1) if match else raw


class StandardOpenAnswerGenerator(Generator):
    def __init__(
        self,
        *,
        llm: BaseLLMRunner,
        log: RunLogger,
        template: Optional[MessageTemplate] = None,
        max_chunks: int = 5,
    ):
        self.llm = llm
        self.log = log
        self._template = template or AnswerOpenQuestionTemplate()
        self._max_chunks = max_chunks

    def generate(
            self,
            qa_pair: QuestionAnswerPair,
            context: List[Chunk],
    ) -> str:
        self.log.info("📜 Building prompt for open question '%s' …", qa_pair.question[:60])
        messages = self._template.build_from_template(qa_pair, context, max_chunks=self._max_chunks)

        self.log.info("🏭 Start generating answer for question %s.", qa_pair.question_id)

        # Use context-aware LLM call for generation
        from common.data_classes.evaluation import LLMCallContext
        raw = self.llm.generate_text(messages=messages, context=LLMCallContext.GENERATION, identifier=qa_pair.question_id).strip()

        return raw


class DummyGenerator(Generator):
    def generate(
            self,
            qa_pair: QuestionAnswerPair,
            context: List[Chunk],
    ) -> str:
        return "F"
