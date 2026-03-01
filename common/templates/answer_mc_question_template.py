from typing import List, Dict

from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk
from common.llm.message_template import MessageTemplate


class AnswerMCQuestionTemplate(MessageTemplate):
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. "
        "Answer the user's question using the given context (if available). Stop thinking as soon as you found the answer and respond. "
        "Only return the label of the multiple choice option (If you don't know guess any one of the options). "
        "For example, if the answer is B, respond only with 'B' — NOTHING ELSE, ONLY THE LETTER."
    )

    QUESTION_PROMPT = "USER QUESTION:\n{query}"

    def build_from_template(
        self,
        qa_pair: QuestionAnswerPair,
        chunks: List[Chunk],
        max_chunks: int = 5,
    ) -> List[Dict[str, str]]:
        selected_chunks = chunks[:max_chunks]

        context_block = ""
        if selected_chunks:
            context_block = "CONTEXT:\n" + "\n".join(
                f"{idx}. {chunk.text.strip()}" for idx, chunk in enumerate(selected_chunks, start=1)
            )

        choices_block = "\n".join(f"{c.label}. {c.text}" for c in qa_pair.choices)
        question_block = f"{qa_pair.question}\n\nChoices:\n{choices_block}"

        # Merge system prompt and context into one system message
        system_content = self.SYSTEM_PROMPT
        if context_block:
            system_content += f"\n\n{context_block}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self.QUESTION_PROMPT.format(query=question_block)},
        ]
