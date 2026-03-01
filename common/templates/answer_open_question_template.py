from typing import List, Dict

from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk
from common.llm.message_template import MessageTemplate


class AnswerOpenQuestionTemplate(MessageTemplate):
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. "
        "Answer the user's question using the given context (if available). "
        "Provide a clear, concise, and accurate answer based on the information provided. "
        "ONLY return the answer - do not include any explanations, reasoning, or additional text. "
        "The answer should not be in a sentence but just the answer. E.g. if the answer is 'John Doe', you should return 'John Doe' only, NOT 'The answer is John Doe'. (Without the ' symbols)"
        "If you don't know the answer, respond with 'I don't know.'."
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

        question_block = qa_pair.question

        # Merge system prompt and context into one system message
        system_content = self.SYSTEM_PROMPT
        if context_block:
            system_content += f"\n\n{context_block}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self.QUESTION_PROMPT.format(query=question_block)},
        ]
