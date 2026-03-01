from typing import List, Dict

from common.llm.message_template import MessageTemplate


class OneShotTripletExtractionTemplate(MessageTemplate):
    """Template that primes the model with one worked example."""

    SYSTEM_PROMPT = (
        """Your task is to extract knowledge triplets from text. Knowledge Triplets are a (Subject, Relationship, Object) triplets. The Subject and Objects are SINGULAR SHORT Entities like Names (e.g. of people, places) or singular Concepts (e.g. professions, things, etc). Respond with a JSON list of triples. The triples should reflect the relevant relationships between two Entities. Most sentences don't have a relevant triplet and some text don't have any, only extract really relevant ones.
        You shoud NEVER extract triplets like this:\n
        [\"Tom", \"though\", \"it looked deep\"]" (Bad because not relevant relation of two entities)
        [\"Jason", \"warned about'\", \"The impending doom of financial ruin due to the recession\"] (Bad because too long)
        """
    )

    USER_PROMPT = "Extract entities and relations from the following text:\n{text}"

    # -- one‑shot demonstration ------------------------------------------------
    DEMO_TEXT = (
        "The Dursley family of number four, Privet Drive, was the reason that "
        "Harry never enjoyed his summer holidays. Uncle Vernon, Aunt Petunia, "
        "and their son, Dudley, were Harry’s only living relatives. They were "
        "Muggles. Harry’s dead parents, who had been a witch and wizard "
        "themselves, were never mentioned under the Dursleys’ roof. These days "
        "they lived in terror of anyone finding out that Harry had spent most "
        "of the last two years at Hogwarts School of Witchcraft and Wizardry."
    )

    DEMO_OUTPUT = """{
      \"knowledge_triplets\": [
        [\"The Dursley family\", \"lives at\", \"number four Privet Drive\"],
        [\"Aunt Petunia\", \"is\", \"Harry's relative\"],
        [\"Uncle Vernon\", \"is\", \"Harry's relative\"],
        [\"Dudley\", \"is\", \"Harry's relative\"],
        [\"The Dursleys\", \"are\", \"Muggles\"],
        [\"Harry's dead parents\", \"were\", \"Witch and Wizards\"],
        [\"Harry\", \"spent most of the last two years at\", \"Hogwarts\"]
      ]
    }"""

    def build_from_template(self, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(text=self.DEMO_TEXT)},
            {"role": "assistant", "content": self.DEMO_OUTPUT},
            {"role": "user", "content": self.USER_PROMPT.format(text=text)},
        ]
