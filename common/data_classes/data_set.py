import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from common.data_classes import documents
from common.data_classes.documents import Document
from common.data_classes.qa import QuestionAnswerPair


@dataclass
class DataSet:
    documents: List[Document]
    qa_pairs: List[QuestionAnswerPair] = None

    def __init__(self, data_set_path: Path):
        self.documents = self.load_documents(data_set_path)
        self.qa_pairs = self.load_qa(data_set_path)

    def load_documents(self, root: str | Path) -> List[Document]:
        root = Path(root)
        docs: List[Document] = []
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                docs.append(Document.from_folder(sub))
        return docs

    def load_qa(self, qa_path: str | Path) -> List[QuestionAnswerPair]:
        qa_path = qa_path / "QA.json"
        qa_pairs: List[QuestionAnswerPair] = []
        if qa_path.exists():
            try:
                with qa_path.open(encoding="utf-8") as f:
                    qa_list = json.load(f) or []
                qa_pairs = [QuestionAnswerPair.from_dict(q) for q in qa_list]
            except json.JSONDecodeError:
                qa_pairs = []
        for document in self.documents:
            qa_pairs.extend(document.qa_pairs)
        return qa_pairs

