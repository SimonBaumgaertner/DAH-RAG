from pathlib import Path
import sys
import math

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.data_classes.data_set import DataSet


def analyze_dataset(data_set: DataSet, name: str = "Selected Dataset"):
    documents = data_set.documents
    qa_pairs = data_set.qa_pairs

    num_documents = len(documents)
    document_qa_pairs = sum(len(doc.qa_pairs or []) for doc in documents)
    dataset_qa_pairs = len(qa_pairs or [])

    total_words = 0
    total_chunks = 0
    chunk_size = 1200
    
    for doc in documents:
        if doc.text:
            doc_words = len(doc.text.split())
            doc_tokens = int(doc_words * 1.3)
            # Calculate chunks per document and round up
            doc_chunks = math.ceil(doc_tokens / chunk_size)
            total_chunks += doc_chunks
            total_words += doc_words
    
    total_tokens = int(total_words * 1.3)

    print(f"Dataset: {name}")
    print(f"  Documents: {num_documents:,}".replace(",", " "))
    print(f"  Document specific QA pairs: {document_qa_pairs:,}".replace(",", " "))
    print(f"  Dataset QA pairs: {dataset_qa_pairs:,}".replace(",", " "))
    print(f"  Approximate tokens: {total_tokens:,}".replace(",", " "))
    print(f"  Approximate Chunks (for {chunk_size} Chunk size): {total_chunks:,}".replace(",", " "))

    return {
        "documents": num_documents,
        "document_qa_pairs": document_qa_pairs,
        "dataset_qa_pairs": dataset_qa_pairs,
        "tokens": total_tokens,
        "chunks": total_chunks,
    }


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parent.parent.parent / "data"

    for folder in data_root.iterdir():
        if folder.is_dir():
            try:
                data_set = DataSet(folder)
                analyze_dataset(data_set, folder.name)
            except Exception as e:
                print(f"Skipping {folder.name} due to error: {e}")
