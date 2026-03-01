import pytest
import sys
from pathlib import Path

# Add the project root to Python path so it works from wherever you are
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.data_classes.enums import Encoder, ChunkingStrategy, LLMBackend, LLMName
from common.data_classes.documents import Document
from experiments.base_experiment import prepare_log, prepare_llm
from dah_rag_app.dah_rag_database import DAHRAGDatabase


def test_dah_rag_database_integration():
    """
    Integration test for DAHRAGDatabase.
    Requires Neo4j and a valid environment.
    """
    # 1. Prepare Standard Log
    log = prepare_log("test_dah_rag_db_integration")

    # 2. Prepare Standard LLM (OpenRouter Llama 3 for fast processing, or dummy if preferred)
    llm = prepare_llm(
        run_on_cluster=False,
        log=log,
        backend=LLMBackend.OpenRouter.value,
        llm_name=LLMName.Llama_3_3_70B.value,
        max_concurrent_llm_executions=6
    )

    # 3. Initialize Database Wrapper
    # Use standard encoder and chunking
    db = DAHRAGDatabase(
        llm=llm, 
        log=log, 
        encoder=Encoder.Qwen3_4B, 
        chunking_strategy=ChunkingStrategy.ContextualizedChunker,
        checkpoint_name="test_dah_rag_interface"
    )

    # 4. Test Initialization (wiping database)
    success = db.initialize_database(wipe_at_start=True)
    assert success is True, "Database failed to initialize"
    
    # 5. Add two documents from HotpotQA_100
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "HotpotQA_100"
    
    # Check if dataset path exists
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"
    
    # Get first two document folders
    doc_folders = [d for d in dataset_path.iterdir() if d.is_dir()][:2]
    assert len(doc_folders) >= 2, "Need at least 2 documents in HotpotQA_100 to test"
    
    doc1 = Document.from_folder(doc_folders[0])
    doc2 = Document.from_folder(doc_folders[1])
    
    print(f"Adding documents: {doc1.id} and {doc2.id}")
    
    success1 = db.add_document(doc1)
    assert success1 is True, f"Failed to add document 1 ({doc1.id})"
    
    success2 = db.add_document(doc2)
    assert success2 is True, f"Failed to add document 2 ({doc2.id})"
    
    # 6. Test documents count
    count = db.get_all_documents_count()
    assert count == 2, f"Expected 2 documents in DB, found {count}"
    
    # 7. Query Database
    query = "What is the capital of France?" # Dummy query, won't match but should retrieve top_k
    chunks = db.query(query_text=query, top_k=2)
    
    assert len(chunks) <= 2, "Returned more chunks than requested"
    print(f"Query returned {len(chunks)} chunks")
    if chunks:
        print(f"Top chunk: {chunks[0].chunk_id}")

    # 8. Test get_all_documents
    all_docs = db.get_all_documents()
    assert len(all_docs) == 2, f"Expected 2 documents, got {len(all_docs)}"
    
    # Check that documents have basic properties and text
    for d in all_docs:
        assert d.id is not None, "Document ID is missing"
        assert d.title is not None, "Document title is missing"
        assert d.text is not None and len(d.text) > 0, f"Document text is missing or empty for {d.id}"
    
    print("get_all_documents passed successfully.")
    
    # 9. Test get_document_by_id
    target_id = doc1.id
    retrieved_doc = db.get_document_by_id(target_id)
    
    assert retrieved_doc is not None, f"Failed to retrieve document by id {target_id}"
    assert retrieved_doc.id == target_id, f"Retrieved document ID {retrieved_doc.id} does not match {target_id}"
    assert retrieved_doc.text is not None and len(retrieved_doc.text) > 0, "Retrieved document text is empty"
    
    print(f"get_document_by_id passed successfully for {target_id}.")

    # 10. Test remove_document
    count_before = db.get_all_documents_count()
    success_remove = db.remove_document(target_id)
    assert success_remove is True, f"Failed to remove document {target_id}"
    
    count_after = db.get_all_documents_count()
    assert count_after == count_before - 1, f"Expected {count_before - 1} documents after removal, got {count_after}"
    
    # Verify exact deletion by trying to retrieve it
    deleted_doc = db.get_document_by_id(target_id)
    assert deleted_doc is None, f"Document {target_id} was not fully removed, it can still be retrieved"
    
    driver = db._rag_system.retriever.env.get_driver()
    with driver.session() as session:
        # Check Chunks
        res_chunks = session.run("MATCH (c:Chunk) WHERE c.doc_id = $doc_id RETURN count(c) as count", doc_id=target_id)
        if res_chunks.single()["count"] > 0:
            print(f"Warning: Chunks for document {target_id} might not be properly deleted if they have doc_id set.")
            
    print(f"remove_document passed successfully for {target_id}. Document count is now {count_after}.")

    print("Test finished successfully")
if __name__ == "__main__":
    test_dah_rag_database_integration()
