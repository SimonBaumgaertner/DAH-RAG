# Document-Aware Hybrid-RAG (MastersThesis)

Codebase for experiments conducted during my master's thesis on retrieval‑augmented generation (RAG) and a custom Document-Aware Hybrid-RAG.  It provides reusable components for building RAG systems, multiple end‑to‑end approaches, and scripts to run indexing and retrieval experiments. Note it is not a product in any way but primarily the scientific experiments conducted for my Masters Thesis.

## Project layout

```
├── common/             # Reusable building blocks shared across approaches
├── dah_rag_app/        # Document-aware app structures and database wrappers
├── data/               # Datasets used in experiments
├── experiments/        # Scripts wiring components into full pipelines
├── logs_and_tracks/    # Run logs and metric CSVs
├── rag_approaches/     # Different RAG system implementations
├── util/               # Data conversion and helper scripts
├── hipporag_requirements.txt # Specific dependencies for HippoRAG approach
└── requirements.txt    # Python dependencies
```

### `common/`

Foundational modules that implement the core logic of the project:

| Submodule | Purpose |
|-----------|---------|
| `analysis/` | Utilities such as `LogAnalyzer` for computing recall, accuracy and timing metrics from experiment CSV logs |
| `data_classes/` | Typed data containers for documents, QA pairs and RAG interfaces like `Indexer`, `Retriever` and `Generator` |
| `evaluation/` | Pipelines that index documents and evaluate retrieval/generation, writing structured logs for later analysis |
| `llm/` | Runners that wrap local and remote language models behind a common interface |
| `logging/` | `RunLogger` with nanosecond stopwatches and CSV tracking of metrics |
| `neo4j/` | Helpers to spin up a self-contained Neo4j environment for graph-based retrieval |
| `strategies/` | Pluggable strategies for chunking, encoding, NER, graph search, reranking and more |
| `templates/` | Prompt templates, e.g. for multiple-choice answering |

### `rag_approaches`

RAG systems composed from the building blocks above:

* **Document-Aware Hybrid RAG** – indexes documents into a Neo4j graph with knowledge triplets and uses personalized PageRank to retrieve supporting chunks.  
* **BM25** - sparse baseline that uses lexical matching
* **RAPTOR** - GraphRAG approach that uses Hierachical Tree Building
* **HippoRAG 2** - GraphRAG approach that uses Personalized Page Rank
* **NaiveVectorDBRAG** – lightweight in-memory vector database with cross-encoder reranking for retrieval.  
* **NoRAGGeneration** – baseline that bypasses retrieval and only calls the generator.  

### `experiments`

Entry points that orchestrate indexing and retrieval pipelines. 

The main file for running most experiments is `adjustable_experiment.py`, which provides a configurable harness to test various retrieval pipeline stages, chunking strategies, llms, or generation methods. 
Other scripts include `scaling_experiment.py` to evaluate performance at different corpus sizes, as well as test scripts for specific functionalities (like `llm_test_experiment.py` or `ui_showcase.py`).

### Data and utilities

The `data/` directory contains the processed HotpotQA_100 dataset (default test set), and `util/` houses scripts for converting and cleaning sources such as HotpotQA and NovelQA. For the full dataset either use the script to format it or contact me, I am happy to provide the formatted HotpotQA, MultiHopRAG and PubMedQA dataset

## Getting started

To reproduce the experiments, you must first provide the necessary external resources:

1.  **Neo4j & Java Archives:** The project embeds Neo4j for graph-based retrieval. You must place the following (or equivalent) downloaded tarballs/jars into the `common/neo4j/tar/` directory:
    *   `neo4j-community-2025.08.0-unix.tar.gz`
    *   `OpenJDK21U-jre_x64_linux_hotspot_21.0.8_9.tar.gz`
    *   `neo4j-graph-data-science-2.21.0.jar`
2.  **OpenRouter API Key:** If using OpenRouter for the LLMs, create a text file at `common/llm/openrouter.txt` and paste your API key inside it.

Install Python dependencies (Python 3.12.4 is recommended) and explore the experiments:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python "experiments/adjustable_experiment.py"
```

Each run writes detailed logs and metrics to `logs_and_tracks/`, which can be analyzed with the scripts in `common/analysis/`.

## Troubleshooting

### Neo4j Connection Issues
If Neo4j fails to start or connect (e.g., `Connection refused`), it could be due to a corrupted environment or lingering lock files. You can use the provided cleanup script to wipe the database space so it can be re-installed cleanly on the next run:

```bash
chmod +x common/neo4j/neo4j.sh
./common/neo4j/neo4j.sh
```

## Acknowledgements & Modified Dependencies

This thesis project builds upon and integrates several open-source technologies:

*   **Neo4j Community Edition:** Used extensively for graph-based retrieval. Neo4j Community Edition is licensed under the GPLv3 license and is free for academic and university projects.
*   **HippoRAG:** [https://github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) (MIT License). This repository contains a modified version of HippoRAG integrated into the experimental pipelines. The original MIT License is included in `rag_approaches/hippo_rag/HippoRAG/LICENSE`.
*   **RAPTOR:** [https://github.com/parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) (MIT License). This repository contains a modified version of RAPTOR for recursive abstractive processing. The original MIT License is included in `rag_approaches/raptor/raptor_src/LICENSE`.

Original copyrights for these dependencies remain with their respective authors. Modifications are provided under the existing license structure.
