# DATASTRUCTURE.md

## Overview
This document describes the graph schema used by the GraphRAG system in Neo4j.  
We represent documents, their chunks, extracted entities, knowledge assertions, and derived centroids as nodes, linked by relationships.

---

## Node Types

### `Document`
Represents an ingested document.  
**Properties:**
- `id: STRING (unique)` – identifier for the document  
- `title: STRING?` – optional title  
- `author: STRING?` – optional author  
- `source: STRING?` – optional source identifier (e.g. URL, filename)  
- `publication_date: DATE?` – optional publication date  
- `references: LIST<STRING>` – optional list of references  

**Indexes/Constraints:**
- Unique constraint on `id`  
- Full-text index on `title`, `source`, `author`

---

### `Chunk`
Represents a passage or segment of text from a document.  
**Properties:**
- `chunk_name: STRING (unique)` – identifier for the chunk  
- `text: STRING` – chunk text  
- `embedding: LIST<FLOAT>` – vector embedding for retrieval  

**Indexes/Constraints:**
- Unique constraint on `chunk_name`  
- Vector index on `embedding`  

**Relationships:**
- `(:Document)-[:HAS_CHUNK]->(:Chunk)`

---

### `Entity`
Represents an entity mentioned in a document.  
**Properties:**
- `name: STRING (unique)` – canonical entity name  
- `type: STRING?` – optional type/class of the entity  
- `aliases: LIST<STRING>` – list of aliases  

**Indexes/Constraints:**
- Unique constraint on `name`  
- Full-text index on `name`, `aliases`

---

### `Assertion`
Represents a knowledge assertion (triplet).  
**Properties:**
- `id: STRING (unique)` – hash of subject|predicate|object  
- `predicate: STRING` – relation/predicate  
- `subject: STRING` – subject text  
- `object: STRING` – object text  
- `embedding: LIST<FLOAT>` – embedding of the assertion text  
- `rank: INT` – accumulated rank/weight  

**Indexes/Constraints:**
- Unique constraint on `id`  
- Vector index on `embedding`  
- Full-text index on `predicate`  

**Relationships:**
- `(:Assertion)-[:SUBJECT]->(:Entity)`  
- `(:Assertion)-[:OBJECT]->(:Entity)`  
- `(:Assertion)-[:DERIVED_FROM]->(:Chunk)`

---

### `DocCentroid`
Represents a centroid embedding computed from a document’s chunk embeddings via k-means clustering.  

**Properties:**
- `id: STRING (unique)` – identifier (`<doc_id>_c<i>`)  
- `embedding: LIST<FLOAT>` – centroid vector  

**Indexes/Constraints:**
- Unique constraint on `id`  
- Vector index on `embedding`  

**Relationships:**
- `(:Document)-[:HAS_CENTROID]->(:DocCentroid)`

---

## Relationship Summary

- `(:Document)-[:HAS_CHUNK]->(:Chunk)`  
- `(:Document)-[:HAS_CENTROID]->(:DocCentroid)`  
- `(:Assertion)-[:SUBJECT]->(:Entity)`  
- `(:Assertion)-[:OBJECT]->(:Entity)`  
- `(:Assertion)-[:DERIVED_FROM]->(:Chunk)`

---

This schema ensures:
- **Documents** organize **chunks**, centroids, and metadata  
- **Chunks** carry embeddings for retrieval  
- **Entities** unify mentions with aliases  
- **Assertions** link entities and provenance  
- **DocCentroids** represent document-level embeddings for fast similarity search  
