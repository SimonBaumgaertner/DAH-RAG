#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "🗑️  Clearing Neo4j binaries and data folders..."
rm -rf "$PROJECT_ROOT/databases" \
       "$PROJECT_ROOT/transactions" \
       "$PROJECT_ROOT/common/neo4j/neo4j_env"

echo "✅ Done. You can now run your Python script."
