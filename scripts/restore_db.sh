#!/bin/bash
set -e

# Directory where backups are stored
BACKUP_DIR="/backups"

# Tag for the backup to restore, passed as an argument
TAG=$1

# Full path to the backup directory
BACKUP_PATH="${BACKUP_DIR}/${TAG}"

# Stop Neo4j database service within the container
docker exec genai-stack-database-1 neo4j stop

# Restore the database using load
docker exec genai-stack-database-1 neo4j-admin database load --from-path="${BACKUP_PATH}" neo4j --overwrite-destination=true

# Start Neo4j database service within the container
docker exec genai-stack-database-1 neo4j start

echo "Restore was successful from ${BACKUP_PATH}"
