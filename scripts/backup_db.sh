#!/bin/bash
set -e

# Directory to store and retrieve backups
BACKUP_DIR="/backups"

# Tag for the backup, passed as an argument
TAG=$1

# Full path for the backup directory
BACKUP_PATH="${BACKUP_DIR}/${TAG}"

# Ensure the backup directory exists
docker exec genai-stack-database-1 mkdir -p "${BACKUP_PATH}"

# Stop Neo4j database service within the container
docker exec genai-stack-database-1 neo4j stop

# Running the backup command using dump
docker exec genai-stack-database-1 neo4j-admin database dump --to-path="${BACKUP_PATH}" neo4j

# Restart Neo4j database service within the container
docker exec genai-stack-database-1 neo4j start

echo "Backup was successful, stored at ${BACKUP_PATH}"
