#!/bin/bash
set -e

# Directory to store and retrieve backups
BACKUP_DIR="/backups"

# Tag for the backup, passed as an argument
TAG=$1

# Full path for the backup directory
BACKUP_PATH="${BACKUP_DIR}/${TAG}"

# # ====backup Neo4j Graph database
# Ensure the backup directory exists
docker exec genai-stack-database-1 mkdir -p "${BACKUP_PATH}"

# Stop Neo4j database service within the container
docker exec genai-stack-database-1 neo4j stop

# Running the backup command using dump
docker exec genai-stack-database-1 neo4j-admin database dump --to-path="${BACKUP_PATH}" neo4j

# Restart Neo4j database service within the container
docker exec genai-stack-database-1 neo4j start
echo "Backup Neo4j DB was successful, stored at ${BACKUP_PATH}"

# ====backup Chroma vector database
CHROMA_HOST_BACKUP="$PWD/backups_chroma/${TAG}"
if [ ! -d "$CHROMA_HOST_BACKUP" ]; then
    # Stop the container
    docker stop genai-stack-chroma_server-1

    sudo mkdir -p "${CHROMA_HOST_BACKUP}"
    # Create a backup of the container's filesystem:
    sudo docker cp genai-stack-chroma_server-1:/data "${CHROMA_HOST_BACKUP}"

    # Restart chroma database container
    docker start genai-stack-chroma_server-1

    echo "Backup Chroma DB was successful, stored at ${CHROMA_HOST_BACKUP}"
fi

