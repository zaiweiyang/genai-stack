#!/bin/bash
set -e

# Directory where backups are stored
BACKUP_DIR="/backups"

# Tag for the backup to restore, passed as an argument
TAG=$1
# ====Restore Neo4j Graph database
# Full path to the backup directory
BACKUP_PATH="${BACKUP_DIR}/${TAG}"

# Stop Neo4j database service within the container
docker exec genai-stack-database-1 neo4j stop

# Restore the database using load
docker exec genai-stack-database-1 neo4j-admin database load --from-path="${BACKUP_PATH}" neo4j --overwrite-destination=true

# Start Neo4j database service within the container
docker exec genai-stack-database-1 neo4j start

echo "Restore Neo4j db was successful from ${BACKUP_PATH}"

# # ====Restore Chroma vector database
CHROMA_HOST_BACKUP="$PWD/backups_chroma/${TAG}"
# Check if the backup directory exists, create if it does not
if [ -d "$CHROMA_HOST_BACKUP" ]; then
  docker stop genai-stack-chroma_server-1

  # Copy the contents inside the host's data directory to the container's /data directory
  find "${CHROMA_HOST_BACKUP}/data" -mindepth 1 -maxdepth 1 -exec sudo docker cp {} genai-stack-chroma_server-1:/data/ \;

# Restart chroma database container
  docker start genai-stack-chroma_server-1

  echo "Restore Chroma DB was successful, stored at ${CHROMA_HOST_BACKUP}"
fi

