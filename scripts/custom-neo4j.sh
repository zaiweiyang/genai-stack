#!/bin/bash
# Custom script to manage Neo4j start-up and ensure the container does not exit after Neo4j stops

# Start Neo4j in the background
neo4j start

# Wait indefinitely to keep the container running
tail -f /dev/null
