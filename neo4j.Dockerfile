# Use the official Neo4j image as a base
FROM neo4j:5.11

# Set environment variables if needed
# ENV NEO4J_AUTH=neo4j/test

# Copy custom scripts or configurations
COPY ./scripts/custom-neo4j.sh /custom-neo4j.sh

# Make sure the script is executable
RUN chmod +x /custom-neo4j.sh

# Set the default command to execute
CMD ["/custom-neo4j.sh"]

