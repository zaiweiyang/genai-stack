FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt
RUN pip install arize-phoenix[evals]
# RUN mkdir -p /app/data
# RUN mkdir -p /app/data_backups
# Install the required dependencies
RUN pip install --no-cache-dir python-docx python-pptx \
    grpcio \
    grpcio-tools \
    google-api-core \
    google-auth \
    google-cloud-core  \
    google-ai-generativelanguage

# Generate gRPC code
COPY grpc-svr/file_service.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. file_service.proto


COPY library_bot.py .
COPY utils.py .
COPY chains.py .
COPY config/config.json .
EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "library_bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
