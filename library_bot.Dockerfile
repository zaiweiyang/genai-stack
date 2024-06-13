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

COPY pdf_bot_chroma.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "pdf_bot_chroma.py", "--server.port=8503", "--server.address=0.0.0.0"]
