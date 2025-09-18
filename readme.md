# SentimentSift-AI-Review-analysis-Platform

## Access the Services

- **Codelab:** [https://damg-7245.github.io/SentimentSift-AI-Powered-Review-Analysis-Platform/#0]
- **Streamlit UI:** [https://streamlit-ui-362196063135.us-central1.run.app/ ]
- **Demo:** https://youtu.be/3A0pTpScfq8

---

## Project Overview

This project implements an end-to-end data pipeline and AI system for analyzing Boston café data, providing rich insights through sentiment analysis, topic modeling, and AI-powered interfaces.

![Architecture](WorkFlow.jpg)
---

## Features

- Processes and analyzes café and review data
- Stores structured data in Snowflake for analytics
- Creates vector embeddings of reviews in Pinecone
- Provides AI-powered interfaces through RAG, chatbot, and SQL agents
- Enables users to explore data through natural language queries
- AI-generated review summaries


---

## Tech Stack

- Python 3.8+
- Docker and Docker Compose
- Airflow (ETL orchestration)
- Snowflake (cloud data warehouse)
- Pinecone (vector database)
- Streamlit (user interface)
- LangChain & OpenAI GPT-4 (RAG-powered chatbot integration)
- Amazon S3 (data storage)
- PyABSA (aspect-based sentiment analysis)
- BERTopic (topic modeling)
- Gemini API

---

## Getting Started

### Cloning the Repository

```bash
git clone https://github.com/DAMG-7245/SentimentSift-AI-Powered-Review-Analysis-Platform.git

```

### Creating and Configuring .env File

- Create `.env` files in the `airflow/tasks` and `api` directories.
- Add your API keys and database credentials:

  ```
  SNOWFLAKE_ACCOUNT=
  SNOWFLAKE_USER= 
  SNOWFLAKE_PASSWORD=
  SNOWFLAKE_ROLE=
  SNOWFLAKE_WAREHOUSE=
  SNOWFLAKE_DATABASE=
  SNOWFLAKE_SCHEMA=
  PINECONE_API_KEY=
  PINECONE_ENVIRONMENT=
  GEMINI_API_KEY=
  PINECONE_INDEX=
  ```

### Installing Dependencies

```bash
pip install -r requirements.txt
```


### Running the Application

```bash
docker-compose up -d
```

---

## Folder Structure

```
│   .gitignore
│   codelab.md
│   docker-compose.yaml
│   Dockerfile
│   readme.md
│   requirements.txt
│   WorkFlow.jpg
│   workflow.md
│
├───.vscode
│       launch.json
│
├───airflow
│   │   requirements.txt
│   │
│   ├───dags
│   │       cafe_pipeline_dag.py
│   │
│   └───tasks
│           .env
│           data_fetch.py
│           data_process.py
│           merge.py
│           sentiment_analysis.py
│           setup_snowflake.sql
│           snowflake_sync.py
│           theme_analysis.py
│           __init__.py
│
├───api
│       .env
│       chatbot_agent.py
│       main.py
│       py_setup.py
│       rag_agent.py
│       sql_agent.py
│       __init__.py
│
├───boston_cafes_data
│       boston_cafes.json
│       reviews.json
│
├───data
│   │   integrated_cafes.json
│   │
│   └───merge
├───docs
│   │   codelab.json
│   │   index.html
│   │
│   └───img
│           4f50ee79a0d3433f.jpg
│
├───frontend
│       chroma.sqlite3
│       package.json
│       test_app.py
│       vanna_helper.py
│
└───img
        review.jpg
        sentimentsift.png
```


---

## How It Works

The Boston Cafés Data & AI System integrates traditional ETL processes with modern AI capabilities, creating a comprehensive solution that:

1. Processes and analyzes café and review data
2. Stores structured data in Snowflake for analytics
3. Creates vector embeddings of reviews in Pinecone
4. Provides AI-powered interfaces through RAG, chatbot, and SQL agents

The system enables users to explore Boston café data through natural language queries, get AI-generated review summaries, and perform advanced analytics.

---

## Usage

### Running the Pipeline

1. Access the Airflow UI at `[ ]`
2. Trigger the `cafe_pipeline_dag` DAG to start the ETL process
3. Monitor the pipeline execution through the Airflow UI

### Using the AI Interfaces

#### Chatbot Interface



---


## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

---



## Authors

- Sicheng Bao (@Jellysillyfish13)
- Yung Rou Ko (@KoYungRou)
- Anuj Rajendraprasad Nene (@Neneanuj)

---

## Contact

For questions, reach out via Big Data Course or open an issue on GitHub.

---
