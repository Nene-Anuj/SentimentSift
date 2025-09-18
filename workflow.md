---
config:
  theme: redux
  look: neo
---
```mermaid
flowchart TD
    subgraph Data_Ingestion [Data Ingestion]
        A1[Google API] --> B1[Collect Reviews and Ratings]
        A2[Yelp API] --> B1
        A3[Twitter API] --> B1
        B1 --> C1[Store Raw JSON Files in S3]
    end
    subgraph Data_Storage [Data Storage]
        C1 --> D1[Preprocess Data in Snowflake]
        D1 --> E1[Validate Data with dbt]
    end
    subgraph Processing [Processing & Transformation]
        E1 --> F1[Aspect-Based Sentiment Analysis using PyABSA]
        F1 --> G1[Generate Sentiment Scores for Food, Service, Ambiance]
        G1 --> H1[Theme Extraction using BERTopic]
        H1 --> I1[Store Embeddings in Pinecone]
    end
    subgraph Streamlit_UI [Streamlit UI & Querying]
        subgraph Chatbot_Use_Case [Chatbot Integration]
            J1[RAG Pipeline Setup] --> K1[Retrieve Relevant Information from Pinecone]
            K1 --> LLM_Fine_Tuning[Fine-Tune LLM for Context-Aware Responses]
            LLM_Fine_Tuning --> M2[Chatbot for Advanced Queries]
            M2 --> Q1["What are the top-rated Italian restaurants in Boston?"]
            M2 --> Q2["Which restaurants are open late with outdoor seating?"]
            M2 --> Q3["What are common complaints about service in low-rated restaurants?"]
        end
        subgraph Market_Research [Market Research Insights]
            I1 --> MR_Insights["Analyze dining trends by cuisine or amenities"]
            I1 --> MR_Trends["Identify emerging customer preferences"]
            MR_Trends --> MR_Report["Generate tiered summary reports for restaurant owners"]
        end
        subgraph Customer_Use_Case [Customer Insights]
            I1 --> CU_Ratings["Access unbiased, normalized ratings"]
            CU_Ratings --> CU_Trends["Query restaurant trends like outdoor seating or vegan options"]
            CU_Trends --> CU_Menu["Explore menu recommendations based on preferences"]
        end
        subgraph Restaurant_Owner_Use_Case [Restaurant Owner Insights]
            I1 --> RO_Strengths["Identify strengths/weaknesses grouped by tiers"]
            RO_Strengths --> RO_Feedback["Analyze recurring themes from customer feedback"]
            RO_Feedback --> RO_Decisions["Make data-driven decisions to improve operations"]
        end
    end
    Streamlit_UI --> AA[Final Testing and Deployment]
    classDef primary fill:#FFDDC1,stroke:#000,stroke-width:2px;  %% Pastel Peach
    classDef secondary fill:#D1FFBD,stroke
