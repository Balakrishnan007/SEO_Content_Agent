# SEO Content Agent

Transforms internal knowledge from meeting transcripts and product documents into SESTdigital's SEO-optimized German website articles.

---

## How it works

```
User enters a relevant query
     ↓
Node 1 Agent:
  1. Searches Qdrant with user query (top_k=10)
  2. Scrapes existing articles on sest.gmbh/news/ (inorder to avoid the repition of the same topics)
  3. Extracts top 5 keywords from retrieved chunks
  4. Searches internet for current trends (Tavily)
  5. LLM suggests topic, category and 10 SEO keywords
     ↓
Human Checkpoint 1 - Either Approve or pick one of the three category(strategy, training, technology) and it runs the Node 1 Agent again
     ↓
Node 2 Agent:
  1. Combines approved topic + approved keywords as a single query
  2. Searches Qdrant for transcripts (top_k=5)
  3. Searches Qdrant for product docs (top_k=5)
  4. LLM acts as a reranker which reads all 10 chunks and keeps only the ones truly relevant to the approved topic (50/50 balance maintained)
     ↓
Human Checkpoint 2 - Choose content type and article tone which will decide how the article should sound
     ↓
Node 3:
  Receives approved topic, keywords, category, content type, tone and filtered chunks from Node 2.
  Writes a full German SEO article (1000+ words)
  Returns title, outline, keywords and category tag
```

---

## Tech Stack

- **LLM** - GPT-4o
- **Embeddings** - Jina v2 base de (German-optimized, runs locally)
- **Vector DB** - Qdrant
- **Web search** - Tavily
- **UI** - Streamlit

---

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Create a `.env` file**
```
OPENAI_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
TAVILY_API_KEY=
QDRANT_COLLECTION_NAME=sest_knowledge_base
```

**3. Run ingestion (for initial setup of vector databse)**
```bash
cd src
python ingestion.py
```

---

## Run

**Streamlit UI**
```bash
cd src
streamlit run app.py
```

**CLI**
```bash
cd src
python main.py
```