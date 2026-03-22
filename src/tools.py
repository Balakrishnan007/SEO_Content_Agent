import os
import requests
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Qdrant and embedding config
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "sest_knowledge_base")
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-de"

# lazy-loaded singletons to avoid reloading on every call
_qdrant_client   = None
_embedding_model = None


def get_qdrant_client() -> QdrantClient:
    """Returns a shared Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def get_embedding_model() -> SentenceTransformer:
    """Returns a shared Jina embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _embedding_model


def web_scraper(url: str) -> str:
    """
    Scrapes article titles from a webpage.
    Used in Node 1 to check existing articles on sest.gmbh/news/
    so the agent does not suggest duplicate topics.
    """
    try:
        headers  = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup.find_all(["nav", "footer", "header"]):
            tag.decompose()

        titles = [
            tag.get_text(strip=True)
            for tag in soup.find_all(["h1", "h2", "h3"])
            if tag.get_text(strip=True) and len(tag.get_text(strip=True)) > 20
        ]

        if not titles:
            return "Keine bestehenden Artikel auf dieser Seite gefunden."

        return "Bereits veröffentlichte Artikel auf SESTdigital:\n" + "\n".join(
            f"- {t}" for t in titles
        )

    except Exception as e:
        return f"Fehler beim Scraping von {url}: {str(e)}"


def web_search(query: str) -> str:
    """
    Searches the internet for trending content using Tavily.
    Used in Node 1 to enrich SEO keywords with current web trends.
    """
    try:
        tavily   = TavilySearch(max_results=5, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily.invoke(query)

        if isinstance(response, dict):
            results = response.get("results", [])
        elif isinstance(response, list):
            results = response
        else:
            return str(response)

        output = []
        for r in results:
            title   = r.get("title", "Kein Titel")
            snippet = r.get("content", "")[:300]
            url     = r.get("url", "")
            output.append(f"Titel: {title}\nAuszug: {snippet}\nURL: {url}")

        return "\n---\n".join(output) if output else "Keine Ergebnisse gefunden."

    except Exception as e:
        return f"Fehler bei der Websuche: {str(e)}"


def vector_search_text(query: str, top_k: int = 8) -> str:
    """
    Searches the Qdrant knowledge base and returns formatted text for LLM consumption.
    Used in Node 1 for broad topic discovery and keyword extraction.
    """
    try:
        client       = get_qdrant_client()
        model        = get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points

        if not results:
            return "Keine relevanten Inhalte in der Wissensdatenbank gefunden."

        output = [
            f"[{r.payload.get('source_type')} | {r.payload.get('source_file')} | Score: {round(r.score, 4)}]\n{r.payload.get('text', '')}"
            for r in results
        ]

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"Fehler bei der Vektorsuche: {str(e)}"


def vector_search_filtered(query: str, top_k: int = 5, source_type: str = None) -> str:
    """
    Searches the Qdrant knowledge base with optional source type filter.
    Used in Node 2 to retrieve targeted chunks for article writing.
    Fetches top_k * 3 results first, then filters to ensure enough matches.
    """
    try:
        client       = get_qdrant_client()
        model        = get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        fetch_limit = top_k * 3 if source_type else top_k

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=fetch_limit,
        ).points

        if not results:
            return "Keine relevanten Chunks für dieses Thema gefunden."

        if source_type:
            results = [r for r in results if r.payload.get("source_type") == source_type]
            results = results[:top_k]

        if not results:
            return f"Keine {source_type}-Chunks für dieses Thema gefunden."

        output = [
            f"[{r.payload.get('source_type')} | {r.payload.get('source_file')} | Score: {round(r.score, 4)}]\n{r.payload.get('text', '')}"
            for r in results
        ]

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"Fehler bei der Vektorsuche: {str(e)}"


def vector_search_for_ui(query: str, top_k: int = 5, source_type: str = None) -> list:
    """
    Same as vector_search_raw but returns a structured list of dicts.
    Used in Node 2 to display chunk scores and sources in the Streamlit UI.
    """
    try:
        client       = get_qdrant_client()
        model        = get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        fetch_limit = top_k * 3 if source_type else top_k

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=fetch_limit,
        ).points

        if not results:
            return []

        if source_type:
            results = [r for r in results if r.payload.get("source_type") == source_type]
            results = results[:top_k]

        return [
            {
                "score":       round(r.score, 4),
                "source_type": r.payload.get("source_type", ""),
                "source_file": r.payload.get("source_file", ""),
                "text":        r.payload.get("text", ""),
            }
            for r in results
        ]

    except Exception as e:
        return []