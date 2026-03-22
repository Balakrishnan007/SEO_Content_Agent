import os
import re
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from tools import vector_search_text, vector_search_filtered, vector_search_for_ui, web_scraper, web_search
from prompts import (
    NODE1_SYSTEM_PROMPT, NODE1_USER_PROMPT,
    NODE2_SYSTEM_PROMPT, NODE2_USER_PROMPT,
    NODE3_SYSTEM_PROMPT, NODE3_USER_PROMPT,
)

load_dotenv(find_dotenv())

# LLM for Node 1 and Node 2 — topic selection and retrieval
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=2000,
)

# LLM for Node 3 — higher token limit for full article generation
llm_writer = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=6000,
)


def node1_topic_agent(state: AgentState) -> dict:
    """
    Node 1 — Topic Agent.

    1. Searches Qdrant with user query (top_k=10)
    2. Scrapes sest.gmbh/news/ to get existing article titles
    3. Extracts top 5 keywords from retrieved chunks
    4. Searches Tavily with those keywords for current web trends
    5. LLM combines everything to suggest topic, category and 10 SEO keywords
    """
    print("\nNode 1 — Topic Agent")

    user_query = state.get("user_query", "")

    # search internal knowledge base with user query
    rag_chunks = vector_search_text(query=user_query, top_k=10)
    print(f"  Qdrant: {len(rag_chunks)} characters retrieved")

    # scrape existing articles to avoid duplicate topics
    existing_articles = web_scraper("https://sest.gmbh/news/")
    if len(existing_articles) < 50:
        existing_articles = "Keine bestehenden Artikel gefunden."

    # extract top 5 keywords from internal chunks
    keyword_prompt = f"""Lies diese internen Dokumente und extrahiere die TOP 5 
relevantesten SEO Keywords auf Deutsch.

Dokumente:
{rag_chunks}

Antworte NUR mit einem JSON-Array: ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
Kein Text davor oder danach."""

    keyword_response = llm.invoke(keyword_prompt)
    raw_keywords     = keyword_response.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        match          = re.search(r'\[.*\]', raw_keywords, re.DOTALL)
        top_5_keywords = json.loads(match.group(0)) if match else []
        if len(top_5_keywords) < 1:
            raise ValueError("No keywords extracted")
    except Exception:
        top_5_keywords = [user_query]

    print(f"  Keywords from docs: {top_5_keywords}")

    # Search internet with extracted keywords for current trends
    keyword_string = " ".join(top_5_keywords)
    web_results    = web_search(keyword_string)
    print(f"  Tavily query: {keyword_string}")

    # inject user category/topic preference if human rejected previous suggestion
    preferred_category = state.get("user_preferred_category", "")
    preferred_topic    = state.get("user_preferred_topic", "")

    user_preference = ""
    if preferred_category or preferred_topic:
        lines = ["\n\nVORGABE DES NUTZERS (zwingend einhalten):"]
        if preferred_category:
            lines.append(f"- Kategorie: {preferred_category}")
        if preferred_topic:
            lines.append(f"- Thema: {preferred_topic}")
        lines.append("Schlage NUR ein Thema vor, das dieser Vorgabe entspricht.")
        user_preference = "\n".join(lines)

    user_prompt = NODE1_USER_PROMPT.format(
        user_query=user_query,
        rag_chunks=rag_chunks,
        existing_articles=existing_articles,
        web_results=web_results,
    ) + user_preference

    response   = llm.invoke([SystemMessage(content=NODE1_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    raw_output = response.content.strip()

    try:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON found", raw_output, 0)
        result = json.loads(match.group(0))

        suggested_topic    = result.get("vorgeschlagenes_thema", "")
        suggested_keywords = result.get("seo_keywords", [])
        suggested_category = result.get("kategorie", "")

        print(f"  Topic    : {suggested_topic}")
        print(f"  Category : {suggested_category}")

    except json.JSONDecodeError:
        suggested_topic    = raw_output[:200]
        suggested_keywords = []
        suggested_category = "Unbekannt"

    return {
        "rag_context_chunks": [{"text": rag_chunks}],
        "existing_articles":  [existing_articles],
        "suggested_topic":    suggested_topic,
        "suggested_keywords": suggested_keywords,
        "suggested_category": suggested_category,
    }


def node2_rag_retriever(state: AgentState) -> dict:
    """
    Node 2 — RAG Retriever Agent.

    1. Searches Qdrant for transcripts (top_k=5)
    2. Searches Qdrant for product docs (top_k=5)
    3. LLM filters and ranks by relevance keeping 50/50 balance
    4. Also retrieves scored chunks for Streamlit UI display
    """
    print("\nNode 2 — RAG Retriever")

    approved_topic    = state.get("approved_topic", "")
    approved_keywords = state.get("approved_keywords", [])
    search_query      = f"{approved_topic} {' '.join(approved_keywords)}".strip()

    # retrieve transcripts and product docs separately for 50/50 balance
    transcript_chunks = vector_search_filtered(query=search_query, top_k=5, source_type="transcript")
    product_chunks    = vector_search_filtered(query=search_query, top_k=5, source_type="product_doc")
    combined_chunks   = transcript_chunks + "\n\n---\n\n" + product_chunks

    print(f"  Transcripts  : {len(transcript_chunks)} characters")
    print(f"  Product docs : {len(product_chunks)} characters")

    # LLM filters chunks for relevance
    user_prompt = NODE2_USER_PROMPT.format(
        approved_topic=approved_topic,
        approved_keywords=", ".join(approved_keywords),
        retrieved_chunks=combined_chunks,
    )

    response        = llm.invoke([SystemMessage(content=NODE2_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    filtered_chunks = response.content.strip()

    print(f"  Filtered     : {len(filtered_chunks)} characters passed to writer")

    # retrieve scored chunks for UI display
    scored_chunks = (
        vector_search_for_ui(search_query, top_k=5, source_type="transcript") +
        vector_search_for_ui(search_query, top_k=5, source_type="product_doc")
    )

    return {
        "retrieved_chunks": [{"text": filtered_chunks}],
        "scored_chunks":    scored_chunks,
    }


def node3_writer_agent(state: AgentState) -> dict:
    """
    Node 3 — Writer Agent.

    Writes a full German SEO article (1000+ words) based on
    approved topic, keywords, category, content type and tone.
    Returns title, outline, draft, final keywords and category tag.
    """
    print("\nNode 3 — Writer Agent")

    approved_topic    = state.get("approved_topic", "")
    approved_keywords = state.get("approved_keywords", [])
    approved_category = state.get("approved_category", "")
    tone              = state.get("tone", "Professionell")
    inhaltstyp        = state.get("inhaltstyp", "How-To")

    # flatten retrieved chunks into single text block
    chunks_list    = state.get("retrieved_chunks", [])
    retrieved_text = "\n\n---\n\n".join(
        c.get("text", "") for c in chunks_list if c.get("text")
    )

    print(f"  Topic      : {approved_topic}")
    print(f"  Category   : {approved_category}")
    print(f"  Type       : {inhaltstyp}")
    print(f"  Tone       : {tone}")

    system_prompt = NODE3_SYSTEM_PROMPT.format(ton=tone, inhaltstyp=inhaltstyp)

    user_prompt = NODE3_USER_PROMPT.format(
        approved_topic=approved_topic,
        approved_keywords=", ".join(approved_keywords),
        approved_category=approved_category,
        retrieved_chunks=retrieved_text,
        inhaltstyp=inhaltstyp,
        tone=tone,
    )

    response   = llm_writer.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    raw_output = response.content.strip()

    try:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON found", raw_output, 0)
        result = json.loads(match.group(0))

        article_title   = result.get("artikel_titel", "")
        article_outline = result.get("inhalt", [])
        article_draft   = result.get("artikel_entwurf", "")
        final_keywords  = result.get("finale_keywords", [])
        category_tag    = result.get("kategorie_tag", approved_category)

        print(f"  Article written: {len(article_draft)} characters")
        print(f"  Title: {article_title}")

    except json.JSONDecodeError:
        article_title   = approved_topic
        article_outline = []
        article_draft   = raw_output
        final_keywords  = approved_keywords
        category_tag    = approved_category

    return {
        "article_title":   article_title,
        "article_outline": article_outline,
        "article_draft":   article_draft,
        "final_keywords":  final_keywords,
        "category_tag":    category_tag,
    }