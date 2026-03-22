from typing import Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Shared state passed between all agent nodes.
    Each node reads from and writes to this state.
    """

    # user input
    user_query:          Optional[str]        

    # Node 1 outputs
    existing_articles:   Optional[list[str]]  # existing article titles scraped from sest.gmbh/news/
    rag_context_chunks:  Optional[list[dict]] # top 10 chunks retrieved from Qdrant
    suggested_topic:     Optional[str]        # article topic suggested by LLM
    suggested_keywords:  Optional[list[str]]  # 10 SEO keywords (internal + web combined)
    suggested_category:  Optional[str]        # Strategie / Training / Technologie

    # Human Checkpoint 1 outputs
    approved_topic:      Optional[str]        # topic approved by human
    approved_keywords:   Optional[list[str]]  # keywords approved by human
    approved_category:   Optional[str]        # category approved by human

    # Node 2 outputs
    retrieved_chunks:    Optional[list[dict]] # filtered chunks passed to writer
    scored_chunks:       Optional[list[dict]] # chunks with scores for UI display

    # Human Checkpoint 2 outputs
    tone:                Optional[str]        # Professionell / Edukativ / Praktisch
    inhaltstyp:          Optional[str]        # How-To / Framework / Story / Facts & Data / Case Study

    # Node 3 outputs
    article_title:       Optional[str]        # final article title
    article_outline:     Optional[str]        # article section outline
    article_draft:       Optional[str]        # full first draft in German
    final_keywords:      Optional[list[str]]  # final SEO keywords used in article
    category_tag:        Optional[str]        # final category tag

    # Human Checkpoint 1 re-run support
    user_preferred_category: Optional[str]   # category preference if human rejects Node 1
    user_preferred_topic:    Optional[str]   # topic preference if human rejects Node 1