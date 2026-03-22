import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from agents import node1_topic_agent, node2_rag_retriever, node3_writer_agent

st.set_page_config(
    page_title="SESTdigital SEO Content Agent",
    page_icon="🤖",
    layout="wide",
)

# SESTdigital brand colors: black background, teal #00c4a0 accent
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .sest-header {
        background-color: #000000;
        padding: 24px 32px;
        margin-bottom: 8px;
    }
    .sest-logo {
        font-size: 24px;
        color: #ffffff;
    }
    .sest-logo .bold { font-weight: 900; }
    .sest-logo .light { font-weight: 300; }
    .sest-tagline { font-size: 13px; color: #aaaaaa; margin: 4px 0 0 0; }

    .stage-header {
        border-left: 5px solid #00c4a0;
        background-color: #f8f8f8;
        padding: 12px 20px;
        margin: 24px 0 16px 0;
        font-weight: 700;
        font-size: 15px;
    }
    .info-card {
        border: 1px solid #e0e0e0;
        border-top: 3px solid #00c4a0;
        border-radius: 4px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .kw-tag {
        display: inline-block;
        background-color: #e6faf7;
        color: #007a63;
        border: 1px solid #00c4a0;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px 3px 3px 0;
        font-size: 12px;
    }
    .cat-badge {
        display: inline-block;
        background-color: #000000;
        color: #00c4a0;
        padding: 3px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .article-box {
        border: 1px solid #e0e0e0;
        border-left: 4px solid #00c4a0;
        border-radius: 4px;
        padding: 24px 28px;
        font-size: 15px;
        line-height: 1.8;
        white-space: pre-wrap;
        margin-top: 8px;
    }
    .sest-divider { border-top: 1px solid #e8e8e8; margin: 24px 0; }

    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stDownloadButton"] > button[kind="primary"] {
        background-color: #00c4a0 !important;
        border-color: #00c4a0 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 2px solid #000000 !important;
        color: #000000 !important;
        border-radius: 30px !important;
    }
</style>
""", unsafe_allow_html=True)


# session state defaults
defaults = {
    "stage": "start",
    "user_query": "",
    "suggested_topic": "", "suggested_category": "", "suggested_keywords": [],
    "approved_topic": "", "approved_category": "", "approved_keywords": [],
    "retrieved_chunks": [],
    "scored_chunks":    [],
    "article_title": "", "article_outline": [], "article_draft": "",
    "final_keywords": [], "category_tag": "",
    "user_preferred_category": "", "user_preferred_topic": "",
    "inhaltstyp": "How-To", "tone": "Professionell",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def kw_tags_html(keywords):
    """Renders keywords as styled HTML tags."""
    return " ".join(f'<span class="kw-tag">{kw}</span>' for kw in keywords)


def build_output_text():
    """Builds the plain text content for article download."""
    lines = [
        f"TITEL: {st.session_state.article_title}",
        f"KATEGORIE: {st.session_state.category_tag}",
        f"KEYWORDS: {', '.join(st.session_state.final_keywords)}",
        "\nGLIEDERUNG:",
    ]
    for item in st.session_state.article_outline:
        lines.append(f"  {item}")
    lines += ["\nARTIKEL:\n", st.session_state.article_draft]
    return "\n".join(lines)


# header
st.markdown("""
<div class="sest-header">
    <div class="sest-logo"><span class="bold">SEST</span><span class="light">digital</span> SEO Content Agent</div>
    <p class="sest-tagline">Verwandelt internes Wissen aus Transkripten und Produktdokumenten in SEO-optimierte Website-Artikel.</p>
</div>
""", unsafe_allow_html=True)


# Step 1 — Topic suggestion
st.markdown('<div class="stage-header">Schritt 1 — Themenvorschlag</div>', unsafe_allow_html=True)

if st.session_state.stage == "start":
    st.markdown("Der Agent durchsucht die interne Wissensdatenbank, prüft bestehende Artikel und schlägt autonom ein Thema, eine Kategorie und SEO-Keywords vor.")

    user_query_input = st.text_input(
        "Worüber soll der Artikel handeln?",
        placeholder="z.B. KI Training für Mitarbeiter oder KI Automatisierung",
        key="user_query_input",
    )

    if st.button("Agent starten", type="primary", use_container_width=True):
        if not user_query_input.strip():
            st.warning("Bitte gib zuerst ein Thema ein.")
        else:
            st.session_state.user_query = user_query_input.strip()
            with st.spinner("Node 1 läuft - Wissensdatenbank wird durchsucht..."):
                result = node1_topic_agent(st.session_state)
                st.session_state.suggested_topic    = result["suggested_topic"]
                st.session_state.suggested_category = result["suggested_category"]
                st.session_state.suggested_keywords = result["suggested_keywords"]
                st.session_state.stage              = "node1_done"
            st.rerun()

if st.session_state.stage in ["node1_done", "node2_done", "done"]:
    st.markdown(f"""
    <div class="info-card">
        <p style="margin:0 0 10px 0;"><strong>Thema:</strong> {st.session_state.suggested_topic}</p>
        <p style="margin:0 0 10px 0;"><strong>Kategorie:</strong> <span class="cat-badge">{st.session_state.suggested_category}</span></p>
        <p style="margin:0 0 8px 0;"><strong>SEO-Keywords:</strong></p>
        {kw_tags_html(st.session_state.suggested_keywords)}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.stage == "node1_done":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Genehmigen & weiter", type="primary", use_container_width=True):
                st.session_state.approved_topic    = st.session_state.suggested_topic
                st.session_state.approved_category = st.session_state.suggested_category
                st.session_state.approved_keywords = st.session_state.suggested_keywords
                with st.spinner("Node 2 läuft - relevante Inhalte werden abgerufen..."):
                    result = node2_rag_retriever(st.session_state)
                    st.session_state.retrieved_chunks = result["retrieved_chunks"]
                    st.session_state.scored_chunks    = result.get("scored_chunks", [])
                    st.session_state.stage            = "node2_done"
                st.rerun()
        with col2:
            with st.expander("Ablehnen — andere Kategorie wählen"):
                cats    = ["Strategie", "Training", "Technologie"]
                idx     = cats.index(st.session_state.suggested_category) if st.session_state.suggested_category in cats else 0
                new_cat = st.selectbox("Kategorie:", cats, index=idx, key="reject_cat")
                if st.button("Node 1 wiederholen", use_container_width=True):
                    st.session_state.user_preferred_category = new_cat
                    st.session_state.user_preferred_topic    = ""
                    with st.spinner(f"Node 1 läuft erneut mit Kategorie: {new_cat}..."):
                        result = node1_topic_agent(st.session_state)
                        st.session_state.suggested_topic    = result["suggested_topic"]
                        st.session_state.suggested_category = result["suggested_category"]
                        st.session_state.suggested_keywords = result["suggested_keywords"]
                    st.rerun()

st.markdown('<hr class="sest-divider">', unsafe_allow_html=True)


# Step 2 — Content retrieval and style selection
if st.session_state.stage in ["node2_done", "done"]:
    st.markdown('<div class="stage-header">Schritt 2 — Inhalte abgerufen & Schreibstil wählen</div>', unsafe_allow_html=True)

    # show retrieved chunks with scores
    scored = st.session_state.get("scored_chunks", [])
    if scored:
        total      = len(scored)
        transcript = sum(1 for c in scored if c["source_type"] == "transcript")
        product    = sum(1 for c in scored if c["source_type"] == "product_doc")
        with st.expander(f"Abgerufene Chunks: {total} gesamt ({transcript} Transkripte + {product} Produktdokumente)"):
            for i, c in enumerate(scored, 1):
                label = "Transkript" if c["source_type"] == "transcript" else "Produktdokument"
                st.markdown(f"**{i}. {label}** — `{c['source_file']}` — Score: **{c['score']}**")
                st.caption(c["text"][:200] + "...")
                if i < len(scored):
                    st.markdown("---")

    if st.session_state.stage == "node2_done":
        col1, col2 = st.columns(2)
        with col1:
            inhaltstyp = st.selectbox("Inhaltstyp", ["How-To", "Framework", "Story", "Facts & Data", "Case Study"])
        with col2:
            tone = st.selectbox("Schreibstil", ["Professionell", "Edukativ", "Praktisch"])

        if st.button("Artikel schreiben", type="primary", use_container_width=True):
            st.session_state.inhaltstyp = inhaltstyp
            st.session_state.tone       = tone
            with st.spinner("Node 3 läuft - SEO-Artikel wird geschrieben (1000+ Wörter)..."):
                result = node3_writer_agent(st.session_state)
                st.session_state.article_title   = result["article_title"]
                st.session_state.article_outline = result["article_outline"]
                st.session_state.article_draft   = result["article_draft"]
                st.session_state.final_keywords  = result["final_keywords"]
                st.session_state.category_tag    = result["category_tag"]
                st.session_state.stage           = "done"
            st.rerun()

    st.markdown('<hr class="sest-divider">', unsafe_allow_html=True)


# Step 3 — Final article display
if st.session_state.stage == "done":
    st.markdown('<div class="stage-header">Schritt 3 — Fertiger SEO-Artikel</div>', unsafe_allow_html=True)

    st.markdown(f"### {st.session_state.article_title}")
    st.markdown(f'<span class="cat-badge">{st.session_state.category_tag}</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Finale SEO-Keywords:**")
    st.markdown(kw_tags_html(st.session_state.final_keywords), unsafe_allow_html=True)
    st.markdown("")

    with st.expander("Inhalt"):
        for i, item in enumerate(st.session_state.article_outline, 1):
            st.markdown(f"{i}. {item}")

    st.markdown("**Artikel-Entwurf:**")
    st.markdown(f'<div class="article-box">{st.session_state.article_draft}</div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns([3, 1])
    with col1:
        safe_title = "".join(c for c in st.session_state.article_title if c.isalnum() or c in " -_")[:50].strip()
        st.download_button(
            label="Artikel herunterladen (.txt)",
            data=build_output_text().encode("utf-8"),
            file_name=f"{safe_title}.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
        )
    with col2:
        if st.button("Neu starten", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()