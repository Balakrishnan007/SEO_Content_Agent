import re
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Paths to source documents
TRANSCRIPTS_DIR = Path(__file__).parent.parent / "Docs" / "Transkripte"
PRODUCTS_DIR    = Path(__file__).parent.parent / "Docs" / "Produkte"

# Qdrant and embedding config
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "SESTdigital")
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-de"
VECTOR_DIM      = 768

# Chunking config for transcripts
TURNS_PER_CHUNK      = 3     # number of speaker turns per chunk
OVERLAP              = 1     # overlapping turns between chunks for context continuity
MAX_TRANSCRIPT_CHARS = 2000  

# Chunking config for product docs
MAX_PRODUCT_CHARS = 2300  


# Read and Parse Transcript PDFs
def read_transcript(pdf_path: Path) -> str:
    """Converts a transcript PDF to plain text using Docling."""
    converter = DocumentConverter()
    result    = converter.convert(str(pdf_path))
    return result.document.export_to_text()


def parse_turns(text: str) -> list:
    """
    Parses raw transcript text into a list of speaker turns.
    Each turn contains timestamp, speaker name and spoken text.
    """
    pattern = re.compile(r'\[(\d{2}:\d{2})\]\s+(\[?[^\[\:]+?\]?):\s+')
    parts   = pattern.split(text)
    turns   = []
    i = 1
    while i + 2 <= len(parts):
        timestamp = parts[i].strip()
        speaker   = parts[i + 1].strip()
        content   = parts[i + 2].strip() if i + 2 < len(parts) else ""
        content   = re.split(r'\[\d{2}:\d{2}\]', content)[0].strip()
        if timestamp and speaker and content:
            turns.append({
                "timestamp": timestamp,
                "speaker":   speaker,
                "text":      content,
            })
        i += 3
    return turns


def make_transcript_chunks(turns: list) -> list:
    """
    Groups speaker turns into overlapping chunks for RAG indexing.
    Uses a sliding window of TURNS_PER_CHUNK with OVERLAP turns
    repeated between chunks to preserve conversational context.
    """
    chunks = []
    step   = TURNS_PER_CHUNK - OVERLAP
    i      = 0

    while i < len(turns):
        window = turns[i: i + TURNS_PER_CHUNK]

        # trim window if combined text exceeds max character limit
        while len(window) > 1:
            combined = " ".join(t["text"] for t in window)
            if len(combined) <= MAX_TRANSCRIPT_CHARS:
                break
            window = window[:-1]

        text = "\n\n".join(
            f"{t['speaker']} [{t['timestamp']}]: {t['text']}"
            for t in window
        )

        chunks.append({
            "text":            text,
            "timestamp_start": window[0]["timestamp"],
            "timestamp_end":   window[-1]["timestamp"],
            "speakers":        list(dict.fromkeys(t["speaker"] for t in window)),
        })

        i += step

    return chunks


# Read and Parse Product Docs
def read_product_pages(pdf_path: Path) -> list:
    """
    Extracts text from a product PDF page by page using Docling.
    Returns a list of dicts with page number and text content.
    """
    converter  = DocumentConverter()
    result     = converter.convert(str(pdf_path))
    doc        = result.document
    page_texts = {}

    for item, _ in doc.iterate_items():
        try:
            pno = item.prov[0].page_no if item.prov else None
        except Exception:
            pno = None
        if pno is None:
            continue
        text = getattr(item, "text", None)
        if text and text.strip():
            page_texts.setdefault(pno, []).append(text.strip())

    pages = []
    for pno in sorted(page_texts.keys()):
        combined = "\n".join(page_texts[pno])
        pages.append({"page_number": pno, "text": combined})

    return pages


def extract_heading(text: str) -> str:
    """Returns the first non-empty line of a text block as its heading."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def make_product_chunks(pages: list) -> list:
    """
    Splits product doc pages into chunks within MAX_PRODUCT_CHARS limit.
    Attaches previous page heading and tail for context continuity.
    """
    chunks = []

    for idx, page in enumerate(pages):
        text    = page["text"]
        page_no = page["page_number"]

        if idx > 0:
            prev_text    = pages[idx - 1]["text"]
            prev_heading = extract_heading(prev_text)
            prev_tail    = prev_text[-100:].strip()
        else:
            prev_heading = ""
            prev_tail    = ""

        if len(text) > MAX_PRODUCT_CHARS:
            # split long pages into sub-chunks by line
            sub_chunks = []
            current    = ""
            for line in text.splitlines():
                if len(current) + len(line) > MAX_PRODUCT_CHARS and current:
                    sub_chunks.append(current.strip())
                    current = line
                else:
                    current += "\n" + line
            if current.strip():
                sub_chunks.append(current.strip())

            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "text":         sub_text,
                    "page_number":  page_no,
                    "sub_chunk":    sub_idx + 1,
                    "prev_heading": prev_heading if sub_idx == 0 else extract_heading(text),
                    "prev_tail":    prev_tail    if sub_idx == 0 else sub_chunks[sub_idx - 1][-100:].strip(),
                })
        else:
            chunks.append({
                "text":         text,
                "page_number":  page_no,
                "sub_chunk":    1,
                "prev_heading": prev_heading,
                "prev_tail":    prev_tail,
            })

    return chunks


# Main ingestion pipeline
def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    # connect to Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # delete existing collection and recreate fresh
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME}")

    global_id = 0

    # process transcript PDFs
    print("\nProcessing transcripts...")
    for pdf_path in sorted(TRANSCRIPTS_DIR.glob("*.pdf")):
        text   = read_transcript(pdf_path)
        turns  = parse_turns(text)
        chunks = make_transcript_chunks(turns)

        print(f"  {pdf_path.name} → {len(turns)} turns, {len(chunks)} chunks")

        texts      = [c["text"] for c in chunks]
        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        points = [
            PointStruct(
                id      = global_id + idx,
                vector  = vector,
                payload = {
                    "text":            chunk["text"],
                    "source_type":     "transcript",
                    "source_file":     pdf_path.name,
                    "timestamp_start": chunk["timestamp_start"],
                    "timestamp_end":   chunk["timestamp_end"],
                    "speakers":        chunk["speakers"],
                },
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Stored {len(points)} points.")
        global_id += len(points)

    # process product doc PDFs
    print("\nProcessing product docs...")
    for pdf_path in sorted(PRODUCTS_DIR.glob("*.pdf")):
        pages  = read_product_pages(pdf_path)
        chunks = make_product_chunks(pages)

        print(f"  {pdf_path.name} → {len(pages)} pages, {len(chunks)} chunks")

        texts      = [c["text"] for c in chunks]
        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        points = [
            PointStruct(
                id      = global_id + idx,
                vector  = vector,
                payload = {
                    "text":         chunk["text"],
                    "source_type":  "product_doc",
                    "source_file":  pdf_path.name,
                    "page_number":  chunk["page_number"],
                    "sub_chunk":    chunk["sub_chunk"],
                    "prev_heading": chunk["prev_heading"],
                    "prev_tail":    chunk["prev_tail"],
                },
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Stored {len(points)} points.")
        global_id += len(points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nIngestion complete. Total vectors in '{COLLECTION_NAME}': {count}")


if __name__ == "__main__":
    main()