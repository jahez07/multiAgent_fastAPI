"""
knowledge_base/ingest_directives.py - Ingest EU directive markdown files into Qdrant.

Reads markdown files from a directory. Each file is one directive.
The script splits each file into chunks by heading (## or ###), 
so each article/section becomes a separate searchable document. 

Expected markdown format:
    # EU Water Framework Directive (2000/60/EC)

    ## Article 1 - Purpose
    This Directive establishes a framework for the protection .....

    ## Article 4 - Environmental objectives
    Member States shall implement the necessary measures...

Each chunk becomes one document in Qdrant:
    text:       "Directive: EU Water Framework Directive (2000/60/EC) |
                Article 4 - Environmental objectives |
                Member States shall implement the necessary measures...."
    metadata:   {directives: "EU Water Framework..", article:"Article 4- ...", source_file: ".."}

Run:
  python -m knowledge_base.ingest_directives --dir knowledge_base/data/directives/
  python -m knowledge_base.ingest_directives --dir knowledge_base/data/directives/ --recreate
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings
from knowledge_base.embeddings import embed_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("ingest_directives")


def parse_markdown_file(file_path: Path) -> list[dict]:
    """
    Parse a markdown fiel into chunks by heading.

    Strategy:
        - The H1 heading (#...) becomes the directive name
        - Each H2/H3 heading (##... or ### ... ) starts a new chunk
        - The chunk includes all text until the next heading.
        - Chunks shorter than 50 chars are merged with the previous chunk
          (catches stubs like '## Article 3 - [Reserved]")
    
    Returns:
        [{"directive": "EU Water ....", "article": "Article 4 - ..", "text}]
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    directive_name = file_path.stem
    chunks = []
    current_article = None
    current_text = []

    for line in lines:
        # H1 = directive name
        if line.startswith("# ") and not line.startswith("## "):
            directive_name = line.lstrip("# ").strip()
            continue

        # H2 or H3 = new article/section
        if re.match(r"^#{2,3}\s", line):
            # Save previous chunk
            if current_article and current_text:
                text = "\n".join(current_text).strip()
                if len(text) > 50:
                    chunks.append({
                        "directive": directive_name,
                        "article": current_article,
                        "text": text,
                    })
                elif chunks:
                    # Merge short stub with previous chunk
                    chunks[-1]["text"] += f"\n\n{current_article}\n{text}"
            
            current_article = line.lstrip("#").strip()
            current_text = []
        else:
            current_text.append(line)

    # Don't forget the last chunk
    if current_article and current_text:
        text = "\n".join(current_text).strip()
        if len(text) > 50:
            chunks.append({
                "directive": directive_name,
                "article": current_article,
                "text": text,
            })
    
    return chunks


def build_document_text(chunk: dict) -> str:
    """
    Build the text that gets embedded.

    Includes directive name + article heading + full text so the
    embedding captures both the regulatory context and the content.
    """
    # Truncate very long articles to -1500 chars for embedding quality
    text = chunk["text"][:1500]

    return (
        f"Directive: {chunk['directive']} |"
        f"Section: {chunk['article']} |"
        f"Content: {text}"
    )


async def ingest(dir_path: str, recreate: bool = False):
    """
    Main ingestion pipeline:
        1. Find all .md files in the directory
        2. Parse each into article chunks
        3. Embed all chunks via Ollama
        4. Create/recreate Qdrant collection
        5. Upload vectors with metadata
    """
    collection = settings.qdrant_directives_collection
    directive_dir = Path(dir_path)

    if not directive_dir.is_dir():
        logger.error("Directory not found: %s", dir_path)
        sys.exit(1)

    # ── Step 1: Find markdown files ──
    md_files = sorted(directive_dir.glob("*.md"))
    if not md_files:
        logger.error("No .md files found in %s", dir_path)
        sys.exit(1)

    logger.info("Found %d directive files", len(md_files))

    # ── Step 2: Parse into chunks ──
    all_chunks = []
    for md_file in md_files:
        chunks = parse_markdown_file(md_file)
        logger.info("   %s: %d sections", md_file.name, len(chunks))
        for chunk in chunks:
            chunk["source_file"] = md_file.name
        all_chunks.extend(chunks)
    
    logger.info("Total chunks: %d", len(all_chunks))

    if not all_chunks:
        logger.error("No chunks extracted. Check your markdown format.")
        sys.exit(1)

    
    # ── Step 3: Embed ──
    texts = [build_document_text(c) for c in all_chunks]
    logger.info("Embedding %d chunks via Ollama (%s)...", len(texts), settings.embedding_model)

    # Batch in groups of 50 to avoid timeout on large directive sets
    batch_size = 50
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = await embed_batch(batch, prefix="search_document")
        all_vectors.extend(vectors)
        logger.info("   Embedding %d/%d", min(i + batch_size, len(texts)), len(texts))

    logger.info("Embedding complete (dimension=%d)", len(all_vectors[0]))

    # ── Step 4: Create Qdrant collection ──
    client = QdrantClient(url=settings.qdrant_url)

    if recreate:
        try:
            client.delete_collection(collection)
            logger.info("Deleted existing collection '%s'", collection)
        except Exception:
            pass
    
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimenstions,
                distance=Distance.COSINE,
            )
        )
        logger.info("Created collection '%s'", collection)
    except Exception as e:
        if "already exists" in str(e):
            logger.info("Collection '%s' already exists", collection)
        else:
            raise

    
    # ── Step 5: Upload points ──
    points = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, all_vectors)):
        points.append(PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "directive": chunk["directive"],
                "article": chunk["article"],
                "text": chunk["text"][:2000],
                "source_file": chunk["source_file"],
                "type": "directive_article",
            },
        ))
    
    # Upload in batches of 100
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        client.upsert(collection_name=collection, points=batch)
    
    logger.info("Uploaded %d points to '%s'", len(points), collection)

    # ── Verifty ──
    info = client.get_collection(collection)
    logger.info("Collection '%s' now has %d points", collection, info.points_count)

    logger.info("Directive ingestion complete!")

def main():
    parser = argparse.ArgumentParser(description="Ingest EU directive markdown files into Qdrant")
    parser.add_argument(
        "--dir", "-d",
        required=True,
        help="Directory containing .md files (one per directive)",   
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection",
    )
    args = parser.parse_args()

    asyncio.run(ingest(args.dir, args.recreate))

if __name__ == "__main__":
    main()