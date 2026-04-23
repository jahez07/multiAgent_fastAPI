"""
knowledge_base/ingest_products.py - Ingest product catalog into Qdrant.

Reads the Excel/CSV file where each row is one feature:

    | Product Name      | Features           | Description          |
    | Gas Clamp Sensor  | Non-intrusive inst.| Clamps externally ...|
    |                   | Real-time monito.. | Continuously ....    |
    | Water Clamp Sen...| Proactive leak de..| Detects Abnormal ... |

Each feature becomes one document in Qdrant with this structure:
    text:       "Product: Water Clamp Sensor | Feature: Proactive water leak detection |
                Description: Detects abnormal flow patterns that indicate potential leaks."
    metadata:   {product:"Water Clamp Sensor", feature:"Proactive water leak detection"}

This means when Agent 3 searches "city losing water through old pipes",
Qdrant returns the specific feature with its product context. 

Run:
    python -m knowledge_base.ingest_products --file knowledge_base/data/product_features.xlsx
    python -m knowledge_base.ingest_products --file knowledge_base/data/product_features.csv
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue,
)

from app.config import settings
from knowledge_base.embeddings import embed_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
)
logger = logging.getLogger("ingest_products")

def load_product_data(file_path: str) -> list[dict]:
    """
    Load the product spreadsheet and normalize it.

    Handles the "merged cell" pattern where Product Name is only
    filled in on the row of each product group. We forward-fill
    the product name down to every feature row. 

    Returns a list of dicts:
        [{"product":"Gas Clamp Sensor", "feature":"Non-intrusive..", "description":"Clamps..."}]
    """
    path = Path(file_path)

    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .xlsx, .xls, or .csv")
    
    # Normalize column names (handle case variations, extra spaces)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")

    # Expected columns (flexible naming)
    product_col = next((c for c in df.columns if "product" in c), None)
    feature_col = next((c for c in df.columns if "feature" in c), None)
    desc_col = next((c for c in df.columns if "desc" in c), None)

    if not all([product_col, feature_col, desc_col]):
        found = list(df.columns)
        raise ValueError(
            f"Could not find required columns. Found: {found}."
            f"Expected columns containing: 'product', 'feature', 'desc'"
        )
    
    # Forward-fill product name (handles the merged-cell pattern)
    df[product_col] = df[product_col].ffill()

    # Drop rows where feature or description is empty
    df = df.dropna(subset=[feature_col, desc_col])

    features = []
    for _, row in df.iterrows():
        features.append({
            "product": str(row[product_col]).strip(),
            "feature": str(row[feature_col]).strip(),
            "description": str(row[desc_col]).strip(),
        })
    
    return features

def build_document_text(item: dict) -> str:
    """
    Build the text that gets embedded. 

    Format: "Product: X | Feature: Y | Description: Z"

    This concatenation ensures the embedding captures the relationship
    between the product, the feature name, and what it does. When Agent 3
    searches for a problem, the description carries the semantic match 
    while product and feature provide context.
    """
    return (
        f"Product: {item['product']} | "
        f"Feature: {item['feature']} | "
        f"Description: {item['description']}"
    )


async def ingest(file_path: str, recreate: bool = False):
    """
    Main ingestion pipeline:
        1. Load spreadsheet
        2. Build document texts
        3. Embed all documnets via Ollama
        4. Create/recreate Qdrant collection
        5. Upload vectors with metadata
    """
    collection = settings.qdrant_products_collection

    # ── Step 1: Load data ──
    logger.info("Loading product data from %s", file_path)
    features = load_product_data(file_path)
    logger.info("Found %d features across products", len(features))

    # Show summary 
    products = set(f["product"] for f in features)
    for product in sorted(products):
        count = sum(1 for f in features if f["product"] == product)
        logger.info("   %s: %d features", product, count)

    # ── Step 2: Build document texts ──
    texts = [build_document_text(f) for f in features]

    # ── Step 3: Embed ──
    logger.info("Embedding %d documents via Ollama (%s)...", len(texts), settings.embedding_model)
    vectors = await embed_batch(texts, prefix="search_document")
    logger.info("Embedding complete (dimension=%d)", len(vectors[0]))

    # ── Step 4: Create Qdrant collection.
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
            ),
        )
        logger.info("Created collection '%s'", collection)
    except Exception as e:
        if "already exists" in str(e):
            logger.info("Collection '%s' already exists", collection)
        else:
            raise

    
    # ── Step 5: Upload points ──
    points = []
    for i, (feature, vector) in enumerate(zip(features, vectors)):
        points.append(PointStruct(
            id = str(uuid4()),
            vector=vector,
            payload={
                "product": feature["product"],
                "feature": feature["feature"],
                "description": feature["description"],
                "text": texts[i],
                "type": "product_features",
            },
        ))
    
    client.upsert(collection_name=collection, points=points)
    logger.info("Uploaded %d points to '%s'", len(points), collection)

    # ── Verify ──
    info = client.get_collection(collection)
    logger.info("Collection '%s' now has %d points", collection, info.points_count)

    logger.info("Product ingestion complete!")


def main():
    parser = argparse.ArgumentParser(description="Ingest product catalog into Qdrant")
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the product spreadsheet (.xlsx, .xls, or .csv)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (fresh start)",
    )
    args = parser.parse_args()

    asyncio.run(ingest(args.file, args.recreate))


if __name__ == "__main__":
    main()