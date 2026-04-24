"""
knowledge_base/search.py — Search utility for testing Qdrant retrieval.

Use this to verify your knowledge base returns relevant results
before wiring it into Agent 3.

Run interactively:
  python -m knowledge_base.search

Run with a specific query:
  python -m knowledge_base.search --query "city losing water through old pipes"
  python -m knowledge_base.search --query "energy efficiency requirements" --collection directives
"""

import argparse
import asyncio
import logging

from qdrant_client import QdrantClient

from app.config import settings
from knowledge_base.embeddings import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("search")


async def search_collection(
    query: str,
    collection: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search a Qdrant collection and return the top results.

    Uses "search_query" prefix for the embedding (vs "search_document"
    used during ingestion). This asymmetric prefix improves retrieval
    accuracy with nomic-embed-text.
    """
    # Embed the query
    query_vector = await embed_text(query, prefix="search_query")

    # Search Qdrant
    client = QdrantClient(url=settings.qdrant_url)
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": point.score,
            "payload": point.payload,
        }
        for point in results.points
    ]


async def search_products(query: str, top_k: int = 5) -> list[dict]:
    """Search the products collection."""
    return await search_collection(
        query, settings.qdrant_products_collection, top_k
    )


async def search_directives(query: str, top_k: int = 5) -> list[dict]:
    """Search the directives collection."""
    return await search_collection(
        query, settings.qdrant_directives_collection, top_k
    )


def print_results(results: list[dict], collection: str):
    """Pretty-print search results."""
    print(f"\n{'─' * 60}")
    print(f"  Collection: {collection} | {len(results)} results")
    print(f"{'─' * 60}")

    for i, result in enumerate(results, 1):
        score = result["score"]
        payload = result["payload"]

        if payload.get("type") == "product_feature":
            print(f"\n  {i}. [{score:.3f}] {payload['product']}")
            print(f"     Feature: {payload['feature']}")
            print(f"     {payload['description'][:120]}")
        elif payload.get("type") == "directive_article":
            print(f"\n  {i}. [{score:.3f}] {payload['directive']}")
            print(f"     {payload['article']}")
            print(f"     {payload['text'][:120]}...")
        else:
            print(f"\n  {i}. [{score:.3f}] {payload}")


async def interactive_mode():
    """Interactive search loop."""
    print("\n" + "=" * 60)
    print("  Knowledge Base Search")
    print("  Type a query to search both collections")
    print("  Commands: /products, /directives, /both (default), /quit")
    print("=" * 60)

    mode = "both"

    while True:
        try:
            query = input(f"\n[{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query == "/quit":
            break
        elif query == "/products":
            mode = "products"
            print("  Switched to products only")
            continue
        elif query == "/directives":
            mode = "directives"
            print("  Switched to directives only")
            continue
        elif query == "/both":
            mode = "both"
            print("  Switched to both collections")
            continue

        if mode in ("products", "both"):
            results = await search_products(query, top_k=3)
            print_results(results, settings.qdrant_products_collection)

        if mode in ("directives", "both"):
            results = await search_directives(query, top_k=3)
            print_results(results, settings.qdrant_directives_collection)


async def single_query(query: str, collection: str, top_k: int):
    """Run a single query and print results."""
    if collection == "products":
        results = await search_products(query, top_k)
        print_results(results, settings.qdrant_products_collection)
    elif collection == "directives":
        results = await search_directives(query, top_k)
        print_results(results, settings.qdrant_directives_collection)
    else:
        results = await search_products(query, top_k)
        print_results(results, settings.qdrant_products_collection)
        results = await search_directives(query, top_k)
        print_results(results, settings.qdrant_directives_collection)


def main():
    parser = argparse.ArgumentParser(description="Search the knowledge base")
    parser.add_argument("--query", "-q", help="Search query (omit for interactive mode)")
    parser.add_argument("--collection", "-c", default="both",
                        choices=["products", "directives", "both"])
    parser.add_argument("--top-k", "-k", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        asyncio.run(single_query(args.query, args.collection, args.top_k))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()