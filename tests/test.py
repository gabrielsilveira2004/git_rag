from pathlib import Path
import sys
from collections import Counter

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.retrieve import retrieve_documents


def inspect_retrieval(query: str, k: int = 12):
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    docs = retrieve_documents(query=query, k_per_query=k)

    print(f"\nDocuments retrieved: {len(docs)}\n")

    source_counter = Counter()

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "UNKNOWN")
        section = doc.metadata.get("section_title", "N/A")

        source_counter[source] += 1

        print(f"Result {i}")
        print(f"Source        : {source}")
        print(f"Section title : {section}")
        print(f"Content size  : {len(doc.page_content)} chars")
        print("Content preview:")
        print(doc.page_content[:250].replace("\n", " "), "...")
        print("-" * 60)

    print("\n📊 Source distribution:")
    for source, count in source_counter.most_common():
        print(f"{count:>2} chunks -> {source}")

    print("\nUnique sources:", len(source_counter))
    print("=" * 80)
    print()


if __name__ == "__main__":
    inspect_retrieval("What is a Git branch?")
    inspect_retrieval("How does git rebase work?")
    inspect_retrieval("What is the git config file used for?")
