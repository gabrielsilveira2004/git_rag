from typing import List, Set
import re
from langchain_core.documents import Document
from vectorstore import load_vectorstore

# Retrieval settings
SEARCH_K_PER_QUERY = 4
FINAL_TOP_K = 8
MMR_LAMBDA = 0.7

# Expand query into multiple related sub-queries to improve semantic recall
def expand_query(query: str) -> List[str]:
    expanded = [query]

    tokens = re.findall(r"[a-zA-Z\-]{3,}", query.lower())

    for token in tokens:
        expanded.append(token)
        expanded.append(f"git {token}")

    seen: Set[str] = set()
    result: List[str] = []

    for q in expanded:
        if q not in seen:
            seen.add(q)
            result.append(q)

    return result


# Deduplicate documents based on source and section metadata
def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen_keys = set()
    unique_docs: List[Document] = []

    for doc in documents:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("section"),
        )

        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(doc)

    return unique_docs

# Retrieve documents relevant to the query
def retrieve_documents(query: str) -> List[Document]:
    vectorstore = load_vectorstore()
    expanded_queries = expand_query(query)

    all_results: List[Document] = []

    for q in expanded_queries:
        results = vectorstore.max_marginal_relevance_search(
            q,
            k=SEARCH_K_PER_QUERY,
            lambda_mult=MMR_LAMBDA,
        )
        all_results.extend(results)

    unique_results = deduplicate_documents(all_results)

    return unique_results[:FINAL_TOP_K]


if __name__ == "__main__":
    test_queries = [
        "How does git checkout-index work?",
        "What is the difference between git merge and git rebase?",
        "Why should I use git branch before pushing changes?"
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}\n")

        results = retrieve_documents(query)

        for i, doc in enumerate(results, start=1):
            print(f"Result {i}")
            print(f"Source : {doc.metadata.get('source')}")
            print(f"Topic  : {doc.metadata.get('topic')}")
            print(f"Section: {doc.metadata.get('section')}")
            print("-" * 40)
            print(doc.page_content[:500].strip())
            print()
