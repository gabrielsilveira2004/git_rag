from typing import List
import re
from langchain_core.documents import Document
from app.rag.vectorstore import load_vectorstore

# -----------------------------
# Retrieval settings
# -----------------------------
SEARCH_K_PER_QUERY = 6
FINAL_TOP_K = 6
MMR_LAMBDA = 0.7

# Section priority per intent
INTENT_SECTION_PRIORITY = {
    "DEFINITION": ["NAME", "INTRO"],
    "HOW": ["DESCRIPTION", "OPTIONS", "EXAMPLES"],
    "WHY": ["NOTES", "DESCRIPTION"],
    "COMPARE": ["DESCRIPTION", "NOTES"],
    "GENERAL": ["DESCRIPTION", "INTRO", "OPTIONS"],
}


# -----------------------------
# Query understanding
# -----------------------------

def extract_git_command(query: str) -> str | None:
    match = re.search(r"git\s+([a-z\-]+)", query.lower())
    return match.group(1) if match else None


def detect_intent(query: str) -> str:
    q = query.lower()

    if any(w in q for w in ["what is", "what does", "define", "meaning of"]):
        return "DEFINITION"
    if any(w in q for w in ["how", "how to", "works", "do i"]):
        return "HOW"
    if any(w in q for w in ["why", "when", "should"]):
        return "WHY"
    if any(w in q for w in ["difference", "compare", "vs"]):
        return "COMPARE"

    return "GENERAL"


# -----------------------------
# Query expansion
# -----------------------------

def expand_query(query: str) -> List[str]:
    expanded = [query]

    command = extract_git_command(query)
    intent = detect_intent(query)

    if command:
        expanded.append(f"git {command}")
        expanded.append(f"Git Command: {command}")

        for section in INTENT_SECTION_PRIORITY.get(intent, []):
            expanded.append(f"Git Command: {command} Section: {section}")

    else:
        # fallback for generic Git questions
        expanded.append("Git version control system")
        expanded.append("Git distributed version control")

    return list(dict.fromkeys(expanded))


# -----------------------------
# Deduplication
# -----------------------------

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []

    for doc in docs:
        key = (
            doc.metadata.get("command"),
            doc.metadata.get("section"),
            doc.metadata.get("chunk_index"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique


# -----------------------------
# Reranking (crucial)
# -----------------------------

def rerank_results(results: List[Document], query: str, intent: str) -> List[Document]:
    qtokens = set(re.findall(r"\w+", query.lower()))
    preferred_sections = INTENT_SECTION_PRIORITY.get(intent, [])

    scored = []

    for idx, doc in enumerate(results):
        score = 0

        cmd = (doc.metadata.get("command") or "").lower()
        sec = (doc.metadata.get("section") or "").upper()
        content = (doc.page_content or "").lower()
        length = len(doc.page_content)

        # Base: earlier results matter
        score += max(0, 20 - idx)

        # Section priority
        if sec in preferred_sections:
            score += 15

        # Command match
        if any(t in cmd for t in qtokens):
            score += 10

        # Content keyword match
        score += sum(1 for t in qtokens if t in content)

        # Penalize very long chunks (bad for simple answers)
        if length > 1500:
            score -= 5

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


# -----------------------------
# Main retrieval
# -----------------------------

def retrieve_documents(query: str, k: int = FINAL_TOP_K, vectorstore=None) -> List[Document]:
    vectorstore = load_vectorstore()

    intent = detect_intent(query)
    expanded_queries = expand_query(query)

    all_results: List[Document] = []

    for q in expanded_queries:
        hits = vectorstore.max_marginal_relevance_search(
            q,
            k=SEARCH_K_PER_QUERY,
            lambda_mult=MMR_LAMBDA,
        )
        all_results.extend(hits)

    unique = deduplicate_documents(all_results)
    reranked = rerank_results(unique, query, intent)

    return reranked[:k]


# -----------------------------
# Manual test
# -----------------------------
if __name__ == "__main__":
    tests = [
        "What is Git?",
        "How to create a new branch?",
        "Explain git rebase",
        "Why use git stash?",
        "Difference between git merge and rebase",
    ]

    for q in tests:
        print("\n" + "=" * 80)
        print(f"Query: {q}\n")

        docs = retrieve_documents(q)

        for i, d in enumerate(docs, 1):
            print(f"[{i}] {d.metadata.get('command')} | {d.metadata.get('section')}")
            print(d.page_content[:300].strip())
            print()
