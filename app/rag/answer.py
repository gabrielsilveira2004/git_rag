from typing import List
import re
import torch
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# Model configuration
# -----------------------------
MODEL_NAME = "google/flan-t5-small"

_llm_instance = None  # Singleton cache


def get_llm():
    """Load FLAN-T5 model once and reuse (API-safe)."""
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    hf_pipeline = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=120,
        min_length=10,
        do_sample=False,  # deterministic (important for API)
        device=0 if device == "cuda" else -1,
    )

    _llm_instance = HuggingFacePipeline(pipeline=hf_pipeline)
    return _llm_instance


# -----------------------------
# Intent detection
# -----------------------------
def detect_intent(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["difference", "compare", "vs"]):
        return "comparison"
    if q.startswith("why"):
        return "reasoning"
    if q.startswith("how"):
        return "procedural"
    if q.startswith("what"):
        return "definition"

    return "general"


# -----------------------------
# Context builder
# -----------------------------
def build_context(documents: List[Document], max_chars: int = 700) -> str:
    """
    Build compact, structured context.
    FLAN-T5 hates long and noisy inputs.
    """
    blocks = []
    total = 0

    for doc in documents:
        command = doc.metadata.get("command", "unknown")
        section = doc.metadata.get("section", "")

        text = doc.page_content.strip()

        # Remove semantic headers if present
        if text.startswith("Git Command:"):
            lines = text.splitlines()
            text = "\n".join(lines[3:]).strip()

        if not text:
            continue

        block = f"Command: {command}\nSection: {section}\n{text[:350]}"

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(intent: str) -> PromptTemplate:
    """
    Explicit prompts per intent.
    FLAN-T5 performs MUCH better this way.
    """

    if intent == "definition":
        instruction = (
            "Give a short, clear definition in one sentence."
        )
    elif intent == "procedural":
        instruction = (
            "Explain briefly how it works in one or two sentences."
        )
    elif intent == "comparison":
        instruction = (
            "Explain the difference clearly in two short sentences."
        )
    elif intent == "reasoning":
        instruction = (
            "Explain the reason briefly and clearly."
        )
    else:
        instruction = (
            "Answer briefly and clearly in one or two sentences."
        )

    template = f"""You are answering questions about Git.

{instruction}

Context:
{{context}}

Question: {{question}}

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


# -----------------------------
# Answer generation
# -----------------------------
def answer_question(question: str, documents: List[Document]) -> str:
    intent = detect_intent(question)
    context = build_context(documents)

    prompt = build_prompt(intent)
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke(
            {"context": context, "question": question}
        )
    except Exception:
        return "I couldn't generate an answer at this time."

    if not result:
        return "I couldn't generate an answer."

    return post_process_answer(result)


# -----------------------------
# Post-processing
# -----------------------------
def post_process_answer(answer: str) -> str:
    if not answer:
        return answer

    # Remove bullet/code artifacts
    lines = [
        ln for ln in answer.splitlines()
        if not ln.strip().startswith(("-", "`", "["))
    ]

    text = " ".join(ln.strip() for ln in lines if ln.strip())

    # Keep only first 2 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    short = " ".join(sentences[:2]).strip()

    # Hard cap
    if len(short) > 220:
        short = short[:217].rsplit(" ", 1)[0] + "..."

    return short


# -----------------------------
# Manual test
# -----------------------------
if __name__ == "__main__":
    from app.rag.retrieve import retrieve_documents

    tests = [
        "What is Git?",
        "Explain git commit",
        "How does git branch work?",
        "Difference between git merge and git rebase",
        "Why use git stash?",
    ]

    for q in tests:
        print("\n" + "=" * 80)
        print(f"Question: {q}")

        docs = retrieve_documents(q)
        answer = answer_question(q, docs)

        print("\nAnswer:\n")
        print(answer)
