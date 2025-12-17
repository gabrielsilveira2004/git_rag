from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

MODEL_NAME = "google/flan-t5-base"


def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    hf_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=hf_pipe)


def detect_intent(question: str) -> str:
    q = question.lower()

    if "difference" in q or "compare" in q:
        return "comparison"
    if q.startswith("why"):
        return "reasoning"
    if q.startswith("how"):
        return "procedural"
    if q.startswith("what"):
        return "definition"

    return "general"


def build_context(documents: List[Document], max_chars: int = 5000) -> str:
    blocks = []
    total = 0

    for i, doc in enumerate(documents, start=1):
        text = doc.page_content.strip()
        if not text:
            continue

        header = f"[Source {i}]"
        block = f"{header}\n{text}"

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


def build_prompt(intent: str) -> PromptTemplate:
    base_rules = """
You are an assistant specialized in Git documentation.
Use only the provided context.
If the answer cannot be derived from the context, say you do not know.
"""

    intent_instructions = {
        "procedural": "Explain the process step by step in clear language.",
        "reasoning": "Explain the reasoning, purpose, and consequences.",
        "comparison": "Compare the concepts, highlighting differences and use cases.",
        "definition": "Provide a clear and concise definition with context.",
        "general": "Provide a clear and helpful explanation."
    }

    template = f"""
{base_rules}

Instruction:
{intent_instructions[intent]}

Context:
{{context}}

Question:
{{question}}

Answer:
""".strip()

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


def answer_question(question: str, documents: List[Document]) -> str:
    intent = detect_intent(question)
    context = build_context(documents)

    prompt = build_prompt(intent)
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from vectorstore import load_vectorstore
    from retrieve import retrieve_documents

    print("Loading vectorstore...")
    vs = load_vectorstore()

    tests = [
        "How does git checkout-index work?",
        "Why should I use git branch before pushing changes?",
        "What is the difference between git merge and git rebase?"
    ]

    for q in tests:
        intent = detect_intent(q)
        k = 6 if intent == "comparison" else 4

        print("\n" + "-" * 40)
        print(f"Question: {q}")
        print(f"Detected intent: {intent}")

        docs = retrieve_documents(q)
        answer = answer_question(q, docs)

        print("\nAnswer:")
        print(answer)
