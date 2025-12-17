from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

MODEL_NAME = "google/flan-t5-large"

# Create and configure the language model pipeline
def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    hf_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )

    return HuggingFacePipeline(pipeline=hf_pipe)


# Detect the user's intent based on simple linguistic patterns
def detect_intent(question: str) -> str:
    q = question.lower().strip()

    if "difference" in q or "compare" in q:
        return "comparison"
    if q.startswith("why"):
        return "reasoning"
    if q.startswith("how"):
        return "procedural"
    if q.startswith("what"):
        return "definition"

    return "general"


# Build a bounded textual context from retrieved documents
def build_context(documents: List[Document], max_chars: int = 5000) -> str:
    blocks = []
    total_chars = 0

    for idx, doc in enumerate(documents, start=1):
        text = doc.page_content.strip()

        if not text:
            continue

        source_block = f"[Source {idx}]\n{text}"

        if total_chars + len(source_block) > max_chars:
            break

        blocks.append(source_block)
        total_chars += len(source_block)

    return "\n\n".join(blocks)


# Build a prompt template based on the detected intent
def build_prompt(intent: str) -> PromptTemplate:
    base_rules = """
You are an assistant specialized in Git documentation.
Use the provided context as your primary source.
Explain concepts clearly and in sufficient detail.
If the answer cannot be inferred from the context, say you do not know.
"""

    intent_instructions = {
        "procedural": "Explain the process step by step, using clear and practical language.",
        "reasoning": "Explain the reasoning, purpose, and implications behind the concept.",
        "comparison": "Compare the concepts, highlighting key differences and use cases.",
        "definition": "Provide a clear and concise definition, adding context when helpful.",
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


# Orchestrate retrieval, prompt construction, and answer generation
def answer_question(question: str, documents: List[Document], vectorstore=None) -> str:
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

    from .vectorstore import load_vectorstore
    from .retrieve import retrieve_documents

    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    test_questions = [
        "How does git checkout-index work?",
        "Why should I use git branch before pushing changes?",
        "What is the difference between git merge and git rebase?"
    ]

    for question in test_questions:
        intent = detect_intent(question)
        k = 6 if intent == "comparison" else 4

        print("\n" + "-" * 50)
        print(f"Question: {question}")
        print(f"Detected intent: {intent}")

        documents = retrieve_documents(question, k=k)
        answer = answer_question(question, documents)

        print("\nAnswer:")
        print(answer)
