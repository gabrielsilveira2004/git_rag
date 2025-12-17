from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_NAME = "google/flan-t5-base"


def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        temperature=0.0
    )

    return HuggingFacePipeline(pipeline=pipe)


def build_prompt():
    return PromptTemplate(
        input_variables=["text"],
        template=(
            "Summarize the Git documentation excerpt below.\n"
            "Focus only on factual and procedural information.\n"
            "Do not add explanations or opinions.\n\n"
            "Text:\n{text}\n\n"
            "Summary:"
        )
    )


def summarize_chunks(documents: List[Document], max_chunks: int = 5) -> List[str]:
    llm = get_llm()
    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()

    summaries = []

    for doc in documents[:max_chunks]:
        text = doc.page_content.strip()
        if not text:
            continue

        summary = chain.invoke({"text": text})
        summaries.append(summary.strip())

    return summaries


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from retrieve import retrieve_documents

    question = "How does git checkout-index work?"
    docs = retrieve_documents(question)

    summaries = summarize_chunks(docs)

    print("\nSummaries:\n")
    for i, s in enumerate(summaries, 1):
        print(f"[{i}] {s}\n")
