# Git RAG – Chat With Git Docs

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)
![GitPython](https://img.shields.io/badge/GitPython-3.1.45-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-red.svg)

Projeto para ingestão da documentação oficial do Git e criação de um sistema RAG (Retrieval-Augmented Generation) completo, servindo como base para a feature **"Chat With Docs AI – Git"**.

O sistema clona o repositório oficial do Git, processa arquivos de documentação (`.adoc`, `.md`, `.txt`), realiza chunking semântico por seções, constrói um vector store para busca semântica e fornece uma API REST para responder perguntas sobre Git usando IA generativa.

## Como Funciona

1. **Ingestão**: Clona o repositório Git e carrega arquivos de documentação.
2. **Chunking**: Divide os documentos em chunks semânticos por seções.
3. **Indexação**: Constrói um vector store com embeddings para busca eficiente.
4. **API**: Fornece endpoints para chat com a documentação via RAG.

## Funcionalidades

- ✅ Clonagem automática do repositório Git (com verificação para evitar re-clones desnecessários).
- ✅ Processamento de documentação: `.adoc`, `.md`, `.txt`.
- ✅ Chunking semântico por seções AsciiDoc/Markdown.
- ✅ Vector store com FAISS e embeddings HuggingFace.
- ✅ API REST com FastAPI para chat inteligente.
- ✅ Detecção de intenção da pergunta (procedural, reasoning, comparison, etc.).
- ✅ Respostas geradas com LLM (FLAN-T5).
- ✅ Testes automatizados para todos os módulos.

## Estrutura do Projeto

```
git_rag/
├── app/
# Git RAG – Chat With Git Docs

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)
![GitPython](https://img.shields.io/badge/GitPython-3.1.45-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-red.svg)

Projeto para ingestão da documentação oficial do Git e criação de um sistema RAG (Retrieval-Augmented Generation).

Resumo rápido — o que este repositório faz de diferente:

- Foca especificamente na documentação oficial do Git (clona https://github.com/git/git) e preserva metadados de commit para rastreabilidade.
- Usa chunking por seções (AsciiDoc/Markdown-aware) para manter contexto técnico por tópicos/sections, reduzindo ruído em buscas semânticas.
- Constrói um vector store local com FAISS + `sentence-transformers/all-MiniLM-L6-v2` para embeddings leves e rápidos.
- Recuperação com MMR (max-marginal-relevance) e expansão controlada de consultas para melhores resultados relevantes e menos duplicação.
- Geração de respostas com um LLM local via `transformers` (ex.: FLAN-T5) encapsulado em `langchain_huggingface` para pipelines text2text.
- Oferece uma API FastAPI simples que carrega o vectorstore e o LLM na inicialização, e ajusta `top_k` dinamicamente segundo a intenção da pergunta.

O objetivo é ser uma base prática para construir assistentes técnicos orientados a documentação, com ênfase em reprodutibilidade (índice salvo em `data/vectorstore/`) e explicabilidade (metadados de fonte e commit).

## Como Funciona (visão rápida)

1. **Ingestão**: `scripts/ingest_git.py` clona (ou atualiza) o repositório oficial do Git e carrega arquivos de documentação (`.adoc`, `.md`, `.txt`). Cada documento recebe metadados incluindo o hash do commit.
2. **Chunking**: `app/rag/chunk.py` divide cada documento por seções/headers para preservar contexto técnico por tópico.
3. **Indexação**: `app/rag/vectorstore.py` cria um index FAISS a partir dos chunks usando embeddings da família `sentence-transformers`.
4. **Recuperação**: `app/rag/retrieve.py` usa busca semântica com MMR e expansão de query para obter documentos relevantes e desduplicar resultados.
5. **Geração**: `app/rag/answer.py` monta um prompt (com detecção de intenção) e gera resposta via pipeline HF (ex.: `google/flan-t5-large`).
6. **API**: `app/api/main.py` expõe `/chat` para requisições, carregando o vectorstore e o LLM na inicialização.

## Funcionalidades principais

- Clonagem/atualização do repositório Git e extração de documentos de `Documentation/`.
- Chunking por seções para preservar contexto técnico.
- Vector store FAISS local com embeddings HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`).
- Recuperação com MMR e expansão de consulta para reduzir duplicatas e melhorar cobertura.
- Prompting com detecção de intenção (procedural, reasoning, comparison, definition, general).
- Respostas geradas por um LLM local via `transformers` integrado com `langchain_huggingface`.

Nota: alguns testes e módulos utilitários estão incluídos, e o vectorstore é salvo em `data/vectorstore/` para reutilização sem reindexação.

## Estrutura do Projeto

```
git_rag/
├── app/
│   ├── api/
│   │   └── main.py          # API FastAPI
│   └── rag/
│       ├── __init__.py
│       ├── ingest.py        # Ingestão de documentos
│       ├── chunk.py         # Chunking por seções
│       ├── vectorstore.py   # Construção e gerenciamento do vector store
│       ├── retrieve.py      # Recuperação de documentos relevantes
│       └── answer.py        # Geração de respostas com LLM
├── data/
│   └── git_repo/            # Repositório Git clonado (ignorar no Git)
├── scripts/
│   └── ingest_git.py        # Script para ingestão completa
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Pré-requisitos

- Python 3.10 ou superior
- Git instalado
- Pelo menos 4GB de RAM (embeddings + index); para a geração com FLAN-T5, mais memória/VRAM pode ser necessária dependendo do backend (CPU vs GPU).

## Instalação e execução rápida

1. Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/gabrielsilveira2004/git_rag.git
cd git_rag
```

2. Crie e ative um ambiente virtual (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Se bloqueado: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Instale dependências:

```bash
pip install -r requirements.txt
```

4. Execute a ingestão e construção do índice:

```bash
python scripts/ingest_git.py
```

5. Inicie a API:

```bash
uvicorn app.api.main:app --reload
```

Abra `http://127.0.0.1:8000/docs` para a interface interativa.

## Uso

Siga estes passos principais:

1. `python scripts/ingest_git.py` — realiza clone/atualização, normalização, chunking, e salva o vectorstore em `data/vectorstore/`.
2. `uvicorn app.api.main:app --reload` — inicia a API que carrega o vectorstore e o LLM na inicialização.
3. `POST /chat` — endpoint principal para enviar perguntas. O corpo aceita `{"question": "...", "top_k": 4}`; se `top_k` não for fornecido, o sistema ajusta automaticamente com base na intenção detectada.

Exemplo de resposta inclui `answer`, `intent` e `sources` com snippets e nomes de arquivos.

## Testes

Execute os testes (quando disponíveis):

```bash
python -m pytest tests/
```

## Desenvolvimento

Para contribuir:

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/nome`
3. Faça commits claros e pequenos
4. `git push origin feature/nome` e abra um PR

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).