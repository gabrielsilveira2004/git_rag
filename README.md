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
│   ├── ingest_test.py       # Testes de ingestão
│   ├── chunk_test.py        # Testes de chunking
│   └── vectorstore_test.py  # Testes do vector store
├── requirements.txt
├── README.md
└── .gitignore
```

## Pré-requisitos

- Python 3.10 ou superior
- Git instalado
- Pelo menos 4GB de RAM (para embeddings e LLM)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/gabrielsilveira2004/git_rag.git
   cd git_rag
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Construir o Índice (Vector Store)

Execute a ingestão completa para preparar os dados:

```bash
python scripts/ingest_git.py
```

Este script:
- Clona/atualiza o repositório Git
- Carrega e chunk os documentos
- Constrói o vector store em `data/vectorstore/`

### 2. Executar a API

Inicie o servidor FastAPI:

```bash
uvicorn app.api.main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`

### 3. Testar a API

- **Página inicial**: `GET /` - Mensagem de boas-vindas
- **Documentação interativa**: `GET /docs` - Interface Swagger UI para testar endpoints
- **Chat**: `POST /chat` - Envie perguntas sobre Git

Exemplo de request para `/chat`:

```json
{
  "question": "Como funciona o git commit?",
  "top_k": 4
}
```

Resposta:
```json
{
  "answer": "O comando git commit salva as mudanças no repositório local...",
  "intent": "procedural",
  "sources": [
    {
      "file": "Documentation/git-commit.txt",
      "snippet": "git-commit - Record changes to the repository..."
    }
  ]
}
```

### 4. Testes

Execute os testes automatizados:

```bash
python -m pytest tests/  # ou
python tests/ingest_test.py
python tests/chunk_test.py
python tests/vectorstore_test.py
```

## API Documentation

### Endpoints

- `GET /`: Status da API
- `POST /chat`: Chat com a documentação Git
  - **Body**: `{"question": "string", "top_k": 4}`
  - **Response**: `{"answer": "string", "intent": "string", "sources": [...]}`

A documentação completa está disponível em `/docs` quando o servidor estiver rodando.

## Desenvolvimento

Para contribuir:

1. Fork o repositório
2. Crie uma branch para sua feature: `git checkout -b feature/nova-funcionalidade`
3. Faça commits: `git commit -m "Adiciona nova funcionalidade"`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).