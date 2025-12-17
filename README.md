# Git RAG Ingest

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.1.3-green.svg)
![GitPython](https://img.shields.io/badge/GitPython-3.1.45-orange.svg)

## Descrição

Projeto para ingestão da documentação do Git e transformação em **Document objects** prontos para RAG (Retrieval-Augmented Generation).
Clona o repositório oficial do Git, processa arquivos `.adoc`, `.md` e `.txt`, e realiza chunking por seções.

## Funcionalidades

- Clona o repositório Git se não existir.
- Lê documentação (`.adoc`, `.md`, `.txt`) e converte em documentos para RAG.
- Chunking semântico por seções, preservando metadados como título, índice e texto.
- Armazena commit do Git para referência.
- Organiza e prepara os documentos para consulta com RAG.

## Estrutura do Projeto

```
git_rag/
├── app/rag/
│   ├── ingest.py    # Ingestão de documentos
│   ├── chunk.py     # Funções de chunking
│   └── ...
├── data/git_repo/   # Repositório clonado (não versionar)
├── scripts/         # Scripts utilitários
├── tests/           # Testes de partes do processo (ingestão de dados e chunking)
├── requirements.txt
└── .gitignore
```

## Pré-requisitos

- Python 3.11 ou superior
- Git instalado no sistema

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/git_rag.git
   cd git_rag
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual:
   - Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para executar a ingestão da documentação:

```bash
python app/rag/ingest.py
```

Isso irá clonar o repositório Git (se necessário) e processar os documentos.

## Contribuição

Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
