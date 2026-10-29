# AI Study Assistant — Hugging Face Pipelines

A Streamlit application for text summarization and context-based question answering.

## Features
- Summarize a long passage with a pretrained Transformer.
- Control summary length.
- Process longer documents in word-based chunks.
- Ask questions using a supplied context passage.
- View answer confidence scores and question history.
- Export results as JSON or TXT.
- Input validation and user-friendly error messages.

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first model download
- RAM and disk space for the pretrained models

## Installation (Windows)
Open Command Prompt or the VS Code terminal inside this project folder:

```bash
python -m pip install -r requirements.txt
```

If `python` is not recognized, try `py -m pip install -r requirements.txt`.

## Run
```bash
python -m streamlit run app.py
```

## Folder structure
```text
ai_study_assistant/
├── app.py
├── requirements.txt
└── README.md
```

## How it works
1. Streamlit collects the document and settings.
2. Hugging Face's `summarization` pipeline generates a shorter version.
3. The `question-answering` pipeline extracts an answer from the supplied context.
4. The app calculates word reduction and records questions in session state.
5. Users can download a JSON data file or readable TXT report.

## Important notes
- The first run downloads model files and can take time.
- The app uses pretrained models; output quality depends on the input and model.
- QA confidence is not a guarantee of correctness.
- Word-based chunks are a practical demonstration, not exact tokenizer-aware chunking.
- For production use, add tokenizer-aware chunking, document upload support, stronger exception logging, and tests.
