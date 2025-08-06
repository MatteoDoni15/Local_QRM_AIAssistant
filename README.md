# QRM AI Assistant

This project provides a complete pipeline for analyzing, extracting, and indexing data from PDF and Excel documents into Azure Cognitive Search, with a PyQt6 desktop frontend.

## Project Structure

- **[QRMGui.py](QRMGui.py)**  
  PyQt6 desktop GUI for uploading, analyzing, and indexing documents.
- **[AIProcessPdfUploadIndexFunction.py](AIProcessPdfUploadIndexFunction.py)**  
  Core logic for processing PDFs and Excel files: OCR (Mistral), text/image/table extraction, language and keyword detection, embedding generation, and indexing to Azure Search.
- **[blob_storage_manager.py](blob_storage_manager.py)**  
  Utilities for uploading/downloading files and managing metadata on Azure Blob Storage.
- **[call_azure_function_processing_pdf.py](call_azure_function_processing_pdf.py)**  
  Example for calling Azure Functions for PDF processing.
- **[collections_json_schema_str.py](collections_json_schema_str.py)**  
  JSON schemas for structured data extraction.
- **[index_indexer_manager.py](index_indexer_manager.py)**  
  Management and triggering of Azure Cognitive Search indexers.
- **[list_questions.py](list_questions.py)**  
  Lists of questions for automated document analysis.
- **[questioning.py](questioning.py)**  
  Functions for sending questions and receiving answers from LLM/PromptFlow.
- **[questions_editor_dialog.py](questions_editor_dialog.py)**  
  PyQt6 dialog for editing question lists.
- **[requirements.txt](requirements.txt)**  
  Python dependencies.
- **[.env](.env)**  
  Environment variables (Azure endpoints, keys, configuration).

## Main Features

- **Upload and storage** of PDF/XLSX files to Azure Blob Storage.
- **Text, image, and table extraction** from PDFs (including OCR with Mistral).
- **Excel to Markdown conversion** and data preparation for AI Search.
- **Vector embedding** of content using Azure OpenAI.
- **Indexing** into Azure Cognitive Search.
- **Automated analysis** via LLM (PromptFlow/OpenAI).
- **Desktop frontend** for batch management and result visualization.
- **PromptFlow integration:** Receives and displays JSON responses from Azure AI Foundry (GPT-4.1).

## Requirements

- Python >= 3.9 for running this application (recommended to use the latest supported version).
- The Azure Function for file processing currently requires Python 3.9.
- Azure Subscription (Blob Storage, Cognitive Search, OpenAI)
- See `requirements.txt` for required Python libraries.

## Setup

1. Copy `.env.example` to `.env` and fill in your Azure keys/endpoints.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Start the frontend:
   ```sh
   python QRMGui.py
   ```

## Data Files

- **questions.xlsx**  
  External Excel file containing all the analysis questions, organized by group.  
  This file is not versioned in the repository for privacy reasons. You must create or provide it before running the application.

- **prompts.json**  
  External JSON file containing the system prompts and instructions for the LLM.  
  This file is not versioned in the repository for privacy reasons. You must create or provide it before running the application.

**Note:**  
Both files must be placed in the same folder as the executable or main script.  
Sample templates can be provided upon request.

## Usage Examples

- Upload one or more PDF/XLSX files from the frontend and start batch indexing.
- Run automated analyses (Private Equity, Private Debt, BPM, Prosp Analysis).
- View results and JSON responses from PromptFlow/Azure AI Foundry directly in the application.

## Notes

- Some features require active and configured Azure services.
- The application does **not** save results in HTML format; PromptFlow responses are handled as JSON.

---

For details on individual functions, see the docstrings in the Python files.