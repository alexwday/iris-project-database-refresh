# Iris Project Database Refresh Pipelines

This project contains automated pipelines for processing various data file types into databases compatible with the Iris project's agentic retrieval system. Each folder represents a distinct method tailored to specific input data characteristics.

## Methods

### `catalog search - small`
-   **Input:** Standard documents (e.g., PDF, DOCX).
-   **Process:** Summarizes documents at the document level only.
-   **Retrieval:** The agent selects relevant documents based on these summaries and retrieves the full document content.
-   **Contains:** The Python scripts for this pipeline.

### `catalog search - large`
-   **Input:** Large documents with distinct sections or chapters.
-   **Process:** Summarizes documents at both the document level and the section/chapter level.
-   **Retrieval:** A two-stage agentic process. First, selects relevant documents based on document summaries. Second, selects specific relevant sections/chapters based on their summaries to retrieve, optimizing for large content.
-   **Contains:** (Currently empty)

### `catalog search - vision`
-   **Input:** Infographics or image-based documents.
-   **Process:** Uses a vision model (e.g., Qwen2) to preprocess the image into Markdown (MD) format. This MD content is then processed using the `catalog search - small` method.
-   **Retrieval:** Same as `catalog search - small`.
-   **Contains:** (Currently empty)

### `catalog search - excel`
-   **Input:** Excel spreadsheets (.xlsx).
-   **Process:** Preprocesses the Excel file, converting each relevant row (or group of rows) from the table into individual Markdown (MD) files. These MD files are then processed using the `catalog search - small` method.
-   **Retrieval:** Same as `catalog search - small`.
-   **Contains:** (Currently empty)

### `semantic search - large`
-   **Input:** Very large documents, like textbooks.
-   **Process:** Breaks down the document into chapters/sections, then further chunks the text within these sections. Embeddings are generated for these chunks.
-   **Retrieval:** Uses similarity search on the chunks, followed by reranking, to retrieve the most relevant text snippets as context, bypassing the agentic catalog search.
-   **Contains:** (Currently empty)
