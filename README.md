# PDF Explorer + RAG (ChatPDF)

An interactive Streamlit web app that lets you explore, search, summarize, and ask questions about any PDF—combining traditional document browsing with a Retrieval‑Augmented Generation (RAG) pipeline powered by Together AI and HuggingFace embeddings.

---

## Objective & Problem Statement

**Problem:**  
Professionals in research, legal, education, and other domains often wrestle with long, multi‑modal PDF documents containing text, tables, and images. Finding specific details or understanding complex figures requires manual skimming, which is time‑consuming and error‑prone.

**Objective:**  
Build a user‑friendly web application that:

1. **Extracts** and **previews** text blocks, tables, and images from any PDF.
2. **Summarizes** each block via an LLM, so users get concise overviews of text passages and table contents.
3. **Indexes** all content into a vector database for fast semantic search.
4. **Answers** natural‑language questions about the document by retrieving relevant snippets and generating context‑aware responses.

---

## 🛠 How It Works

1. **PDF Partitioning**  
   - Uses [Unstructured](https://github.com/Unstructured-IO/unstructured) + Poppler’s `pdfinfo` to split a PDF into:
     - **Text blocks**  
     - **Table elements**  
     - **Embedded images**

2. **Summarization & Description**  
   - **Text & Tables**: Sent to Together AI’s “Mixtral-8x7B-Instruct” model (via a `/v1/chat/completions` REST call) for concise summaries.  
   - **Images**: Treated as “placeholders” with a fixed prompt asking the model to imagine a typical diagram (e.g., transformer architecture).

3. **Vector Indexing**  
   - Transforms summaries into embeddings using HuggingFace’s **all‑MiniLM‑L6‑v2** model.  
   - Stores embeddings in a [Chroma](https://github.com/chroma-core/chroma) vector store.  
   - Keeps an in‑memory mapping from each embedding back to the original content (text, HTML table, or base64 image).

4. **Retrieval‑Augmented Generation (RAG)**  
   - **Retrieval function**: Given a user query, finds the top‑k most semantically similar summaries.  
   - **Answer function**: Builds a prompt combining retrieved text summaries and image‑placeholder descriptions, then calls Together AI again to generate a concise answer, including source references.

5. **Streamlit UI**  
   - **Sidebar**:  
     - Upload or select a default PDF  
     - Search within raw text blocks  
     - Browse and preview tables and images  
     - Ask a natural‑language question  
   - **Main area**: Displays search results, table/dataframe previews, images, and AI‑generated Q&A answers with source context.

---

## Tech Stack & Dependencies

- **Python 3.9+**  
- **Streamlit** for the web UI  
- **Unstructured** + **Poppler** (`pdfinfo`) for PDF partitioning  
- **Pandas** & **Pillow** for table and image handling  
- **Together AI** REST API (`/v1/chat/completions`) for all LLM calls  
- **Requests** for HTTP communication  
- **LangChain** + **Chroma** + **HuggingFaceEmbeddings** for vector indexing & retrieval  
- **Sentence‑Transformers** (`all‑MiniLM‑L6‑v2`) for free, on‑device embeddings  

---

## Installation & Usage

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your‑username/pdf-explorer-rag.git
   cd pdf-explorer-rag
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your Together AI API key**
   ```bash
   export TOGETHER_API_KEY="your_together_api_key"      # macOS/Linux
   setx TOGETHER_API_KEY "your_together_api_key"       # Windows (then open a new PowerShell)
   ```
4. **Run the app**
   ```bash
   streamlit cache clear
   streamlit run streamlit_app.py
   ```
5. **Explore!**
   * Upload a PDF or use the default
   * Search text, browse tables & images
   * Ask questions and get instant AI‑powered answers
  
## Challenges & Solutions

- **Multimodal Summaries:** Together AI doesn’t directly accept images → solved by prompting the model to “imagine” standard diagram conventions.
- **Empty‑list Embeddings:** Chroma errors when upserting ```[]``` → added guards to skip empty modalities.
- **Streamlit Caching:** Inner functions aren’t pickleable → cached only the retriever object and regenerated lambda functions on demand.

## Future Work

- **True Vision Support:** Integrate GPT‑4 Vision or an open‑source vision‑LLM (e.g., BLIP) to analyze actual image pixels.
- **Persistent Storage:** Swap InMemoryStore for a persistent database (e.g., Redis, PostgreSQL) so indexes survive server restarts.
- **Incremental Updates:** Allow users to add new documents to the existing index without a full rebuild.
- **Authentication & Deployment:** Add user authentication and deploy to a cloud platform (Heroku, AWS, Streamlit Cloud).
- **Enhanced UI/UX:** Provide interactive chart previews, highlight answer spans in the original PDF, and allow export of Q&A transcripts.
