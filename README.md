# ArtiTech: Adaptive Artwork Recommendation System


## 📊 LangGraph Visualization
![LangGraph Visualization](Artitech_LangGraph.png)

## Overview

This project implements an adaptive Retrieval-Augmented Generation (RAG) system combined with an artwork recommender. The primary goal is to create an AI assistant that can engage in empathetic conversations, understand user emotions, and suggest relevant artworks or creative activities to support emotional well-being.

The system leverages LangChain and LangGraph to build a flexible and stateful workflow. It uses CLIP embeddings for multimodal understanding (text and image) to power artwork recommendations and integrates various techniques like web search, document retrieval, and emotion detection to provide contextually relevant and supportive responses.

## Features

*   **Adaptive RAG:** Uses LangGraph to create a dynamic workflow that routes queries based on context, detected emotion, and intermediate results (e.g., document relevance).
*   **Emotion Detection:** Analyzes user input to detect emotions from a predefined list, influencing the conversation flow and art suggestions.
*   **Artwork Recommendation:**
    *   Utilizes OpenAI's CLIP model to find artworks semantically related to emotional prompts.
    *   Incorporates color analysis (dominant colors) to enhance recommendation relevance.
    *   Uses predefined mappings between emotions, colors, subjects, and styles.
    *   Supports efficient similarity search using FAISS (if installed).
*   **Hybrid Retrieval:** Employs an `EnsembleRetriever` combining keyword search (BM25) and semantic search (FAISS) for robust document retrieval from PDF sources.
*   **Web Search Capability:** Integrates Tavily Search for answering queries outside the scope of local documents.
*   **Hallucination & Relevance Grading:** Includes steps to check if generated responses are grounded in retrieved context and relevant to the user's query.
*   **Conversation Memory:** Maintains conversation history for context-aware interactions within a session.
*   **Configurable:** Key parameters like model names, file paths, and feature flags are managed via `src/config.py`.
*   **Caching:** Caches generated artwork embeddings to speed up initialization on subsequent runs.

## Project Structure

```
.
├── data/                     # Data files for the project
│   ├── art_csv/              # CSV files with artwork metadata
│   │   └── painting_info_with_descriptions.csv  # Example artwork data file
│   ├── docs/                 # PDF documents for the RAG system
│   │   └── ... (your PDF files)
│   └── images/               # Artwork image files (named by Object ID)
│       └── ... (e.g., 12345.jpg)
├── notebooks/                # Jupyter notebooks for experimentation and interaction
│   └── artitech_final.ipynb  # Main notebook demonstrating the system
├── scripts/                  # Helper scripts
│   ├── emotion/              # Scripts related to emotion processing and graph routing
│   │   ├── __init__.py
│   │   ├── art_recommendation_connector.py
│   │   ├── art_response_handler.py
│   │   ├── art_suggestion.py
│   │   ├── emotion_detection.py
│   │   └── multi_turn_router.py
│   ├── __init__.py
│   └── fix_api_key.py        # Utility for setting up API keys
├── src/                      # Source code for the core application logic
│   ├── __init__.py
│   ├── artwork_recommender.py # Core class for artwork embeddings and recommendations
│   ├── chain.py              # Functions for creating LangChain RAG chains
│   ├── config.py             # Configuration settings (paths, models, flags)
│   ├── data_processing.py    # PDF loading and splitting utilities
│   ├── display_utils.py      # Utilities for displaying recommendations
│   ├── emotion_nodes.py      # LangGraph nodes specific to emotion detection
│   ├── grading.py            # LangChain components for grading relevance/hallucinations
│   ├── graph_builder.py      # Builds the main LangGraph workflow
│   ├── graph_nodes.py        # Core LangGraph nodes (retrieve, generate, check, etc.)
│   ├── prompts.py            # Definition of various system and user prompts
│   ├── recommender_setup.py  # Initialization logic for the ArtworkRecommender
│   ├── retriever.py          # Functions for creating retrievers (Ensemble)
│   ├── rewriting.py          # LangChain component for query rewriting
│   ├── routing.py            # LangChain components for routing decisions
│   ├── state.py              # Definition of the GraphState TypedDict
│   └── tools.py              # Web search tool setup (Tavily)
├── emotion_art_cache/        # Default directory for cached embeddings (auto-created)
├── .env                      # API keys (user needs to create this)
├── requirements.txt          # Project dependencies (user should generate/verify)
└── README.md                 # This file
```

## Setup and Installation

1.  **Prerequisites:**
    *   Python (3.9+ recommended)
    *   Git
    *   Access to an environment where you can install packages (virtual environment recommended).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your actual repository URL
    cd <your-project-directory>     # Navigate to the project root
    ```

3.  **Set up a Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    *   A `requirements.txt` file is highly recommended for reproducibility. If one is not present, you might need to install the core libraries manually based on the imports in the `.py` files. Key dependencies include:
        ```
        langchain langchain-openai langgraph langchain-community
        pandas numpy torch transformers pillow scikit-learn tqdm
        faiss-cpu # Or faiss-gpu if you have CUDA setup
        matplotlib python-dotenv tavily-python
        # Add any other specific libraries identified from imports
        ```
    *   Install using pip:
        ```bash
        pip install -r requirements.txt
        # Or install individually: pip install langchain langchain-openai ...
        ```
    *   **Note on FAISS:** FAISS (`faiss-cpu` or `faiss-gpu`) provides significant speedup for artwork recommendations but is optional. The system falls back to standard (slower) similarity search if FAISS is not installed.

5.  **API Key Configuration:**
    *   **CRITICAL:** This project requires API keys for OpenAI and Tavily Search to function correctly. You **must** provide your own keys.
    *   Create a file named `.env` in the project's root directory (if it doesn't exist).
    *   Add your API keys to this `.env` file in the following format:
        ```dotenv
        OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        TAVILY_API_KEY="tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```
    *   Replace `sk-...` and `tvly-...` with your actual API keys.
    *   **Never commit your `.env` file to Git.** The included `.gitignore` file should prevent this automatically.
    *   The `scripts/fix_api_key.py` script (specifically the `setup_api_keys` function called in the notebook) handles the process of reading this `.env` file and setting the corresponding environment variables (`OPENAI_API_KEY`, `TAVILY_API_KEY`) for the application to use. The application will fail with an error message if the required `OPENAI_API_KEY` is missing or invalid.

6.  **Data Setup:**
    *   **Artwork Metadata:** Place your artwork metadata CSV (e.g., `painting_info_with_descriptions.csv`) inside the `data/art_csv/` directory. Ensure it contains an `Object ID` column (or modify `src/artwork_recommender.py` accordingly).
    *   **Artwork Images:** Place the corresponding artwork image files inside the `data/images/` directory. Images **must** be named according to their `Object ID` from the CSV file (e.g., if `Object ID` is `12345`, the image should be named `12345.jpg`). The default expected extension is `.jpg`.
    *   **Text Documents:** Place any PDF documents you want the RAG system to use for context into the `data/docs/` directory.

## Usage

The primary way to interact with and demonstrate the system is through the Jupyter Notebook (`notebooks/artitech_final.ipynb`).

1.  **Start Jupyter:**
    ```bash
    jupyter notebook
    # Or use Jupyter Lab:
    # jupyter lab
    ```
2.  **Open the Notebook:** Navigate to the `notebooks` directory and open `artitech_final.ipynb`.
3.  **Run the Cells:** Execute the notebook cells sequentially. The notebook guides you through:
    *   Importing libraries and setting up API keys.
    *   Loading and processing PDF documents and artwork data.
    *   Initializing the `ArtworkRecommender` (which includes generating/loading embeddings).
    *   Setting up the RAG components (retriever, graders, prompts, LLM).
    *   Building the `LangGraph` application (`adaptive_rag_app`).
    *   Running example queries through the graph.
    *   Displaying the final responses and any generated artwork recommendations.
    Running the notebook cells allows users to interactively test the system and observe the step-by-step execution of the RAG and recommendation processes.

## Configuration

Several key parameters can be adjusted in `src/config.py`:

*   `RECOMMENDER_MODEL_NAME`: The specific CLIP model used for artwork embeddings (e.g., `"openai/clip-vit-large-patch14"`).
*   `CSV_PATH`: Path to the artwork metadata CSV file.
*   `IMAGE_FOLDER`: Path to the directory containing artwork images.
*   `CACHE_DIR`: Directory for storing cached embeddings.
*   Font settings for Matplotlib CJK character support.

Other configurations like retriever weights (`src/retriever.py`) or LLM parameters (model name, temperature in `src/chain.py` and `src/grading.py` factories) can also be adjusted within the respective files or during component initialization in the notebook.

## Technical Details

*   **Core Framework:** LangChain & LangGraph
*   **LLMs:** OpenAI models (GPT-4o, GPT-3.5-turbo are common choices, configurable)
*   **Embeddings:** OpenAI Embeddings (for text RAG), CLIP (for multimodal artwork recommendation via `transformers`)
*   **Vector Search:** FAISS (optional, for artwork), built-in LangChain vector stores.
*   **Keyword Search:** BM25 (via LangChain)
*   **Web Search:** Tavily Search API
*   **Data Handling:** Pandas
*   **Image Processing:** Pillow, NumPy
*   **Visualization:** Matplotlib
