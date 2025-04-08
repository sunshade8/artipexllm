# EmpathyArtRAG

EmpathyArtRAG is an advanced AI system that combines conversational intelligence with emotion-aware artwork recommendations, creating a uniquely empathetic user experience.

## 🌟 Key Features

- **Emotion Intelligence**: Automatically detects emotions from user messages and conversation context
- **Smart Retrieval**: Adaptively decides when to retrieve information based on the type of query
- **Artwork Recommendations**: Suggests relevant artwork that resonates with the user's emotional state
- **Flexible Architecture**: Modular design with a configurable workflow graph for easy customization

## 🧠 How It Works

The system processes user interactions through a sophisticated workflow:

1. **Initial Processing**: Analyzes the user's question and extracts key information
2. **Emotion Detection**: Identifies emotions present in the message using keyword analysis
3. **Query Routing**: Determines the optimal path based on question type and emotional content
4. **Contextual Retrieval**: Fetches relevant information when factual responses are needed
5. **Art Recommendation**: Selects artwork that complements the detected emotions
6. **Response Generation**: Creates a thoughtful answer that addresses both informational needs and emotional context

## 📁 Project Structure

```
empathy_art_rag/
├── art/                    # Artwork recommendation components
│   └── artwork_recommender.py
├── generators/             # Response generation modules
│   ├── answer_generator.py
│   └── art_generator.py
├── handlers/               # Query processing systems
│   ├── emotion_handler.py
│   ├── question_handler.py
│   ├── retrieval_handler.py
│   └── routing_handler.py
├── models.py               # Data models and structures
├── workflow.py             # Workflow definition and control
├── config.py               # System configuration
└── main.py                 # Main application interface
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Access to artwork dataset (CSV file and image folder)
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/EmpathyArtRAG.git
   cd EmpathyArtRAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Copy `.env.example` to `.env` and update with your settings:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred editor
   ```

   Key settings to configure:
   - `ART_CSV_PATH`: Path to the artwork metadata CSV
   - `ART_IMAGE_FOLDER`: Path to the folder containing artwork images
   - `MODEL_NAME`: Name of the language model to use

## 💻 Usage Examples

### Basic Usage

```python
from empathy_art_rag import process_query

# Process a simple query
response = process_query(
    query="I'm feeling overwhelmed by work today",
    conversation_id=None  # New conversation
)

# Display the response
print(response.answer)

# Check if artwork is recommended
if response.artwork:
    print("\nRecommended Artwork:")
    print(f"Title: {response.artwork['title']}")
    print(f"Artist: {response.artwork['artist']}")
    print(f"Description: {response.artwork['description']}")
```

### Running the Demo

The project includes a demo script for easy testing:

```bash
# Run example conversation
python demo.py

# Run interactive demo
python demo.py --interactive
```

### Web Interface

A web application is available for visual interaction:

```bash
# Start the web application
python web_app.py
```

Then visit `http://localhost:8000` in your browser.

## 🔧 Customization

The system is designed to be easily extended:

- **Add emotions**: Expand the emotion detection by adding new keywords in `emotion_handler.py`
- **Update artwork**: Add new artwork to the dataset to expand recommendations
- **Modify workflow**: Adjust the workflow graph in `workflow.py` to change processing logic
- **Change LLM**: Integrate with different language models by updating `config.py`

## 📊 Performance Notes

- First-time queries may be slower as the system caches artwork embeddings
- Subsequent interactions with similar emotional content will be faster
- The system is optimized for conversational flow rather than high-volume processing

## 🔍 Troubleshooting

Common issues and solutions:

- **Missing artwork**: Verify the paths in `.env` are correct and the CSV file contains valid entries
- **Slow responses**: Check if the CLIP model is loading correctly and embeddings are being cached
- **Undetected emotions**: Try using more explicit emotional language or add new emotion keywords

## 📝 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

This project integrates components from EmotionArtSystem and adaptive-rag-ko-improved systems, enhanced with custom emotion detection and artwork recommendation algorithms. 