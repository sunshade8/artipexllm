# Using the EmpathyArtRAG System

This guide explains how to run and test the EmpathyArtRAG system.

## Prerequisites

Before running the system, make sure you have completed the installation steps in README.md and have the following:

1. The artwork metadata CSV file at the path specified in `config.py`
2. The artwork image folder at the path specified in `config.py`
3. All the required Python dependencies installed

## Running the Demo

The system includes a CLI demo script that showcases its functionality:

```bash
# Run the demo with example conversations
python demo.py

# Run the demo in interactive mode
python demo.py --interactive
```

### Demo Options

- **Example mode**: Runs through a series of predefined examples to showcase the system's emotion detection and artwork recommendation capabilities.
- **Interactive mode**: Allows you to have a free-form conversation with the system and test its responses to your inputs.

## Running the Web Application

The system also includes a simple web interface:

```bash
# Run the web application on the default host and port (127.0.0.1:8000)
python web_app.py

# Run the web application with custom host and port
python web_app.py --host 0.0.0.0 --port 5000
```

### Web Interface Features

The web application offers a more user-friendly way to interact with the system:

1. Chat interface with conversation history
2. Artwork recommendations displayed in a sidebar
3. Conversation persistence between page refreshes
4. Option to clear conversation history

## Testing Emotion Detection

Try the following prompts to test the emotion detection:

- "I'm feeling really happy today!"
- "I've been sad lately and don't know what to do."
- "I'm anxious about my upcoming presentation."
- "I feel nostalgic when I look at old photographs."
- "I'm amazed by the beauty of nature."

## Testing Art Recommendations

Try these prompts to test the art recommendation functionality:

- "Can you recommend some art for me?"
- "I feel sad, what artwork might help?"
- "Show me art related to joy."
- "I need some peaceful artwork to calm my mind."
- "What artwork represents wonder and awe?"

## Troubleshooting

If you encounter issues:

1. **Artwork not displaying**: Check that the art image folder path is correct in `config.py` and that the images are named according to their Object IDs.

2. **Emotion not detected**: Try using explicit emotion words like "happy", "sad", "anxious", etc. in your queries.

3. **System errors**: Check the console output for error messages that may indicate configuration issues.

4. **Web interface not loading**: Make sure all required dependencies are installed, especially FastAPI, Uvicorn, and Jinja2.

## Extending the System

The modular design of EmpathyArtRAG makes it easy to extend:

- Add new emotions to `DETECTABLE_EMOTIONS` in `emotion_handler.py`
- Integrate different LLM providers by modifying `call_llm` in `answer_generator.py`
- Add new artwork data by updating the CSV file
- Extend the workflow graph in `workflow.py` with additional processing nodes 