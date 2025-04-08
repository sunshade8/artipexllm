#!/usr/bin/env python3
"""
Web application for the EmpathyArtRAG system.

This script provides a simple web interface for interacting with the
EmpathyArtRAG system using FastAPI and HTML templates.
"""

import os
import json
import uuid
import uvicorn
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request, Form, Cookie, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from empathy_art_rag import process_query
from empathy_art_rag.config import ART_IMAGE_FOLDER

# Create FastAPI app
app = FastAPI(
    title="EmpathyArtRAG Web Interface",
    description="A web interface for the EmpathyArtRAG system",
    version="0.1.0"
)

# Setup template and static directories
# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# In-memory conversation store
conversations: Dict[str, List[Dict[str, Any]]] = {}

# Request/response models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ArtworkInfo(BaseModel):
    title: str
    artist: str
    date: str
    image_path: Optional[str] = None
    description: Optional[str] = None
    emotion: str

class QueryResponse(BaseModel):
    answer: str
    conversation_id: str
    detected_emotion: Optional[str] = None
    artwork: Optional[ArtworkInfo] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, conversation_id: Optional[str] = Cookie(None)):
    """Render the home page."""
    # Create a new conversation ID if none exists
    if not conversation_id or conversation_id not in conversations:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []
    
    # Get conversation history
    history = conversations.get(conversation_id, [])
    
    # Render the template
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "conversation_id": conversation_id,
            "history": history
        }
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return a response."""
    # Get or create conversation
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to conversation history
    conversations[conversation_id].append({
        "role": "user",
        "content": request.query
    })
    
    # Process the query
    response = process_query(
        query=request.query,
        conversation_id=conversation_id
    )
    
    # Add assistant message to conversation history
    conversations[conversation_id].append({
        "role": "assistant",
        "content": response.answer
    })
    
    # Extract artwork info if present
    artwork = None
    if response.artwork:
        art_data = response.artwork["artwork"]
        
        # Construct image path if an image folder is configured
        image_path = None
        if ART_IMAGE_FOLDER and "Object ID" in art_data:
            object_id = art_data["Object ID"]
            # Try common image extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                potential_path = os.path.join(ART_IMAGE_FOLDER, f"{object_id}{ext}")
                if os.path.exists(potential_path):
                    image_path = f"/images/{object_id}{ext}"
                    break
        
        artwork = ArtworkInfo(
            title=art_data.get("Title", "Untitled"),
            artist=art_data.get("Artist Display Name", "Unknown Artist"),
            date=art_data.get("Object Date", "Unknown date"),
            image_path=image_path,
            description=art_data.get("General_Text_Description", ""),
            emotion=response.artwork["emotion"]
        )
    
    # Construct response
    return QueryResponse(
        answer=response.answer,
        conversation_id=conversation_id,
        detected_emotion=response.metadata.get("detected_emotion"),
        artwork=artwork
    )

@app.get("/clear-conversation")
async def clear_conversation(conversation_id: Optional[str] = Cookie(None)):
    """Clear the current conversation history."""
    if conversation_id and conversation_id in conversations:
        conversations[conversation_id] = []
    
    return RedirectResponse(url="/")

# Mount art images if available
if ART_IMAGE_FOLDER and os.path.exists(ART_IMAGE_FOLDER):
    app.mount("/images", StaticFiles(directory=ART_IMAGE_FOLDER), name="images")

# Create template file
@app.on_event("startup")
async def create_templates():
    """Create necessary template files if they don't exist."""
    # Create index.html template
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmpathyArtRAG</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7f9;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #3498db;
            color: white;
            padding: 15px 0;
            text-align: center;
        }
        .chat-container {
            display: flex;
            height: calc(100vh - 200px);
            margin-top: 20px;
        }
        .chat {
            flex: 2;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            background-color: white;
        }
        .artwork-display {
            flex: 1;
            margin-left: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            background-color: white;
            padding: 20px;
            overflow-y: auto;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 18px;
            max-width: 70%;
        }
        .user {
            background-color: #e1f5fe;
            align-self: flex-end;
            margin-left: auto;
        }
        .assistant {
            background-color: #f1f1f1;
            align-self: flex-start;
        }
        .chat-input {
            display: flex;
            padding: 15px;
            border-top: 1px solid #eee;
        }
        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .chat-input button {
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            margin-left: 10px;
            cursor: pointer;
            font-size: 16px;
        }
        .artwork-info {
            margin-top: 20px;
        }
        .artwork-image {
            width: 100%;
            max-height: 300px;
            object-fit: contain;
            margin-bottom: 15px;
        }
        .emotion-tag {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            margin-top: 5px;
        }
        .actions {
            margin-top: 20px;
            text-align: right;
        }
        .actions a {
            color: #3498db;
            text-decoration: none;
            margin-left: 15px;
        }
        .hidden {
            display: none;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <header>
        <h1>EmpathyArtRAG System</h1>
        <p>Emotion-Aware Conversational AI with Artwork Recommendations</p>
    </header>
    
    <div class="container">
        <div class="chat-container">
            <div class="chat">
                <div class="chat-messages" id="messageContainer">
                    {% for msg in history %}
                        <div class="message {{ msg.role }}">
                            {{ msg.content | replace('\\n', '<br>') | safe }}
                        </div>
                    {% endfor %}
                </div>
                <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Type your message here..." />
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <div class="artwork-display" id="artworkDisplay">
                <h2>Artwork Recommendations</h2>
                <p id="noArtworkMessage">Artwork recommendations will appear here when available.</p>
                <div id="artworkInfo" class="artwork-info hidden">
                    <img id="artworkImage" class="artwork-image" src="" alt="Artwork" />
                    <h3 id="artworkTitle"></h3>
                    <p id="artworkArtist"></p>
                    <p id="artworkDate"></p>
                    <p id="artworkDescription"></p>
                    <div id="emotionTag" class="emotion-tag"></div>
                </div>
            </div>
        </div>
        
        <div class="actions">
            <a href="/clear-conversation">Clear Conversation</a>
        </div>
    </div>

    <script>
        // Store the conversation ID
        const conversationId = "{{ conversation_id }}";
        
        // Send a message to the server
        async function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (message) {
                // Add user message to UI
                addMessage('user', message);
                
                // Clear input
                messageInput.value = '';
                
                try {
                    // Send to server
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            query: message,
                            conversation_id: conversationId
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        
                        // Add assistant response to UI
                        addMessage('assistant', data.answer);
                        
                        // Update artwork display if available
                        if (data.artwork) {
                            displayArtwork(data.artwork);
                        }
                    } else {
                        addMessage('assistant', 'Sorry, there was an error processing your request.');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('assistant', 'Sorry, there was an error processing your request.');
                }
            }
        }
        
        // Add a message to the UI
        function addMessage(role, content) {
            const messageContainer = document.getElementById('messageContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = content.replace(/\\n/g, '<br>');
            messageContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messageContainer.scrollTop = messageContainer.scrollHeight;
        }
        
        // Display artwork in the sidebar
        function displayArtwork(artwork) {
            // Hide no artwork message
            document.getElementById('noArtworkMessage').style.display = 'none';
            
            // Show artwork info
            const artworkInfo = document.getElementById('artworkInfo');
            artworkInfo.classList.remove('hidden');
            
            // Update artwork details
            document.getElementById('artworkTitle').textContent = artwork.title;
            document.getElementById('artworkArtist').textContent = `By ${artwork.artist}`;
            document.getElementById('artworkDate').textContent = artwork.date;
            
            if (artwork.description) {
                document.getElementById('artworkDescription').textContent = artwork.description;
            } else {
                document.getElementById('artworkDescription').textContent = '';
            }
            
            document.getElementById('emotionTag').textContent = artwork.emotion;
            
            // Update image if available
            const artworkImage = document.getElementById('artworkImage');
            if (artwork.image_path) {
                artworkImage.src = artwork.image_path;
                artworkImage.style.display = 'block';
            } else {
                artworkImage.style.display = 'none';
            }
        }
        
        // Event listener for pressing Enter in the input field
        document.getElementById('messageInput').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""
    
    # Create index.html file if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    index_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write(index_html)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the EmpathyArtRAG web application")
    parser.add_argument(
        "--host", 
        type=str,
        default="127.0.0.1",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    args = parser.parse_args()
    
    print(f"Starting web application at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port) 