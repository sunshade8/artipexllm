// src/App.js
import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import Message from './components/Message';
import Spinner from './components/Spinner';
import DrawingCanvas from './components/DrawingCanvas';
// Import MUI icons
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import SendIcon from '@mui/icons-material/Send';
import SaveIcon from '@mui/icons-material/Save';
import DeleteIcon from '@mui/icons-material/Delete';
import UndoIcon from '@mui/icons-material/Undo';

// *** IMPORTANT: Replace with your deployed Render backend URL ***
const API_ENDPOINT = "https://artipexllm-backend-abcd.onrender.com/chat"; // Replace this placeholder

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [canvasImage, setCanvasImage] = useState(null);
  const messageEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  
  // API key should be stored in environment variables, not in source code
  // const OPENAI_API_KEY = process.env.REACT_APP_OPENAI_API_KEY;

  // Brush settings
  const [brushColor, setBrushColor] = useState("#000000");
  const [brushRadius, setBrushRadius] = useState(3);
  const [canvasWidth, setCanvasWidth] = useState(800);
  const [canvasHeight, setCanvasHeight] = useState(500);
  
  // Drawing history for undo functionality
  const [drawingHistory, setDrawingHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Scroll to bottom of messages
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Calculate canvas size on mount
  useEffect(() => {
    function updateCanvasSize() {
      // Adjust canvas size based on container size
      const container = document.querySelector('.canvas-wrapper');
      if (container) {
        const containerWidth = container.clientWidth;
        // Make canvas wider but keep reasonable height
        setCanvasWidth(Math.min(800, containerWidth - 20)); // subtract padding
        setCanvasHeight(Math.min(500, window.innerHeight * 0.5));
      }
    }
    
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);

  // Handle sending a message
  const handleSend = async () => {
    if ((!input.trim() && !uploadedImage && !canvasImage) || loading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: input,
      image: uploadedImage || canvasImage,
      sender: 'user'
    };
    
    setMessages([...messages, userMessage]);
    setInput('');
    setUploadedImage(null);
    setCanvasImage(null);
    setLoading(true);

    try {
      // Prepare the request payload (same as before)
      const apiPayload = {
        history: messages.map(m => ({
          role: m.sender,
          text: m.text,
          image: m.image || null
        })),
        text: input,
        image: uploadedImage || canvasImage || null
      };
      
      console.log("Sending payload to API:", apiPayload);

      // Make API call to the deployed backend
      const response = await fetch(API_ENDPOINT, { // Use the new API_ENDPOINT
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiPayload)
      });

      // Log raw response for debugging
      const rawResponse = await response.text();
      console.log("Raw API response:", rawResponse);
      
      let data;
      try {
        data = JSON.parse(rawResponse);
        console.log("Parsed API response:", data);
      } catch (parseError) {
        console.error("Error parsing JSON response:", parseError);
        throw new Error(`Failed to parse response: ${rawResponse}`);
      }
      
      if (!response.ok) {
        // Attempt to get error message from backend response
        const errorMsg = data?.detail || data?.error?.message || 'Error communicating with the backend';
        throw new Error(errorMsg);
      }

      if (!data.choices || !data.choices[0] || !data.choices[0].message || 
          !data.choices[0].message.content) {
        console.error("Unexpected API response format:", data);
        throw new Error("The response format from the backend is incorrect");
      }

      const botMessage = {
        id: Date.now(),
        text: data.choices[0].message.content,
        sender: 'bot'
      };
      
      setMessages(prevMessages => [...prevMessages, botMessage]);

    } catch (error) {
      console.error('Error in handleSend:', error);
      const errorMessage = {
        id: Date.now(),
        text: `Sorry, I encountered an error: ${error.message}`,
        sender: 'bot'
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Handle file upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Only accept image files
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setUploadedImage(reader.result);
      setCanvasImage(null); // Clear canvas image when uploading new image
    };
    reader.readAsDataURL(file);
  };

  // Handle saving canvas drawing
  const handleSaveDrawing = () => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const dataURL = canvas.toDataURL('image/png');
      setCanvasImage(dataURL);
      
      // Save to drawing history
      const newHistory = [...drawingHistory.slice(0, historyIndex + 1), dataURL];
      setDrawingHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
      
      alert('Drawing saved! Click send to include it in your message.');
    }
  };

  // Handle clearing canvas
  const handleClearCanvas = () => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      setCanvasImage(null);
      
      // Save cleared state to history
      const clearedCanvas = canvas.toDataURL('image/png');
      const newHistory = [...drawingHistory, clearedCanvas];
      setDrawingHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
  };

  // Handle undo last canvas action
  const handleUndoCanvas = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      const previousState = drawingHistory[historyIndex - 1];
      
      // Load previous state
      const canvas = document.querySelector('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0);
      };
      img.src = previousState;
      
      // Update canvas image if needed
      setCanvasImage(previousState);
    }
  };

  // Initialize canvas with white background and save to history
  const handleCanvasInit = (canvas) => {
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const initialState = canvas.toDataURL('image/png');
      setDrawingHistory([initialState]);
      setHistoryIndex(0);
    }
  };

  // Trigger file input click
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Remove uploaded image
  const removeImage = () => {
    setUploadedImage(null);
    setCanvasImage(null);
  };

  // Color options for drawing
  const colorOptions = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"];

  return (
    <div className="app">
      <header className="app-header">
        <h1>IPEXLLM</h1>
      </header>
      
      <div className="main-container">
        {/* Drawing Canvas Section - Now takes 2/3 of the screen */}
        <div className="canvas-container">
          <h2>Draw Something</h2>
          <div className="canvas-controls">
            <div className="brush-controls">
              <div className="color-picker">
                {colorOptions.map((color) => (
                  <div
                    key={color}
                    className={`color-option ${color === brushColor ? 'selected' : ''}`}
                    style={{ backgroundColor: color }}
                    onClick={() => setBrushColor(color)}
                  />
                ))}
              </div>
              <div className="brush-size">
                <label>Size: </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={brushRadius}
                  onChange={(e) => setBrushRadius(parseInt(e.target.value))}
                />
              </div>
            </div>
            <div className="canvas-actions">
              <button className="canvas-btn save-btn" onClick={handleSaveDrawing} title="Save Drawing">
                <SaveIcon />
              </button>
              <button className="canvas-btn clear-btn" onClick={handleClearCanvas} title="Clear Canvas">
                <DeleteIcon />
              </button>
              <button 
                className="canvas-btn undo-btn" 
                onClick={handleUndoCanvas} 
                title="Undo Last Action"
                disabled={historyIndex <= 0}
              >
                <UndoIcon />
              </button>
            </div>
          </div>
          <div className="canvas-wrapper">
            <DrawingCanvas 
              width={canvasWidth}
              height={canvasHeight}
              brushColor={brushColor}
              brushRadius={brushRadius}
              onSave={(dataUrl) => setCanvasImage(dataUrl)}
            />
          </div>
          {canvasImage && (
            <div className="canvas-saved">
              <p>Drawing saved and ready to send!</p>
            </div>
          )}
        </div>
        
        {/* Chat Container - Now takes 1/3 of the screen */}
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="empty-chat">
                <p>No messages yet. Start a conversation!</p>
              </div>
            )}
            
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            
            {loading && <Spinner />}
            
            <div ref={messageEndRef} />
          </div>
          
          <div className="input-container">
            {(uploadedImage || canvasImage) && (
              <div className="image-preview">
                <img src={uploadedImage || canvasImage} alt="Upload preview" />
                <button className="remove-image-btn" onClick={removeImage}>×</button>
              </div>
            )}
            
            <div className="input-controls">
              <button 
                className="upload-btn" 
                onClick={triggerFileInput}
                disabled={loading}
              >
                <CameraAltIcon />
              </button>
              
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                style={{ display: 'none' }}
                accept="image/*"
              />
              
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message..."
                disabled={loading}
              />
              
              <button 
                className="send-btn" 
                onClick={handleSend}
                disabled={(!input.trim() && !uploadedImage && !canvasImage) || loading}
              >
                <SendIcon />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;