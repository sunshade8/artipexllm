// src/components/DrawingCanvas.js
import React, { useRef, useState, useEffect } from 'react';

const DrawingCanvas = ({ width, height, brushColor, brushRadius, onSave }) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPosition, setLastPosition] = useState({ x: 0, y: 0 });
  const contextRef = useRef(null);
  
  // Initialize canvas when component mounts or when dimensions change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Set dimensions
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    
    // Set up context
    const context = canvas.getContext('2d');
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.strokeStyle = brushColor;
    context.lineWidth = brushRadius * 2;
    
    // Fill with white background
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    contextRef.current = context;
  }, [width, height]);
  
  // Update brush style when color or radius changes
  useEffect(() => {
    if (contextRef.current) {
      contextRef.current.strokeStyle = brushColor;
      contextRef.current.lineWidth = brushRadius * 2;
    }
  }, [brushColor, brushRadius]);
  
  // Start drawing
  const startDrawing = (e) => {
    const { offsetX, offsetY } = getCoordinates(e);
    contextRef.current.beginPath();
    contextRef.current.moveTo(offsetX, offsetY);
    setLastPosition({ x: offsetX, y: offsetY });
    setIsDrawing(true);
  };
  
  // Draw
  const draw = (e) => {
    if (!isDrawing) return;
    
    const { offsetX, offsetY } = getCoordinates(e);
    
    // Smooth line drawing with quadratic curves
    contextRef.current.quadraticCurveTo(
      lastPosition.x,
      lastPosition.y,
      (offsetX + lastPosition.x) / 2,
      (offsetY + lastPosition.y) / 2
    );
    
    setLastPosition({ x: offsetX, y: offsetY });
    contextRef.current.stroke();
  };
  
  // Stop drawing
  const stopDrawing = () => {
    if (!isDrawing) return;
    contextRef.current.closePath();
    setIsDrawing(false);
    
    // Save current state and notify parent if onSave provided
    if (onSave && canvasRef.current) {
      const dataUrl = canvasRef.current.toDataURL('image/png');
      onSave(dataUrl);
    }
  };
  
  // Handle touch events and mouse events coordinates
  const getCoordinates = (e) => {
    if (!canvasRef.current) return { offsetX: 0, offsetY: 0 };
    
    // For mouse events
    if (e.nativeEvent.offsetX !== undefined) {
      return { 
        offsetX: e.nativeEvent.offsetX, 
        offsetY: e.nativeEvent.offsetY 
      };
    } 
    // For touch events
    else if (e.touches && e.touches[0]) {
      const rect = canvasRef.current.getBoundingClientRect();
      return {
        offsetX: e.touches[0].clientX - rect.left,
        offsetY: e.touches[0].clientY - rect.top
      };
    } 
    else {
      return { offsetX: 0, offsetY: 0 };
    }
  };
  
  // Public methods that can be called from parent component
  React.useImperativeHandle(React.forwardRef((props, ref) => ref), () => ({
    clear: () => {
      if (!canvasRef.current || !contextRef.current) return;
      
      contextRef.current.fillStyle = 'white';
      contextRef.current.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    },
    
    getDataURL: () => {
      if (!canvasRef.current) return null;
      return canvasRef.current.toDataURL('image/png');
    }
  }));
  
  return (
    <canvas
      ref={canvasRef}
      onMouseDown={startDrawing}
      onMouseMove={draw}
      onMouseUp={stopDrawing}
      onMouseLeave={stopDrawing}
      onTouchStart={startDrawing}
      onTouchMove={draw}
      onTouchEnd={stopDrawing}
      style={{
        border: '1px solid #ccc',
        borderRadius: '4px',
        touchAction: 'none', // Prevents scrolling while drawing on touch devices
        display: 'block' // Removes extra space below canvas
      }}
    />
  );
};

export default DrawingCanvas;