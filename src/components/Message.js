// src/components/Message.js
import React from 'react';

function Message({ message }) {
  const { text, sender, image } = message;
  
  return (
    <div className={`message ${sender}-message`}>
      <div className="message-content">
        {image && (
          <div className="message-image">
            <img src={image} alt="User uploaded" />
          </div>
        )}
        {text && <p>{text}</p>}
      </div>
    </div>
  );
}

export default Message;