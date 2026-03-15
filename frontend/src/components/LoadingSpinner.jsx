import React from 'react';

export default function LoadingSpinner({ text = 'Loading...', size = 40 }) {
  return (
    <div className="loading-overlay">
      <div className="spinner" style={{ width: size, height: size }} />
      <div className="loading-text">{text}</div>
    </div>
  );
}