import React, { useState, useRef } from 'react';
import { Camera, Upload, X, Image as ImageIcon } from 'lucide-react';

export default function ImageInput({ onSolve, loading }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (f) => {
    if (!f) return;
    const allowed = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!allowed.includes(f.type)) {
      alert('Please upload a JPG, PNG, or WebP image.');
      return;
    }
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {!preview ? (
        <div
          className={`upload-zone ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <Camera size={48} color="var(--text-muted)" style={{ marginBottom: 12 }} />
          <p style={{ fontSize: 16, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Drop an image here or click to upload
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            JPG, PNG, or WebP - Clear photo of a math problem
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      ) : (
        <div style={{ position: 'relative' }}>
          <img
            src={preview}
            alt="Uploaded math problem"
            style={{
              width: '100%',
              maxHeight: 300,
              objectFit: 'contain',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-color)',
              background: 'var(--bg-primary)',
            }}
          />
          <button
            className="btn btn-ghost btn-icon"
            onClick={handleClear}
            style={{
              position: 'absolute',
              top: 8,
              right: 8,
              background: 'rgba(0,0,0,0.6)',
              borderRadius: '50%',
            }}
          >
            <X size={16} color="white" />
          </button>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <ImageIcon size={14} color="var(--text-muted)" />
              <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                {file.name} ({(file.size / 1024).toFixed(0)} KB)
              </span>
            </div>
            <button
              className="btn btn-primary btn-lg"
              onClick={() => onSolve(file)}
              disabled={loading}
            >
              <Upload size={16} />
              {loading ? 'Processing...' : 'Extract & Solve'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}