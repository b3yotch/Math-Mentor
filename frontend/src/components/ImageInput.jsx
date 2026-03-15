import React, { useState, useRef } from 'react';
import {
  Camera, Upload, X, Image as ImageIcon, Edit3,
  Check, RotateCcw, AlertTriangle, Loader,
} from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';

export default function ImageInput({ onSolve, loading }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  // HITL extraction state
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState(null); // { extracted_text, confidence }
  const [editedText, setEditedText] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  const handleFile = (f) => {
    if (!f) return;
    const allowed = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!allowed.includes(f.type)) {
      toast.error('Please upload a JPG, PNG, or WebP image.');
      return;
    }
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);

    // Reset extraction state when a new file is selected
    setExtraction(null);
    setEditedText('');
    setIsEditing(false);
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
    setExtraction(null);
    setEditedText('');
    setIsEditing(false);
    if (inputRef.current) inputRef.current.value = '';
  };

  // Step 1: Extract text from image (HITL)
  const handleExtract = async () => {
    if (!file) return;
    setExtracting(true);
    try {
      const result = await api.extractImage(file);
      setExtraction(result);
      setEditedText(result.extracted_text || '');
      if (!result.extracted_text || !result.extracted_text.trim()) {
        toast.error('Could not extract text. Try a clearer image.');
      } else {
        toast.success('Text extracted! Review and edit below.');
      }
    } catch (err) {
      toast.error(err.message || 'Extraction failed');
    } finally {
      setExtracting(false);
    }
  };

  // Step 2: Solve with (possibly edited) text
  const handleSolveExtracted = () => {
    if (!editedText.trim()) {
      toast.error('Please enter or edit the math problem text.');
      return;
    }

    const wasEdited = extraction && editedText !== extraction.extracted_text;
    const confidence = extraction ? extraction.confidence : 1.0;

    // Pass HITL metadata through to SolveTab via onSolve
    onSolve(editedText, {
      inputType: 'image',
      confidence,
      wasHumanEdited: wasEdited,
    });
  };

  const handleRevert = () => {
    if (extraction) {
      setEditedText(extraction.extracted_text);
      setIsEditing(false);
    }
  };

  const confidenceColor = (val) => {
    if (val >= 0.7) return 'var(--accent-green)';
    if (val >= 0.5) return 'var(--accent-yellow)';
    return 'var(--accent-red)';
  };

  const confidenceLabel = (val) => {
    if (val >= 0.7) return 'High';
    if (val >= 0.5) return 'Medium';
    return 'Low';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Upload zone */}
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
            JPG, PNG, or WebP — Clear photo of a math problem
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
        <div>
          {/* Image preview */}
          <div style={{ position: 'relative', marginBottom: 12 }}>
            <img
              src={preview}
              alt="Uploaded math problem"
              style={{
                width: '100%',
                maxHeight: 260,
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
                position: 'absolute', top: 8, right: 8,
                background: 'rgba(0,0,0,0.6)', borderRadius: '50%',
              }}
            >
              <X size={16} color="white" />
            </button>
          </div>

          {/* File info */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <ImageIcon size={14} color="var(--text-muted)" />
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
              {file.name} ({(file.size / 1024).toFixed(0)} KB)
            </span>
          </div>

          {/* Extraction not done yet: show Extract button */}
          {!extraction && !extracting && (
            <button
              className="btn btn-primary btn-lg"
              onClick={handleExtract}
              disabled={loading}
              style={{ width: '100%' }}
            >
              <Camera size={16} />
              Extract Text (OCR)
            </button>
          )}

          {/* Extracting spinner */}
          {extracting && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: 10, padding: 20, color: 'var(--text-muted)',
            }}>
              <Loader size={20} className="spin" />
              <span>Running OCR...</span>
            </div>
          )}

          {/* Extraction complete: HITL review panel */}
          {extraction && (
            <div style={{
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
            }}>
              {/* Confidence bar */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Extracted Text
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    fontSize: 12,
                    color: confidenceColor(extraction.confidence),
                    fontWeight: 600,
                  }}>
                    {confidenceLabel(extraction.confidence)} confidence ({(extraction.confidence * 100).toFixed(0)}%)
                  </span>
                  {extraction.confidence < 0.7 && (
                    <AlertTriangle size={14} color="var(--accent-yellow)" />
                  )}
                </div>
              </div>

              {/* Low confidence warning */}
              {extraction.confidence < 0.7 && (
                <div style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--accent-yellow-bg)',
                  border: '1px solid rgba(245, 158, 11, 0.2)',
                  fontSize: 12,
                  color: 'var(--accent-yellow)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}>
                  <AlertTriangle size={14} />
                  Low confidence — please review and correct the text below
                </div>
              )}

              {/* Editable text area */}
              <textarea
                className="textarea"
                value={editedText}
                onChange={(e) => {
                  setEditedText(e.target.value);
                  setIsEditing(true);
                }}
                rows={4}
                placeholder="Extracted math problem text..."
                style={{ fontSize: 14 }}
              />

              {/* Edit indicator */}
              {isEditing && editedText !== extraction.extracted_text && (
                <div style={{
                  fontSize: 12, color: 'var(--accent-cyan)',
                  display: 'flex', alignItems: 'center', gap: 4,
                }}>
                  <Edit3 size={12} />
                  Text has been edited
                </div>
              )}

              {/* Action buttons */}
              <div style={{ display: 'flex', gap: 8 }}>
                {isEditing && editedText !== extraction.extracted_text && (
                  <button
                    className="btn btn-ghost"
                    onClick={handleRevert}
                    style={{ flex: '0 0 auto' }}
                  >
                    <RotateCcw size={14} /> Revert
                  </button>
                )}

                <button
                  className="btn btn-ghost"
                  onClick={handleExtract}
                  disabled={extracting}
                  style={{ flex: '0 0 auto' }}
                >
                  <Camera size={14} /> Re-extract
                </button>

                <button
                  className="btn btn-primary btn-lg"
                  onClick={handleSolveExtracted}
                  disabled={loading || !editedText.trim()}
                  style={{ flex: 1 }}
                >
                  <Check size={16} />
                  {loading ? 'Solving...' : 'Solve Problem'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}