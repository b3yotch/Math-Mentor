import React, { useState, useRef } from 'react';
import {
  Mic, X, FileAudio, Square, Edit3,
  Check, RotateCcw, AlertTriangle, Loader,
} from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';

export default function AudioInput({ onSolve, loading }) {
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const inputRef = useRef(null);

  // HITL extraction state
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState(null);
  const [editedText, setEditedText] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
    setExtraction(null);
    setEditedText('');
    setIsEditing(false);
  };

  const handleClear = () => {
    setFile(null);
    setAudioUrl(null);
    setExtraction(null);
    setEditedText('');
    setIsEditing(false);
    if (inputRef.current) inputRef.current.value = '';
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([blob], 'recording.webm', { type: 'audio/webm' });
        handleFile(audioFile);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      toast.error('Microphone access denied. Please allow microphone permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  // Step 1: Transcribe audio (HITL)
  const handleTranscribe = async () => {
    if (!file) return;
    setExtracting(true);
    try {
      const result = await api.extractAudio(file);
      setExtraction(result);
      setEditedText(result.extracted_text || '');
      if (!result.extracted_text || !result.extracted_text.trim()) {
        toast.error('Could not transcribe audio. Please try again.');
      } else {
        toast.success('Audio transcribed! Review and edit below.');
      }
    } catch (err) {
      toast.error(err.message || 'Transcription failed');
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

    onSolve(editedText, {
      inputType: 'audio',
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
      {/* Upload or Record (when no file yet) */}
      {!file && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Upload */}
          <div
            className="upload-zone"
            onClick={() => inputRef.current?.click()}
            style={{ padding: 24 }}
          >
            <FileAudio size={36} color="var(--text-muted)" style={{ marginBottom: 8 }} />
            <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-secondary)' }}>
              Upload Audio
            </p>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              WAV, MP3, OGG, FLAC
            </p>
            <input
              ref={inputRef}
              type="file"
              accept="audio/*"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>

          {/* Record */}
          <div
            className="upload-zone"
            onClick={recording ? stopRecording : startRecording}
            style={{
              padding: 24,
              borderColor: recording ? 'var(--accent-red)' : 'var(--border-color)',
              background: recording ? 'rgba(239, 68, 68, 0.05)' : undefined,
            }}
          >
            {recording ? (
              <>
                <Square size={36} color="var(--accent-red)" style={{ marginBottom: 8 }} />
                <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--accent-red)' }}>
                  Recording...
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Click to stop</p>
              </>
            ) : (
              <>
                <Mic size={36} color="var(--text-muted)" style={{ marginBottom: 8 }} />
                <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-secondary)' }}>
                  Record Audio
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Click to start</p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Audio file loaded — show preview + HITL flow */}
      {file && audioUrl && (
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
          padding: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}>
          {/* File info + clear button */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <FileAudio size={14} color="var(--accent-blue)" />
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                {file.name} ({(file.size / 1024).toFixed(0)} KB)
              </span>
            </div>
            <button className="btn btn-ghost btn-icon" onClick={handleClear}>
              <X size={14} />
            </button>
          </div>

          {/* Audio player */}
          <audio controls src={audioUrl} style={{ width: '100%' }} />

          {/* Transcribe button (Step 1) */}
          {!extraction && !extracting && (
            <button
              className="btn btn-primary btn-lg"
              onClick={handleTranscribe}
              disabled={loading}
              style={{ width: '100%' }}
            >
              <Mic size={16} />
              Transcribe Audio
            </button>
          )}

          {/* Transcribing spinner */}
          {extracting && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: 10, padding: 20, color: 'var(--text-muted)',
            }}>
              <Loader size={20} className="spin" />
              <span>Transcribing audio...</span>
            </div>
          )}

          {/* Transcription result: HITL review panel */}
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
              {/* Confidence */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Transcribed Text
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
                  Low confidence — please review and correct the transcription below
                </div>
              )}

              {/* Math conversion hints */}
              <div style={{
                padding: '6px 10px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--accent-blue-bg)',
                border: '1px solid rgba(59, 130, 246, 0.15)',
                fontSize: 11,
                color: 'var(--accent-blue)',
              }}>
                💡 Tip: You can edit math notation. E.g., "x squared" → x², "square root of x" → √x
              </div>

              {/* Editable text area */}
              <textarea
                className="textarea"
                value={editedText}
                onChange={(e) => {
                  setEditedText(e.target.value);
                  setIsEditing(true);
                }}
                rows={4}
                placeholder="Transcribed math problem text..."
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
                  onClick={handleTranscribe}
                  disabled={extracting}
                  style={{ flex: '0 0 auto' }}
                >
                  <Mic size={14} /> Re-transcribe
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