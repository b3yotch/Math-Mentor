import React, { useState, useRef } from 'react';
import { Mic, Upload, X, FileAudio, Square } from 'lucide-react';

export default function AudioInput({ onSolve, loading }) {
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const inputRef = useRef(null);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
  };

  const handleClear = () => {
    setFile(null);
    setAudioUrl(null);
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
        setFile(audioFile);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      alert('Microphone access denied. Please allow microphone permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Upload or Record */}
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
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Click to stop
              </p>
            </>
          ) : (
            <>
              <Mic size={36} color="var(--text-muted)" style={{ marginBottom: 8 }} />
              <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-secondary)' }}>
                Record Audio
              </p>
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Click to start
              </p>
            </>
          )}
        </div>
      </div>

      {/* Preview */}
      {file && audioUrl && (
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
          padding: 16,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
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

          <audio controls src={audioUrl} style={{ width: '100%', marginBottom: 12 }} />

          <button
            className="btn btn-primary btn-lg btn-full"
            onClick={() => onSolve(file)}
            disabled={loading}
          >
            <Mic size={16} />
            {loading ? 'Transcribing...' : 'Transcribe & Solve'}
          </button>
        </div>
      )}
    </div>
  );
}