import React, { useState } from 'react';
import { Send, Sparkles, FileText, Camera, Mic } from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';
import LoadingSpinner from './LoadingSpinner';
import SolutionDisplay from './SolutionDisplay';
import RAGContext from './RAGContext';
import EvaluationCard from './EvaluationCard';
import FeedbackSection from './FeedbackSection';
import ImageInput from './ImageInput';
import AudioInput from './AudioInput';

const EXAMPLES = [
  'Solve x^2 - 5x + 6 = 0',
  'Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3',
  'What is the probability of getting exactly 2 heads in 3 coin tosses?',
  'Find the determinant of matrix [[1,2],[3,4]]',
  'Integrate 3x^2 + 2x dx',
  'A train travels at 60 km/h for 2 hours. Find the distance.',
];

const INPUT_MODES = [
  { id: 'text', label: 'Text', icon: FileText },
  { id: 'image', label: 'Image (OCR)', icon: Camera },
  { id: 'audio', label: 'Audio (ASR)', icon: Mic },
];

export default function SolveTab({ settings }) {
  const [inputMode, setInputMode] = useState('text');
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const startLoading = (mode) => {
    setLoading(true);
    setError(null);
    setResult(null);

    const steps = {
      text: ['Checking guardrails...', 'Searching knowledge base...', 'Solving with AI...', 'Evaluating quality...'],
      image: ['Checking guardrails...', 'Searching knowledge base...', 'Solving with AI...', 'Evaluating quality...'],
      audio: ['Checking guardrails...', 'Searching knowledge base...', 'Solving with AI...', 'Evaluating quality...'],
    };

    let idx = 0;
    const msgs = steps[mode || inputMode] || steps.text;
    const interval = setInterval(() => {
      if (idx < msgs.length) {
        setLoadingStep(msgs[idx]);
        idx++;
      }
    }, 2500);

    return interval;
  };

  // Text input: solve directly
  const handleSolveText = async () => {
    if (!question.trim()) {
      toast.error('Please enter a math problem');
      return;
    }
    const interval = startLoading('text');

    try {
      const data = await api.solve(question, settings.topK, settings.includeEval);
      setResult(data);
      if (data.status === 'success') toast.success('Problem solved!');
      else if (data.status === 'blocked') toast.error('Input blocked by guardrails');
    } catch (err) {
      setError(err.message);
      toast.error(err.message);
    } finally {
      clearInterval(interval);
      setLoading(false);
      setLoadingStep('');
    }
  };

  // Image HITL: receives extracted text + metadata from ImageInput
  const handleSolveImage = async (extractedText, hitlMeta = {}) => {
    if (!extractedText || !extractedText.trim()) {
      toast.error('No text to solve');
      return;
    }

    const interval = startLoading('image');

    try {
      const data = await api.solve(
        extractedText,
        settings.topK,
        settings.includeEval,
        {
          inputType: hitlMeta.inputType || 'image',
          confidence: hitlMeta.confidence || 1.0,
          wasHumanEdited: hitlMeta.wasHumanEdited || false,
        }
      );
      setResult(data);
      if (data.status === 'success') toast.success('Image problem solved!');
      else if (data.status === 'blocked') toast.error('Input blocked by guardrails');
    } catch (err) {
      setError(err.message);
      toast.error(err.message);
    } finally {
      clearInterval(interval);
      setLoading(false);
      setLoadingStep('');
    }
  };

  // Audio HITL: receives transcribed text + metadata from AudioInput
  const handleSolveAudio = async (extractedText, hitlMeta = {}) => {
    if (!extractedText || !extractedText.trim()) {
      toast.error('No text to solve');
      return;
    }

    const interval = startLoading('audio');

    try {
      const data = await api.solve(
        extractedText,
        settings.topK,
        settings.includeEval,
        {
          inputType: hitlMeta.inputType || 'audio',
          confidence: hitlMeta.confidence || 1.0,
          wasHumanEdited: hitlMeta.wasHumanEdited || false,
        }
      );
      setResult(data);
      if (data.status === 'success') toast.success('Audio problem solved!');
      else if (data.status === 'blocked') toast.error('Input blocked by guardrails');
    } catch (err) {
      setError(err.message);
      toast.error(err.message);
    } finally {
      clearInterval(interval);
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSolveText();
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Input Card */}
      <div className="card">
        <div className="card-header">
          <Sparkles size={20} color="var(--accent-blue)" />
          <div>
            <div className="card-title">Solve a Math Problem</div>
            <div className="card-subtitle">Text, image, or voice input</div>
          </div>
        </div>

        {/* Input Mode Tabs */}
        <div className="tabs" style={{ marginBottom: 20 }}>
          {INPUT_MODES.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`tab ${inputMode === id ? 'active' : ''}`}
              onClick={() => { setInputMode(id); setResult(null); setError(null); }}
            >
              <Icon size={16} /> {label}
            </button>
          ))}
        </div>

        {/* Text Input */}
        {inputMode === 'text' && (
          <>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  className="btn btn-ghost"
                  style={{ fontSize: 12, padding: '4px 10px', borderRadius: 9999 }}
                  onClick={() => setQuestion(ex)}
                >
                  {ex.length > 40 ? ex.slice(0, 40) + '...' : ex}
                </button>
              ))}
            </div>

            <textarea
              className="textarea"
              placeholder="e.g., Solve x^2 - 5x + 6 = 0"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={4}
              style={{ marginBottom: 16 }}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Ctrl+Enter to solve</span>
              <button className="btn btn-primary btn-lg" onClick={handleSolveText} disabled={loading || !question.trim()}>
                <Send size={18} /> {loading ? 'Solving...' : 'Solve Problem'}
              </button>
            </div>
          </>
        )}

        {/* Image Input — HITL: Extract → Edit → Solve */}
        {inputMode === 'image' && (
          <ImageInput onSolve={handleSolveImage} loading={loading} />
        )}

        {/* Audio Input — HITL: Transcribe → Edit → Solve */}
        {inputMode === 'audio' && (
          <AudioInput onSolve={handleSolveAudio} loading={loading} />
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="card slide-up">
          <LoadingSpinner text={loadingStep || 'Processing...'} />
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="card slide-up" style={{ borderColor: 'var(--accent-red)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--accent-red)' }}>
            <span style={{ fontSize: 24, fontWeight: 700 }}>!</span>
            <div>
              <div style={{ fontWeight: 600 }}>Error</div>
              <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>{error}</div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Pipeline Steps */}
          {result.pipeline_steps?.length > 0 && (
            <div className="pipeline-steps">
              {result.pipeline_steps.map((step, i) => (
                <React.Fragment key={step}>
                  {i > 0 && <span className="pipeline-arrow">&#8594;</span>}
                  <span className="pipeline-step">{step}</span>
                </React.Fragment>
              ))}
              {result.latency_ms > 0 && (
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
                  {result.latency_ms.toFixed(0)}ms
                </span>
              )}
            </div>
          )}

          {/* Extraction info for image/audio */}
          {result.input_type !== 'text' && (
            <div className="card" style={{ borderColor: 'rgba(6, 182, 212, 0.3)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                  {result.input_type === 'image' ? '📷 OCR' : '🎤 ASR'} Input
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {result.was_human_edited && (
                    <span className="badge badge-cyan" style={{ fontSize: 11 }}>✏️ Human Edited</span>
                  )}
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    Confidence: {(result.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="math-block">{result.question}</div>
            </div>
          )}

          {/* Blocked */}
          {result.status === 'blocked' && (
            <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
              <h3 style={{ color: 'var(--accent-red)', marginBottom: 8 }}>Input Blocked</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>{result.blocked_reason}</p>
              {result.suggestions?.length > 0 && (
                <ul style={{ paddingLeft: 20, color: 'var(--text-secondary)', fontSize: 13 }}>
                  {result.suggestions.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              )}
            </div>
          )}

          {/* Solution */}
          {result.status === 'success' && (
            <>
              {settings.showRag && result.rag_sources && <RAGContext sources={result.rag_sources} />}
              <SolutionDisplay result={result} />
              {result.evaluation && <EvaluationCard evaluation={result.evaluation} />}
              {result.problem_id && <FeedbackSection problemId={result.problem_id} />}
            </>
          )}
        </div>
      )}
    </div>
  );
}