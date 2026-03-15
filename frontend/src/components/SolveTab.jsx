import React, { useState } from 'react';
import { Send, Sparkles, BookOpen, Lightbulb } from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';
import LoadingSpinner from './LoadingSpinner';
import SolutionDisplay from './SolutionDisplay';
import RAGContext from './RAGContext';
import EvaluationCard from './EvaluationCard';
import FeedbackSection from './FeedbackSection';

const EXAMPLES = [
  'Solve x^2 - 5x + 6 = 0',
  'Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3',
  'What is the probability of getting exactly 2 heads in 3 coin tosses?',
  'Find the determinant of matrix [[1,2],[3,4]]',
  'Integrate 3x^2 + 2x dx',
  'Find the limit of (x^2 - 1)/(x - 1) as x approaches 1',
  'A train travels at 60 km/h for 2 hours. Find the distance.',
];

export default function SolveTab({ settings }) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSolve = async () => {
    if (!question.trim()) {
      toast.error('Please enter a math problem');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const steps = [
      'Checking guardrails...',
      'Normalizing input...',
      'Searching knowledge base...',
      'Solving with AI...',
      'Validating response...',
      'Evaluating quality...',
    ];

    let stepIndex = 0;
    const stepInterval = setInterval(() => {
      if (stepIndex < steps.length) {
        setLoadingStep(steps[stepIndex]);
        stepIndex++;
      }
    }, 2000);

    try {
      const data = await api.solve(question, settings.topK, settings.includeEval);
      setResult(data);

      if (data.status === 'success') {
        toast.success('Problem solved!');
      } else if (data.status === 'blocked') {
        toast.error('Input blocked by guardrails');
      } else if (data.status === 'error') {
        toast.error(data.error || 'An error occurred');
      }
    } catch (err) {
      setError(err.message);
      toast.error(err.message);
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSolve();
    }
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Input Card */}
      <div className="card">
        <div className="card-header">
          <Sparkles size={20} color="var(--accent-blue)" />
          <div>
            <div className="card-title">Solve a Math Problem</div>
            <div className="card-subtitle">Enter any JEE-level math question</div>
          </div>
        </div>

        {/* Examples */}
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

        {/* Text Input */}
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
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Ctrl+Enter to solve
          </span>
          <button
            className="btn btn-primary btn-lg"
            onClick={handleSolve}
            disabled={loading || !question.trim()}
          >
            <Send size={18} />
            {loading ? 'Solving...' : 'Solve Problem'}
          </button>
        </div>
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
            <span style={{ fontSize: 24 }}>!</span>
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
          {result.pipeline_steps && result.pipeline_steps.length > 0 && (
            <div className="pipeline-steps">
              {result.pipeline_steps.map((step, i) => (
                <React.Fragment key={step}>
                  {i > 0 && <span className="pipeline-arrow">&#8594;</span>}
                  <span className="pipeline-step">{step}</span>
                </React.Fragment>
              ))}
              {result.latency_ms && (
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
                  {result.latency_ms.toFixed(0)}ms
                </span>
              )}
            </div>
          )}

          {/* Blocked Message */}
          {result.status === 'blocked' && (
            <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
              <h3 style={{ color: 'var(--accent-red)', marginBottom: 8 }}>Input Blocked</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
                {result.blocked_reason}
              </p>
              {result.suggestions && result.suggestions.length > 0 && (
                <div>
                  <p style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>Suggestions:</p>
                  <ul style={{ paddingLeft: 20, color: 'var(--text-secondary)', fontSize: 13 }}>
                    {result.suggestions.map((s, i) => <li key={i} style={{ marginBottom: 4 }}>{s}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Solution */}
          {result.status === 'success' && (
            <>
              {/* Topic Badge */}
              {result.detected_topic && (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <BookOpen size={14} color="var(--accent-purple)" />
                  <span className="badge badge-purple">{result.detected_topic}</span>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {result.rag_results_count} sources found
                  </span>
                </div>
              )}

              {/* RAG Context */}
              {settings.showRag && result.rag_sources && (
                <RAGContext sources={result.rag_sources} />
              )}

              {/* Main Solution */}
              <SolutionDisplay result={result} />

              {/* Evaluation */}
              {result.evaluation && (
                <EvaluationCard evaluation={result.evaluation} />
              )}

              {/* Feedback */}
              {result.problem_id && (
                <FeedbackSection problemId={result.problem_id} />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}