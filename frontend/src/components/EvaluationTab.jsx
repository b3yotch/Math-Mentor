import React, { useState } from 'react';
import {
  BarChart3, Play, Settings, Download,
  CheckCircle, XCircle, AlertTriangle,
} from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';
import LoadingSpinner from './LoadingSpinner';

export default function EvaluationTab() {
  const [config, setConfig] = useState({
    topic: '',
    maxCases: 10,
    includeRag: true,
    includeSolutions: true,
    includeGuardrails: true,
  });
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    setLoadingMsg('Starting evaluation...');

    const messages = [
      'Preparing test cases...',
      'Evaluating RAG retrieval...',
      'Testing solution accuracy...',
      'Checking guardrails...',
      'Generating report...',
    ];

    let msgIndex = 0;
    const interval = setInterval(() => {
      if (msgIndex < messages.length) {
        setLoadingMsg(messages[msgIndex]);
        msgIndex++;
      }
    }, 3000);

    try {
      const data = await api.runBatchEvaluation(config.topic, config.maxCases);
      setReport(data);
      toast.success('Evaluation complete!');
    } catch (err) {
      setError(err.message);
      toast.error('Evaluation failed');
    } finally {
      clearInterval(interval);
      setLoading(false);
      setLoadingMsg('');
    }
  };

  const downloadReport = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `eval_report_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Configuration */}
      <div className="card">
        <div className="card-header">
          <Settings size={20} color="var(--accent-blue)" />
          <div>
            <div className="card-title">Evaluation Configuration</div>
            <div className="card-subtitle">
              Test system performance against known math problems
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div className="input-group">
            <label className="input-label">Topic Filter</label>
            <select
              className="select"
              value={config.topic}
              onChange={(e) => setConfig({ ...config, topic: e.target.value })}
            >
              <option value="">All Topics</option>
              <option value="algebra">Algebra</option>
              <option value="calculus">Calculus</option>
              <option value="probability">Probability</option>
              <option value="statistics">Statistics</option>
              <option value="linear_algebra">Linear Algebra</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Max Test Cases: {config.maxCases}</label>
            <input
              type="range"
              min="5" max="26"
              value={config.maxCases}
              onChange={(e) => setConfig({ ...config, maxCases: parseInt(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: 16, marginBottom: 20 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={config.includeRag}
              onChange={(e) => setConfig({ ...config, includeRag: e.target.checked })}
              style={{ accentColor: 'var(--accent-blue)' }}
            />
            RAG Retrieval
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={config.includeSolutions}
              onChange={(e) => setConfig({ ...config, includeSolutions: e.target.checked })}
              style={{ accentColor: 'var(--accent-blue)' }}
            />
            Solutions
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={config.includeGuardrails}
              onChange={(e) => setConfig({ ...config, includeGuardrails: e.target.checked })}
              style={{ accentColor: 'var(--accent-blue)' }}
            />
            Guardrails
          </label>
        </div>

        <button
          className="btn btn-primary btn-lg btn-full"
          onClick={handleRun}
          disabled={loading}
        >
          <Play size={18} />
          {loading ? 'Running Evaluation...' : 'Run Evaluation'}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="card slide-up">
          <LoadingSpinner text={loadingMsg || 'Evaluating...'} />
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
          <p style={{ color: 'var(--accent-red)' }}>Error: {error}</p>
        </div>
      )}

      {/* Results */}
      {report && !loading && (
        <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Overall Score */}
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 40, alignItems: 'center', marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 48, fontWeight: 800, background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  {report.overall_score?.toFixed(1)}%
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Overall Score</div>
              </div>
              <div className={`grade-badge grade-${report.overall_grade}`} style={{ width: 64, height: 64, fontSize: 28 }}>
                {report.overall_grade}
              </div>
            </div>
            <div className="progress-bar" style={{ height: 8, marginBottom: 8 }}>
              <div
                className="progress-fill"
                style={{
                  width: `${report.overall_score || 0}%`,
                  background: report.overall_score >= 70 ? 'var(--accent-green)' : report.overall_score >= 50 ? 'var(--accent-orange)' : 'var(--accent-red)',
                }}
              />
            </div>
          </div>

          {/* Component Results */}
          {report.components && Object.entries(report.components).map(([name, comp]) => (
            <ComponentResult key={name} name={name} data={comp} />
          ))}

          {/* Download */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <button className="btn btn-secondary" onClick={downloadReport}>
              <Download size={16} /> Download Report (JSON)
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function ComponentResult({ name, data }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="card">
      <div
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <BarChart3 size={18} color="var(--accent-blue)" />
          <span style={{ fontWeight: 600 }}>{name}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{
            fontSize: 18, fontWeight: 700,
            color: data.score >= 70 ? 'var(--accent-green)' : data.score >= 50 ? 'var(--accent-orange)' : 'var(--accent-red)',
          }}>
            {data.score?.toFixed(1)}%
          </span>
        </div>
      </div>

      {expanded && data.metrics && (
        <div style={{ marginTop: 16 }}>
          <div className="metric-grid">
            {data.metrics.map((metric, i) => (
              <div key={i} className="metric-card">
                <div className="metric-value" style={{
                  fontSize: 20,
                  background: 'none',
                  WebkitTextFillColor: metric.percentage >= 70 ? 'var(--accent-green)' : metric.percentage >= 50 ? 'var(--accent-orange)' : 'var(--accent-red)',
                  color: metric.percentage >= 70 ? 'var(--accent-green)' : metric.percentage >= 50 ? 'var(--accent-orange)' : 'var(--accent-red)',
                }}>
                  {metric.percentage?.toFixed(1)}%
                </div>
                <div className="metric-label">{metric.name}</div>
              </div>
            ))}
          </div>

          {data.errors && data.errors.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--accent-orange)', marginBottom: 8 }}>
                <AlertTriangle size={14} />
                <span style={{ fontSize: 13, fontWeight: 500 }}>{data.errors.length} errors</span>
              </div>
              {data.errors.slice(0, 5).map((err, i) => (
                <p key={i} style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
                  {err}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}