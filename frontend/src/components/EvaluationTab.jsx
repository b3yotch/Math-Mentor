import React, { useState } from 'react';
import {
  Settings, Play, AlertTriangle, CheckCircle,
  BarChart3, ChevronDown, ChevronUp,
} from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';
import LoadingSpinner from './LoadingSpinner';

export default function EvaluationTab() {
  const [topicFilter, setTopicFilter] = useState('');
  const [maxCases, setMaxCases] = useState(10);
  const [includeRag, setIncludeRag] = useState(true);
  const [includeSolutions, setIncludeSolutions] = useState(true);
  const [includeGuardrails, setIncludeGuardrails] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedComponent, setExpandedComponent] = useState(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.runBatchEvaluation(topicFilter, maxCases);
      setResult(data);
      if (data.status === 'success') {
        toast.success(`Evaluation complete: ${data.overall_grade}`);
      }
    } catch (err) {
      setError(err.message);
      toast.error('Evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  const gradeColor = (score) => {
    if (score >= 70) return 'var(--accent-green)';
    if (score >= 50) return 'var(--accent-yellow)';
    return 'var(--accent-red)';
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Configuration */}
      <div className="card">
        <div className="card-header">
          <Settings size={20} color="var(--accent-blue)" />
          <div>
            <div className="card-title">Evaluation Configuration</div>
            <div className="card-subtitle">Test system performance against known math problems</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
          {/* Topic filter */}
          <div className="input-group">
            <label className="input-label">Topic Filter</label>
            <select
              className="select"
              value={topicFilter}
              onChange={(e) => setTopicFilter(e.target.value)}
            >
              <option value="">All Topics</option>
              <option value="algebra">Algebra</option>
              <option value="calculus">Calculus</option>
              <option value="probability">Probability</option>
              <option value="statistics">Statistics</option>
              <option value="linear_algebra">Linear Algebra</option>
              <option value="trigonometry">Trigonometry</option>
            </select>
          </div>

          {/* Max cases */}
          <div className="input-group">
            <label className="input-label">Max Test Cases: {maxCases}</label>
            <input
              type="range"
              min={1}
              max={26}
              value={maxCases}
              onChange={(e) => setMaxCases(parseInt(e.target.value))}
            />
          </div>
        </div>

        {/* Component toggles */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
          <CheckboxToggle
            label="RAG Retrieval"
            checked={includeRag}
            onChange={setIncludeRag}
          />
          <CheckboxToggle
            label="Solutions"
            checked={includeSolutions}
            onChange={setIncludeSolutions}
          />
          <CheckboxToggle
            label="Guardrails"
            checked={includeGuardrails}
            onChange={setIncludeGuardrails}
          />
        </div>

        <button
          className="btn btn-primary btn-lg"
          onClick={handleRun}
          disabled={loading}
          style={{ width: '100%' }}
        >
          <Play size={16} />
          {loading ? 'Running Evaluation...' : 'Run Evaluation'}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="card slide-up">
          <LoadingSpinner text="Running evaluation against test dataset..." />
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="card slide-up" style={{ borderColor: 'var(--accent-red)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--accent-red)' }}>
            <AlertTriangle size={20} />
            <div>
              <div style={{ fontWeight: 600 }}>Evaluation Failed</div>
              <div style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 4 }}>{error}</div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Overall Score Card */}
          <div className="card" style={{ borderColor: gradeColor(result.overall_score) }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              flexWrap: 'wrap', gap: 16,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                {/* Score ring */}
                <div style={{
                  width: 72, height: 72, borderRadius: '50%',
                  border: `3px solid ${gradeColor(result.overall_score)}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexDirection: 'column',
                }}>
                  <span style={{
                    fontSize: 22, fontWeight: 700,
                    color: gradeColor(result.overall_score),
                  }}>
                    {result.overall_score?.toFixed(0)}%
                  </span>
                </div>

                <div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Grade: {result.overall_grade}
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
                    {result.test_cases} test cases
                    {result.topic_filter ? ` • ${result.topic_filter}` : ' • All topics'}
                    {result.total_time_ms ? ` • ${(result.total_time_ms / 1000).toFixed(1)}s` : ''}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <BarChart3 size={16} color="var(--text-muted)" />
                <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                  {Object.keys(result.components || {}).length} components evaluated
                </span>
              </div>
            </div>
          </div>

          {/* Component Results */}
          {result.components && Object.entries(result.components).map(([name, comp]) => (
            <div key={name} className="card">
              <button
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  width: '100%', background: 'none', border: 'none',
                  color: 'var(--text-primary)', cursor: 'pointer',
                  padding: 0, fontFamily: 'inherit',
                }}
                onClick={() => setExpandedComponent(expandedComponent === name ? null : name)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <CheckCircle
                    size={18}
                    color={gradeColor(comp.overall_score)}
                  />
                  <span style={{ fontSize: 15, fontWeight: 600 }}>{name}</span>
                  <span style={{
                    fontSize: 13, fontWeight: 600,
                    color: gradeColor(comp.overall_score),
                  }}>
                    {comp.overall_score?.toFixed(1)}%
                  </span>
                </div>
                {expandedComponent === name
                  ? <ChevronUp size={16} color="var(--text-muted)" />
                  : <ChevronDown size={16} color="var(--text-muted)" />
                }
              </button>

              {expandedComponent === name && (
                <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {/* Metrics grid */}
                  {comp.metrics && comp.metrics.length > 0 && (
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${Math.min(comp.metrics.length, 4)}, 1fr)`,
                      gap: 12,
                    }}>
                      {comp.metrics.map((metric, i) => (
                        <div key={i} style={{
                          padding: 12,
                          borderRadius: 'var(--radius-sm)',
                          background: 'var(--bg-tertiary)',
                        }}>
                          <div style={{
                            fontSize: 11, color: 'var(--text-muted)',
                            textTransform: 'uppercase', letterSpacing: 0.5,
                            marginBottom: 6,
                          }}>
                            {metric.name}
                          </div>
                          <div style={{
                            fontSize: 20, fontWeight: 700,
                            color: gradeColor(metric.percentage),
                          }}>
                            {metric.percentage?.toFixed(1)}%
                          </div>
                          {metric.details && (
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                              {metric.details}
                            </div>
                          )}
                          {/* Progress bar */}
                          <div style={{
                            height: 4, borderRadius: 2,
                            background: 'var(--bg-hover)',
                            marginTop: 8, overflow: 'hidden',
                          }}>
                            <div style={{
                              height: '100%', borderRadius: 2,
                              background: gradeColor(metric.percentage),
                              width: `${Math.min(metric.percentage, 100)}%`,
                              transition: 'width 0.6s ease',
                            }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Errors */}
                  {comp.errors && comp.errors.length > 0 && (
                    <div style={{
                      padding: 12,
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--accent-red-bg)',
                      border: '1px solid rgba(239, 68, 68, 0.2)',
                    }}>
                      <div style={{
                        fontSize: 12, fontWeight: 600,
                        color: 'var(--accent-red)', marginBottom: 6,
                      }}>
                        ⚠️ {comp.errors.length} error(s)
                      </div>
                      {comp.errors.slice(0, 5).map((err, i) => (
                        <div key={i} style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                          ❌ {err}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CheckboxToggle({ label, checked, onChange }) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 8,
      cursor: 'pointer', fontSize: 14, color: 'var(--text-secondary)',
    }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: 'var(--accent-blue)' }}
      />
      {label}
    </label>
  );
}