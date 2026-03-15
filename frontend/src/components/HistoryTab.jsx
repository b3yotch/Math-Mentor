import React, { useState, useEffect } from 'react';
import {
  Clock, FileText, Camera, Mic, ChevronDown,
  ChevronUp, Search, Filter, RefreshCw,
} from 'lucide-react';
import api from '../api/client';
import LoadingSpinner from './LoadingSpinner';
import ReactMarkdown from 'react-markdown';

const INPUT_ICONS = {
  text: FileText,
  image: Camera,
  audio: Mic,
};

export default function HistoryTab() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [limit, setLimit] = useState(10);
  const [topicFilter, setTopicFilter] = useState('');
  const [expandedId, setExpandedId] = useState(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getHistory(limit, topicFilter);
      setHistory(data.problems || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [limit, topicFilter]);

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <Clock size={20} color="var(--accent-blue)" />
          <div>
            <div className="card-title">Problem History</div>
            <div className="card-subtitle">Previously solved problems and solutions</div>
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <div className="input-group" style={{ flex: 1, minWidth: 200 }}>
            <label className="input-label">
              <Filter size={12} style={{ display: 'inline', marginRight: 4 }} />
              Topic Filter
            </label>
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

          <div className="input-group" style={{ width: 120 }}>
            <label className="input-label">Show</label>
            <select
              className="select"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value))}
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button className="btn btn-secondary" onClick={fetchHistory}>
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Loading */}
      {loading && <LoadingSpinner text="Loading history..." />}

      {/* Error */}
      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
          <p style={{ color: 'var(--accent-red)' }}>Error: {error}</p>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && history.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
          <Clock size={48} color="var(--text-muted)" style={{ marginBottom: 12 }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: 16 }}>
            No problems solved yet
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 4 }}>
            Start by entering a math problem in the Solve tab
          </p>
        </div>
      )}

      {/* History List */}
      {!loading && history.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {history.map((problem, index) => {
            const Icon = INPUT_ICONS[problem.input_type] || FileText;
            const isExpanded = expandedId === problem.id;

            return (
              <div key={problem.id || index} className="collapsible">
                <button
                  className="collapsible-header"
                  onClick={() => toggleExpand(problem.id)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 0 }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: 8,
                      background: 'var(--bg-primary)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      flexShrink: 0,
                    }}>
                      <Icon size={16} color="var(--accent-blue)" />
                    </div>

                    <div style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
                      <div style={{
                        fontSize: 14, fontWeight: 500,
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      }}>
                        {problem.question || 'Unknown problem'}
                      </div>
                      <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                        {problem.topic && (
                          <span className="badge badge-purple" style={{ fontSize: 11 }}>
                            {problem.topic}
                          </span>
                        )}
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {problem.created_at}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <ConfidenceBadge value={problem.confidence} />
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                </button>

                {isExpanded && (
                  <div className="collapsible-body">
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                      <div>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Input Type</span>
                        <p style={{ fontSize: 14, fontWeight: 500 }}>{problem.input_type}</p>
                      </div>
                      <div>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Confidence</span>
                        <p style={{ fontSize: 14, fontWeight: 500 }}>{(problem.confidence * 100).toFixed(0)}%</p>
                      </div>
                    </div>

                    <div style={{ marginBottom: 16 }}>
                      <span style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                        Problem
                      </span>
                      <div className="math-block">{problem.question}</div>
                    </div>

                    <div>
                      <span style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                        Solution
                      </span>
                      <div className="solution-content" style={{ fontSize: 13 }}>
                        <ReactMarkdown>
                          {problem.solution_preview || 'No solution recorded'}
                        </ReactMarkdown>
                      </div>
                    </div>

                    {problem.was_human_edited && (
                      <div style={{ marginTop: 12 }}>
                        <span className="badge badge-orange">Human Edited</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ConfidenceBadge({ value }) {
  const pct = (value * 100).toFixed(0);
  const cls = value >= 0.7 ? 'badge-green' : value >= 0.5 ? 'badge-orange' : 'badge-red';
  return <span className={`badge ${cls}`} style={{ fontSize: 11 }}>{pct}%</span>;
}