import React, { useState } from 'react';
import { BarChart3, ChevronDown, ChevronUp, AlertTriangle, Lightbulb } from 'lucide-react';

export default function EvaluationCard({ evaluation }) {
  const [open, setOpen] = useState(false);

  if (!evaluation) return null;

  const gradeClass = `grade-${evaluation.grade}`;

  return (
    <div className="collapsible">
      <button className="collapsible-header" onClick={() => setOpen(!open)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <BarChart3 size={16} color="var(--accent-blue)" />
          <span>Response Quality</span>
          <div className={`grade-badge ${gradeClass}`} style={{ width: 32, height: 32, fontSize: 14 }}>
            {evaluation.grade}
          </div>
          <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            {(evaluation.overall_score * 100).toFixed(0)}%
          </span>
        </div>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div className="collapsible-body">
          {/* Score Grid */}
          <div className="metric-grid" style={{ marginBottom: 20 }}>
            <ScoreCard label="Overall" value={evaluation.overall_score} />
            <ScoreCard label="RAG Relevance" value={evaluation.rag_relevance} />
            <ScoreCard label="Solution" value={evaluation.solution_quality} />
            <ScoreCard label="Explanation" value={evaluation.explanation_clarity} />
          </div>

          {/* Assessments */}
          {evaluation.solution_assessment && (
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
              <strong>Solution:</strong> {evaluation.solution_assessment}
            </p>
          )}
          {evaluation.explanation_assessment && (
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
              <strong>Explanation:</strong> {evaluation.explanation_assessment}
            </p>
          )}

          {/* Issues */}
          {evaluation.issues && evaluation.issues.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--accent-orange)' }}>
                <AlertTriangle size={14} />
                <span style={{ fontSize: 13, fontWeight: 500 }}>Issues Found</span>
              </div>
              {evaluation.issues.map((issue, i) => (
                <p key={i} style={{ fontSize: 13, color: 'var(--text-secondary)', paddingLeft: 20, marginBottom: 4 }}>
                  - {issue}
                </p>
              ))}
            </div>
          )}

          {/* Suggestions */}
          {evaluation.suggestions && evaluation.suggestions.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--accent-cyan)' }}>
                <Lightbulb size={14} />
                <span style={{ fontSize: 13, fontWeight: 500 }}>Suggestions</span>
              </div>
              {evaluation.suggestions.map((sug, i) => (
                <p key={i} style={{ fontSize: 13, color: 'var(--text-secondary)', paddingLeft: 20, marginBottom: 4 }}>
                  - {sug}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ScoreCard({ label, value }) {
  const pct = ((value || 0) * 100).toFixed(0);
  const color = value >= 0.7 ? 'var(--accent-green)' : value >= 0.5 ? 'var(--accent-orange)' : 'var(--accent-red)';

  return (
    <div className="metric-card">
      <div className="metric-value" style={{ background: 'none', WebkitTextFillColor: color, color }}>{pct}%</div>
      <div className="metric-label">{label}</div>
      <div className="progress-bar" style={{ marginTop: 8 }}>
        <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}