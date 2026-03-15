import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

export default function RAGContext({ sources }) {
  const [open, setOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="collapsible">
      <button className="collapsible-header" onClick={() => setOpen(!open)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <BookOpen size={16} color="var(--accent-cyan)" />
          <span>Retrieved Knowledge ({sources.length} sources)</span>
        </div>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div className="collapsible-body">
          {sources.map((source, i) => (
            <div
              key={i}
              style={{
                padding: '12px 0',
                borderBottom: i < sources.length - 1 ? '1px solid var(--border-color)' : 'none',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontWeight: 500, fontSize: 14 }}>
                  [{i + 1}] {source.topic}/{source.subtopic}
                </span>
                <ScoreBadge score={source.score} />
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                {source.content_preview}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ScoreBadge({ score }) {
  const pct = (score * 100).toFixed(0);
  const cls = score >= 0.6 ? 'badge-green' : score >= 0.4 ? 'badge-orange' : 'badge-red';
  return <span className={`badge ${cls}`}>{pct}%</span>;
}