import React, { useState } from 'react';
import {
  FileText, CheckCircle, Lightbulb, Target, BookOpen,
  ChevronDown, ChevronUp, Award, Zap, Database,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';


// ============================================================
// Math-aware Markdown renderer
// ============================================================

function MathMarkdown({ children }) {
  if (!children) return null;

  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[
        [rehypeKatex, {
          throwOnError: false,
          strict: false,
        }],
      ]}
    >
      {String(children)}
    </ReactMarkdown>
  );
}


// ============================================================
// Main Component
// ============================================================

export default function SolutionDisplay({ result }) {
  const [showSolution, setShowSolution] = useState(true);

  const parsed = result.parsed_problem || {};
  const solution = result.solution || '';
  const answer = result.final_answer || '';
  const verification = result.verification || '';
  const explanation = result.explanation || '';
  const topic = result.detected_topic || '';
  const fromCache = result.from_cache || false;
  const cacheSimilarity = result.cache_similarity;

  return (
    <div className="solution-container">

      {/* Cache Hit Banner */}
      {fromCache && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 16px',
          borderRadius: 'var(--radius-md)',
          background: 'var(--accent-green-bg)',
          border: '1px solid rgba(34, 197, 94, 0.25)',
          marginBottom: 4,
        }}>
          <Zap size={18} color="var(--accent-green)" />
          <div>
            <span style={{
              fontSize: 14, fontWeight: 600,
              color: 'var(--accent-green)',
            }}>
              Instant Answer — Retrieved from Memory
            </span>
            <span style={{
              fontSize: 12, color: 'var(--text-muted)', marginLeft: 8,
            }}>
              {cacheSimilarity !== null && cacheSimilarity !== undefined
                ? `${(cacheSimilarity * 100).toFixed(0)}% match`
                : 'Exact match'
              }
              {result.latency_ms > 0 && ` • ${result.latency_ms.toFixed(0)}ms`}
            </span>
          </div>
          <Database size={14} color="var(--accent-green)" style={{ marginLeft: 'auto' }} />
        </div>
      )}

      {/* Problem Understanding */}
      {(parsed.type || parsed.what_to_find || parsed.given) && (
        <div className="solution-section" style={{ borderLeft: '3px solid var(--accent-cyan)' }}>
          <h3>
            <Target size={18} color="var(--accent-cyan)" />
            Problem Analysis
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            {parsed.type && (
              <div>
                <div style={{
                  fontSize: 11, color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4,
                }}>Type</div>
                <span className="badge badge-purple">{parsed.type}</span>
              </div>
            )}
            {parsed.what_to_find && (
              <div>
                <div style={{
                  fontSize: 11, color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4,
                }}>Find</div>
                <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                  <MathMarkdown>{parsed.what_to_find}</MathMarkdown>
                </div>
              </div>
            )}
            {parsed.given && (
              <div>
                <div style={{
                  fontSize: 11, color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4,
                }}>Given</div>
                <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                  <MathMarkdown>{parsed.given}</MathMarkdown>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════
          SOLUTION — Single flowing document with LaTeX
          ═══════════════════════════════════════════════════════ */}
      {solution && (
        <div className="solution-section" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', marginBottom: 16,
          }}>
            <h3 style={{ margin: 0 }}>
              <FileText size={18} color="var(--accent-blue)" />
              Solution
            </h3>
            <button
              onClick={() => setShowSolution(!showSolution)}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                background: 'none',
                border: '1px solid var(--border-color)',
                borderRadius: 6, padding: '4px 10px',
                color: 'var(--text-secondary)',
                cursor: 'pointer', fontSize: 12,
              }}
            >
              {showSolution
                ? <><ChevronUp size={14} /> Collapse</>
                : <><ChevronDown size={14} /> Expand</>
              }
            </button>
          </div>

          {showSolution && (
            <div style={{
              fontSize: 15,
              color: 'var(--text-secondary)',
              lineHeight: 1.8,
            }}>
              <MathMarkdown>{solution}</MathMarkdown>
            </div>
          )}
        </div>
      )}

      {/* Final Answer */}
      {answer && (
        <div className="solution-section" style={{
          borderLeft: '3px solid var(--accent-green)',
          background: 'var(--bg-tertiary)',
        }}>
          <h3>
            <Award size={18} color="var(--accent-green)" />
            Final Answer
          </h3>
          <div style={{
            padding: 16, borderRadius: 8,
            background: 'var(--bg-primary)',
            border: '1px solid var(--accent-green)',
            fontSize: 16, fontWeight: 600,
            color: 'var(--text-primary)', lineHeight: 1.6,
          }}>
            <MathMarkdown>{answer}</MathMarkdown>
          </div>
        </div>
      )}

      {/* Verification */}
      {verification && (
        <div className="solution-section" style={{ borderLeft: '3px solid var(--accent-yellow, #f59e0b)' }}>
          <h3>
            <CheckCircle size={18} color="var(--accent-yellow, #f59e0b)" />
            Verification
          </h3>
          <div style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <MathMarkdown>{verification}</MathMarkdown>
          </div>
        </div>
      )}

      {/* Explanation */}
      {explanation && (
        <div className="solution-section" style={{ borderLeft: '3px solid var(--accent-purple, #a855f7)' }}>
          <h3>
            <Lightbulb size={18} color="var(--accent-purple, #a855f7)" />
            Explanation
          </h3>
          <div style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <MathMarkdown>{explanation}</MathMarkdown>
          </div>
        </div>
      )}

      {/* Topic */}
      {topic && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '8px 12px', borderRadius: 6,
          background: 'var(--bg-tertiary)',
          fontSize: 12, color: 'var(--text-muted)', marginTop: 8,
        }}>
          <BookOpen size={14} />
          <span>
            Topic: <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
              {topic}
            </span>
          </span>
          {fromCache && (
            <span style={{
              marginLeft: 'auto', color: 'var(--accent-green)',
              display: 'flex', alignItems: 'center', gap: 4,
            }}>
              <Zap size={12} /> From Memory
            </span>
          )}
        </div>
      )}
    </div>
  );
}