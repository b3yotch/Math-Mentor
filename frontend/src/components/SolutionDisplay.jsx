import React, { useState } from 'react';
import {
  FileText, CheckCircle, Lightbulb, Target, BookOpen,
  AlertTriangle, ChevronDown, ChevronUp, Award,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function SolutionDisplay({ result }) {
  const [showAllSteps, setShowAllSteps] = useState(true);

  const steps = result.solution_steps || [];
  const parsed = result.parsed_problem || {};
  const answer = result.final_answer || '';
  const verification = result.verification || '';
  const explanation = result.explanation || '';
  const topic = result.detected_topic || '';

  return (
    <div className="solution-container">
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
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>
                  Type
                </div>
                <span className="badge badge-purple">{parsed.type}</span>
              </div>
            )}
            {parsed.what_to_find && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>
                  Find
                </div>
                <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>{parsed.what_to_find}</div>
              </div>
            )}
            {parsed.given && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>
                  Given
                </div>
                <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>{parsed.given}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Solution Steps */}
      <div className="solution-section" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ margin: 0 }}>
            <FileText size={18} color="var(--accent-blue)" />
            Solution Steps
            {steps.length > 0 && (
              <span style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>
                ({steps.length} {steps.length === 1 ? 'step' : 'steps'})
              </span>
            )}
          </h3>
          {steps.length > 1 && (
            <button
              onClick={() => setShowAllSteps(!showAllSteps)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                background: 'none',
                border: '1px solid var(--border-color)',
                borderRadius: 6,
                padding: '4px 10px',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: 12,
                transition: 'all 0.2s ease',
              }}
            >
              {showAllSteps ? (
                <>
                  <ChevronUp size={14} />
                  Collapse
                </>
              ) : (
                <>
                  <ChevronDown size={14} />
                  Expand All
                </>
              )}
            </button>
          )}
        </div>

        {steps.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {steps.map((step, index) => (
              <div
                key={index}
                style={{
                  display: showAllSteps || index === 0 || index === steps.length - 1 ? 'flex' : 'none',
                  gap: 12,
                  padding: 12,
                  borderRadius: 8,
                  background: 'var(--bg-tertiary)',
                  transition: 'all 0.2s ease',
                }}
              >
                <div
                  style={{
                    minWidth: 28,
                    height: 28,
                    borderRadius: '50%',
                    background: 'var(--accent-blue)',
                    color: '#fff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 13,
                    fontWeight: 600,
                    flexShrink: 0,
                  }}
                >
                  {index + 1}
                </div>
                <div style={{ flex: 1, fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {typeof step === 'string' ? (
                    <ReactMarkdown>{step}</ReactMarkdown>
                  ) : (
                    <>
                      {step.title && (
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
                          {step.title}
                        </div>
                      )}
                      <ReactMarkdown>{step.content || step.description || ''}</ReactMarkdown>
                    </>
                  )}
                </div>
              </div>
            ))}
            {!showAllSteps && steps.length > 2 && (
              <div
                style={{
                  textAlign: 'center',
                  padding: '8px 0',
                  color: 'var(--text-muted)',
                  fontSize: 13,
                  fontStyle: 'italic',
                }}
              >
                ... {steps.length - 2} more {steps.length - 2 === 1 ? 'step' : 'steps'} hidden
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: 14, fontStyle: 'italic' }}>
            No detailed steps available.
          </div>
        )}
      </div>

      {/* Final Answer */}
      {answer && (
        <div
          className="solution-section"
          style={{
            borderLeft: '3px solid var(--accent-green)',
            background: 'var(--bg-tertiary)',
          }}
        >
          <h3>
            <Award size={18} color="var(--accent-green)" />
            Final Answer
          </h3>
          <div
            style={{
              padding: 16,
              borderRadius: 8,
              background: 'var(--bg-primary)',
              border: '1px solid var(--accent-green)',
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--text-primary)',
              lineHeight: 1.6,
            }}
          >
            <ReactMarkdown>{answer}</ReactMarkdown>
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
            <ReactMarkdown>{verification}</ReactMarkdown>
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
            <ReactMarkdown>{explanation}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* Detected Topic */}
      {topic && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 12px',
            borderRadius: 6,
            background: 'var(--bg-tertiary)',
            fontSize: 12,
            color: 'var(--text-muted)',
            marginTop: 8,
          }}
        >
          <BookOpen size={14} />
          <span>
            Topic: <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{topic}</span>
          </span>
        </div>
      )}
    </div>
  );
}