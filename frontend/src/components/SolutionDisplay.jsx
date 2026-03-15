import React from 'react';
import { FileText, CheckCircle, Lightbulb } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function SolutionDisplay({ result }) {
  return (
    <div className="solution-container">
      {/* Solution */}
      <div className="solution-section">
        <h3>
          <FileText size={18} color="var(--accent-blue)" />
          Solution
        </h3>
        <div className="solution-content">
          <ReactMarkdown>{result.solution || 'No solution generated'}</ReactMarkdown>
        </div>
      </div>

      {/* Verification */}
      {result.verification && (
        <div className="solution-section" style={{ borderColor: 'rgba(16, 185, 129, 0.3)' }}>
          <h3>
            <CheckCircle size={18} color="var(--accent-green)" />
            Verification
          </h3>
          <div className="solution-content">
            <ReactMarkdown>{result.verification}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* Explanation */}
      {result.explanation && (
        <div className="solution-section" style={{ borderColor: 'rgba(139, 92, 246, 0.3)' }}>
          <h3>
            <Lightbulb size={18} color="var(--accent-purple)" />
            Explanation
          </h3>
          <div className="solution-content">
            <ReactMarkdown>{result.explanation}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}