import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, MessageSquare, Send, X } from 'lucide-react';
import api from '../api/client';
import toast from 'react-hot-toast';

export default function FeedbackSection({ problemId }) {
  const [submitted, setSubmitted] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [comment, setComment] = useState('');
  const [corrected, setCorrected] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleFeedback = async (isCorrect) => {
    if (!isCorrect) {
      setShowForm(true);
      return;
    }

    try {
      setSubmitting(true);
      await api.recordFeedback(problemId, true);
      setSubmitted(true);
      toast.success('Thanks for your feedback!');
    } catch (err) {
      toast.error('Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitNegative = async () => {
    try {
      setSubmitting(true);
      await api.recordFeedback(problemId, false, comment, corrected);
      setSubmitted(true);
      setShowForm(false);
      toast.success('Feedback recorded. We will improve!');
    } catch (err) {
      toast.error('Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="card" style={{ textAlign: 'center', borderColor: 'rgba(16, 185, 129, 0.3)' }}>
        <p style={{ color: 'var(--accent-green)', fontWeight: 500 }}>
          Thank you for your feedback!
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <MessageSquare size={18} color="var(--accent-blue)" />
        <span className="card-title" style={{ fontSize: 16 }}>Was this helpful?</span>
      </div>

      {!showForm ? (
        <div style={{ display: 'flex', gap: 12 }}>
          <button
            className="btn btn-success"
            style={{ flex: 1 }}
            onClick={() => handleFeedback(true)}
            disabled={submitting}
          >
            <ThumbsUp size={16} /> Correct
          </button>
          <button
            className="btn btn-danger"
            style={{ flex: 1 }}
            onClick={() => handleFeedback(false)}
            disabled={submitting}
          >
            <ThumbsDown size={16} /> Incorrect
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="input-group">
            <label className="input-label">What was wrong?</label>
            <textarea
              className="textarea"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Describe the issue..."
              rows={3}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Correct answer (optional)</label>
            <textarea
              className="textarea"
              value={corrected}
              onChange={(e) => setCorrected(e.target.value)}
              placeholder="Provide the correct solution if you know it..."
              rows={3}
            />
          </div>

          <div style={{ display: 'flex', gap: 12 }}>
            <button
              className="btn btn-primary"
              style={{ flex: 1 }}
              onClick={handleSubmitNegative}
              disabled={submitting || !comment.trim()}
            >
              <Send size={16} />
              {submitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => {
                setShowForm(false);
                setComment('');
                setCorrected('');
              }}
              disabled={submitting}
            >
              <X size={16} /> Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}