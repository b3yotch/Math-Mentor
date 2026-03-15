/**
 * API Client for Math Mentor FastAPI backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';

class ApiClient {
  async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
      }

      return data;
    } catch (error) {
      if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
        throw new Error('Cannot connect to API server. Is it running on port 8000?');
      }
      throw error;
    }
  }

  // Health
  async health() {
    return this.request('/health');
  }

  // Solve
  async solve(question, topK = 3, includeEvaluation = true) {
    return this.request('/solve', {
      method: 'POST',
      body: JSON.stringify({
        question,
        top_k: topK,
        include_evaluation: includeEvaluation,
      }),
    });
  }

  // RAG Retrieve
  async retrieve(query, topK = 3, topicFilter = null) {
    return this.request('/retrieve', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: topK,
        topic_filter: topicFilter,
      }),
    });
  }

  // Guardrails
  async checkGuardrails(text) {
    return this.request('/guardrails/check', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  // History
  async getHistory(limit = 10, topic = '') {
    const params = new URLSearchParams({ limit, topic });
    return this.request(`/memory/history?${params}`);
  }

  // Feedback
  async recordFeedback(problemId, isCorrect, comment = '', correctedSolution = '') {
    return this.request('/memory/feedback', {
      method: 'POST',
      body: JSON.stringify({
        problem_id: problemId,
        is_correct: isCorrect,
        comment,
        corrected_solution: correctedSolution,
      }),
    });
  }

  // Stats
  async getStats() {
    return this.request('/memory/stats');
  }

  // Evaluate solution
  async evaluateSolution(question, solution, explanation = '') {
    return this.request('/evaluate/solution', {
      method: 'POST',
      body: JSON.stringify({ question, solution, explanation }),
    });
  }

  // Batch evaluation
  async runBatchEvaluation(topic = '', maxCases = 10) {
    return this.request('/evaluate/batch', {
      method: 'POST',
      body: JSON.stringify({
        topic: topic || null,
        max_cases: maxCases,
        include_rag: true,
        include_solutions: true,
        include_guardrails: true,
      }),
    });
  }
}

export const api = new ApiClient();
export default api;