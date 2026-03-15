/**
 * API Client for Math Mentor FastAPI backend.
 * Includes per-user tracking, file upload, and HITL extraction support.
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';

// Generate or retrieve persistent user ID
function getUserId() {
  let userId = localStorage.getItem('math_mentor_user_id');
  if (!userId) {
    userId = 'user_' + crypto.randomUUID();
    localStorage.setItem('math_mentor_user_id', userId);
  }
  return userId;
}

class ApiClient {
  constructor() {
    this.userId = getUserId();
  }

  async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    };

    let response;
    try {
      response = await fetch(url, config);
    } catch (error) {
      throw new Error(
        'Cannot connect to API server. Make sure the backend is running.\n\n' +
        'Run: uvicorn api.main:app --reload --port 8000'
      );
    }

    const text = await response.text();
    if (!text || text.trim() === '') {
      throw new Error('Empty response. Is the API server running?');
    }

    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error('Invalid response from server');
    }

    if (!response.ok) {
      throw new Error(data.detail || data.error || `HTTP ${response.status}`);
    }

    return data;
  }

  // Health
  async health() {
    return this.request('/health');
  }

  // Solve text (also used for HITL-edited image/audio extractions)
  async solve(question, topK = 3, includeEvaluation = true, {
    inputType = 'text',
    confidence = 1.0,
    wasHumanEdited = false,
  } = {}) {
    return this.request('/solve', {
      method: 'POST',
      body: JSON.stringify({
        question,
        top_k: topK,
        include_evaluation: includeEvaluation,
        user_id: this.userId,
        input_type: inputType,
        confidence,
        was_human_edited: wasHumanEdited,
      }),
    });
  }

  // Solve from image (direct, no HITL)
  async solveImage(file, topK = 3, includeEvaluation = true) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_k', topK.toString());
    formData.append('include_evaluation', includeEvaluation.toString());
    formData.append('user_id', this.userId);

    return this.request('/solve/image', {
      method: 'POST',
      headers: {},
      body: formData,
    });
  }

  // Solve from audio (direct, no HITL)
  async solveAudio(file, topK = 3, includeEvaluation = true) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_k', topK.toString());
    formData.append('include_evaluation', includeEvaluation.toString());
    formData.append('user_id', this.userId);

    return this.request('/solve/audio', {
      method: 'POST',
      headers: {},
      body: formData,
    });
  }

  // Extract text from image (for HITL preview — step 1)
  async extractImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    return this.request('/solve/extract/image', {
      method: 'POST',
      headers: {},
      body: formData,
    });
  }

  // Extract text from audio (for HITL preview — step 1)
  async extractAudio(file) {
    const formData = new FormData();
    formData.append('file', file);

    return this.request('/solve/extract/audio', {
      method: 'POST',
      headers: {},
      body: formData,
    });
  }

  // RAG Retrieve
  async retrieve(query, topK = 3, topicFilter = null) {
    return this.request('/retrieve', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK, topic_filter: topicFilter }),
    });
  }

  // Guardrails
  async checkGuardrails(text) {
    return this.request('/guardrails/check', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  // History (per-user)
  async getHistory(limit = 10, topic = '') {
    const params = new URLSearchParams();
    params.set('limit', limit);
    params.set('user_id', this.userId);
    if (topic) params.set('topic', topic);
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
        user_id: this.userId,
      }),
    });
  }

  // Stats (per-user)
  async getStats() {
    return this.request(`/memory/stats?user_id=${this.userId}`);
  }

  // Evaluate
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

  // Get user ID
  getUserId() {
    return this.userId;
  }
}

export const api = new ApiClient();
export default api;