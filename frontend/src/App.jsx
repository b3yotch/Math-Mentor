import React, { useState } from 'react';
import {
  Sparkles, Clock, BarChart3,
} from 'lucide-react';
import Layout from './components/Layout';
import SolveTab from './components/SolveTab';
import HistoryTab from './components/HistoryTab';
import EvaluationTab from './components/EvaluationTab';

const TABS = [
  { id: 'solve', label: 'Solve', icon: Sparkles },
  { id: 'history', label: 'History', icon: Clock },
  { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('solve');
  const [settings, setSettings] = useState({
    topK: 3,
    showRag: true,
    includeEval: true,
  });

  const renderTab = () => {
    switch (activeTab) {
      case 'solve':
        return <SolveTab settings={settings} />;
      case 'history':
        return <HistoryTab />;
      case 'evaluation':
        return <EvaluationTab />;
      default:
        return <SolveTab settings={settings} />;
    }
  };

  return (
    <Layout settings={settings} onSettingsChange={setSettings}>
      {/* Header */}
      <div className="header">
        <div>
          <h2 style={{
            fontSize: 22,
            fontWeight: 700,
            background: 'var(--gradient-primary)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Math Mentor
          </h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            AI-Powered JEE Math Tutor
          </p>
        </div>

        {/* Tabs */}
        <div className="tabs">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`tab ${activeTab === id ? 'active' : ''}`}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>

        {/* Right side placeholder */}
        <div style={{ width: 120 }} />
      </div>

      {/* Content */}
      <div className="content-area">
        {renderTab()}
      </div>

      {/* Footer */}
      <div style={{
        padding: '16px 32px',
        textAlign: 'center',
        borderTop: '1px solid var(--border-color)',
        fontSize: 12,
        color: 'var(--text-muted)',
      }}>
        Protected by Guardrails | Human-in-the-Loop for low-confidence extractions | Learning from corrections
      </div>
    </Layout>
  );
}