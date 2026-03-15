import React, { useState, useEffect } from 'react';
import {
  Brain, BookOpen, Shield, Activity, Database,
  Settings, ChevronDown, Wifi, WifiOff,
} from 'lucide-react';
import api from '../api/client';

export default function Sidebar({ settings, onSettingsChange }) {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.getStats().then(setStats).catch(() => {});
  }, []);

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div style={{ textAlign: 'center', paddingBottom: 8 }}>
        <div style={{ fontSize: 36, marginBottom: 4 }}>
          <Brain size={36} style={{ color: 'var(--accent-blue)' }} />
        </div>
        <h1 style={{ fontSize: 20, fontWeight: 700, background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Math Mentor
        </h1>
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>AI-Powered JEE Math Tutor</p>
      </div>

      <div style={{ height: 1, background: 'var(--border-color)' }} />

      {/* Settings */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Settings size={14} color="var(--text-muted)" />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>Settings</span>
        </div>

        <div className="input-group" style={{ marginBottom: 12 }}>
          <label className="input-label">RAG Results</label>
          <input
            type="range"
            min="1" max="5"
            value={settings.topK}
            onChange={(e) => onSettingsChange({ ...settings, topK: parseInt(e.target.value) })}
            style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
          />
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{settings.topK} documents</span>
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer', marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={settings.showRag}
            onChange={(e) => onSettingsChange({ ...settings, showRag: e.target.checked })}
            style={{ accentColor: 'var(--accent-blue)' }}
          />
          Show RAG Context
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer', marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={settings.includeEval}
            onChange={(e) => onSettingsChange({ ...settings, includeEval: e.target.checked })}
            style={{ accentColor: 'var(--accent-blue)' }}
          />
          Include Evaluation
        </label>
      </div>

      <div style={{ height: 1, background: 'var(--border-color)' }} />

      {/* System Status */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Activity size={14} color="var(--text-muted)" />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>System Status</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <StatusItem label="API Server" ok={!!health} />
          <StatusItem label="Groq LLM" ok={health?.groq_api} />
          <StatusItem label="RAG Engine" ok={health?.rag_available} />
          <StatusItem label="Guardrails" ok={health?.guardrails_active} />
          <StatusItem label="Memory DB" ok={health?.memory_available} />
        </div>
      </div>

      <div style={{ height: 1, background: 'var(--border-color)' }} />

      {/* Memory Stats */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Database size={14} color="var(--text-muted)" />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>Memory</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <MiniStat label="Solved" value={stats?.total_problems || 0} />
          <MiniStat label="Feedback" value={stats?.feedback?.total_feedback || 0} />
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 'auto', fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
        <Shield size={12} style={{ display: 'inline', marginRight: 4 }} />
        Protected by Guardrails
      </div>
    </aside>
  );
}

function StatusItem({ label, ok }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      {ok ? (
        <span style={{ color: 'var(--accent-green)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
          <Wifi size={12} /> Connected
        </span>
      ) : (
        <span style={{ color: 'var(--text-muted)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
          <WifiOff size={12} /> Offline
        </span>
      )}
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div style={{ background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', padding: '10px 12px', textAlign: 'center' }}>
      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-blue)' }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</div>
    </div>
  );
}