import React from 'react';
import Sidebar from './Sidebar';

export default function Layout({ children, settings, onSettingsChange }) {
  return (
    <div className="app-layout">
      <Sidebar settings={settings} onSettingsChange={onSettingsChange} />
      <div className="main-content">
        {children}
      </div>
    </div>
  );
}