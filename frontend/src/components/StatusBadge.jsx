import React from 'react';
import { CheckCircle, XCircle, AlertTriangle, Clock } from 'lucide-react';

const configs = {
  success: { icon: CheckCircle, className: 'badge-green', label: 'Success' },
  error: { icon: XCircle, className: 'badge-red', label: 'Error' },
  blocked: { icon: XCircle, className: 'badge-red', label: 'Blocked' },
  warning: { icon: AlertTriangle, className: 'badge-orange', label: 'Warning' },
  pending: { icon: Clock, className: 'badge-blue', label: 'Pending' },
};

export default function StatusBadge({ status, label }) {
  const config = configs[status] || configs.pending;
  const Icon = config.icon;

  return (
    <span className={`badge ${config.className}`}>
      <Icon size={12} />
      {label || config.label}
    </span>
  );
}