# src/evaluation/eval_report.py
"""
Evaluation report generation for Math Mentor.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np


def _sanitize(obj):
    """Recursively convert numpy/non-serializable types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


@dataclass
class MetricResult:
    """A single metric result."""
    name: str
    value: float
    max_value: float = 1.0
    details: str = ""
    
    @property
    def percentage(self) -> float:
        return (self.value / self.max_value * 100) if self.max_value > 0 else 0
    
    @property
    def grade(self) -> str:
        pct = self.percentage
        if pct >= 90: return "A"
        elif pct >= 80: return "B"
        elif pct >= 70: return "C"
        elif pct >= 60: return "D"
        else: return "F"


@dataclass
class ComponentReport:
    """Report for a single component."""
    component: str
    metrics: List[MetricResult] = field(default_factory=list)
    details: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.percentage for m in self.metrics) / len(self.metrics)


@dataclass
class EvalReport:
    """Complete evaluation report."""
    timestamp: datetime = field(default_factory=datetime.now)
    component_reports: Dict[str, ComponentReport] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        if not self.component_reports:
            return 0.0
        scores = [r.overall_score for r in self.component_reports.values()]
        return sum(scores) / len(scores)
    
    @property
    def overall_grade(self) -> str:
        score = self.overall_score
        if score >= 90: return "A"
        elif score >= 80: return "B"
        elif score >= 70: return "C"
        elif score >= 60: return "D"
        else: return "F"
    
    def add_component(self, name: str, report: ComponentReport):
        self.component_reports[name] = report
    
    def to_dict(self) -> Dict:
        raw = {
            "timestamp": self.timestamp.isoformat(),
            "overall_score": round(float(self.overall_score), 1),
            "overall_grade": self.overall_grade,
            "components": {
                name: {
                    "score": round(float(report.overall_score), 1),
                    "metrics": [
                        {
                            "name": m.name,
                            "value": round(float(m.value), 3),
                            "percentage": round(float(m.percentage), 1),
                            "grade": m.grade,
                        }
                        for m in report.metrics
                    ],
                    "details": _sanitize(report.details),   # ← sanitize details
                    "errors": report.errors,
                }
                for name, report in self.component_reports.items()
            },
            "metadata": _sanitize(self.metadata),            # ← sanitize metadata
        }
        return _sanitize(raw)  # ← final pass to catch anything remaining
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []
        lines.append(f"# Math Mentor Evaluation Report")
        lines.append(f"**Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Overall Score:** {self.overall_score:.1f}% ({self.overall_grade})")
        lines.append("")
        lines.append("---")
        
        for name, report in self.component_reports.items():
            lines.append(f"\n## {name}")
            lines.append(f"**Score:** {report.overall_score:.1f}%")
            lines.append("")
            
            if report.metrics:
                lines.append("| Metric | Value | Score | Grade |")
                lines.append("|--------|-------|-------|-------|")
                for m in report.metrics:
                    lines.append(
                        f"| {m.name} | {m.value:.3f} | {m.percentage:.1f}% | {m.grade} |"
                    )
            
            if report.errors:
                lines.append(f"\n**Errors ({len(report.errors)}):**")
                for err in report.errors[:5]:
                    lines.append(f"- {err}")
            
            lines.append("")
        
        return "\n".join(lines)