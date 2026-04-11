import React, { useState } from 'react';
import {
  FileText, CheckCircle, Lightbulb, Target, BookOpen,
  ChevronDown, ChevronUp, Award, Zap, Database,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';


// ============================================================
// Sanitize LLM Output — detect raw JSON, fix broken LaTeX
// ============================================================

function sanitizeForDisplay(text) {
  if (!text || typeof text !== 'string') return text || '';

  let cleaned = text.trim();

  // Detect raw JSON
  if (cleaned.startsWith('{') && cleaned.endsWith('}')) {
    try {
      const parsed = JSON.parse(cleaned);

      const textField =
        parsed.explanation ||
        parsed.verification_summary ||
        parsed.verification ||
        parsed.final_answer ||
        parsed.solution ||
        '';

      if (textField) {
        cleaned = String(textField);

        const extras = [];

        if (parsed.intuition) {
          extras.push(`\n\n💡 **Key Insight:** ${parsed.intuition}`);
        }
        if (parsed.exam_tip) {
          extras.push(`\n\n📝 **JEE Tip:** ${parsed.exam_tip}`);
        }
        if (Array.isArray(parsed.key_concepts)) {
          extras.push(
            `\n\n**Key Concepts:**\n${parsed.key_concepts.map(c => `- ${c}`).join('\n')}`
          );
        }
        if (Array.isArray(parsed.common_mistakes)) {
          extras.push(
            `\n\n**Common Mistakes:**\n${parsed.common_mistakes.map(m => `- ${m}`).join('\n')}`
          );
        }

        cleaned += extras.join('');
      }
    } catch {
      const expMatch = cleaned.match(/"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"/);
      if (expMatch) {
        cleaned = expMatch[1]
          .replace(/\\"/g, '"')
          .replace(/\\n/g, '\n')
          .replace(/\\\\/g, '\\');
      }
    }
  }

  // Fix LaTeX wrapping
  if (!cleaned.includes('$')) {
    if (/^\\(frac|sqrt|int|sum|prod|lim|begin|left|right)/.test(cleaned.trim())) {
      cleaned = `\n$$\n${cleaned.trim()}\n$$\n`;
    }
  } else {
    cleaned = fixNestedDollarSigns(cleaned);
  }

  return cleaned;
}


// ============================================================
// Fix nested $ inside LaTeX
// ============================================================

function fixNestedDollarSigns(text) {
  let result = text;

  const parts = result.split('$');

  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(
      /(\\(?:frac|binom)\{[^}]*\}\{[^}]*\}|\\(?:sqrt|text|mathrm)\{[^}]*\})/g,
      (match) => {
        const clean = match.replace(/\$/g, '');
        return `$$${clean}$$`;
      }
    );
  }

  result = parts.join('$');
  result = result.replace(/\$\$\$/g, '$$');

  return result;
}


// ============================================================
// Preprocess Math
// ============================================================

function preprocessMath(text) {
  if (!text) return text;

  let result = text;

  // Normalize delimiters
  result = result.replace(/\\\(/g, '$');
  result = result.replace(/\\\)/g, '$');
  result = result.replace(/\\\[/g, '$$');
  result = result.replace(/\\\]/g, '$$');

  if (/\$/.test(result)) return result;

  // Bare LaTeX
  result = result.replace(
    /(\\(?:frac|binom)\{[^}]*\}\{[^}]*\})/g,
    (m) => `$$${m}$$`
  );

  result = result.replace(
    /(\\(?:sqrt|text|mathrm|mathbf|overline|hat|vec|bar)\{[^}]*\})/g,
    (m) => `$$${m}$$`
  );

  result = result.replace(
    /(\\(?:int|sum|prod|lim)(?:_\{[^}]*\})?(?:\^\{[^}]*\})?)/g,
    (m) => `$$${m}$$`
  );

  // Exponents
  result = result.replace(
    /([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)/g,
    '$$$1^{$2}$$'
  );

  // sqrt()
  result = result.replace(
    /sqrt\(([^)]+)\)/g,
    (_, inner) => `$\\sqrt{${inner}}$`
  );

  // fractions
  result = result.replace(
    /(\d+)\s*\/\s*(\d+)/g,
    (_, a, b) => `$\\frac{${a}}{${b}}$`
  );

  // symbols
  result = result.replace(/\+-/g, '$\\pm$');
  result = result.replace(/!=|≠/g, '$\\neq$');
  result = result.replace(/>=|≥/g, '$\\geq$');
  result = result.replace(/<=|≤/g, '$\\leq$');
  result = result.replace(/∞/g, '$\\infty$');

  return result;
}


// ============================================================
// Markdown Renderer
// ============================================================

function MathMarkdown({ children }) {
  if (!children) return null;

  const sanitized = sanitizeForDisplay(String(children));
  const processed = preprocessMath(sanitized);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
    >
      {processed}
    </ReactMarkdown>
  );
}


// ============================================================
// MAIN COMPONENT
// ============================================================

export default function SolutionDisplay({ result }) {
  const [showAllSteps, setShowAllSteps] = useState(true);

  const steps = result.solution_steps || [];
  const answer = result.final_answer || '';
  const verification = result.verification || '';
  const explanation = result.explanation || '';
  const topic = result.detected_topic || '';
  const fromCache = result.from_cache || false;

  return (
    <div>

      {fromCache && (
        <div>
          <Zap /> Cached Result
        </div>
      )}

      <h3><FileText /> Solution</h3>

      {steps.map((step, i) => (
        <div key={i} style={{ display: showAllSteps ? 'block' : 'none' }}>
          <b>{i + 1}.</b>
          <MathMarkdown>
            {typeof step === 'string' ? step : step.content}
          </MathMarkdown>
        </div>
      ))}

      {answer && (
        <>
          <h3><Award /> Final Answer</h3>
          <MathMarkdown>{answer}</MathMarkdown>
        </>
      )}

      {verification && (
        <>
          <h3><CheckCircle /> Verification</h3>
          <MathMarkdown>{verification}</MathMarkdown>
        </>
      )}

      {explanation && (
        <>
          <h3><Lightbulb /> Explanation</h3>
          <MathMarkdown>{explanation}</MathMarkdown>
        </>
      )}

      {topic && (
        <div>
          <BookOpen /> {topic}
        </div>
      )}

    </div>
  );
}