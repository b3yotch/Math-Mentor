import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

export default function MathText({ text }) {
  if (!text) return null;

  const segments = splitMathSegments(text);

  return (
    <span>
      {segments.map((seg, i) => {
        if (seg.type === 'block') {
          return <BlockMath key={i} math={seg.content} />;
        }
        if (seg.type === 'inline') {
          return <InlineMath key={i} math={seg.content} />;
        }
        return <span key={i}>{seg.content}</span>;
      })}
    </span>
  );
}

function splitMathSegments(text) {
  const segments = [];

  // ✅ FIXED REGEX
  const regex =
    /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\$(.*?)\$|\\\((.*?)\\\)/g;

  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: 'text',
        content: text.slice(lastIndex, match.index),
      });
    }

    if (match[1] || match[2]) {
      segments.push({
        type: 'block',
        content: (match[1] || match[2]).trim(),
      });
    } else if (match[3] || match[4]) {
      segments.push({
        type: 'inline',
        content: (match[3] || match[4]).trim(),
      });
    }

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    segments.push({
      type: 'text',
      content: text.slice(lastIndex),
    });
  }

  return segments;
}