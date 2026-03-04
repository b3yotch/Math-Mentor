# app.py
"""
Math Mentor - AI-Powered JEE Math Tutor
Complete Version with Text, Image, and Audio Input + HITL + Guardrails
"""

import streamlit as st
import time
import os
import sys
import tempfile
import json
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# ============================================================
# SUPPRESS NOISY LOGS (Must be before other imports)
# ============================================================
warnings.filterwarnings("ignore")
for logger_name in ["httpx", "httpcore", "faiss", "transformers", 
                    "sentence_transformers", "huggingface_hub"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Math Mentor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_shared_resources():
    """Initialize all shared resources including embedding model."""
    from src.core.embedding_manager import get_embedding_manager, preload_model
    
    # Preload the embedding model
    em = get_embedding_manager()
    _ = em.model  # Trigger loading
    
    return em


shared_embedding_manager = initialize_shared_resources()

# ============================================================
# Imports from our modules
# ============================================================
from src.input_processing.schemas import CanonicalInput
from src.input_processing.math_normalizer import MathNormalizer
from src.input_processing.ocr_processor import OCRProcessor
from src.rag.retriever import MathRAGRetriever
from src.memory.memory_manager import MemoryManager

# Import improved guardrails
from src.guardrails.guardrails_manager import (
    GuardrailsManager, 
    GuardrailsReport,
    CheckStatus
)

# Import HITL
from src.hitl.hitl_manager import (
    HITLManager,
    HITLTrigger,
    HITLStatus,
    HITLRequest,
    HITLStage,
    get_hitl_manager,
)




# ============================================================
# Cached Resource Loaders
# ============================================================
@st.cache_resource
def load_ocr():
    """Load OCR processor (cached)."""
    return OCRProcessor()


@st.cache_resource
def load_asr():
    """Load ASR processor (cached)."""
    try:
        from src.input_processing.asr_processor import ASRProcessor
        return ASRProcessor(model_size="base")
    except Exception as e:
        return None


@st.cache_resource
def load_normalizer():
    """Load math normalizer (cached)."""
    return MathNormalizer()


@st.cache_resource
def load_retriever():
    """Load RAG retriever with shared embedding model and reranker."""
    from src.rag.retriever import MathRAGRetriever
    return MathRAGRetriever(
        embedding_manager=shared_embedding_manager,
        use_reranker=True,
        reranker_type="hybrid"
    )


@st.cache_resource
def load_memory():
    """Load memory manager (cached)."""
    return MemoryManager()


@st.cache_resource
def load_groq_client():
    """Load Groq client directly."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        st.error("Please install groq: pip install groq")
        return None


@st.cache_resource
def load_guardrails():
    """Load guardrails with shared embedding model."""
    from src.guardrails.guardrails_manager import GuardrailsManager
    
    # Pass embedding manager to guardrails
    manager = GuardrailsManager(strict_mode=False)
    
    # Initialize the classifier with shared model
    manager.content_filter._embedding_manager = shared_embedding_manager
    
    return manager


@st.cache_resource
def load_hitl_manager():
    """Load HITL manager with memory integration."""
    memory = load_memory()
    return get_hitl_manager(memory_manager=memory)


# ============================================================
# Helper Functions for Guardrails
# ============================================================
def get_detected_topic(text: str) -> str:
    """
    Get detected math topic from text.
    Uses content filter's topic classification.
    """
    guardrails = load_guardrails()
    
    # Get the math score and metadata from content filter
    sanitized = guardrails.sanitize(text)
    safety_check = guardrails.content_filter.check_input(sanitized)
    
    # Category from safety check often contains topic info
    category = safety_check.category
    
    # Map category to display-friendly topic
    topic_map = {
        "approved": "general",
        "math": "general",
        "algebra": "Algebra",
        "calculus": "Calculus",
        "trigonometry": "Trigonometry",
        "probability": "Probability",
        "statistics": "Statistics",
        "geometry": "Geometry",
        "linear_algebra": "Linear Algebra",
    }
    
    return topic_map.get(category, "Mathematics")


def display_guardrails_error(report: GuardrailsReport):
    """Display guardrails error with suggestions."""
    guardrails = load_guardrails()
    
    # Get rejection message
    rejection_msg = guardrails.get_rejection_message(report)
    st.error(f"🛡️ **Input Blocked:** {rejection_msg}")
    
    # Get and display suggestions
    suggestions = guardrails.get_suggestions(report)
    if suggestions:
        st.info("💡 **Suggestions:**")
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            st.markdown(f"- {suggestion}")


def display_guardrails_warnings(report: GuardrailsReport):
    """Display any warnings from guardrails."""
    if report.warnings:
        for warning in report.warnings:
            st.warning(f"⚠️ {warning}")


# ============================================================
# Input Processing Functions
# ============================================================
def process_text_input(text: str) -> CanonicalInput:
    """Process text input."""
    normalizer = load_normalizer()
    normalized = normalizer.normalize(text)
    
    return CanonicalInput(
        input_type="text",
        extracted_text=normalized,
        original_extraction=text,
        confidence_score=1.0
    )


def process_image_input(image_file) -> CanonicalInput:
    """Process image input with OCR."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(image_file.getvalue())
        tmp_path = tmp.name
    
    ocr = load_ocr()
    result = ocr.process(tmp_path)
    
    # Apply learned corrections
    memory = load_memory()
    corrected_text = memory.apply_corrections(result.extracted_text, 'image')
    result.extracted_text = corrected_text
    
    os.unlink(tmp_path)
    
    return result


def process_audio_input(audio_file) -> CanonicalInput:
    """Process audio input with ASR."""
    asr = load_asr()
    
    if asr is None:
        return CanonicalInput(
            input_type="audio",
            extracted_text="",
            original_extraction="",
            confidence_score=0.0,
            metadata={"error": "ASR not available. Install whisper: pip install openai-whisper"}
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name
    
    result = asr.process(tmp_path)
    
    memory = load_memory()
    corrected_text = memory.apply_corrections(result.extracted_text, 'audio')
    result.extracted_text = corrected_text
    
    os.unlink(tmp_path)
    
    return result


def get_rag_context(query: str, top_k: int = 3, use_reranker: bool = True):
    """Get relevant context from RAG with reranking."""
    retriever = load_retriever()
    if retriever is None:
        return []
    
    try:
        results = retriever.retrieve(
            query, 
            top_k=top_k,
            rerank=use_reranker
        )
        # Format results for display
        formatted = []
        for r in results:
            formatted.append({
                'title': r.get('type', 'Unknown').title(),
                'topic': f"{r.get('topic', 'general')}/{r.get('subtopic', '')}",
                'content': r.get('content', ''),
                'score': r.get('score', 0),
                'semantic_score': r.get('semantic_score', 0),
                'keyword_score': r.get('keyword_score', 0),
                'rerank_score': r.get('rerank_score', 0),
                'chapter': r.get('chapter', ''),
                'section': r.get('section', ''),
            })
        return formatted
    except Exception as e:
        st.warning(f"RAG retrieval error: {e}")
        return []


def save_to_memory(canonical: CanonicalInput, rag_context, result, topic=None):
    """Save solved problem to memory."""
    import numpy as np
    
    memory = load_memory()
    
    def make_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return make_serializable(obj.to_dict())
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            try:
                return float(obj) if '.' in str(obj) else int(obj)
            except (ValueError, TypeError):
                return str(obj)
    
    # Convert entire rag_context to serializable format
    serializable_context = make_serializable(rag_context)
    
    problem_id = memory.save_solved_problem(
        input_type=canonical.input_type,
        original_input=canonical.original_extraction,
        extracted_text=canonical.extracted_text,
        topic=topic,
        solution=result.get('solution', ''),
        explanation=result.get('explanation', ''),
        verification_result=result.get('verification'),
        rag_context=serializable_context,
        confidence_score=float(canonical.confidence_score),
        was_human_edited=canonical.was_human_edited
    )
    
    return problem_id


# ============================================================
# Core Solving Function
# ============================================================
def solve_with_llm(problem_text: str, rag_context: list, similar_problems: list = None):
    """Solve math problem using LLM with RAG context."""
    client = load_groq_client()
    
    if client is None:
        return {
            "error": "Groq API not configured",
            "solution": "Please add GROQ_API_KEY to .env file",
            "verification": "N/A",
            "explanation": "API key required for solving"
        }
    
    # Build context from RAG results
    context_text = "\n\n".join([
        f"**{doc['title']}** ({doc['topic']}):\n{doc['content']}"
        for doc in rag_context
    ]) if rag_context else "No specific context available."
    
    # Build similar problems context
    similar_text = ""
    if similar_problems:
        similar_text = "\n\nSimilar previously solved problems:\n" + "\n".join([
            f"- {p.get('extracted_text', '')}: {p.get('solution', '')}"
            for p in similar_problems[:2]
        ])
    
    system_prompt = """You are an expert JEE Mathematics tutor. You solve problems step-by-step with clear explanations.

Your response MUST be in this exact JSON format:
{
    "parsed_problem": {
        "type": "algebra/calculus/probability/linear_algebra",
        "what_to_find": "description of what needs to be found",
        "given": "what information is given"
    },
    "solution_steps": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
    ],
    "final_answer": "The final answer",
    "verification": "How to verify this answer is correct",
    "explanation": "Student-friendly explanation of the approach and key concepts"
}

Be thorough, show all work, and explain your reasoning."""

    user_prompt = f"""Solve this math problem:

**Problem:** {problem_text}

**Relevant Knowledge from Database:**
{context_text}
{similar_text}

Provide a complete solution with all steps. Return valid JSON only."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        try:
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            
            solution_text = "\n".join(result.get("solution_steps", ["No steps provided"]))
            final_answer = result.get("final_answer", "")
            
            return {
                "parsed_problem": result.get("parsed_problem", {}),
                "solution": f"{solution_text}\n\n**Final Answer:** {final_answer}",
                "verification": result.get("verification", "Verification not provided"),
                "explanation": result.get("explanation", "Explanation not provided"),
                "raw_response": content
            }
            
        except json.JSONDecodeError:
            return {
                "solution": content,
                "verification": "Response was not in expected format",
                "explanation": "See solution above for explanation",
                "raw_response": content
            }
    
    except Exception as e:
        error_msg = str(e)
        return {
            "error": error_msg,
            "solution": f"Error occurred: {error_msg}",
            "verification": "Could not verify due to error",
            "explanation": "An error occurred while solving. Please try again."
        }


# ============================================================
# HITL Functions
# ============================================================
def render_extraction_hitl(
    canonical: CanonicalInput,
    hitl_request: Optional[HITLRequest],
    input_type: str
) -> Optional[CanonicalInput]:
    """
    Render HITL UI for extraction review - SIMPLIFIED VERSION.
    
    Args:
        canonical: The canonical input to review
        hitl_request: HITL request if triggered (can be None)
        input_type: "image" or "audio"
    
    Returns:
        Updated CanonicalInput if ready to proceed, None if waiting for action
    """
    hitl_manager = load_hitl_manager()
    guardrails = load_guardrails()
    
    # Generate unique keys
    key_suffix = f"{input_type}_{canonical.input_id[:8]}"
    edit_key = f"edit_text_{key_suffix}"
    
    # Show confidence indicator
    render_confidence_indicator(canonical.confidence_score)
    
    # Show HITL alert if triggered
    if hitl_request:
        display = hitl_manager.get_display_info(hitl_request)
        st.warning(f"**{display['title']}**")
        st.caption(display['reason'])
        
        with st.expander("💡 Review Suggestions", expanded=True):
            for suggestion in display['suggestions'][:5]:
                st.markdown(f"• {suggestion}")
    
    # Show original extraction if available
    if canonical.original_extraction and canonical.original_extraction != canonical.extracted_text:
        with st.expander("📄 Original Extraction"):
            st.code(canonical.original_extraction[:500])
    
    # Get the current text to display
    # Priority: session state edited text > canonical extracted text
    stored_key = f"stored_text_{key_suffix}"
    if stored_key in st.session_state:
        display_text = st.session_state[stored_key]
    else:
        display_text = canonical.extracted_text
        st.session_state[stored_key] = display_text
    
    # Always show editable text area
    st.markdown("**Review and edit the extracted text:**")
    edited_text = st.text_area(
        "Extracted text:",
        value=display_text,
        height=120,
        key=edit_key,
        label_visibility="collapsed"
    )
    
    # Update stored text when user types
    st.session_state[stored_key] = edited_text
    
    # Check if text was modified
    original_text = canonical.original_extraction or canonical.extracted_text
    text_was_modified = edited_text != original_text
    
    # Show modification indicator
    
    
    st.markdown("---")
    
    # Action buttons
    if hitl_request:
        # HITL mode - show all three buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            approve_btn = st.button(
                "✅ Use Original",
                use_container_width=True,
                key=f"approve_{key_suffix}",
                help="Use the original extracted text without changes"
            )
        
        with col2:
            save_btn = st.button(
                "💾 Save & Continue",
                type="primary",
                use_container_width=True,
                key=f"save_{key_suffix}",
                help="Save your edits and continue"
            )
        
        with col3:
            skip_btn = st.button(
                "⏭️ Skip Review",
                use_container_width=True,
                key=f"skip_{key_suffix}",
                help="Skip review and use original"
            )
    else:
        # No HITL triggered - simpler UI
        col1, col2 = st.columns(2)
        
        with col1:
            save_btn = st.button(
                "✅ Continue" if not text_was_modified else "💾 Save & Continue",
                type="primary",
                use_container_width=True,
                key=f"continue_{key_suffix}"
            )
        
        with col2:
            if text_was_modified:
                revert_btn = st.button(
                    "↩️ Revert Changes",
                    use_container_width=True,
                    key=f"revert_{key_suffix}"
                )
                if revert_btn:
                    st.session_state[stored_key] = original_text
                    st.rerun()
        
        approve_btn = False
        skip_btn = False
    
    # Handle button actions
    final_text = None
    action_taken = None
    
    if hitl_request:
        if approve_btn:
            final_text = canonical.extracted_text  # Original
            action_taken = "approved"
            hitl_manager.approve_request(hitl_request.request_id)
            
        elif save_btn:
            final_text = edited_text
            action_taken = "edited" if text_was_modified else "approved"
            if text_was_modified:
                hitl_manager.edit_request(hitl_request.request_id, edited_text)
            else:
                hitl_manager.approve_request(hitl_request.request_id)
            
        elif skip_btn:
            final_text = canonical.extracted_text  # Original
            action_taken = "skipped"
            hitl_manager.skip_request(hitl_request.request_id)
    else:
        if save_btn:
            final_text = edited_text
            action_taken = "edited" if text_was_modified else "approved"
    
    # Process the action
    if final_text is not None:
        # Validate the final text
        report = guardrails.check_input(final_text)
        if not report.passed:
            display_guardrails_error(report)
            return None
        
        # Update canonical with final text
        if final_text != canonical.extracted_text:
            # Text was changed - learn from correction
            memory = load_memory()
            memory.learn_extraction_correction(
                input_type,
                canonical.extracted_text,
                final_text
            )
            
            canonical.original_extraction = canonical.extracted_text
            canonical.extracted_text = final_text
            canonical.was_human_edited = True
        
        # Mark as reviewed
        canonical.mark_reviewed(action_taken, final_text if action_taken == "edited" else None)
        
        # Clear stored text
        if stored_key in st.session_state:
            del st.session_state[stored_key]
        
        # Show success
        if action_taken == "edited":
            st.success("✏️ Changes saved! Learning signal stored.")
        elif action_taken == "approved":
            st.success("✅ Approved!")
        elif action_taken == "skipped":
            st.info("⏭️ Skipped review")
        
        # Show topic
        topic = report.metadata.get('category', 'general')
        render_topic_badge(topic)
        display_guardrails_warnings(report)
        
        return canonical
    
    # No action taken yet
    st.caption("👆 Click a button above to continue")
    return None

def render_evaluation_tab():
    """Render the evaluation tab in the UI."""
    st.subheader("📊 System Evaluation")
    st.markdown("Run comprehensive tests to evaluate system performance.")
    
    from src.evaluation.eval_manager import get_eval_manager
    from src.evaluation.test_datasets import get_test_dataset
    
    dataset = get_test_dataset()
    summary = dataset.get_summary()
    
    # Dataset info
    with st.expander("📋 Test Dataset", expanded=False):
        st.markdown(f"**Total test cases:** {summary['total']}")
        for topic, count in summary['by_topic'].items():
            st.markdown(f"- {topic.title()}: {count} cases")
    
    # Evaluation options
    st.markdown("### Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        eval_rag = st.checkbox("Evaluate RAG", value=True)
        eval_solution = st.checkbox("Evaluate Solutions", value=True)
    with col2:
        eval_guardrails = st.checkbox("Evaluate Guardrails", value=True)
        eval_memory = st.checkbox("Evaluate Memory", value=True)
    
    topic_options = ["All Topics"] + list(summary['by_topic'].keys())
    selected_topic = st.selectbox("Topic Filter:", topic_options)
    topic_filter = None if selected_topic == "All Topics" else selected_topic
    
    max_cases = st.slider("Max Test Cases", 5, summary['total'], min(20, summary['total']))
    
    # Run evaluation
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        eval_manager = get_eval_manager()
        
        progress = st.progress(0)
        status = st.empty()
        
        def progress_callback(pct, msg):
            progress.progress(pct, text=msg)
            status.info(msg)
        
        with st.spinner("Running evaluation..."):
            report = eval_manager.run_full_evaluation(
                include_rag=eval_rag,
                include_solution=eval_solution,
                include_guardrails=eval_guardrails,
                include_memory=eval_memory,
                topic_filter=topic_filter,
                max_cases=max_cases,
                progress_callback=progress_callback
            )
        
        progress.empty()
        status.empty()
        
        # Display results
        st.markdown("---")
        
        # Overall score
        score = report.overall_score
        grade = report.overall_grade
        
        col1, col2, col3 = st.columns(3)
        with col1:
            color = "green" if score >= 70 else "orange" if score >= 50 else "red"
            st.markdown(f"### Overall: :{color}[{score:.1f}%]")
        with col2:
            st.markdown(f"### Grade: **{grade}**")
        with col3:
            st.markdown(f"### Components: **{len(report.component_reports)}**")
        
        st.markdown("---")
        
        # Component results
        for name, comp_report in report.component_reports.items():
            with st.expander(f"📊 {name} ({comp_report.overall_score:.1f}%)", expanded=True):
                
                if comp_report.metrics:
                    cols = st.columns(len(comp_report.metrics[:4]))
                    for i, metric in enumerate(comp_report.metrics[:4]):
                        with cols[i]:
                            color = "green" if metric.percentage >= 70 else "orange" if metric.percentage >= 50 else "red"
                            st.metric(
                                metric.name,
                                f"{metric.percentage:.1f}%",
                                help=metric.details
                            )
                    
                    # Show remaining metrics
                    if len(comp_report.metrics) > 4:
                        for metric in comp_report.metrics[4:]:
                            st.caption(f"**{metric.name}:** {metric.percentage:.1f}% - {metric.details}")
                
                if comp_report.errors:
                    st.warning(f"⚠️ {len(comp_report.errors)} errors during evaluation")
                    with st.expander("Show errors"):
                        for err in comp_report.errors[:10]:
                            st.caption(f"❌ {err}")
                
                if comp_report.details:
                    with st.expander("Detailed Results"):
                        st.json(comp_report.details[:10])
        
        # Download report
        st.markdown("---")
        report_md = report.to_markdown()
        st.download_button(
            "📥 Download Report (Markdown)",
            report_md,
            file_name=f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
        
        report_json = json.dumps(report.to_dict(), indent=2)
        st.download_button(
            "📥 Download Report (JSON)",
            report_json,
            file_name=f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
def render_recheck_button(solution: str, problem_text: str):
    """Render a re-check request button for the solution."""
    hitl_manager = load_hitl_manager()
    
    # Initialize session state for re-check
    if 'recheck_active' not in st.session_state:
        st.session_state.recheck_active = False
        st.session_state.recheck_request_id = None
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if not st.session_state.recheck_active:
            if st.button("🔄 Request Re-check", help="Request manual verification"):
                request = hitl_manager.create_user_recheck_request(
                    content=solution,
                    stage="solution",
                    reason="User requested verification"
                )
                st.session_state.recheck_active = True
                st.session_state.recheck_request_id = request.request_id
                st.rerun()
    
    # Show re-check UI if active
    if st.session_state.recheck_active and st.session_state.recheck_request_id:
        st.markdown("---")
        st.info("🔄 **Re-check Mode Active**")
        st.markdown("Please review the solution and provide your assessment:")
        
        feedback = st.text_area(
            "Your feedback (optional):",
            placeholder="Describe any issues or concerns...",
            key="recheck_feedback_input"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Looks Correct", type="primary"):
                hitl_manager.approve_request(
                    st.session_state.recheck_request_id,
                    feedback
                )
                st.session_state.recheck_active = False
                st.session_state.recheck_request_id = None
                st.success("✅ Marked as correct! Thanks for verifying.")
                st.rerun()
        
        with col2:
            if st.button("⚠️ Has Issues"):
                hitl_manager.resolve_request(
                    st.session_state.recheck_request_id,
                    HITLStatus.REJECTED,
                    human_feedback=feedback or "User reported issues"
                )
                st.session_state.recheck_active = False
                st.session_state.recheck_request_id = None
                st.warning("⚠️ Issue reported. Thanks for your feedback!")
                st.rerun()
        
        with col3:
            if st.button("❌ Cancel"):
                hitl_manager.skip_request(st.session_state.recheck_request_id)
                st.session_state.recheck_active = False
                st.session_state.recheck_request_id = None
                st.rerun()


def render_hitl_status():
    """Show HITL statistics in sidebar."""
    hitl_manager = load_hitl_manager()
    stats = hitl_manager.get_statistics()
    
    st.subheader("👤 HITL")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reviews", stats['total'])
    with col2:
        edit_rate = stats.get('edit_rate', 0)
        st.metric("Edits", f"{edit_rate:.0%}")
    
    # Show current state
    if st.session_state.get('hitl_done'):
        final = st.session_state.get('final_canonical')
        if final and final.was_human_edited:
            st.success("✅ Current: Edited")
        else:
            st.success("✅ Current: Reviewed")
    elif st.session_state.get('processed_canonical'):
        st.info("⏳ Current: Awaiting review")
    
    with st.expander("Session Details", expanded=False):
        if stats['by_action']:
            st.caption("**By Action:**")
            for action, count in stats['by_action'].items():
                icon = {"approved": "✅", "edited": "✏️", "rejected": "❌", "skipped": "⏭️"}.get(action, "•")
                st.caption(f"  {icon} {action}: {count}")
        
        st.caption(f"📚 Learning signals: {stats.get('learning_signals', 0)}")
# ============================================================
# UI Components
# ============================================================
def render_sidebar():
    """Render sidebar with settings."""
    with st.sidebar:
        st.title("🧮 Math Mentor")
        st.markdown("*AI-Powered JEE Math Tutor*")
        
        st.divider()
        
        # Input mode selector
        st.subheader("📥 Input Mode")
        input_mode = st.radio(
            "Choose input:",
            ["📝 Text", "📷 Image (OCR)", "🎤 Audio (ASR)"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        show_rag = st.checkbox("Show RAG Context", value=True)
        show_debug = st.checkbox("Show Debug Info", value=False)
        top_k = st.slider("RAG Results", 1, 5, 3)
        use_reranker = st.checkbox("Enable Reranking", value=True, help="Use cross-encoder to rerank results")
        
        st.divider()
        
        # API Status
        st.subheader("🔌 Status")
        
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Groq API Missing")
        
        # RAG Status
        retriever = load_retriever()
        if retriever:
            st.success("✅ RAG Knowledge Base")
            if hasattr(retriever, 'reranker') and retriever.reranker:
                st.success("✅ Reranker Active")
        else:
            st.warning("⚠️ RAG Not Available")
        
        # ASR Status
        asr = load_asr()
        if asr:
            st.success("✅ Whisper ASR Ready")
        else:
            st.warning("⚠️ Whisper ASR Not Available")
            st.caption("pip install openai-whisper")
        
        st.divider()
        
        # 🛡️ GUARDRAILS STATUS
        render_guardrails_status()
        
        st.divider()
        
        # 👤 HITL STATUS
        render_hitl_status()
        
        st.divider()
        
        # Memory Stats
        st.subheader("🧠 Memory")
        try:
            memory = load_memory()
            stats = memory.get_statistics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Solved", stats.get('total_problems', 0))
            with col2:
                feedback = stats.get('feedback', {})
                st.metric("Feedback", feedback.get('total_feedback', 0))
        except:
            st.caption("No data yet")
        
        return input_mode, show_rag, show_debug, top_k, use_reranker


def render_guardrails_status():
    """Show guardrails status in sidebar."""
    st.subheader("🛡️ Guardrails")
    
    with st.expander("Active Protections", expanded=False):
        st.markdown("""
        ✅ Input validation & sanitization  
        ✅ Prompt injection detection  
        ✅ Content safety filter  
        ✅ Math topic enforcement  
        ✅ Word problem support  
        ✅ Output validation  
        ✅ Harmful content blocking
        """)
    
    guardrails = load_guardrails()
    mode = "Strict" if guardrails.strict_mode else "Standard"
    st.caption(f"Mode: {mode}")


def render_topic_badge(topic: str):
    """Display topic badge if detected."""
    if topic and topic.lower() not in ["general", "mathematics", "approved", "low_confidence"]:
        st.success(f"📚 Detected Topic: **{topic.replace('_', ' ').title()}**")


def render_confidence_indicator(confidence: float):
    """Display confidence indicator."""
    if confidence >= 0.7:
        color = "green"
        label = "High"
    elif confidence >= 0.5:
        color = "orange"
        label = "Medium"
    else:
        color = "red"
        label = "Low"
    
    st.markdown(f"**Confidence:** :{color}[{confidence:.0%} ({label})]")


def get_file_hash(file_obj) -> str:
    """Get a hash of the uploaded file to detect changes."""
    import hashlib
    if file_obj is None:
        return ""
    content = file_obj.getvalue()
    return hashlib.md5(content).hexdigest()


def clear_hitl_state():
    """Clear HITL-related session state."""
    st.session_state['processed_canonical'] = None
    st.session_state['processed_file_hash'] = None
    st.session_state['hitl_completed'] = False
    st.session_state['current_hitl_request_id'] = None


def render_input_section(input_mode: str):
    """Render input section with HITL integration."""
    canonical = None
    guardrails = load_guardrails()
    hitl_manager = load_hitl_manager()
    
    # ==================== TEXT INPUT ====================
    if input_mode == "📝 Text":
        # Clear any file-based state
        for key in list(st.session_state.keys()):
            if key.startswith(('processed_', 'stored_text_', 'edit_text_', 'hitl_')):
                del st.session_state[key]
        
        st.subheader("📝 Enter Your Math Problem")
        
        examples = [
            "Select an example...",
            "Solve x^2 - 5x + 6 = 0",
            "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
            "Integrate 3x^2 + 2x dx",
            "What is the probability of getting exactly 2 heads in 3 coin tosses?",
            "Find the determinant of matrix [[1,2],[3,4]]",
            "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1",
            "John has 5 apples. Mary gives him 3 more. How many does he have?",
            "A train travels at 60 km/h for 2 hours. Find the distance."
        ]
        
        selected = st.selectbox("📋 Quick Examples:", examples)
        default_text = selected if selected != examples[0] else ""
        
        text_input = st.text_area(
            "Type your problem:",
            value=default_text,
            height=100,
            placeholder="e.g., Solve x^2 - 5x + 6 = 0"
        )
        
        if text_input:
            report = guardrails.check_input(text_input)
            
            if not report.passed:
                display_guardrails_error(report)
                return None
            
            display_guardrails_warnings(report)
            
            topic = report.metadata.get('category', 'general')
            render_topic_badge(topic)
            
            confidence = report.metadata.get('math_confidence', 1.0)
            if confidence < 0.8:
                st.caption(f"Math confidence: {confidence:.0%}")
            
            canonical = process_text_input(text_input)
            
            with st.expander("🔍 Processed Input", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Original:")
                    st.code(canonical.original_extraction)
                with col2:
                    st.caption("Normalized:")
                    st.code(canonical.extracted_text)
    
    # ==================== IMAGE INPUT ====================
    elif input_mode == "📷 Image (OCR)":
        st.subheader("📷 Upload Math Problem Image")
        
        uploaded_file = st.file_uploader(
            "Upload image:",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a math problem",
            key="image_uploader"
        )
        
        if uploaded_file:
            file_id = f"image_{uploaded_file.name}_{uploaded_file.size}"
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                # Check if we need to process this file
                if st.session_state.get('processed_file_id') != file_id:
                    # New file - process it
                    with st.spinner("🔍 Running OCR..."):
                        canonical = process_image_input(uploaded_file)
                    
                    st.session_state['processed_file_id'] = file_id
                    st.session_state['processed_canonical'] = canonical
                    st.session_state['hitl_done'] = False
                else:
                    # Same file - use cached canonical
                    canonical = st.session_state.get('processed_canonical')
                    
                    if canonical is None:
                        with st.spinner("🔍 Running OCR..."):
                            canonical = process_image_input(uploaded_file)
                        st.session_state['processed_canonical'] = canonical
                
                # Check for errors
                if canonical.metadata.get('error'):
                    st.error(f"❌ OCR Error: {canonical.metadata['error']}")
                    return None
                
                # Check if HITL already completed
                if st.session_state.get('hitl_done') and st.session_state.get('final_canonical'):
                    final_canonical = st.session_state['final_canonical']
                    
                    st.success("✅ Extraction reviewed and ready")
                    render_confidence_indicator(final_canonical.confidence_score)
                    
                    if final_canonical.was_human_edited:
                        st.info("✏️ Text was edited by user")
                    
                    st.text_area(
                        "Final text:",
                        value=final_canonical.extracted_text,
                        height=100,
                        disabled=True,
                        key="final_display_image"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔄 Edit Again", key="reedit_image"):
                            st.session_state['hitl_done'] = False
                            # Clear stored text so we start fresh
                            for k in list(st.session_state.keys()):
                                if k.startswith('stored_text_image') or k.startswith('edit_text_image'):
                                    del st.session_state[k]
                            st.rerun()
                    
                    # Validate and return
                    report = guardrails.check_input(final_canonical.extracted_text)
                    if report.passed:
                        topic = report.metadata.get('category', 'general')
                        render_topic_badge(topic)
                        return final_canonical
                    else:
                        display_guardrails_error(report)
                        return None
                
                # HITL not done - show HITL UI
                hitl_request = hitl_manager.check_extraction_hitl(
                    input_type="image",
                    extracted_text=canonical.extracted_text,
                    confidence=canonical.confidence_score,
                    metadata=canonical.metadata
                )
                
                result = render_extraction_hitl(canonical, hitl_request, "image")
                
                if result is not None:
                    st.session_state['hitl_done'] = True
                    st.session_state['final_canonical'] = result
                    st.session_state['processed_canonical'] = result
                    return result
                else:
                    return None
        else:
            # Clear state when no file
            for key in ['processed_file_id', 'processed_canonical', 'hitl_done', 'final_canonical']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # ==================== AUDIO INPUT ====================
    elif input_mode == "🎤 Audio (ASR)":
        st.subheader("🎤 Speak Your Math Problem")
        
        asr = load_asr()
        if asr is None:
            st.error("❌ Whisper ASR is not installed!")
            st.code("pip install openai-whisper")
            return None
        
        audio_tab1, audio_tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])
        
        audio_source = None
        audio_id = None
        
        with audio_tab1:
            uploaded_audio = st.file_uploader(
                "Upload audio file:",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                help="Upload an audio file with your math question",
                key="audio_uploader"
            )
            if uploaded_audio:
                st.audio(uploaded_audio, format="audio/wav")
                audio_source = uploaded_audio
                audio_id = f"audio_{uploaded_audio.name}_{uploaded_audio.size}"
        
        with audio_tab2:
            st.info("🎙️ Click below to record your question")
            recorded_audio = st.audio_input("Record your math question:", key="audio_recorder")
            if recorded_audio:
                audio_source = recorded_audio
                audio_id = f"recorded_{len(recorded_audio.getvalue())}"
        
        if audio_source and audio_id:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Audio Input:**")
                if hasattr(audio_source, 'name'):
                    st.caption(f"File: {audio_source.name}")
                else:
                    st.caption("Recorded audio")
            
            with col2:
                # Check if we need to process this audio
                if st.session_state.get('processed_file_id') != audio_id:
                    # New audio - process it
                    with st.spinner("🎤 Transcribing audio..."):
                        canonical = process_audio_input(audio_source)
                    
                    st.session_state['processed_file_id'] = audio_id
                    st.session_state['processed_canonical'] = canonical
                    st.session_state['hitl_done'] = False
                else:
                    # Same audio - use cached canonical
                    canonical = st.session_state.get('processed_canonical')
                    
                    if canonical is None:
                        with st.spinner("🎤 Transcribing audio..."):
                            canonical = process_audio_input(audio_source)
                        st.session_state['processed_canonical'] = canonical
                
                # Check for errors
                if canonical.metadata.get('error'):
                    st.error(f"❌ ASR Error: {canonical.metadata['error']}")
                    return None
                
                # Check if HITL already completed
                if st.session_state.get('hitl_done') and st.session_state.get('final_canonical'):
                    final_canonical = st.session_state['final_canonical']
                    
                    st.success("✅ Transcription reviewed and ready")
                    render_confidence_indicator(final_canonical.confidence_score)
                    
                    if final_canonical.was_human_edited:
                        st.info("✏️ Text was edited by user")
                    
                    st.text_area(
                        "Final text:",
                        value=final_canonical.extracted_text,
                        height=100,
                        disabled=True,
                        key="final_display_audio"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔄 Edit Again", key="reedit_audio"):
                            st.session_state['hitl_done'] = False
                            # Clear stored text
                            for k in list(st.session_state.keys()):
                                if k.startswith('stored_text_audio') or k.startswith('edit_text_audio'):
                                    del st.session_state[k]
                            st.rerun()
                    
                    # Show math conversion info
                    with st.expander("🔢 Math Phrase Conversions"):
                        st.markdown("""
                        - "square root of" → √
                        - "x squared" → x²
                        - "raised to the power of" → ^
                        - "integral of" → ∫
                        - "derivative of" → d/dx
                        """)
                    
                    # Validate and return
                    report = guardrails.check_input(final_canonical.extracted_text)
                    if report.passed:
                        topic = report.metadata.get('category', 'general')
                        render_topic_badge(topic)
                        return final_canonical
                    else:
                        display_guardrails_error(report)
                        return None
                
                # HITL not done - show HITL UI
                hitl_request = hitl_manager.check_extraction_hitl(
                    input_type="audio",
                    extracted_text=canonical.extracted_text,
                    confidence=canonical.confidence_score,
                    metadata=canonical.metadata
                )
                
                result = render_extraction_hitl(canonical, hitl_request, "audio")
                
                if result is not None:
                    st.session_state['hitl_done'] = True
                    st.session_state['final_canonical'] = result
                    st.session_state['processed_canonical'] = result
                    
                    # Show math conversion info
                    with st.expander("🔢 Math Phrase Conversions Applied"):
                        st.markdown("""
                        - "square root of" → √
                        - "x squared" → x²
                        - "raised to the power of" → ^
                        - "integral of" → ∫
                        - "derivative of" → d/dx
                        """)
                    
                    return result
                else:
                    return None
        else:
            # Clear state when no audio
            for key in ['processed_file_id', 'processed_canonical', 'hitl_done', 'final_canonical']:
                if key in st.session_state:
                    del st.session_state[key]
    
    return canonical


def render_solution_section(
    canonical: CanonicalInput,
    show_rag: bool,
    show_debug: bool,
    top_k: int,
    use_reranker: bool = True
):
    """Render solution with output guardrails, HITL, and per-query evaluation."""
    
    def render_rag_context(rag_results: List[Dict]):
        """Render RAG context with improved display."""
        if not rag_results:
            st.info("No relevant knowledge found in the database.")
            return
        
        with st.expander(f"📚 Retrieved Knowledge ({len(rag_results)} sources)", expanded=False):
            for i, doc in enumerate(rag_results, 1):
                score = doc.get('score', 0)
                rerank_score = doc.get('rerank_score', 0)
                
                # Score badge
                if score >= 0.6:
                    score_badge = "🟢 High"
                elif score >= 0.4:
                    score_badge = "🟡 Medium"
                else:
                    score_badge = "🔴 Low"
                
                # Header with score
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**[{i}] {doc.get('title', 'Content')}** ({doc.get('topic', '')})")
                with col2:
                    st.caption(f"{score_badge} {score:.0%}")
                
                # Show rerank score if available
                if rerank_score > 0:
                    st.caption(f"🎯 Rerank Score: {rerank_score:.0%}")
                
                # Chapter/section info
                chapter = doc.get('chapter', '')
                section = doc.get('section', '')
                if chapter or section:
                    st.caption(f"📖 {chapter}" + (f" > {section}" if section else ""))
                
                # Content
                content = doc.get('content', '')
                if len(content) > 600:
                    preview = content[:500].rsplit(' ', 1)[0] + "..."
                    st.markdown(preview)
                    with st.expander("📄 Show full content"):
                        st.markdown(content)
                else:
                    st.markdown(content)
                
                if i < len(rag_results):
                    st.divider()
    
    guardrails = load_guardrails()
    hitl_manager = load_hitl_manager()
    
    if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
        st.session_state['current_canonical'] = canonical
        
        # Reset re-check state
        st.session_state.recheck_active = False
        st.session_state.recheck_request_id = None
        
        progress = st.progress(0, text="Starting...")
        status = st.empty()
        
        # Track total time for evaluation
        import time
        solve_start_time = time.time()
        
        # ==================== Step 1: RAG Retrieval ====================
        rerank_text = " with reranking" if use_reranker else ""
        status.info(f"📚 Searching knowledge base{rerank_text}...")
        progress.progress(15)
        
        retriever = load_retriever()
        
        # Get expanded query info for debugging
        from src.rag.query_expander import get_query_expander
        expander = get_query_expander()
        expanded_query = expander.expand(canonical.extracted_text)
        
        rag_results = retriever.retrieve(
            canonical.extracted_text,
            top_k=top_k,
            min_score=0.2,
            rerank=use_reranker
        )
        
        st.session_state['rag_context'] = rag_results
        
        # ==================== Step 2: Memory Lookup ====================
        status.info("🧠 Checking memory for similar problems...")
        progress.progress(30)
        memory = load_memory()
        similar = memory.find_similar_problems(canonical.extracted_text, limit=2)
        
        # ==================== Step 3: LLM Solve ====================
        status.info("🤖 Solving with AI...")
        progress.progress(45)
        result = solve_with_llm(canonical.extracted_text, rag_results, similar)
        
        # ==================== Step 4: Output Guardrails ====================
        status.info("🛡️ Validating response...")
        progress.progress(60)
        
        output_report = guardrails.check_output(result)
        
        if not output_report.passed:
            progress.empty()
            status.empty()
            
            st.error(f"🛡️ **Response Filtered:** {guardrails.get_rejection_message(output_report)}")
            st.warning("The AI response didn't pass safety checks. Please try rephrasing your question.")
            
            if show_debug:
                with st.expander("🐛 Debug Info"):
                    st.json({
                        "status": output_report.status.value,
                        "blocked_reason": output_report.blocked_reason,
                        "warnings": output_report.warnings
                    })
            return
        
        # Show output warnings
        display_guardrails_warnings(output_report)
        
        # Format output safely
        result = guardrails.format_output_for_display(result)
        
        # ==================== Step 5: Save to Memory ====================
        status.info("💾 Saving to memory...")
        progress.progress(75)
        
        topic = rag_results[0].get('topic') if rag_results else None
        problem_id = save_to_memory(canonical, rag_results, result, topic)
        st.session_state['problem_id'] = problem_id
        st.session_state['result'] = result
        
        # ==================== Step 6: Per-Query Evaluation ====================
        status.info("📊 Evaluating response quality...")
        progress.progress(88)
        
        from src.evaluation.evaluator_agent import get_evaluator_agent
        
        eval_agent = get_evaluator_agent()
        total_latency = (time.time() - solve_start_time) * 1000  # ms
        
        try:
            query_eval = eval_agent.evaluate_query(
                question=canonical.extracted_text,
                solution=result.get('solution', ''),
                explanation=result.get('explanation', ''),
                verification=result.get('verification', ''),
                rag_context=rag_results,
                input_type=canonical.input_type,
                confidence=float(canonical.confidence_score),
                was_human_edited=canonical.was_human_edited,
                latency_ms=total_latency,
                topic=topic or "",
                query_id=problem_id,
            )
            st.session_state['last_evaluation'] = query_eval
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            st.session_state['last_evaluation'] = None
        
        # ==================== Done ====================
        progress.progress(100, text="✅ Complete!")
        status.empty()
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        st.divider()
        
        # ---- Status Badges Row ----
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            input_badges = {
                "text": "📝 Text",
                "image": "📷 OCR",
                "audio": "🎤 ASR"
            }
            st.caption(f"Input: {input_badges.get(canonical.input_type, 'Unknown')}")
        with col2:
            st.caption(f"RAG Docs: {len(rag_results)}")
        with col3:
            rerank_status = "🎯 On" if use_reranker else "Off"
            st.caption(f"Rerank: {rerank_status}")
        with col4:
            status_color = "green" if output_report.status == CheckStatus.PASSED else "orange"
            st.caption(f"Validation: :{status_color}[{output_report.status.value}]")
        with col5:
            # Show eval grade
            query_eval = st.session_state.get('last_evaluation')
            if query_eval:
                eval_color = "green" if query_eval.overall_score >= 0.7 else "orange" if query_eval.overall_score >= 0.5 else "red"
                st.caption(f"Quality: :{eval_color}[{query_eval.grade}]")
            else:
                st.caption("Quality: N/A")
        
        # ---- HITL Info ----
        if canonical.was_human_edited:
            st.caption("✏️ Input was human-edited")
        
        # ---- Debug Info ----
        if show_debug:
            with st.expander("🐛 Debug Info"):
                debug_data = {
                    "input_type": canonical.input_type,
                    "confidence": float(canonical.confidence_score),
                    "was_edited": canonical.was_human_edited,
                    "hitl_reviewed": canonical.hitl_info.was_reviewed,
                    "hitl_action": canonical.hitl_info.review_action,
                    "rag_docs": len(rag_results),
                    "reranker_enabled": use_reranker,
                    "similar_problems": len(similar),
                    "guardrails_status": output_report.status.value,
                    "guardrails_warnings": output_report.warnings,
                    "metadata": output_report.metadata,
                    "total_latency_ms": round(total_latency, 1),
                }
                
                # Add evaluation data if available
                query_eval = st.session_state.get('last_evaluation')
                if query_eval:
                    debug_data["evaluation"] = {
                        "overall_score": round(query_eval.overall_score, 3),
                        "grade": query_eval.grade,
                        "rag_relevance": round(query_eval.rag_relevance_score, 3),
                        "solution_quality": round(query_eval.solution_quality_score, 3),
                        "explanation_clarity": round(query_eval.explanation_clarity_score, 3),
                        "issues": query_eval.issues_found,
                        "suggestions": query_eval.suggestions,
                    }
                
                st.json(debug_data)
                
                # Show RAG scores
                if rag_results:
                    st.markdown("**RAG Scores:**")
                    for i, r in enumerate(rag_results[:5]):
                        st.caption(
                            f"{i+1}. {r.get('title', 'Unknown')}: "
                            f"combined={r.get('score', 0):.3f}, "
                            f"rerank={r.get('rerank_score', 0):.3f}"
                        )
                
                if result.get('raw_response'):
                    st.text_area(
                        "Raw Response",
                        result.get('raw_response', '')[:2000],
                        height=200
                    )
        
        # ---- Error Display ----
        if result.get('error'):
            st.error(f"⚠️ {result['error']}")
        
        # ---- RAG Context ----
        if show_rag:
            render_rag_context(rag_results)
        
        # ---- Solution ----
        st.subheader("📝 Solution")
        solution = result.get('solution', 'No solution generated')
        if solution:
            st.markdown(solution)
        
        # ---- Re-check Button ----
        render_recheck_button(solution, canonical.extracted_text)
        
        # ---- Verification ----
        verification = result.get('verification', '')
        if verification and verification != "N/A":
            st.subheader("✅ Verification")
            st.markdown(verification)
        
        # ---- Explanation ----
        explanation = result.get('explanation', '')
        if explanation:
            st.subheader("📖 Explanation")
            st.markdown(explanation)
        
        st.divider()
        
        # ---- Per-Query Evaluation Card ----
        query_eval = st.session_state.get('last_evaluation')
        render_query_evaluation(query_eval)
        
        st.divider()
        
        # ---- Feedback Section ----
        render_feedback_section()

def render_query_evaluation(evaluation):
    """Display per-query evaluation results."""
    from src.evaluation.evaluator_agent import QueryEvaluation
    
    if evaluation is None:
        return
    
    with st.expander(
        f"📊 Response Quality: {evaluation.grade} ({evaluation.overall_score:.0%})",
        expanded=False
    ):
        # Score cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "green" if evaluation.overall_score >= 0.7 else "orange" if evaluation.overall_score >= 0.5 else "red"
            st.metric("Overall", f"{evaluation.overall_score:.0%}")
        with col2:
            st.metric("RAG Relevance", f"{evaluation.rag_relevance_score:.0%}")
        with col3:
            st.metric("Solution", f"{evaluation.solution_quality_score:.0%}")
        with col4:
            st.metric("Explanation", f"{evaluation.explanation_clarity_score:.0%}")
        
        # Assessments
        if evaluation.solution_assessment:
            st.markdown(f"**Solution:** {evaluation.solution_assessment}")
        if evaluation.explanation_assessment:
            st.markdown(f"**Explanation:** {evaluation.explanation_assessment}")
        
        # Issues
        if evaluation.issues_found:
            st.warning("**Issues Found:**")
            for issue in evaluation.issues_found:
                st.markdown(f"- ⚠️ {issue}")
        
        # Suggestions
        if evaluation.suggestions:
            st.info("**Suggestions:**")
            for suggestion in evaluation.suggestions:
                st.markdown(f"- 💡 {suggestion}")


def render_feedback_section():
    """Render feedback buttons."""
    st.subheader("📊 Was this helpful?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("✅ Correct", type="primary", use_container_width=True):
            if 'problem_id' in st.session_state:
                memory = load_memory()
                memory.record_feedback(st.session_state['problem_id'], is_correct=True)
                st.success("Thanks! 🎉")
    
    with col2:
        if st.button("❌ Incorrect", use_container_width=True):
            st.session_state['show_feedback_form'] = True
    
    if st.session_state.get('show_feedback_form'):
        with st.form("feedback_form"):
            comment = st.text_area("What was wrong?", placeholder="Describe the issue...")
            correct_answer = st.text_area("Correct answer (optional):", placeholder="Provide the correct solution...")
            
            if st.form_submit_button("Submit Feedback", type="primary"):
                if 'problem_id' in st.session_state:
                    memory = load_memory()
                    memory.record_feedback(
                        st.session_state['problem_id'],
                        is_correct=False,
                        comment=comment,
                        corrected_solution=correct_answer
                    )
                    st.success("Feedback recorded! We'll use this to improve. 📝")
                    st.session_state['show_feedback_form'] = False
                    st.rerun()


def render_history_tab():
    """Render history tab."""
    st.subheader("📜 Recent Problems")
    
    memory = load_memory()
    history = memory.get_problem_history(limit=10)
    
    if not history:
        st.info("No problems solved yet! Start by entering a math problem.")
        return
    
    for i, problem in enumerate(history, 1):
        input_icon = {"text": "📝", "image": "📷", "audio": "🎤"}.get(problem.get('input_type', ''), "❓")
        extracted_text = problem.get('extracted_text', 'Unknown problem')
        
        with st.expander(f"{input_icon} {i}. {extracted_text[:50]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Input Type:** {problem.get('input_type', 'Unknown')}")
                st.markdown(f"**Topic:** {problem.get('topic', 'Unknown')}")
            with col2:
                confidence = problem.get('confidence_score', 0)
                st.markdown(f"**Confidence:** {confidence:.0%}")
                st.caption(f"Solved: {problem.get('created_at', 'Unknown')}")
            
            st.markdown("**Problem:**")
            st.markdown(extracted_text)
            
            solution = problem.get('solution', 'N/A')
            st.markdown("**Solution:**")
            st.markdown(solution[:500] + "..." if len(solution) > 500 else solution)


# ============================================================
# Main App
# ============================================================
def main():
    """Main application."""
    
    # Initialize session state
    if 'show_feedback_form' not in st.session_state:
        st.session_state['show_feedback_form'] = False
    if 'recheck_active' not in st.session_state:
        st.session_state['recheck_active'] = False
    if 'recheck_request_id' not in st.session_state:
        st.session_state['recheck_request_id'] = None
    
    # HITL state management
    if 'processed_canonical' not in st.session_state:
        st.session_state['processed_canonical'] = None
    if 'processed_file_hash' not in st.session_state:
        st.session_state['processed_file_hash'] = None
    if 'hitl_completed' not in st.session_state:
        st.session_state['hitl_completed'] = False
    if 'current_hitl_request_id' not in st.session_state:
        st.session_state['current_hitl_request_id'] = None
    
    # Render sidebar and get settings
    input_mode, show_rag, show_debug, top_k, use_reranker = render_sidebar()
    
    # Main content
    st.title("🧮 Math Mentor")
    st.markdown("*Your AI-powered JEE Math Tutor with HITL & Safety Guardrails*")
    
    # Tabs - ADD EVALUATION TAB
    tab1, tab2, tab3 = st.tabs(["🆕 Solve Problem", "📜 History", "📊 Evaluation"])
    
    with tab1:
        canonical = render_input_section(input_mode)
        
        if canonical and canonical.extracted_text:
            st.divider()
            render_solution_section(canonical, show_rag, show_debug, top_k, use_reranker)
    
    with tab2:
        render_history_tab()
    
    with tab3:
        render_evaluation_tab()
    
    # Footer
    st.divider()
    st.caption("🛡️ All inputs/outputs validated by guardrails | 👤 HITL for low-confidence extractions | 🧠 Learning from corrections")
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()