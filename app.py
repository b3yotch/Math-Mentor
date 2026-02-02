# app.py
"""
Math Mentor - AI-Powered JEE Math Tutor
Complete Version with Text, Image, and Audio Input + Guardrails
"""

import streamlit as st
import os
import sys
import tempfile
import json
from datetime import datetime
from typing import Tuple, Optional
from dotenv import load_dotenv

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

# ============================================================
# Imports from our modules
# ============================================================
from src.input_processing.schemas import CanonicalInput
from src.input_processing.math_normalizer import MathNormalizer
from src.input_processing.ocr_processor import OCRProcessor
from src.rag.retriever import RAGRetriever
from src.memory.memory_manager import MemoryManager
from src.guardrails.guardrails_manager import GuardrailsManager, GuardrailsReport


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
        st.warning(f"ASR not available: {e}")
        return None

@st.cache_resource
def load_normalizer():
    """Load math normalizer (cached)."""
    return MathNormalizer()

@st.cache_resource
def load_retriever():
    """Load RAG retriever (cached)."""
    return RAGRetriever()

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
    """Load guardrails manager."""
    return GuardrailsManager()


# ============================================================
# Input Processing with Guardrails
# ============================================================
def process_input_with_guardrails(text: str, input_type: str) -> Tuple[Optional[CanonicalInput], GuardrailsReport]:
    """
    Process input with guardrails validation.
    
    Returns:
        Tuple of (CanonicalInput or None, GuardrailsReport)
    """
    guardrails = load_guardrails()
    
    # Run input guardrails
    report = guardrails.check_input(text)
    
    if not report.passed:
        return None, report
    
    # Process if passed
    if input_type == "text":
        canonical = process_text_input(text)
    else:
        canonical = CanonicalInput(
            input_type=input_type,
            extracted_text=text,
            confidence_score=1.0
        )
    
    return canonical, report


# ============================================================
# Display Guardrails Status in Sidebar
# ============================================================
def render_guardrails_status():
    """Show guardrails status in sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("🛡️ Guardrails")
    
    guardrails = load_guardrails()
    
    with st.sidebar.expander("Active Protections"):
        st.markdown("""
        - ✅ Input validation
        - ✅ Prompt injection detection
        - ✅ Content safety filter
        - ✅ Topic enforcement
        - ✅ Output validation
        - ✅ Hallucination detection
        """)
    
    st.sidebar.caption("All inputs and outputs are validated")


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


def get_rag_context(query: str, top_k: int = 3):
    """Get relevant context from RAG."""
    retriever = load_retriever()
    return retriever.retrieve(query, top_k=top_k)


def save_to_memory(canonical: CanonicalInput, rag_context, result, topic=None):
    """Save solved problem to memory."""
    memory = load_memory()
    
    problem_id = memory.save_solved_problem(
        input_type=canonical.input_type,
        original_input=canonical.original_extraction,
        extracted_text=canonical.extracted_text,
        topic=topic,
        solution=result.get('solution', ''),
        explanation=result.get('explanation', ''),
        verification_result=result.get('verification'),
        rag_context=rag_context,
        confidence_score=canonical.confidence_score,
        was_human_edited=canonical.was_human_edited
    )
    
    return problem_id


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
        
        st.divider()
        
        # API Status
        st.subheader("🔌 Status")
        
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Groq API Missing")
        
        asr = load_asr()
        if asr:
            st.success("✅ Whisper ASR Ready")
        else:
            st.warning("⚠️ Whisper ASR Not Available")
            st.caption("pip install openai-whisper")
        
        # 🛡️ GUARDRAILS STATUS
        render_guardrails_status()
        
        st.divider()
        
        # Memory Stats
        st.subheader("🧠 Memory")
        try:
            memory = load_memory()
            stats = memory.get_statistics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Solved", stats['total_problems'])
            with col2:
                st.metric("Feedback", stats['feedback']['total_feedback'])
        except:
            st.caption("No data yet")
        
        return input_mode, show_rag, show_debug, top_k


def render_input_section(input_mode: str):
    """Render input section with guardrails."""
    canonical = None
    guardrails = load_guardrails()
    
    # ==================== TEXT INPUT ====================
    if input_mode == "📝 Text":
        st.subheader("📝 Enter Your Math Problem")
        
        examples = [
            "Select an example...",
            "Solve x^2 - 5x + 6 = 0",
            "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
            "Integrate 3x^2 + 2x dx",
            "What is the probability of getting exactly 2 heads in 3 coin tosses?",
            "Find the determinant of matrix [[1,2],[3,4]]",
            "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1"
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
            # 🛡️ GUARDRAILS CHECK
            report = guardrails.check_input(text_input)
            
            if not report.passed:
                st.error(f"🛡️ **Input Blocked:** {report.blocked_reason}")
                
                # Show helpful suggestions
                if report.input_check and report.input_check.suggestions:
                    st.info("💡 **Suggestions:**")
                    for suggestion in report.input_check.suggestions:
                        st.markdown(f"- {suggestion}")
                
                return None
            
            # Show warnings
            for warning in report.warnings:
                st.warning(f"⚠️ {warning}")
            
            # Show detected topic badge
            topic = guardrails.get_detected_topic(text_input)
            if topic != "general":
                st.success(f"📚 Topic: **{topic.replace('_', ' ').title()}**")
            
            # Process input
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
            help="Upload a clear image of a math problem"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner("🔍 Running OCR..."):
                    canonical = process_image_input(uploaded_file)
                
                # 🛡️ GUARDRAILS CHECK on extracted text
                if canonical.extracted_text:
                    report = guardrails.check_input(canonical.extracted_text)
                    
                    if not report.passed:
                        st.error(f"🛡️ **Input Blocked:** {report.blocked_reason}")
                        if report.input_check and report.input_check.suggestions:
                            st.info("💡 **Suggestions:**")
                            for suggestion in report.input_check.suggestions:
                                st.markdown(f"- {suggestion}")
                        return None
                    
                    # Show warnings
                    for warning in report.warnings:
                        st.warning(f"⚠️ {warning}")
                    
                    # Show detected topic
                    topic = guardrails.get_detected_topic(canonical.extracted_text)
                    if topic != "general":
                        st.success(f"📚 Topic: **{topic.replace('_', ' ').title()}**")
                
                # Confidence indicator
                conf_color = "green" if canonical.confidence_score > 0.7 else "orange" if canonical.confidence_score > 0.5 else "red"
                st.markdown(f"**Confidence:** :{conf_color}[{canonical.confidence_score:.0%}]")
                
                # Editable text
                edited_text = st.text_area(
                    "Edit extracted text if needed:",
                    value=canonical.extracted_text,
                    height=100,
                    key="ocr_edit"
                )
                
                if edited_text != canonical.extracted_text:
                    # Re-check guardrails on edited text
                    edit_report = guardrails.check_input(edited_text)
                    if not edit_report.passed:
                        st.error(f"🛡️ **Edit Blocked:** {edit_report.blocked_reason}")
                        return None
                    
                    memory = load_memory()
                    memory.learn_extraction_correction('image', canonical.extracted_text, edited_text)
                    canonical.original_extraction = canonical.extracted_text
                    canonical.extracted_text = edited_text
                    canonical.was_human_edited = True
                    st.success("✏️ Correction saved for learning!")
                
                if canonical.needs_hitl():
                    st.warning("⚠️ Low confidence - please verify the text above")
    
    # ==================== AUDIO INPUT ====================
    elif input_mode == "🎤 Audio (ASR)":
        st.subheader("🎤 Speak Your Math Problem")
        
        asr = load_asr()
        if asr is None:
            st.error("❌ Whisper ASR is not installed!")
            st.code("pip install openai-whisper")
            st.info("After installing, restart the app.")
            return None
        
        audio_tab1, audio_tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])
        
        audio_source = None
        
        with audio_tab1:
            uploaded_audio = st.file_uploader(
                "Upload audio file:",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                help="Upload an audio file with your math question"
            )
            if uploaded_audio:
                st.audio(uploaded_audio, format="audio/wav")
                audio_source = uploaded_audio
        
        with audio_tab2:
            st.info("🎙️ Click below to record your question")
            recorded_audio = st.audio_input("Record your math question:")
            if recorded_audio:
                audio_source = recorded_audio
        
        if audio_source:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Audio Input:**")
                if hasattr(audio_source, 'name'):
                    st.caption(f"File: {audio_source.name}")
                else:
                    st.caption("Recorded audio")
            
            with col2:
                with st.spinner("🎤 Transcribing audio..."):
                    canonical = process_audio_input(audio_source)
                
                if canonical.metadata.get('error'):
                    st.error(canonical.metadata['error'])
                else:
                    # 🛡️ GUARDRAILS CHECK on transcribed text
                    if canonical.extracted_text:
                        report = guardrails.check_input(canonical.extracted_text)
                        
                        if not report.passed:
                            st.error(f"🛡️ **Input Blocked:** {report.blocked_reason}")
                            if report.input_check and report.input_check.suggestions:
                                st.info("💡 **Suggestions:**")
                                for suggestion in report.input_check.suggestions:
                                    st.markdown(f"- {suggestion}")
                            return None
                        
                        for warning in report.warnings:
                            st.warning(f"⚠️ {warning}")
                        
                        topic = guardrails.get_detected_topic(canonical.extracted_text)
                        if topic != "general":
                            st.success(f"📚 Topic: **{topic.replace('_', ' ').title()}**")
                    
                    conf_color = "green" if canonical.confidence_score > 0.7 else "orange" if canonical.confidence_score > 0.5 else "red"
                    st.markdown(f"**Confidence:** :{conf_color}[{canonical.confidence_score:.0%}]")
                    
                    if canonical.original_extraction:
                        st.caption("Raw transcription:")
                        st.text(canonical.original_extraction)
                    
                    edited_text = st.text_area(
                        "Edit transcription if needed:",
                        value=canonical.extracted_text,
                        height=100,
                        key="asr_edit"
                    )
                    
                    if edited_text != canonical.extracted_text:
                        # Re-check guardrails on edited text
                        edit_report = guardrails.check_input(edited_text)
                        if not edit_report.passed:
                            st.error(f"🛡️ **Edit Blocked:** {edit_report.blocked_reason}")
                            return None
                        
                        memory = load_memory()
                        memory.learn_extraction_correction('audio', canonical.extracted_text, edited_text)
                        canonical.original_extraction = canonical.extracted_text
                        canonical.extracted_text = edited_text
                        canonical.was_human_edited = True
                        st.success("✏️ Correction saved for learning!")
                    
                    if canonical.needs_hitl():
                        st.warning("⚠️ Low confidence - please verify the text above")
                    
                    with st.expander("🔢 Math Phrase Conversions Applied"):
                        st.markdown("""
                        The following spoken phrases are automatically converted:
                        - "square root of" → √
                        - "x squared" → x^2
                        - "raised to the power of" → ^
                        - "integral of" → ∫
                        - "derivative of" → d/dx
                        - "theta", "pi", "alpha" → θ, π, α
                        - "divided by" → /
                        - "times" → *
                        """)
    
    return canonical


def render_solution_section(canonical: CanonicalInput, show_rag: bool, show_debug: bool, top_k: int):
    """Render solution with output guardrails."""
    guardrails = load_guardrails()
    
    if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
        st.session_state['current_canonical'] = canonical
        
        progress = st.progress(0, text="Starting...")
        status = st.empty()
        
        # Step 1: RAG
        status.info("📚 Searching knowledge base...")
        progress.progress(20)
        rag_context = get_rag_context(canonical.extracted_text, top_k=top_k)
        st.session_state['rag_context'] = rag_context
        
        # Step 2: Memory
        status.info("🧠 Checking memory for similar problems...")
        progress.progress(35)
        memory = load_memory()
        similar = memory.find_similar_problems(canonical.extracted_text, limit=2)
        
        # Step 3: Solve
        status.info("🤖 Solving with AI...")
        progress.progress(50)
        result = solve_with_llm(canonical.extracted_text, rag_context, similar)
        
        # 🛡️ OUTPUT GUARDRAILS
        status.info("🛡️ Validating response...")
        progress.progress(70)
        
        # ✅ FIX: Pass the entire result dictionary, not just the solution string
        output_report = guardrails.check_output(result)
        
        if not output_report.passed:
            progress.empty()
            status.empty()
            st.error(f"🛡️ **Response Filtered:** {output_report.blocked_reason}")
            st.warning("The AI response didn't pass safety checks. Please try rephrasing your question.")
            
            if show_debug:
                with st.expander("🐛 Debug Info"):
                    st.json({
                        "errors": output_report.output_check.errors if output_report.output_check else [],
                        "warnings": output_report.warnings
                    })
            return
        
        # Show output warnings
        if output_report.warnings:
            for warning in output_report.warnings:
                st.warning(f"⚠️ {warning}")
        
        # Format output safely
        result = guardrails.format_output(result)
        
        # Step 4: Save
        status.info("💾 Saving to memory...")
        progress.progress(85)
        
        topic = rag_context[0].get('topic') if rag_context else None
        problem_id = save_to_memory(canonical, rag_context, result, topic)
        st.session_state['problem_id'] = problem_id
        st.session_state['result'] = result
        
        progress.progress(100, text="✅ Complete!")
        status.empty()
        
        # Display Results
        st.divider()
        
        # Guardrails confidence badge
        if output_report.output_check and hasattr(output_report.output_check, 'confidence'):
            conf = output_report.output_check.confidence
            conf_color = "green" if conf > 0.8 else "orange" if conf > 0.5 else "red"
            st.markdown(f"🛡️ Response Confidence: :{conf_color}[{conf:.0%}]")
        
        # Input type badge
        input_badges = {
            "text": "📝 Text Input",
            "image": "📷 Image Input (OCR)",
            "audio": "🎤 Audio Input (ASR)"
        }
        st.caption(input_badges.get(canonical.input_type, "Unknown"))
        
        # Debug info
        if show_debug:
            with st.expander("🐛 Debug Info"):
                st.json({
                    "input_type": canonical.input_type,
                    "confidence": canonical.confidence_score,
                    "was_edited": canonical.was_human_edited,
                    "rag_docs": len(rag_context),
                    "similar_problems": len(similar),
                    "guardrails_passed": output_report.passed,
                    "guardrails_warnings": output_report.warnings
                })
                if result.get('raw_response'):
                    st.code(result.get('raw_response', '')[:1000])
        
        # Error display
        if result.get('error'):
            st.error(f"⚠️ {result['error']}")
        
        # RAG Context
        if show_rag and rag_context:
            with st.expander("📚 Retrieved Knowledge", expanded=False):
                for i, doc in enumerate(rag_context, 1):
                    st.markdown(f"**[{i}] {doc['title']}** ({doc['topic']})")
                    st.caption(doc['content'][:300] + "...")
                    if i < len(rag_context):
                        st.divider()
        
        # Solution
        st.subheader("📝 Solution")
        solution = result.get('solution', 'No solution generated')
        if solution:
            st.markdown(solution)
        
        # Verification
        verification = result.get('verification', '')
        if verification:
            st.subheader("✅ Verification")
            st.markdown(verification)
        
        # Explanation
        explanation = result.get('explanation', '')
        if explanation:
            st.subheader("📖 Explanation")
            st.markdown(explanation)
        
        st.divider()
        
        # Feedback
        render_feedback_section()


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
            comment = st.text_area("What was wrong?")
            correct_answer = st.text_area("Correct answer (optional):")
            
            if st.form_submit_button("Submit Feedback"):
                if 'problem_id' in st.session_state:
                    memory = load_memory()
                    memory.record_feedback(
                        st.session_state['problem_id'],
                        is_correct=False,
                        comment=comment,
                        corrected_solution=correct_answer
                    )
                    st.success("Feedback recorded! 📝")
                    st.session_state['show_feedback_form'] = False


def render_history_tab():
    """Render history tab."""
    st.subheader("📜 Recent Problems")
    
    memory = load_memory()
    history = memory.get_problem_history(limit=10)
    
    if not history:
        st.info("No problems solved yet!")
        return
    
    for i, problem in enumerate(history, 1):
        input_icon = {"text": "📝", "image": "📷", "audio": "🎤"}.get(problem['input_type'], "❓")
        
        with st.expander(f"{input_icon} {i}. {problem['extracted_text'][:50]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Input Type:** {problem['input_type']}")
                st.markdown(f"**Topic:** {problem.get('topic', 'Unknown')}")
            with col2:
                st.markdown(f"**Confidence:** {problem.get('confidence_score', 0):.0%}")
                st.caption(f"Solved: {problem['created_at']}")
            
            st.markdown("**Solution:**")
            st.markdown(problem.get('solution', 'N/A')[:300] + "...")


# ============================================================
# Main App
# ============================================================
def main():
    """Main application."""
    
    # Initialize session state
    if 'show_feedback_form' not in st.session_state:
        st.session_state['show_feedback_form'] = False
    
    # Render sidebar and get settings
    input_mode, show_rag, show_debug, top_k = render_sidebar()
    
    # Main content
    st.title("🧮 Math Mentor")
    st.markdown("*Your AI-powered JEE Math Tutor with Safety Guardrails*")
    
    # Tabs
    tab1, tab2 = st.tabs(["🆕 Solve Problem", "📜 History"])
    
    with tab1:
        canonical = render_input_section(input_mode)
        
        if canonical and canonical.extracted_text:
            st.divider()
            render_solution_section(canonical, show_rag, show_debug, top_k)
    
    with tab2:
        render_history_tab()


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()