import streamlit as st
from transformers import pipeline
from collections import Counter
import re
import json
from datetime import datetime

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🧠",
    layout="wide"
)

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"

@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer():
    return pipeline("summarization", model=SUMMARIZER_MODEL)

@st.cache_resource(show_spinner="Loading question-answering model...")
def load_qa():
    return pipeline("question-answering", model=QA_MODEL)

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))

def split_into_chunks(text, max_words=450):
    """Simple word-based chunking to keep long inputs manageable."""
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

def summarize_document(text, summarizer, max_length, min_length):
    chunks = split_into_chunks(text, max_words=450)
    summaries = []

    for chunk in chunks:
        # Avoid asking for a longer summary than the input can support.
        input_tokens = len(chunk.split())
        local_max = min(max_length, max(20, input_tokens))
        local_min = min(min_length, max(5, local_max - 5))

        result = summarizer(
            chunk,
            max_length=local_max,
            min_length=local_min,
            do_sample=False,
            truncation=True
        )
        summaries.append(result[0]["summary_text"].strip())

    if len(summaries) == 1:
        return summaries[0]

    # Summarize the combined chunk summaries to produce one final result.
    combined = " ".join(summaries)
    if count_words(combined) > 500:
        combined = " ".join(combined.split()[:500])

    final = summarizer(
        combined,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True
    )
    return final[0]["summary_text"].strip()

def answer_question(context, question, qa_pipeline):
    # Sliding word windows with overlap help answers near chunk boundaries.
    words = context.split()
    chunk_size = 350
    overlap = 60
    step = chunk_size - overlap
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
    ] or [context]

    candidates = []
    for index, chunk in enumerate(chunks):
        result = qa_pipeline(question=question, context=chunk)
        candidates.append({
            "answer": result.get("answer", "").strip(),
            "score": float(result.get("score", 0.0)),
            "chunk": index + 1
        })

    return max(candidates, key=lambda item: item["score"])

def make_export(summary, qa_history, original_text):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "original_word_count": count_words(original_text),
        "summary": summary,
        "question_answer_history": qa_history
    }

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "last_document" not in st.session_state:
    st.session_state.last_document = ""

st.title("🧠 AI Study Assistant")
st.caption("Hugging Face Pipelines • Text Summarization • Context-Based Question Answering")

with st.sidebar:
    st.header("⚙️ Model information")
    st.write("**Summarization**")
    st.code(SUMMARIZER_MODEL, language="text")
    st.write("**Question answering**")
    st.code(QA_MODEL, language="text")
    st.info("The first run downloads model files. Internet access is required for the initial download.")
    if st.button("Clear question history", use_container_width=True):
        st.session_state.qa_history = []
        st.rerun()

tab_summary, tab_qa, tab_export = st.tabs([
    "📝 Summarize", "❓ Ask Questions", "📥 Export Results"
])

with tab_summary:
    st.subheader("1. Summarize a document")
    document = st.text_area(
        "Paste your study notes or article",
        height=260,
        placeholder="Paste a passage about AI, databases, cybersecurity, or another subject..."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        summary_length = st.selectbox(
            "Maximum summary length (tokens)",
            options=[60, 80, 100, 130, 160],
            index=2
        )
    with col2:
        minimum_length = st.selectbox(
            "Minimum summary length (tokens)",
            options=[10, 15, 20, 25, 30],
            index=2
        )
    with col3:
        st.metric("Input words", count_words(document))

    if st.button("✨ Generate Summary", type="primary", use_container_width=True):
        cleaned = clean_text(document)
        if len(cleaned) < 100:
            st.error("Please enter at least 100 characters for meaningful summarization.")
        elif minimum_length >= summary_length:
            st.error("Minimum summary length must be smaller than maximum summary length.")
        else:
            try:
                summarizer = load_summarizer()
                with st.spinner("Reading and summarizing your document..."):
                    result = summarize_document(
                        cleaned, summarizer, summary_length, minimum_length
                    )
                st.session_state.summary = result
                st.session_state.last_document = cleaned
                st.session_state.qa_history = []
                st.success("Summary generated.")
            except Exception as exc:
                st.error("The summarization model could not process this text.")
                st.caption(f"Technical details: {exc}")

    if st.session_state.summary:
        st.markdown("### Generated summary")
        st.write(st.session_state.summary)
        original = st.session_state.last_document
        original_count = count_words(original)
        summary_count = count_words(st.session_state.summary)
        reduction = (
            (original_count - summary_count) / original_count * 100
            if original_count else 0
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Original words", original_count)
        m2.metric("Summary words", summary_count)
        m3.metric("Word reduction", f"{reduction:.1f}%")
        with st.expander("View original document"):
            st.write(original)

with tab_qa:
    st.subheader("2. Ask questions about a passage")
    st.write("Use the document below as the evidence source. The model extracts an answer from that context.")
    qa_context = st.text_area(
        "Context passage",
        value=st.session_state.last_document,
        height=230,
        placeholder="Paste the document you want to ask questions about..."
    )
    question = st.text_input(
        "Your question",
        placeholder="Example: What are the applications of artificial intelligence?"
    )

    if st.button("🔎 Find Answer", type="primary", use_container_width=True):
        context_clean = clean_text(qa_context)
        question_clean = clean_text(question)
        if len(context_clean) < 50:
            st.error("Please provide a context of at least 50 characters.")
        elif len(question_clean) < 5:
            st.error("Please enter a complete question.")
        else:
            try:
                qa_pipeline = load_qa()
                with st.spinner("Searching the context for an answer..."):
                    answer = answer_question(context_clean, question_clean, qa_pipeline)
                record = {
                    "question": question_clean,
                    "answer": answer["answer"],
                    "confidence": round(answer["score"], 4),
                    "chunk": answer["chunk"]
                }
                st.session_state.qa_history.insert(0, record)
                st.markdown("### Answer")
                st.success(answer["answer"] or "No clear answer found.")
                st.metric("Model confidence score", f'{answer["score"]:.1%}')
                if answer["score"] < 0.20:
                    st.warning("Low confidence: check the context or rephrase the question. The answer may be unreliable.")
                st.caption(f"Best matching passage chunk: {answer['chunk']}")
            except Exception as exc:
                st.error("The question-answering model could not process this request.")
                st.caption(f"Technical details: {exc}")

    if st.session_state.qa_history:
        st.markdown("### Question history")
        for i, item in enumerate(st.session_state.qa_history, start=1):
            with st.expander(f"{i}. {item['question']}"):
                st.write("**Answer:**", item["answer"])
                st.write("**Confidence:**", f"{item['confidence']:.1%}")
                st.write("**Chunk:**", item["chunk"])

with tab_export:
    st.subheader("3. Export your work")
    summary = st.session_state.summary
    history = st.session_state.qa_history
    source_text = st.session_state.last_document

    if not summary and not history:
        st.info("Generate a summary or answer at least one question before exporting.")
    else:
        payload = make_export(summary, history, source_text)
        st.download_button(
            "Download results as JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name="ai_study_assistant_results.json",
            mime="application/json",
            use_container_width=True
        )
        report_lines = [
            "AI STUDY ASSISTANT REPORT",
            f"Created: {payload['created_at']}",
            f"Original word count: {payload['original_word_count']}",
            "",
            "SUMMARY",
            summary or "(No summary generated)",
            "",
            "QUESTION AND ANSWER HISTORY"
        ]
        if history:
            for i, item in enumerate(history, start=1):
                report_lines.extend([
                    f"{i}. Question: {item['question']}",
                    f"   Answer: {item['answer']}",
                    f"   Confidence: {item['confidence']:.1%}",
                    ""
                ])
        else:
            report_lines.append("(No questions answered)")
        st.download_button(
            "Download readable report as TXT",
            data="\n".join(report_lines),
            file_name="ai_study_assistant_report.txt",
            mime="text/plain",
            use_container_width=True
        )

st.divider()
st.caption("Educational prototype. Model answers and confidence scores can be wrong; verify important information against the original text.")
