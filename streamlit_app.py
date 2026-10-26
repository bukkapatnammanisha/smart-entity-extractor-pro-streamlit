import re
import html
import json
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
from pypdf import PdfReader
from pdf2image import convert_from_path

st.set_page_config(page_title="Smart Entity Extractor Pro", page_icon="🧠", layout="wide")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

try:
    nlp = load_nlp()
    MODEL_ERROR = None
except Exception as exc:
    nlp = None
    MODEL_ERROR = str(exc)

COLORS = {
    "PERSON": "#FFD6A5", "ORG": "#CAFFBF", "GPE": "#9BF6FF", "LOC": "#9BF6FF",
    "DATE": "#BDB2FF", "TIME": "#BDB2FF", "MONEY": "#FDFFB6", "PERCENT": "#FFC6FF",
    "QUANTITY": "#FDFFB6", "CARDINAL": "#E2E2E2", "ORDINAL": "#E2E2E2",
    "PRODUCT": "#A0C4FF", "EVENT": "#FFADAD", "DURATION": "#D8F3DC",
    "NORP": "#FDE2E4", "FAC": "#CFBAF0"
}
PATTERNS = {
    "DATE": [r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", r"\b\d{4}-\d{1,2}-\d{1,2}\b",
             r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
             r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"],
    "MONEY": [r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:crore|crores|lakh|lakhs|million|billion))?",
              r"\b\d+(?:\.\d+)?\s?(?:rupees|INR|USD|dollars|euros|pounds)\b", r"\b\d+(?:\.\d+)?\s?(?:crores?|lakhs?)\b"],
    "PERCENT": [r"\b\d+(?:\.\d+)?\s?%"],
    "DURATION": [r"\b\d+(?:\.\d+)?\s?(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"]
}
SAMPLE_TEXT = ("Google CEO Sundar Pichai announced in Hyderabad, India, on 20-10-2026 that employee Manisha "
               "earned ₹50,000 per month, received a 10% bonus, worked for 3 years, and helped launch "
               "the Pixel 10 product at the Global Technology Summit.")
SAMPLE_EVALUATION = json.dumps([
    {"text": "Google hired Ravi in Chennai.", "entities": [{"text": "Google", "label": "ORG"}, {"text": "Ravi", "label": "PERSON"}, {"text": "Chennai", "label": "GPE"}]},
    {"text": "Microsoft paid ₹5000 on 20-10-2026.", "entities": [{"text": "Microsoft", "label": "ORG"}, {"text": "₹5000", "label": "MONEY"}, {"text": "20-10-2026", "label": "DATE"}]}
], indent=2)

def find_rule_entities(text):
    results = []
    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                results.append({"text": match.group(), "label": label, "start": match.start(), "end": match.end()})
    return results

def extract_entities(text):
    if not text or not text.strip():
        return []
    if nlp is None:
        raise RuntimeError("The spaCy English model could not load. Please check the app deployment logs.")
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
    for rule in find_rule_entities(text):
        entities = [ent for ent in entities if not (rule["start"] < ent["end"] and rule["end"] > ent["start"])]
        entities.append(rule)
    unique = {(ent["start"], ent["end"], ent["label"]): ent for ent in entities}
    return sorted(unique.values(), key=lambda ent: (ent["start"], ent["end"]))

def get_input_text(text, uploaded_file):
    if uploaded_file is None:
        return text or ""
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            path = tmp.name
        reader = PdfReader(path)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        for i, page_text in enumerate(page_texts):
            if not page_text.strip():
                try:
                    images = convert_from_path(path, dpi=200, first_page=i + 1, last_page=i + 1)
                    if images:
                        page_texts[i] = pytesseract.image_to_string(images[0])
                except Exception as exc:
                    st.warning(f"OCR could not process PDF page {i+1}: {exc}. If this is a scanned PDF, OCR system packages may be missing.")
        extracted = "\n".join(page_texts)
        if not extracted.strip():
            raise ValueError("No readable text found. The PDF may be blank, damaged, or unclear.")
        return extracted
    raise ValueError("Please upload a PDF or TXT file.")

def highlight_text(text, entities):
    if not text:
        return "No text to display."
    output, cursor = [], 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        if start < cursor:
            continue
        output.append(html.escape(text[cursor:start]))
        label = ent["label"]
        color = COLORS.get(label, "#E2E2E2")
        output.append(f'<span style="background:{color};padding:4px;border-radius:5px;margin:2px;display:inline-block;">{html.escape(text[start:end])} <b style="font-size:11px;">[{html.escape(label)}]</b></span>')
        cursor = end
    output.append(html.escape(text[cursor:]))
    return '<div style="line-height:2.3;overflow-wrap:anywhere;">' + "".join(output) + "</div>"

def entities_to_dataframe(entities):
    return pd.DataFrame([{"Entity": e["text"], "Category": e["label"], "Start": e["start"], "End": e["end"]} for e in entities], columns=["Entity", "Category", "Start", "End"])

def create_chart(entities):
    counts = Counter(e["label"] for e in entities)
    fig, ax = plt.subplots(figsize=(8, 4))
    if counts:
        ax.bar(list(counts.keys()), list(counts.values()))
        ax.set_title("Entities by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
    else:
        ax.text(.5, .5, "No entities detected", ha="center", va="center")
        ax.axis("off")
    return fig

def compare_documents(text1, text2):
    e1, e2 = extract_entities(text1), extract_entities(text2)
    set1 = {(e["text"].lower(), e["label"]) for e in e1}
    set2 = {(e["text"].lower(), e["label"]) for e in e2}
    common, only1, only2 = set1 & set2, set1 - set2, set2 - set1
    summary = f"**Document 1 entities:** {len(set1)}  \n\n**Document 2 entities:** {len(set2)}  \n\n**Common entities:** {len(common)}  \n\n**Only in Document 1:** {len(only1)}  \n\n**Only in Document 2:** {len(only2)}"
    rows = ([{"Entity": x, "Category": y, "Document": "Both"} for x, y in sorted(common)] +
            [{"Entity": x, "Category": y, "Document": "Document 1 only"} for x, y in sorted(only1)] +
            [{"Entity": x, "Category": y, "Document": "Document 2 only"} for x, y in sorted(only2)])
    return summary, pd.DataFrame(rows, columns=["Entity", "Category", "Document"])

def evaluate_model(json_data):
    try:
        examples = json.loads(json_data)
        if not isinstance(examples, list) or not examples:
            raise ValueError("Provide a non-empty JSON list.")
        tp = fp = fn = 0
        for example in examples:
            expected = {(e["text"].lower(), e["label"].upper()) for e in example["entities"]}
            predicted = {(e["text"].lower(), e["label"].upper()) for e in extract_entities(example["text"])}
            tp += len(expected & predicted); fp += len(predicted - expected); fn += len(expected - predicted)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return f"**True positives:** {tp}  \n\n**False positives:** {fp}  \n\n**False negatives:** {fn}  \n\n**Precision:** {precision:.2%}  \n\n**Recall:** {recall:.2%}  \n\n**F1-score:** {f1:.2%}  \n\n*Scores compare exact entity text and category against your labels.*"
    except Exception as exc:
        return f"Invalid evaluation data: {exc}"

st.title("🧠 Smart Entity Extractor Pro")
st.subheader("NLP-powered Named Entity Recognition Application")
st.write("Upload PDF/TXT files or enter text to identify important information. Scanned PDFs are processed using OCR when supported.")
if MODEL_ERROR:
    st.error("The spaCy English model did not load. The app is showing this message so the issue is visible in deployment.")
    st.code(MODEL_ERROR)

extract_tab, compare_tab, evaluate_tab = st.tabs(["Entity Extractor", "Compare Documents", "Model Evaluation"])
with extract_tab:
    left, right = st.columns(2)
    with left:
        text_input = st.text_area("Enter text", height=180, placeholder="Paste your text here...")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], key="main_file")
        if st.button("Try Sample Text"):
            st.session_state["sample_text"] = SAMPLE_TEXT
        if "sample_text" in st.session_state and not text_input:
            st.info("Sample text is ready below. Copy it into the text box, or click Analyze after pasting it.")
            st.code(SAMPLE_TEXT)
        category = st.selectbox("Filter by category", ["All"] + sorted(COLORS.keys()))
        search_term = st.text_input("Search entity", placeholder="Example: Google")
        analyze_clicked = st.button("Analyze Text / File", type="primary")
    if analyze_clicked:
        try:
            text = get_input_text(text_input, uploaded_file)
            if not text.strip():
                st.warning("Please enter text or upload a PDF/TXT file.")
            else:
                entities = extract_entities(text)
                filtered = entities
                if category != "All":
                    filtered = [e for e in filtered if e["label"] == category]
                if search_term.strip():
                    filtered = [e for e in filtered if search_term.strip().lower() in e["text"].lower()]
                word_count = len(re.findall(r"\b\w+\b", text))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Words", word_count); m2.metric("Total entities", len(entities)); m3.metric("Categories", len(set(e["label"] for e in entities))); m4.metric("Displayed after filters", len(filtered))
                st.markdown("### Highlighted Text")
                st.markdown(highlight_text(text, filtered), unsafe_allow_html=True)
                st.markdown("### Entity Chart")
                st.pyplot(create_chart(entities))
                df = entities_to_dataframe(filtered)
                st.markdown("### Extracted Entities")
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV Report", data=df.to_csv(index=False).encode("utf-8"), file_name="extracted_entities.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not analyze this input: {exc}")
with compare_tab:
    st.write("Paste text from two documents to compare their entities.")
    c1, c2 = st.columns(2)
    with c1: text1 = st.text_area("Document 1", height=180, key="doc1")
    with c2: text2 = st.text_area("Document 2", height=180, key="doc2")
    if st.button("Compare Documents", type="primary"):
        if not text1.strip() or not text2.strip():
            st.warning("Please enter text in both document boxes.")
        else:
            try:
                summary, comparison = compare_documents(text1, text2)
                st.markdown("### Document Comparison")
                st.markdown(summary)
                st.dataframe(comparison, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not compare documents: {exc}")
with evaluate_tab:
    st.write("Enter labelled JSON examples to evaluate entity extraction. Edit the sample data to test your own examples.")
    evaluation_input = st.text_area("Ground-truth JSON", value=SAMPLE_EVALUATION, height=300)
    if st.button("Evaluate Model", type="primary"):
        try:
            st.markdown("### Evaluation Results")
            st.markdown(evaluate_model(evaluation_input))
        except Exception as exc:
            st.error(str(exc))
st.divider()
st.caption("Technologies: Python, spaCy, Streamlit, Pandas, Matplotlib, pypdf, OCR. OCR accuracy depends on PDF image quality. The NLP model can make mistakes, so verify important extracted information.")
