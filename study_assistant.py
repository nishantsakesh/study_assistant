# study_assistant
import streamlit as st
import pdfplumber, io, re, json, numpy as np, pandas as pd, nltk, requests, docx
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ctransformers import AutoModelForCausalLM
import graphviz

# --- Page Config ---
st.set_page_config(page_title="Ultimate AI Study Assistant", layout="wide")

# --- NLTK Punkt Tokenizer ---
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)

# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading Open Source LLM (Phi-3)... This may take a while.")
def load_llm():
    return AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct-gguf", model_file="Phi-3-mini-4k-instruct-q4.gguf",
        model_type="phi3", gpu_layers=0, context_length=4000
    )

# --- Text Extraction & Core Logic Functions ---
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
    return text

def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"DOCX parsing error: {e}")
        return ""

def extract_text_from_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs])
    except requests.exceptions.RequestException as e:
        st.error(f"URL fetch error: {e}")
        return ""

def get_youtube_transcript(video_url):
    try:
        video_id = None
        if "v=" in video_url: video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url: video_id = video_url.split("youtu.be/")[1].split("?")[0]
        if not video_id:
            st.error("Invalid YouTube URL.")
            return ""
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d['text'] for d in transcript_list])
    except Exception as e:
        st.error(f"YouTube transcript error: {e}")
        return ""

def split_text_into_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

@st.cache_data
def embed_texts(_model, texts):
    return _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

# --- Feature Generation Functions ---
def generate_abstractive_summary(llm, text):
    prompt = f"<|user|>\nSummarize the text concisely in bullet points.\n\nText:\n---\n{text[:3500]}\n---\nSummary:<|end|>\n<|assistant|>"
    return llm(prompt)

def answer_question(llm, question, sentences, sentence_embeddings, embedding_model):
    question_embedding = embed_texts(embedding_model, [question])[0].reshape(1, -1)
    similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]
    context = "\n".join([sentences[i] for i in top_indices])
    prompt = f"""<|user|>
    Based ONLY on the context below, answer the question. If the answer isn't in the context, say so.
    Context: --- {context} --- Question: {question}<|end|>
    <|assistant|>"""
    return llm(prompt)

def generate_flashcards(sentences, num_cards=5):
    if not sentences or len(sentences) < num_cards: return []
    long_sentences = [s for s in sentences if len(s) > 100]
    if not long_sentences: return []
    selected_sentences = np.random.choice(long_sentences, size=min(num_cards, len(long_sentences)), replace=False)
    flashcards = []
    for sent in selected_sentences:
        words = [w.strip(".,!?") for w in sent.split() if len(w) > 5]
        if not words: continue
        keyword = np.random.choice(words)
        blanked_sent = re.sub(re.escape(keyword), "_______", sent, flags=re.IGNORECASE, count=1)
        flashcards.append({"blank": blanked_sent, "answer": keyword})
    return flashcards

def generate_smart_mcqs(llm, text, num_mcqs=3):
    prompt = f"""<|user|>
    Generate {num_mcqs} multiple-choice questions from the text. Provide three plausible but incorrect distractors.
    Format as a valid JSON list of objects with keys: "question", "options", "answer".
    Text: --- {text[:3500]} --- JSON Output:<|end|>
    <|assistant|>"""
    response = llm(prompt)
    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match: return json.loads(json_match.group(0))
        return []
    except (json.JSONDecodeError, TypeError): return []

def generate_anki_deck(flashcards, mcqs):
    anki_data = []
    for fc in flashcards: anki_data.append({"Front": fc['blank'], "Back": fc['answer']})
    for mcq in mcqs:
        front = f"{mcq['question']}\n\nOptions:\n" + "\n".join([f"- {opt}" for opt in mcq['options']])
        back = mcq['answer']
        anki_data.append({"Front": front, "Back": back})
    if not anki_data: return ""
    df = pd.DataFrame(anki_data)
    return df.to_csv(index=False, header=False).encode('utf-8')

def generate_knowledge_graph(llm, text):
    prompt = f"""<|user|>
    Extract the main concepts and their relationships from the text as a list of triplets.
    Format the output as a valid JSON list of lists. Example: [["AI", "is a branch of", "Computer Science"]]
    Text: --- {text[:3000]} --- JSON Output:<|end|>
    <|assistant|>"""
    response = llm(prompt)
    try:
        json_match = re.search(r'\[\s*\[.*\]\s*\]', response, re.DOTALL)
        if json_match:
            triplets = json.loads(json_match.group(0))
            dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'splines': 'true'})
            dot.attr('node', shape='box', style='rounded', color='skyblue')
            dot.attr('edge', color='gray')
            for subj, rel, obj in triplets:
                dot.node(subj, subj); dot.node(obj, obj)
                dot.edge(subj, obj, label=rel)
            return dot
        return None
    except (json.JSONDecodeError, IndexError): return None

# --- Streamlit UI ---
st.title("🧠 Ultimate AI Study Assistant")
st.caption("From PDF, URL, or YouTube to Summaries, Quizzes, Mind Maps & Anki cards!")

# Load models once
embedding_model = load_embedding_model()
llm = load_llm()

# Initialize session state for text and quiz answers
if 'raw_text' not in st.session_state: st.session_state.raw_text = ""
if 'user_answers' not in st.session_state: st.session_state.user_answers = {}

# --- Input UI in an Expander ---
with st.expander("📚 Step 1: Provide Your Content", expanded=True):
    input_method = st.radio("Choose input method:", ["File Upload", "Web URL", "YouTube URL"], horizontal=True, label_visibility="collapsed")
    
    raw_text_input = ""
    if input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"], label_visibility="collapsed")
        if uploaded_file:
            if uploaded_file.type == "application/pdf": raw_text_input = extract_text_from_pdf(uploaded_file.read())
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": raw_text_input = extract_text_from_docx(uploaded_file.read())
            else: raw_text_input = uploaded_file.read().decode('utf-8')
    elif input_method == "Web URL":
        url = st.text_input("Enter a URL to an article:")
        if url: raw_text_input = extract_text_from_url(url)
    elif input_method == "YouTube URL":
        yt_url = st.text_input("Enter a YouTube video URL:")
        if yt_url: raw_text_input = get_youtube_transcript(yt_url)

    if st.button("Analyze Content"):
        if raw_text_input:
            st.session_state.raw_text = raw_text_input
            st.session_state.user_answers = {}
            st.success("Content loaded! Your study kit is ready below.")
        else:
            st.warning("Please provide some content to analyze.")

# --- Main App Logic (displays only after analysis) ---
if st.session_state.raw_text:
    sentences = split_text_into_sentences(st.session_state.raw_text)
    
    if not sentences:
        st.warning("Could not extract enough text from the content. Please try a different source.")
    else:
        st.header("🚀 Your Study Kit")
        sentence_embeddings = embed_texts(embedding_model, sentences)
        
        tab_titles = ["📖 Summaries", "🗂️ Flashcards & Quiz", "🧠 Mind Map", "💬 Chat with Document"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1:
            with st.spinner("Generating AI summary..."):
                st.subheader("🤖 AI-Generated Abstractive Summary")
                st.info(generate_abstractive_summary(llm, st.session_state.raw_text))
        
        with tab2:
            st.subheader("Fill-in-the-Blank Flashcards & MCQs")
            flashcards = generate_flashcards(sentences)
            with st.spinner("Generating smart MCQs..."):
                mcqs = generate_smart_mcqs(llm, st.session_state.raw_text, num_mcqs=3)

            if not flashcards and not mcqs:
                st.warning("Could not generate flashcards or MCQs for this text.")
            else:
                anki_csv = generate_anki_deck(flashcards, mcqs)
                st.download_button("Export to Anki (CSV)", anki_csv, "anki_deck.csv", "text/csv")
                
                for i, mcq in enumerate(mcqs):
                    st.markdown(f"**Question {i+1}:** {mcq['question']}")
                    answer = st.radio("Options:", mcq['options'], key=f"mcq_{i}", index=None)
                    st.session_state.user_answers[i] = {"selected": answer, "correct": mcq['answer']}

                if st.button("Check Answers"):
                    score = 0; total = len(mcqs)
                    for i, result in st.session_state.user_answers.items():
                        if result['selected'] == result['correct']: score += 1
                    st.metric("Your Score:", f"{score}/{total}")

        with tab3:
            st.subheader("Concept Mind Map")
            with st.spinner("Generating knowledge graph..."):
                graph = generate_knowledge_graph(llm, st.session_state.raw_text)
            if graph:
                st.graphviz_chart(graph)
            else:
                st.warning("Could not generate a mind map for this text.")

        with tab4:
            st.subheader("Ask a Question")
            user_question = st.text_input("Ask anything about your document:", key="qna_input")
            if user_question:
                with st.spinner("Searching for the answer..."):
                    answer = answer_question(llm, user_question, sentences, sentence_embeddings, embedding_model)
                    st.info(answer)