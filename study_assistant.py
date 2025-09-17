# study_assistant_deployment_ready.py
import streamlit as st
import pdfplumber, io, re, numpy as np, nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ctransformers import AutoModelForCausalLM

# --- Page Config & NLTK Setup ---
st.set_page_config(page_title="AI PDF Assistant", layout="wide")
try:
    nltk.data.path.append('./nltk_data')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("Downloading one-time NLTK data for sentence splitting...")
    nltk.download('punkt', quiet=True)


# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner="Loading AI models... This may take a moment.")
def load_models():
    """Loads both the embedding model and the LLM."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # --- MODEL FIXED FOR DEPLOYMENT ---
    # Switched back to TinyLlama as it fits in Streamlit Cloud's memory.
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama", # Model type is now 'llama'
        gpu_layers=0,
        context_length=2048
    )

    return embedding_model, llm

# --- Core Logic Functions ---
def extract_text_from_file(file_bytes, file_type):
    """Extracts text from a file, skipping pages with errors."""
    text = ""
    if file_type == "application/pdf":
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text(x_tolerance=2)
                        if page_text: text += page_text + "\n"
                    except: continue
        except Exception as e:
            st.error(f"PDF parsing error: {e}")
    elif file_type == "text/plain":
        text = file_bytes.decode('utf-8', errors='ignore')
    return text

def clean_and_split_text(text):
    """Cleans up the extracted text and splits it into sentences."""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

@st.cache_data
def embed_texts(_model, texts):
    """Generates embeddings for a list of texts."""
    return _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

# --- Feature Generation Functions (Prompts fixed for TinyLlama) ---
def generate_summary(llm, text):
    """Generates a concise, AI-powered summary."""
    # --- PROMPT UPDATED FOR TINYLLAMA ---
    prompt = f"<|im_start|>user\nSummarize the following text in a few concise bullet points.\n\nText:\n---\n{text[:1800]}\n---\nSummary:<|im_end|>\n<|im_start|>assistant\n"
    try:
        return llm(prompt, max_new_tokens=512, temperature=0.5)
    except Exception as e:
        return f"Could not generate summary: {e}"

def answer_question(llm, question, sentences, sentence_embeddings, embedding_model):
    """Answers a user's question about the text using RAG."""
    question_embedding = embed_texts(embedding_model, [question])[0].reshape(1, -1)
    similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]
    context = "\n".join([sentences[i] for i in top_indices])
    
    # --- PROMPT UPDATED FOR TINYLLAMA ---
    prompt = f"<|im_start|>user\nBased ONLY on the context provided below, answer the user's question. If the answer is not in the context, state that clearly.\n\nContext:\n--- {context} ---\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
    try:
        return llm(prompt, max_new_tokens=512, temperature=0.5)
    except Exception as e:
        return f"Could not generate an answer: {e}"

# --- Streamlit UI ---
st.title("📄 AI Document Assistant")
st.caption("Upload a PDF or TXT file to get an instant AI-powered summary and ask questions about the content.")

embedding_model, llm = load_models()

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"], label_visibility="collapsed")

if uploaded_file:
    with st.spinner("Processing document..."):
        raw_text = extract_text_from_file(uploaded_file.read(), uploaded_file.type)
        sentences = clean_and_split_text(raw_text)

    if not sentences:
        st.warning("Could not extract enough readable text from this document. Please try another file.")
    else:
        sentence_embeddings = embed_texts(embedding_model, sentences)
        
        st.header("🤖 AI-Generated Summary", divider='rainbow')
        with st.spinner("Generating summary..."):
            summary = generate_summary(llm, raw_text)
            st.markdown(summary)

        st.header("💬 Chat with Your Document", divider='rainbow')
        user_question = st.text_input("Ask a question about the content:")

        if user_question:
            with st.spinner("Searching for the answer..."):
                answer = answer_question(llm, user_question, sentences, sentence_embeddings, embedding_model)
                st.info(answer)

else:
    st.info("Please upload a PDF or TXT file to get started.")
