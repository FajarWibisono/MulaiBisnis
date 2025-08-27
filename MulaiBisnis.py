# TD_WIRA.py
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI API & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────

# Groq_API KEY
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="TEDIWIRA",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
        .analysis-box { background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 1rem 0; }
        .startup-info { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
        .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("📓TEMAN DISKUSI KEWIRAUSAHAAN")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Tentang Kewirausahaan
    **Chat Bot ini akan membantu Anda memahami lebih dalam tentang dunia KEWIRAUSAHAAN** dan berbagai hal-hal yang perlu di perhatikan baik pada masa persiapan, pelaksanaan, pengembangan,dan bahkan exit strategy. **Pastikanlah Anda memiliki koneksi internet yang baik dan stabil.**
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'proposal_analysis' not in st.session_state:
    st.session_state.proposal_analysis = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_INDONESIA = """\
Anda adalah seorang Ahli ENTREPRENEURSHIP yang KREATIF dan berpengalaman lebih dari 25 tahun. 
Jawab pertanyaan pengguna dalam bahasa Indonesia yang baik dan terstruktur.
Selalu berikan jawaban terbaik yang dapat kamu berikan dengan tone memotivasi dengan gaya santai dan informal tapi tetap santun.

Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=PROMPT_INDONESIA
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI UNTUK REVIEW PROPOSAL BISNIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_business_proposal(pdf_file):
    """
    Menganalisis proposal bisnis dari file PDF dan memberikan ringkasan, keuntungan,
    risiko, serta pertanyaan lanjutan untuk investor.
    """
    try:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Gabungkan semua teks
        full_text = "\n".join([doc.page_content for doc in documents])

        # Prompt untuk analisis proposal
        analysis_prompt = f"""
        Anda adalah seorang ahli bisnis dan investor profesional. Analisis proposal bisnis berikut dan berikan:

        1. **Ringkasan Proposal (9-18 kalimat)**: Berikan gambaran komprehensif tentang ide bisnis, model bisnis, target pasar, dan rencana implementasi.
        2. **Potensi Keuntungan**: Sebutkan hal-hal yang membuat proposal ini menarik atau potensial menguntungkan.
        3. **Risiko atau Hal yang Perlu Diwaspadai**: Sebutkan risiko utama atau kelemahan dalam proposal ini.
        4. **Pertanyaan Investor (3-7 pertanyaan)**: Buat daftar pertanyaan penting yang harus diajukan oleh investor kepada pengusul bisnis untuk memastikan dan meimilkan risiko yang mungkin tersembunyi.

        Proposal Bisnis:
        {full_text}

        Jawaban:
        """

        # Inisialisasi LLM
        llm = ChatGroq(
            temperature=0.3,
            model_name="openai/gpt-oss-120b",
            max_tokens=4096
        )

        # Dapatkan jawaban
        response = llm.invoke(analysis_prompt)
        
        # Hapus file sementara
        os.unlink(tmp_file_path)
        
        return response.content

    except Exception as e:
        # Hapus file sementara jika ada error
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        return f"Terjadi kesalahan saat menganalisis proposal: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. INISIALISASI LLM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    """Menginisialisasi dan meng-cache LLM"""
    return ChatGroq(
        temperature=0.72,
        model_name="openai/gpt-oss-120b",
        max_tokens=4096
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR UNTUK UPLOAD DAN ANALISIS PROPOSAL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Review Proposal Bisnis")
    st.markdown("Upload file PDF proposal bisnis untuk dianalisis.")
    uploaded_file = st.file_uploader("📁 Pilih file PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Menganalisis proposal..."):
            summary = analyze_business_proposal(uploaded_file)
            st.session_state.proposal_analysis = summary

# ─────────────────────────────────────────────────────────────────────────────
# 7. TAMPILKAN HASIL ANALISIS PROPOSAL (JIKA ADA)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.proposal_analysis:
    st.subheader("🔍 Hasil Analisis Proposal Bisnis")
    st.markdown(f'<div class="analysis-box">{st.session_state.proposal_analysis}</div>', unsafe_allow_html=True)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("💬 Diskusi Kewirausahaan")

# 8.1 Tampilkan riwayat chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 8.2 Chat Input
prompt = st.chat_input("✍️tuliskan pertanyaan Anda tentang KEWIRAUSAHAAN disini")
if prompt:
    # Tambahkan pertanyaan user ke riwayat chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 8.3 Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            try:
                # Format riwayat chat
                chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[:-1]])
                
                # Dapatkan LLM
                llm = get_llm()
                
                # Buat prompt dengan riwayat
                formatted_prompt = INDO_PROMPT_TEMPLATE.format(
                    chat_history=chat_history_str,
                    question=prompt
                )
                
                # Dapatkan jawaban
                response = llm.invoke(formatted_prompt)
                answer = response.content
                
                st.write(answer)
                
                # Tambahkan ke riwayat
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 9. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan **AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.**
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)




