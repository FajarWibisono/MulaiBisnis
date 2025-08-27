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
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA - MENTOR AHLI
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_MENTOR_AHLI = """\
Anda adalah seorang MENTOR KEWIRAUSAHAAN yang sangat ahli dengan pengalaman lebih dari 25 tahun di bidang:
- Perencanaan bisnis komprehensif
- Validasi ide dan model bisnis
- Strategi go-to-market yang efektif
- Pengembangan skala bisnis
- Manajemen risiko dan pertumbuhan
- Exit strategy yang menguntungkan

GAYA BERKOMUNIKASI:
- Bersikap seperti mentor pribadi yang tajam tapi supportive
- Gunakan bahasa yang mudah dipahami tapi tetap profesional
- Berikan insight yang actionable dan spesifik
- Tanyakan pertanyaan probing untuk membantu user berpikir lebih dalam
- Jangan hanya menjawab, tapi juga arahkan dan tantang user untuk berpikir kritis

INSTRUKSI:
1. Analisis pertanyaan dengan tajam - identifikasi inti masalah/kesempatan
2. Berikan jawaban yang komprehensif dengan struktur yang jelas
3. Sertakan contoh konkret atau analogi yang relevan
4. Tawarkan perspektif yang mungkin belum dipertimbangkan user
5. Akhiri dengan actionable insights atau pertanyaan untuk refleksi

Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban (sebagai mentor ahli):
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=PROMPT_MENTOR_AHLI
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROMPT UNTUK ANALISIS PROPOSAL BISNIS
# ─────────────────────────────────────────────────────────────────────────────
PROPOSAL_ANALYSIS_PROMPT = """\
Anda adalah seorang MENTOR KEWIRAUSAHAAN dan INVESTOR profesional dengan pengalaman lebih dari 25 tahun. 
Tugas Anda adalah menganalisis proposal bisnis ini dengan lensa seorang mentor yang tajam tapi constructive.

ANALISIS YANG DIBUTUHKAN:

1. **🔍 VALIDASI IDE BISNIS** (2-3 paragraf)
   - Apakah ide ini menyelesaikan masalah nyata?
   - Bagaimana potensi market demand?
   - Apa unique value proposition yang kuat?

2. **📊 MODEL BISNIS & FINANSIAL** (2-3 paragraf)
   - Apakah model bisnisnya sustainable?
   - Bagaimana struktur pendapatan dan cost structure?
   - Apakah proyeksi finansialnya realistis?

3. **🎯 GO-TO-MARKET STRATEGY** (2 paragraf)
   - Apakah strategi pemasaran dan sales sudah tepat sasaran?
   - Bagaimana positioning terhadap kompetitor?

4. **⚡ KEKUATAN & KELEMAHAN UTAMA** (Bullet points)
   • Strengths: Hal-hal yang membuat proposal ini menonjol
   • Weaknesses: Area yang perlu perhatian serius
   • Opportunities: Kesempatan yang bisa dimanfaatkan
   • Threats: Risiko yang perlu di-anticipate

5. **💡 REKOMENDASI MENTOR** (3-4 poin actionable)
   - Rekomendasi spesifik untuk improvement
   - Pertanyaan kritis yang harus dijawab founder
   - Next steps yang harus diambil

6. **🎯 PERTANYAAN MENTOR** (3-5 pertanyaan probing)
   - Pertanyaan yang menggugah pemikiran mendalam
   - Pertanyaan tentang asumsi yang mendasari proposal

Proposal Bisnis:
{proposal_text}

Jawaban (sebagai mentor ahli):
"""

# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI UNTUK REVIEW PROPOSAL BISNIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_business_proposal(pdf_file):
    """
    Menganalisis proposal bisnis dari file PDF dengan pendekatan mentor ahli
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

        # Inisialisasi LLM
        llm = ChatGroq(
            temperature=0.72,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=4096 # Meningkatkan token untuk analisis komprehensif
        )

        # Buat prompt analisis
        analysis_prompt = PROPOSAL_ANALYSIS_PROMPT.format(proposal_text=full_text)
        
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
# 6. INISIALISASI LLM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm(temperature=0.72):
    """Menginisialisasi dan meng-cache LLM"""
    return ChatGroq(
        temperature=temperature,
        model_name="openai/gpt-oss-120b",
        max_tokens=3006
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR UNTUK UPLOAD DAN ANALISIS PROPOSAL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Review Proposal Bisnis")
    st.markdown("Upload file PDF proposal bisnis untuk dianalisis.")
    uploaded_file = st.file_uploader("📁 Pilih file PDF", type="pdf")

    if uploaded_file:
        with st.spinner("🧠 Mentor sedang menganalisis proposal Anda dengan seksama..."):
            summary = analyze_business_proposal(uploaded_file)
            st.session_state.proposal_analysis = summary

# ─────────────────────────────────────────────────────────────────────────────
# 8. TAMPILKAN HASIL ANALISIS PROPOSAL (JIKA ADA)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.proposal_analysis:
    st.subheader("🔍 Hasil Analisis Proposal Bisnis oleh TEDI")
    st.markdown(f'<div class="analysis-box">{st.session_state.proposal_analysis}</div>', unsafe_allow_html=True)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 9. ANTARMUKA CHAT - MENTOR INTERAKTIF
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("💬 Diskusi Interaktif dengan **TEMAN DISKUSI Kewirausahaan**")
st.markdown("*Ajukan pertanyaan spesifik tentang ide bisnis, strategi, atau tantangan yang Anda hadapi*")

# 9.1 Tampilkan riwayat chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 9.2 Chat Input
prompt = st.chat_input("✍️Tulis pertanyaan Anda tentang kewirausahaan...")
if prompt:
    # Tambahkan pertanyaan user ke riwayat chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 9.3 Generate Response
    with st.chat_message("assistant"):
        with st.spinner("🧠 TEDIWIRA sedang memikirkan jawaban terbaik untuk Anda..."):
            try:
                # Format riwayat chat
                chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[:-1]])
                
                # Dapatkan LLM dengan temperature yang tepat untuk mentoring
                llm = get_llm(temperature=0.5)  # Slightly more creative for mentoring
                
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
                error_msg = f"Maaf, terjadi kesalahan: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 10. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **💡 Tips Interaksi dengan Mentor:**
    - Ajukan pertanyaan spesifik tentang tantangan nyata yang Anda hadapi
    - Jangan ragu untuk meminta klarifikasi atau contoh lebih detail
    - Gunakan prompt seperti "Apa yang saya lewatkan?" atau "Bagaimana jika..."
    - Untuk hasil terbaik, berikan konteks yang cukup dalam pertanyaan Anda
    
    **Disclaimer:**
    - Sistem ini menggunakan AI sebagai alat bantu mentoring
    - Selalu verifikasi informasi penting dengan sumber terpercaya
    - Keputusan bisnis akhir tetap menjadi tanggung jawab Anda
    - **Data TIDAK REAL-TIME**, karena tidak terkoneksi dengan data terbaru di internet
    """
)






