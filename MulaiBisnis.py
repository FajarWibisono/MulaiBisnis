# MulaiBisnis.py
# TEMAN DISKUSI KEWIRAUSAHAAN (TEDIWIRA)
import hashlib
import html
import io
import json
import os
from datetime import datetime

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI APLIKASI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WIRAUSAHA",
    page_icon="💰",
    layout="wide",
)

MODEL_CHAT = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
MODEL_PROPOSAL = os.getenv("GROQ_PROPOSAL_MODEL", "openai/gpt-oss-20b")

MAX_TOKENS_CHAT = int(os.getenv("GROQ_MAX_TOKENS_CHAT", "3000"))
MAX_TOKENS_PROPOSAL = int(os.getenv("GROQ_MAX_TOKENS_PROPOSAL", "4096"))

TEMP_CHAT = 0.5
TEMP_PROPOSAL = 0.35

# Batas agar prompt tidak melebihi context window model / tagihan API
MAX_PROPOSAL_CHARS = 30_000      # ± 8-10 ribu token
MAX_HISTORY_MESSAGES = 12        # hanya pesan terakhir yang dikirim ke LLM
MAX_HISTORY_CHARS = 12_000
MAX_PROPOSAL_CONTEXT_CHARS = int(os.getenv("MAX_PROPOSAL_CONTEXT_CHARS", "4000"))

PROMPT_VERSION = "v3"            # ubah jika prompt di bawah diubah (cache buster)

try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    st.error(
        "GROQ_API_KEY belum diset. Tambahkan `GROQ_API_KEY = \"gsk_...\"` "
        "di `.streamlit/secrets.toml` atau environment variable, lalu jalankan ulang aplikasi."
    )
    st.stop()

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
        .analysis-box { background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; }
        .startup-info { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
        .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; color: #721c24; }
        .attach-box { background-color: #eef7ee; border: 1px solid #cfe8cf; padding: 0.75rem 1rem; border-radius: 0.5rem; margin: 0.5rem 0; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul Aplikasi
st.title("💰TEMAN DISKUSI KEWIRAUSAHAAN")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Tentang Kewirausahaan

    **Chat Bot ini akan membantu Anda memahami lebih dalam tentang dunia KEWIRAUSAHAAN** dan berbagai hal-hal yang perlu di perhatikan baik pada masa persiapan, pelaksanaan, pengembangan, dan bahkan exit strategy.

    Apps ini juga dapat membantu Anda untuk **MERIVIEW PROPOSAL TAWARAN BISNIS atau KERJASAMA yang datang pada ANDA**.

    **Pastikanlah Anda memiliki koneksi internet yang baik dan stabil.**
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "chat_history": [],          # [{role, content}]
    "proposal_analysis": None,   # hasil analisis LLM (markdown)
    "proposal_error": None,      # pesan error analisis
    "proposal_file": None,       # {name, bytes}
    "proposal_hash": None,       # sha256 file, penanda file baru
    "proposal_pages": None,
    "proposal_chars": 0,
    "proposal_truncated": False,
    "analysis_at": None,
    "include_proposal_context": True,  # kirim hasil analisis proposal ke konteks chat
    "analysis_nonce": 0,         # dinaikkan saat user minta analisis ulang
    "proposal_version": 0,       # dinaikkan saat lampiran dihapus (reset uploader)
    "last_failed_prompt": None,
}
for _key, _value in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT MENTOR (CHAT)
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

KONTEKS PROPOSAL BISNIS PENGGUNA (hasil analisis Anda sebelumnya, gunakan sebagai acuan bila pertanyaan berkaitan dengan proposal):
{proposal_context}

Riwayat Chat:
{chat_history}

Pertanyaan: {question}

Jawaban (sebagai mentor ahli):"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["proposal_context", "chat_history", "question"],
    template=PROMPT_MENTOR_AHLI,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROMPT ANALISIS PROPOSAL
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

Jawaban (sebagai mentor ahli):"""


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI BANTU (EKSTRAKSI PDF, ANALISIS, TRANSCRIPT)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """Ekstrak teks PDF langsung dari memori (tanpa file sementara)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip(), len(reader.pages)


@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float, max_tokens: int) -> ChatGroq:
    """Cache instance LLM per konfigurasi agar tidak dibuat ulang tiap rerun."""
    return ChatGroq(model_name=model, temperature=temperature, max_tokens=max_tokens)


@st.cache_data(show_spinner=False)
def analyze_proposal_text(proposal_text: str, nonce: int) -> str:
    """
    Analisis proposal. `nonce` membuat cache bisa dipaksa lewat tombol Refresh.
    Hasil di-cache per teks proposal agar analisis tidak diulang saat user chat.
    """
    llm = get_llm(MODEL_PROPOSAL, TEMP_PROPOSAL, MAX_TOKENS_PROPOSAL)
    prompt = PROPOSAL_ANALYSIS_PROMPT.format(proposal_text=proposal_text)
    return llm.invoke(prompt).content


def run_proposal_analysis() -> None:
    """Jalankan analisis untuk lampiran yang tersimpan di session_state."""
    proposal = st.session_state.proposal_file
    if not proposal:
        return

    st.session_state.proposal_error = None
    try:
        text, pages = extract_pdf_text(proposal["bytes"])
        st.session_state.proposal_pages = pages
        st.session_state.proposal_chars = len(text)

        if not text.strip():
            raise ValueError(
                "Tidak ada teks yang bisa dibaca dari PDF ini (kemungkinan hasil scan/gambar). "
                "Gunakan PDF berbasis teks atau jalankan OCR terlebih dahulu."
            )

        truncated = text[:MAX_PROPOSAL_CHARS]
        st.session_state.proposal_truncated = len(text) > MAX_PROPOSAL_CHARS
        with st.spinner("🧠 Mentor sedang menganalisis proposal Anda dengan seksama..."):
            st.session_state.proposal_analysis = analyze_proposal_text(
                truncated, st.session_state.analysis_nonce
            )
        st.session_state.analysis_at = datetime.now().isoformat(timespec="seconds")
    except Exception as exc:
        st.session_state.proposal_analysis = None
        st.session_state.proposal_error = str(exc)


def build_chat_history_str(history: list[dict]) -> str:
    """Ambil hanya beberapa pesan terakhir agar prompt tetap ringkas."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    while recent:
        lines = [f"{msg['role']}: {msg['content']}" for msg in recent]
        if len("\n".join(lines)) <= MAX_HISTORY_CHARS:
            return "\n".join(lines)
        recent.pop(0)
    return ""


def _token_stream(llm_stream):
    """Ubah stream AIMessageChunk menjadi stream teks biasa (kompatibel semua versi Streamlit)."""
    for chunk in llm_stream:
        text = getattr(chunk, "content", chunk)
        if text:
            yield text if isinstance(text, str) else str(text)


def build_proposal_context() -> str:
    """
    Ringkas hasil analisis proposal untuk dijadikan konteks chat.
    Dibatasi agar prompt tidak membengkak dan biaya token tetap terkendali.
    """
    analysis = st.session_state.proposal_analysis
    if not st.session_state.include_proposal_context or not analysis:
        return "(tidak ada proposal bisnis yang dilampirkan. Jawab sebagai mentor umum.)"

    proposal = st.session_state.proposal_file
    header = (
        f"File: {proposal['name']} ({st.session_state.proposal_pages or '?'} halaman)"
        if proposal
        else ""
    )

    analysis = analysis.strip()
    if len(analysis) > MAX_PROPOSAL_CONTEXT_CHARS:
        analysis = (
            analysis[:MAX_PROPOSAL_CONTEXT_CHARS].rstrip()
            + "\n[... ringkasan analisis dipotong karena batas konteks ...]"
        )

    return "\n".join(part for part in (header, analysis) if part)


def generate_answer(question: str) -> str:
    """Panggil LLM dan stream jawabannya ke UI. Mengembalikan teks jawaban."""
    history_str = build_chat_history_str(st.session_state.chat_history[:-1])
    llm = get_llm(MODEL_CHAT, TEMP_CHAT, MAX_TOKENS_CHAT)
    formatted_prompt = INDO_PROMPT_TEMPLATE.format(
        proposal_context=build_proposal_context(),
        chat_history=history_str or "(belum ada riwayat)",
        question=question,
    )
    answer = st.write_stream(_token_stream(llm.stream(formatted_prompt)))
    return answer if isinstance(answer, str) else "".join(str(chunk) for chunk in answer)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def transcript_available() -> bool:
    return bool(st.session_state.chat_history or st.session_state.proposal_analysis)


TRANSCRIPT_CSS = """
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --brand:#2563eb; --brand-2:#7c3aed; --bg:#eef2f7;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  line-height:1.7;font-size:16px}
.page{max-width:880px;margin:0 auto;padding:32px 20px 56px}
.hero{background:linear-gradient(135deg,var(--brand),var(--brand-2));color:#fff;
  padding:30px 32px;border-radius:20px;box-shadow:0 16px 40px rgba(37,99,235,.28)}
.hero .top{display:flex;justify-content:space-between;gap:12px;align-items:center;
  font-size:13px;letter-spacing:.02em;opacity:.9;flex-wrap:wrap}
.hero h1{margin:14px 0 6px;font-size:28px;line-height:1.25;font-weight:800}
.hero p{margin:0;opacity:.92;font-size:15px}
.chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0 26px}
.chip{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.chip span{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.chip strong{display:block;margin-top:2px;font-size:14px;word-break:break-word}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:24px 26px;margin:0 0 26px}
.card>h2{margin:0 0 14px;font-size:18px}
.chat>h2{font-size:18px;margin:0 0 16px}
.msg{margin:0 0 18px;display:flex;flex-direction:column;max-width:92%}
.msg.user{margin-left:auto;align-items:flex-end}
.msg.mentor{margin-right:auto;align-items:flex-start}
.who{font-size:12px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;
  color:var(--muted);margin:0 6px 6px}
.bubble{padding:16px 20px;border-radius:16px;width:100%}
.msg.user .bubble{background:#dbeafe;border:1px solid #bfdbfe;border-bottom-right-radius:4px}
.msg.mentor .bubble{background:#fff;border:1px solid var(--line);
  border-bottom-left-radius:4px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.md>*:first-child{margin-top:0}
.md>*:last-child{margin-bottom:0}
.md h1,.md h2,.md h3,.md h4{line-height:1.35;margin:18px 0 8px}
.md h1{font-size:20px}
.md h2{font-size:18px}
.md h3{font-size:16px}
.md p{margin:9px 0}
.md ul,.md ol{margin:9px 0;padding-left:22px}
.md li{margin:4px 0}
.md strong{font-weight:700}
.md a{color:var(--brand);word-break:break-word}
.md code{background:#f1f5f9;border:1px solid var(--line);border-radius:6px;
  padding:1px 5px;font-size:.9em}
.md pre{background:#0f172a;color:#e2e8f0;border-radius:12px;padding:14px 16px;overflow:auto}
.md pre code{background:none;border:0;color:inherit;padding:0}
.md blockquote{margin:10px 0;padding:8px 14px;border-left:3px solid var(--brand);
  background:#f8fafc;color:#334155}
.md table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
.md th,.md td{border:1px solid var(--line);padding:8px 10px;text-align:left}
.md th{background:#f8fafc}
.sources{margin:8px 6px 0;font-size:13px;color:#334155}
.sources summary{cursor:pointer;font-weight:600;color:var(--brand)}
.sources ul{margin:8px 0 0;padding-left:20px}
.sources li{margin:4px 0}
.empty{color:var(--muted);font-style:italic}
footer{margin-top:8px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}
footer strong{color:#334155}
@media print{
  body{background:#fff}
  .page{max-width:none;padding:0}
  .hero{border-radius:0;box-shadow:none;padding:22px 24px}
  .card,.bubble{box-shadow:none}
  .msg{max-width:100%}
  footer{page-break-inside:avoid}
}
@page{margin:14mm}
"""


def _md_to_html(text: str) -> str:
    """Konversi markdown ke HTML; '<' di-escape agar HTML mentah tidak ikut tersisip."""
    escaped = (text or "").strip().replace("<", "&lt;")
    if not escaped:
        return ""
    try:
        import markdown as _markdown

        return _markdown.markdown(
            escaped, extensions=["extra", "sane_lists", "nl2br"]
        )
    except Exception:
        return "<p>" + escaped.replace("\n", "<br>") + "</p>"


def build_transcript_html() -> str:
    """Bangun transkrip HTML mandiri (tanpa aset eksternal) untuk diunduh."""
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    proposal = st.session_state.proposal_file
    history = st.session_state.chat_history

    chips = [
        ("Lampiran", html.escape(proposal["name"]) if proposal else "Tidak ada"),
        ("Ukuran", human_size(len(proposal["bytes"])) if proposal else "-"),
        ("Halaman", str(st.session_state.proposal_pages or "-")),
        ("Jumlah pesan", str(len(history))),
    ]
    chips_html = "".join(
        f'<div class="chip"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in chips
    )

    analysis_html = ""
    if st.session_state.proposal_analysis:
        analysis_html = (
            '<section class="card"><h2>🔍 Analisis Proposal Bisnis</h2>'
            f'<div class="md">{_md_to_html(st.session_state.proposal_analysis)}</div>'
            "</section>"
        )

    messages_html = []
    for message in history:
        is_user = message.get("role") == "user"
        label = "Anda" if is_user else "Mentor (TEDIWIRA)"
        cls = "user" if is_user else "mentor"
        sources_html = ""
        sources = message.get("sources") or []
        if sources:
            items = "".join(
                "<li><strong>{}</strong>{}<br><span>{}</span></li>".format(
                    html.escape(str(source.get("file", "?"))),
                    f" · hal. {source['page']}" if source.get("page") else "",
                    html.escape(str(source.get("snippet", ""))),
                )
                for source in sources
            )
            sources_html = (
                f'<details open class="sources"><summary>📚 Sumber ({len(sources)})</summary>'
                f"<ul>{items}</ul></details>"
            )
        messages_html.append(
            f'<article class="msg {cls}"><div class="who">{label}</div>'
            f'<div class="bubble md">{_md_to_html(message.get("content", ""))}</div>'
            f"{sources_html}</article>"
        )
    if not messages_html:
        messages_html.append('<p class="empty">Belum ada percakapan.</p>')

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transkrip Diskusi Kewirausahaan — TEDIWIRA</title>
<style>
{TRANSCRIPT_CSS}
</style>
</head>
<body>
<main class="page">
  <header class="hero">
    <div class="top"><span>📓 TEDIWIRA</span><span>{now}</span></div>
    <h1>Transkrip Diskusi Kewirausahaan</h1>
    <p>Teman Diskusi Kewirausahaan · diekspor otomatis dari aplikasi</p>
  </header>
  <section class="chips">{chips_html}</section>
  {analysis_html}
  <section class="chat">
    <h2>💬 Percakapan</h2>
    {"".join(messages_html)}
  </section>
  <footer>
    <strong>Disklaimer:</strong> Jawaban dihasilkan oleh AI dan dapat tidak akurat.
    Verifikasi informasi penting dengan sumber terpercaya sebelum mengambil keputusan bisnis.
  </footer>
</main>
</body>
</html>"""


def build_transcript_json() -> str:
    proposal = st.session_state.proposal_file
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "proposal": (
            {
                "name": proposal["name"],
                "size_bytes": len(proposal["bytes"]),
                "pages": st.session_state.proposal_pages,
                "chars": st.session_state.proposal_chars,
                "analyzed_at": st.session_state.analysis_at,
                "analysis": st.session_state.proposal_analysis,
            }
            if proposal
            else None
        ),
        "chat_history": st.session_state.chat_history,
        "model": MODEL_CHAT,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def reset_proposal_state() -> None:
    """Callback: hapus lampiran + analisis dan reset widget uploader."""
    st.session_state.proposal_version += 1
    st.session_state.proposal_file = None
    st.session_state.proposal_hash = None
    st.session_state.proposal_analysis = None
    st.session_state.proposal_error = None
    st.session_state.proposal_pages = None
    st.session_state.proposal_chars = 0
    st.session_state.proposal_truncated = False
    st.session_state.analysis_at = None
    st.session_state.analysis_nonce += 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR: UPLOAD, STATUS LAMPIRAN, REFRESH & RESET
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Review Proposal Bisnis")
    st.markdown("Unggah file PDF proposal bisnis untuk dianalisis oleh mentor AI.")

    uploaded_file = st.file_uploader(
        "📁 Pilih file PDF",
        type="pdf",
        key=f"proposal_uploader_v{st.session_state.proposal_version}",
    )

    # Analisis otomatis HANYA saat file baru diunggah (bukan tiap rerun/chat).
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash != st.session_state.proposal_hash:
            st.session_state.proposal_hash = file_hash
            st.session_state.proposal_file = {
                "name": uploaded_file.name,
                "bytes": file_bytes,
            }
            st.session_state.proposal_analysis = None
            st.session_state.proposal_error = None
            st.session_state.analysis_nonce += 1
            run_proposal_analysis()

    # Status lampiran
    if st.session_state.proposal_file:
        proposal = st.session_state.proposal_file
        pages = st.session_state.proposal_pages or "-"
        st.markdown(
            f"""
            <div class="attach-box">
                📎 <strong>{html.escape(proposal['name'])}</strong><br>
                Ukuran: {human_size(len(proposal['bytes']))} · Halaman: {pages}<br>
                Status: {"✅ sudah dianalisis" if st.session_state.proposal_analysis else "⏳ belum dianalisis"}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.checkbox(
        "🧠 Sertakan analisis proposal di konteks chat",
        key="include_proposal_context",
        disabled=not st.session_state.proposal_analysis,
        help=(
            "Jika aktif, mentor mengingat hasil analisis proposal saat menjawab "
            "pertanyaan lanjutan. Matikan untuk menghemat token."
        ),
    )

    col_refresh, col_reset = st.columns(2)
    with col_refresh:
        refresh_all = st.button(
            "🔄 Refresh",
            help="Analisis ulang lampiran & muat ulang tampilan obrolan.",
            use_container_width=True,
            type="primary",
        )
    with col_reset:
        st.button(
            "🗑️ Hapus",
            help="Hapus lampiran dan hasil analisisnya.",
            on_click=reset_proposal_state,
            use_container_width=True,
        )

    if refresh_all:
        st.cache_data.clear()  # buang cache teks & analisis
        st.session_state.analysis_nonce += 1
        if st.session_state.proposal_file:
            run_proposal_analysis()
        st.toast("Obrolan dan lampiran dimuat ulang ✅")
        st.rerun()

    if st.session_state.proposal_error:
        st.error(f"Analisis gagal: {st.session_state.proposal_error}")
        if st.button("🔁 Coba analisis lagi", use_container_width=True):
            st.cache_data.clear()
            st.session_state.analysis_nonce += 1
            run_proposal_analysis()
            st.rerun()

    st.markdown("---")
    if st.button(
        "🧹 Mulai percakapan baru",
        help="Hapus riwayat obrolan (lampiran tetap).",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.session_state.last_failed_prompt = None
        st.toast("Riwayat obrolan dibersihkan")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 7. HASIL ANALISIS PROPOSAL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.proposal_analysis:
    head_left, head_right = st.columns([4, 1])
    with head_left:
        st.subheader("🔍 Hasil Analisis Proposal Bisnis oleh TEDIWIRA")
        if st.session_state.analysis_at:
            st.caption(
                "Dianalisis pada "
                + datetime.fromisoformat(st.session_state.analysis_at).strftime(
                    "%d-%m-%Y %H:%M"
                )
            )
    with head_right:
        if st.button("🧠 Analisis ulang", use_container_width=True):
            st.cache_data.clear()
            st.session_state.analysis_nonce += 1
            run_proposal_analysis()
            st.rerun()

    if st.session_state.proposal_truncated:
        st.warning(
            f"Teks proposal lebih dari {MAX_PROPOSAL_CHARS:,} karakter, jadi hanya bagian awal "
            "yang dianalisis. Untuk analisis menyeluruh, unggah proposal yang lebih ringkas "
            "atau pecah per bab."
        )

    with st.container(border=True):
        st.markdown(st.session_state.proposal_analysis)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────
chat_left, chat_right = st.columns([5, 1])
with chat_left:
    st.subheader("💬 Diskusi Interaktif dengan **TEMAN DISKUSI Kewirausahaan**")
    st.markdown(
        "*Ajukan pertanyaan spesifik tentang ide bisnis, strategi, atau tantangan yang Anda hadapi*"
    )
    if st.session_state.proposal_analysis and st.session_state.include_proposal_context:
        st.caption(
            "🧠 Konteks aktif: jawaban mentor mempertimbangkan hasil analisis proposal "
            "yang Anda lampirkan."
        )
with chat_right:
    st.write("")
    st.download_button(
        "⬇️ Unduh chat (.html)",
        data=build_transcript_html(),
        file_name=f"transkrip_mulaibisnis_{datetime.now():%Y%m%d_%H%M}.html",
        mime="text/html",
        use_container_width=True,
        disabled=not transcript_available(),
    )

if transcript_available():
    st.download_button(
        "⬇️ Unduh data lengkap (.json)",
        data=build_transcript_json(),
        file_name=f"data_mulaibisnis_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
    )

# 8.1 Riwayat chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8.2 Retry jika permintaan terakhir gagal
if st.session_state.last_failed_prompt:
    st.warning("Jawaban terakhir gagal dibuat.")
    if st.button("🔁 Coba jawab ulang pertanyaan terakhir"):
        question = st.session_state.last_failed_prompt
        with st.chat_message("assistant"):
            try:
                answer = generate_answer(question)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_failed_prompt = None
                st.rerun()
            except Exception as exc:
                st.error(f"Maaf, terjadi kesalahan: {exc}")

# 8.3 Chat input
prompt = st.chat_input("✍️Tulis pertanyaan Anda tentang kewirausahaan...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.last_failed_prompt = None
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = generate_answer(prompt)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as exc:
            st.session_state.last_failed_prompt = prompt
            st.error(f"Maaf, terjadi kesalahan: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FOOTER & DISCLAIMER
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
