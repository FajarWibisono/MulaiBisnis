# MulaiBisnis — Teman Diskusi Kewirausahaan (TEDIWIRA)

Chatbot mentoring kewirausahaan berbasis Streamlit + Groq, dilengkapi review proposal bisnis (PDF) dan ekspor transkrip percakapan.

## Fitur

- Chat mentor AI berbahasa Indonesia dengan jawaban streaming.
- Review proposal bisnis PDF: ringkasan, potensi keuntungan, risiko, dan pertanyaan investor.
- Hasil analisis proposal otomatis dipakai sebagai konteks chat (bisa dimatikan lewat checkbox).
- Analisis proposal hanya dijalankan sekali per file (hash + cache), tidak diulang setiap chat.
- Batas aman konteks: proposal 30.000 karakter, riwayat chat 12 pesan terakhir / 12.000 karakter.
- Batas **7 pertanyaan per sesi browser** untuk menjaga kuota API.
- Unduh transkrip percakapan + analisis dalam format **HTML** yang rapi dan siap dicetak ke PDF.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
```

Buat file `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_xxxxxxxx"
```

Lalu jalankan:

```bash
streamlit run MulaiBisnis.py
```

## Konfigurasi opsional (environment variable)

| Variabel | Default | Keterangan |
|---|---|---|
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-120b` | Model untuk chat |
| `GROQ_PROPOSAL_MODEL` | `openai/gpt-oss-20b` | Model untuk analisis proposal |
| `GROQ_MAX_TOKENS_CHAT` | `3000` | Maksimum token jawaban chat |
| `GROQ_MAX_TOKENS_PROPOSAL` | `4096` | Maksimum token analisis proposal |
| `MAX_PROPOSAL_CONTEXT_CHARS` | `4000` | Panjang analisis yang dikirim ke konteks chat |

## Catatan penting

- Percakapan hanya tersimpan di `st.session_state`. **Refresh/close tab akan menghapus riwayat** — unduh transkrip HTML bila ingin menyimpannya.
- Teks proposal dan pertanyaan dikirim ke layanan Groq untuk diproses; tidak ada penyimpanan di server aplikasi.
- Jangan commit `.streamlit/secrets.toml` (sudah masuk `.gitignore`).
- Untuk Streamlit Community Cloud: tambahkan `GROQ_API_KEY` melalui menu **Secrets**, lalu deploy dari repo ini.
