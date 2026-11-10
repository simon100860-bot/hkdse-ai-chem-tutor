import os
import io
import re
import base64
import zipfile
import logging
import threading
import requests
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader

# 關閉 pypdf 的 CFF 字型警告洗版，大幅提升雲端解析 PDF 的啟動速度
logging.getLogger("pypdf").setLevel(logging.ERROR)

st.set_page_config(
    page_title="HKDSE Chem AI Tutor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 鎖定中轉站專屬模型名稱
FLASH_MODEL = "aistudio-gemini-3.8-flash"
PRO_MODEL = "gemini-3.1-pro-high"
TEACHER_PASSWORD = st.secrets.get("TEACHER_PASSWORD", "chem5star")

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

st.title("🧪 HKDSE Chemistry 24/7 AI Tutor (Bilingual Edition)")
st.caption("English-Primary & Chinese-Supplemented · Instant Streaming · Handwriting Grading · Auto-Sync to Google Sheets")

def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return ""
    return url if url.endswith("/v1") else f"{url}/v1"

# 側邊欄設定
with st.sidebar:
    st.header("🔑 API & System Settings")

    flash_api_key = st.secrets.get("FLASH_API_KEY") or st.secrets.get("GEMINI_API_KEY") or ""
    if not flash_api_key:
        flash_api_key = st.text_input("1A. Flash API Key (sk-)", type="password")

    pro_api_key = st.secrets.get("PRO_API_KEY") or st.secrets.get("GEMINI_API_KEY") or ""
    if not pro_api_key:
        pro_api_key = st.text_input("1B. Pro API Key (sk-)", type="password")

    raw_base_url = st.secrets.get("API_BASE_URL") or ""
    if not raw_base_url:
        raw_base_url = st.text_input("2. Proxy Base URL", placeholder="https://api.xxx.com/v1")
    api_base_url = normalize_base_url(raw_base_url)

    apps_script_url = st.secrets.get("APPS_SCRIPT_URL") or ""
    if not apps_script_url:
        apps_script_url = st.text_input("3. Google Apps Script URL")

    gdrive_urls_secret = st.secrets.get("GDRIVE_URLS") or ""

    routing_strategy = st.selectbox(
        "🧠 Model Routing Strategy",
        options=[
            f"⚡ Smart Routing (Concept: {FLASH_MODEL} / Long Q: {PRO_MODEL})",
            f"🚀 Speed First: Always use {FLASH_MODEL}",
            f"🎯 Depth First: Always use {PRO_MODEL}"
        ],
        index=0
    )

    enable_web_search = st.checkbox(
        "🌐 Enable Live Web Search Fallback",
        value=False,
        help="預設關閉以獲得最快秒回速度；勾選後若筆記無紀錄會額外搜尋 Chemguide/RSC 網頁。"
    )
    student_id = st.text_input("🧑‍🎓 Class & Name (選填)", placeholder="e.g., 5A (12) Chan Tai Man")

if (not flash_api_key and not pro_api_key) or not api_base_url:
    st.info("👈 請先在 `.streamlit/secrets.toml` 或點擊左上角展開側邊欄填入 API 密鑰與 Base URL。")
    st.stop()

if not flash_api_key:
    flash_api_key = pro_api_key
if not pro_api_key:
    pro_api_key = flash_api_key

@st.cache_resource
def get_clients(f_key: str, p_key: str, base_u: str):
    return (
        OpenAI(api_key=f_key, base_url=base_u, timeout=60.0, max_retries=1),
        OpenAI(api_key=p_key, base_url=base_u, timeout=60.0, max_retries=1)
    )

client_flash, client_pro = get_clients(flash_api_key, pro_api_key, api_base_url)

def encode_pil_image_fast(image: Image.Image) -> str:
    buffered = io.BytesIO()
    rgb_im = image.convert("RGB")
    rgb_im.thumbnail((800, 800), Image.Resampling.BILINEAR)
    rgb_im.save(buffered, format="JPEG", quality=75, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 輔助函式：將文字切塊並建立關鍵字集合
def index_text_segments(text: str, source_label: str, indexed_chunks: list):
    for i in range(0, len(text), 650):
        segment = text[i:i + 750].strip()
        if len(segment) > 40:
            tokens = set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", segment.lower()))
            indexed_chunks.append({
                "source": source_label,
                "text": segment,
                "tokens": tokens
            })

def parse_pdf_bytes(pdf_bytes: bytes, filename: str, indexed_chunks: list):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            index_text_segments(text, f"{filename} (P.{page_num+1})", indexed_chunks)
    except Exception:
        pass

# 自動從 Google Drive 分享網址或純 ID 提取 File ID
def extract_gdrive_id(item: str) -> str:
    item = item.strip()
    if not item:
        return ""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", item)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", item)
    if m:
        return m.group(1)
    return item

def download_from_gdrive(file_id: str) -> bytes:
    session = requests.Session()
    # 使用 drive.usercontent.google.com 自動跳過大檔案病毒掃描確認頁
    direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        res = session.get(direct_url, timeout=60)
        if res.status_code == 200 and not res.content[:15].lower().startswith(b"<!doctype html"):
            return res.content
        fallback_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        res = session.get(fallback_url, timeout=60)
        if res.status_code == 200 and not res.content[:15].lower().startswith(b"<!doctype html"):
            return res.content
    except Exception:
        pass
    return b""

# 1. 知識庫預編譯索引（支援本地 data/ 資料夾 + Google Drive 私密載入 PDF 或 ZIP 壓縮包）
@st.cache_resource(show_spinner="📚 正在從安全雲端載入並索引化學教材庫...")
def load_and_index_knowledge_base(gdrive_config: str):
    indexed_chunks = []

    # (A) 如果本地 data/ 資料夾有檔案（方便你在自己電腦測試），優先讀取
    if os.path.exists("data"):
        for fname in os.listdir("data"):
            fpath = os.path.join("data", fname)
            try:
                if fname.lower().endswith(".pdf"):
                    with open(fpath, "rb") as f:
                        parse_pdf_bytes(f.read(), fname, indexed_chunks)
                elif fname.lower().endswith((".txt", ".md")):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        index_text_segments(f.read(), fname, indexed_chunks)
            except Exception:
                pass

    # (B) 從 Streamlit Secrets 的 GDRIVE_URLS 私密下載教材（支援單一/多個 PDF 連結，或打包好的 .zip 連結）
    if gdrive_config:
        raw_items = [x.strip() for x in re.split(r"[,\n;]+", gdrive_config) if x.strip()]
        for idx, item in enumerate(raw_items):
            fid = extract_gdrive_id(item)
            if not fid:
                continue
            file_bytes = download_from_gdrive(fid)
            if not file_bytes:
                continue

            # 自動判斷下載的是 .zip 壓縮包還是單一 .pdf 檔案
            if file_bytes[:4] == b"PK\x03\x04":
                try:
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                        for zname in z.namelist():
                            if zname.startswith("__MACOSX") or zname.endswith("/"):
                                continue
                            clean_name = os.path.basename(zname)
                            if clean_name.lower().endswith(".pdf"):
                                parse_pdf_bytes(z.read(zname), clean_name, indexed_chunks)
                            elif clean_name.lower().endswith((".txt", ".md")):
                                text = z.read(zname).decode("utf-8", errors="ignore")
                                index_text_segments(text, clean_name, indexed_chunks)
                except Exception:
                    pass
            elif file_bytes[:4] == b"%PDF":
                parse_pdf_bytes(file_bytes, f"Cloud_Note_{idx+1}.pdf", indexed_chunks)
            else:
                try:
                    text = file_bytes.decode("utf-8", errors="ignore")
                    index_text_segments(text, f"Cloud_Note_{idx+1}.txt", indexed_chunks)
                except Exception:
                    pass

    return indexed_chunks

kb_chunks = load_and_index_knowledge_base(gdrive_urls_secret)

def retrieve_top_chunks_fast(query: str, top_k: int = 3):
    if not kb_chunks or not query.strip():
        return []
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", query.lower()))
    if not q_tokens:
        return []
    scored = []
    for item in kb_chunks:
        overlap = q_tokens.intersection(item["tokens"])
        score = sum(3 if len(t) > 2 else 1 for t in overlap)
        if score >= 2:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

@st.cache_data(ttl=3600, show_spinner=False)
def search_authoritative_chem_sites(query: str):
    if not HAS_DDGS:
        return "", []
    search_q = f"{query} (site:chemguide.co.uk OR site:chem.libretexts.org OR site:rsc.org OR site:hkeaa.edu.hk)"
    snippets, links = [], []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_q, max_results=3))
            for r in results:
                title, href, body = r.get("title", "Reference"), r.get("href", ""), r.get("body", "")
                snippets.append(f"[{title}]: {body}")
                if href:
                    links.append(f"- [{title}]({href})")
    except Exception:
        pass
    return "\n\n".join(snippets), links

def async_save_log(url, payload):
    def _task():
        try:
            requests.post(url, json=payload, timeout=8)
        except Exception:
            pass
    if url:
        threading.Thread(target=_task, daemon=True).start()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_logs_cached(url: str):
    if not url:
        return None, "尚未設定 Google Apps Script 網址。"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            return res.json(), None
        return None, f"讀取失敗，HTTP 狀態碼：{res.status_code}"
    except Exception as e:
        return None, f"連線 Apps Script 發生錯誤：{e}"

def get_system_prompt(mode: str, has_image: bool) -> str:
    bilingual_rule = """
【LANGUAGE & FORMATTING RULE (CRITICAL - MUST FOLLOW)】
- You MUST answer in an **English-Primary, Traditional Chinese-Supplemented (英文為主、中文為輔)** format.
- **English MUST ALWAYS come first**, followed immediately by a concise Traditional Chinese explanation/translation (先寫英文標準表述，後附繁體中文輔助解析).
- All chemical terms, equations, and marking scheme keywords MUST follow official **HKEAA HKDSE Chemistry** standards.
- Example format for every point:
  * **English:** Ethanoic acid is a weak monobasic acid that only partially ionizes in water to give mobile hydrogen ions ($H^+(aq)$).
  * **中文輔助：** 乙酸（Ethanoic acid）是一元弱酸，在水中僅部分電離（partially ionizes）產生可移動的氫離子。
"""

    image_instruction = """
【IMAGE & HANDWRITING GRADING INSTRUCTION】
- Carefully examine the uploaded image (printed question / experimental setup / student's handwritten equations & calculations).
- Perform a **Line-by-Line Red-Pen Grading (逐行紅筆批改)** in English first, then Chinese:
  1. Check chemical formulas, balancing coefficients, state symbols ((s), (l), (g), (aq)), ionic charges, and reversible arrows ($\rightleftharpoons$).
  2. Check calculation steps (moles, molarity, enthalpy change $\Delta H$), units ($\text{mol dm}^{-3}$, $\text{kJ mol}^{-1}$), and significant figures.
  3. Explicitly mark "✅ **Awarded [1M]**" and "❌ **Deducted / Common Trap (扣分位與原因)**".
""" if has_image else ""

    analytics_footer_rule = """
【SYSTEM ANALYTICS FOOTER (MANDATORY)】
At the very end of your response (after the Marking Scheme), start a new line with the exact separator `===ANALYTICS_META===`, followed on the next line by:
Topic Name | Concise summary of student's misconception or error (in bilingual format, under 35 words)

Choose the Topic Name strictly from:
Planet Earth / Microscopic World I / Metals / Acids and Bases / Fossil Fuels and Carbon Compounds / Microscopic World II / Redox, Chemical Cells and Electrolysis / Chemical Reactions and Energy / Rate of Reaction / Chemical Equilibrium / Chemistry of Carbon Compounds / Patterns in the Chemical World / Analytical Chemistry
"""

    if "Concept" in mode:
        return f"""
You are a top-tier HKDSE Chemistry teacher in Hong Kong. The student selected **Concept Clarification Mode**.
{bilingual_rule}
{image_instruction}
Strictly structure your response into these 4 sections:

1. **【1. Misconception Alert (常見迷思大破解)】**
   - ❌ **Common Misconception (常犯迷思):** State the wrong intuition in English first, then Chinese.
   - ✅ **Correct HKDSE Concept (正確 DSE 概念):** State the accurate HKEAA concept in English first, then Chinese.

2. **【2. In-Depth Microscopic Explanation (核心概念微觀詳解)】**
   - Explain from the microscopic particle level (ions, electrons, chemical bonds, intermolecular forces) or energetics.
   - Include full chemical equations and ionic equations with state symbols. (English explanation first, Chinese summary second).

3. **【3. Diagnostic Practice (概念診斷實戰題)】**
   - Create 1 high-quality HKDSE-style Multiple Choice Question (MC with options A, B, C, D) and 1 True/False statement in English (with Chinese translation in brackets).

4. **【4. Feedback & Marking Scheme (評卷參考與反饋)】**
   - Output the exact separator line `===MARKING_SCHEME===` first.
   - Below the separator, provide the correct answers and explain why the distractors are wrong (English first, Chinese second).
{analytics_footer_rule}
"""
    else:
        return f"""
You are a senior HKDSE Chemistry teacher and HKEAA marker. The student selected **Long Question Breakdown & Grading Mode**.
{bilingual_rule}
{image_instruction}
Strictly structure your response into these 5 sections:

1. **【1. Question Breakdown & Handwriting Grading (審題拆解與手寫批改)】**
   - Analyze command words (*Explain*, *Deduce*, *Describe*, *Calculate*) and grade the student's uploaded work if an image is provided (English first, Chinese second).

2. **【2. Step-by-Step Scoring Points (逐分得分位拆解)】**
   - List official HKEAA scoring points using `* [1M]` to highlight exact English keywords required for marks, followed by Chinese explanation.
   - Highlight **Fatal Traps (奪命失分陷阱)** (e.g., omitting "mobile ions", missing state symbols, wrong units).

3. **【3. HKDSE Full-Mark Model Answer (DSE 滿分示範答案)】**
   - Provide a rigorous, exam-ready English Model Answer first (with step-by-step calculation or balanced equations), followed by a Traditional Chinese reference translation.

4. **【4. Follow-Up Exam Practice (同類題型進階實戰)】**
   - Create 1 similar HKDSE Structured Question (indicating marks, e.g., `[3 marks]`) in English first, followed by Chinese translation.

5. **【5. Practice Marking Scheme (實戰練習評卷參考)】**
   - Output the exact separator line `===MARKING_SCHEME===` first.
   - Below the separator, provide the point-by-point `[1M]` Marking Scheme (English first, Chinese second).
{analytics_footer_rule}
"""

def resolve_target_model(mode: str, strategy: str, has_image: bool) -> str:
    if f"Always use {FLASH_MODEL}" in strategy:
        return FLASH_MODEL
    if f"Always use {PRO_MODEL}" in strategy:
        return PRO_MODEL
    return PRO_MODEL if ("Long Question" in mode or has_image) else FLASH_MODEL

def stream_chat_response(model_name: str, prompt_text: str, system_prompt: str, pil_image: Image.Image = None):
    messages = [{"role": "system", "content": system_prompt}]
    if pil_image is not None:
        b64_img = encode_pil_image_fast(pil_image)
        user_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ]
    else:
        user_content = prompt_text
    messages.append({"role": "user", "content": user_content})

    if model_name == PRO_MODEL:
        fallback_chain = [PRO_MODEL, FLASH_MODEL]
    else:
        fallback_chain = [FLASH_MODEL, PRO_MODEL]

    last_err = None
    for candidate in fallback_chain:
        active_client = client_pro if candidate == PRO_MODEL else client_flash
        try:
            stream = active_client.chat.completions.create(
                model=candidate,
                messages=messages,
                temperature=0.25,
                stream=True
            )
            return stream, candidate
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"模型調用失敗（請確認中轉站模型名稱 `{PRO_MODEL}` / `{FLASH_MODEL}` 與密鑰餘額）。詳情：{last_err}")

with st.sidebar:
    st.markdown("---")
    st.success(f"📚 Local KB: {len(kb_chunks)} chunks indexed")

tab_student, tab_teacher = st.tabs(["🙋‍♂️ Student Q&A & Grading (學生發問區)", "👨‍🏫 Teacher Analytics (阿 Sir 後台)"])

with tab_student:
    col1, col2 = st.columns([3, 2])
    with col1:
        selected_mode = st.radio(
            "👇 Step 1: Select Question Mode (選擇發問模式)",
            options=[
                "💡 Ask Concept (Clarify Misconceptions / 釐清概念迷思)",
                "📝 Ask Long Question / Grade Handwriting (拆解得分位與批改)"
            ]
        )
    with col2:
        uploaded_file = st.file_uploader(
            "📸 Step 2 (Optional): Upload Photo / Screenshot (上傳題目或手寫算式)",
            type=["png", "jpg", "jpeg"]
        )

    pil_image = Image.open(uploaded_file) if uploaded_file else None
    if pil_image:
        with st.expander("🖼️ Preview Uploaded Image (預覽圖片)", expanded=False):
            st.image(pil_image, width=300)

    student_query = st.chat_input("Ask any HKDSE Chemistry question in English or Chinese...")

    if student_query:
        with st.chat_message("user"):
            st.markdown(f"**[{selected_mode.split('(')[0].strip()}]**\n\n{student_query}")
            if pil_image:
                st.image(pil_image, width=240)

        with st.chat_message("assistant"):
            target_model = resolve_target_model(selected_mode, routing_strategy, pil_image is not None)
            system_prompt = get_system_prompt(selected_mode, pil_image is not None)

            matched_chunks = retrieve_top_chunks_fast(student_query, top_k=3)
            web_links = []

            if matched_chunks:
                kb_context = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in matched_chunks])
                prompt_text = f"【Internal School Notes Reference】:\n{kb_context}\n\n【Student Question】:\n{student_query}\nPlease prioritize the above school notes and answer in English-First, Chinese-Second format."
                source_prefix = "📚 Source: Internal School Notes (校內筆記庫)"
            else:
                web_context = ""
                if enable_web_search:
                    web_context, web_links = search_authoritative_chem_sites(student_query)
                prompt_text = f"【Authoritative Reference】:\n{web_context if web_context else 'Follow official HKEAA HKDSE Chemistry, Chemguide, RSC, and LibreTexts standards.'}\n\n【Student Question】:\n{student_query}\nPlease answer in English-First, Chinese-Second format."
                source_prefix = "🌐 Source: HKEAA / Authoritative Chemistry Knowledge Base"

            status_box = st.empty()
            stream_placeholder = st.empty()
            full_response = ""

            with st.spinner(f"⚡ Connecting to {target_model}..."):
                stream_gen, used_model = stream_chat_response(target_model, prompt_text, system_prompt, pil_image)

            key_tag = "Pro Key" if used_model == PRO_MODEL else "Flash Key"
            status_box.info(f"{source_prefix} · Powered by `{used_model}` ({key_tag})")

            chunk_counter = 0
            for chunk in stream_gen:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    chunk_counter += 1
                    if chunk_counter % 4 == 0:
                        visible_part = full_response.split("===MARKING_SCHEME===")[0].split("===ANALYTICS_META===")[0]
                        stream_placeholder.markdown(visible_part + " ▌")

            clean_answer = full_response
            topic_tag = "Uncategorized"
            issue_summary = "General Concept / Question Breakdown"

            if "===ANALYTICS_META===" in full_response:
                clean_answer, meta_part = full_response.split("===ANALYTICS_META===", 1)
                meta_line = meta_part.strip().split("\n")[0]
                if "|" in meta_line:
                    parts = meta_line.split("|", 1)
                    topic_tag, issue_summary = parts[0].strip(), parts[1].strip()

            if "===MARKING_SCHEME===" in clean_answer:
                explanation_part, ms_part = clean_answer.split("===MARKING_SCHEME===", 1)
                stream_placeholder.markdown(explanation_part)
                with st.expander("✅ Click to View Marking Scheme & Detailed Feedback (展開查看評卷參考)"):
                    st.markdown(ms_part.strip())
            else:
                stream_placeholder.markdown(clean_answer)

            if web_links:
                with st.expander("🔗 Authoritative Web References"):
                    for link in set(web_links):
                        st.markdown(link)

            async_save_log(apps_script_url, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "student_id": student_id.strip() if student_id.strip() else "Anonymous",
                "mode": "Concept 迷思" if "Concept" in selected_mode else "Long Q / 批改",
                "topic": topic_tag,
                "question": student_query[:150].replace("\n", " "),
                "key_issue": issue_summary,
                "source": "校內筆記庫" if matched_chunks else "權威知識庫",
                "has_img": "是" if pil_image else "否"
            })

with tab_teacher:
    st.subheader("📊 Cloud Student Error & Misconception Analytics")
    col_pwd, col_btn = st.columns([3, 1])
    with col_pwd:
        pwd_input = st.text_input("🔒 Teacher Password", type="password")
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Cloud Data"):
            fetch_logs_cached.clear()

    if pwd_input == TEACHER_PASSWORD:
        records, err_msg = fetch_logs_cached(apps_script_url)
        if err_msg:
            st.error(err_msg)
        elif not records:
            st.info("No student questions logged in Google Sheets yet.")
        else:
            df = pd.DataFrame(records)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Questions", f"{len(df)}")
            m2.metric("Concept Queries", f"{len(df[df['發問模式'] == 'Concept 迷思'])}")
            m3.metric("Long Q / Grading", f"{len(df[df['發問模式'] == 'Long Q / 批改'])}")
            top_topic = df["DSE 課題分類"].mode()[0] if not df["DSE 課題分類"].mode().empty else "N/A"
            m4.metric("Top Struggled Topic", str(top_topic).split("(")[0].strip())

            st.markdown("**📈 Questions by HKDSE Topic**")
            st.bar_chart(df["DSE 課題分類"].value_counts())

            st.markdown("**📋 Detailed Student Misconception Log**")
            st.dataframe(df.iloc[::-1], use_container_width=True)
    elif pwd_input:
        st.error("Incorrect password!")