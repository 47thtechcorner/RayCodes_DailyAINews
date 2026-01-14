import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import feedparser
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from huggingface_hub import HfApi, hf_hub_download
from dateutil import parser
import pandas as pd

# ==========================================
# ⚙️ CONFIGURATION & SETUP
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USER_NAME = os.getenv("USER_NAME")

DEFAULT_MODEL = "gemini-1.5-flash"
REQUESTED_MODEL = "gemini-3-flash-preview" # Using latest confirmed flash
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
CONFIG_REPO_ID = f"{USER_NAME}/user-preferences"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("⚠️ GEMINI_API_KEY not found!")

# ==========================================
# 🎨 MOBILE-FIRST UI THEME & CSS
# ==========================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --primary-glow: rgba(99, 102, 241, 0.4);
    --bg-dark: #020617;
    --card-bg: rgba(15, 23, 42, 0.8);
    --text-main: #f8fafc;
    --text-dim: #94a3b8;
    --radius-lg: 28px;
    --radius-md: 18px;
}

body { 
    font-family: 'Outfit', sans-serif; 
    background-color: var(--bg-dark); 
    color: var(--text-main);
    margin: 0;
    padding: 0;
}

/* Mobile-First Container (Narrows Desktop view to Mobile App aspect) */
.gradio-container { 
    max-width: 550px !important; 
    margin: 0 auto !important;
    padding: 10px !important;
    border: none !important;
    min-height: 100vh;
}

.hero-header {
    background: linear-gradient(160deg, #4f46e5 0%, #7c3aed 100%);
    padding: 2.5rem 1rem;
    border-radius: var(--radius-lg);
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -10px var(--primary-glow);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-title { 
    color: white; 
    font-size: 2.8rem; 
    font-weight: 800; 
    margin: 0; 
    letter-spacing: -0.04em;
    text-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.hero-subtitle { 
    color: rgba(255,255,255,0.9); 
    font-size: 1.1rem; 
    margin-top: 0.5rem; 
    font-weight: 400;
    letter-spacing: 0.02em;
}

.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
}

.primary-btn {
    background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: var(--radius-md) !important;
    height: 64px !important; /* Larger touch target */
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
    font-size: 1.1rem !important;
    width: 100%;
}

.primary-btn:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 12px 24px -6px var(--primary-glow) !important; 
}

.primary-btn:active {
    transform: scale(0.98);
}

/* Tabs Styling for Mobile */
.tabs {
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
}

.tab-nav button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}

/* Input Fields Overhaul */
.pref-input textarea {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
    color: #fff !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    font-size: 1rem !important;
    font-family: 'Outfit', sans-serif !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    resize: none !important;
}

.pref-input textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15), inset 0 2px 4px rgba(0,0,0,0.2) !important;
}

.log-terminal textarea {
    font-family: 'Fira Code', monospace;
    background-color: #000000 !important;
    color: #4ade80 !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    font-size: 0.85rem !important;
}

.prose {
    font-family: 'Outfit', sans-serif !important;
    line-height: 1.6;
}

.prose h2 { 
    color: #c7d2fe; 
    font-size: 1.4rem;
    margin-top: 2rem;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
}

.prose table { 
    display: block;
    overflow-x: auto;
    border-radius: var(--radius-md);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Final Progress Bar Fix - No Overlap, Clean Flow */
.progress-view {
    background: var(--card-bg) !important;
    backdrop-filter: blur(20px) !important;
    padding: 24px 0 !important;
    position: static !important; /* Back to normal flow */
    margin-bottom: 20px !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.progress-wrap {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 12px !important;
}

.progress-text {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-main) !important;
    text-align: center !important;
}

.progress-level-2 {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    background: var(--primary) !important;
    padding: 2px 10px !important;
    border-radius: 8px !important;
}

.progress-bar-wrap {
    height: 10px !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    width: 90% !important;
    overflow: hidden !important;
}

/* Aggressive Spinner Removal */
input::-webkit-outer-spin-button,
input::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}

input[type=number] {
    -moz-appearance: textfield !important;
}

/* Ensure components don't overlap vertically */
.glass-card > * {
    margin-bottom: 15px !important;
}

.glass-card > *:last-child {
    margin-bottom: 0 !important;
}
"""

# ==========================================
# 🛠️ BACKEND LOGIC (Remains Same)
# ==========================================

class PersistenceManager:
    def __init__(self):
        self.api = HfApi(token=HF_TOKEN)
        self.local_file = "config.json"
        
    def load_prefs(self):
        default_prefs = {"topics": ["Artificial Intelligence", "Space Exploration", "Quantum Computing"], "country": "IN"}
        if not CONFIG_REPO_ID: return default_prefs
        try:
            path = hf_hub_download(repo_id=CONFIG_REPO_ID, filename=self.local_file, repo_type="dataset")
            with open(path, 'r') as f: return json.load(f)
        except Exception: return default_prefs

    def save_prefs(self, topics, country):
        if not CONFIG_REPO_ID: return "Persistence not configured."
        data = {"topics": topics, "country": country, "last_updated": str(datetime.now())}
        with open(self.local_file, 'w') as f: json.dump(data, f, indent=2)
        try:
            self.api.upload_file(path_or_fileobj=self.local_file, path_in_repo=self.local_file, repo_id=CONFIG_REPO_ID, repo_type="dataset")
            return "✅ Saved to Cloud."
        except Exception as e: return f"❌ Save failed: {str(e)}"

class NewsFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_google_rss(self, query, country="IN"):
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-{country}&gl={country}&ceid={country}:en"
        feed = feedparser.parse(url)
        return [{"title": e.title, "source": e.source.title if 'source' in e else "Google News", "published": parser.parse(e.published)} for e in feed.entries[:8]]

    def deduplicate(self, items):
        seen = set()
        unique = []
        for itm in items:
            slug = "".join(e for e in itm['title'].lower() if e.isalnum())[:40]
            if slug not in seen:
                unique.append(itm)
                seen.add(slug)
        unique.sort(key=lambda x: x['published'], reverse=True)
        return unique

class GeminiAgent:
    def summarize(self, all_topics_data):
        if not all_topics_data:
            yield "No data gathered."
            return

        context_str = ""
        for topic, items in all_topics_data.items():
            context_str += f"\n### TOPIC: {topic}\n"
            context_str += "\n".join([f"- {itm['title']} ({itm['source']})" for itm in items])

        prompt = f"""
        Analyze these news datasets and provide a consolidated intelligence report.
        For EACH topic, provide:
        1. A 2-sentence 'Impact Summary'.
        2. A small table: | Key Event | Sentiment | Potential Outlook |
        
        Final section: 'Cross-Topic Synthesis' - How these areas might intersect.
        Tone: Crisp, visual, minimal wall-of-text. Use bolding and emojis.
        
        Data:
        {context_str}
        """

        try:
            model = genai.GenerativeModel(REQUESTED_MODEL)
            response = model.generate_content(prompt, stream=True)
            accumulated = ""
            for chunk in response:
                if chunk.text:
                    accumulated += chunk.text
                    yield accumulated
        except Exception as e:
            yield f"❌ Error: {str(e)}"

# ==========================================
# 🚀 CORE APP ORCHESTRATOR
# ==========================================

fetcher = NewsFetcher()
agent = GeminiAgent()
prefs_mgr = PersistenceManager()

def process_all(t1, t2, t3, country, progress=gr.Progress()):
    logs = []
    def log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        return "\n".join(logs)

    valid_topics = [t for t in [t1, t2, t3] if t and t.strip()]
    if not valid_topics:
        yield "⚠️ Please enter at least one topic.", log("Aborted: No topics entered."), pd.DataFrame(), gr.update(selected=0)
        return

    yield "📡 Initializing scan...", log(f"Starting analysis for {len(valid_topics)} topics..."), pd.DataFrame(), gr.update(selected=1)
    
    all_data = {}
    master_table = []

    for idx, topic in enumerate(valid_topics):
        progress((idx/len(valid_topics)) * 0.5, desc=f"Scanning {topic}...")
        yield gr.update(), log(f"🔍 Fetching RSS for: {topic}"), pd.DataFrame(), gr.update()
        
        raw = fetcher.fetch_google_rss(topic, country)
        clean = fetcher.deduplicate(raw)
        all_data[topic] = clean
        
        for itm in clean:
            master_table.append({"Topic": topic, "Source": itm['source'], "Title": itm['title']})

    df = pd.DataFrame(master_table)
    
    progress(0.6, desc="Synthesizing with AI...")
    summary_stream = agent.summarize(all_data)
    
    for summary in summary_stream:
        yield summary, log(f"🧠 AI is thinking..."), df, gr.update()
    
    yield gr.update(), log("✅ Success. Intelligence briefing ready."), df, gr.update()

def start_up():
    stored = prefs_mgr.load_prefs()
    ts = stored.get("topics", ["", "", ""])
    while len(ts) < 3: ts.append("")
    return ts[0], ts[1], ts[2], stored.get("country", "IN")

# ==========================================
# 🖥️ GRADIO INTERFACE
# ==========================================

with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS, title="Neura Pro") as demo:
    
    with gr.Column(elem_classes="hero-header"):
        gr.HTML("""
            <h1 class='hero-title'>NEURA PRO</h1>
            <p class='hero-subtitle'>AI News Intelligence</p>
        """)

    with gr.Tabs() as main_tabs:
        
        with gr.TabItem("⚙️ CONFIG", id=0) as tab_config:
            with gr.Column():
                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown("### 🛠️ Parameters")
                    t1 = gr.Textbox(label="Primary Topic", placeholder="e.g. AI Trends", elem_classes="pref-input", lines=2)
                    t2 = gr.Textbox(label="Secondary Topic", placeholder="e.g. Market Shifts", elem_classes="pref-input", lines=2)
                    t3 = gr.Textbox(label="Tertiary Topic", placeholder="e.g. Policy Changes", elem_classes="pref-input", lines=2)
                    country = gr.Dropdown(choices=["US", "IN", "UK", "CA", "DE"], label="Base Region", value="IN")
                    
                    analyze_btn = gr.Button("🚀 START SCAN", elem_classes="primary-btn")
                    save_btn = gr.Button("💾 Save Configuration", variant="secondary")

                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown("### 📡 Terminal")
                    log_box = gr.Textbox(show_label=False, lines=8, elem_classes="log-terminal", value="System Ready")

        with gr.TabItem("📊 FEED", id=1) as tab_results:
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("## 🧠 Briefing")
                ai_out = gr.Markdown("Ready to sync...", elem_classes="prose")
                
                with gr.Accordion("📂 Raw Data", open=False):
                    raw_df = gr.Dataframe(headers=["Topic", "Source", "Title"], interactive=False)

    # Persistence
    demo.load(start_up, outputs=[t1, t2, t3, country])
    
    save_btn.click(lambda a,b,c,d: gr.Info(prefs_mgr.save_prefs([a,b,c], d)), inputs=[t1, t2, t3, country])

    # Core Action
    analyze_btn.click(
        fn=process_all,
        inputs=[t1, t2, t3, country],
        outputs=[ai_out, log_box, raw_df, main_tabs]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Base(), css=CUSTOM_CSS)
