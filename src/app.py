from __future__ import annotations
import os, tempfile, time, traceback, runpy, json, csv, re, statistics, unicodedata, requests, pathlib, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit_ace import st_ace
import numpy as np
import faiss  # pip install faiss-cpu
from pypdf import PdfReader       # pip install pypdf
import html2text                  # pip install html2text
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# ====== Feedback persistence (JSONL) ======
import uuid, json
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "feedback_logs")
FEEDBACK_JSONL = os.path.join(LOG_DIR, "feedback.jsonl")

def _ensure_log_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass

def _append_jsonl(record: dict, path: str = FEEDBACK_JSONL):
    _ensure_log_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _build_feedback_record(choice_label: str, comment: str, alt_code: str | None = None) -> dict:
    code_before = (st.session_state.get("code_input_generate") or "").strip()
    gen3 = st.session_state.get("gen3", {}) or {}
    after_hetero = (gen3.get("hetero", {}) or {}).get("code", "") or ""
    after_simple = (gen3.get("simple", {}) or {}).get("code", "") or ""
    after_ast    = (gen3.get("ast", {}) or {}).get("code", "") or ""
    after_llm    = (gen3.get("llm", {}) or {}).get("code", "") or ""

    choice_key = None
    label2key = {
        "Hétérogène": "hetero",
        "Simple": "simple",
        "Simple (KB)": "simple",
        "AST": "ast",
        "LLM seul": "llm",
    }
    if choice_label in label2key:
        choice_key = label2key[choice_label]

    after_selected = ""
    if choice_key == "hetero":
        after_selected = after_hetero
    elif choice_key == "simple":
        after_selected = after_simple
    elif choice_key == "ast":
        after_selected = after_ast
    elif choice_key == "llm":
        after_selected = after_llm

    rec = {
        "id": str(uuid.uuid4()),
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "page": "rag",
        "choice_label": choice_label,
        "choice_key": choice_key,
        "comment": comment or "",
        "code_before": code_before,
        "code_after_hetero": after_hetero,
        "code_after_simple": after_simple,
        "code_after_ast": after_ast,
        "code_after_llm": after_llm,
        "code_after_selected": after_selected,
    }
    if choice_label == "Aucun" and (alt_code or "").strip():
        rec["alt_code"] = (alt_code or "").strip()
    return rec


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
SHOW_DEBUG = False

DEFAULT_RAG_URLS = [
    "https://pythonspeed.com/articles/different-ways-speed/",
    "https://www.docstring.fr/formations/faq/optimisation/comment-optimiser-les-performances-de-mon-code-python/",
    "https://realpython.com/python-concurrency/",
    "https://pandas.pydata.org/docs/user_guide/enhancingperf.html",
    "https://numpy.org/doc/stable/user/absolute_beginners.html",
]

LOCAL_RULES_DIR = r"C:/Users/ghita.chahdi/Téléchargements/regles bonne pratiques"
LOCAL_RAG_DOCS = [
    os.path.join(LOCAL_RULES_DIR, "sonar_python_rules"),
    os.path.join(LOCAL_RULES_DIR, "sonar_python_rules_text"),
    os.path.join(LOCAL_RULES_DIR, "CAST Rules python.pdf"),
]

# ────────────────────────────── Thème & Styles ──────────────────────────────
st.set_page_config(page_title="Green Code Optimizer", page_icon="🌱", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{
  --bg:#0e0e0e; --bg-2:#171616; --fg:#ffffff;
  --green:#205f2a; --green-2:#1b5123; --chip:#102114;
  --card:#121212; --card-b:#1e1e1e; --muted:#a6b0a4;
  --ok:#16a34a; --warn:#d97706; --bad:#ef4444;
}
[data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--fg); }
[data-testid="stHeader"]{ background:var(--bg-2); }
[data-testid="stHeader"] *{ color:#e9efe7 !important; }
section[data-testid="stSidebar"]{ background:var(--bg-2); border-right:1px solid #111; }
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{
  min-height:100vh; display:flex; flex-direction:column; gap:12px; padding:12px 14px !important;
}
h1,h2,h3,h4{ color:var(--fg); letter-spacing:.2px; }
div[data-testid="stWidgetLabel"] > label p,
.stTextArea label p, .stSelectbox label p, label{ color:#fff !important; }
.badge{ display:inline-block; padding:.2rem .6rem; border-radius:999px;
  border:1px solid #2a3b2a; background:var(--chip); color:#bfe8c1; font-size:.78rem; }
.field-label{ color:#fff !important; font-size:.88rem; font-weight:400; letter-spacing:.2px; margin:2px 0 6px; opacity:1 !important; }
.section-card{ background:var(--card); border:1px solid var(--card-b); border-radius:16px;
  padding:18px 20px; box-shadow:0 8px 24px rgba(0,0,0,.28); }
.stButton{ display:flex; justify-content:center; }
.stButton > button{ min-width:150px !important; background:var(--green) !important; color:#fff !important;
  border:none; border-radius:14px; padding:.45rem 1.2rem; box-shadow:0 6px 18px rgba(0,0,0,.35);
  font-weight:700; letter-spacing:.2px; }
.stButton > button:hover{ background:var(--green-2) !important; }
.hero{ max-width:860px; margin:24px auto 8px; text-align:center;
  background:linear-gradient(180deg,#151515 0%,#0f0f0f 100%); border:1px solid rgba(255,255,255,.08);
  border-radius:18px; padding:28px 26px; box-shadow:0 12px 28px rgba(0,0,0,.35); }
.hero h1{ font-size:2rem; margin:0; }
.history-empty{ color:var(--muted); text-align:center; padding:6px 0 2px; font-size:.95rem; }
.history-card{ position:relative; border-radius:14px; padding:10px 12px;
  background:linear-gradient(180deg,#151515 0%,#0f0f0f 100%); border:1px solid rgba(255,255,255,.06);
  box-shadow:0 8px 22px rgba(0,0,0,.25); transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
.history-card:hover{ transform:translateY(-1px); box-shadow:0 12px 26px rgba(0,0,0,.35); border-color:rgba(255,255,255,.14); }
.hdr{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:6px; }
.chip{ display:inline-flex; align-items:center; gap:6px; padding:.18rem .55rem; border-radius:999px;
  font-size:.78rem; font-weight:700; border:1px solid rgba(255,255,255,.12); }
.tool-cc{ background:#0f2a18; color:#b7f4c4; border-color:#1e5e36; }
.badge-co2{ padding:.22rem .55rem; border-radius:999px; font-weight:800; font-size:.80rem; background:#1a1a1a; border:1px solid rgba(255,255,255,.12); }
.lv-ok{ color:#86efac; border-color:#234f2b; background:#102115; }
.lv-warn{ color:#fbbf24; border-color:#4f3b1a; background:#211a10; }
.lv-bad{ color:#fca5a5; border-color:#5a1e1e; background:#210f10; }
.code-preview{ color:#d7dbd5; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  font-size:.82rem; line-height:1.35; opacity:.95; margin:2px 0 8px; display:-webkit-box; -webkit-line-clamp:2;
  -webkit-box-orient:vertical; overflow:hidden; }
.meta{ display:flex; gap:10px; flex-wrap:wrap; color:#aeb4ac; font-size:.8rem; }
.meta .pill{ background:#0f0f0f; border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:.2rem .45rem; color:#cfd7cc; }
.result-wrap{ margin:8px 0 24px; }
.result-card{ background:var(--card); border:1px solid var(--card-b); border-radius:16px;
  padding:18px 20px; box-shadow:0 8px 24px rgba(0,0,0,.28); }
.kpi-grid{ display:grid; gap:12px; grid-template-columns:repeat(3, minmax(0,1fr)); grid-auto-rows:1fr; }
.kpi{ background:linear-gradient(180deg,#151515 0%,#0f0f0f 100%); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:16px;
  display:grid; grid-template-rows:28px 36px; row-gap:8px; justify-items:center; align-items:center; text-align:center; }
.kpi h4{ margin:0; font-weight:700; font-size:1.1rem; color:#e9efe7; white-space:nowrap; line-height:1; font-variant-numeric:tabular-nums; }
.kpi .val{ color:#fff; font-size:1.1rem; }
.energy-grid{ display:grid; gap:12px; margin:16px 0 36px; grid-template-columns:repeat(auto-fit, minmax(170px,1fr)); }
.energy-card{ background:linear-gradient(180deg,#151515 0%,#0f0f0f 100%); border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px; text-align:center; }
.energy-card .title{ color:#e9efe7; font-size:.95rem; margin-bottom:8px; font-weight:700; }
.energy-card .val{ color:#fff; font-weight:400; font-size:1.1rem; line-height:1; }
.eq-inline{ white-space:nowrap; }
div[data-testid="stComponent"]{ border:1px solid rgba(255,255,255,.14) !important; border-radius:12px !important; background:transparent !important; box-shadow:none !important; padding:0 !important; overflow:hidden !important; }
div[data-testid="stComponent"]:hover{ border-color:rgba(255,255,255,.24) !important; }
div[data-testid="stComponent"]:focus-within{ border-color:#2e7d32 !important; box-shadow:0 0 0 2px rgba(46,125,50,.18) inset !important; }
iframe[title^="streamlit_ace.st_ace"]{ border:0 !important; border-radius:12px !important; background:transparent !important; }
details[data-testid="stExpander"]{ border:1px solid rgba(255,255,255,.14); border-radius:12px; background:transparent !important; overflow:hidden; }
details[data-testid="stExpander"] > summary{ background:#141414 !important; color:#e9efe7 !important; border-radius:12px; padding:.60rem .85rem; list-style:none; cursor:pointer; }
details[data-testid="stExpander"][open] > summary{ background:#161616 !important; color:#e9efe7 !important; }
.feedback-fab { position:fixed; right:18px; bottom:18px; z-index:9999; }
.feedback-card{ width:320px; background:var(--card); border:1px solid var(--card-b); border-radius:14px; box-shadow:0 10px 28px rgba(0,0,0,.45); padding:12px 12px 10px; }
.feedback-card h4{ margin:0 0 8px 0; font-size:1rem; color:#e9efe7; }
.feedback-small{ color:#cfd7cc; font-size:.8rem; opacity:.9; }

/* Option cards */
.opt-wrap { display:flex; flex-direction:column; gap:10px; }
.opt-card {
  display:flex; align-items:center; gap:12px;
  padding:10px 12px; border-radius:12px;
  background:#171717; border:1px solid rgba(255,255,255,.10);
  box-shadow:0 4px 14px rgba(0,0,0,.25);
}
.opt-card:hover { border-color:rgba(255,255,255,.18); }
.opt-title { color:#fff; font-weight:700; line-height:1.1; }
.opt-desc  { color:#cfd7cc; font-size:.86rem; opacity:.9; margin-top:2px; }
.opt-disabled { opacity:.55; }
.opt-badge {
  display:inline-block; padding:.08rem .45rem; border-radius:999px;
  border:1px solid rgba(255,255,255,.14); font-size:.72rem; color:#cfead2;
  background:linear-gradient(180deg,#0f2a18 0%, #0f1f15 100%);
  margin-left:.45rem;
}

/* Sidebar header */
.sidebar-head{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:10px; margin:14px 8px 16px;
}
.sidebar-head .sprout{ font-size:64px; line-height:1; filter: drop-shadow(0 2px 6px rgba(0,0,0,.35)); }
.sidebar-head .title{ font-weight:800; font-size:1.25rem; color:#e9efe7; letter-spacing:.2px; }
.sidebar-head .divider{
  width:100%; height:1px; margin-top:2px;
  background:linear-gradient(90deg, rgba(255,255,255,.06), rgba(255,255,255,.14), rgba(255,255,255,.06));
  border-radius:999px;
}

/* Home button fixe (noir/blanc) */
#home-anchor { position:fixed; top:12px; left:24px; z-index:1000; }
#home-anchor ~ div .stButton > button,
#home-anchor ~ div button {
  position:fixed; top:12px; left:24px; z-index:1001;
  min-width:32px !important; width:32px !important; height:32px !important; padding:0 !important;
  border-radius:8px !important; background:#000 !important; color:#fff !important;
  border:1px solid rgba(255,255,255,.25) !important; box-shadow:0 4px 12px rgba(0,0,0,.45) !important;
}
#home-anchor ~ div .stButton > button::before,
#home-anchor ~ div button::before{
  content:"←"; display:inline-block; font-size:14px; line-height:1; transform: translateY(1px);
}
#home-anchor ~ div .stButton > button > div,
#home-anchor ~ div button > div { display:none !important; }
#home-anchor ~ div .stButton > button:hover,
#home-anchor ~ div .stButton > button:focus,
#home-anchor ~ div .stButton > button:active,
#home-anchor ~ div button:hover,
#home-anchor ~ div button:focus,
#home-anchor ~ div button:active {
  background:#000 !important; color:#fff !important;
  border:1px solid rgba(255,255,255,.25) !important; box-shadow:0 4px 12px rgba(0,0,0,.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────── Session ─────────────────────────────
ss = st.session_state
ss.setdefault("gen3", {})
ss.setdefault("history", [])
ss.setdefault("tool_select", "CodeCarbon")
ss.setdefault("code_input_analyse", "")
ss.setdefault("code_input_generate", "")
ss.setdefault("generated_code", "")
ss.setdefault("rag_sources", [])
ss.setdefault("feedback", [])
ss.setdefault("page", "home")

# ordre demandé : AST > Simple > Hétérogène
ss.setdefault("rag_selected", {"ast": True, "simple": True, "hetero": True})

# KB (unique) pour Simple & AST
ss.setdefault("kb_df_loaded_simple", False)
ss.setdefault("kb_faiss_ready_simple", False)
ss.setdefault("kb_df_loaded_ast", False)
ss.setdefault("kb_faiss_ready_ast", False)

# caches techniques
ss.setdefault("_global_emb", None)     # NVIDIAEmbeddings singleton
ss.setdefault("_kb_faiss_obj", None)   # retriever faiss pour KB
ss.setdefault("_het_vs", None)         # index FAISS hétérogène (persist session)

# ───────────────────────────── Utils formatting ─────────────────────────────
def _fmt_num(x: Optional[float], f=None) -> str:
    if x is None: return "—"
    try: return f(x) if f else str(x)
    except Exception: return str(x)

def _co2_fmt_kg(kg: Optional[float]) -> str:
    if kg is None: return "—"
    if 0 < kg < 1e-5: return f"{kg:.2e} kgCO₂"
    return f"{kg:.5f} kgCO₂"

def _fmt_s(v: Optional[float]) -> str:
    return _fmt_num(v, lambda x: f"{x:.2f} s")

def _fmt_wh(v: Optional[float]) -> str:
    return _fmt_num(v, lambda x: f"{x*1000:.3f} Wh")

def _fmt_g(v: Optional[float]) -> str:
    return _fmt_num(v, lambda x: f"{x:.3f} g")

def _fmt_si(n: float, unit: str, steps=(1, 1e3, 1e6, 1e9), labels=("","k","M","G")) -> str:
    for step, lab in zip(reversed(steps), reversed(labels)):
        if abs(n) >= step:
            out = f"{n/step:.3f} {lab}{unit}"
            return out.replace(".000","")
    return f"{n:.3f} {unit}".replace(".000","")

def _fmt_joules_from_kwh(kwh: Optional[float]) -> str:
    if kwh is None: return "—"
    j = kwh * 3_600_000.0
    return _fmt_si(j, "J")

def _co2_level(kg: Optional[float]) -> str:
    if kg is None: return "lv-ok"
    if kg >= 1e-2: return "lv-bad"
    if kg >= 1e-3: return "lv-warn"
    return "lv-ok"

def _tool_chip_cls(tool: str) -> str:
    t = tool.lower()
    if "codecarbon" in t: return "tool-cc"
    return ""

def _short_code_preview(code: str, n=160) -> str:
    code1 = (code or "").strip().replace("\n", " ")
    return (code1[:n] + "…") if len(code1) > n else code1

# ───────────────────── Helpers généraux ─────────────────────
def detect_language(code: str) -> str:
    return "python"

def preflight_compile(code: str) -> Tuple[bool, Optional[str]]:
    try:
        compile(code, "<snippet>", "exec")
        return True, None
    except Exception:
        return False, traceback.format_exc()

def _write_snippet(code: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="code_")) / "snippet.py"
    tmp.write_text(code, encoding="utf-8")
    return tmp

# ───────────────────── Backends de mesure ─────────────────────
def measure_with_codecarbon(code: str) -> Dict[str, Any]:
    try:
        from codecarbon import EmissionsTracker
    except Exception as e:
        return {"error":"codecarbon_missing","notes":"Installe : pip install codecarbon psutil","stderr":str(e)}
    out_dir = Path(tempfile.mkdtemp(prefix="cc_run_")); csv_path = out_dir / "emissions.csv"
    os.environ.setdefault("CODECARBON_LOG_LEVEL","error")
    tracker = EmissionsTracker(output_dir=str(out_dir), output_file="emissions.csv",
                               measure_power_secs=1, save_to_file=True, log_level="error")
    run_err, err_text, emissions_kg = False, "", None; tmp = _write_snippet(code)
    try:
        tracker.start()
        try: runpy.run_path(str(tmp), run_name="__main__")
        except SystemExit: pass
        except Exception:
            run_err, err_text = True, traceback.format_exc()
        finally: emissions_kg = tracker.stop()
    finally:
        time.sleep(0.1)
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
    res = {"duration_s": None, "energy_kwh": None, "cpu_energy_kwh": None,
           "gpu_energy_kwh": None, "ram_energy_kwh": None,
           "emissions_kg": float(emissions_kg) if emissions_kg is not None else None}
    try:
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f: rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                def ffloat(x):
                    try: return float(x) if x not in (None,"","None") else None
                    except: return None
                res["duration_s"]     = ffloat(last.get("duration"))
                res["energy_kwh"]     = ffloat(last.get("energy_consumed"))
                res["cpu_energy_kwh"] = ffloat(last.get("cpu_energy"))
                res["gpu_energy_kwh"] = ffloat(last.get("gpu_energy"))
                res["ram_energy_kwh"] = ffloat(last.get("ram_energy"))
    except Exception: pass
    if run_err: res["run_error"]=True; res["stderr"]=err_text.strip()
    return res

# ───────────────────────────── UI Résultats ─────────────────────────────
def render_result(res: Dict[str, Any]) -> None:
    kg = res.get("emissions_kg"); co2g = res.get("co2eq_g")
    duration = res.get("duration_s"); energy_kwh = res.get("energy_kwh")
    cpu = res.get("cpu_energy_kwh"); gpu = res.get("gpu_energy_kwh"); ram = res.get("ram_energy_kwh")

    co2_g_txt    = _fmt_g(co2g) if isinstance(co2g,(int,float)) else (_fmt_g(kg*1000.0) if isinstance(kg,(int,float)) else "—")
    duration_txt = _fmt_s(duration)
    energy_wh_txt = _fmt_wh(energy_kwh)
    energy_j_txt  = _fmt_joules_from_kwh(energy_kwh)

    st.markdown(f"""
<div class="section-card">
  <div class="kpi-grid">
    <div class="kpi"><h4>Durée</h4><div class="val">{duration_txt}</div></div>
    <div class="kpi"><h4>Énergie</h4><div class="val">{energy_wh_txt} &nbsp;=&nbsp; {energy_j_txt}</div></div>
    <div class="kpi"><h4>CO₂eq</h4><div class="val">{co2_g_txt}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    cards = []
    for label, kwh in [("CPU", cpu), ("GPU", gpu), ("RAM", ram)]:
        wh = _fmt_wh(kwh) if isinstance(kwh, (int, float)) else "—"
        jj = _fmt_joules_from_kwh(kwh) if isinstance(kwh, (int, float)) else "—"
        cards.append(
            f'''<div class="energy-card">
                <div class="title">{label}</div>
                <div class="val eq-inline">{wh}&nbsp;=&nbsp;{jj}</div>
                </div>'''
        )
    st.markdown(f'<div class="energy-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

def render_history_sidebar():
    st.markdown(
        """
        <div class="sidebar-head">
          <div class="sprout">🌱</div>
          <div class="title">Historique</div>
          <div class="divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hist = st.session_state.get("history", [])
    if not hist:
        st.markdown('<div class="history-empty">Aucun Run pour le moment.</div>', unsafe_allow_html=True)
        return
    for h in reversed(hist):
        tool_name = h.get("tool", "?")
        res = h.get("res", {}) or {}
        kg = res.get("emissions_kg")
        co2_txt = _co2_fmt_kg(kg) if isinstance(kg, (int, float)) else "—"
        level = _co2_level(kg if isinstance(kg, (int, float)) else None)
        ts = h.get("timestamp", "")
        dur = res.get("duration_s")
        en  = res.get("energy_kwh")
        dur_txt   = _fmt_s(dur)
        en_txt_wh = _fmt_wh(en)
        en_txt_j  = _fmt_joules_from_kwh(en)
        code_prev = _short_code_preview(h.get("code", ""))
        st.markdown(
            f"""
            <div class="history-card">
              <div class="hdr">
                <span class="chip {_tool_chip_cls(tool_name)}">{tool_name}</span>
                <span class="badge-co2 {level}">{co2_txt}</span>
              </div>
              <div class="code-preview">{code_prev}</div>
              <div class="meta">
                <span class="pill">Durée&nbsp;: {dur_txt}</span>
                <span class="pill">Énergie&nbsp;: {en_txt_wh} · {en_txt_j}</span>
                <span class="pill">{ts}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ───────── Bouton Home fixé en haut ─────────
def render_top_home_button():
    st.markdown('<div id="home-anchor"></div>', unsafe_allow_html=True)
    clicked = st.button("← Accueil", key="__top_home_btn__")
    if clicked:
        st.session_state.page = "home"
        _set_query_page("home")
        st.rerun()

# ───────────────────────────── Mapping backends ─────────────────────────────
MEASURERS = {"CodeCarbon": measure_with_codecarbon}

# ───────────────────────────── Query params ─────────────────────────────
def _get_query_page():
    try:
        return st.query_params.get("page", None)
    except Exception:
        q = st.experimental_get_query_params()
        return (q.get("page", [None]) or [None])[0]

def _set_query_page(page: str):
    try:
        st.query_params.update({"page": page})
    except Exception:
        st.experimental_set_query_params(page=page)

qp = _get_query_page()
if qp in ("home", "eval", "rag"):
    ss.page = qp

# ───────────────────────────── NVIDIA Key ─────────────────────────────
def require_nvidia_key() -> bool:
    try:
        if "NVIDIA_API_KEY" in getattr(st, "secrets", {}):
            os.environ["NVIDIA_API_KEY"] = str(st.secrets["NVIDIA_API_KEY"]).strip()
    except Exception:
        pass
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        return True
    k = st.text_input("NVIDIA_API_KEY", type="password", key="nv_key_input")
    if k:
        os.environ["NVIDIA_API_KEY"] = k.strip()
        st.success("Clé NVIDIA enregistrée.")
        return True
    st.info("Définis NVIDIA_API_KEY dans un fichier .env (ou st.secrets) avant de lancer la génération.")
    return False

# ----- Token helpers (512 max “safe”) -----
try:
    import tiktoken
    _TOK_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOK_ENC = None

def _tok_count(text: str) -> int:
    if _TOK_ENC: return len(_TOK_ENC.encode(text))
    return max(1, int(len(text) / 4))

def _tok_truncate(text: str, max_tokens: int = 480) -> str:
    if _TOK_ENC:
        toks = _TOK_ENC.encode(text)
        if len(toks) <= max_tokens: return text
        return _TOK_ENC.decode(toks[:max_tokens])
    return text[: max_tokens * 4]

def _tok_split(text: str, chunk_tokens: int = 480) -> list[str]:
    if _tok_count(text) <= chunk_tokens:
        return [text]
    if _TOK_ENC:
        toks = _TOK_ENC.encode(text)
        return [_TOK_ENC.decode(toks[i:i+chunk_tokens]) for i in range(0, len(toks), chunk_tokens)]
    approx = chunk_tokens * 4
    return [text[i:i+approx] for i in range(0, len(text), approx)]

# ───────────────────────────── Embeddings global (cache) ─────────────────────────────
_HET_EMB_CACHE: dict[str, list[float]] = {}

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _get_global_emb():
    if st.session_state.get("_global_emb") is None:
        st.session_state["_global_emb"] = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            base_url="https://integrate.api.nvidia.com/v1",
            truncate="END",
        )
    return st.session_state["_global_emb"]

def _embed_text_token_safe(emb, text: str) -> list[float]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return [0.0] * 1024
    if _tok_count(text) <= 480:
        small = _tok_truncate(text, 480)
        k = _sha1(small)
        if k in _HET_EMB_CACHE: return _HET_EMB_CACHE[k]
        v = emb.embed_documents([small])[0]
        _HET_EMB_CACHE[k] = v
        return v
    segs = _tok_split(text, 480)
    vecs = []
    for s in segs:
        k = _sha1(s)
        if k in _HET_EMB_CACHE:
            vecs.append(_HET_EMB_CACHE[k])
        else:
            v = emb.embed_documents([s])[0]
            _HET_EMB_CACHE[k] = v
            vecs.append(v)
    arr = np.array(vecs, dtype="float32")
    mean = arr.mean(axis=0)
    mean = mean / (float((np.linalg.norm(mean) + 1e-12)))
    return mean.tolist()

# ───────────────────────────── RAG Hétérogène ─────────────────────────────
def _hetero_add_web_sources(urls: List[str], dest_dir: str) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    headers = {"User-Agent": "GreenRAG/1.0 (+research use)"}
    for u in urls or []:
        try:
            r = requests.get(u, headers=headers, timeout=25)
            r.raise_for_status()
            txt_raw = r.text or ""
            txt = html2text.html2text(txt_raw)
            # un minimum de signal pour éviter les pages quasi vides
            txt = re.sub(r"\s+", " ", txt).strip()
            if len(txt) < 400:
                continue
            tag = f"[URL:{u}] "
            content = tag + txt
            name = re.sub(r"[^A-Za-z0-9._-]+", "-", re.sub(r"^https?://", "", u)).strip("-")[:120] or "url"
            fpath = os.path.join(dest_dir, f"url_{name}.txt")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            saved.append(fpath)
        except Exception:
            if SHOW_DEBUG:
                st.write(f"⚠️ URL échouée: {u}")
    return saved


def _hetero_ingest_files(dirpath: str) -> List[str]:
    out=[]
    for name in os.listdir(dirpath):
        p=os.path.join(dirpath,name)
        if not os.path.isfile(p): continue
        ext = Path(p).suffix.lower()
        try:
            if ext==".pdf":
                r=PdfReader(p); parts=[]
                # on lit toutes les pages mais on normalise le texte
                for i,pg in enumerate(r.pages):
                    t=pg.extract_text() or ""
                    t=re.sub(r"\s+"," ", unicodedata.normalize("NFKC", t)).strip()
                    if t: parts.append(f"[PDF:{name}|p{i+1}] {t}")
                txt="\n\n".join(parts)
            elif ext in {".html",".htm"}:
                html=open(p,"r",encoding="utf-8",errors="ignore").read()
                txt=f"[HTML:{name}] " + re.sub(r"\s+"," ", html2text.html2text(html)).strip()
            else:
                txt=f"[{ext.upper()[1:]}:{name}] " + re.sub(r"\s+"," ", open(p,"r",encoding="utf-8",errors="ignore").read()).strip()
            if txt.strip(): out.append(txt)
        except Exception as e:
            if SHOW_DEBUG: st.write("⚠️ lecture:", p, e)
    return out

def _hetero_build_vs(chunks: List[str]):
    emb = _get_global_emb()
    cleaned = []
    for t in chunks:
        t = re.sub(r"\s+", " ", (t or "")).strip()
        if _tok_count(t) > 1200:
            t = _tok_truncate(t, 1200)
        if t:
            cleaned.append(t)

    # ❌ Plus de “règles par défaut” si aucune source
    if not cleaned:
        return {"index": None, "chunks": [], "emb": emb}

    vecs = [_embed_text_token_safe(emb, t) for t in cleaned]
    X = np.array(vecs, dtype="float32")
    faiss.normalize_L2(X)
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    return {"index": idx, "chunks": cleaned, "emb": emb}


def _hetero_search(vs, q: str, k: int = 6):
    if not vs or not vs.get("index") or not vs.get("chunks"):
        return []
    q = re.sub(r"\s+", " ", q or "").strip()
    q = _tok_truncate(q, 480)
    qv = _embed_text_token_safe(vs["emb"], q)
    Q = np.array([qv], dtype="float32")
    faiss.normalize_L2(Q)
    D, I = vs["index"].search(Q, k)
    out = []
    for rank, ix in enumerate(I[0].tolist()):
        if ix < 0:
            continue
        out.append((vs["chunks"][ix], D[0][rank]))
    return out


def _get_hetero_vs_cached():
    """
    Construit l'index hétérogène à partir des **URLs** et des **fichiers locaux**.
    Aucun fallback interne : si rien n'est ingéré, on le signale et la génération renverra vide.
    """
    if st.session_state.get("_het_vs") is not None:
        return st.session_state["_het_vs"]

    updir = os.path.join(tempfile.gettempdir(), "rag_builtin_sources")
    os.makedirs(updir, exist_ok=True)

    present_local_docs = []

    # 1) URLs (optionnel si réseau OK)
    try:
        _hetero_add_web_sources(DEFAULT_RAG_URLS, updir)
    except Exception as e:
        if SHOW_DEBUG:
            st.write("⚠️ Echec chargement URLs:", e)

    # 2) Fichiers locaux (optionnels)
    for src in LOCAL_RAG_DOCS:
        try:
            if os.path.exists(src):
                dest = os.path.join(updir, os.path.basename(src))
                with open(src, "rb") as f_in, open(dest, "wb") as f_out:
                    f_out.write(f_in.read())
                present_local_docs.append(src)
        except Exception as e:
            if SHOW_DEBUG:
                st.write(f"⚠️ Copie impossible {src}: {e}")

    # 3) Ingestion
    chunks = _hetero_ingest_files(updir)
    if SHOW_DEBUG:
        st.write(f"ℹ️ Hétérogène: {len(chunks)} chunks | Locaux: {len(present_local_docs)} | URLs: {len(DEFAULT_RAG_URLS)}")

    vs = _hetero_build_vs(chunks) if chunks else {"index": None, "chunks": [], "emb": _get_global_emb()}
    st.session_state["_het_vs"] = {"vs": vs, "present_local": present_local_docs}
    return st.session_state["_het_vs"]



def rag_hetero_generate(code_pas_green: str, temperature: float = 0.25, max_tokens: int = 1000):
    """
    RAG Hétérogène = LLM + docs locaux/URLs (pas de KB, pas d'AST).
    → Plus conservateur que Simple: n'applique pas de vectorisation/refonte
      à moins qu'un extrait la prescrive EXPLICITEMENT.
    """
    cache = _get_hetero_vs_cached()
    vs = cache["vs"]
    present_local_docs = cache["present_local"]

    # Pas d’index → pas de génération
    if not vs or not vs.get("index") or not vs.get("chunks"):
        return {
            "code": "",
            "changes": "- aucune génération : aucune source exploitable (docs locaux/URLs non ingérés).",
            "sources": "N/A"
        }

    # Recherche d'extraits pertinents dans les sources
    ctx = _hetero_search(vs, "python optimization performance memory io vectorization pandas numpy", k=6)
    if not ctx:
        return {
            "code": "",
            "changes": "- aucune génération : la recherche n'a rien retourné d'utile.",
            "sources": "N/A"
        }

    parts = [f"[score={sc:.3f}] {txt[:1500]}" for (txt, sc) in ctx]
    stuffed = "\n\n".join(parts)

    # Sources affichées
    src_lines = []
    if present_local_docs:
        src_lines.append("=== Fichiers locaux ===")
        src_lines.extend([os.path.basename(p) for p in present_local_docs])
    sources_txt = "\n".join(src_lines) if src_lines else "N/A"

    system = (
        "Tu optimises LÉGÈREMENT le code en t'appuyant UNIQUEMENT sur les extraits fournis "
        "(provenant de documents locaux et/ou URLs déjà ingérés). "
        "N'UTILISE ni AST, ni base de connaissances privée, ni règles implicites non citées. "
        "Ne fais pas de refonte structurelle sauf si un extrait le prescrit explicitement. "
        "Reste dans le même langage et renvoie un code exécutable."
        "NE REMPLACE PAS des boucles par de la vectorisation pandas/NumPy, ne réécris pas la logique, "
        "SAUF si un extrait le PRESCRIT explicitement et montre comment faire. "
        "Pas de 'bonnes pratiques' implicites. "
        "Reste dans le même langage et renvoie un code exécutable."

    )

    user = (
        f"=== CONTEXTE ===\n{stuffed}\n\n"
        "=== TÂCHE ===\nRéécris le code avec des ajustements mineurs justifiés par les extraits ci-dessus.\n\n"
        f"=== CODE_ORIGINAL ===\n{code_pas_green}\n\n"
        "RÈGLE DE SORTIE (OBLIGATOIRE) : renvoie un bloc [CODE] avec des backticks.\n"
        "FORMAT EXACT :\n"
        "[CODE]\n```python\n...code...\n```\n"
        "[SOURCES]\n- liste ou N/A\n"
        "[CHANGES]\n- puces concises (uniquement justifiées par les extraits)"
    )

    llm = ChatNVIDIA(
        model="mistralai/mistral-small-24b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=temperature,
        max_tokens=max_tokens
    )
    content = llm.invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ).content

    # ⤵️ Extraction tolérante
    code = _extract_code_block(content, tag="CODE")

    # (facultatif) mini-retry si le bloc n'a pas été rendu correctement
    # if not code:
    #     content2 = llm.invoke([
    #         {"role": "system", "content": "Rends UNIQUEMENT un bloc de code entre backticks."},
    #         {"role": "user", "content": f"Réécris ce code sans changer sa logique.\n\n```python\n{code_pas_green}\n```"}
    #     ]).content
    #     code = _extract_code_block(content2, tag="CODE")

    m_changes = re.search(r"\[CHANGES\]\s*([\s\S]*?)\Z", content)
    changes = _sanitize_changes((m_changes.group(1).strip() if m_changes else "").strip()) \
              or "- aucune modification majeure"

    return {"code": code, "changes": changes, "sources": sources_txt}




# ───────────────────────────── RAG Simple / AST (KB Excel) ─────────────────────────────
def build_faiss_from_kb(df: pd.DataFrame):
    emb = _get_global_emb()

    def row_to_text(r):
        parts = [
            f"ID: {r['id']}",
            f"Langage: {r['langage']}",
            f"Contexte: {r['contexte']}",
            f"Anti-pattern: {r['anti_pattern']}",
            f"Conseil: {r['advice']}",
            "Exemple (avant):\n"+str(r["exemple_avant"]),
            "Exemple (après):\n"+str(r["exemple_apres"]),
            f"Gains: CPU={r['gain_cpu']} | RAM={r['gain_ram']}",
            f"Outils de mesure: {r['outil_mesure']}",
            f"Estimation CO2: {r['estimation_co2']}",
        ]
        return "\n".join(parts)

    docs=[]
    for _, r in df.iterrows():
        meta = {k: r[k] for k in ("id","langage","contexte","gain_cpu","gain_ram","outil_mesure","estimation_co2")}
        docs.append(Document(page_content=row_to_text(r), metadata=meta))

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=340, chunk_overlap=48
    )
    chunks=[]
    for d in docs:
        for c in splitter.split_text(d.page_content):
            chunks.append(Document(page_content=c, metadata=d.metadata))

    vs = FAISS.from_documents(chunks, emb)
    retr = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 32, "lambda_mult": 0.5}
    )
    return retr

def resolve_kb_path_for_simple() -> str | None:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        os.environ.get("RAG_KB_PATH", "").strip() or None,
        script_dir / "knowledge_base.updated2.xlsx",
        script_dir / "knowledge_base_with_rules_fixed.xlsx",
        script_dir / "knowledge_base_with_rules.xlsx",
        script_dir / "knowledge_base.xlsx",
    ]
    for p in candidates:
        if not p:
            continue
        p = Path(p)
        if p.exists() and p.is_file():
            return str(p)
    return None

KB_PATH_SIMPLE = resolve_kb_path_for_simple()

def load_kb_simple_once():
    if st.session_state.get("kb_df_loaded_simple"):
        return
    if not KB_PATH_SIMPLE:
        st.warning("KB (Simple) introuvable. Place un fichier Excel à côté du script ou définis RAG_KB_PATH.")
        return
    try:
        import openpyxl
        df = pd.read_excel(KB_PATH_SIMPLE, engine="openpyxl", sheet_name=0)
    except Exception as e:
        st.error(f"Erreur de lecture de la KB Simple ({KB_PATH_SIMPLE}) : {e}")
        return
    st.session_state.kb_df_simple = df
    st.session_state.kb_df_loaded_simple = True
    st.session_state.kb_faiss_ready_simple = False

def load_kb_ast_once():
    if st.session_state.get("kb_df_loaded_ast"):
        return
    if not st.session_state.get("kb_df_loaded_simple"):
        load_kb_simple_once()
    if not st.session_state.get("kb_df_loaded_simple"):
        st.warning("KB (Simple) introuvable — nécessaire aussi pour l’AST.")
        return
    st.session_state.kb_df_ast = st.session_state.kb_df_simple
    st.session_state.kb_df_loaded_ast = True
    st.session_state.kb_faiss_ready_ast = False

# chargements
load_kb_simple_once()
load_kb_ast_once()

# build FAISS (cache retriever pour la session)
if st.session_state.get("kb_df_loaded_simple") and not st.session_state.get("kb_faiss_ready_simple"):
    with st.spinner("Chargement..."):
        st.session_state.retriever_simple = build_faiss_from_kb(st.session_state.kb_df_simple)
        st.session_state.kb_faiss_ready_simple = True

if st.session_state.get("kb_df_loaded_ast") and not st.session_state.get("kb_faiss_ready_ast"):
    with st.spinner("Chargement..."):
        st.session_state.retriever_ast = build_faiss_from_kb(st.session_state.kb_df_ast)
        st.session_state.kb_faiss_ready_ast = True

def _sanitize_changes(text: str) -> str:
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines()]
    keep = []
    for l in lines:
        low = l.lower()
        if not l:
            continue
        if low.startswith("[raisonnement") or low.startswith("[ast_hints"):
            continue
        if low in {"anti-pattern(s)", "anti-pattern", "optimisations", "optimisation", "règles", "regles"}:
            continue
        if "règle" in low or "regle" in low:
            continue
        if "anti-pattern" in low or "antipattern" in low or "anti pattern" in low:
            continue
        keep.append(l)
    formatted = []
    for l in keep:
        if l.startswith(("•", "-")):
            formatted.append(l)
        else:
            formatted.append(f"- {l}")
    return "\n".join(formatted)

def _extract_code_block(content: str, tag: str = "CODE") -> str:
    """
    Récupère le code même si le modèle ne respecte pas parfaitement le format.
    Ordre:
      1) [TAG] + bloc ``` ```
      2) premier bloc ``` ```
      3) tout ce qui suit [TAG]
    """
    if not content:
        return ""

    # 1) Format strict: [TAG] + ```...```
    m = re.search(rf"\[{tag}\]\s*```[^\n]*\n([\s\S]*?)```", content, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # 2) Premier bloc de backticks
    m2 = re.search(r"```[^\n]*\n([\s\S]*?)```", content)
    if m2 and m2.group(1).strip():
        return m2.group(1).strip()

    # 3) Tout ce qui suit [TAG]
    m3 = re.search(rf"\[{tag}\]\s*([\s\S]+)", content, re.IGNORECASE)
    if m3 and m3.group(1).strip():
        return m3.group(1).strip()

    return ""


# -------- AST helpers (pas de règles ajoutées ; uniquement signaux structurels) --------
def _ast_hints(code: str) -> List[str]:
    import ast
    hints=[]
    try:
        tree = ast.parse(code)

        loops = 0
        sleep_calls = 0
        apply_calls = 0
        imports = set()
        func_names = set()
        self_recursion = False

        class V(ast.NodeVisitor):
            def __init__(self):
                self.current_func = None

            def visit_FunctionDef(self, n):  # type: ignore
                nonlocal func_names
                func_names.add(n.name)
                prev = self.current_func
                self.current_func = n.name
                self.generic_visit(n)
                self.current_func = prev

            def visit_AsyncFunctionDef(self, n):  # type: ignore
                self.visit_FunctionDef(n)

            def visit_For(self, n):  # type: ignore
                nonlocal loops; loops += 1
                self.generic_visit(n)

            def visit_While(self, n):  # type: ignore
                nonlocal loops; loops += 1
                self.generic_visit(n)

            def visit_Import(self, n):  # type: ignore
                for a in n.names:
                    imports.add(a.name.split('.')[0])
                self.generic_visit(n)

            def visit_ImportFrom(self, n):  # type: ignore
                if n.module:
                    imports.add(n.module.split('.')[0])
                self.generic_visit(n)

            def visit_Call(self, n):  # type: ignore
                nonlocal sleep_calls, apply_calls, self_recursion
                # sleep()
                if isinstance(n.func, ast.Name) and n.func.id == "sleep":
                    sleep_calls += 1
                if isinstance(n.func, ast.Attribute):
                    if n.func.attr == "sleep":
                        sleep_calls += 1
                    if n.func.attr == "apply":
                        apply_calls += 1
                # self-recursion: foo() inside def foo(...)
                target = None
                if isinstance(n.func, ast.Name):
                    target = n.func.id
                elif isinstance(n.func, ast.Attribute):
                    target = n.func.attr
                if self.current_func and target == self.current_func:
                    self_recursion = True
                self.generic_visit(n)

        V().visit(tree)
        pandas = ("pandas" in imports)
        numpy = ("numpy" in imports)

        if loops > 0: hints.append(f"{loops} boucles")
        if sleep_calls > 0: hints.append("sleep bloquant")
        if apply_calls > 0 or pandas: hints.append("pandas → vectorisation")
        if numpy: hints.append("numpy détecté")
        if self_recursion: hints.append("auto-récursion détectée")
    except Exception:
        hints.append("AST invalide")
    return hints


def _ast_lint_text(code: str) -> str:
    import ast
    lines=[]
    try:
        tree=ast.parse(code)
        loops=0; appends=0; applies=0; comps=0
        class V(ast.NodeVisitor):
            def visit_For(self, n):  # type: ignore
                nonlocal loops; loops += 1; self.generic_visit(n)
            def visit_Attribute(self, n):  # type: ignore
                nonlocal appends
                if n.attr == "append": appends += 1
                self.generic_visit(n)
            def visit_Call(self, n):  # type: ignore
                nonlocal applies
                if isinstance(n.func, ast.Attribute) and n.func.attr=="apply": applies += 1
                self.generic_visit(n)
            def visit_ListComp(self, n):  # type: ignore
                nonlocal comps; comps += 1; self.generic_visit(n)
        V().visit(tree)
        lines = [
            "AST LINT:",
            f"- Boucles: {loops}",
            f"- append() dans boucle: {appends}",
            f"- .apply() pandas: {applies}",
            f"- List comprehensions: {comps}",
        ]
    except Exception:
        lines = ["AST LINT: indisponible"]
    return "\n".join(lines)

# --------- RAG Simple (KB seule) ---------
def rag_simple_generate(retriever, user_code: str, language: str = "python"):
    query = re.sub(r"\s+", " ", user_code.strip())
    docs = retriever.get_relevant_documents(query)
    stuffed_context = "\n\n".join([d.page_content for d in docs[:6]])
    rule_ids = [d.metadata.get("id", "?") for d in docs[:6]]

    SYSTEM = (
        "Tu es un assistant 'Green Code' qui optimise le code selon la base de connaissances (KB) uniquement.\n"
        "N'ajoute aucune règle externe et ne modifie pas la logique du programme."
    )
    USER = (
        f"[EXTRAITS_KB]\n{stuffed_context}\n\n"
        f"[CODE_UTILISATEUR]\n```{language}\n{user_code}\n```\n\n"
        "FORMAT EXACT :\n"
        "[CODE_OPTIMISE]\n```{language}\n...code...\n```\n"
        "[CHANGES]\n- Liste concise des optimisations justifiées par la KB uniquement."
    )

    llm_simple = ChatNVIDIA(
        model="mistralai/mistral-small-24b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.2,
        max_tokens=1100,
    )

    response = llm_simple.invoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER}
    ])
    content = getattr(response, "content", str(response))

    # ⤵️ Extraction tolérante (tag différent)
    optimized_code = _extract_code_block(content, tag="CODE_OPTIMISE")

    m_changes = re.search(r"\[CHANGES\]\s*([\s\S]*?)\Z", content)
    changes = (m_changes.group(1).strip() if m_changes else "").strip() or "- aucune optimisation KB"

    return {
        "niveau": "Simple (KB only)",
        "rule_ids": rule_ids,
        "code": optimized_code,
        "changes": changes,
        "raw": content
    }


# --------- RAG AST (le plus optimal – KB only) ---------
def rag_ast_generate(retriever, user_code: str, language: str = "python"):
    hints = _ast_hints(user_code)
    lint = _ast_lint_text(user_code)
    hints_text = "; ".join(hints) or "aucun motif détecté"

    query = user_code + " " + hints_text
    docs = retriever.get_relevant_documents(query)
    stuffed_context = "\n\n".join([d.page_content for d in docs[:6]])
    rule_ids = [d.metadata.get("id", "?") for d in docs[:6]]

    SYSTEM = (
        "Tu es un assistant 'Green Code' utilisant UNIQUEMENT la KB.\n"
        "Tu disposes d'une carte AST qui te guide dans les optimisations autorisées.\n"
        "N'introduis aucune règle externe ni phrases génériques.\n"
        "Explique uniquement les VRAIES modifications effectuées sur le code."
    )

    USER = (
        f"[CARTE_AST]\n{lint}\n\n"
        f"[INDICES_AST]\n{hints_text}\n\n"
        f"[EXTRAITS_KB]\n{stuffed_context}\n\n"
        f"[CODE_UTILISATEUR]\n```{language}\n{user_code}\n```\n\n"
        "FORMAT EXACT :\n"
        "[CODE_OPTIMISE]\n```{language}\n...code...\n```\n"
        "[MODIFICATIONS]\n- Liste concise des modifications réellement apportées au code."
    )

    llm_ast = ChatNVIDIA(
        model="mistralai/mistral-small-24b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.12,
        max_tokens=1200,
    )

    resp = llm_ast.invoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER}
    ])
    content = getattr(resp, "content", str(resp))

    # ⤵️ Extraction tolérante (tag différent)
    code_opt = _extract_code_block(content, tag="CODE_OPTIMISE")

    m_reason = re.search(r"\[MODIFICATIONS\]\s*([\s\S]*?)\Z", content)
    reasoning = (m_reason.group(1).strip() if m_reason else "").strip()
    if not reasoning and content:
        after_code = content.split("```")[-1].strip()
        reasoning = _sanitize_changes(after_code)

    reasoning = _sanitize_changes(reasoning)

    return {
        "niveau": "AST (Optimal – KB only)",
        "rule_ids": rule_ids,
        "code": code_opt,
        "changes": reasoning,
        "hints": hints,
        "raw": content
    }


def llm_only_generate(user_code: str, language: str = "python",
                      temperature: float = 0.15, max_tokens: int = 1100):
    """
    Génération sans KB ni web : l'LLM propose des micro-optimisations
    sûres et lisibles, sans changer la sémantique.
    """
    SYSTEM = (
        "Tu es un assistant 'Green Code' qui optimise légèrement le code "
        "pour réduire CPU/RAM/énergie/CO2 tout en gardant strictement le même comportement. "
        "Ne fais pas de refontes lourdes. Privilégie des micro-ajustements sûrs et lisibles. "
        "Reste dans le même langage et renvoie un code exécutable."
    )

    USER = (
        f"[CODE_UTILISATEUR]\n```{language}\n{user_code}\n```\n\n"
        "FORMAT EXACT :\n"
        "[CODE]\n```{language}\n...code optimisé...\n```\n"
        "[CHANGES]\n- puces très concises, uniquement des ajustements sûrs"
    )

    llm = ChatNVIDIA(
        model="mistralai/mistral-small-24b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = llm.invoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]).content

    m_code = re.search(r"\[CODE\]\s*```[^\n]*\n([\s\S]*?)```", content)
    code = (m_code.group(1).strip() if m_code else "").strip()

    m_changes = re.search(r"\[CHANGES\]\s*([\s\S]*?)\Z", content)
    changes = _sanitize_changes((m_changes.group(1).strip() if m_changes else "").strip()) \
              or "- aucune modification majeure"

    return {"code": code, "changes": changes, "sources": ""}


# ───────────────────────────── VUES ─────────────────────────────
def render_home_view():
    st.markdown('<div style="height:100px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🌱 Green Code Optimizer</h1></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:100px"></div>', unsafe_allow_html=True)

    center = st.columns([1, 2, 1])[1]
    with center:
        b1, b2 = st.columns([1, 1], gap="large")
        with b1:
            if st.button("Évaluation d’empreinte", key="home_eval_btn"):
                ss.page = "eval"; _set_query_page("eval"); st.rerun()
        with b2:
            if st.button("Génération du code green", key="home_rag_btn"):
                ss.page = "rag"; _set_query_page("rag"); st.rerun()

def render_eval_view():
    st.title("Évaluation d’empreinte")
    left, right = st.columns(2, gap="large")

    with left:
        tool = "CodeCarbon"
        ss["tool_select"] = "CodeCarbon"
        st.markdown('<span class="badge">Outil de mesure: Code Carbon</span>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Code à analyser :</div>', unsafe_allow_html=True)

        code_to_analyse = st_ace(
            value=ss.get("code_input_analyse", ""),
            language="python",
            theme="tomorrow_night",
            keybinding="vscode",
            min_lines=16,
            height=226,
            tab_size=4,
            wrap=False,
            show_gutter=True,
            show_print_margin=False,
            auto_update=True,
            key="ace_analyse",
        ) or ""
        ss["code_input_analyse"] = code_to_analyse
        run_btn = st.button("Analyser", key="btn_analyser")

    with right:
        if st.button("Aller à la génération →", key="go_to_rag"):
            if ss.get("code_input_analyse", "").strip():
                ss["code_input_generate"] = ss["code_input_analyse"]
            ss.page = "rag"
            _set_query_page("rag")
            st.rerun()

    if run_btn and ss["code_input_analyse"].strip():
        lang = detect_language(ss["code_input_analyse"])
        ok_syntax, tb = preflight_compile(ss["code_input_analyse"])

        with st.spinner("Mesure en cours…"):
            if not ok_syntax:
                res = {"run_error": True, "stderr": tb}
            else:
                fn = MEASURERS.get(ss["tool_select"])
                res = fn(ss["code_input_analyse"]) if fn else {"error":"unsupported_tool","notes":"Outil non pris en charge."}

        ss["history"].append({"tool": ss["tool_select"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "code": ss["code_input_analyse"], "res": res})
        st.subheader("Résultats d’analyse")

        if res.get("error"):
            st.error(res.get("notes") or f"Erreur {res.get('error')}")
            if res.get("stderr"):
                with st.expander("Voir le détail de l’erreur (traceback)"):
                    st.code(res["stderr"])
        elif res.get("run_error") or res.get("stderr"):
            st.warning("⚠️ Alerte d’exécution : erreur Python — le code n’a pas pu être lancé.")
            if res.get("stderr"):
                with st.expander("Voir le détail de l’erreur (traceback)"):
                    st.code(res["stderr"])
            if any(res.get(k) for k in ("duration_s","energy_kwh","emissions_kg","cpu_energy_kwh","gpu_energy_kwh","ram_energy_kwh")):
                render_result(res)
        else:
            render_result(res)

def _eval_this_version(code: str):
    if not code or not code.strip():
        return
    st.session_state["code_input_analyse"] = code
    st.session_state.page = "eval"
    _set_query_page("eval")
    st.session_state["__goto_eval__"] = True

def render_rag_view():
    st.title("Génération")
    if not require_nvidia_key():
        st.stop()

    left, right = st.columns([3, 2], gap="large")

    # =======================
    # Zone gauche : éditeur + bouton
    # =======================
    with left:
        st.markdown('<div class="field-label">Code non green pour génération :</div>', unsafe_allow_html=True)
        code_to_generate = st_ace(
            value=st.session_state.get("code_input_generate", ""),
            language="python",
            theme="tomorrow_night",
            keybinding="vscode",
            min_lines=22,
            height=355,
            tab_size=4,
            wrap=False,
            show_gutter=True,
            show_print_margin=False,
            auto_update=True,
            key="ace_generate",
        ) or ""
        st.session_state["code_input_generate"] = code_to_generate

        c1, c2 = st.columns([1, 1])
        with c1:
            gen_btn = st.button("Générer", key="btn_generer_3")
        with c2:
            pass

    # =======================
    # Zone droite : choix des générateurs
    # =======================
    with right:
        st.markdown('<div class="field-label">Choix des générateurs :</div>', unsafe_allow_html=True)

        kb_needed_simple = not st.session_state.get("kb_df_loaded_simple", False)
        kb_needed_ast = kb_needed_simple  # AST dépend de Simple

        st.markdown('<div class="opt-wrap">', unsafe_allow_html=True)

        def opt_card(title: str, desc: str, key: str, value: bool, disabled: bool = False):
            c1o, c2o = st.columns([1, 7], vertical_alignment="center")
            with c1o:
                on = st.toggle("", value=value, key=key, disabled=disabled)
            with c2o:
                klass = "opt-card opt-disabled" if disabled else "opt-card"
                st.markdown(
                    f"""
                    <div class="{klass}">
                      <div style="display:flex; flex-direction:column;">
                        <div class="opt-title">{title}</div>
                        <div class="opt-desc">{desc}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            return on

        # Ordre visuel : AST > Simple > Hétérogène > LLM seul
        ch_ast = opt_card(
            "RAG + AST",
            "Analyse AST + Recherche dans la Knowledge base. ",
            key="opt_rag_ast",
            value=st.session_state.get("rag_selected", {}).get("ast", True),
            disabled=kb_needed_ast,
        )

        ch_simple = opt_card(
            "RAG + Knowledge base",
            "Recherche dans la Knowledge base et usage des extraits comme contexte. ",
            key="opt_rag_simple",
            value=st.session_state.get("rag_selected", {}).get("simple", True),
            disabled=kb_needed_simple,
        )

        ch_hetero = opt_card(
            "RAG + Bonnes pratiques",
            "Combine des sources (fichiers locaux + URLs) comme contexte. ",
            key="opt_rag_hetero",
            value=st.session_state.get("rag_selected", {}).get("hetero", True),
            disabled=False,
        )

        ch_llm = opt_card(
            "LLM Vanilla",
            "Génération uniquement guidée par le prompt. ",
            key="opt_rag_llm",
            value=st.session_state.get("rag_selected", {}).get("llm", True),
            disabled=False,
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["rag_selected"] = {
            "ast": ch_ast, "simple": ch_simple, "hetero": ch_hetero, "llm": ch_llm
        }

        if kb_needed_simple and (ch_simple or ch_ast):
            st.info("KB requise : place un fichier Excel à côté du script ou définis `RAG_KB_PATH`.")

    # =======================
    # Lancement des générations (dans l'ordre)
    # =======================
    if gen_btn and st.session_state.get("code_input_generate", "").strip():
        code_src = st.session_state["code_input_generate"]
        st.session_state["gen3"] = {}

        # AST
        if ch_ast and st.session_state.get("kb_faiss_ready_ast"):
            try:
                with st.spinner("RAG AST…"):
                    st.session_state["gen3"]["ast"] = rag_ast_generate(
                        st.session_state.retriever_ast, code_src, "python"
                    )
            except Exception as e:
                st.error(f"Erreur AST : {e}")

        # Simple (KB)
        if ch_simple and st.session_state.get("kb_faiss_ready_simple"):
            try:
                with st.spinner("RAG Simple (KB)…"):
                    st.session_state["gen3"]["simple"] = rag_simple_generate(
                        st.session_state.retriever_simple, code_src, "python"
                    )
            except Exception as e:
                st.error(f"Erreur Simple : {e}")

        # Hétérogène
        if ch_hetero:
            try:
                with st.spinner("RAG Hétérogène…"):
                    st.session_state["gen3"]["hetero"] = rag_hetero_generate(code_src)
            except Exception as e:
                st.error(f"Erreur Hétérogène : {e}")

        # LLM seul
        if ch_llm:
            try:
                with st.spinner("LLM seul…"):
                    st.session_state["gen3"]["llm"] = llm_only_generate(code_src, "python")
            except Exception as e:
                st.error(f"Erreur LLM seul : {e}")

    # =======================
    # Rendu des résultats
    # =======================
    out_ast     = (st.session_state.get("gen3") or {}).get("ast")
    out_simple  = (st.session_state.get("gen3") or {}).get("simple")
    out_hetero  = (st.session_state.get("gen3") or {}).get("hetero")
    out_llm     = (st.session_state.get("gen3") or {}).get("llm")

    if out_ast:
        render_rag_block("AST", "ast", out_ast, show_sources=False)
    if out_simple:
        render_rag_block("Simple (KB)", "simple", out_simple, show_sources=False)
    if out_hetero:
        render_rag_block("Hétérogène", "hetero", out_hetero, show_sources=True)
    if out_llm:
        render_rag_block("LLM seul", "llm", out_llm, show_sources=False)

    # Liste des générations disponibles (pour le feedback)
    st.session_state["last_gen_rags"] = [
        k for k in ("ast", "simple", "hetero", "llm")
        if (st.session_state.get("gen3", {}).get(k) and
            (st.session_state["gen3"][k].get("code", "").strip()))
    ]

    # =======================
    # Carte de feedback
    # =======================
    with st.container():
        st.markdown('<div class="feedback-fab"><div class="feedback-card">', unsafe_allow_html=True)
        st.markdown("<h4>Votre avis</h4>", unsafe_allow_html=True)

        _rag_labels = {"ast": "AST", "simple": "Simple (KB)", "hetero": "Hétérogène", "llm": "LLM seul"}
        available_rags = st.session_state.get("last_gen_rags", [])

        def _clear_feedback_state():
            for k in ("fb_text", "fb_alt_code", "fb_rag_choice_label"):
                st.session_state.pop(k, None)
            st.session_state.pop("__fb_inv_map__", None)

        def _submit_feedback_cb():
            choice_label = st.session_state.get("fb_rag_choice_label")
            comment = (st.session_state.get("fb_text") or "").strip()
            alt_code = (st.session_state.get("fb_alt_code") or "").strip()
            record = _build_feedback_record(choice_label=choice_label, comment=comment, alt_code=alt_code)
            try:
                _append_jsonl(record)
                st.toast("Merci pour votre retour !")
            except Exception as e:
                st.toast(f"⚠️ Impossible d'enregistrer le feedback: {e}")
            _clear_feedback_state()

        if not available_rags:
            st.info("Générez d’abord du code")
        else:
            display_opts = [_rag_labels[k] for k in available_rags] + ["Aucun"]
            inv_map = {v: k for k, v in _rag_labels.items()}
            st.session_state["__fb_inv_map__"] = inv_map

            default_index = 0
            if "fb_rag_choice_label" in st.session_state and st.session_state["fb_rag_choice_label"] in display_opts:
                default_index = display_opts.index(st.session_state["fb_rag_choice_label"])

            choice_label = st.radio(
                "Quelle génération vous paraît la plus pertinente ?",
                display_opts,
                index=default_index,
                horizontal=True,
                key="fb_rag_choice_label",
            )

            st.text_area(
                "Un commentaire ?",
                key="fb_text",
                label_visibility="visible",
                height=90,
                placeholder="Dites-nous ce qui va / ne va pas…",
            )

            if choice_label == "Aucun":
                st.markdown(
                    '<div class="feedback-small" style="margin:6px 0 6px;">Optionnel — proposez un code que vous jugez plus optimisé :</div>',
                    unsafe_allow_html=True,
                )
                st_ace(
                    value=st.session_state.get("fb_alt_code", ""),
                    language="python",
                    theme="tomorrow_night",
                    keybinding="vscode",
                    min_lines=12,
                    height=180,
                    tab_size=4,
                    wrap=False,
                    show_gutter=True,
                    show_print_margin=False,
                    auto_update=True,
                    key="fb_alt_code",
                )

            st.button("Envoyer l’avis", key="fb_send_btn", on_click=_submit_feedback_cb)

        st.markdown("</div></div>", unsafe_allow_html=True)


def render_rag_block(title: str, key_prefix: str, out: dict, show_sources: bool = False):
    code_txt = (out or {}).get("code", "") or ""
    changes  = (out or {}).get("changes", "") or ""
    sources  = (out or {}).get("sources", "") or ""

    # 🧹 Nettoyage AST : on retire les mentions automatiques et on garde seulement les vraies modifs
    if title.startswith("AST"):
        lines = []
        for l in changes.splitlines():
            low = l.lower().strip()
            if not l.strip():
                continue
            if any(phrase in low for phrase in [
                "optimisations justifiées par la kb",
                "règles rag citées",
                "gains estimés",
            ]):
                continue
            if re.search(r"rule_\d+", low):
                continue
            lines.append(l)
        changes = "\n".join(lines).strip()

    # 🧹 Nettoyage Simple (KB) : on enlève les rule_xxx entre parenthèses
    if title.startswith("Simple"):
        changes = re.sub(r"\s*\(rule_\d+\)\s*", "", changes)

    st.subheader(f"Code optimisé — {title}")
    st.code(code_txt if code_txt.strip() else "# Aucun code généré", language="python")

    if changes.strip():
        st.markdown("**Modifications :**\n\n" + changes)

    if show_sources and sources.strip():
        with st.expander("Sources"):
            st.text(sources)

    if code_txt.strip():
        btn_key = f"eval_{key_prefix}_" + hashlib.sha1(code_txt.encode("utf-8")).hexdigest()[:8]
        st.button(
            " Calculer l’empreinte du code optimisé",
            key=btn_key,
            on_click=_eval_this_version,
            args=(code_txt,),
        )

# ───────────────────────────── Rendu page courante ─────────────────────────────
if ss.page in ("rag", "eval"):
    render_top_home_button()

if ss.page == "home":
    render_home_view()
elif ss.page == "rag":
    render_rag_view()
else:
    render_eval_view()

# ───────────────────────────── Sidebar : Historique ─────────────────────────────
with st.sidebar:
    render_history_sidebar()

# ---------- Rerun déclenché hors callback ----------
if st.session_state.get("__goto_eval__"):
    st.session_state["__goto_eval__"] = False
    st.rerun()
