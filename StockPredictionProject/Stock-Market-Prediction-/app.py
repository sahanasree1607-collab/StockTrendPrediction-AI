"""
YourStockPredict - Advanced Stock Prediction Dashboard
Professional stock analysis with multiple ML models and user authentication
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="YourStockPredict",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@700;800;900&family=DM+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════════ */
@keyframes fadeUp {
    0%   { opacity:0; transform:translateY(24px); }
    100% { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn {
    0%  { opacity:0; }
    100%{ opacity:1; }
}
@keyframes slideRight {
    0%   { opacity:0; transform:translateX(-24px); }
    100% { opacity:1; transform:translateX(0); }
}
@keyframes popIn {
    0%   { opacity:0; transform:scale(0.88); }
    70%  { transform:scale(1.03); }
    100% { opacity:1; transform:scale(1); }
}
@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(20,184,166,0.0); }
    50%      { box-shadow: 0 0 24px 4px rgba(20,184,166,0.18); }
}
@keyframes borderFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0%,100% { transform:translateY(0px); }
    50%      { transform:translateY(-6px); }
}
@keyframes shimmerSlide {
    0%   { background-position:-600px 0; }
    100% { background-position: 600px 0; }
}

/* ══════════════════════════════════════════
   BASE
══════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background: #f0faf8;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(20,184,166,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(99,102,241,0.06) 0%, transparent 50%);
    min-height: 100vh;
}

/* ══════════════════════════════════════════
   SIDEBAR  — deep slate
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f172a 0%, #1e293b 60%, #0f2027 100%) !important;
    border-right: 1px solid rgba(20,184,166,0.2);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.01em;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(20,184,166,0.08) !important;
    border: 1px solid rgba(20,184,166,0.3) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #f1f5f9 !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.75rem 1.2rem;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 4px 15px rgba(20,184,166,0.3);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 25px rgba(20,184,166,0.45);
}

/* Popover dropdown */
[data-baseweb="popover"]        { background:#1e293b !important; border:1px solid rgba(20,184,166,0.25) !important; border-radius:12px !important; }
[data-baseweb="popover"] ul     { background:#1e293b !important; }
[data-baseweb="popover"] ul li  { background:#1e293b !important; color:#e2e8f0 !important; }
[data-baseweb="popover"] ul li:hover,
[data-baseweb="popover"] ul li[aria-selected="true"] {
    background: rgba(20,184,166,0.2) !important;
    color: #14b8a6 !important;
}

/* ══════════════════════════════════════════
   HEADER
══════════════════════════════════════════ */
.app-header {
    background: linear-gradient(135deg, #0f2027 0%, #134e4a 40%, #0f766e 100%);
    border-radius: 22px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(20,184,166,0.2), 0 2px 8px rgba(0,0,0,0.1);
    animation: fadeUp 0.6s cubic-bezier(0.22,1,0.36,1) both;
}
.app-header::before {
    content:'';
    position:absolute; top:-60%; right:-8%;
    width:380px; height:380px;
    background: radial-gradient(circle, rgba(20,184,166,0.18) 0%, transparent 65%);
    pointer-events:none;
    animation: float 6s ease-in-out infinite;
}
.app-header::after {
    content:'';
    position:absolute; bottom:-50%; left:20%;
    width:250px; height:250px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 65%);
    pointer-events:none;
}
.app-header h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.03em;
}
.app-header p { color: rgba(255,255,255,0.75) !important; font-size: 1rem; margin:0; font-weight:400; }
.header-badge {
    display:inline-block;
    background: rgba(20,184,166,0.25);
    border: 1px solid rgba(20,184,166,0.5);
    color: #5eead4 !important;
    padding: 0.22rem 0.8rem;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 700;
    margin-right: 0.4rem;
    margin-top: 0.75rem;
    letter-spacing: 0.04em;
}
.welcome-chip {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff !important;
    padding: 0.4rem 1.1rem;
    border-radius: 20px;
    font-size: 0.88rem;
    font-weight: 600;
    backdrop-filter: blur(8px);
}

/* ══════════════════════════════════════════
   LOGIN PAGE
══════════════════════════════════════════ */
.login-hero {
    background: linear-gradient(135deg, #0f2027 0%, #134e4a 45%, #0f766e 100%);
    border-radius: 28px;
    padding: 4rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(20,184,166,0.25);
    animation: fadeUp 0.7s cubic-bezier(0.22,1,0.36,1) both;
}
.login-hero::before {
    content:'';
    position:absolute; top:-40%; right:-10%;
    width:400px; height:400px;
    background: radial-gradient(circle, rgba(20,184,166,0.15) 0%, transparent 65%);
    animation: float 8s ease-in-out infinite;
    pointer-events:none;
}
.login-hero::after {
    content:'';
    position:absolute; bottom:-50%; left:5%;
    width:300px; height:300px;
    background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 65%);
    pointer-events:none;
}
.login-hero h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.04em;
    margin-bottom: 0.6rem;
}
.login-hero p { color: rgba(255,255,255,0.72) !important; font-size: 1.05rem; margin:0; }
.login-box {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.06);
    animation: fadeUp 0.8s cubic-bezier(0.22,1,0.36,1) 0.1s both;
}

/* ══════════════════════════════════════════
   INPUTS
══════════════════════════════════════════ */
.stTextInput > div > div > input {
    background: #f8fffe !important;
    border: 1.5px solid #99f6e4 !important;
    border-radius: 11px !important;
    color: #0f172a !important;
    font-size: 0.95rem;
    padding: 0.65rem 1rem !important;
    transition: all 0.25s;
}
.stTextInput > div > div > input:focus {
    border-color: #14b8a6 !important;
    box-shadow: 0 0 0 3px rgba(20,184,166,0.15) !important;
    background: #ffffff !important;
}
.stTextInput label {
    color: #475569 !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.72rem 1.5rem;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 3px 12px rgba(20,184,166,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    box-shadow: 0 8px 24px rgba(20,184,166,0.4);
    transform: translateY(-2px) scale(1.02);
}

/* ══════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════ */
.metric-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    margin: 0.3rem 0;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.metric-card::before {
    content:'';
    position:absolute; top:0; left:0;
    width:4px; height:100%;
    background: linear-gradient(180deg, #14b8a6, #6366f1);
    border-radius:18px 0 0 18px;
}
.metric-card:hover {
    border-color: #14b8a6;
    box-shadow: 0 12px 32px rgba(20,184,166,0.15);
    transform: translateY(-4px);
}
.metric-label {
    color: #64748b !important;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.metric-value {
    color: #0f172a !important;
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
    animation: popIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.metric-value.positive { color: #059669 !important; }
.metric-value.negative { color: #dc2626 !important; }

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 0.4rem;
    gap: 0.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b !important;
    border-radius: 11px;
    font-weight: 600;
    font-size: 0.83rem;
    padding: 0.55rem 1rem;
    border: none !important;
    transition: all 0.25s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0f172a !important;
    background: #f0fdf4;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(20,184,166,0.35);
}

/* ══════════════════════════════════════════
   SECTION TITLE
══════════════════════════════════════════ */
.section-title {
    color: #0f172a !important;
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    margin: 1.6rem 0 0.9rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #ccfbf1;
    position: relative;
    animation: slideRight 0.4s cubic-bezier(0.22,1,0.36,1) both;
}
.section-title::after {
    content:'';
    position:absolute; bottom:-2px; left:0;
    width:48px; height:2px;
    background: linear-gradient(90deg, #14b8a6, #6366f1);
    border-radius:2px;
}

/* ══════════════════════════════════════════
   MODEL CARDS (next-day predictions)
══════════════════════════════════════════ */
.model-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.4rem;
    margin: 0.4rem 0;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.model-card:hover {
    border-color: #14b8a6;
    box-shadow: 0 12px 32px rgba(20,184,166,0.15);
    transform: translateY(-4px);
}
.model-card .model-name {
    color: #64748b !important;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.model-card .model-price {
    color: #0f172a !important;
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0.4rem 0 0.2rem;
}
.model-card .model-change { font-size: 0.9rem; font-weight: 700; }
.model-card .model-r2 { color: #94a3b8 !important; font-size: 0.75rem; margin-top: 0.3rem; }

/* ══════════════════════════════════════════
   SIGNAL CARDS
══════════════════════════════════════════ */
.signal-card {
    border-radius: 18px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin: 1rem 0;
    animation: popIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.signal-buy  {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac;
    box-shadow: 0 6px 24px rgba(5,150,105,0.1);
}
.signal-sell {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #fca5a5;
    box-shadow: 0 6px 24px rgba(220,38,38,0.1);
}
.signal-hold {
    background: linear-gradient(135deg, #fefce8, #fef9c3);
    border: 2px solid #fde047;
    box-shadow: 0 6px 24px rgba(161,98,7,0.08);
}
.signal-card .signal-label {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.signal-buy  .signal-label { color: #065f46 !important; }
.signal-sell .signal-label { color: #991b1b !important; }
.signal-hold .signal-label { color: #92400e !important; }
.signal-card p { color: #475569 !important; margin: 0.3rem 0; font-size: 0.95rem; }

/* ══════════════════════════════════════════
   BEST MODEL CARD
══════════════════════════════════════════ */
.best-model-card {
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 60%, #e0f2fe 100%);
    border: 2px solid #5eead4;
    border-radius: 22px;
    padding: 2.4rem 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(20,184,166,0.15);
    animation: popIn 0.6s cubic-bezier(0.22,1,0.36,1) both;
    position: relative;
    overflow: hidden;
}
.best-model-card::before {
    content:'🏆';
    position:absolute; top:-10px; right:16px;
    font-size:5rem; opacity:0.07;
    pointer-events:none;
}
.best-model-card h2 {
    color: #0f766e !important;
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.best-model-card .score {
    color: #0f172a !important;
    font-size: 3rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    animation: popIn 0.7s cubic-bezier(0.22,1,0.36,1) 0.1s both;
}
.best-model-card p { color: #475569 !important; font-size: 0.95rem; margin: 0.2rem 0; }

/* ══════════════════════════════════════════
   INSIGHT + STRENGTH CARDS
══════════════════════════════════════════ */
.insight-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-left: 4px solid #14b8a6;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.5rem 0;
    transition: all 0.25s;
    animation: slideRight 0.4s cubic-bezier(0.22,1,0.36,1) both;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.insight-card:hover {
    border-left-color: #0d9488;
    box-shadow: 0 6px 20px rgba(20,184,166,0.12);
    transform: translateX(5px);
}
.insight-card p { color: #334155 !important; margin:0; font-size:0.92rem; }
.insight-card strong { color: #0f172a !important; }

.strength-card {
    background: linear-gradient(135deg, #f0fdfa, #f8fffe);
    border: 1.5px solid #ccfbf1;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.5rem 0;
    transition: all 0.25s;
    animation: fadeIn 0.5s ease both;
}
.strength-card:hover {
    border-color: #5eead4;
    box-shadow: 0 4px 16px rgba(20,184,166,0.1);
    transform: translateY(-2px);
}
.strength-card p { color: #334155 !important; margin:0; font-size:0.92rem; }

/* ══════════════════════════════════════════
   DISCLAIMER
══════════════════════════════════════════ */
.disclaimer {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1.5px solid #fcd34d;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease both;
}
.disclaimer p { color: #78350f !important; font-size: 0.9rem; margin:0; }

/* ══════════════════════════════════════════
   ALERTS
══════════════════════════════════════════ */
.stSuccess { background:#f0fdf4 !important; border-color:#86efac !important; border-radius:12px; }
.stSuccess p { color:#065f46 !important; }
.stError   { background:#fff1f2 !important; border-color:#fca5a5 !important; border-radius:12px; }
.stError p { color:#991b1b !important; }
.stWarning { background:#fefce8 !important; border-color:#fde047 !important; border-radius:12px; }
.stWarning p { color:#92400e !important; }
.stInfo    { background:#f0fdfa !important; border-color:#5eead4 !important; border-radius:12px; }
.stInfo p  { color:#0f766e !important; }

/* ══════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 14px;
    border: 1.5px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    overflow: hidden;
}

/* ══════════════════════════════════════════
   STREAMLIT METRICS
══════════════════════════════════════════ */
[data-testid="stMetricValue"] { color:#0f172a !important; font-weight:800 !important; font-family:'Outfit',sans-serif; }
[data-testid="stMetricLabel"] { color:#64748b !important; font-weight:600 !important; }
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    transition: all 0.25s;
}
[data-testid="metric-container"]:hover {
    border-color: #14b8a6;
    box-shadow: 0 6px 20px rgba(20,184,166,0.1);
    transform: translateY(-2px);
}

/* ══════════════════════════════════════════
   FOOTER
══════════════════════════════════════════ */
.app-footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1.5px solid #e2e8f0;
    animation: fadeIn 0.5s ease both;
}
.app-footer p { color: #94a3b8 !important; font-size: 0.82rem; }

/* ══════════════════════════════════════════
   GENERAL
══════════════════════════════════════════ */
h1,h2,h3,h4 { color:#0f172a !important; font-family:'Outfit',sans-serif; }
.app-header h1, .login-hero h1 { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; background:none !important; }
p,li,span   { color:#334155; }
strong      { color:#0f172a; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════
def load_users():
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        return {}
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
        return {}

def authenticate(username, password):
    users = load_users()
    return username in users and users[username] == password

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
        return True
    except:
        return False

# ══════════════════════════════════════════
# ML
# ══════════════════════════════════════════
class AdvancedStockPredictor:
    def __init__(self, ticker, data=None):
        self.ticker = ticker
        self.data   = data
        self.scaler = StandardScaler()

    def prepare_features(self, df):
        df = df.copy()
        df['Day']              = np.arange(len(df))
        df['MA_5']             = df['Close'].rolling(5).mean()
        df['MA_10']            = df['Close'].rolling(10).mean()
        df['MA_20']            = df['Close'].rolling(20).mean()
        df['MA_50']            = df['Close'].rolling(50).mean()
        df['Volatility_5']     = df['Close'].rolling(5).std()
        df['Volatility_20']    = df['Close'].rolling(20).std()
        df['Returns_1d']       = df['Close'].pct_change()
        df['Returns_5d']       = df['Close'].pct_change(5)
        df['Returns_20d']      = df['Close'].pct_change(20)
        df['High_Low_Ratio']   = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['Volume_MA_5']      = df['Volume'].rolling(5).mean()
        df['Volume_Ratio']     = df['Volume'] / df['Volume'].rolling(20).mean()
        delta = df['Close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss
        df['RSI']       = 100 - (100 / (1 + rs))
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std          = df['Close'].rolling(20).std()
        df['BB_Upper']  = df['BB_Middle'] + bb_std * 2
        df['BB_Lower']  = df['BB_Middle'] - bb_std * 2
        df = df.dropna()
        feature_cols = ['Day','MA_5','MA_10','MA_20','MA_50',
                        'Volatility_5','Volatility_20','Returns_1d','Returns_5d',
                        'Returns_20d','High_Low_Ratio','Close_Open_Ratio',
                        'Volume_Ratio','RSI','BB_Upper','BB_Lower']
        return df[feature_cols].values, df['Close'].values, df, feature_cols

    def train_selected_models(self, selected_models):
        try:
            X, y, df_features, feature_cols = self.prepare_features(self.data)
            if len(X) < 10:
                st.error(f"Not enough data ({len(X)} samples). Use a period of 3 months or more.")
                return None, None, None
            split   = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            X_tr = self.scaler.fit_transform(X_train)
            X_te = self.scaler.transform(X_test)
            catalogue = {
                'Linear Regression':  LinearRegression(),
                'Ridge Regression':   Ridge(alpha=1.0),
                'Lasso Regression':   Lasso(alpha=1.0),
                'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
                'SVR':                SVR(kernel='rbf', C=100, gamma='auto'),
            }
            results = {}
            for name in selected_models:
                if name in catalogue:
                    m = catalogue[name]
                    m.fit(X_tr, y_train)
                    results[name] = self._evaluate(m, X_te, y_test, X_tr, y_train, X, y)
                    if name in ('Random Forest', 'Gradient Boosting'):
                        results[name]['feature_importance'] = m.feature_importances_
            results['features'] = feature_cols
            return results, df_features.index[split:], y_test
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return None, None, None

    def _evaluate(self, model, X_test, y_test, X_train, y_train, X_full, y_full):
        preds    = model.predict(X_test)
        last_sc  = self.scaler.transform(X_full[-1].reshape(1, -1))
        next_day = model.predict(last_sc)[0]
        mae      = mean_absolute_error(y_test, preds)
        rmse     = np.sqrt(mean_squared_error(y_test, preds))
        r2       = r2_score(y_test, preds)
        mape     = np.mean(np.abs((y_test - preds) / y_test)) * 100
        return {'model': model, 'predictions': preds, 'next_day': next_day,
                'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

# ══════════════════════════════════════════
# DATA & CHARTS
# ══════════════════════════════════════════
def get_stock_data(ticker, period='2y'):
    try:
        stock = yf.Ticker(ticker)
        data  = stock.history(period=period)
        if data.empty:
            return None, None
        return data, stock.info
    except Exception as e:
        st.error(f"Fetch error: {str(e)}")
        return None, None

# Clean white chart layout — maximum readability
PLOT_LAYOUT = dict(
    template='plotly_white',
    paper_bgcolor='#ffffff',
    plot_bgcolor='#fafffe',
    font=dict(family='Plus Jakarta Sans', color='#334155', size=12),
    xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0', linecolor='#e2e8f0'),
    yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0', linecolor='#e2e8f0'),
    hoverlabel=dict(bgcolor='#0f172a', font_color='#f1f5f9', bordercolor='#14b8a6'),
)

def create_advanced_charts(data, ticker):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Price with Moving Averages', 'Trading Volume',
                        'RSI Indicator', 'Bollinger Bands',
                        'Price Distribution', 'Returns Distribution'),
        vertical_spacing=0.13, horizontal_spacing=0.1
    )
    # 1. Price + MAs
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close',
                             line=dict(color='#0f766e', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(),
                             name='MA20', line=dict(color='#f59e0b', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(),
                             name='MA50', line=dict(color='#6366f1', width=1.5, dash='dash')), row=1, col=1)
    # 2. Volume
    colors = ['#f43f5e' if data['Close'].iloc[i] < data['Close'].iloc[i-1] else '#10b981'
              for i in range(1, len(data))]
    colors.insert(0, '#10b981')
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                         marker_color=colors, opacity=0.8), row=1, col=2)
    # 3. RSI
    delta = data['Close'].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))
    fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI',
                             line=dict(color='#8b5cf6', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#f43f5e', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#10b981', row=2, col=1)
    # 4. Bollinger
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price',
                             line=dict(color='#0f766e')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=bb_mid+bb_std*2, name='BB+',
                             line=dict(color='#f43f5e', dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=bb_mid-bb_std*2, name='BB−',
                             line=dict(color='#10b981', dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=bb_mid, name='BB mid',
                             line=dict(color='#f59e0b', dash='dot')), row=2, col=2)
    # 5. Price dist
    fig.add_trace(go.Histogram(x=data['Close'], nbinsx=30,
                               marker_color='#14b8a6', name='Price Dist',
                               marker_line=dict(color='#0d9488', width=0.5)), row=3, col=1)
    # 6. Returns dist
    returns = data['Close'].pct_change().dropna() * 100
    fig.add_trace(go.Histogram(x=returns, nbinsx=50,
                               marker_color='#8b5cf6', name='Returns Dist',
                               marker_line=dict(color='#7c3aed', width=0.5)), row=3, col=2)
    fig.update_layout(height=1050, title_text=f"<b>{ticker}</b> — Advanced Technical Analysis",
                      showlegend=True, title_font=dict(size=18, color='#0f172a'), **PLOT_LAYOUT)
    return fig

def create_prediction_comparison(results, dates, y_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, name='Actual',
                             line=dict(color='#0f172a', width=3)))
    palette = ['#14b8a6','#6366f1','#f59e0b','#f43f5e','#10b981','#8b5cf6']
    i = 0
    for name, res in results.items():
        if name != 'features':
            fig.add_trace(go.Scatter(x=dates, y=res['predictions'], name=name,
                                     line=dict(color=palette[i % len(palette)], width=2, dash='dash')))
            i += 1
    fig.update_layout(title='<b>Model Predictions vs Actual</b>', xaxis_title='Date',
                      yaxis_title='Price (USD)', height=550, hovermode='x unified',
                      title_font=dict(size=16, color='#0f172a'), **PLOT_LAYOUT)
    return fig

def create_error_heatmap(results):
    models, zvals = [], []
    for name, res in results.items():
        if name != 'features' and 'mae' in res:
            models.append(name)
            zvals.append([res['mae'], res['rmse'], res['mape']])
    if not zvals:
        return None
    # Use a light-to-dark teal colorscale — fully visible on white background
    fig = go.Figure(go.Heatmap(
        z=zvals, x=['MAE ($)', 'RMSE ($)', 'MAPE (%)'], y=models,
        colorscale=[
            [0.0,  '#f0fdfa'],
            [0.25, '#99f6e4'],
            [0.5,  '#2dd4bf'],
            [0.75, '#0f766e'],
            [1.0,  '#134e4a'],
        ],
        text=np.round(zvals, 2),
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=13, color='#0f172a'),
        showscale=True,
        colorbar=dict(title='Value', tickfont=dict(color='#334155'))
    ))
    fig.update_layout(
        title='<b>Model Error Comparison</b>',
        xaxis_title='Error Metrics',
        yaxis_title='Models',
        height=420,
        title_font=dict(size=16, color='#0f172a'),
        xaxis=dict(tickfont=dict(color='#334155', size=13), gridcolor='#e2e8f0'),
        yaxis=dict(tickfont=dict(color='#334155', size=13), gridcolor='#e2e8f0'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#fafffe',
        font=dict(family='Plus Jakarta Sans', color='#334155'),
    )
    return fig

def generate_trading_recommendation(results, current_price):
    recommendations, weights = [], []
    for name, res in results.items():
        if name != 'features' and 'next_day' in res:
            pred_change = ((res['next_day'] - current_price) / current_price) * 100
            weight = max(0.01, res['r2'])
            recommendations.append(pred_change)
            weights.append(weight)
    if recommendations and sum(weights) > 0:
        wpc = np.average(recommendations, weights=weights)
        if   wpc > 2:    sig,cls,conf = "STRONG BUY",  "signal-buy",  "High"
        elif wpc > 0.5:  sig,cls,conf = "BUY",         "signal-buy",  "Medium"
        elif wpc > -0.5: sig,cls,conf = "HOLD",        "signal-hold", "Low"
        elif wpc > -2:   sig,cls,conf = "SELL",        "signal-sell", "Medium"
        else:            sig,cls,conf = "STRONG SELL", "signal-sell", "High"
        return {'signal':sig,'class':cls,'expected_change':wpc,'confidence':conf,
                'next_day_price': current_price*(1+wpc/100)}
    return None

# ══════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════
def login_page():
    st.markdown("""
    <div class="login-hero">
       <h1 style="color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;">📈 YourStockPredict</h1>
        <p>Intelligent Stock Predictions Powered by Advanced Machine Learning</p>
        <div style="margin-top:1rem;">
            <span class="header-badge">Real-time Analysis</span>
            <span class="header-badge">6 ML Models</span>
            <span class="header-badge">Professional Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome Back")
        st.markdown("Login to access your personalized dashboard")
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        with tab1:
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            if st.button("Login", key="login_btn", use_container_width=True):
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username']      = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        with tab2:
            new_username     = st.text_input("Username",         key="reg_username",  placeholder="Choose a username")
            new_password     = st.text_input("Password",         type="password", key="reg_password",  placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            if st.button("Register", key="reg_btn", use_container_width=True):
                if not new_username or not new_password:
                    st.error("❌ Please fill all fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif register_user(new_username, new_password):
                    st.success("✅ Registration successful! Please login.")
                else:
                    st.error("❌ Username already exists")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════
def main_app():
    user = st.session_state['username']

    st.markdown(f"""
    <div class="app-header">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
            <div>
               <h1 style="color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;">📈 YourStockPredict</h1>
                <p>Intelligent Stock Predictions Powered by Advanced Machine Learning</p>
                <div style="margin-top:0.75rem;">
                    <span class="header-badge">6 ML Models</span>
                    <span class="header-badge">Real-time</span>
                    <span class="header-badge">Technical Analysis</span>
                </div>
            </div>
            <span class="welcome-chip">👋 Welcome, {user}!</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 Stock Selection")
        st.markdown("---")
        popular_stocks = {
            'Select a stock...': '',
            'Apple Inc. (AAPL)': 'AAPL',
            'Microsoft Corp. (MSFT)': 'MSFT',
            'Google (GOOGL)': 'GOOGL',
            'Amazon.com Inc. (AMZN)': 'AMZN',
            'Tesla Inc. (TSLA)': 'TSLA',
            'Meta Platforms (META)': 'META',
            'NVIDIA Corp. (NVDA)': 'NVDA',
            'Netflix Inc. (NFLX)': 'NFLX',
            'The Walt Disney Co. (DIS)': 'DIS',
            'Coca-Cola Co. (KO)': 'KO',
            'Custom Ticker...': 'custom'
        }
        stock_option = st.selectbox("Select Stock", list(popular_stocks.keys()))
        if stock_option == 'Custom Ticker...':
            ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
        else:
            ticker = popular_stocks[stock_option]
            if ticker and ticker != 'custom':
                st.info(f"✅ Selected: {stock_option}")
        if not ticker:
            st.warning("⚠️ Please select a stock")
            return
        period = st.selectbox("Analysis Period", ["3mo","6mo","1y","2y","5y"], index=2)
        st.markdown("---")
        st.markdown("## 🤖 Select ML Models")
        st.markdown("Choose the models you want to use for predictions:")
        model_options = {
            'Linear Regression':  True,
            'Ridge Regression':   True,
            'Lasso Regression':   False,
            'Random Forest':      True,
            'Gradient Boosting':  True,
            'SVR':                False,
        }
        selected_models = [m for m, d in model_options.items() if st.checkbox(m, value=d)]
        if not selected_models:
            st.warning("⚠️ Please select at least one model")
            return
        st.markdown(f"**✅ Selected Models:** {len(selected_models)}")
        st.markdown("---")
        st.markdown("## 📊 What You Can See Here:")
        st.markdown("""
        - ✅ Technical Indicators
        - ✅ Moving Averages (5,10,20,50)
        - ✅ RSI & Bollinger Bands
        - ✅ Volume Analysis
        - ✅ Price Patterns
        - ✅ Risk Metrics
        """)
        analyze_btn = st.button("🚀 Analyze Stock", use_container_width=True)
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()

    # ── RUN ANALYSIS ──────────────────────────────────────────────────────────
    if analyze_btn:
        with st.spinner(f"Fetching and analyzing {ticker}..."):
            data, info = get_stock_data(ticker, period)
            if data is not None and not data.empty:
                st.session_state.update({'stock_data': data, 'stock_info': info, 'ticker': ticker})
                st.success(f"✅ Successfully fetched data for {ticker}")
                with st.spinner(f"Training {len(selected_models)} machine learning models..."):
                    predictor = AdvancedStockPredictor(ticker, data)
                    results, test_dates, y_test = predictor.train_selected_models(selected_models)
                    if results:
                        st.session_state.update({'ml_results': results,
                                                 'test_dates': test_dates, 'y_test': y_test})
                        st.success(f"✅ {len(selected_models)} models trained successfully!")
                    else:
                        st.error("❌ Model training failed. Please try with more data (3 months or longer).")
            else:
                st.error(f"❌ No data found for ticker {ticker}")
                return

    if 'stock_data' not in st.session_state:
        st.info("👈 Select a stock and click **Analyze Stock** to begin.")
        return

    data       = st.session_state['stock_data']
    info       = st.session_state['stock_info']
    ticker     = st.session_state['ticker']
    results    = st.session_state.get('ml_results')
    test_dates = st.session_state.get('test_dates')
    y_test     = st.session_state.get('y_test')

    current_price    = data['Close'].iloc[-1]
    prev_price       = data['Close'].iloc[-2]
    price_change     = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    # ── METRIC ROW ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">💰 Current Price</div>
            <div class="metric-value">${current_price:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        cls = "positive" if price_change >= 0 else "negative"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">📊 Daily Change</div>
            <div class="metric-value {cls}">{price_change:+.2f} ({price_change_pct:+.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">📈 Volume</div>
            <div class="metric-value">{data['Volume'].iloc[-1]:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        mcap = info.get('marketCap', 0) / 1e9
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">🏢 Market Cap</div>
            <div class="metric-value">${mcap:.1f}B</div>
        </div>""", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 DASHBOARD", "📈 TECHNICAL", "🤖 MODELS",
        "🔮 FORECAST",  "📉 RISK",      "🏆 SUMMARY"
    ])

    # ── TAB 1: DASHBOARD ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("## 📋 Stock Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🏢 Company Details")
            st.write(f"**Name:** {info.get('longName','N/A')}")
            st.write(f"**Sector:** {info.get('sector','N/A')}")
            st.write(f"**Industry:** {info.get('industry','N/A')}")
            st.write(f"**Country:** {info.get('country','N/A')}")
            st.write(f"**Website:** {info.get('website','N/A')}")
        with col2:
            st.markdown("### 📊 Trading Information")
            st.write(f"**52W High:** ${info.get('fiftyTwoWeekHigh',0):.2f}")
            st.write(f"**52W Low:** ${info.get('fiftyTwoWeekLow',0):.2f}")
            st.write(f"**50-Day Avg:** ${info.get('fiftyDayAverage',0):.2f}")
            st.write(f"**200-Day Avg:** ${info.get('twoHundredDayAverage',0):.2f}")
            st.write(f"**P/E Ratio:** {info.get('trailingPE','N/A')}")
        st.markdown("### 📊 Recent Price Data")
        recent_data = data.tail(10)[['Open','High','Low','Close','Volume']].round(2)
        recent_data.index = recent_data.index.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)

    # ── TAB 2: TECHNICAL ──────────────────────────────────────────────────────
    with tab2:
        st.markdown("## 📈 Advanced Technical Analysis")
        st.plotly_chart(create_advanced_charts(data, ticker), use_container_width=True)
        st.markdown("## 📊 Correlation Analysis")
        corr_data = data[['Open','High','Low','Close','Volume']].corr()
        fig_corr  = px.imshow(corr_data, text_auto=True, aspect="auto",
                              title="Price Correlation Matrix",
                              color_continuous_scale=[
                                  [0,'#f0fdfa'],[0.5,'#2dd4bf'],[1,'#134e4a']
                              ])
        fig_corr.update_layout(height=600, paper_bgcolor='#ffffff',
                               plot_bgcolor='#fafffe',
                               title_font=dict(size=16, color='#0f172a'),
                               font=dict(color='#334155'))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── TAB 3: MODELS ─────────────────────────────────────────────────────────
    with tab3:
        if results:
            st.markdown("## 🤖 Model Performance Comparison")
            metrics_data = []
            for model_name, result in results.items():
                if model_name != 'features':
                    metrics_data.append({
                        'Model':        model_name,
                        'MAE ($)':      f"{result['mae']:.2f}",
                        'RMSE ($)':     f"{result['rmse']:.2f}",
                        'R² Score':     f"{result['r2']:.3f}",
                        'MAPE (%)':     f"{result['mape']:.2f}",
                        'Next Day ($)': f"${result['next_day']:.2f}"
                    })
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            st.markdown("## 📊 Error Metrics Heatmap")
            fig_error = create_error_heatmap(results)
            if fig_error:
                st.plotly_chart(fig_error, use_container_width=True)

            tree_models = [n for n, r in results.items()
                           if n != 'features' and 'feature_importance' in r]
            if tree_models:
                st.markdown("## 🔑 Feature Importance")
                col1, col2 = st.columns(2)
                with col1:
                    if 'Random Forest' in tree_models:
                        rf_imp = pd.DataFrame({
                            'Feature':    results['features'],
                            'Importance': results['Random Forest']['feature_importance']
                        }).sort_values('Importance', ascending=True)
                        fig_rf = px.bar(rf_imp.tail(10), x='Importance', y='Feature',
                                        title='Random Forest — Top 10 Features',
                                        orientation='h', color='Importance',
                                        color_continuous_scale=[
                                            [0,'#f0fdfa'],[0.5,'#2dd4bf'],[1,'#0f766e']
                                        ])
                        fig_rf.update_layout(
                            paper_bgcolor='#ffffff', plot_bgcolor='#fafffe',
                            title_font=dict(size=14, color='#0f172a'),
                            font=dict(color='#334155'),
                            xaxis=dict(tickfont=dict(color='#334155'), gridcolor='#e2e8f0'),
                            yaxis=dict(tickfont=dict(color='#334155')),
                            coloraxis_colorbar=dict(tickfont=dict(color='#334155'))
                        )
                        st.plotly_chart(fig_rf, use_container_width=True)
                with col2:
                    if 'Gradient Boosting' in tree_models:
                        gb_imp = pd.DataFrame({
                            'Feature':    results['features'],
                            'Importance': results['Gradient Boosting']['feature_importance']
                        }).sort_values('Importance', ascending=True)
                        fig_gb = px.bar(gb_imp.tail(10), x='Importance', y='Feature',
                                        title='Gradient Boosting — Top 10 Features',
                                        orientation='h', color='Importance',
                                        color_continuous_scale=[
                                            [0,'#faf5ff'],[0.5,'#a78bfa'],[1,'#5b21b6']
                                        ])
                        fig_gb.update_layout(
                            paper_bgcolor='#ffffff', plot_bgcolor='#fafffe',
                            title_font=dict(size=14, color='#0f172a'),
                            font=dict(color='#334155'),
                            xaxis=dict(tickfont=dict(color='#334155'), gridcolor='#e2e8f0'),
                            yaxis=dict(tickfont=dict(color='#334155')),
                            coloraxis_colorbar=dict(tickfont=dict(color='#334155'))
                        )
                        st.plotly_chart(fig_gb, use_container_width=True)
        else:
            st.info("Run analysis first.")

    # ── TAB 4: FORECAST ───────────────────────────────────────────────────────
    with tab4:
        if results and test_dates is not None:
            st.markdown("## 🔮 Price Predictions")
            st.plotly_chart(create_prediction_comparison(results, test_dates, y_test),
                            use_container_width=True)

            st.markdown("## 📉 Prediction Errors Distribution")
            fig_errors = go.Figure()
            for model_name, result in results.items():
                if model_name != 'features':
                    errors = np.abs(result['predictions'] - y_test)
                    fig_errors.add_trace(go.Box(y=errors, name=model_name,
                                                marker_color='#14b8a6',
                                                line_color='#0f766e'))
            fig_errors.update_layout(title='<b>Prediction Error Distribution by Model</b>',
                                     yaxis_title='Absolute Error ($)',
                                     height=600, title_font=dict(size=16, color='#0f172a'),
                                     **PLOT_LAYOUT)
            st.plotly_chart(fig_errors, use_container_width=True)

            st.markdown("## 🎯 Next Day Predictions")
            model_count = len([m for m in results.keys() if m != 'features'])
            if model_count > 0:
                next_day_cols = st.columns(min(model_count, 4))
                idx = 0
                for model_name, result in results.items():
                    if model_name != 'features' and idx < len(next_day_cols):
                        pchg  = (result['next_day'] - current_price) / current_price * 100
                        color = "#059669" if pchg >= 0 else "#dc2626"
                        with next_day_cols[idx]:
                            st.markdown(f"""
                            <div class="model-card">
                                <div class="model-name">{model_name}</div>
                                <div class="model-price">${result['next_day']:.2f}</div>
                                <div class="model-change" style="color:{color};">{pchg:+.2f}%</div>
                                <div class="model-r2">R² Score: {result['r2']:.3f}</div>
                            </div>""", unsafe_allow_html=True)
                        idx += 1
        else:
            st.info("Run analysis first.")

    # ── TAB 5: RISK ───────────────────────────────────────────────────────────
    with tab5:
        st.markdown("## 📊 Risk & Statistical Analysis")
        returns = data['Close'].pct_change().dropna()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📈 Returns Statistics")
            st.metric("Mean Daily Return", f"{returns.mean()*100:.3f}%")
            st.metric("Std Deviation",     f"{returns.std()*100:.3f}%")
            st.metric("Sharpe Ratio",      f"{returns.mean()/returns.std()*np.sqrt(252):.3f}")
            st.metric("Skewness",          f"{returns.skew():.3f}")
            st.metric("Kurtosis",          f"{returns.kurtosis():.3f}")
        with col2:
            st.markdown("### ⚠️ Risk Metrics")
            st.metric("Value at Risk (95%)",  f"{np.percentile(returns,5)*100:.2f}%")
            st.metric("Conditional VaR",      f"{returns[returns<=np.percentile(returns,5)].mean()*100:.2f}%")
            st.metric("Max Drawdown",         f"{(data['Close'].min()/data['Close'].max()-1)*100:.2f}%")
            st.metric("Volatility (Annual)",  f"{returns.std()*np.sqrt(252)*100:.2f}%")
            st.metric("Calmar Ratio",         f"{returns.mean()*252/(abs(data['Close'].min()/data['Close'].max()-1)):.2f}")
        with col3:
            st.markdown("### 📉 Price Statistics")
            st.metric("All-time High",  f"${data['Close'].max():.2f}")
            st.metric("All-time Low",   f"${data['Close'].min():.2f}")
            st.metric("Average Price",  f"${data['Close'].mean():.2f}")
            st.metric("Median Price",   f"${data['Close'].median():.2f}")
            st.metric("Price Range",    f"${data['Close'].max()-data['Close'].min():.2f}")

        if results:
            st.markdown("## 💡 Trading Recommendation")
            recommendation = generate_trading_recommendation(results, current_price)
            if recommendation:
                st.markdown(f"""
                <div class="signal-card {recommendation['class']}">
                    <div class="signal-label">{recommendation['signal']} SIGNAL</div>
                    <p><strong>Expected Price:</strong> ${recommendation['next_day_price']:.2f}</p>
                    <p><strong>Expected Change:</strong> {recommendation['expected_change']:+.2f}%</p>
                    <p><strong>Confidence Level:</strong> {recommendation['confidence']}</p>
                </div>""", unsafe_allow_html=True)

    # ── TAB 6: SUMMARY ────────────────────────────────────────────────────────
    with tab6:
        st.markdown("## 🏆 Model Performance Summary")

        if results:
            best_model = None
            best_r2    = -float('inf')
            best_mae   = float('inf')
            for model_name, result in results.items():
                if model_name != 'features' and result['r2'] > best_r2:
                    best_r2    = result['r2']
                    best_model = model_name
                    best_mae   = result['mae']

            st.markdown(f"""
            <div class="best-model-card">
                <h2>🏆 Best Performing Model: {best_model}</h2>
                <div class="score">{best_r2:.3f}</div>
                <p>R² Score &nbsp;·&nbsp; MAE: ${best_mae:.2f}</p>
                <p>This model explains {best_r2*100:.1f}% of the price variance</p>
            </div>""", unsafe_allow_html=True)

            # Model Rankings
            st.markdown("## 📊 Model Rankings")
            rankings = []
            for model_name, result in results.items():
                if model_name != 'features':
                    rankings.append({
                        'Model':    model_name,
                        'R² Score': result['r2'],
                        'MAE ($)':  result['mae'],
                        'RMSE ($)': result['rmse'],
                        'MAPE (%)': result['mape']
                    })
            rankings_df = pd.DataFrame(rankings).sort_values('R² Score', ascending=False)
            rankings_df.index = range(1, len(rankings_df)+1)
            st.dataframe(rankings_df, use_container_width=True)

            # Investment Recommendations
            st.markdown("## 💡 Investment Recommendations")
            if best_r2 > 0.8:
                st.success("✅ **Highly Reliable Predictions** - Models show excellent performance. Consider using these predictions for investment decisions.")
            elif best_r2 > 0.6:
                st.info("📊 **Moderately Reliable Predictions** - Models show good performance. Use predictions with proper risk management.")
            else:
                st.warning("⚠️ **Limited Reliability** - Models show moderate performance. Consider using as a supplementary tool.")

            # Key Insights
            st.markdown("## 🎯 Key Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Best Features for Prediction:**")
                if 'feature_importance' in results.get('Random Forest', {}):
                    importance_df = pd.DataFrame({
                        'Feature':    results['features'],
                        'Importance': results['Random Forest']['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    for i, row in importance_df.head(5).iterrows():
                        st.markdown(f"""
                        <div class="insight-card">
                            <p>• <strong>{row['Feature']}</strong>: {row['Importance']:.3f}</p>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="insight-card">
                        <p>• Train <strong>Random Forest</strong> to see feature importance</p>
                    </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("**Model Strengths:**")
                strengths = [
                    ("✅ Random Forest",     "Best for capturing non-linear patterns"),
                    ("✅ Gradient Boosting", "Good for complex relationships"),
                    ("✅ Ridge Regression",  "Handles multicollinearity well"),
                    ("✅ SVR",              "Useful for small datasets"),
                ]
                for title, desc in strengths:
                    st.markdown(f"""
                    <div class="strength-card">
                        <p><strong>{title}:</strong> {desc}</p>
                    </div>""", unsafe_allow_html=True)

            # Risk Disclaimer
            st.markdown("""
            <div class="disclaimer">
                <p>⚠️ <strong>Risk Disclaimer:</strong> This dashboard provides predictions based on
                historical data and machine learning models. Past performance does not guarantee
                future results. Always conduct your own research and consult with financial advisors
                before making investment decisions.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Run analysis first.")

    # Footer
    st.markdown("""
    <div class="app-footer">
        <p><strong>📊 YourStockPredict</strong> | Intelligent Stock Predictions Powered by Machine Learning</p>
        <p>Data Source: Yahoo Finance &nbsp;·&nbsp; Real-time Analysis &nbsp;·&nbsp; For Educational Purpose Only</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════
def main():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if not st.session_state['authenticated']:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()