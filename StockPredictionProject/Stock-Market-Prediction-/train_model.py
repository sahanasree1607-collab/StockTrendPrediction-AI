"""
Model training script for pre-trained models
You can run this separately to train and save models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import streamlit as st
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YourStockPredict — Model Trainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0f1117; color: #e2e8f0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27 !important;
        border-right: 1px solid #1e2d45;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1e2d45 !important;
        border: 1px solid #2d4a6e !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span { color: #f1f5f9 !important; }

    /* Dropdown */
    [data-baseweb="popover"] { background-color: #1e2d45 !important; }
    [data-baseweb="popover"] ul { background-color: #1e2d45 !important; }
    [data-baseweb="popover"] ul li { background-color: #1e2d45 !important; color: #f1f5f9 !important; }
    [data-baseweb="popover"] ul li:hover,
    [data-baseweb="popover"] ul li[aria-selected="true"] {
        background-color: #2563eb !important; color: #ffffff !important;
    }

    /* Sidebar multiselect tags */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #2563eb !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span { color: #ffffff !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.7rem 1.5rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 20px rgba(37,99,235,0.4);
        transform: translateY(-1px);
    }

    /* ── Header ── */
    .trainer-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .trainer-header::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .trainer-header h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .trainer-header p { color: #94a3b8 !important; font-size: 0.98rem; margin: 0; }
    .header-badge {
        display: inline-block;
        background: rgba(37,99,235,0.2);
        border: 1px solid rgba(37,99,235,0.4);
        color: #60a5fa !important;
        padding: 0.22rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.7rem;
    }

    /* ── Cards ── */
    .info-card {
        background: #161b27;
        border: 1px solid #1e2d45;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 0.4rem 0;
        transition: border-color 0.2s, transform 0.2s;
    }
    .info-card:hover { border-color: #2563eb; transform: translateY(-2px); }

    .info-card .card-label {
        color: #64748b !important;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .info-card .card-value {
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .info-card .card-sub { color: #475569 !important; font-size: 0.8rem; margin-top: 0.2rem; }

    /* ── Stock queue table ── */
    .stock-queue {
        background: #161b27;
        border: 1px solid #1e2d45;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .stock-queue .sq-title {
        color: #94a3b8 !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.8rem;
    }
    .stock-chip {
        display: inline-block;
        background: rgba(37,99,235,0.15);
        border: 1px solid rgba(37,99,235,0.35);
        color: #60a5fa !important;
        padding: 0.3rem 0.85rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 0.2rem;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Log console ── */
    .log-console {
        background: #0d1117;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        min-height: 180px;
        max-height: 340px;
        overflow-y: auto;
    }
    .log-line { margin: 0.15rem 0; }
    .log-info    { color: #60a5fa !important; }
    .log-success { color: #34d399 !important; }
    .log-warn    { color: #fbbf24 !important; }
    .log-error   { color: #f87171 !important; }
    .log-dim     { color: #475569 !important; }

    /* ── Result row ── */
    .result-row {
        background: #161b27;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    /* ── Section title ── */
    .section-title {
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.6rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e2d45;
    }

    /* ── Progress ── */
    .stProgress > div > div { background-color: #2563eb !important; border-radius: 4px; }
    .stProgress > div { background-color: #1e2d45 !important; border-radius: 4px; }

    /* ── Alerts ── */
    .stSuccess { background-color: rgba(16,185,129,0.1) !important; border-color: rgba(16,185,129,0.3) !important; }
    .stSuccess p { color: #34d399 !important; }
    .stError   { background-color: rgba(239,68,68,0.1) !important;  border-color: rgba(239,68,68,0.3) !important; }
    .stError p { color: #f87171 !important; }
    .stWarning { background-color: rgba(234,179,8,0.1) !important;  border-color: rgba(234,179,8,0.3) !important; }
    .stWarning p { color: #fbbf24 !important; }
    .stInfo    { background-color: rgba(37,99,235,0.1) !important;  border-color: rgba(37,99,235,0.3) !important; }
    .stInfo p  { color: #60a5fa !important; }

    /* ── General ── */
    h1,h2,h3,h4 { color: #f1f5f9 !important; font-family: 'Space Grotesk', sans-serif; }
    p, li, span { color: #cbd5e1; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 1.8rem;
        margin-top: 3rem;
        border-top: 1px solid #1e2d45;
    }
    .app-footer p { color: #475569 !important; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Core training logic (unchanged) ──────────────────────────────────────────

def train_and_save_model(ticker='AAPL', period='2y'):
    """Train and save models for a specific stock"""
    stock = yf.Ticker(ticker)
    data  = stock.history(period=period)

    if data.empty:
        return None, f"No data found for {ticker}"

    df = data.copy()
    df['Day']        = np.arange(len(df))
    df['MA_7']       = df['Close'].rolling(window=7).mean()
    df['MA_21']      = df['Close'].rolling(window=21).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df['Returns']    = df['Close'].pct_change()
    df = df.dropna()

    feature_cols = ['Day', 'MA_7', 'MA_21', 'Volatility', 'Returns']
    X = df[feature_cols].values
    y = df['Close'].values

    split_idx = int(len(X) * 0.8)
    X_train   = X[:split_idx]
    y_train   = y[:split_idx]

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, f'models/lr_model_{ticker}.pkl')
    joblib.dump(rf_model, f'models/rf_model_{ticker}.pkl')
    joblib.dump(scaler,   f'models/scaler_{ticker}.pkl')

    return {
        'ticker':      ticker,
        'samples':     len(df),
        'train_size':  split_idx,
        'features':    feature_cols,
        'last_price':  round(float(df['Close'].iloc[-1]), 2),
        'period':      period,
    }, None

# ── UI ────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="trainer-header">
    <h1 style="color:#ffffff !important;">🧠 Model Trainer</h1>
    <p>Pre-train and save ML models for fast predictions in YourStockPredict</p>
    <span class="header-badge">Linear Regression</span>
    <span class="header-badge">Random Forest</span>
    <span class="header-badge">StandardScaler</span>
    <span class="header-badge">joblib</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Training Config")
    st.markdown("---")

    PRESET = {
        'Apple Inc. (AAPL)':       'AAPL',
        'Microsoft Corp. (MSFT)':  'MSFT',
        'Google (GOOGL)':          'GOOGL',
        'Amazon.com (AMZN)':       'AMZN',
        'Tesla Inc. (TSLA)':       'TSLA',
        'NVIDIA Corp. (NVDA)':     'NVDA',
        'Meta Platforms (META)':   'META',
        'Netflix Inc. (NFLX)':     'NFLX',
    }

    selected_names = st.multiselect(
        "Stocks to Train",
        list(PRESET.keys()),
        default=['Apple Inc. (AAPL)', 'Google (GOOGL)', 'Tesla Inc. (TSLA)',
                 'Microsoft Corp. (MSFT)', 'Amazon.com (AMZN)']
    )

    custom_raw = st.text_input("Add Custom Tickers",
                               placeholder="e.g. NFLX, BABA, COIN",
                               help="Comma-separated tickers")

    period = st.selectbox("Training Period", ["1y", "2y", "3y", "5y"], index=1)

    st.markdown("---")
    st.markdown("**Output directory:** `./models/`")
    st.markdown("**Saved files per stock:**")
    st.markdown("- `lr_model_<TICKER>.pkl`")
    st.markdown("- `rf_model_<TICKER>.pkl`")
    st.markdown("- `scaler_<TICKER>.pkl`")
    st.markdown("---")

    run_btn = st.button("🚀 Start Training", use_container_width=True)

# Build ticker list
selected_tickers = [PRESET[n] for n in selected_names]
if custom_raw.strip():
    extras = [t.strip().upper() for t in custom_raw.split(',') if t.strip()]
    selected_tickers += extras
selected_tickers = list(dict.fromkeys(selected_tickers))  # dedup, preserve order

# ── Queue preview ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="section-title">Training Queue</div>', unsafe_allow_html=True)
    if selected_tickers:
        chips = " ".join(f'<span class="stock-chip">{t}</span>' for t in selected_tickers)
        st.markdown(f"""
        <div class="stock-queue">
            <div class="sq-title">📋 {len(selected_tickers)} stock(s) queued</div>
            {chips}
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No stocks selected. Pick from the sidebar.")

with col_right:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="info-card">
            <div class="card-label">Queue Size</div>
            <div class="card-value">{len(selected_tickers)}</div>
            <div class="card-sub">stocks</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="info-card">
            <div class="card-label">Models / Stock</div>
            <div class="card-value">2</div>
            <div class="card-sub">LR + RF</div>
        </div>""", unsafe_allow_html=True)

    total_files = len(selected_tickers) * 3   # lr + rf + scaler
    st.markdown(f"""<div class="info-card">
        <div class="card-label">Total Files to Save</div>
        <div class="card-value">{total_files}</div>
        <div class="card-sub">pkl files → ./models/</div>
    </div>""", unsafe_allow_html=True)

# ── Training run ──────────────────────────────────────────────────────────────
if run_btn:
    if not selected_tickers:
        st.warning("⚠️ Add at least one stock before training.")
    else:
        st.markdown('<div class="section-title">Training Log</div>', unsafe_allow_html=True)

        log_placeholder  = st.empty()
        prog_placeholder = st.empty()
        stat_placeholder = st.empty()

        logs        = []
        results_ok  = []
        results_err = []

        def render_log():
            lines = "".join(
                f'<div class="log-line {l["cls"]}">{l["text"]}</div>'
                for l in logs
            )
            log_placeholder.markdown(
                f'<div class="log-console">{lines}</div>',
                unsafe_allow_html=True
            )

        def add_log(text, cls="log-info"):
            ts = datetime.now().strftime("%H:%M:%S")
            logs.append({"text": f"[{ts}]  {text}", "cls": cls})
            render_log()

        add_log("═" * 52, "log-dim")
        add_log("  YourStockPredict — Model Trainer  v1.0", "log-info")
        add_log("═" * 52, "log-dim")
        add_log(f"Period : {period}   |   Queue : {len(selected_tickers)} stocks", "log-dim")
        add_log("", "log-dim")

        for idx, ticker in enumerate(selected_tickers):
            progress = (idx) / len(selected_tickers)
            prog_placeholder.progress(progress, text=f"Training {ticker}…")

            add_log(f"▶  Fetching  {ticker} …", "log-info")

            try:
                info, err = train_and_save_model(ticker, period)
                if err:
                    add_log(f"✗  {ticker} — {err}", "log-error")
                    results_err.append(ticker)
                else:
                    add_log(f"   Samples  : {info['samples']}  (train {info['train_size']})", "log-dim")
                    add_log(f"   Last px  : ${info['last_price']}", "log-dim")
                    add_log(f"✓  {ticker} — saved lr · rf · scaler", "log-success")
                    results_ok.append(info)
            except Exception as e:
                add_log(f"✗  {ticker} — {str(e)}", "log-error")
                results_err.append(ticker)

            add_log("", "log-dim")

        prog_placeholder.progress(1.0, text="Done!")

        add_log("═" * 52, "log-dim")
        add_log(f"  ✓ Success : {len(results_ok)}    ✗ Failed : {len(results_err)}", "log-success" if not results_err else "log-warn")
        add_log("═" * 52, "log-dim")

        # ── Summary cards ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="info-card">
                <div class="card-label">✅ Trained</div>
                <div class="card-value" style="color:#34d399 !important;">{len(results_ok)}</div>
                <div class="card-sub">models ready</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="info-card">
                <div class="card-label">❌ Failed</div>
                <div class="card-value" style="color:#f87171 !important;">{len(results_err)}</div>
                <div class="card-sub">check log above</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="info-card">
                <div class="card-label">💾 Files Saved</div>
                <div class="card-value">{len(results_ok)*3}</div>
                <div class="card-sub">in ./models/</div>
            </div>""", unsafe_allow_html=True)

        if results_ok:
            st.markdown('<div class="section-title">Trained Model Details</div>', unsafe_allow_html=True)
            df_out = pd.DataFrame(results_ok)[['ticker','period','samples','train_size','last_price']]
            df_out.columns = ['Ticker','Period','Total Samples','Train Samples','Last Price ($)']
            df_out.index   = range(1, len(df_out)+1)
            st.dataframe(df_out, use_container_width=True)

        if results_err:
            st.error(f"Failed tickers: {', '.join(results_err)}")

        if not results_err:
            st.success("🎉 All models trained and saved successfully!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>🧠 YourStockPredict — Model Trainer &nbsp;·&nbsp; Models saved to <code>./models/</code> &nbsp;·&nbsp; For educational purposes only</p>
</div>""", unsafe_allow_html=True)