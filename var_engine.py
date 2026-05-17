"""
═══════════════════════════════════════════════════════════════════════════════
 VaR ENGINE — Equity Risk Analytics Dashboard
═══════════════════════════════════════════════════════════════════════════════

A Streamlit dashboard for stock risk analysis using:
  • Monte Carlo Simulation
  • Value at Risk (VaR) & Expected Shortfall (ES)
  • Comparison of Standard Deviation vs Parkinson's Volatility

Run locally:
    streamlit run var_engine.py

Author: Built as an FRM portfolio project
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VaR Engine | Equity Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — modern fintech look
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        padding-top: 1rem;
    }
    
    /* Color palette */
    :root {
        --navy: #0B2545;
        --teal: #13796E;
        --coral: #E63946;
        --gold: #F4A261;
        --slate: #5C6B73;
        --sand: #F7F3E9;
    }
    
    /* Headers */
    h1 {
        color: #0B2545;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #0B2545;
        font-weight: 700;
    }
    
    /* KPI tile styling */
    [data-testid="stMetric"] {
        background-color: #F7F3E9;
        border: 2px solid #13796E;
        border-radius: 8px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        color: #5C6B73 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #0B2545 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0B2545;
    }
    
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Section divider */
    .section-divider {
        background: linear-gradient(90deg, #13796E 0%, #0B2545 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        margin: 25px 0 15px 0;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
    }
    
    /* Insight box */
    .insight-box {
        background-color: #E0F0EC;
        border-left: 5px solid #13796E;
        padding: 18px 22px;
        border-radius: 6px;
        margin: 15px 0;
        color: #0B2545;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .warning-box {
        background-color: #FCE4E6;
        border-left: 5px solid #E63946;
        padding: 18px 22px;
        border-radius: 6px;
        margin: 15px 0;
        color: #0B2545;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #13796E;
        color: white;
        font-weight: 700;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
    }
    
    .stButton button:hover {
        background-color: #0B2545;
        color: white;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 6px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLC data from Yahoo Finance via yfinance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                        progress=False, auto_adjust=False)
        if df.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_company_name(ticker: str) -> str:
    """Get full company name from ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info.get('longName', info.get('shortName', ticker))
    except:
        return ticker


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def parkinson_volatility(high: pd.Series, low: pd.Series) -> float:
    """Parkinson's daily volatility estimator."""
    return np.sqrt(np.mean((1 / (4 * np.log(2))) * (np.log(high / low)) ** 2))


def run_monte_carlo(mean: float, sigma: float, n_sim: int, seed: int = None) -> np.ndarray:
    """Generate N simulated returns from N(mean, sigma)."""
    if seed is not None:
        np.random.seed(seed)
    z = np.random.standard_normal(n_sim)
    return mean + sigma * z


def compute_var_es(returns: np.ndarray, confidence: float, investment: float) -> dict:
    """Compute VaR and ES at given confidence."""
    alpha = 1 - confidence
    var_pct = np.percentile(returns, alpha * 100)
    pnl = investment * (np.exp(returns) - 1)
    var_dollar = np.percentile(pnl, alpha * 100)
    es_dollar = pnl[pnl <= var_dollar].mean()
    return {
        'var_pct': var_pct,
        'var_dollar': var_dollar,
        'es_dollar': es_dollar,
    }


def fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"


def fmt_dollar(x: float) -> str:
    if x < 0:
        return f"-${abs(x):,.2f}"
    return f"${x:,.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# ⚙️ VaR Engine")
    st.markdown("---")
    
    st.markdown("### 📌 Stock Selection")
    ticker = st.text_input(
        "Ticker Symbol",
        value="TSLA",
        help="Examples: TSLA, AAPL, NVDA, RELIANCE.NS, ICICIBANK.NS, HDFCBANK.NS"
    ).upper().strip()
    
    st.markdown("**Quick Picks:**")
    col1, col2 = st.columns(2)
    quick_picks = {
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "NVDA": "NVIDIA",
        "RELIANCE.NS": "Reliance",
    }
    for i, (sym, name) in enumerate(quick_picks.items()):
        target_col = col1 if i % 2 == 0 else col2
        if target_col.button(name, key=f"qp_{sym}", use_container_width=True):
            ticker = sym
            st.session_state.ticker_override = sym
            st.rerun()
    
    if 'ticker_override' in st.session_state:
        ticker = st.session_state.ticker_override
        del st.session_state.ticker_override
    
    st.markdown("---")
    st.markdown("### 📅 Date Range")
    
    today = datetime.now().date()
    default_start = today - timedelta(days=730)  # ~2 years
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=today - timedelta(days=30),
    )
    end_date = st.date_input(
        "End Date",
        value=today,
        max_value=today,
    )
    
    st.markdown("---")
    st.markdown("### 💰 Position & Risk")
    
    investment = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=10_000_000,
        value=10_000,
        step=1000,
        format="%d",
    )
    
    confidence_pct = st.select_slider(
        "Confidence Level",
        options=[90, 95, 99],
        value=95,
        help="Probability that loss will NOT exceed VaR"
    )
    confidence = confidence_pct / 100
    
    n_sim = st.select_slider(
        "Monte Carlo Simulations",
        options=[1000, 5000, 10000, 25000],
        value=5000,
        help="More simulations = more stable estimates"
    )
    
    st.markdown("---")
    
    if st.button("🎲 Re-run Simulation", use_container_width=True,
                 help="Generate fresh random scenarios"):
        st.session_state.seed = np.random.randint(0, 1_000_000)
    
    if 'seed' not in st.session_state:
        st.session_state.seed = 42
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This dashboard estimates portfolio tail risk using Monte Carlo "
        "simulation, comparing two volatility estimators."
    )
    st.markdown(
        "**Data Source:** Yahoo Finance via `yfinance`  \n"
        "**Methods:** Std Dev vs Parkinson's Vol  \n"
        "**Metrics:** VaR, Expected Shortfall"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — TITLE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 📊 VaR Engine — Equity Risk Analytics Dashboard")
st.markdown(
    "<div style='color: #5C6B73; font-size: 1.05rem; margin-top: -10px;'>"
    "Monte Carlo · Value at Risk · Expected Shortfall · Parkinson's Volatility"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner(f"Fetching {ticker} data from Yahoo Finance..."):
    df = fetch_stock_data(ticker, str(start_date), str(end_date))
    company_name = get_company_name(ticker) if df is not None else ticker

if df is None or len(df) < 30:
    st.error(
        f"⚠️ Could not fetch sufficient data for `{ticker}`. "
        f"Check the ticker symbol or try a different date range. "
        f"For Indian stocks, use `.NS` suffix (e.g., `RELIANCE.NS`)."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

log_returns = compute_log_returns(df['Close'])
mean_return = log_returns.mean()
std_dev_daily = log_returns.std()
std_dev_annual = std_dev_daily * np.sqrt(252)

park_daily = parkinson_volatility(df['High'], df['Low'])
park_annual = park_daily * np.sqrt(252)

vol_ratio = park_daily / std_dev_daily
latest_close = float(df['Close'].iloc[-1])
first_close = float(df['Close'].iloc[0])
total_return = (latest_close / first_close - 1)

# Monte Carlo simulations
sim_std = run_monte_carlo(mean_return, std_dev_daily, n_sim, seed=st.session_state.seed)
sim_park = run_monte_carlo(mean_return, park_daily, n_sim, seed=st.session_state.seed + 1)

# VaR and ES at each confidence level
risk_metrics = {}
for cl in [0.90, 0.95, 0.99]:
    risk_metrics[cl] = {
        'std': compute_var_es(sim_std, cl, investment),
        'park': compute_var_es(sim_park, cl, investment),
    }

selected_metrics = risk_metrics[confidence]


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK HEADER
# ═══════════════════════════════════════════════════════════════════════════════

col_a, col_b = st.columns([3, 1])
with col_a:
    st.markdown(f"### {company_name} ({ticker})")
    st.markdown(
        f"<span style='color: #5C6B73;'>"
        f"{len(df)} trading days · "
        f"{df.index[0].strftime('%b %d, %Y')} → {df.index[-1].strftime('%b %d, %Y')}"
        f"</span>",
        unsafe_allow_html=True
    )
with col_b:
    delta_color = "normal" if total_return >= 0 else "inverse"
    st.metric(
        "Period Return",
        f"{total_return*100:+.1f}%",
        f"${latest_close - first_close:+,.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KPI TILES — TOP ROW
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-divider">KEY METRICS</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", fmt_dollar(latest_close))
c2.metric("Position Size", f"${investment:,.0f}")
c3.metric("StdDev (Annual)", fmt_pct(std_dev_annual, 1))
c4.metric("Parkinson (Annual)", fmt_pct(park_annual, 1))

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"VaR {confidence_pct}% — StdDev",
    fmt_dollar(selected_metrics['std']['var_dollar']),
    fmt_pct(selected_metrics['std']['var_pct']),
    delta_color="inverse",
)
c2.metric(
    f"VaR {confidence_pct}% — Parkinson",
    fmt_dollar(selected_metrics['park']['var_dollar']),
    fmt_pct(selected_metrics['park']['var_pct']),
    delta_color="inverse",
)
c3.metric(
    f"ES {confidence_pct}% — StdDev",
    fmt_dollar(selected_metrics['std']['es_dollar']),
    "Avg loss in worst tail",
    delta_color="inverse",
)
c4.metric(
    f"ES {confidence_pct}% — Parkinson",
    fmt_dollar(selected_metrics['park']['es_dollar']),
    "Avg loss in worst tail",
    delta_color="inverse",
)


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE HISTORY CHART
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-divider">PRICE HISTORY</div>', unsafe_allow_html=True)

fig_price = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("Closing Price", "Daily Intraday Range (High − Low)"),
)

# Close price with high-low band
fig_price.add_trace(
    go.Scatter(
        x=df.index, y=df['High'],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=1
)
fig_price.add_trace(
    go.Scatter(
        x=df.index, y=df['Low'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(19, 121, 110, 0.15)',
        name='Daily Range', hoverinfo='skip',
    ), row=1, col=1
)
fig_price.add_trace(
    go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', line=dict(color='#0B2545', width=2),
        name='Close',
        hovertemplate='%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>',
    ), row=1, col=1
)

# Intraday range
fig_price.add_trace(
    go.Bar(
        x=df.index, y=df['High'] - df['Low'],
        marker_color='#F4A261',
        name='Intraday Range',
        hovertemplate='%{x|%b %d, %Y}<br>Range: $%{y:.2f}<extra></extra>',
    ), row=2, col=1
)

fig_price.update_layout(
    height=520,
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=11, color='#0B2545'),
)
fig_price.update_xaxes(showgrid=True, gridcolor='#EEEEEE')
fig_price.update_yaxes(showgrid=True, gridcolor='#EEEEEE', title_text="Price ($)", row=1, col=1)
fig_price.update_yaxes(showgrid=True, gridcolor='#EEEEEE', title_text="Range ($)", row=2, col=1)

st.plotly_chart(fig_price, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-divider">VOLATILITY COMPARISON</div>', unsafe_allow_html=True)

vc1, vc2, vc3 = st.columns(3)

with vc1:
    st.markdown("#### Standard Deviation")
    st.markdown(
        f"""
        <div style='background: #F7F3E9; border: 2px solid #13796E; border-radius: 8px; padding: 18px;'>
            <div style='color: #5C6B73; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;'>Daily Vol</div>
            <div style='color: #0B2545; font-size: 1.8rem; font-weight: 800;'>{fmt_pct(std_dev_daily, 3)}</div>
            <div style='color: #5C6B73; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-top: 12px;'>Annualized</div>
            <div style='color: #0B2545; font-size: 1.8rem; font-weight: 800;'>{fmt_pct(std_dev_annual, 1)}</div>
            <div style='color: #5C6B73; font-size: 0.85rem; margin-top: 12px; font-style: italic;'>
                Uses only closing prices. Captures overnight gaps but misses intraday swings.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

with vc2:
    st.markdown("#### Parkinson's Volatility")
    st.markdown(
        f"""
        <div style='background: #F7F3E9; border: 2px solid #F4A261; border-radius: 8px; padding: 18px;'>
            <div style='color: #5C6B73; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;'>Daily Vol</div>
            <div style='color: #0B2545; font-size: 1.8rem; font-weight: 800;'>{fmt_pct(park_daily, 3)}</div>
            <div style='color: #5C6B73; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-top: 12px;'>Annualized</div>
            <div style='color: #0B2545; font-size: 1.8rem; font-weight: 800;'>{fmt_pct(park_annual, 1)}</div>
            <div style='color: #5C6B73; font-size: 0.85rem; margin-top: 12px; font-style: italic;'>
                Uses daily High-Low range. Captures intraday swings but blind to overnight gaps.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

with vc3:
    st.markdown("#### Ratio & Gap")
    ratio_color = '#E63946' if vol_ratio < 0.95 else '#13796E' if vol_ratio > 1.05 else '#5C6B73'
    st.markdown(
        f"""
        <div style='background: #F7F3E9; border: 2px solid #0B2545; border-radius: 8px; padding: 18px;'>
            <div style='color: #5C6B73; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;'>Parkinson ÷ StdDev</div>
            <div style='color: {ratio_color}; font-size: 1.8rem; font-weight: 800;'>{vol_ratio:.3f}x</div>
            <div style='color: #5C6B73; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-top: 12px;'>Annual Gap (Park − SD)</div>
            <div style='color: #0B2545; font-size: 1.8rem; font-weight: 800;'>{fmt_pct(park_annual - std_dev_annual, 2)}</div>
            <div style='color: #5C6B73; font-size: 0.85rem; margin-top: 12px; font-style: italic;'>
                {"Parkinson lower → overnight gaps dominate" if vol_ratio < 1 else "Parkinson higher → intraday swings dominate"}
            </div>
        </div>
        """, unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="section-divider">MONTE CARLO DISTRIBUTION — {n_sim:,} scenarios per method</div>',
    unsafe_allow_html=True
)

# Build distribution comparison chart
larger_sigma = max(std_dev_daily, park_daily)
x_range = np.linspace(-4 * larger_sigma, 4 * larger_sigma, 200)
theoretical_pdf_std = stats.norm.pdf(x_range, mean_return, std_dev_daily)
theoretical_pdf_park = stats.norm.pdf(x_range, mean_return, park_daily)

# Compute histogram bins manually for consistent overlay
bins = np.linspace(-4 * larger_sigma, 4 * larger_sigma, 51)
hist_std, edges = np.histogram(sim_std, bins=bins, density=True)
hist_park, _ = np.histogram(sim_park, bins=bins, density=True)
bin_centers = (edges[:-1] + edges[1:]) / 2

fig_dist = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"Monte Carlo: Standard Deviation (σ = {fmt_pct(std_dev_daily, 3)})",
        f"Monte Carlo: Parkinson's (σ = {fmt_pct(park_daily, 3)})",
    ),
    horizontal_spacing=0.10,
)

# StdDev panel
fig_dist.add_trace(
    go.Bar(
        x=bin_centers, y=hist_std,
        marker_color='#13796E', opacity=0.75,
        name='Simulated', showlegend=True,
    ), row=1, col=1
)
fig_dist.add_trace(
    go.Scatter(
        x=x_range, y=theoretical_pdf_std,
        mode='lines', line=dict(color='#0B2545', width=3, dash='dash'),
        name='Theoretical Normal', showlegend=True,
    ), row=1, col=1
)

# VaR & ES lines for StdDev
var_std = selected_metrics['std']['var_pct']
es_std_pct = np.log(1 + selected_metrics['std']['es_dollar'] / investment)
fig_dist.add_vline(
    x=var_std, line_color='#E63946', line_width=2, line_dash='solid',
    annotation_text=f"VaR {confidence_pct}%",
    annotation_position="top", row=1, col=1
)
fig_dist.add_vline(
    x=es_std_pct, line_color='#8B0000', line_width=2, line_dash='dot',
    annotation_text=f"ES {confidence_pct}%",
    annotation_position="bottom", row=1, col=1
)

# Parkinson panel
fig_dist.add_trace(
    go.Bar(
        x=bin_centers, y=hist_park,
        marker_color='#F4A261', opacity=0.75,
        name='Simulated', showlegend=False,
    ), row=1, col=2
)
fig_dist.add_trace(
    go.Scatter(
        x=x_range, y=theoretical_pdf_park,
        mode='lines', line=dict(color='#0B2545', width=3, dash='dash'),
        name='Theoretical Normal', showlegend=False,
    ), row=1, col=2
)

var_park = selected_metrics['park']['var_pct']
es_park_pct = np.log(1 + selected_metrics['park']['es_dollar'] / investment)
fig_dist.add_vline(
    x=var_park, line_color='#E63946', line_width=2, line_dash='solid',
    annotation_text=f"VaR {confidence_pct}%",
    annotation_position="top", row=1, col=2
)
fig_dist.add_vline(
    x=es_park_pct, line_color='#8B0000', line_width=2, line_dash='dot',
    annotation_text=f"ES {confidence_pct}%",
    annotation_position="bottom", row=1, col=2
)

fig_dist.update_layout(
    height=480,
    bargap=0.05,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=11, color='#0B2545'),
    legend=dict(
        orientation='h', yanchor='bottom', y=-0.18,
        xanchor='center', x=0.25,
    ),
    margin=dict(l=10, r=10, t=60, b=60),
)
fig_dist.update_xaxes(
    title_text="Daily Return", tickformat='.1%',
    showgrid=True, gridcolor='#EEEEEE',
)
fig_dist.update_yaxes(title_text="Density", showgrid=True, gridcolor='#EEEEEE')

st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RISK METRICS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-divider">RISK METRICS — ALL CONFIDENCE LEVELS</div>', unsafe_allow_html=True)

rows = []
for cl in [0.90, 0.95, 0.99]:
    sd = risk_metrics[cl]['std']
    pk = risk_metrics[cl]['park']
    rows.append({
        'Confidence': f"{int(cl*100)}%",
        'VaR % (StdDev)': fmt_pct(sd['var_pct']),
        'VaR $ (StdDev)': fmt_dollar(sd['var_dollar']),
        'ES $ (StdDev)': fmt_dollar(sd['es_dollar']),
        'VaR % (Parkinson)': fmt_pct(pk['var_pct']),
        'VaR $ (Parkinson)': fmt_dollar(pk['var_dollar']),
        'ES $ (Parkinson)': fmt_dollar(pk['es_dollar']),
        'Gap ($)': fmt_dollar(pk['var_dollar'] - sd['var_dollar']),
    })

risk_df = pd.DataFrame(rows)
st.dataframe(risk_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATED INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-divider">INTERPRETATION</div>', unsafe_allow_html=True)

# Generate rules-based interpretation
park_vs_std_pct = (park_daily / std_dev_daily - 1) * 100

if vol_ratio < 0.90:
    interpretation = f"""
    <strong>⚠️ Parkinson's volatility is {abs(park_vs_std_pct):.1f}% LOWER than close-to-close Std Dev.</strong>
    This is the opposite of the textbook expectation and indicates that {ticker}'s risk is dominated by 
    <strong>overnight gaps</strong> rather than intraday swings. Parkinson's estimator ignores price moves 
    between yesterday's close and today's open — for stocks where earnings reactions, macro news, or 
    after-hours announcements drive most price action (typical for high-beta names like {ticker}), 
    Parkinson systematically <strong>underestimates</strong> true risk. The Std Dev VaR of 
    <strong>{fmt_dollar(selected_metrics['std']['var_dollar'])}</strong> at {confidence_pct}% confidence 
    is the more honest number for this position.
    """
    box_class = "warning-box"
elif vol_ratio > 1.10:
    interpretation = f"""
    <strong>📈 Parkinson's volatility is {park_vs_std_pct:.1f}% HIGHER than close-to-close Std Dev.</strong>
    This matches the textbook expectation: Parkinson's High-Low estimator captures intraday price swings 
    that closing prices flatten out. For {ticker}, the intraday range carries more information than the 
    close-to-close path, suggesting risk lives during the trading day rather than overnight. The Parkinson 
    VaR of <strong>{fmt_dollar(selected_metrics['park']['var_dollar'])}</strong> at {confidence_pct}% confidence 
    is the more conservative — and likely more realistic — risk estimate.
    """
    box_class = "insight-box"
else:
    interpretation = f"""
    <strong>≈ Parkinson's volatility is approximately equal to Std Dev ({vol_ratio:.2f}x ratio).</strong>
    For {ticker}, overnight gaps and intraday swings contribute roughly equally to total volatility. 
    This is typical for steady, less news-driven stocks. Either estimator gives a similar risk picture: 
    VaR at {confidence_pct}% confidence is approximately 
    <strong>{fmt_dollar((selected_metrics['std']['var_dollar'] + selected_metrics['park']['var_dollar'])/2)}</strong> 
    on a {fmt_dollar(investment)} position.
    """
    box_class = "insight-box"

st.markdown(f'<div class="{box_class}">{interpretation}</div>', unsafe_allow_html=True)

# Generic explanation of metrics
st.markdown(
    f"""
    <div class="insight-box">
    <strong>How to read these numbers:</strong> A VaR of {fmt_dollar(selected_metrics['std']['var_dollar'])} at 
    {confidence_pct}% confidence means there is a {100-confidence_pct}% probability of losing MORE than this on a 
    typical day. Expected Shortfall ({fmt_dollar(selected_metrics['std']['es_dollar'])}) is the AVERAGE loss 
    in those worst {100-confidence_pct}% of cases — capturing the severity of tail events, not just their frequency. 
    The historical observation period was {len(df)} trading days; the {n_sim:,} simulations assume returns follow 
    a normal distribution with the observed mean and chosen volatility.
    </div>
    """,
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY EXPANDER
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📚 Methodology Notes"):
    st.markdown("""
    ### Log Returns
    Computed as ln(P_t / P_{t-1}). Preferred over simple returns because they are additive over time and 
    approximately normally distributed.
    
    ### Standard Deviation (Close-to-Close)
    Sample standard deviation of daily log returns. Captures total day-to-day price variability including 
    overnight gaps but ignores within-day swings.
    
    ### Parkinson's Volatility (1980)
    Estimates volatility from daily High-Low range:
    
    $$\\sigma_P = \\sqrt{\\frac{1}{4 \\ln 2} \\cdot \\frac{1}{n}\\sum_{i=1}^{n} \\left[\\ln\\left(\\frac{H_i}{L_i}\\right)\\right]^2}$$
    
    Approximately 5× more efficient than close-to-close StdDev under continuous Brownian motion. 
    Major limitation: assumes no overnight gaps and no drift.
    
    ### Monte Carlo Simulation
    Generates N random returns from N(μ, σ²) where μ is the historical mean and σ is the chosen volatility. 
    Simulated prices follow geometric Brownian motion: P_sim = P_0 · exp(simulated_return).
    
    ### Value at Risk (VaR)
    The α-th percentile of the simulated P&L distribution, where α = 1 - confidence. A 95% VaR is the 5th 
    percentile worst outcome.
    
    ### Expected Shortfall (ES / CVaR)
    The average loss conditional on losing more than VaR. Captures the severity of tail events beyond the 
    VaR threshold.
    
    ### Limitations
    - **Normal distribution assumption** underestimates fat tails (real returns have kurtosis > 3)
    - **Static volatility** ignores volatility clustering (GARCH effects)
    - **No correlation modeling** — single-asset only
    - **Parkinson ignores overnight gaps** — addressed by Garman-Klass (1980), Rogers-Satchell (1991), 
      Yang-Zhang (2000)
    
    ### Possible Extensions
    - Student's t-distribution for fatter tails
    - GARCH(1,1) for time-varying volatility
    - Multi-asset portfolio with Cholesky decomposition for correlated returns
    - Backtesting VaR breaches against actual realized losses
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("🗂️ View Raw Historical Data"):
    df_display = df.copy()
    df_display['Log Return'] = np.log(df_display['Close'] / df_display['Close'].shift(1))
    df_display['HL Range'] = df_display['High'] - df_display['Low']
    df_display = df_display.sort_index(ascending=False)
    
    df_display['Open'] = df_display['Open'].apply(lambda x: f"${x:,.2f}")
    df_display['High'] = df_display['High'].apply(lambda x: f"${x:,.2f}")
    df_display['Low'] = df_display['Low'].apply(lambda x: f"${x:,.2f}")
    df_display['Close'] = df_display['Close'].apply(lambda x: f"${x:,.2f}")
    df_display['HL Range'] = df_display['HL Range'].apply(lambda x: f"${x:,.2f}")
    df_display['Log Return'] = df_display['Log Return'].apply(
        lambda x: f"{x*100:+.4f}%" if pd.notna(x) else ""
    )
    
    st.dataframe(df_display, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #5C6B73; font-size: 0.85rem; padding: 20px 0;'>
        Built with Streamlit · Data from Yahoo Finance · An FRM portfolio project<br>
        <em>This dashboard is for educational purposes only and does not constitute financial advice.</em>
    </div>
    """,
    unsafe_allow_html=True
)
