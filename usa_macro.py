import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
import os
from dotenv import load_dotenv
import datetime
import io
import time
import re
import requests
import calendar
import math
import yfinance as yf

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

st.set_page_config(page_title="US Macro Dashboard", layout="wide", page_icon="🇺🇸")

st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #161b26; border: 1px solid #2a2f3e;
    border-radius: 8px; padding: 12px 16px; text-align: center;
  }
  .metric-label { font-size: 0.68rem; color: #8a94a6; letter-spacing: 0.08em;
                  text-transform: uppercase; margin-bottom: 3px; }
  .metric-value { font-size: 1.25rem; font-weight: 700; }
  .metric-delta { font-size: 0.72rem; margin-top: 2px; }
  .positive { color: #26a69a; }
  .negative { color: #ef5350; }
  .neutral  { color: #e0e0e0; }
  .section-header {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8a94a6; margin: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #2a2f3e; padding-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

TEMPLATE   = "plotly_dark"
PAPER_BG   = "#0e1117"
PLOT_BG    = "#161b26"
GRID_COLOR = "#2a2f3e"
RECESSION_COLOR = "rgba(180,60,60,0.12)"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch(series_id: str, label: str, start: str, end: str = None) -> pd.DataFrame:
    for attempt in range(3):
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            df = pd.DataFrame({label: s.values}, index=pd.to_datetime(s.index))
            df.index.name = "date"
            time.sleep(0.15)
            return df
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate Limit" in str(e):
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                continue
            st.warning(f"Could not load {label} ({series_id}): {e}")
            return pd.DataFrame()
    st.warning(f"Rate limit: could not load {label} ({series_id}) after 3 attempts")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_recessions(start: str, end: str) -> list:
    """Return list of (start, end) tuples for NBER recessions."""
    rec = fetch("USREC", "USREC", start, end)
    if rec.empty:
        return []
    periods, in_rec, rec_start = [], False, None
    for date, val in rec["USREC"].items():
        if val == 1 and not in_rec:
            in_rec, rec_start = True, date
        elif val == 0 and in_rec:
            periods.append((rec_start, date))
            in_rec = False
    if in_rec:
        periods.append((rec_start, rec.index[-1]))
    return periods

def add_recessions(fig, recessions, rows=None, cols=None):
    """Shade recession bands on a figure."""
    for r_start, r_end in recessions:
        if rows and cols:
            for row, col in zip(rows, cols):
                fig.add_vrect(x0=r_start, x1=r_end, fillcolor=RECESSION_COLOR,
                              layer="below", line_width=0, row=row, col=col)
        else:
            fig.add_vrect(x0=r_start, x1=r_end, fillcolor=RECESSION_COLOR,
                          layer="below", line_width=0)
    return fig

def base_layout(title="", height=480):
    return dict(
        template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        height=height,
        margin=dict(l=50, r=50, t=45, b=30),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )

def dual_axis_layout(title, y1_title, y2_title, height=480):
    layout = base_layout(title, height)
    layout.update(
        yaxis =dict(title=y1_title, gridcolor=GRID_COLOR),
        yaxis2=dict(title=y2_title, overlaying="y", side="right", gridcolor=GRID_COLOR),
    )
    return layout

def mom_yoy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[f"{col} MoM %", f"{col} YoY %"])
    out = pd.DataFrame(index=df.index)
    out[f"{col} MoM %"] = df[col].pct_change() * 100
    out[f"{col} YoY %"] = df[col].pct_change(12) * 100
    return out

def nfp_change(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[f"{col} MoM Change (k)"])
    out = pd.DataFrame(index=df.index)
    out[f"{col} MoM Change (k)"] = df[col].diff()
    return out

def csv_download(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    st.download_button(f"⬇ CSV", buf, file_name=f"{label}.csv",
                       mime="text/csv", key=f"dl_{label}_{id(df)}")

def render_two_col(charts):
    """Render (title, fig [, df]) tuples in 2-column layout."""
    n, i = len(charts), 0
    while i < n:
        if i == n - 1 and n % 2 != 0:
            item = charts[i]
            st.plotly_chart(item[1], use_container_width=True)
            if len(item) > 2 and item[2] is not None:
                csv_download(item[2], item[0])
            i += 1
        else:
            c1, c2 = st.columns(2)
            for col, item in [(c1, charts[i]), (c2, charts[i+1])]:
                with col:
                    st.plotly_chart(item[1], use_container_width=True)
                    if len(item) > 2 and item[2] is not None:
                        csv_download(item[2], item[0])
            i += 2

# ── Fed Funds implied-probability (WIRP/FedWatch-style) ────────────────────────
# Replicates CME's published Fed Funds futures methodology (not a scrape of CME's own
# tool - there's no public API for that). ZQ = CME 30-Day Fed Funds futures; a contract's
# price implies the average daily effective rate for its whole delivery month.
#
# Reading a meeting's OWN contract month requires solving a days-weighted split between
# the pre- and post-meeting rate, which is numerically unstable whenever the meeting falls
# close to month-end (a small "days after" denominator amplifies ordinary price noise into
# huge implied-rate swings - confirmed by inspecting raw prices, which are smooth; the
# instability is in that formula, not the data). Since FOMC meetings are always 5+ weeks
# apart, the calendar month right after any meeting is essentially always meeting-free, so
# its whole-month average price directly *is* the implied post-meeting rate - no division,
# no instability, and each meeting is read independently instead of chained (so one bad
# read can't cascade into every later meeting).

FOMC_MONTH_CODE = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
RATE_STEP = 0.25  # standard FOMC move size, in percentage points (A.R.M. in Bloomberg's WIRP)

# Every FOMC decision date (2nd day of each 2-day meeting) the Fed has published so far.
# The Fed only confirms each date at the meeting immediately prior - there's no formula
# for these, so this table has to be refreshed by hand from
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm once today's date gets
# within ~2 years of the last entry below (currently covers through Dec 2027).
FOMC_ALL_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

def _zq_ticker(date):
    return f"ZQ{FOMC_MONTH_CODE[date.month]}{date.strftime('%y')}.CBT"

def _month_after(date):
    return pd.Timestamp(year=date.year + (date.month == 12), month=1 if date.month == 12 else date.month + 1, day=1)

def _zq_price(ticker):
    hist = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1].iloc[0] if hasattr(hist["Close"].iloc[-1], "iloc") else hist["Close"].iloc[-1])

def _zq_close_series(ticker, period="4mo"):
    hist = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if hist.empty:
        return None
    close = hist["Close"]
    return close.iloc[:, 0] if hasattr(close, "columns") else close

SNAPSHOT_OFFSETS = {"Today": 0, "1W Ago": -5, "1M Ago": -21, "3M Ago": -63}
SNAPSHOT_COLORS = {"Today": "cyan", "1W Ago": "orange", "1M Ago": "green", "3M Ago": "magenta"}

@st.cache_data(ttl=21600)
def get_fedwatch_history(years_ahead=2):
    # Same meetings/clean-month read as get_fedwatch_probabilities, but pulls each
    # contract's own price history to snapshot the implied curve at past points in time -
    # i.e. what the market was pricing for these same meetings, as of each snapshot date.
    today = pd.Timestamp.today().normalize()
    window_end = today + pd.DateOffset(years=years_ahead)
    meetings = [pd.Timestamp(d) for d in FOMC_ALL_DATES if today <= pd.Timestamp(d) <= window_end]

    rows = []
    for meeting in meetings:
        series = _zq_close_series(_zq_ticker(_month_after(meeting)))
        row = {"Meeting": meeting.strftime("%Y-%m-%d")}
        if series is not None:
            for label, offset in SNAPSHOT_OFFSETS.items():
                idx = len(series) - 1 + offset
                row[label] = 100 - float(series.iloc[idx]) if idx >= 0 else None
        rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data(ttl=21600)
def get_fedwatch_probabilities(years_ahead=2):
    effr = fred.get_series("EFFR").dropna()
    current_rate = float(effr.iloc[-1])

    today = pd.Timestamp.today().normalize()
    window_end = today + pd.DateOffset(years=years_ahead)
    meetings = [pd.Timestamp(d) for d in FOMC_ALL_DATES if today <= pd.Timestamp(d) <= window_end]

    implied = {}  # meeting -> implied post-meeting rate, computed independently per meeting
    for meeting in meetings:
        clean_month = _month_after(meeting)
        price = _zq_price(_zq_ticker(clean_month))
        if price is None:
            # fall back to the meeting's own contract month (day-weighted solve) only if the
            # cleaner next-month contract isn't available
            price = _zq_price(_zq_ticker(meeting))
            if price is None:
                continue
            days_in_month = calendar.monthrange(meeting.year, meeting.month)[1]
            days_before, days_after = meeting.day, days_in_month - meeting.day
            prior = implied.get(meetings[meetings.index(meeting) - 1], current_rate) if meeting != meetings[0] else current_rate
            avg_rate = 100 - price
            implied[meeting] = avg_rate if days_after <= 0 else (avg_rate * days_in_month - days_before * prior) / days_after
        else:
            implied[meeting] = 100 - price

    rows = []
    prior_rate = current_rate
    for meeting in meetings:
        if meeting not in implied:
            continue
        implied_rate = implied[meeting]

        # This-meeting-only move, relative to the previous meeting's implied rate
        # (Bloomberg's "%Hike/Cut") ...
        meeting_delta_bps = (implied_rate - prior_rate) * 100
        step_bps = int(RATE_STEP * 100)
        lower = math.floor(meeting_delta_bps / step_bps) * step_bps
        upper = lower + step_bps
        frac_upper = 0.0 if upper == lower else min(max((meeting_delta_bps - lower) / step_bps, 0), 1)
        probs = {lower: 1 - frac_upper} if frac_upper == 0 else {lower: 1 - frac_upper, upper: frac_upper}

        # ... vs. cumulative move from today (Bloomberg's "Imp. Rate Δ" / "#Hikes/Cuts")
        cum_delta = implied_rate - current_rate

        rows.append({
            "Meeting": meeting.strftime("%Y-%m-%d"), "Implied Rate": implied_rate,
            "Imp. Rate Delta": cum_delta, "Hikes/Cuts": cum_delta / RATE_STEP,
            "This-Meeting Move (bps)": meeting_delta_bps, "Probabilities": probs,
        })
        prior_rate = implied_rate  # this meeting's own (independently-read) rate anchors the next delta

    return pd.DataFrame(rows), current_rate

# ── Treasury auctions (issuance / maturity / outstanding / bid-to-cover) ───────

AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

def _get_json(url, retries=5, backoff=2, timeout=30):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return r.json()
        except requests.exceptions.RequestException:
            pass
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Treasury API failed after {retries} retries: {url}")

def _download_all_auctions():
    data, result = [], _get_json(f"{AUCTIONS_URL}?filter=record_date:gt:1900-01-01&page[size]=10000")
    data.extend(result["data"])
    while result["links"]["next"] is not None:
        # next link is "&page[number]=N..." with no leading "?" - must not be
        # concatenated onto a bare url or it 404s past page 1.
        result = _get_json(f'{AUCTIONS_URL}?{result["links"]["next"].lstrip("&")}')
        data.extend(result["data"])
        time.sleep(0.2)
    return pd.DataFrame(data)

@st.cache_data(ttl=21600)  # Treasury auction calendar only changes a few times/week
def load_auctions_data():
    df = _download_all_auctions()
    for col in ["issue_date", "auction_date", "maturity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["offering_amt", "total_accepted", "bid_to_cover_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _parse_tenor(term):
    num = re.findall(r"\d+\.?\d*", term)
    unit = "W" if "Week" in term else "M" if "Month" in term else "Y"
    return (float(num[0]) if num else 0), unit

def _sort_by_tenor(df, col):
    df = df.copy()
    parsed = df[col].apply(_parse_tenor)
    df["_wk"] = [v * (1 if u == "W" else 4 if u == "M" else 52) for v, u in parsed]
    return df.sort_values("_wk").drop(columns="_wk")

def get_upcoming_issuances(df, days_ahead):
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    upcoming = df[(df["issue_date"] >= today) & (df["issue_date"] <= cutoff)].copy()
    if upcoming.empty:
        return upcoming, pd.DataFrame(columns=["security_term_week_year", "Total Issuance (Billion $)"])
    upcoming["offering_amt_bil"] = upcoming["offering_amt"] / 1e9
    upcoming = upcoming[["auction_date", "issue_date", "security_type", "security_term_week_year", "cusip", "offering_amt_bil"]].sort_values("issue_date")
    summary = upcoming.groupby("security_term_week_year")["offering_amt_bil"].sum().reset_index()
    summary = summary.rename(columns={"offering_amt_bil": "Total Issuance (Billion $)"})
    return upcoming, _sort_by_tenor(summary, "security_term_week_year")

def get_maturing_treasuries(df, days_ahead):
    # Amount of previously-issued debt whose maturity_date falls in the same forward
    # window - i.e. what needs to be rolled over/refinanced alongside new issuance.
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    maturing = df[(df["maturity_date"] >= today) & (df["maturity_date"] <= cutoff)].copy()
    if maturing.empty:
        return maturing, pd.DataFrame(columns=["security_term_week_year", "Total Maturing (Billion $)"])
    maturing["maturing_amt_bil"] = maturing["total_accepted"].fillna(maturing["offering_amt"]) / 1e9
    maturing_out = maturing[["maturity_date", "security_type", "security_term_week_year", "cusip", "maturing_amt_bil"]].sort_values("maturity_date")
    summary = maturing.groupby("security_term_week_year")["maturing_amt_bil"].sum().reset_index()
    summary = summary.rename(columns={"maturing_amt_bil": "Total Maturing (Billion $)"})
    return maturing_out, _sort_by_tenor(summary, "security_term_week_year")

def get_outstanding_by_tenor(df):
    # Still-alive securities (issued in the past, not yet matured) summed by original
    # tenor - reopenings included, so it's the full marketable outstanding balance.
    # Excludes intragovernmental holdings and non-marketable debt (savings bonds, SLGS),
    # so this will run below Total Public Debt Outstanding - that's expected.
    today = pd.Timestamp.today().normalize()
    outstanding = df[(df["issue_date"] <= today) & (df["maturity_date"] > today)].copy()
    outstanding["amt_bil"] = outstanding["total_accepted"].fillna(outstanding["offering_amt"]) / 1e9
    summary = outstanding.groupby("security_term_week_year")["amt_bil"].sum().reset_index()
    summary = summary.rename(columns={"amt_bil": "Outstanding (Billion $)"})
    return outstanding, _sort_by_tenor(summary, "security_term_week_year")

# ── Date range ────────────────────────────────────────────────────────────────
st.title("🇺🇸 US Macro Dashboard")

col_d1, col_d2 = st.columns([3,1])
with col_d1:
    date_range = st.slider(
        "Date Range", min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today(),
        value=(datetime.date.today().replace(year=datetime.date.today().year - 5), datetime.date.today()),
        format="YYYY-MM-DD"
    )
START = date_range[0].strftime("%Y-%m-%d")
END   = date_range[1].strftime("%Y-%m-%d")

recessions = fetch_recessions(START, END)

# ── Summary bar ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Latest Readings</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_summary_metrics(end):
    metrics = {
        "CPI YoY":      ("CPIAUCSL",   "pct_yoy",  "%"),
        "Core CPI YoY": ("CPILFESL",   "pct_yoy",  "%"),
        "PCE YoY":      ("PCEPI",      "pct_yoy",  "%"),
        "Unemp":        ("UNRATE",     "level",     "%"),
        "NFP MoM":      ("PAYEMS",     "diff_k",    "k"),
        "10Y Yield":    ("DGS10",      "level",     "%"),
        "2Y Yield":     ("DGS2",       "level",     "%"),
        "2s10s":        ("T10Y2Y",     "level_bps","bps"),
        "Fed Funds":    ("FEDFUNDS",   "level",     "%"),
        "M2 YoY":       ("M2SL",       "pct_yoy",  "%"),
    }
    results = {}
    for name, (sid, calc, unit) in metrics.items():
        try:
            s = fred.get_series(sid)
            s = s.dropna()
            latest = float(s.iloc[-1])
            prev   = float(s.iloc[-2]) if len(s) > 1 else latest
            if calc == "pct_yoy":
                val = s.pct_change(12).iloc[-1] * 100
                delta = val - s.pct_change(12).iloc[-2] * 100
            elif calc == "diff_k":
                val = s.diff().iloc[-1]
                delta = val - s.diff().iloc[-2]
            elif calc == "level_bps":
                # FRED reports this series in percentage points (e.g. 0.51 = 51bps)
                val, delta = latest * 100, (latest - prev) * 100
            else:
                val, delta = latest, latest - prev
            results[name] = (val, delta, unit)
        except:
            results[name] = (None, None, "")
    return results

with st.spinner("Loading summary metrics…"):
    summary = get_summary_metrics(END)

cols = st.columns(len(summary))
for col, (name, (val, delta, unit)) in zip(cols, summary.items()):
    with col:
        if val is None:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{name}</div><div class="metric-value neutral">N/A</div></div>', unsafe_allow_html=True)
            continue
        val_str   = f"{val:+.0f}{unit}" if unit in ("k", "bps") else f"{val:.2f}{unit}"
        if delta is None:
            delta_str = ""
        else:
            delta_str = f"{delta:+.0f}{unit}" if unit in ("k", "bps") else f"{delta:+.2f}{unit}"
        delta_cls = "positive" if (delta or 0) > 0 else "negative" if (delta or 0) < 0 else "neutral"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{name}</div>
          <div class="metric-value neutral">{val_str}</div>
          <div class="metric-delta {delta_cls}">{delta_str} MoM</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Inflation & Consumer",
    "👷 Labour Market",
    "🏠 Housing",
    "💵 Monetary & Rates",
    "🔭 Leading Indicators",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Inflation & Consumer
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Inflation & Consumer")
    with st.spinner("Loading inflation data…"):
        cpi      = mom_yoy(fetch("CPIAUCSL", "CPI", START, END), "CPI")
        core_cpi = mom_yoy(fetch("CPILFESL", "Core CPI", START, END), "Core CPI")
        pce      = mom_yoy(fetch("PCEPI", "PCE", START, END), "PCE")
        core_pce = mom_yoy(fetch("PCEPILFE", "Core PCE", START, END), "Core PCE")
        ppi      = mom_yoy(fetch("PPIACO", "PPI", START, END), "PPI")
        rsa      = mom_yoy(fetch("RSAFS", "Retail Sales", START, END), "Retail Sales")
        exp_pi   = mom_yoy(fetch("IQ", "Export Price Index", START, END), "Export Price Index")
        imp_pi   = mom_yoy(fetch("IR", "Import Price Index", START, END), "Import Price Index")
        be_5y    = fetch("T5YIE",  "5Y Breakeven", START, END)
        be_10y   = fetch("T10YIE", "10Y Breakeven", START, END)
        umich    = fetch("UMCSENT", "UMich Sentiment", START, END)
        inf_exp1 = fetch("MICH", "1Y Inf Expectation", START, END)
        inf_exp5 = fetch("EXPINF5YR", "5Y Inf Expectation", START, END)

    # CPI vs Core CPI
    fig_cpi = go.Figure()
    for col, color in [("CPI YoY %","#ef5350"),("Core CPI YoY %","#ff9800"),
                       ("CPI MoM %","#ef535055"),("Core CPI MoM %","#ff980055")]:
        src = pd.concat([cpi, core_cpi], axis=1)
        if col not in src.columns: continue
        ax = "y2" if "MoM" in col else "y"
        fig_cpi.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines",
                                     yaxis=ax, line=dict(width=1.5 if "YoY" in col else 1, dash="solid" if "YoY" in col else "dot")))
    fig_cpi.update_layout(**dual_axis_layout("CPI vs Core CPI", "YoY %", "MoM %"))
    add_recessions(fig_cpi, recessions)

    # PCE vs Core PCE
    fig_pce = go.Figure()
    for col, color in [("PCE YoY %","#26a69a"),("Core PCE YoY %","#80cbc4"),
                       ("PCE MoM %","#26a69a55"),("Core PCE MoM %","#80cbc455")]:
        src = pd.concat([pce, core_pce], axis=1)
        if col not in src.columns: continue
        ax = "y2" if "MoM" in col else "y"
        fig_pce.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines",
                                     yaxis=ax, line=dict(width=1.5 if "YoY" in col else 1, dash="solid" if "YoY" in col else "dot")))
    fig_pce.update_layout(**dual_axis_layout("PCE vs Core PCE (Fed's Preferred)", "YoY %", "MoM %"))
    add_recessions(fig_pce, recessions)

    # PPI
    fig_ppi = go.Figure()
    for col in ppi.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_ppi.add_trace(go.Scatter(x=ppi.index, y=ppi[col], name=col, mode="lines", yaxis=ax))
    fig_ppi.update_layout(**dual_axis_layout("PPI (MoM & YoY)", "YoY %", "MoM %"))
    add_recessions(fig_ppi, recessions)

    # Retail Sales
    fig_rsa = go.Figure()
    for col in rsa.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_rsa.add_trace(go.Scatter(x=rsa.index, y=rsa[col], name=col, mode="lines", yaxis=ax))
    fig_rsa.update_layout(**dual_axis_layout("Retail Sales (MoM & YoY)", "YoY %", "MoM %"))
    add_recessions(fig_rsa, recessions)

    # Breakeven inflation
    fig_be = go.Figure()
    if not be_5y.empty:
        fig_be.add_trace(go.Scatter(x=be_5y.index, y=be_5y["5Y Breakeven"], name="5Y Breakeven", line=dict(color="#26a69a")))
    if not be_10y.empty:
        fig_be.add_trace(go.Scatter(x=be_10y.index, y=be_10y["10Y Breakeven"], name="10Y Breakeven", line=dict(color="#ff9800")))
    fig_be.update_layout(**base_layout("Market-Implied Inflation Expectations (TIPS Breakevens)"))
    add_recessions(fig_be, recessions)

    # UMich Sentiment + Inflation Expectations
    fig_umich = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Consumer Sentiment", "Inflation Expectations"),
                              vertical_spacing=0.1)
    if not umich.empty:
        fig_umich.add_trace(go.Scatter(x=umich.index, y=umich["UMich Sentiment"],
                                       name="UMich Sentiment", line=dict(color="#90a4d4")), row=1, col=1)
    if not inf_exp1.empty:
        fig_umich.add_trace(go.Scatter(x=inf_exp1.index, y=inf_exp1["1Y Inf Expectation"],
                                       name="1Y Inf Exp", line=dict(color="#ef5350")), row=2, col=1)
    if not inf_exp5.empty:
        fig_umich.add_trace(go.Scatter(x=inf_exp5.index, y=inf_exp5["5Y Inf Expectation"],
                                       name="5Y Inf Exp", line=dict(color="#ff9800")), row=2, col=1)
    fig_umich.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                            height=500, margin=dict(l=10,r=10,t=45,b=30),
                            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    fig_umich.update_xaxes(gridcolor=GRID_COLOR)
    fig_umich.update_yaxes(gridcolor=GRID_COLOR)
    add_recessions(fig_umich, recessions, rows=[1,1,2,2], cols=[1,1,1,1])

    # Export/Import Prices
    fig_pi = go.Figure()
    for col in pd.concat([exp_pi, imp_pi], axis=1).columns:
        src = pd.concat([exp_pi, imp_pi], axis=1)
        ax  = "y2" if "MoM" in col else "y"
        fig_pi.add_trace(go.Scatter(x=src.index, y=src[col], name=col, mode="lines", yaxis=ax))
    fig_pi.update_layout(**dual_axis_layout("Export & Import Price Indices", "YoY %", "MoM %"))
    add_recessions(fig_pi, recessions)

    inflation_charts = [
        ("CPI vs Core CPI", fig_cpi, pd.concat([cpi, core_cpi], axis=1)),
        ("PCE vs Core PCE", fig_pce, pd.concat([pce, core_pce], axis=1)),
        ("PPI", fig_ppi, ppi),
        ("Retail Sales", fig_rsa, rsa),
        ("TIPS Breakevens", fig_be, pd.concat([be_5y, be_10y], axis=1)),
        ("UMich & Inflation Expectations", fig_umich, pd.concat([umich, inf_exp1, inf_exp5], axis=1)),
        ("Export & Import Prices", fig_pi, pd.concat([exp_pi, imp_pi], axis=1)),
    ]
    render_two_col(inflation_charts)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Labour Market
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Labour Market")
    with st.spinner("Loading labour data…"):
        wages   = mom_yoy(fetch("CES0500000003", "Avg Hourly Earnings", START, END), "Avg Hourly Earnings")
        nfp     = nfp_change(fetch("PAYEMS", "NFP", START, END), "NFP")
        lfpr    = fetch("CIVPART", "Labour Force Participation Rate", START, END)
        prime_lfpr = fetch("LNS11300060", "Prime-Age LFPR (25-54)", START, END)

        unemp_data = pd.concat([
            fetch(sid, lbl, START, END)
            for sid, lbl in [
                ("U1RATE","U1"), ("U2RATE","U2"), ("UNRATE","U3"),
                ("U4RATE","U4"), ("U5RATE","U5"), ("U6RATE","U6"), ("CGBD25O","U7 BA+"),
            ]
        ], axis=1)

        demo = pd.concat([
            fetch(sid, lbl, START, END)
            for sid, lbl in [
                ("LNS14000003","Men 20+"), ("LNS14000002","Women 20+"),
                ("LNS14000006","Teenagers"), ("LNS14000009","Black/AA"),
                ("LNS14000012","Hispanic"), ("LNS14027662","White"),
            ]
        ], axis=1)

        claims = fetch("ICSA", "Initial Claims", START, END)
        if not claims.empty and "Initial Claims" in claims.columns:
            claims["4W MA"]  = claims["Initial Claims"].rolling(4).mean()
            claims["12W MA"] = claims["Initial Claims"].rolling(12).mean()

        # JOLTS
        jolts_openings = fetch("JTSJOL",  "Job Openings (k)", START, END)
        jolts_quits    = fetch("JTSQUR",  "Quits Rate", START, END)
        jolts_layoffs  = fetch("JTSLDR",  "Layoffs Rate", START, END)
        jolts_hire     = fetch("JTSHIR",  "Hire Rate", START, END)

        # ADP sectors
        adp_ids = {
            "Construction":       "ADPWINDCONNERSA",
            "Information":        "ADPWINDINFONERSA",
            "Prof & Business":    "ADPWINDPROBUSNERSA",
            "Leisure & Hosp":     "ADPWINDLSHPNERSA",
            "Education & Health": "ADPWINDEDHLTNERSA",
            "Trade & Transport":  "ADPWINDTTUNERSA",
            "Financial":          "ADPWINDFINNERSA",
        }
        adp_sectors = pd.concat([
            nfp_change(fetch(sid, name, START, END), name)
            for name, sid in adp_ids.items()
        ], axis=1)


    # Wages
    fig_wages = go.Figure()
    for col in wages.columns:
        ax = "y2" if "MoM" in col else "y"
        fig_wages.add_trace(go.Scatter(x=wages.index, y=wages[col], name=col, mode="lines", yaxis=ax))
    fig_wages.update_layout(**dual_axis_layout("Avg Hourly Earnings", "YoY %", "MoM %"))
    add_recessions(fig_wages, recessions)

    # NFP bar
    fig_nfp = go.Figure()
    colors_nfp = ["#26a69a" if v >= 0 else "#ef5350" for v in nfp["NFP MoM Change (k)"].fillna(0)]
    fig_nfp.add_trace(go.Bar(x=nfp.index, y=nfp["NFP MoM Change (k)"],
                             marker_color=colors_nfp, name="NFP MoM"))
    fig_nfp.update_layout(**base_layout("Nonfarm Payrolls MoM Change (k)"))
    add_recessions(fig_nfp, recessions)

    # LFPR
    fig_lfpr = go.Figure()
    if not lfpr.empty:
        fig_lfpr.add_trace(go.Scatter(x=lfpr.index, y=lfpr["Labour Force Participation Rate"],
                                      name="Overall LFPR", line=dict(color="#90a4d4")))
    if not prime_lfpr.empty:
        fig_lfpr.add_trace(go.Scatter(x=prime_lfpr.index, y=prime_lfpr["Prime-Age LFPR (25-54)"],
                                      name="Prime-Age (25-54)", line=dict(color="#26a69a")))
    fig_lfpr.update_layout(**base_layout("Labour Force Participation Rate"))
    add_recessions(fig_lfpr, recessions)

    # Unemployment U1-U7
    fig_unemp = go.Figure()
    colors_u = ["#ef5350","#ff7043","#ff9800","#ffc107","#26a69a","#42a5f5","#ab47bc"]
    for col, color in zip(unemp_data.columns, colors_u):
        fig_unemp.add_trace(go.Scatter(x=unemp_data.index, y=unemp_data[col], name=col,
                                        mode="lines", line=dict(color=color)))
    fig_unemp.update_layout(**base_layout("Unemployment Rates U1–U7"))
    add_recessions(fig_unemp, recessions)

    # Demographics
    fig_demo = go.Figure()
    for col in demo.columns:
        fig_demo.add_trace(go.Scatter(x=demo.index, y=demo[col], name=col, mode="lines"))
    fig_demo.update_layout(**base_layout("Unemployment by Demographics"))
    add_recessions(fig_demo, recessions)

    # Jobless claims
    fig_claims = go.Figure()
    color_map = {"Initial Claims": "#90a4d4", "4W MA": "#26a69a", "12W MA": "#ff9800"}
    for col in claims.columns:
        if col in color_map:
            fig_claims.add_trace(go.Scatter(
                x=claims.index, y=claims[col], name=col, mode="lines",
                line=dict(color=color_map[col], width=2 if "MA" in col else 1)))
    fig_claims.update_layout(**base_layout("Initial Jobless Claims + Moving Averages"))
    add_recessions(fig_claims, recessions)

    # JOLTS
    fig_jolts = make_subplots(rows=2, cols=2,
                              subplot_titles=("Job Openings (k)", "Quits Rate", "Layoffs Rate", "Hire Rate"),
                              shared_xaxes=False, vertical_spacing=0.12)
    for (df_j, label, color), (row, col) in zip(
        [(jolts_openings,"Job Openings (k)","#26a69a"),
         (jolts_quits,   "Quits Rate",     "#ff9800"),
         (jolts_layoffs, "Layoffs Rate",   "#ef5350"),
         (jolts_hire,    "Hire Rate",      "#90a4d4")],
        [(1,1),(1,2),(2,1),(2,2)]
    ):
        if not df_j.empty:
            fig_jolts.add_trace(go.Scatter(x=df_j.index, y=df_j.iloc[:,0],
                                           name=label, line=dict(color=color)), row=row, col=col)
    fig_jolts.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                            height=520, margin=dict(l=10,r=10,t=50,b=30),
                            showlegend=False)
    fig_jolts.update_xaxes(gridcolor=GRID_COLOR)
    fig_jolts.update_yaxes(gridcolor=GRID_COLOR)
    add_recessions(fig_jolts, recessions, rows=[1,1,2,2], cols=[1,2,1,2])

    # Beveridge curve
    if not jolts_openings.empty and not unemp_data.empty:
        merged_bev = pd.concat([
            jolts_openings["Job Openings (k)"] / 1000,
            unemp_data["U3"]
        ], axis=1).dropna()
        merged_bev.columns = ["openings_m", "unemployment"]
        fig_bev = go.Figure()
        fig_bev.add_trace(go.Scatter(
            x=merged_bev["unemployment"], y=merged_bev["openings_m"],
            mode="markers+lines",
            marker=dict(
                color=list(range(len(merged_bev))),
                colorscale="Viridis", size=5, opacity=0.7,
                colorbar=dict(title="Time →", len=0.5, thickness=10),
            ),
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
            text=merged_bev.index.strftime("%Y-%m"),
            hovertemplate="Date: %{text}<br>Unemployment: %{x:.1f}%<br>Openings: %{y:.2f}M<extra></extra>",
            name="Beveridge Curve",
        ))
        fig_bev.update_layout(**base_layout("Beveridge Curve (Job Openings vs Unemployment Rate)"))
        fig_bev.update_xaxes(title="Unemployment Rate (%)", gridcolor=GRID_COLOR)
        fig_bev.update_yaxes(title="Job Openings (M)", gridcolor=GRID_COLOR)
    else:
        fig_bev = go.Figure()

    # ADP sectors
    fig_adp = go.Figure()
    for col in adp_sectors.columns:
        fig_adp.add_trace(go.Scatter(x=adp_sectors.index, y=adp_sectors[col], name=col, mode="lines"))
    fig_adp.update_layout(**base_layout("ADP Private Employment by Sector (MoM Change k)"))
    add_recessions(fig_adp, recessions)


    labor_charts = [
        ("Wages", fig_wages, wages),
        ("NFP", fig_nfp, nfp),
        ("LFPR", fig_lfpr, pd.concat([lfpr, prime_lfpr], axis=1)),
        ("Unemployment U1-U7", fig_unemp, unemp_data),
        ("Demographics", fig_demo, demo),
        ("Initial Claims", fig_claims, claims),
        ("JOLTS", fig_jolts, pd.concat([jolts_openings, jolts_quits, jolts_layoffs, jolts_hire], axis=1)),
        ("Beveridge Curve", fig_bev, merged_bev if not jolts_openings.empty else None),
        ("ADP Sectors", fig_adp, adp_sectors),
    ]
    render_two_col(labor_charts)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Housing
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Housing")
    with st.spinner("Loading housing data…"):
        home_sales  = fetch("EXHOSLUSM495S", "Existing Home Sales", START, END)
        new_sales   = fetch("HSN1F",         "New Home Sales", START, END)
        starts      = fetch("HOUST",         "Housing Starts", START, END)
        permits     = fetch("PERMIT",        "Building Permits", START, END)
        case_shiller = fetch("CSUSHPINSA",   "Case-Shiller HPI", START, END)

    # Existing vs New Home Sales
    fig_sales = go.Figure()
    if not home_sales.empty:
        fig_sales.add_trace(go.Scatter(x=home_sales.index, y=home_sales["Existing Home Sales"],
                                       name="Existing", line=dict(color="#26a69a")))
    if not new_sales.empty:
        fig_sales.add_trace(go.Scatter(x=new_sales.index, y=new_sales["New Home Sales"],
                                       name="New", line=dict(color="#ff9800"), yaxis="y2"))
    fig_sales.update_layout(**dual_axis_layout("Existing vs New Home Sales", "Existing (k)", "New (k)"))
    add_recessions(fig_sales, recessions)

    # Starts vs Permits
    fig_starts = go.Figure()
    if not starts.empty:
        fig_starts.add_trace(go.Scatter(x=starts.index, y=starts["Housing Starts"],
                                        name="Starts", line=dict(color="#42a5f5")))
    if not permits.empty:
        fig_starts.add_trace(go.Scatter(x=permits.index, y=permits["Building Permits"],
                                        name="Permits", line=dict(color="#ab47bc")))
    fig_starts.update_layout(**base_layout("Housing Starts vs Building Permits (Leading Indicator)"))
    add_recessions(fig_starts, recessions)

    # Case-Shiller
    cs_mom = mom_yoy(case_shiller, "Case-Shiller HPI") if not case_shiller.empty else pd.DataFrame()
    fig_cs = go.Figure()
    if not cs_mom.empty:
        for col in cs_mom.columns:
            ax = "y2" if "MoM" in col else "y"
            fig_cs.add_trace(go.Scatter(x=cs_mom.index, y=cs_mom[col], name=col, mode="lines", yaxis=ax))
    fig_cs.update_layout(**dual_axis_layout("Case-Shiller Home Price Index", "YoY %", "MoM %"))
    add_recessions(fig_cs, recessions)


    housing_charts = [
        ("Home Sales", fig_sales, pd.concat([home_sales, new_sales], axis=1)),
        ("Starts vs Permits", fig_starts, pd.concat([starts, permits], axis=1)),
        ("Case-Shiller HPI", fig_cs, cs_mom),
    ]
    render_two_col(housing_charts)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Monetary & Rates
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Monetary Policy & Rates")
    with st.spinner("Loading monetary data…"):
        fed_total  = fetch("WALCL",        "Fed Total Assets (M)", START, END)
        fed_tres   = fetch("TREAST",       "Fed Treasuries (M)", START, END)
        m2         = fetch("M2SL",         "M2", START, END)
        sofr       = fetch("SOFR",         "SOFR", START, END)
        iorb       = fetch("IORB",         "IORB", START, END)
        rrp        = fetch("RRPONTSYAWARD","ON RRP", START, END)
        effr       = fetch("FEDFUNDS",     "EFFR", START, END)

        maturities = {
            "1M":"DGS1MO","3M":"DGS3MO","6M":"DGS6MO","1Y":"DGS1",
            "2Y":"DGS2","3Y":"DGS3","5Y":"DGS5","7Y":"DGS7",
            "10Y":"DGS10","20Y":"DGS20","30Y":"DGS30"
        }
        yc = pd.concat([fetch(code, label, START, END) for label, code in maturities.items()], axis=1)

        tips_5y  = fetch("DFII5",  "5Y Real Yield", START, END)
        tips_10y = fetch("DFII10", "10Y Real Yield", START, END)
        be_5y2   = fetch("T5YIE",  "5Y Breakeven", START, END)
        be_10y2  = fetch("T10YIE", "10Y Breakeven", START, END)

        ig_oas = fetch("BAMLC0A0CM",  "IG OAS", START, END)
        hy_oas = fetch("BAMLH0A0HYM2","HY OAS", START, END)

    # Fed balance sheet breakdown
    fed_total_T = fed_total["Fed Total Assets (M)"] / 1e6 if not fed_total.empty else pd.Series()
    fig_fed = go.Figure()
    if not fed_total.empty:
        fig_fed.add_trace(go.Scatter(x=fed_total.index, y=fed_total_T,
                                     name="Total Assets", line=dict(color="#90a4d4", width=2)))
    if not fed_tres.empty:
        fig_fed.add_trace(go.Scatter(x=fed_tres.index, y=fed_tres["Fed Treasuries (M)"] / 1e6,
                                     name="Treasuries", fill="tozeroy",
                                     line=dict(color="#42a5f5"), fillcolor="rgba(66,165,245,0.15)"))
    fig_fed.update_layout(**base_layout("Fed Balance Sheet — Total Assets & Treasuries (Trillions)"))
    fig_fed.update_yaxes(ticksuffix="T")
    add_recessions(fig_fed, recessions)

    # M2
    m2_data = pd.DataFrame(index=m2.index) if not m2.empty else pd.DataFrame()
    if not m2.empty:
        m2_data["M2 (Billions)"] = m2["M2"] / 1e3
        m2_data["YoY %"] = m2["M2"].pct_change(12) * 100
        m2_data["MoM %"] = m2["M2"].pct_change() * 100
    fig_m2 = go.Figure()
    if not m2_data.empty:
        fig_m2.add_trace(go.Scatter(x=m2_data.index, y=m2_data["M2 (Billions)"],
                                    name="M2 Level", line=dict(color="#26a69a"), yaxis="y"))
        fig_m2.add_trace(go.Scatter(x=m2_data.index, y=m2_data["YoY %"],
                                    name="YoY %", line=dict(color="#ff9800"), yaxis="y2"))
    fig_m2.update_layout(**dual_axis_layout("M2 Money Supply", "Billions $", "YoY %"))
    add_recessions(fig_m2, recessions)

    # Policy rates
    fig_rates = go.Figure()
    for df_r, col, color in [(effr,"EFFR","#ef5350"),(sofr,"SOFR","#26a69a"),
                              (iorb,"IORB","#ff9800"),(rrp,"ON RRP","#ab47bc")]:
        if not df_r.empty:
            fig_rates.add_trace(go.Scatter(x=df_r.index, y=df_r.iloc[:,0],
                                           name=col, line=dict(color=color)))
    fig_rates.update_layout(**base_layout("Policy Rates — EFFR, SOFR, IORB, ON RRP"))
    add_recessions(fig_rates, recessions)

    # Yield curve snapshots
    if not yc.empty:
        latest_date = yc.dropna(how="all").index[-1]
        snap_labels = {"Latest": 0, "1D Ago": -1, "1W Ago": -5, "1M Ago": -21}
        snap_colors = {"Latest":"cyan","1D Ago":"magenta","1W Ago":"orange","1M Ago":"green"}
        fig_yc = go.Figure()
        for label, offset in snap_labels.items():
            idx = max(0, len(yc) - 1 + offset)
            snap_date = yc.index[idx]
            row_vals  = yc.iloc[idx]
            fig_yc.add_trace(go.Scatter(
                x=list(maturities.keys()), y=row_vals.values,
                mode="lines+markers", name=f"{label} ({snap_date.date()})",
                line=dict(color=snap_colors[label],
                          width=2 if label=="Latest" else 1,
                          dash="solid" if label=="Latest" else "dash"),
            ))
        fig_yc.update_layout(**base_layout(f"US Treasury Yield Curve — Snapshots"))
        fig_yc.update_xaxes(title="Maturity", gridcolor=GRID_COLOR)
        fig_yc.update_yaxes(title="Yield (%)", ticksuffix="%", gridcolor=GRID_COLOR)

        # Yield curve changes (bar)
        latest_row = yc.iloc[-1]
        fig_yc_chg = go.Figure()
        for label, offset in [("1D Ago",-1),("1W Ago",-5),("1M Ago",-21)]:
            idx = max(0, len(yc) - 1 + offset)
            chg = (latest_row - yc.iloc[idx]) * 100  # in bps
            colors_bar = ["#26a69a" if v >= 0 else "#ef5350" for v in chg]
            fig_yc_chg.add_trace(go.Bar(
                x=list(maturities.keys()), y=chg,
                name=label, marker_color=snap_colors[label], opacity=0.8,
            ))
        fig_yc_chg.update_layout(**base_layout("Yield Curve Changes (bps)"), barmode="group")
        fig_yc_chg.update_yaxes(ticksuffix=" bps")
    else:
        fig_yc = fig_yc_chg = go.Figure()

    # Treasury spreads
    spreads = pd.DataFrame()
    if not yc.empty:
        # yc is in percentage points (e.g. 4.15 = 4.15%); *100 converts the spread to bps
        spreads["10Y-2Y"]  = (yc.get("10Y",pd.Series()) - yc.get("2Y",pd.Series())) * 100
        spreads["10Y-3M"]  = (yc.get("10Y",pd.Series()) - yc.get("3M",pd.Series())) * 100
        spreads["5Y-2Y"]   = (yc.get("5Y",pd.Series())  - yc.get("2Y",pd.Series())) * 100
        spreads["30Y-10Y"] = (yc.get("30Y",pd.Series()) - yc.get("10Y",pd.Series())) * 100
        spreads = spreads.dropna(how="all")
    fig_spreads = go.Figure()
    for col in spreads.columns:
        fig_spreads.add_trace(go.Scatter(x=spreads.index, y=spreads[col], name=col, mode="lines"))
    fig_spreads.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_spreads.update_layout(**base_layout("Treasury Yield Spreads (bps)"))
    fig_spreads.update_yaxes(ticksuffix=" bps")
    add_recessions(fig_spreads, recessions)

    # Real yields vs Breakevens
    fig_real = go.Figure()
    for df_r, col, color in [
        (tips_5y, "5Y Real Yield","#26a69a"), (tips_10y,"10Y Real Yield","#42a5f5"),
        (be_5y2,  "5Y Breakeven","#ff9800"),  (be_10y2, "10Y Breakeven","#ef5350"),
    ]:
        if not df_r.empty:
            fig_real.add_trace(go.Scatter(x=df_r.index, y=df_r.iloc[:,0],
                                          name=col, line=dict(color=color)))
    fig_real.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_real.update_layout(**base_layout("Real Yields (TIPS) vs Breakeven Inflation"))
    add_recessions(fig_real, recessions)

    # Credit spreads - FRED reports these OAS series in percentage points, not bps
    fig_credit = go.Figure()
    if not ig_oas.empty:
        fig_credit.add_trace(go.Scatter(x=ig_oas.index, y=ig_oas["IG OAS"] * 100,
                                        name="IG OAS", line=dict(color="#26a69a"), yaxis="y"))
    if not hy_oas.empty:
        fig_credit.add_trace(go.Scatter(x=hy_oas.index, y=hy_oas["HY OAS"] * 100,
                                        name="HY OAS", line=dict(color="#ef5350"), yaxis="y2"))
    fig_credit.update_layout(**dual_axis_layout("Credit Spreads — IG & HY OAS (bps)", "IG OAS (bps)", "HY OAS (bps)"))
    add_recessions(fig_credit, recessions)

    monetary_charts = [
        ("Fed Balance Sheet", fig_fed, pd.concat([fed_total, fed_tres], axis=1)),
        ("M2 Money Supply",   fig_m2,  m2_data),
        ("Policy Rates",      fig_rates, pd.concat([effr, sofr, iorb, rrp], axis=1)),
        ("Yield Curve",       fig_yc,  yc),
        ("Yield Curve Changes", fig_yc_chg, None),
        ("Treasury Spreads",  fig_spreads, spreads),
        ("Real Yields vs Breakevens", fig_real, pd.concat([tips_5y, tips_10y, be_5y2, be_10y2], axis=1)),
        ("Credit Spreads",    fig_credit, pd.concat([ig_oas, hy_oas], axis=1)),
    ]
    render_two_col(monetary_charts)

    # ── Fed Funds implied-probability (WIRP/FedWatch-style) ─────────────────────
    st.markdown('<div class="section-header">Fed Funds Implied Probabilities</div>', unsafe_allow_html=True)
    with st.spinner("Loading Fed Funds futures…"):
        fedwatch_df, current_effr = get_fedwatch_probabilities(years_ahead=2)

    if not fedwatch_df.empty:
        st.markdown(f"**Current EFFR:** {current_effr:.2f}%  |  **Meetings shown:** next {len(fedwatch_df)} (through {fedwatch_df['Meeting'].iloc[-1]})")

        bar_colors = ["#26a69a" if r > current_effr else "#ef5350" if r < current_effr else "#8a94a6"
                      for r in fedwatch_df["Implied Rate"]]
        fig_fedwatch = go.Figure(go.Bar(
            x=fedwatch_df["Meeting"], y=fedwatch_df["Implied Rate"], marker_color=bar_colors,
            text=fedwatch_df["Implied Rate"].round(3), textposition="outside"
        ))
        fig_fedwatch.add_hline(y=current_effr, line_dash="dash", line_color="#e0e0e0",
                                annotation_text=f"Current EFFR ({current_effr:.2f}%)", annotation_position="top left")
        fig_fedwatch.update_layout(**base_layout("Market-Implied Fed Funds Rate by FOMC Meeting"))
        fig_fedwatch.update_yaxes(ticksuffix="%", title="Implied Rate")
        fig_fedwatch.update_xaxes(title="FOMC Meeting Date")

        # How the implied path for these same meetings has shifted over time - same
        # Today/1W/1M/3M snapshot pattern as the Treasury Yield Curve Snapshots chart.
        with st.spinner("Loading historical Fed Funds futures…"):
            fedwatch_hist_df = get_fedwatch_history(years_ahead=2)
        fig_fedwatch_hist = go.Figure()
        for label in SNAPSHOT_OFFSETS:
            if label not in fedwatch_hist_df.columns:
                continue
            fig_fedwatch_hist.add_trace(go.Scatter(
                x=fedwatch_hist_df["Meeting"], y=fedwatch_hist_df[label],
                mode="lines+markers", name=label,
                line=dict(color=SNAPSHOT_COLORS[label], width=2 if label == "Today" else 1,
                           dash="solid" if label == "Today" else "dash"),
            ))
        fig_fedwatch_hist.update_layout(**base_layout("Market-Implied Fed Funds Rate — Snapshots"))
        fig_fedwatch_hist.update_xaxes(title="FOMC Meeting Date")
        fig_fedwatch_hist.update_yaxes(title="Implied Rate", ticksuffix="%")

        render_two_col([
            ("Market-Implied Fed Funds Rate", fig_fedwatch, fedwatch_df[["Meeting", "Implied Rate"]]),
            ("Market-Implied Fed Funds Rate Snapshots", fig_fedwatch_hist, fedwatch_hist_df),
        ])

        # Detail table, Bloomberg WIRP-style: implied rate path + cumulative and
        # this-meeting-only move sizes.
        detail = fedwatch_df[["Meeting", "Implied Rate", "Imp. Rate Delta", "Hikes/Cuts", "This-Meeting Move (bps)"]].copy()
        detail["Implied Rate"] = detail["Implied Rate"].round(3)
        detail["Imp. Rate Delta"] = detail["Imp. Rate Delta"].round(3)
        detail["Hikes/Cuts"] = detail["Hikes/Cuts"].round(2)
        detail["%Hike/Cut (this meeting)"] = (detail["This-Meeting Move (bps)"] / (RATE_STEP * 100) * 100).round(1)
        detail = detail.drop(columns="This-Meeting Move (bps)")
        detail = detail.rename(columns={"Imp. Rate Delta": "Imp. Rate Δ (cum.)", "Hikes/Cuts": "#Hikes/Cuts (cum.)"})
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.caption("A.R.M. (step size): 25bps. \"#Hikes/Cuts (cum.)\" and \"Imp. Rate Δ (cum.)\" are relative to today's "
                   "EFFR; \"%Hike/Cut (this meeting)\" is the incremental move priced in at that specific meeting only "
                   "(chained off the previous meeting's implied rate) - the same simplified 2-outcome-per-meeting view "
                   "as the chart above, not CME's full joint multi-meeting solve.")
        csv_download(detail, "fedwatch_probabilities")
    else:
        st.info("No FOMC meetings in the selected window, or Fed Funds futures data unavailable.")

    # ── Treasury issuance, maturity wall, outstanding & bid-to-cover ───────────
    st.markdown('<div class="section-header">Treasury Issuance & Supply</div>', unsafe_allow_html=True)

    days_ahead = st.select_slider("Forward-looking window (days)", options=[7, 30, 60, 90, 120], value=7, key="treasury_days_ahead")

    with st.spinner("Loading Treasury auction data…"):
        auctions = load_auctions_data()

    upcoming, issuance_summary = get_upcoming_issuances(auctions, days_ahead)
    maturing, maturity_summary = get_maturing_treasuries(auctions, days_ahead)
    outstanding, outstanding_summary = get_outstanding_by_tenor(auctions)

    combined = pd.merge(issuance_summary, maturity_summary, on="security_term_week_year", how="outer").fillna(0)
    combined = _sort_by_tenor(combined, "security_term_week_year")
    combined["Net New Supply (Billion $)"] = combined["Total Issuance (Billion $)"] - combined["Total Maturing (Billion $)"]
    total_issuance, total_maturing = combined["Total Issuance (Billion $)"].sum(), combined["Total Maturing (Billion $)"].sum()

    # Upcoming issuances table
    fig_upcoming = go.Figure(go.Table(
        columnwidth=[100, 100, 60, 70, 90, 90],
        header=dict(values=["Auction Date", "Issue Date", "Type", "Term", "CUSIP", "Offering ($B)"],
                    fill_color=PLOT_BG, font=dict(color="white", size=12), align="center", height=28),
        cells=dict(values=[upcoming["auction_date"].dt.strftime("%Y-%m-%d"), upcoming["issue_date"].dt.strftime("%Y-%m-%d"),
                            upcoming["security_type"], upcoming["security_term_week_year"], upcoming["cusip"],
                            upcoming["offering_amt_bil"].round(2)],
                   fill_color=PAPER_BG, font=dict(color="#e0e0e0", size=11), align="center")
    ))
    fig_upcoming.update_layout(**base_layout(f"Upcoming Issuances — Next {days_ahead}d (${total_issuance:,.1f}B)", height=420))

    # Maturing treasuries table
    fig_maturing = go.Figure(go.Table(
        columnwidth=[110, 60, 70, 90, 90],
        header=dict(values=["Maturity Date", "Type", "Term", "CUSIP", "Maturing ($B)"],
                    fill_color=PLOT_BG, font=dict(color="white", size=12), align="center", height=28),
        cells=dict(values=[maturing["maturity_date"].dt.strftime("%Y-%m-%d"), maturing["security_type"],
                            maturing["security_term_week_year"], maturing["cusip"], maturing["maturing_amt_bil"].round(2)],
                   fill_color=PAPER_BG, font=dict(color="#e0e0e0", size=11), align="center")
    ))
    fig_maturing.update_layout(**base_layout(f"Maturing Treasuries — Next {days_ahead}d (${total_maturing:,.1f}B)", height=420))

    # Issuance vs. maturities by term
    fig_supply_bar = go.Figure()
    fig_supply_bar.add_trace(go.Bar(x=combined["security_term_week_year"], y=combined["Total Issuance (Billion $)"],
                                     name="Issuance", marker_color="#26a69a"))
    fig_supply_bar.add_trace(go.Bar(x=combined["security_term_week_year"], y=-combined["Total Maturing (Billion $)"],
                                     name="Maturing", marker_color="#ef5350"))
    fig_supply_bar.add_trace(go.Scatter(x=combined["security_term_week_year"], y=combined["Net New Supply (Billion $)"],
                                         name="Net New Supply", mode="lines+markers", line=dict(color="#90a4d4", width=2)))
    fig_supply_bar.update_layout(**base_layout(
        f"Issuance vs. Maturities by Term (Net: ${total_issuance - total_maturing:,.1f}B)"))
    fig_supply_bar.update_layout(barmode="relative")

    # Outstanding by tenor
    total_outstanding = outstanding_summary["Outstanding (Billion $)"].sum()
    fig_outstanding = go.Figure(go.Bar(
        x=outstanding_summary["security_term_week_year"], y=outstanding_summary["Outstanding (Billion $)"],
        marker_color="#ab47bc", text=outstanding_summary["Outstanding (Billion $)"].round(0), textposition="outside"
    ))
    fig_outstanding.update_layout(**base_layout(
        f"Outstanding Marketable Treasuries (Total ${total_outstanding:,.0f}B) — "
        f"excl. intragvmt & non-marketable debt"
    ))

    # Bid-to-cover trend - reuses the global date range instead of its own lookback control.
    # If END isn't today (user is looking at a past window), START may be an arbitrary date
    # rather than a meaningful lookback, so fall back to a default 5-year window before END.
    bc_type = st.selectbox("Bid-to-cover: security type", sorted(auctions["security_type"].dropna().unique()),
                            index=sorted(auctions["security_type"].dropna().unique()).index("Note")
                            if "Note" in auctions["security_type"].values else 0, key="bc_type")

    end_ts = pd.Timestamp(END)
    if end_ts.normalize() == pd.Timestamp.today().normalize():
        bc_cutoff = pd.Timestamp(START)
    else:
        bc_cutoff = end_ts - pd.DateOffset(years=5)

    bc_hist = auctions[(auctions["security_type"] == bc_type) & auctions["bid_to_cover_ratio"].notna() &
                        (auctions["auction_date"] >= bc_cutoff) & (auctions["auction_date"] <= end_ts)].sort_values("auction_date")
    fig_btc = go.Figure()
    for term in sorted(bc_hist["security_term_week_year"].dropna().unique(), key=lambda t: _parse_tenor(t)):
        term_df = bc_hist[bc_hist["security_term_week_year"] == term]
        fig_btc.add_trace(go.Scatter(x=term_df["auction_date"], y=term_df["bid_to_cover_ratio"],
                                      mode="lines+markers", name=term, marker=dict(size=4)))
    fig_btc.update_layout(**base_layout(f"Bid-to-Cover Ratio — {bc_type}s ({bc_cutoff.date()} to {end_ts.date()})"))
    add_recessions(fig_btc, recessions)

    treasury_charts = [
        ("Upcoming Issuances", fig_upcoming, upcoming),
        ("Maturing Treasuries", fig_maturing, maturing),
        ("Issuance vs Maturity", fig_supply_bar, combined),
        ("Outstanding by Tenor", fig_outstanding, outstanding_summary),
        ("Bid-to-Cover Trend", fig_btc, bc_hist[["auction_date", "security_term_week_year", "bid_to_cover_ratio"]]),
    ]
    render_two_col(treasury_charts)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Leading Indicators
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Leading Indicators")
    with st.spinner("Loading leading indicator data…"):
        ism_mfg  = fetch("MANEMP",    "ISM Mfg Employment", START, END)
        lei          = fetch("USSLIND",   "Conference Board LEI", START, END)
        cfnai        = fetch("CFNAI",     "CFNAI", START, END)
        wei          = fetch("WEI",       "Weekly Economic Index", START, END)
        rec_prob     = fetch("RECPROUSM156N","Recession Probability (12M)", START, END)
        philly_fed   = fetch("GACDFSA066MSFRBPHI", "Philly Fed Business Outlook", START, END)
        empire_state = fetch("GAFDISA066MSFRBNY","Empire State Mfg", START, END)


    # LEI
    fig_lei = go.Figure()
    if not lei.empty:
        lei_yoy = lei["Conference Board LEI"].pct_change(12) * 100
        fig_lei.add_trace(go.Scatter(x=lei.index, y=lei["Conference Board LEI"],
                                     name="LEI Level", line=dict(color="#26a69a"), yaxis="y"))
        fig_lei.add_trace(go.Scatter(x=lei.index, y=lei_yoy,
                                     name="YoY %", line=dict(color="#ff9800", dash="dot"), yaxis="y2"))
    fig_lei.update_layout(**dual_axis_layout("Conference Board Leading Economic Index", "Level", "YoY %"))
    add_recessions(fig_lei, recessions)

    # CFNAI
    fig_cfnai = go.Figure()
    if not cfnai.empty:
        cfnai_ma3 = cfnai["CFNAI"].rolling(3).mean()
        bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in cfnai["CFNAI"].fillna(0)]
        fig_cfnai.add_trace(go.Bar(x=cfnai.index, y=cfnai["CFNAI"],
                                    marker_color=bar_colors, name="CFNAI", opacity=0.6))
        fig_cfnai.add_trace(go.Scatter(x=cfnai.index, y=cfnai_ma3,
                                        name="3M MA", line=dict(color="white", width=2)))
    fig_cfnai.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_cfnai.add_hline(y=-0.7, line_dash="dot", line_color="#ef5350",
                         annotation_text="Recession threshold (-0.7)")
    fig_cfnai.update_layout(**base_layout("Chicago Fed National Activity Index (CFNAI)"))
    add_recessions(fig_cfnai, recessions)

    # WEI
    fig_wei = go.Figure()
    if not wei.empty:
        fig_wei.add_trace(go.Scatter(x=wei.index, y=wei["Weekly Economic Index"],
                                      name="WEI", line=dict(color="#90a4d4")))
        fig_wei.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_wei.update_layout(**base_layout("Weekly Economic Index (WEI)"))
    add_recessions(fig_wei, recessions)

    # Recession probability
    fig_rec = go.Figure()
    if not rec_prob.empty:
        fig_rec.add_trace(go.Scatter(
            x=rec_prob.index, y=rec_prob["Recession Probability (12M)"],
            fill="tozeroy", name="Recession Probability",
            line=dict(color="#ef5350"), fillcolor="rgba(239,83,80,0.2)",
        ))
    fig_rec.add_hline(y=30, line_dash="dot", line_color="#ff9800",
                       annotation_text="30% threshold")
    fig_rec.update_layout(**base_layout("12-Month Recession Probability (Fed Model)"))
    fig_rec.update_yaxes(ticksuffix="%")
    add_recessions(fig_rec, recessions)

    # Regional Fed surveys
    fig_regional = go.Figure()
    for df_r, col, color in [(philly_fed,"Philly Fed","#42a5f5"),
                              (empire_state,"Empire State","#ff9800")]:
        if not df_r.empty:
            fig_regional.add_trace(go.Scatter(x=df_r.index, y=df_r.iloc[:,0],
                                               name=col, line=dict(color=color)))
    fig_regional.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_regional.update_layout(**base_layout("Regional Fed Manufacturing Surveys"))
    add_recessions(fig_regional, recessions)

    leading_charts = [
        ("Conference Board LEI", fig_lei, lei),
        ("CFNAI", fig_cfnai, cfnai),
        ("Weekly Economic Index", fig_wei, wei),
        ("Recession Probability", fig_rec, rec_prob),
        ("Regional Fed Surveys", fig_regional, pd.concat([philly_fed, empire_state], axis=1)),
    ]
    render_two_col(leading_charts)

st.markdown("---")
st.caption("Data: FRED (Federal Reserve Bank of St. Louis) · Shaded areas = NBER recessions · Refresh rate: 1hr cache")
