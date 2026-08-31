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
EIA_API_KEY = os.getenv("EIA_API_KEY")

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
    # Retry on ANY failure, not just rate-limit-shaped ones - fredapi occasionally raises a
    # bare ValueError(None) (str(e) == "None") on a transient FRED-side hiccup (a malformed
    # response during a brief server restart, a dropped connection), which is not a rate-limit
    # message and would previously skip the retry entirely and fail on the very first attempt.
    # Confirmed live: DGS1 (1-Year Treasury) failed this way once while being a completely
    # healthy, currently-published series - a one-off transient issue, not a broken series.
    last_exc = None
    for attempt in range(3):
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            df = pd.DataFrame({label: s.values}, index=pd.to_datetime(s.index))
            df.index.name = "date"
            time.sleep(0.15)
            return df
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
    st.warning(f"Could not load {label} ({series_id}) after 3 attempts: {last_exc}")
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
    out[f"{col} MoM %"] = (df[col].pct_change() * 100).round(3)
    out[f"{col} YoY %"] = (df[col].pct_change(12) * 100).round(3)
    return out

def nfp_change(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[f"{col} MoM Change (k)"])
    out = pd.DataFrame(index=df.index)
    out[f"{col} MoM Change (k)"] = df[col].diff().round(3)
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

# FOMC decision dates (2nd day of each 2-day meeting). The Fed only confirms each date at
# the meeting immediately prior - there's no formula for these, so *something* has to be
# the source of truth. The Fed publishes no calendar API/feed, only this HTML page, so
# FOMC_DATES_FALLBACK below is a seed used only if the live fetch fails (network error, or
# the page's markup changes enough to break the parser) - refresh it by hand from
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm every year or two.
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_MONTH_NUM = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,
                   "August":8,"September":9,"October":10,"November":11,"December":12}
FOMC_DATES_FALLBACK = [
    "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

@st.cache_data(ttl=604800)  # the Fed's calendar changes at most a few times a year
def get_fomc_dates():
    """Scrape the Fed's own FOMC calendar page. Falls back to FOMC_DATES_FALLBACK on any
    failure (network error, empty/garbled result) so a live-site hiccup can't take down
    the whole section."""
    try:
        resp = requests.get(FOMC_CALENDAR_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
        pair_re = re.compile(
            r'fomc-meeting__month[^>]*><strong>(\w+)</strong></div>\s*'
            r'<div class="fomc-meeting__date[^"]*">([\d\-\*]+)<'
        )
        dates = []
        for year_str, panel_html in zip(*[iter(re.split(r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a></h4>', html)[1:])] * 2):
            year = int(year_str)
            for month_name, day_range in pair_re.findall(panel_html):
                days = re.findall(r'\d+', day_range)
                month_num = FOMC_MONTH_NUM.get(month_name)
                if not days or not month_num:
                    continue
                decision_day = int(days[-1])
                if decision_day > calendar.monthrange(year, month_num)[1]:
                    continue  # cross-month meeting (e.g. "Jan 31-Feb 1") - skip rather than misattribute
                dates.append(f"{year:04d}-{month_num:02d}-{decision_day:02d}")
        dates = sorted(set(dates))
        if len(dates) < 10:  # sanity floor - a near-empty result means the parser broke
            raise ValueError(f"only parsed {len(dates)} meetings, expected 10+")
        return dates
    except Exception:
        return FOMC_DATES_FALLBACK

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
    meetings = [pd.Timestamp(d) for d in get_fomc_dates() if today <= pd.Timestamp(d) <= window_end]

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
    meetings = [pd.Timestamp(d) for d in get_fomc_dates() if today <= pd.Timestamp(d) <= window_end]

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

# Standard Treasury note/bond original-issuance tenors (years) - 2/3/5/7/10-Year Notes and
# 20/30-Year Bonds are the current program; 4-Year existed as a standalone historical
# program (42 real records, distinct from the 2/3/5/7/10 cycle). Used below to snap
# sub-year-precision original_security_term values (e.g. "29-Year 9-Month") to the actual
# program they belong to.
NOTE_BOND_TENORS = [2, 3, 4, 5, 7, 10, 20, 30]

def _parse_year_month(term):
    m = re.match(r"(\d+)-Year(?:\s+(\d+)-Month)?", term)
    if not m:
        return None
    return int(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0)

def _normalize_tenor_bucket(row):
    # Treasury's security_term_week_year is the REMAINING maturity as of each individual
    # auction event, not the security's true original tenor - a 26-week bill reopened with
    # 4 weeks left gets labeled "4-Week", and a 10-year note reopened with 9 years left gets
    # labeled "9-Year", fragmenting one program's outstanding total across multiple buckets.
    # original_security_term is the correct field, but has its own two quirks: Cash
    # Management Bills carry an arbitrary day-count original term (e.g. "119-Day" - real,
    # tactical, irregular issuances, not a disguised standard week-tenor), and month-precision
    # entries (e.g. "29-Year 9-Month", "6-Year 11-Month") are NOT a distinct product - verified
    # live against the Treasury API: the same CUSIP carries the identical original_security_term
    # across every one of its reopenings, so this is a fixed attribute of the security (a
    # genuine 29y9m issue-to-maturity span from Treasury's fixed quarterly refunding calendar),
    # not an auction-event artifact. Truncating to the leading integer (the old approach) split
    # these into spurious standalone buckets - e.g. "29-Year 9-Month" (29.75y) got its own
    # "29-Year" bucket instead of merging into the 30-Year program it actually belongs to.
    # Snapping to the nearest real Treasury program fixes this correctly.
    term = row["original_security_term"]
    if not isinstance(term, str) or term in ("null", ""):
        return row["security_term_week_year"]  # fallback for the small number of true nulls
    if "Day" in term:
        return "CMB"
    if "Week" in term:
        return term  # clean week-denominated bills (8/13/17/26/52-Week) - no drift observed
    decimal_years = _parse_year_month(term)
    if decimal_years is None:
        return term
    nearest = min(NOTE_BOND_TENORS, key=lambda t: abs(t - decimal_years))
    return f"{nearest}-Year"

@st.cache_data(ttl=21600)  # Treasury auction calendar only changes a few times/week
def load_auctions_data():
    df = _download_all_auctions()
    for col in ["issue_date", "auction_date", "maturity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["offering_amt", "total_accepted", "bid_to_cover_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Bucket by true original tenor everywhere downstream, not remaining-maturity-at-auction.
    df["security_term_week_year"] = df.apply(_normalize_tenor_bucket, axis=1)
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

def get_issuance_by_tenor_over_time(df, start, end):
    # How much of each tenor Treasury has actually issued (auction-settled amount) each
    # quarter, over the selected date range - shows both total issuance growth and how the
    # tenor mix has shifted (e.g. more bills during a cash-management-heavy phase, more
    # coupons when terming out debt). Quarterly matches the cadence of Treasury's own
    # refunding announcements, and smooths out the lumpiness of e.g. monthly 10Y/30Y auctions
    # vs weekly bills.
    window = df[(df["issue_date"] >= pd.Timestamp(start)) & (df["issue_date"] <= pd.Timestamp(end))].copy()
    if window.empty:
        return pd.DataFrame()
    window["amt_bil"] = window["total_accepted"].fillna(window["offering_amt"]) / 1e9
    window["period"] = window["issue_date"].dt.to_period("Q")
    pivot = window.pivot_table(index="period", columns="security_term_week_year", values="amt_bil", aggfunc="sum").fillna(0)
    pivot.index = pivot.index.to_timestamp()
    tenor_order = _sort_by_tenor(pd.DataFrame({"security_term_week_year": pivot.columns}), "security_term_week_year")["security_term_week_year"]
    return pivot[tenor_order]

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

# Remaining-maturity ladder: buckets every currently-outstanding security by years left to
# maturity (maturity_date - today), snapped to the NEAREST tenor on this ladder - answers "how
# much comes due around year X from now" rather than "how much was originally issued as a
# 10-Year". Its finer 20-30Y granularity (22.5Y/25Y/27.5Y) is exactly what original-tenor
# bucketing can't show, since Treasury has never issued a standalone 22.5/25/27.5-Year original
# tenor - those buckets only make sense as remaining-maturity midpoints.
MATURITY_LADDER = {
    "3M": 0.25, "6M": 0.5, "12M": 1, "2Y": 2, "3Y": 3, "4Y": 4, "5Y": 5, "7Y": 7,
    "10Y": 10, "12Y": 12, "15Y": 15, "20Y": 20, "22.5Y": 22.5, "25Y": 25, "27.5Y": 27.5, "30Y": 30,
}
_MATURITY_LADDER_ITEMS = list(MATURITY_LADDER.items())
_MATURITY_LADDER_ORDER = {label: i for i, label in enumerate(MATURITY_LADDER)}

def _nearest_maturity_bucket(years_left):
    return min(_MATURITY_LADDER_ITEMS, key=lambda kv: abs(kv[1] - years_left))[0]

def get_outstanding_by_remaining_maturity(df):
    today = pd.Timestamp.today().normalize()
    outstanding = df[(df["issue_date"] <= today) & (df["maturity_date"] > today)].copy()
    outstanding["years_to_maturity"] = (outstanding["maturity_date"] - today).dt.days / 365.25
    outstanding["amt_bil"] = outstanding["total_accepted"].fillna(outstanding["offering_amt"]) / 1e9
    outstanding["maturity_bucket"] = outstanding["years_to_maturity"].apply(_nearest_maturity_bucket)
    summary = outstanding.groupby("maturity_bucket")["amt_bil"].sum().reset_index()
    summary = summary.rename(columns={"amt_bil": "Outstanding (Billion $)"})
    summary = summary.iloc[summary["maturity_bucket"].map(_MATURITY_LADDER_ORDER).argsort()]
    return outstanding, summary

def get_upcoming_issuance_by_maturity_bucket(df, days_ahead):
    # Same remaining-maturity-ladder bucketing as get_outstanding_by_remaining_maturity, applied
    # to not-yet-issued securities - so a 17-Week bill (maturity ~17 weeks from today, since
    # issue_date falls within the forward window) lands in the 3M bucket, a 26-Week bill in 6M,
    # etc. Uses maturity_date - today() (not issue_date), same reference point as the
    # outstanding side, so the two are directly stackable/addable bucket-for-bucket.
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    upcoming = df[(df["issue_date"] >= today) & (df["issue_date"] <= cutoff)].copy()
    if upcoming.empty:
        return pd.DataFrame(columns=["maturity_bucket", "Total Issuance (Billion $)"])
    upcoming["years_to_maturity"] = (upcoming["maturity_date"] - today).dt.days / 365.25
    upcoming["offering_amt_bil"] = upcoming["offering_amt"] / 1e9
    upcoming["maturity_bucket"] = upcoming["years_to_maturity"].apply(_nearest_maturity_bucket)
    summary = upcoming.groupby("maturity_bucket")["offering_amt_bil"].sum().reset_index()
    summary = summary.rename(columns={"offering_amt_bil": "Total Issuance (Billion $)"})
    summary = summary.iloc[summary["maturity_bucket"].map(_MATURITY_LADDER_ORDER).argsort()]
    return summary

# Yield-curve maturities FRED actually quotes (DGS series), in years - used to interpolate an
# estimated yield at the MATURITY_LADDER's tenors that FRED doesn't directly quote (4Y, 12Y,
# 15Y, 22.5Y, 25Y, 27.5Y). Standard practitioner technique (linear interpolation between real
# quoted points), not fabricated data.
YIELD_CURVE_MATURITY_YEARS = {"1M":1/12,"3M":0.25,"6M":0.5,"1Y":1,"2Y":2,"3Y":3,"5Y":5,"7Y":7,"10Y":10,"20Y":20,"30Y":30}

def _interp_yield_curve(row_vals, target_years):
    pairs = sorted((YIELD_CURVE_MATURITY_YEARS[lbl], row_vals[lbl]) for lbl in row_vals.index
                   if lbl in YIELD_CURVE_MATURITY_YEARS and pd.notna(row_vals[lbl]))
    if not pairs:
        return [None] * len(target_years)
    xs, ys = zip(*pairs)
    return list(np.interp(target_years, xs, ys))

# ── Fiscal accounts (TGA, debt limit, interest expense, MTS, spending by category) ─────

FISCAL_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

def _get_json_params(url, params, retries=5, backoff=2, timeout=30):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and r.text.strip():
                return r.json()
        except requests.exceptions.RequestException:
            pass
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Treasury Fiscal Data API failed after {retries} retries: {url}")

@st.cache_data(ttl=21600)
def get_tga_balance(start, end):
    j = _get_json_params(f"{FISCAL_BASE}/v1/accounting/dts/operating_cash_balance", {
        "filter": f"account_type:eq:Treasury General Account (TGA) Closing Balance,record_date:gte:{start},record_date:lte:{end}",
        "fields": "record_date,open_today_bal",
        "sort": "record_date", "page[size]": 10000,
    })
    data = j.get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["TGA Balance (Billion $)"] = pd.to_numeric(df["open_today_bal"], errors="coerce") / 1000
    df = df.set_index("record_date")[["TGA Balance (Billion $)"]]
    df.index.name = "date"
    return df

@st.cache_data(ttl=21600)
def get_debt_subject_to_limit(start, end):
    j = _get_json_params(f"{FISCAL_BASE}/v1/accounting/dts/debt_subject_to_limit", {
        "filter": f"record_date:gte:{start},record_date:lte:{end}",
        "fields": "record_date,debt_catg,close_today_bal",
        "sort": "record_date", "page[size]": 10000,
    })
    data = j.get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["close_today_bal"] = pd.to_numeric(df["close_today_bal"], errors="coerce")
    pivot = df.pivot_table(index="record_date", columns="debt_catg", values="close_today_bal", aggfunc="last")
    out = pd.DataFrame(index=pivot.index)
    out["Debt Subject to Limit ($T)"] = (
        pivot.get("Debt Held by the Public", 0) + pivot.get("Intragovernmental Holdings", 0)
        + pivot.get("Other Debt Subject to Limit", 0)
    ) / 1e6
    out["Statutory Limit ($T)"] = pivot.get("Statutory Debt Limit", pd.Series(dtype=float)) / 1e6
    # During a debt-ceiling suspension Treasury reports the statutory limit as 0 - that means
    # "not currently in effect," not "zero headroom." Drop those rows rather than chart a fake breach.
    out.loc[out["Statutory Limit ($T)"] <= 0, "Statutory Limit ($T)"] = np.nan
    out.index.name = "date"
    return out

MTS_MONTH_ORDER = ["October","November","December","January","February","March","April","May","June",
                    "July","August","September"]

@st.cache_data(ttl=21600)
def get_mts_monthly():
    # Each MTS "vintage" (record_date) carries the full current + prior fiscal year table -
    # find the latest vintage first, then pull that one vintage's monthly rows.
    latest_j = _get_json_params(f"{FISCAL_BASE}/v1/accounting/mts/mts_table_1", {
        "fields": "record_date", "sort": "-record_date", "page[size]": 1,
    })
    latest_data = latest_j.get("data", [])
    if not latest_data:
        return pd.DataFrame()
    latest_date = latest_data[0]["record_date"]

    j = _get_json_params(f"{FISCAL_BASE}/v1/accounting/mts/mts_table_1", {
        "filter": f"record_date:eq:{latest_date}", "page[size]": 500,
    })
    mts = pd.DataFrame(j.get("data", []))
    if mts.empty:
        return pd.DataFrame()
    headers_df = mts[mts["data_type_cd"] == "S"][["classification_id", "classification_desc"]].rename(
        columns={"classification_id": "parent_id", "classification_desc": "fy_label"})
    months = mts[mts["record_type_cd"] == "MTH"].merge(headers_df, on="parent_id", how="left")
    months = months.drop_duplicates(subset=["fy_label", "classification_desc"])
    for c in ["current_month_gross_rcpt_amt", "current_month_gross_outly_amt", "current_month_dfct_sur_amt"]:
        months[c] = pd.to_numeric(months[c], errors="coerce") / 1e9  # billions
    months["month_idx"] = months["classification_desc"].apply(lambda m: MTS_MONTH_ORDER.index(m) if m in MTS_MONTH_ORDER else -1)
    fy_num = months["fy_label"].str.extract(r"(\d+)").astype(float)[0]
    months["sort_key"] = fy_num * 100 + months["month_idx"]
    months = months[months["month_idx"] >= 0].sort_values("sort_key")
    months["Period"] = months["classification_desc"].str[:3] + " " + months["fy_label"].str.extract(r"(\d+)")[0].str[-2:]
    return months[["Period", "current_month_gross_rcpt_amt", "current_month_gross_outly_amt",
                    "current_month_dfct_sur_amt"]].reset_index(drop=True)

# Salient spending programs, grouped from their real (fragmented) DTS transaction_catg line
# items - e.g. defense pay, health and retirement are three separate rows in the raw data.
FISCAL_SPEND_GROUPS = {
    "Interest on Debt": ["Interest on Treasury Securities"],
    "Defense": ["Dept of Defense (DoD) - misc", "DoD - Health", "DoD - Military Active Duty Pay", "DoD - Military Retirement"],
    "Medicare": ["HHS - Federal Hospital Insr Trust Fund", "HHS - Federal Supple Med Insr Trust Fund",
                 "HHS - Medicare Prescription Drugs", "HHS - Othr Cent Medicare & Medicaid Serv"],
    "Medicaid": ["HHS - Grants to States for Medicaid"],
    "Social Security": ["SSA - Benefits Payments", "SSA - Supplemental Security Income", "Social Security Admin (SSA) - misc"],
    "Veterans Affairs": ["Dept of Veterans Affairs (VA)", "VA - Benefits"],
    "Unemployment Insurance": ["DOL - Unemployment Benefits"],
    "Food Assistance (SNAP/WIC)": ["USDA - Supp Nutrition Assist Prog (SNAP)", "USDA - Supp Nutrition Assist Prog (WIC)"],
}

@st.cache_data(ttl=21600)
def get_category_spend(categories: tuple, start: str):
    # A transaction_catg value containing its own parentheses (e.g. "...(SNAP)") breaks the
    # API's :in:(...) list filter once a SECOND parenthesised value joins the same list -
    # confirmed live: SNAP+WIC combined silently returns 0 rows, each alone returns data fine.
    # Fetching each category separately via :eq: and summing client-side sidesteps this for
    # every group, not just the ones with parens in the name.
    per_cat = []
    for cat in categories:
        j = _get_json_params(f"{FISCAL_BASE}/v1/accounting/dts/deposits_withdrawals_operating_cash", {
            "fields": "record_date,transaction_fytd_amt",
            "filter": f"transaction_catg:eq:{cat},record_date:gte:{start}",
            "sort": "record_date", "page[size]": 10000,
        })
        data = j.get("data", [])
        if not data:
            continue
        df = pd.DataFrame(data)
        df["record_date"] = pd.to_datetime(df["record_date"])
        df["transaction_fytd_amt"] = pd.to_numeric(df["transaction_fytd_amt"], errors="coerce")
        per_cat.append(df.set_index("record_date")["transaction_fytd_amt"].resample("ME").last().rename(cat))
    if not per_cat:
        return pd.DataFrame()
    combined = pd.concat(per_cat, axis=1).sum(axis=1, min_count=1).dropna()
    out = pd.DataFrame({"FYTD (Billion $)": combined / 1000})
    out["YoY %"] = (combined / combined.shift(12) - 1) * 100
    out.index.name = "date"
    return out

# ── US Markets (equities, cross-asset, via yfinance) ────────────────────────────
# yfinance/Yahoo Finance is well known to rate-limit or transiently block requests more
# readily than a residential/dev IP - confirmed live on a different dashboard in this same
# project (a ticker silently vanished from a ranking for a full day because one failed fetch
# got cached as "no data"). Retry with backoff, and keep the cache TTL short (1hr) so any
# fetch that does fail all retries self-heals quickly instead of persisting all day.

def _yf_retry(fn, retries=3, backoff=1.5):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("yfinance call failed")

@st.cache_data(ttl=3600)
def fetch_yf_close(ticker: str, label: str, start: str, end: str = None) -> pd.DataFrame:
    try:
        hist = _yf_retry(lambda: yf.download(ticker, start=start, end=end, progress=False,
                                             auto_adjust=True))
        if hist.empty:
            return pd.DataFrame()
        close = hist["Close"]
        s = close.iloc[:, 0] if hasattr(close, "columns") else close
        df = pd.DataFrame({label: s.values}, index=pd.to_datetime(s.index).tz_localize(None))
        df.index.name = "date"
        return df.dropna()
    except Exception as e:
        st.warning(f"Could not load {label} ({ticker}): {e}")
        return pd.DataFrame()

Z_SCORE_WINDOWS = {"1M": 21, "3M": 63, "1Y": 252}  # trading days - same convention as the CIX
                                                     # backtester's dependent-variable z-score

def calculate_price_zscores(series: pd.Series, windows=Z_SCORE_WINDOWS) -> dict:
    """Latest rolling z-score of a price level against its own trailing history, per window -
    how stretched a ticker is relative to its own recent range, not a return z-score."""
    out = {}
    for label, window in windows.items():
        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std()
        z = (series - roll_mean) / roll_std
        out[label] = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None
    return out

def calculate_period_returns(series: pd.Series, windows=Z_SCORE_WINDOWS) -> dict:
    """% return over each trailing window (1M/3M/1Y trading days), independent of the global
    date-range slider - same fixed-window convention as calculate_price_zscores, so the return
    and z-score bars for the same ticker are directly comparable side by side."""
    out = {}
    for label, window in windows.items():
        if len(series) <= window:
            out[label] = None
            continue
        past = series.iloc[-1 - window]
        out[label] = float((series.iloc[-1] / past - 1) * 100) if past else None
    return out

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Cons. Discretionary", "XLP": "Cons. Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Comm. Services",
}
MARKET_INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^RUT": "Russell 2000", "^DJI": "Dow Jones"}

# ── Oil & Gas (EIA) ──────────────────────────────────────────────────────────────
EIA_BASE = "https://api.eia.gov/v2"

def _eia_get(path, params, retries=3, backoff=1.5):
    params = {**params, "api_key": EIA_API_KEY}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{EIA_BASE}/{path}", params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    raise last_exc

@st.cache_data(ttl=21600)  # SPR data only updates weekly
def get_spr_level() -> pd.DataFrame:
    """U.S. Strategic Petroleum Reserve crude oil ending stocks, weekly, full history since
    1982 - EIA series WCSSTUS1 (thousand barrels, converted to million barrels here)."""
    if not EIA_API_KEY:
        st.warning("EIA_API_KEY not set - Oil & Gas tab needs a free key from eia.gov/opendata/register.php")
        return pd.DataFrame()
    try:
        all_rows, offset = [], 0
        while True:
            j = _eia_get("petroleum/stoc/wstk/data/", {
                "frequency": "weekly", "data[0]": "value", "facets[series][]": "WCSSTUS1",
                "sort[0][column]": "period", "sort[0][direction]": "asc",
                "length": 5000, "offset": offset,
            })
            rows = j["response"]["data"]
            all_rows.extend(rows)
            if len(rows) < 5000:
                break
            offset += 5000
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["period"])
        df["SPR (Million Barrels)"] = pd.to_numeric(df["value"], errors="coerce") / 1000
        out = df.set_index("date")[["SPR (Million Barrels)"]].sort_index()
        return out
    except Exception as e:
        st.warning(f"Could not load SPR data: {e}")
        return pd.DataFrame()

# Statutory floor for the SPR's non-emergency ("routine") drawdown authority - 42 U.S.C.
# 6241(h): petroleum cannot be drawn down under this authority "if there are fewer than
# 252,400,000 barrels ... in the Reserve", and a drawdown under this specific subsection
# additionally requires "the Secretary of Defense has found that action taken under this
# subsection will not impair national security" - the "SecDef-authorized floor". This does
# NOT apply to the President's separate emergency drawdown authority (subsection (d)), which
# has no statutory minimum. Verified live against the US Code (Cornell Law, 42 USC 6241).
SPR_SECDEF_FLOOR = 252.4  # million barrels

def estimate_time_to_floor(series: pd.Series, floor: float, freq: str = "weekly") -> dict:
    """Projects weeks (or months) until `series` reaches `floor` at the current drawdown pace.
    Weekly series: drawdown rate = 4-period moving average of the period-over-period change
    (smooths weekly noise). Monthly series: drawdown rate = latest single month's change only
    (no averaging - one data point per month is already a smoothed read). Returns
    periods_to_floor=None if the series isn't currently declining (undefined ETA)."""
    diffs = series.diff().dropna()
    if diffs.empty:
        return {"rate_per_period": None, "periods_to_floor": None, "eta_date": None}
    rate = diffs.tail(4).mean() if freq == "weekly" else diffs.iloc[-1]
    current = series.iloc[-1]
    if rate >= 0:  # flat or refilling - no defined time-to-floor
        return {"rate_per_period": float(rate), "periods_to_floor": None, "eta_date": None}
    periods = max((current - floor) / abs(rate), 0)
    unit_days = 7 if freq == "weekly" else 30
    eta_date = series.index[-1] + pd.Timedelta(days=periods * unit_days)
    return {"rate_per_period": float(rate), "periods_to_floor": float(periods), "eta_date": eta_date}

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

    try:
        d2  = fred.get_series("DGS2").dropna()
        d5  = fred.get_series("DGS5").dropna()
        d10 = fred.get_series("DGS10").dropna()
        d30 = fred.get_series("DGS30").dropna()
        curve = pd.concat([d2, d5, d10, d30], axis=1, keys=["2Y", "5Y", "10Y", "30Y"]).ffill().dropna()
        fly = (2 * curve["5Y"] - curve["10Y"] - curve["2Y"]) * 100  # bps
        results["2s5s10s Fly"] = (float(fly.iloc[-1]), float(fly.iloc[-1] - fly.iloc[-2]), "bps")
        curve_5s30s = (curve["30Y"] - curve["5Y"]) * 100  # bps
        results["5s30s"] = (float(curve_5s30s.iloc[-1]), float(curve_5s30s.iloc[-1] - curve_5s30s.iloc[-2]), "bps")
    except Exception:
        results["2s5s10s Fly"] = (None, None, "")
        results["5s30s"] = (None, None, "")

    return results

# ── Economic calendar ───────────────────────────────────────────────────────────
# Release SCHEDULE dates come from FRED's release/dates endpoint (same API key, no
# scraping) - it tracks the official publication calendar for each release, including
# future not-yet-happened dates when include_release_dates_with_no_data=true is set.
# There's no free source for economist CONSENSUS estimates (that's commercial data -
# Bloomberg/Refinitiv/TradingEconomics paid tier), so this shows next date + previous
# actual, not a forecast-vs-actual surprise calendar.
ECONOMIC_RELEASES = [
    # (Event name, FRED release_id, FRED series_id, calc type, unit)
    ("Nonfarm Payrolls",       50,  "PAYEMS",   "diff_k",  "k"),
    ("Initial Jobless Claims", 180, "ICSA",     "level",   ""),
    ("JOLTS Job Openings",     192, "JTSJOL",   "level",   "k"),
    ("Consumer Price Index",   10,  "CPIAUCSL", "pct_mom", "%"),
    ("Producer Price Index",   46,  "PPIACO",   "pct_mom", "%"),
    ("PCE Price Index",        54,  "PCEPI",    "pct_mom", "%"),
    ("Retail Sales",           9,   "RSAFS",    "pct_mom", "%"),
    ("GDP (QoQ SAAR)",         53,  "GDPC1",    "qoq_saar","%"),
    ("Housing Starts",         27,  "HOUST",    "level",   "k"),
]

@st.cache_data(ttl=86400)  # release schedules only get revised a few times a year
def get_next_release_date(release_id):
    try:
        url = (f"https://api.stlouisfed.org/fred/release/dates?release_id={release_id}"
               f"&include_release_dates_with_no_data=true&sort_order=desc&limit=10"
               f"&file_type=json&api_key={os.getenv('FRED_API_KEY')}")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        today_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        future = sorted(d["date"] for d in r.json().get("release_dates", []) if d["date"] >= today_str)
        return pd.Timestamp(future[0]) if future else None
    except Exception:
        return None

@st.cache_data(ttl=21600)
def get_economic_release_calendar():
    rows = []
    for name, release_id, series_id, calc, unit in ECONOMIC_RELEASES:
        next_date = get_next_release_date(release_id)
        if next_date is None:
            continue
        try:
            s = fred.get_series(series_id).dropna()
            latest, prev = float(s.iloc[-1]), float(s.iloc[-2])
            if calc == "diff_k":
                prev_val = s.diff().iloc[-1]
            elif calc == "pct_mom":
                prev_val = s.pct_change().iloc[-1] * 100
            elif calc == "qoq_saar":
                prev_val = ((latest / prev) ** 4 - 1) * 100
            else:
                prev_val = latest
            detail = f"Prev: {prev_val:+,.1f}{unit} ({s.index[-1].strftime('%b %Y')})"
        except Exception:
            detail = ""
        rows.append({"Date": next_date, "Event": name, "Type": "Economic Release", "Detail": detail})
    return pd.DataFrame(rows)

with st.spinner("Loading summary metrics…"):
    summary = get_summary_metrics(END)

summary_items = list(summary.items())
ROW_SIZE = 6
for row_start in range(0, len(summary_items), ROW_SIZE):
    row_items = summary_items[row_start:row_start + ROW_SIZE]
    cols = st.columns(ROW_SIZE)
    for col, (name, (val, delta, unit)) in zip(cols, row_items):
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
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Prices",
    "Oil & Gas",
    "Labour Market",
    "Housing",
    "Treasury & Rates",
    "US Markets",
    "Fiscal",
    "Indicators",
    "Economic Calendar",
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

        # CPI components - the standard BLS major expenditure groups (SA), each verified
        # live against FRED. Not exhaustive (BLS publishes finer subcomponents too), but this
        # is the standard "CPI component breakdown" level of granularity.
        CPI_COMPONENTS = [
            ("Food",                       "CPIUFDSL"),
            ("Energy",                     "CPIENGSL"),
            ("Shelter",                    "CUSR0000SAH1"),
            ("Apparel",                    "CPIAPPSL"),
            ("Transportation",             "CPITRNSL"),
            ("Medical Care",               "CPIMEDSL"),
            ("Recreation",                 "CPIRECSL"),
            ("Education & Communication",  "CPIEDUSL"),
            ("Other Goods & Services",     "CPIOGSSL"),
        ]
        cpi_components = {label: mom_yoy(fetch(sid, label, START, END), label) for label, sid in CPI_COMPONENTS}

        # PCE components - BEA's monthly PCE price index release is coarser than BLS's CPI
        # release: it only splits Goods/Services (each further split Durable/Nondurable, plus
        # Food and Energy as cross-cutting cuts) at monthly frequency. The finer CPI-style
        # categories (Housing, Health Care, Transportation Services, Recreation Services etc.)
        # only exist in BEA's underlying-detail tables at quarterly/annual frequency, so
        # they're left out here rather than mixed with monthly series at a different cadence.
        PCE_COMPONENTS = [
            ("Goods",                          "DGDSRG3M086SBEA"),
            ("Services",                       "DSERRG3M086SBEA"),
            ("Durable Goods",                  "DDURRG3M086SBEA"),
            ("Nondurable Goods",                "DNDGRG3M086SBEA"),
            ("Food",                           "DFXARG3M086SBEA"),
            ("Energy Goods & Services",        "DNRGRG3M086SBEA"),
            ("Services Excl. Energy & Housing","IA001260M"),
        ]
        pce_components = {label: mom_yoy(fetch(sid, label, START, END), label) for label, sid in PCE_COMPONENTS}

        # PCE weights - nominal-dollar expenditure shares (current $, distinct from the
        # chain-type price index series above used for MoM/YoY). These update monthly via
        # FRED, unlike CPI's BLS relative-importance weights, which are a static annual table
        # published only as an HTML page BLS blocks scripted access to (confirmed live: direct
        # curl and WebFetch both 403, even against the raw flat-file mirror) - PCE is the only
        # side of this that's realistically live-refreshable, hence weighting PCE only.
        PCE_WEIGHT_SERIES = {
            "Services":                        "PCES",
            "Durable Goods":                   "PCEDG",
            "Nondurable Goods":                "PCEND",
            "Food":                            "DFXARC1M027SBEA",
            "Energy Goods & Services":         "DNRGRC1M027SBEA",
            "Services Excl. Energy & Housing": "LA001260M",  # millions of $; the rest are billions
        }
        pce_total_df = fetch("PCE", "PCE Total", START, END)
        pce_weights = {}
        if not pce_total_df.empty:
            pce_total = pce_total_df["PCE Total"].dropna().iloc[-1]
            pce_levels = {}
            for label, sid in PCE_WEIGHT_SERIES.items():
                df_w = fetch(sid, f"{label} $", START, END)
                if df_w.empty:
                    continue
                val = df_w[f"{label} $"].dropna().iloc[-1]
                if sid == "LA001260M":
                    val = val / 1000  # millions -> billions, to match the other series' units
                pce_levels[label] = val
                pce_weights[label] = val / pce_total * 100
            if "Durable Goods" in pce_levels and "Nondurable Goods" in pce_levels:
                pce_weights["Goods"] = (pce_levels["Durable Goods"] + pce_levels["Nondurable Goods"]) / pce_total * 100

        # GDP vs 30Y Treasury yield - Real & Nominal GDP YoY (quarterly pct_change(4), since
        # mom_yoy() above assumes monthly cadence) plotted as grouped bars against the 30Y
        # yield as a line, all on one shared % axis (levels would need a $ vs % dual axis,
        # which fights against reading the yield line cleanly against the bars).
        real_gdp = fetch("GDPC1", "Real GDP", START, END)
        nom_gdp  = fetch("GDP", "Nominal GDP", START, END)
        gdp_dgs30 = fetch("DGS30", "30Y Treasury", START, END)
        real_gdp_yoy = (real_gdp["Real GDP"].pct_change(4) * 100).round(3).dropna() if not real_gdp.empty else pd.Series(dtype=float)
        nom_gdp_yoy  = (nom_gdp["Nominal GDP"].pct_change(4) * 100).round(3).dropna() if not nom_gdp.empty else pd.Series(dtype=float)
        gdp_y30_aligned = pd.Series(dtype=float)
        if not gdp_dgs30.empty and not real_gdp_yoy.empty:
            dgs30_sorted = gdp_dgs30.rename(columns={"30Y Treasury": "y30"}).dropna().reset_index().sort_values("date")
            gdp_dates = pd.DataFrame({"date": real_gdp_yoy.index}).sort_values("date")
            gdp_y30_aligned = pd.merge_asof(gdp_dates, dgs30_sorted, on="date", direction="forward").set_index("date")["y30"]

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
    fig_be.update_layout(**base_layout("Inflation Expectations (TIPS Breakevens)"))
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

    # CPI components - YoY % history (all 9 groups) + latest MoM/YoY snapshot
    fig_cpi_comp_hist = go.Figure()
    for label, _ in CPI_COMPONENTS:
        df_c = cpi_components[label]
        col = f"{label} YoY %"
        if not df_c.empty and col in df_c.columns:
            fig_cpi_comp_hist.add_trace(go.Scatter(x=df_c.index, y=df_c[col], name=label, mode="lines"))
    fig_cpi_comp_hist.update_layout(**base_layout("CPI Components — YoY %"))
    fig_cpi_comp_hist.update_yaxes(ticksuffix="%")
    add_recessions(fig_cpi_comp_hist, recessions)

    comp_rows = []
    for label, _ in CPI_COMPONENTS:
        df_c = cpi_components[label]
        mom_col, yoy_col = f"{label} MoM %", f"{label} YoY %"
        if df_c.empty or mom_col not in df_c.columns:
            continue
        mom_s, yoy_s = df_c[mom_col].dropna(), df_c[yoy_col].dropna()
        if mom_s.empty or yoy_s.empty:
            continue
        comp_rows.append({"Component": label, "MoM %": mom_s.iloc[-1], "YoY %": yoy_s.iloc[-1], "As Of": df_c.index[-1]})
    comp_df = pd.DataFrame(comp_rows).sort_values("YoY %", ascending=True)
    comp_latest_date = comp_df["As Of"].max().strftime("%b %Y") if not comp_df.empty else ""
    comp_df = comp_df.drop(columns="As Of")

    fig_cpi_comp_snap = go.Figure()
    fig_cpi_comp_snap.add_trace(go.Bar(y=comp_df["Component"], x=comp_df["YoY %"], name="YoY %",
                                        orientation="h", marker_color="#ef5350"))
    fig_cpi_comp_snap.add_trace(go.Bar(y=comp_df["Component"], x=comp_df["MoM %"], name="MoM %",
                                        orientation="h", marker_color="#ff9800"))
    fig_cpi_comp_snap.update_layout(**base_layout(f"CPI Components — Latest MoM & YoY % ({comp_latest_date})", height=420))
    fig_cpi_comp_snap.update_layout(barmode="group")
    fig_cpi_comp_snap.update_xaxes(ticksuffix="%")

    # PCE components - YoY % history (7 groups) + latest MoM/YoY snapshot, same pairing as CPI
    fig_pce_comp_hist = go.Figure()
    for label, _ in PCE_COMPONENTS:
        df_c = pce_components[label]
        col = f"{label} YoY %"
        if not df_c.empty and col in df_c.columns:
            fig_pce_comp_hist.add_trace(go.Scatter(x=df_c.index, y=df_c[col], name=label, mode="lines"))
    fig_pce_comp_hist.update_layout(**base_layout("PCE Components — YoY %"))
    fig_pce_comp_hist.update_yaxes(ticksuffix="%")
    add_recessions(fig_pce_comp_hist, recessions)

    pce_comp_rows = []
    for label, _ in PCE_COMPONENTS:
        df_c = pce_components[label]
        mom_col, yoy_col = f"{label} MoM %", f"{label} YoY %"
        if df_c.empty or mom_col not in df_c.columns:
            continue
        mom_s, yoy_s = df_c[mom_col].dropna(), df_c[yoy_col].dropna()
        if mom_s.empty or yoy_s.empty:
            continue
        mom_latest, yoy_latest = mom_s.iloc[-1], yoy_s.iloc[-1]
        weight = pce_weights.get(label)
        pce_comp_rows.append({
            "Component": label, "MoM %": mom_latest, "YoY %": yoy_latest,
            "Weight %": weight,
            "Weighted MoM (pp)": weight / 100 * mom_latest if weight is not None else None,
            "Weighted YoY (pp)": weight / 100 * yoy_latest if weight is not None else None,
            "As Of": df_c.index[-1],
        })
    pce_comp_df = pd.DataFrame(pce_comp_rows).sort_values("YoY %", ascending=True)
    pce_comp_latest_date = pce_comp_df["As Of"].max().strftime("%b %Y") if not pce_comp_df.empty else ""
    pce_comp_df = pce_comp_df.drop(columns="As Of")

    # Weight-adjusted bars use nominal-$ PCE expenditure shares (pce_weights, fetched above).
    # "Goods" and "Services Excl. Energy & Housing" overlap other rows (Goods = Durable +
    # Nondurable; Services Excl. Energy & Housing is a subset of Services) - included for
    # visibility, but not meant to be summed together with the rows they overlap.
    fig_pce_comp_snap = go.Figure()
    fig_pce_comp_snap.add_trace(go.Bar(y=pce_comp_df["Component"], x=pce_comp_df["YoY %"], name="YoY %",
                                        orientation="h", marker_color="#26a69a"))
    fig_pce_comp_snap.add_trace(go.Bar(y=pce_comp_df["Component"], x=pce_comp_df["MoM %"], name="MoM %",
                                        orientation="h", marker_color="#80cbc4"))
    fig_pce_comp_snap.add_trace(go.Bar(y=pce_comp_df["Component"], x=pce_comp_df["Weighted YoY (pp)"], name="Weighted YoY (pp)",
                                        orientation="h", marker_color="#f5a24c"))
    fig_pce_comp_snap.add_trace(go.Bar(y=pce_comp_df["Component"], x=pce_comp_df["Weighted MoM (pp)"], name="Weighted MoM (pp)",
                                        orientation="h", marker_color="#ffcc80"))
    fig_pce_comp_snap.update_layout(**base_layout(f"PCE Components — Latest MoM & YoY % + Weight-Adjusted Contribution (pp) ({pce_comp_latest_date})", height=440))
    fig_pce_comp_snap.update_layout(barmode="group")
    fig_pce_comp_snap.update_xaxes(ticksuffix="%", title="% (raw) / pp (weight-adjusted contribution)")

    # Real & Nominal GDP (YoY %) vs 30Y Treasury yield - GDP as grouped bars, yield as an
    # overlaid line, all on one shared % axis so the growth-vs-borrowing-cost read (is nominal
    # GDP outgrowing the 30Y yield) doesn't require a dual-axis.
    fig_gdp_30y = go.Figure()
    fig_gdp_30y.add_trace(go.Bar(x=real_gdp_yoy.index, y=real_gdp_yoy.values, name="Real GDP YoY %", marker_color="#4c8bf5"))
    fig_gdp_30y.add_trace(go.Bar(x=nom_gdp_yoy.index, y=nom_gdp_yoy.values, name="Nominal GDP YoY %", marker_color="#f5a24c"))
    if not gdp_y30_aligned.empty:
        fig_gdp_30y.add_trace(go.Scatter(x=gdp_y30_aligned.index, y=gdp_y30_aligned.values, name="30Y Treasury Yield",
                                          mode="lines+markers", line=dict(color="#c85fd6", width=2)))
    fig_gdp_30y.update_layout(**base_layout("Real & Nominal GDP (YoY %) vs. 30Y Treasury Yield"))
    fig_gdp_30y.update_layout(barmode="group")
    fig_gdp_30y.update_yaxes(ticksuffix="%")
    add_recessions(fig_gdp_30y, recessions)

    inflation_charts = [
        ("CPI vs Core CPI", fig_cpi, pd.concat([cpi, core_cpi], axis=1)),
        ("PCE vs Core PCE", fig_pce, pd.concat([pce, core_pce], axis=1)),
        ("PPI", fig_ppi, ppi),
        ("Retail Sales", fig_rsa, rsa),
        ("TIPS Breakevens", fig_be, pd.concat([be_5y, be_10y], axis=1)),
        ("UMich & Inflation Expectations", fig_umich, pd.concat([umich, inf_exp1, inf_exp5], axis=1)),
        ("Export & Import Prices", fig_pi, pd.concat([exp_pi, imp_pi], axis=1)),
        ("CPI Components History", fig_cpi_comp_hist, pd.concat([cpi_components[l] for l, _ in CPI_COMPONENTS], axis=1)),
        ("CPI Components Snapshot", fig_cpi_comp_snap, comp_df),
        ("PCE Components History", fig_pce_comp_hist, pd.concat([pce_components[l] for l, _ in PCE_COMPONENTS], axis=1)),
        ("PCE Components Snapshot", fig_pce_comp_snap, pce_comp_df),
        ("GDP vs 30Y Treasury Yield", fig_gdp_30y, pd.DataFrame({
            "Real GDP YoY %": real_gdp_yoy, "Nominal GDP YoY %": nom_gdp_yoy, "30Y Treasury Yield": gdp_y30_aligned,
        })),
    ]
    render_two_col(inflation_charts)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Oil & Gas
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Oil & Gas")
    st.caption("Strategic Petroleum Reserve - EIA (api.eia.gov), not FRED. A different data domain "
               "from the rest of this dashboard.")

    with st.spinner("Loading SPR data…"):
        spr = get_spr_level()

    if spr.empty:
        st.info("SPR data unavailable this run.")
    else:
        latest_level = spr["SPR (Million Barrels)"].iloc[-1]
        ath = spr["SPR (Million Barrels)"].max()
        ath_date = spr["SPR (Million Barrels)"].idxmax()
        pct_of_ath = latest_level / ath * 100
        proj = estimate_time_to_floor(spr["SPR (Million Barrels)"], SPR_SECDEF_FLOOR, freq="weekly")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Latest SPR Level", f"{latest_level:,.1f}M bbls", help=f"As of {spr.index[-1].strftime('%b %d, %Y')}")
        with m2:
            st.metric("All-Time High", f"{ath:,.1f}M bbls", help=f"{ath_date.strftime('%b %Y')}")
        with m3:
            st.metric("% of All-Time High", f"{pct_of_ath:.1f}%")
        with m4:
            if proj["periods_to_floor"] is not None:
                weeks = proj["periods_to_floor"]
                st.metric("Est. Time to SecDef Floor", f"{weeks:,.0f} weeks",
                         delta=f"~{proj['eta_date'].strftime('%b %Y')}", delta_color="off",
                         help=f"Floor: {SPR_SECDEF_FLOOR:.1f}M bbls (42 USC 6241(h), the non-emergency "
                              f"drawdown minimum requiring Secretary of Defense sign-off). Drawdown rate: "
                              f"4-week moving average of the weekly change, currently {proj['rate_per_period']:+.2f}M bbls/week.")
            else:
                st.metric("Est. Time to SecDef Floor", "N/A",
                         help="Not currently drawing down (4-week average weekly change is flat or positive), "
                              "so time-to-floor is undefined.")

        st.caption(f"**SecDef-authorized floor: {SPR_SECDEF_FLOOR:.1f}M bbls** — the statutory minimum for the "
                   f"SPR's non-emergency drawdown authority (42 U.S.C. § 6241(h)); a drawdown under this "
                   f"specific authority requires the Secretary of Defense to confirm it \"will not impair "
                   f"national security.\" This does not apply to the President's separate emergency-drawdown "
                   f"authority, which has no statutory minimum.")

        fig_spr = go.Figure()
        fig_spr.add_trace(go.Scatter(x=spr.index, y=spr["SPR (Million Barrels)"], name="SPR Level",
                                     line=dict(color="#e08b4f"), fill="tozeroy", fillcolor="rgba(224,139,79,0.12)"))
        fig_spr.add_hline(y=SPR_SECDEF_FLOOR, line_dash="dash", line_color="#ef5350",
                          annotation_text=f"SecDef Floor ({SPR_SECDEF_FLOOR:.1f}M)", annotation_position="bottom right")
        fig_spr.update_layout(**base_layout("SPR Level — Full History (Weekly)"))
        fig_spr.update_yaxes(title="Million Barrels")
        add_recessions(fig_spr, recessions)

        spr_recent = spr[spr.index >= (spr.index[-1] - pd.DateOffset(years=2))].copy()
        spr_recent["WoW Change"] = spr_recent["SPR (Million Barrels)"].diff()
        spr_recent["4W MA"] = spr_recent["WoW Change"].rolling(4).mean()
        fig_spr_chg = go.Figure()
        fig_spr_chg.add_trace(go.Bar(x=spr_recent.index, y=spr_recent["WoW Change"],
                                     marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in spr_recent["WoW Change"].fillna(0)],
                                     name="Weekly Change", opacity=0.6))
        fig_spr_chg.add_trace(go.Scatter(x=spr_recent.index, y=spr_recent["4W MA"],
                                         name="4W MA (Drawdown Rate)", line=dict(color="white", width=2)))
        fig_spr_chg.add_hline(y=0, line_dash="dot", line_color="#555")
        fig_spr_chg.update_layout(**base_layout("SPR Weekly Change + 4W Moving Average (Last 2Y)"))
        fig_spr_chg.update_yaxes(title="Million Barrels")

        render_two_col([
            ("SPR Level", fig_spr, spr),
            ("SPR Weekly Change", fig_spr_chg, spr_recent),
        ])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Labour Market
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
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
        fig_bev.update_layout(**base_layout("Beveridge Curve (Openings vs Unemployment)"))
        fig_bev.update_xaxes(title="Unemployment Rate (%)", gridcolor=GRID_COLOR)
        fig_bev.update_yaxes(title="Job Openings (M)", gridcolor=GRID_COLOR)
    else:
        fig_bev = go.Figure()

    # ADP sectors
    fig_adp = go.Figure()
    for col in adp_sectors.columns:
        fig_adp.add_trace(go.Scatter(x=adp_sectors.index, y=adp_sectors[col], name=col, mode="lines"))
    fig_adp.update_layout(**base_layout("ADP Employment by Sector (MoM, k)"))
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
# TAB 4 — Housing
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Housing")
    with st.spinner("Loading housing data…"):
        home_sales  = fetch("EXHOSLUSM495S", "Existing Home Sales", START, END)
        new_sales   = fetch("HSN1F",         "New Home Sales", START, END)
        starts      = fetch("HOUST",         "Housing Starts", START, END)
        permits     = fetch("PERMIT",        "Building Permits", START, END)
        completions = fetch("COMPUTSA",      "Completions", START, END)
        case_shiller = fetch("CSUSHPINSA",   "Case-Shiller HPI", START, END)

        mort30      = fetch("MORTGAGE30US", "30Y Fixed Mortgage", START, END)
        mort15      = fetch("MORTGAGE15US", "15Y Fixed Mortgage", START, END)
        dgs30_h     = fetch("DGS30",        "30Y Treasury", START, END)
        new_supply  = fetch("MSACSR",       "New Home Months Supply", START, END)
        existing_supply = fetch("HOSSUPUSM673N", "Existing Home Months Supply", START, END)
        median_price = fetch("MSPUS",       "Median New-Home Price", START, END)
        homeownership = fetch("RHORUSQ156N", "Homeownership Rate", START, END)
        affordability = fetch("FIXHAI",     "Housing Affordability Index", START, END)

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

    # Starts vs Permits vs Completions - full construction pipeline; a widening
    # permits-to-completions gap signals a building backlog before it shows up elsewhere.
    fig_starts = go.Figure()
    if not starts.empty:
        fig_starts.add_trace(go.Scatter(x=starts.index, y=starts["Housing Starts"],
                                        name="Starts", line=dict(color="#42a5f5")))
    if not permits.empty:
        fig_starts.add_trace(go.Scatter(x=permits.index, y=permits["Building Permits"],
                                        name="Permits", line=dict(color="#ab47bc")))
    if not completions.empty:
        fig_starts.add_trace(go.Scatter(x=completions.index, y=completions["Completions"],
                                        name="Completions", line=dict(color="#26a69a")))
    fig_starts.update_layout(**base_layout("Housing Starts, Permits & Completions"))
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

    # 30Y vs 15Y fixed mortgage rate
    fig_mortgage = go.Figure()
    if not mort30.empty:
        fig_mortgage.add_trace(go.Scatter(x=mort30.index, y=mort30["30Y Fixed Mortgage"],
                                          name="30Y Fixed", line=dict(color="#42a5f5")))
    if not mort15.empty:
        fig_mortgage.add_trace(go.Scatter(x=mort15.index, y=mort15["15Y Fixed Mortgage"],
                                          name="15Y Fixed", line=dict(color="#8a94a6")))
    fig_mortgage.update_layout(**base_layout("30Y vs 15Y Fixed Mortgage Rate"))
    fig_mortgage.update_yaxes(ticksuffix="%")
    add_recessions(fig_mortgage, recessions)

    # Mortgage-Treasury spread - the standard MBS-market-stress read (30Y mortgage over
    # 30Y Treasury, matching maturities). Peaked well above its historical norm during the
    # 2022-23 stress episode and has been normalizing since.
    mort_spread = pd.DataFrame()
    if not mort30.empty and not dgs30_h.empty:
        spread_df = pd.concat([mort30, dgs30_h], axis=1).ffill().dropna()
        mort_spread = pd.DataFrame(index=spread_df.index)
        mort_spread["Mortgage-Treasury Spread (bps)"] = (
            spread_df["30Y Fixed Mortgage"] - spread_df["30Y Treasury"]) * 100
    fig_mort_spread = go.Figure()
    if not mort_spread.empty:
        fig_mort_spread.add_trace(go.Scatter(x=mort_spread.index, y=mort_spread["Mortgage-Treasury Spread (bps)"],
                                             name="Spread", line=dict(color="#e08b4f"), fill="tozeroy",
                                             fillcolor="rgba(224,139,79,0.15)"))
    fig_mort_spread.update_layout(**base_layout("Mortgage — 30Y Treasury Spread"))
    fig_mort_spread.update_yaxes(ticksuffix=" bps")
    add_recessions(fig_mort_spread, recessions)

    # New home months' supply
    fig_new_supply = go.Figure()
    if not new_supply.empty:
        fig_new_supply.add_trace(go.Scatter(x=new_supply.index, y=new_supply["New Home Months Supply"],
                                            name="Months Supply", line=dict(color="#e08b4f"), fill="tozeroy",
                                            fillcolor="rgba(224,139,79,0.15)"))
    fig_new_supply.update_layout(**base_layout("New Home Months' Supply"))
    add_recessions(fig_new_supply, recessions)

    # Existing home months' supply - NAR restricted data redistribution in 2023-24, so FRED
    # restarted this series from scratch; only a short window of history is available.
    fig_existing_supply = go.Figure()
    if not existing_supply.empty:
        fig_existing_supply.add_trace(go.Scatter(x=existing_supply.index, y=existing_supply["Existing Home Months Supply"],
                                                  name="Months Supply", line=dict(color="#26a69a"), fill="tozeroy",
                                                  fillcolor="rgba(38,166,154,0.15)"))
    fig_existing_supply.update_layout(**base_layout("Existing Home Months' Supply"))

    # Median new-home price vs Case-Shiller - two panels, not one dual-axis chart: different
    # units and scales. Case-Shiller says how much prices moved (%); this says what a home
    # actually costs ($).
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Median New-Home Price ($)", "Case-Shiller Index"),
                              vertical_spacing=0.12)
    if not median_price.empty:
        fig_price.add_trace(go.Scatter(x=median_price.index, y=median_price["Median New-Home Price"],
                                       name="Median Price", line=dict(color="#e08b4f")), row=1, col=1)
    if not case_shiller.empty:
        fig_price.add_trace(go.Scatter(x=case_shiller.index, y=case_shiller["Case-Shiller HPI"],
                                       name="Case-Shiller", line=dict(color="#8a94a6")), row=2, col=1)
    fig_price.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                            height=500, margin=dict(l=10,r=10,t=45,b=30), showlegend=False)
    fig_price.update_xaxes(gridcolor=GRID_COLOR)
    fig_price.update_yaxes(gridcolor=GRID_COLOR)
    add_recessions(fig_price, recessions, rows=[1,2], cols=[1,1])

    # Homeownership rate - slow-moving structural context, not a series people watch move
    # month to month.
    fig_homeownership = go.Figure()
    if not homeownership.empty:
        fig_homeownership.add_trace(go.Scatter(x=homeownership.index, y=homeownership["Homeownership Rate"],
                                                name="Homeownership Rate", line=dict(color="#90a4d4")))
    fig_homeownership.update_layout(**base_layout("Homeownership Rate"))
    fig_homeownership.update_yaxes(ticksuffix="%")
    add_recessions(fig_homeownership, recessions)

    # Housing affordability index - same NAR restart issue as existing-home supply; short
    # history, useful as a current reading (100 = median family exactly qualifies for the
    # median-priced home) more than a trend.
    fig_afford = go.Figure()
    if not affordability.empty:
        fig_afford.add_trace(go.Scatter(x=affordability.index, y=affordability["Housing Affordability Index"],
                                        name="Affordability Index", line=dict(color="#ab47bc")))
    fig_afford.add_hline(y=100, line_dash="dot", line_color="#555",
                          annotation_text="100 = median family qualifies")
    fig_afford.update_layout(**base_layout("Housing Affordability Index"))

    housing_charts = [
        ("Home Sales", fig_sales, pd.concat([home_sales, new_sales], axis=1)),
        ("Starts, Permits & Completions", fig_starts, pd.concat([starts, permits, completions], axis=1)),
        ("Case-Shiller HPI", fig_cs, cs_mom),
        ("30Y vs 15Y Mortgage Rate", fig_mortgage, pd.concat([mort30, mort15], axis=1)),
        ("Mortgage-Treasury Spread", fig_mort_spread, mort_spread),
        ("New Home Months Supply", fig_new_supply, new_supply),
        ("Existing Home Months Supply", fig_existing_supply, existing_supply),
        ("Median Price vs Case-Shiller", fig_price, pd.concat([median_price, case_shiller], axis=1)),
        ("Homeownership Rate", fig_homeownership, homeownership),
        ("Housing Affordability Index", fig_afford, affordability),
    ]
    render_two_col(housing_charts)
    st.caption("Existing Home Months' Supply and the Housing Affordability Index both restart in "
               "2025 - NAR restricted redistribution of this data in 2023-24, and FRED rebuilt these "
               "series from scratch once a new agreement was reached. The same gap affects the "
               "Existing Home Sales series above, which also only carries recent history.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Monetary & Rates
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
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
    fig_fed.update_layout(**base_layout("Fed Balance Sheet ($T)"))
    fig_fed.update_yaxes(ticksuffix="T")
    add_recessions(fig_fed, recessions)

    # M2
    m2_data = pd.DataFrame(index=m2.index) if not m2.empty else pd.DataFrame()
    if not m2.empty:
        m2_data["M2 (Billions)"] = m2["M2"] / 1e3
        m2_data["YoY %"] = (m2["M2"].pct_change(12) * 100).round(3)
        m2_data["MoM %"] = (m2["M2"].pct_change() * 100).round(3)
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
    fig_rates.update_layout(**base_layout("Policy Rates"))
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
        spreads["2s10s"]   = (yc.get("10Y",pd.Series()) - yc.get("2Y",pd.Series())) * 100
        spreads["10Y-3M"]  = (yc.get("10Y",pd.Series()) - yc.get("3M",pd.Series())) * 100
        spreads["2s5s"]    = (yc.get("5Y",pd.Series())  - yc.get("2Y",pd.Series())) * 100
        spreads["10s30s"]  = (yc.get("30Y",pd.Series()) - yc.get("10Y",pd.Series())) * 100
        spreads["5s30s"]   = (yc.get("30Y",pd.Series()) - yc.get("5Y",pd.Series())) * 100
        # Butterfly: 2x the belly minus both wings - same definition as the "2s5s10s Fly"
        # summary metric above, just as a full time series instead of a single latest figure.
        spreads["2s5s10s Fly"] = (2 * yc.get("5Y",pd.Series()) - yc.get("10Y",pd.Series()) - yc.get("2Y",pd.Series())) * 100
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
    fig_real.update_layout(**base_layout("Real Yields vs Breakevens"))
    add_recessions(fig_real, recessions)

    # Credit spreads - FRED reports these OAS series in percentage points, not bps.
    # IG and HY now share one axis (both bps, directly comparable); the HY-IG differential
    # gets its own axis since it moves on a much smaller scale than either level.
    fig_credit = go.Figure()
    if not ig_oas.empty:
        fig_credit.add_trace(go.Scatter(x=ig_oas.index, y=ig_oas["IG OAS"] * 100,
                                        name="IG OAS", line=dict(color="#26a69a"), yaxis="y"))
    if not hy_oas.empty:
        fig_credit.add_trace(go.Scatter(x=hy_oas.index, y=hy_oas["HY OAS"] * 100,
                                        name="HY OAS", line=dict(color="#ef5350"), yaxis="y"))
    credit_diff_df = pd.DataFrame()
    if not ig_oas.empty and not hy_oas.empty:
        credit_diff = ((hy_oas["HY OAS"] - ig_oas["IG OAS"]) * 100).dropna()
        credit_diff_df = pd.DataFrame({"HY-IG Diff (bps)": credit_diff})
        fig_credit.add_trace(go.Scatter(x=credit_diff.index, y=credit_diff.values,
                                        name="HY − IG Differential", line=dict(color="#ff9800", dash="dot"), yaxis="y2"))
    fig_credit.update_layout(**dual_axis_layout("Credit Spreads — IG & HY OAS vs. HY−IG Differential (bps)", "IG & HY OAS (bps)", "HY − IG Differential (bps)"))
    add_recessions(fig_credit, recessions)

    monetary_charts = [
        ("Fed Balance Sheet", fig_fed, pd.concat([fed_total, fed_tres], axis=1)),
        ("M2 Money Supply",   fig_m2,  m2_data),
        ("Policy Rates",      fig_rates, pd.concat([effr, sofr, iorb, rrp], axis=1)),
        ("Yield Curve",       fig_yc,  yc),
        ("Yield Curve Changes", fig_yc_chg, None),
        ("Treasury Spreads",  fig_spreads, spreads),
        ("Real Yields vs Breakevens", fig_real, pd.concat([tips_5y, tips_10y, be_5y2, be_10y2], axis=1)),
        ("Credit Spreads",    fig_credit, pd.concat([ig_oas, hy_oas, credit_diff_df], axis=1)),
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
        fig_fedwatch.update_layout(**base_layout("Market-Implied Fed Funds"))
        # Plotly bars default to a 0-anchored y-axis, which buries a tight cluster of implied
        # rates (e.g. 3.6-4.3%) in the top sliver of the chart. Bloomberg's WIRP view zooms to
        # the data instead - do the same: range from just below the lowest of (current EFFR,
        # implied rates) to just above the highest, not from zero.
        y_vals = list(fedwatch_df["Implied Rate"]) + [current_effr]
        y_min, y_max = min(y_vals), max(y_vals)
        y_pad = (y_max - y_min) * 0.15 or 0.25
        fig_fedwatch.update_yaxes(ticksuffix="%", title="Implied Rate",
                                  range=[y_min - y_pad, y_max + y_pad * 1.6])  # extra headroom for outside bar labels
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
        fig_fedwatch_hist.update_layout(**base_layout("Fed Funds Rate — Snapshots"))
        fig_fedwatch_hist.update_xaxes(title="Dates")
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
    outstanding_mat, outstanding_maturity_summary = get_outstanding_by_remaining_maturity(auctions)
    issuance_maturity_summary = get_upcoming_issuance_by_maturity_bucket(auctions, days_ahead)

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

    # Outstanding by remaining maturity (nearest-tenor ladder), with the yield curve overlaid
    # on a secondary axis so current supply can be read against where the curve sits at each
    # tenor. Reindexed to the full ladder (zero-filled) so the bars and the (interpolated)
    # yield-curve points always line up 1:1 on the shared x-axis, even if a bucket is empty.
    ladder_labels = list(MATURITY_LADDER.keys())
    outstanding_maturity_summary = (
        outstanding_maturity_summary.set_index("maturity_bucket")
        .reindex(ladder_labels, fill_value=0)
        .rename_axis("maturity_bucket").reset_index()
    )
    total_outstanding_maturity = outstanding_maturity_summary["Outstanding (Billion $)"].sum()

    fig_outstanding_maturity = go.Figure()
    fig_outstanding_maturity.add_trace(go.Bar(
        x=outstanding_maturity_summary["maturity_bucket"], y=outstanding_maturity_summary["Outstanding (Billion $)"],
        marker_color="#42a5f5", text=outstanding_maturity_summary["Outstanding (Billion $)"].round(0),
        textposition="outside", name="Outstanding (Billion $)", yaxis="y",
    ))
    if not yc.empty:
        target_years = list(MATURITY_LADDER.values())
        for label, offset in snap_labels.items():
            idx = max(0, len(yc) - 1 + offset)
            interp_yields = _interp_yield_curve(yc.iloc[idx], target_years)
            fig_outstanding_maturity.add_trace(go.Scatter(
                x=ladder_labels, y=interp_yields, mode="lines+markers", name=f"Yield: {label}",
                yaxis="y2", line=dict(color=snap_colors[label], width=2 if label == "Latest" else 1,
                                       dash="solid" if label == "Latest" else "dash"),
                marker=dict(size=4),
            ))
    fig_outstanding_maturity.update_layout(**dual_axis_layout(
        f"Outstanding Treasuries by Remaining Maturity (${total_outstanding_maturity:,.0f}B)",
        "Outstanding (Billion $)", "Yield (%)"))
    fig_outstanding_maturity.update_layout(yaxis2=dict(ticksuffix="%"))

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
    fig_btc.update_layout(**base_layout(f"Bid-to-Cover Ratio — {bc_type}s"))
    add_recessions(fig_btc, recessions)

    # Outstanding + new issuance - current outstanding balance plus whatever's being newly
    # issued in the selected forward window, both bucketed on the same remaining-maturity
    # ladder (a 17-Week bill lands in 3M, a 26-Week bill in 6M, etc. - see
    # get_upcoming_issuance_by_maturity_bucket). Does not net out maturities - that's what the
    # "Issuance vs Maturity" chart above already covers.
    outstanding_plus_new = pd.merge(
        outstanding_maturity_summary, issuance_maturity_summary, on="maturity_bucket", how="left"
    ).fillna(0)
    outstanding_plus_new = outstanding_plus_new.iloc[outstanding_plus_new["maturity_bucket"].map(_MATURITY_LADDER_ORDER).argsort()]
    outstanding_plus_new["Outstanding + New Issuance (Billion $)"] = (
        outstanding_plus_new["Outstanding (Billion $)"] + outstanding_plus_new["Total Issuance (Billion $)"])
    fig_outstanding_plus_new = go.Figure()
    fig_outstanding_plus_new.add_trace(go.Bar(x=outstanding_plus_new["maturity_bucket"], y=outstanding_plus_new["Outstanding (Billion $)"],
                                               name="Current Outstanding", marker_color="#ab47bc"))
    fig_outstanding_plus_new.add_trace(go.Bar(x=outstanding_plus_new["maturity_bucket"], y=outstanding_plus_new["Total Issuance (Billion $)"],
                                               name=f"New Issuance (Next {days_ahead}d)", marker_color="#26a69a"))
    fig_outstanding_plus_new.update_layout(**base_layout(
        f"Outstanding + New Issuance ({days_ahead}d, "
        f"${outstanding_plus_new['Outstanding + New Issuance (Billion $)'].sum():,.0f}B)"))
    fig_outstanding_plus_new.update_layout(barmode="stack")

    # Issuance by tenor over time - quarterly stacked area, selected date range. Answers "how
    # has the magnitude (and mix) of issuance across tenors changed over time" directly: total
    # stack height is total quarterly issuance, band thickness is that tenor's share.
    issuance_over_time = get_issuance_by_tenor_over_time(auctions, START, END)
    fig_issuance_time = go.Figure()
    TENOR_PALETTE = ["#42a5f5", "#29b6f6", "#26c6da", "#26a69a", "#66bb6a", "#9ccc65",
                      "#d4e157", "#ffee58", "#ffca28", "#ffa726", "#ff7043", "#ef5350",
                      "#ec407a", "#ab47bc", "#7e57c2", "#5c6bc0"]
    if not issuance_over_time.empty:
        for i, tenor in enumerate(issuance_over_time.columns):
            fig_issuance_time.add_trace(go.Scatter(
                x=issuance_over_time.index, y=issuance_over_time[tenor], name=tenor,
                mode="lines", stackgroup="one", line=dict(width=0.5, color=TENOR_PALETTE[i % len(TENOR_PALETTE)]),
            ))
    fig_issuance_time.update_layout(**base_layout("Issuance by Tenor Over Time (Quarterly, Billion $)", height=520))
    fig_issuance_time.update_yaxes(title="Issuance (Billion $)")
    add_recessions(fig_issuance_time, recessions)

    treasury_charts = [
        ("Upcoming Issuances", fig_upcoming, upcoming),
        ("Maturing Treasuries", fig_maturing, maturing),
        ("Issuance vs Maturity", fig_supply_bar, combined),
        ("Outstanding by Remaining Maturity", fig_outstanding_maturity, outstanding_maturity_summary),
        ("Bid-to-Cover Trend", fig_btc, bc_hist[["auction_date", "security_term_week_year", "bid_to_cover_ratio"]]),
        ("Outstanding + New Issuance", fig_outstanding_plus_new, outstanding_plus_new),
        ("Issuance by Tenor Over Time", fig_issuance_time, issuance_over_time.reset_index()),
    ]
    render_two_col(treasury_charts)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — US Markets
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("US Markets")
    st.caption("Equities and cross-asset - a different data domain from the rest of this dashboard "
               "(yfinance, not FRED, for most of this tab). Charts respect the global date range above; "
               "z-scores and period returns use a longer fixed lookback under the hood so the latest "
               "reading stays valid even if you narrow the date range.")

    MARKETS_HIST_START = "2015-01-01"  # independent of the global slider - z-scores need a
                                        # trailing window regardless of the selected display range

    with st.spinner("Loading market data…"):
        spy_full = fetch_yf_close("SPY", "SPY", MARKETS_HIST_START)
        tlt_full = fetch_yf_close("TLT", "TLT", MARKETS_HIST_START)
        vix_full = fetch_yf_close("^VIX", "VIX", MARKETS_HIST_START)
        dxy_full = fetch_yf_close("DX-Y.NYB", "DXY", MARKETS_HIST_START)
        gold_full = fetch_yf_close("GC=F", "Gold", MARKETS_HIST_START)
        real_yield_10y = fetch("DFII10", "10Y Real Yield", MARKETS_HIST_START, END)
        idx_full = {t: fetch_yf_close(t, name, MARKETS_HIST_START) for t, name in MARKET_INDICES.items()}
        sector_full = {t: fetch_yf_close(t, name, MARKETS_HIST_START) for t, name in SECTOR_ETFS.items()}

        # VIX term structure - ^VIX9D/^VIX3M only ever return their single current value via
        # yfinance (confirmed live: requesting 2 years of history still returns exactly 1 row),
        # so this can only ever be a snapshot, never a time series, with this data source.
        try:
            vix9d_now = float(_yf_retry(lambda: yf.Ticker("^VIX9D").history(period="5d"))["Close"].iloc[-1])
        except Exception:
            vix9d_now = None
        try:
            vix3m_now = float(_yf_retry(lambda: yf.Ticker("^VIX3M").history(period="5d"))["Close"].iloc[-1])
        except Exception:
            vix3m_now = None
        try:
            spy_pe = _yf_retry(lambda: yf.Ticker("SPY").info.get("trailingPE"))
        except Exception:
            spy_pe = None

    def _clip_mkt(df):
        return df[(df.index >= START) & (df.index <= END)] if not df.empty else df

    # Stock/Bond rolling correlation - your original idea. SPY vs TLT daily returns (price-
    # based, not yield-based, so positive = "stocks and bonds moving together", matching how
    # this is conventionally reported - a yield-diff version would read with the opposite sign
    # since bond prices and yields move inversely).
    fig_corr = go.Figure()
    if not spy_full.empty and not tlt_full.empty:
        combo = pd.concat([spy_full, tlt_full], axis=1).dropna()
        rets = combo.pct_change().dropna()
        corr_60 = rets["SPY"].rolling(21 * 3).corr(rets["TLT"])  # ~60 trading days
        corr_252 = rets["SPY"].rolling(252).corr(rets["TLT"])
        corr_df = _clip_mkt(pd.DataFrame({"60D": corr_60, "1Y": corr_252}).dropna())
        fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df["60D"], name="60D", line=dict(color="#42a5f5")))
        fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df["1Y"], name="1Y", line=dict(color="#ab47bc")))
    fig_corr.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_corr.update_layout(**base_layout("Stock/Bond Rolling Correlation (SPY vs TLT)"))
    add_recessions(fig_corr, recessions)

    # Equity risk premium - current snapshot only. yfinance doesn't expose a historical daily
    # P/E series for SPY/S&P 500, only the current value, so this can't be a time series.
    # y10_nominal goes through the shared fetch() helper (retry + cache) rather than a raw,
    # unretried fred.get_series call, so a transient FRED hiccup here behaves the same way as
    # everywhere else in the file instead of risking an unhandled exception.
    st.markdown('<div class="section-header">Equity Risk Premium</div>', unsafe_allow_html=True)
    y10_nominal_df = fetch("DGS10", "10Y Nominal Yield", MARKETS_HIST_START, END)
    missing = []
    if not spy_pe:
        missing.append("SPY trailing P/E")
    if y10_nominal_df.empty:
        missing.append("10Y Treasury yield")

    if not missing:
        earnings_yield = 100 / spy_pe
        y10_nominal = float(y10_nominal_df["10Y Nominal Yield"].dropna().iloc[-1])
        erp = earnings_yield - y10_nominal
        erp_col1, erp_col2, erp_col3 = st.columns(3)
        with erp_col1:
            st.metric("SPY Earnings Yield", f"{earnings_yield:.2f}%", help=f"1 / trailing P/E ({spy_pe:.1f})")
        with erp_col2:
            st.metric("10Y Treasury Yield", f"{y10_nominal:.2f}%")
        with erp_col3:
            st.metric("Equity Risk Premium", f"{erp:+.2f}%",
                     help="Earnings yield minus 10Y yield. Negative means bonds currently yield more than stock earnings.")
    else:
        st.info(f"Equity risk premium unavailable this run - failed to load: {', '.join(missing)}.")

    st.plotly_chart(fig_corr, use_container_width=True)
    csv_download(corr_df if not spy_full.empty and not tlt_full.empty else pd.DataFrame(), "stock_bond_correlation")

    # Index levels, 1M/3M/1Y returns, and z-scores - returns use the same fixed trailing
    # windows as the z-scores (not the global date-range slider), so the two charts stay
    # directly comparable row-for-row and both are labeled unambiguously by period.
    st.markdown('<div class="section-header">Index Levels & Z-Scores</div>', unsafe_allow_html=True)
    idx_rows, idx_z_rows = [], []
    for t, name in MARKET_INDICES.items():
        full = idx_full[t]
        if full.empty:
            continue
        ret = calculate_period_returns(full[name])
        idx_rows.append({"Index": name, "Level": full[name].iloc[-1], **{f"{k} Return %": v for k, v in ret.items()}})
        z = calculate_price_zscores(full[name])
        idx_z_rows.append({"Index": name, **{f"{k} Z-Score": v for k, v in z.items()}})
    idx_df = pd.DataFrame(idx_rows).sort_values("1M Return %")
    idx_z_df = pd.DataFrame(idx_z_rows).sort_values("1M Z-Score")

    fig_idx_ret = go.Figure()
    for label, color in [("1M Return %", "#42a5f5"), ("3M Return %", "#ab47bc"), ("1Y Return %", "#ff9800")]:
        fig_idx_ret.add_trace(go.Bar(x=idx_df[label], y=idx_df["Index"], name=label,
                                     orientation="h", marker_color=color))
    fig_idx_ret.add_vline(x=0, line_dash="dot", line_color="#555")
    fig_idx_ret.update_layout(**base_layout("Index Returns — 1M / 3M / 1Y (%)", height=300))
    fig_idx_ret.update_layout(barmode="group")
    fig_idx_ret.update_xaxes(ticksuffix="%")

    fig_idx_z = go.Figure()
    for label, color in [("1M Z-Score", "#42a5f5"), ("3M Z-Score", "#ab47bc"), ("1Y Z-Score", "#ff9800")]:
        fig_idx_z.add_trace(go.Bar(x=idx_z_df[label], y=idx_z_df["Index"], name=label,
                                   orientation="h", marker_color=color))
    fig_idx_z.add_vline(x=0, line_dash="dot", line_color="#555")
    fig_idx_z.update_layout(**base_layout("Index Price Z-Scores (vs Own Trailing History)", height=300))
    fig_idx_z.update_layout(barmode="group")

    render_two_col([
        ("Index Returns", fig_idx_ret, idx_df),
        ("Index Z-Scores", fig_idx_z, idx_z_df),
    ])

    # Sector rotation + z-scores - same 1M/3M/1Y treatment as the index chart above.
    st.markdown('<div class="section-header">Sector Rotation & Z-Scores</div>', unsafe_allow_html=True)
    sec_rows, sec_z_rows = [], []
    for t, name in SECTOR_ETFS.items():
        full = sector_full[t]
        if full.empty:
            continue
        ret = calculate_period_returns(full[name])
        sec_rows.append({"Sector": name, **{f"{k} Return %": v for k, v in ret.items()}})
        z = calculate_price_zscores(full[name])
        sec_z_rows.append({"Sector": name, **{f"{k} Z-Score": v for k, v in z.items()}})
    if not spy_full.empty:
        spy_ret = calculate_period_returns(spy_full["SPY"])
        sec_rows.append({"Sector": "S&P 500 (SPY)", **{f"{k} Return %": v for k, v in spy_ret.items()}})
        spy_z = calculate_price_zscores(spy_full["SPY"])
        sec_z_rows.append({"Sector": "S&P 500 (SPY)", **{f"{k} Z-Score": v for k, v in spy_z.items()}})
    sec_df = pd.DataFrame(sec_rows).sort_values("1M Return %")
    sec_z_df = pd.DataFrame(sec_z_rows).sort_values("1M Z-Score") if sec_z_rows else pd.DataFrame()

    fig_sec_ret = go.Figure()
    for label, color in [("1M Return %", "#42a5f5"), ("3M Return %", "#ab47bc"), ("1Y Return %", "#ff9800")]:
        fig_sec_ret.add_trace(go.Bar(x=sec_df[label], y=sec_df["Sector"], name=label,
                                     orientation="h", marker_color=color))
    fig_sec_ret.add_vline(x=0, line_dash="dot", line_color="#555")
    fig_sec_ret.update_layout(**base_layout("Sector Returns — 1M / 3M / 1Y (%)", height=420))
    fig_sec_ret.update_layout(barmode="group")
    fig_sec_ret.update_xaxes(ticksuffix="%")

    fig_sec_z = go.Figure()
    for label, color in [("1M Z-Score", "#42a5f5"), ("3M Z-Score", "#ab47bc"), ("1Y Z-Score", "#ff9800")]:
        fig_sec_z.add_trace(go.Bar(x=sec_z_df[label], y=sec_z_df["Sector"], name=label,
                                   orientation="h", marker_color=color))
    fig_sec_z.add_vline(x=0, line_dash="dot", line_color="#555")
    fig_sec_z.update_layout(**base_layout("Sector Price Z-Scores (vs Own Trailing History)", height=420))
    fig_sec_z.update_layout(barmode="group")

    render_two_col([
        ("Sector Returns", fig_sec_ret, sec_df),
        ("Sector Z-Scores", fig_sec_z, sec_z_df),
    ])

    # VIX level
    fig_vix = go.Figure()
    vix_clipped = _clip_mkt(vix_full)
    if not vix_clipped.empty:
        fig_vix.add_trace(go.Scatter(x=vix_clipped.index, y=vix_clipped["VIX"], name="VIX",
                                     line=dict(color="#ef5350"), fill="tozeroy", fillcolor="rgba(239,83,80,0.12)"))
    fig_vix.update_layout(**base_layout("VIX (Implied Volatility)"))
    add_recessions(fig_vix, recessions)

    # VIX term structure - snapshot only (see note above on why no history exists for these two)
    fig_vix_term = go.Figure()
    vix_now = vix_full["VIX"].iloc[-1] if not vix_full.empty else None
    term_labels, term_vals = [], []
    for lbl, val in [("9D", vix9d_now), ("30D (VIX)", vix_now), ("3M", vix3m_now)]:
        if val is not None:
            term_labels.append(lbl); term_vals.append(val)
    if term_labels:
        fig_vix_term.add_trace(go.Bar(x=term_labels, y=term_vals, marker_color="#ef5350",
                                      hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    fig_vix_term.update_layout(**base_layout("VIX Term Structure (Snapshot Only)", height=420))

    render_two_col([
        ("VIX", fig_vix, vix_clipped),
        ("VIX Term Structure", fig_vix_term, pd.DataFrame({"Tenor": term_labels, "VIX": term_vals})),
    ])
    st.caption("VIX Term Structure is a current snapshot, not a time series - confirmed live that "
               "yfinance only ever returns the single latest value for ^VIX9D and ^VIX3M, regardless "
               "of how much history is requested.")

    # Dollar Index
    fig_dxy = go.Figure()
    dxy_clipped = _clip_mkt(dxy_full)
    if not dxy_clipped.empty:
        fig_dxy.add_trace(go.Scatter(x=dxy_clipped.index, y=dxy_clipped["DXY"], name="DXY",
                                     line=dict(color="#90a4d4")))
    fig_dxy.update_layout(**base_layout("US Dollar Index (DXY)"))
    add_recessions(fig_dxy, recessions)

    # Gold vs 10Y real yield - two panels, not one dual-axis chart: different units.
    fig_gold = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Gold ($/oz)", "10Y Real Yield (%)"),
                             vertical_spacing=0.12)
    gold_clipped = _clip_mkt(gold_full)
    ry_clipped = real_yield_10y[(real_yield_10y.index >= START) & (real_yield_10y.index <= END)] if not real_yield_10y.empty else real_yield_10y
    if not gold_clipped.empty:
        fig_gold.add_trace(go.Scatter(x=gold_clipped.index, y=gold_clipped["Gold"],
                                      name="Gold", line=dict(color="#eda100")), row=1, col=1)
    if not ry_clipped.empty:
        fig_gold.add_trace(go.Scatter(x=ry_clipped.index, y=ry_clipped["10Y Real Yield"],
                                      name="10Y Real Yield", line=dict(color="#ef5350")), row=2, col=1)
    fig_gold.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                           height=500, margin=dict(l=10, r=10, t=45, b=30), showlegend=False)
    fig_gold.update_xaxes(gridcolor=GRID_COLOR)
    fig_gold.update_yaxes(gridcolor=GRID_COLOR)
    add_recessions(fig_gold, recessions, rows=[1, 2], cols=[1, 1])

    render_two_col([
        ("Dollar Index (DXY)", fig_dxy, dxy_clipped),
        ("Gold vs 10Y Real Yield", fig_gold, pd.concat([gold_clipped, ry_clipped], axis=1)),
    ])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — Fiscal Policy & Government Spending
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Fiscal Policy & Government Spending")
    st.caption("US Treasury Fiscal Data API — Daily Treasury Statement, Debt Subject to Limit, "
               "Monthly Treasury Statement.")

    with st.spinner("Loading fiscal data…"):
        tga             = get_tga_balance(START, END)
        rrp_vol         = fetch("RRPONTSYD", "ON RRP (Billion $)", START, END)
        debt_limit      = get_debt_subject_to_limit(START, END)
        mts_monthly     = get_mts_monthly()

    # TGA & RRP — two separate single-axis charts rather than one dual-axis chart: same
    # "where are reserves going" story, but each series keeps its own honest scale.
    fig_tga = go.Figure()
    if not tga.empty:
        fig_tga.add_trace(go.Scatter(x=tga.index, y=tga["TGA Balance (Billion $)"], name="TGA Balance",
                                      line=dict(color="#42a5f5"), fill="tozeroy", fillcolor="rgba(66,165,245,0.12)"))
    fig_tga.update_layout(**base_layout("Treasury General Account Balance"))
    fig_tga.update_yaxes(ticksuffix="B")
    add_recessions(fig_tga, recessions)

    fig_rrp = go.Figure()
    if not rrp_vol.empty:
        fig_rrp.add_trace(go.Scatter(x=rrp_vol.index, y=rrp_vol["ON RRP (Billion $)"], name="ON RRP",
                                      line=dict(color="#ab47bc"), fill="tozeroy", fillcolor="rgba(171,71,188,0.12)"))
    fig_rrp.update_layout(**base_layout("ON RRP Usage"))
    fig_rrp.update_yaxes(ticksuffix="B")
    add_recessions(fig_rrp, recessions)

    # Debt ceiling
    fig_debt = go.Figure()
    if not debt_limit.empty:
        fig_debt.add_trace(go.Scatter(x=debt_limit.index, y=debt_limit["Debt Subject to Limit ($T)"],
                                       name="Debt Subject to Limit", line=dict(color="#42a5f5")))
        fig_debt.add_trace(go.Scatter(x=debt_limit.index, y=debt_limit["Statutory Limit ($T)"],
                                       name="Statutory Limit", line=dict(color="#eda100", dash="dash")))
    fig_debt.update_layout(**base_layout("Debt Subject to Limit vs. Statutory Limit"))
    fig_debt.update_yaxes(ticksuffix="T")
    add_recessions(fig_debt, recessions)

    # Monthly budget — receipts up, outlays down, net as the line. Treasury's own
    # current_month_dfct_sur_amt is signed positive=deficit/negative=surplus (the opposite of
    # receipts-minus-outlays) - confirmed live (e.g. a $27B surplus month reports as -27B). Negating
    # it here makes the line read the same direction as the bars: positive when the green (receipts)
    # bar is taller than the red (outlays) bar, matching receipts - outlays exactly.
    fig_mts = go.Figure()
    if not mts_monthly.empty:
        fig_mts.add_trace(go.Bar(x=mts_monthly["Period"], y=mts_monthly["current_month_gross_rcpt_amt"],
                                  name="Receipts", marker_color="#26a69a"))
        fig_mts.add_trace(go.Bar(x=mts_monthly["Period"], y=-mts_monthly["current_month_gross_outly_amt"],
                                  name="Outlays", marker_color="#ef5350"))
        fig_mts.add_trace(go.Scatter(x=mts_monthly["Period"], y=-mts_monthly["current_month_dfct_sur_amt"],
                                      name="Net (Receipts − Outlays)", line=dict(color="#e0e0e0", width=2)))
    fig_mts.update_layout(**base_layout("Monthly Receipts, Outlays & Net (Billion $)"))
    fig_mts.update_layout(barmode="relative")

    fiscal_charts = [
        ("TGA Balance", fig_tga, tga),
        ("ON RRP Usage", fig_rrp, rrp_vol),
        ("Debt Subject to Limit", fig_debt, debt_limit),
        ("Monthly Budget", fig_mts, mts_monthly),
    ]
    render_two_col(fiscal_charts)
    st.caption("Debt-ceiling chart drops rows during a suspension period — Treasury reports the statutory "
               "limit as $0 then, which means \"not currently in effect,\" not \"zero headroom\"; charting it "
               "as-is would show a fake breach.")

    # ── Spending by category (grouped, dropdown-selectable) ─────────────────────
    st.markdown('<div class="section-header">Spending by Category</div>', unsafe_allow_html=True)
    st.caption("Each option sums several DTS transaction categories that Treasury reports separately "
               "(e.g. defense pay, health and retirement are three different rows) into one program total.")

    cat_choice = st.selectbox("Spending category", list(FISCAL_SPEND_GROUPS.keys()), key="fiscal_cat_choice")
    with st.spinner(f"Loading {cat_choice} spending…"):
        cat_data = get_category_spend(tuple(FISCAL_SPEND_GROUPS[cat_choice]), "2022-01-01")

    n_cats = len(FISCAL_SPEND_GROUPS[cat_choice])
    st.caption(f"{'From' if n_cats == 1 else f'Summed from {n_cats}'} DTS categor{'y' if n_cats == 1 else 'ies'}: "
               + ", ".join(FISCAL_SPEND_GROUPS[cat_choice]))
    if cat_choice == "Defense":
        st.caption("⚠️ Treasury only reports DoD pay/health/retirement as separate line items from FY2024 "
                   "(Oct 2023) onward — before that, defense spending is bundled into a single vendor-payments "
                   "category not comparable to this grouping, so this series starts Oct 2023.")
    elif cat_choice == "Medicare":
        st.caption("⚠️ One constituent line (HHS - Other Centers for Medicare & Medicaid Services) currently "
                   "returns no rows, so this slightly understates total Medicare-related outlays.")
    elif cat_choice == "Interest on Debt":
        st.caption("⚠️ This is Treasury's cash-basis DTS category — interest actually paid out day by day — "
                   "rather than the accrual-based interest expense Treasury reports elsewhere, so it won't "
                   "match an accrual figure exactly month to month.")

    fig_cat_fytd = go.Figure()
    fig_cat_yoy = go.Figure()
    if not cat_data.empty:
        fig_cat_fytd.add_trace(go.Scatter(x=cat_data.index, y=cat_data["FYTD (Billion $)"], name=cat_choice,
                                           line=dict(color="#26a69a"), fill="tozeroy", fillcolor="rgba(38,166,154,0.12)"))
        yoy_s = cat_data["YoY %"].dropna()
        fig_cat_yoy.add_trace(go.Scatter(x=yoy_s.index, y=yoy_s, name=f"{cat_choice} YoY", line=dict(color="#ff9800")))
        fig_cat_yoy.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_cat_fytd.update_layout(**base_layout(f"{cat_choice} — FYTD Cumulative Spend (Billion $)"))
    fig_cat_yoy.update_layout(**base_layout(f"{cat_choice} — YoY Change on FYTD (%)"))
    fig_cat_yoy.update_yaxes(ticksuffix="%")

    render_two_col([
        (f"{cat_choice} FYTD", fig_cat_fytd, cat_data),
        (f"{cat_choice} YoY", fig_cat_yoy, cat_data),
    ])
    st.caption("FYTD resets every October, so the October YoY reading compares one month of new-fiscal-year "
               "spend against a much larger September base and can look like a large swing — that's the "
               "reset, not a real spending spike.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — Leading Indicators
# ════════════════════════════════════════════════════════════════════════════════
with tabs[7]:
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
        lei_yoy = (lei["Conference Board LEI"].pct_change(12) * 100).round(3)
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 9 — Economic Calendar
# ════════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("Economic Calendar")
    st.caption("Next scheduled release dates (FRED) + upcoming FOMC meetings + upcoming Treasury auctions. "
               "No consensus/forecast column - that's commercial data with no free legitimate source; "
               "\"Detail\" shows the previous actual print instead.")

    days_ahead = st.select_slider("Forward-looking window (days)", options=[7, 14, 30, 60], value=30, key="econ_cal_days")
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)

    with st.spinner("Loading economic release calendar…"):
        econ_df = get_economic_release_calendar()
    econ_df = econ_df[(econ_df["Date"] >= today) & (econ_df["Date"] <= cutoff)]

    with st.spinner("Loading FOMC meeting dates…"):
        fomc_dates = [pd.Timestamp(d) for d in get_fomc_dates()]
    fomc_df = pd.DataFrame([
        {"Date": d, "Event": "FOMC Meeting", "Type": "FOMC", "Detail": ""}
        for d in fomc_dates if today <= d <= cutoff
    ])

    with st.spinner("Loading Treasury auction calendar…"):
        auctions_all = load_auctions_data()
    # Filter on auction_date (the actual bidding day - the real "event") rather than
    # issue_date (settlement, a few days later): get_upcoming_issuances filters on
    # issue_date, which can leave auction_date in the past for a "today onward" calendar.
    upcoming_auctions = auctions_all[(auctions_all["auction_date"] >= today) & (auctions_all["auction_date"] <= cutoff)].copy()
    upcoming_auctions["offering_amt_bil"] = upcoming_auctions["offering_amt"] / 1e9
    auction_df = pd.DataFrame([
        {"Date": row["auction_date"], "Event": f"{row['security_term_week_year']} {row['security_type']} Auction",
         "Type": "Treasury Auction", "Detail": f"${row['offering_amt_bil']:.1f}B"}
        for _, row in upcoming_auctions.iterrows()
    ])

    calendar_df = pd.concat([econ_df, fomc_df, auction_df], ignore_index=True)
    if not calendar_df.empty:
        calendar_df = calendar_df.sort_values("Date")
        type_counts = calendar_df["Type"].value_counts()
        st.markdown(f"**{len(calendar_df)} events in the next {days_ahead} days** — "
                    + " | ".join(f"{t}: {c}" for t, c in type_counts.items()))

        display_df = calendar_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d (%a)")
        st.dataframe(display_df[["Date", "Event", "Type", "Detail"]], use_container_width=True, hide_index=True, height=600)
        csv_download(display_df, "economic_calendar")
    else:
        st.info(f"No tracked releases, FOMC meetings, or Treasury auctions in the next {days_ahead} days.")

st.markdown("---")
st.caption("Data: FRED (Federal Reserve Bank of St. Louis) · Shaded areas = NBER recessions · Refresh rate: 1hr cache")
