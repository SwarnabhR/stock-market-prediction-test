
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import json

# ---- Utility functions ----
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.to_flat_index()
    df.columns = [
        "_".join(map(str, c)) if isinstance(c, tuple) else str(c)
        for c in df.columns
    ]
    return df

def attach_time_column(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ["Date", "Datetime", "date", "datetime", "timestamp"]:
        if cand in df.columns:
            df["time"] = pd.to_datetime(df[cand]).dt.strftime("%Y-%m-%d")
            return df
    if df.index.name and "date" in df.index.name.lower():
        df = df.reset_index()
        return attach_time_column(df)
    df["time"] = pd.to_datetime(df.iloc[:, 0]).dt.strftime("%Y-%m-%d")
    return df

def map_ohlcv(df):
    """
    Return dict  {'Open':real_col, 'High':real_col, … , 'Volume':real_col}
    that matches whatever yfinance gave us (with or without suffixes).
    """
    out = {}
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        out[base] = next(
            (c for c in df.columns if base.lower() in c.lower()), None
        )
    out["Time"] = "time"
    return out
# ────────────────────────────────────────────────────────────────────────


# ---- Sidebar (controls) ----
st.sidebar.title("🔍 Symbol & Settings")
symbol = st.sidebar.text_input("Ticker (Yahoo code)", "RELIANCE.NS")
period = st.sidebar.selectbox("History window", ["6mo", "1y", "2y"], index=1)
theme = st.sidebar.selectbox("Theme", ["dark", "light"], index=0)

# ---- Data load ----
@st.cache_data(show_spinner=False)
def load_data(sym, per):
    df = yf.download(sym, period=per, interval="1d", auto_adjust=False)
    df = flatten_columns(df).reset_index(drop=False)
    df = attach_time_column(df)
    df.ta.rsi(length=14, append=True)
    return df

df = load_data(symbol, period)

# ---------- helpers to find column names ----------
def map_ohlcv_columns(df):
    """
    Return a dict that maps canonical keys Open/High/Low/Close/Volume
    to the actual column names present in df, regardless of suffixes
    like 'Open_REL' or 'Close_RELIANCE.NS'.
    """
    out = {}
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        out[base] = next(
            (c for c in df.columns if base.lower() in c.lower()), None
        )
    out["Time"] = "time"
    return out


def cols(df, *keys):
    "Convenience wrapper to get the real column names from the map."
    m = map_ohlcv_columns(df)
    return [m[k] for k in keys]


# ---- Prepare for chart ----
COLOR_UP, COLOR_DOWN = "rgba(38,166,154,0.9)", "rgba(239,83,80,0.9)"

cols = map_ohlcv(df)                         # <── new

candles = json.loads(
    df[[cols["Time"], cols["Open"], cols["High"],
        cols["Low"], cols["Close"]]].to_json(orient="records")
)

volume  = json.loads(
    df.assign(value=df[cols["Volume"]])
      [[cols["Time"], "value"]]
      .to_json(orient="records")
)

rsi     = json.loads(df.assign(value=df.RSI_14)[["time", "value"]].to_json(orient="records"))

chart_cfg = [
    {"height": 400,
     "layout": {"background": {"type": "solid", "color": "#1e1e1e" if theme == "dark" else "white"},
                "textColor": "#d1d4dc" if theme == "dark" else "black"}},
    {"height": 100, "timeScale": {"visible": False}},
    {"height": 120, "timeScale": {"visible": False}}
]

series_cfg = [
    [{"type": "Candlestick", "data": candles,
      "options": {"upColor": COLOR_UP, "downColor": COLOR_DOWN,
                  "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN}}],
    [{"type": "Histogram", "data": volume,
      "options": {"color": "#26a69a", "priceFormat": {"type": "volume"}}}],
    [{"type": "Line", "data": rsi, "options": {"color": "orange", "lineWidth": 2},
      "markers": [{"time": rsi[-1]["time"], "position": "aboveBar",
                  "text": f"RSI {rsi[-1]['value']:.1f}"}]}]
]

st.title(f"📈 {symbol} – TradingView-style dashboard")
renderLightweightCharts(
    [{"chart": c, "series": s} for c, s in zip(chart_cfg, series_cfg)],
    key="mainchart"
)
