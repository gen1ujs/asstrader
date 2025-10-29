# app.py
# TAO/USDT – 1H Excel Viewer (Plotly Candlestick + MA)
# - Excel (.xlsx/.xls) dosyasından saatlik veriyi okur
# - Son 3 ayı filtreleme seçeneğiyle grafikte ve tabloda gösterir
# - Görüntülenen veriyi Excel olarak indirme imkânı sunar

import os
import io
import glob
from datetime import timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -------------------- Page setup --------------------
st.set_page_config(page_title="TAO/USDT – 1H (Excel Viewer)", layout="wide")
st.title("TAO/USDT – 1H Veri Görselleştirici (Excel)")

st.caption(
    "Excel dosyasından saatlik TAO/USDT verisini okur, son 3 ayı isteğe bağlı filtreler, "
    "candlestick + hareketli ortalama ile görselleştirir ve görüntülenen veriyi Excel olarak indirmenizi sağlar."
)

# -------------------- Sidebar: Data source --------------------
st.sidebar.header("Veri Kaynağı (Excel)")
default_folder = st.sidebar.text_input("Veri klasörü", value="data")
pattern = st.sidebar.text_input("Dosya paterni", value="TAOUSDT.xlsx")
uploaded = st.sidebar.file_uploader("Excel yükle (opsiyonel)", type=["xlsx", "xls"])

# -------------------- Loader --------------------
def load_df_from_excel(_folder: str, _pattern: str, _uploaded):
    if _uploaded is not None:
        df = pd.read_excel(_uploaded)
        st.sidebar.success("Yüklenen Excel kullanılıyor.")
    else:
        paths = sorted(glob.glob(os.path.join(_folder, _pattern)))
        if not paths:
            st.sidebar.error("Excel bulunamadı. Klasör/pattern'i kontrol edin veya bir dosya yükleyin.")
            st.stop()
        latest = paths[-1]
        st.sidebar.success(f"Seçilen dosya: {os.path.basename(latest)}")
        df = pd.read_excel(latest)

    # Kolon adlarını normalize et
    cols = {c.lower(): c for c in df.columns}
    mapping = {}

    # timestamp/time/open time/date -> timestamp
    for k in ["timestamp", "time", "open time", "date"]:
        if k in cols:
            mapping[cols[k]] = "timestamp"
            break

    # OHLCV
    for k in ["open", "high", "low", "close", "volume"]:
        if k in cols:
            mapping[cols[k]] = k

    df = df.rename(columns=mapping)

    # Tip dönüşümleri
    # Excel genellikle tz'siz gelir; olası metin formatlarını da kapsayalım
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)

    # timestamp tz'li geldiyse tz'i kaldır (Excel uyumluluğu)
    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    for k in ["open", "high", "low", "close", "volume"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    return df

df_full = load_df_from_excel(default_folder, pattern, uploaded)

# -------------------- Sidebar: View options --------------------
st.sidebar.header("Görünüm")
use_last_3m = st.sidebar.checkbox("Sadece son 40 günü göster", value=True)
ma_window = st.sidebar.number_input("MA Penceresi (saat)", min_value=10, max_value=1000, value=100, step=5)
show_signals = st.sidebar.checkbox("Kırılım oklarını göster", value=True)


# -------------------- Build df_view --------------------
df_view = df_full.copy()
if use_last_3m:
    cutoff = df_full["timestamp"].max() - timedelta(days=40)
    df_view = df_full[df_full["timestamp"] >= cutoff].reset_index(drop=True)

# MA hesapla (görünür veri üzerinde)
df_view["MA"] = df_view["close"].rolling(int(ma_window), min_periods=int(ma_window)).mean()
prev_close = df_view["close"].shift(1)
prev_ma = df_view["MA"].shift(1)

long_cross = (
    (prev_close < prev_ma)
    & (df_view["close"] > df_view["MA"])
    & df_view["MA"].notna()
)
short_cross = (
    (prev_close > prev_ma)
    & (df_view["close"] < df_view["MA"])
    & df_view["MA"].notna()
)

df_view["signal"] = np.select(
    [long_cross, short_cross],
    ["LONG", "SHORT"],
    default=""
)
df_view["period"] = df_view["signal"].ne("").cumsum()

# Sinyal noktalarının x-y koordinatları
long_x  = df_view.loc[long_cross,  "timestamp"]
long_y  = df_view.loc[long_cross,  "close"]
short_x = df_view.loc[short_cross, "timestamp"]
short_y = df_view.loc[short_cross, "close"]

# -------------------- Metrics --------------------
if len(df_view) == 0:
    st.warning("Seçili filtreyle görüntülenecek satır yok. Son 3 ay filtresini kapatmayı deneyin.")
    st.stop()

rows = len(df_view)
span_hours = (df_view["timestamp"].iloc[-1] - df_view["timestamp"].iloc[0]).total_seconds() / 3600
c1, c2, c3 = st.columns(3)
c1.metric("Satır", f"{rows:,}")
c2.metric("Zaman Aralığı (saat)", f"{span_hours:,.0f}")
c3.metric("Dönem", f"{df_view['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {df_view['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

# -------------------- Chart --------------------
st.subheader("Candlestick + MA")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_view["timestamp"],
    open=df_view["open"], high=df_view["high"],
    low=df_view["low"], close=df_view["close"],
    name="TAOUSDT (1H)"
))

fig.add_trace(go.Scatter(
    x=df_view["timestamp"], y=df_view["MA"],
    mode="lines", name=f"MA{ma_window}"
))

fig.update_yaxes(title="Price")

fig.update_layout(
    xaxis_rangeslider_visible=True,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

if show_signals:
    # Long sinyaller: triangle-up
    fig.add_trace(go.Scatter(
        x=long_x, y=long_y,
        mode="markers",
        name="Long Breakout",
        marker=dict(symbol="triangle-up", size=10, line=dict(width=1), color="green"),
        hovertemplate="Long ↗<br>%{x}<br>Close: %{y}<extra></extra>"
    ))
    # Short sinyaller: triangle-down
    fig.add_trace(go.Scatter(
        x=short_x, y=short_y,
        mode="markers",
        name="Short Breakdown",
        marker=dict(symbol="triangle-down", size=10, line=dict(width=1), color="red"),
        hovertemplate="Short ↘<br>%{x}<br>Close: %{y}<extra></extra>"
    ))

st.plotly_chart(fig, use_container_width=True)

# -------------------- Data Table --------------------
st.subheader("Veri Tablosu")
show_cols = ["timestamp", "open", "high", "low", "close", "volume", "MA", "signal", "period"]
show_cols = [c for c in show_cols if c in df_view.columns]
st.dataframe(df_view[show_cols], use_container_width=True, height=420)

# -------------------- Download (Excel of the viewed data) --------------------
st.subheader("Görüntülenen Veriyi İndir (Excel)")
out_buf = io.BytesIO()
with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
    df_view[show_cols].to_excel(writer, index=False, sheet_name="TAO_1H_VIEW")
st.download_button(
    label="İndir (.xlsx)",
    data=out_buf.getvalue(),
    file_name="TAOUSDT_1h_view.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Not: Bu sayfa yalnızca görselleştirme içindir. Backtest aşamasında tam veri seti (df_full) kullanılacaktır.")
