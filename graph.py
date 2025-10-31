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
import streamlit as st
import plotly.graph_objects as go

# -------------------- Page setup --------------------
st.set_page_config(page_title="TAO/USDT – 1H (Excel Viewer)", layout="wide")
st.title("TAO/USDT – 1H Veri Görselleştirici (Excel)")

st.caption(
    "Excel dosyasından saatlik TAO/USDT verisini okur, son 3 ayı isteğe bağlı filtreler, "
    "candlestick + hareketli ortalama ile görselleştirir ve görüntülenen veriyi Excel olarak indirmenizi sağlar."
)

# Full-width ve full-height grafik için basit CSS
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    div[data-testid=stPlotlyChart] { height: auto !important; }
    div[data-testid=stPlotlyChart] .plot-container,
    div[data-testid=stPlotlyChart] .main-svg,
    div[data-testid=stPlotlyChart] .js-plotly-plot { height: auto !important; width: auto !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Sidebar: Data source --------------------
st.sidebar.header("Veri Kaynağı (Excel)")
default_folder = st.sidebar.text_input("Veri klasörü", value="data")
pattern = st.sidebar.text_input("Dosya paterni", value="TAOUSDT.xlsx")
# Varsayılan klasör değeri 'data' ise boş olarak kullan (aynı dizin)
if default_folder == "data":
    default_folder = ""
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
signal_threshold_pct = st.sidebar.slider(
    "Kırılım doğrulama eşiği (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.5,
    help="Fiyatın hareketli ortalamanın kaç %% üzerine/altına çıktığında kırılımın geçerli sayılacağını belirler."
)
show_signals = st.sidebar.checkbox("Kırılım oklarını göster", value=True)


# -------------------- Sidebar: Strategy params --------------------
st.sidebar.header("Strateji")
stop_threshold_pct = st.sidebar.number_input(
    "Stop (MA'ya gore %)", min_value=0.0, max_value=50.0, value=1.0, step=0.1,
    help="Pozisyona girildiginde stop fiyati MA'ye gore % uzaklikla sabitlenir (LONG: MA-% | SHORT: MA+%)."
)
tp_long_pct = st.sidebar.number_input(
    "TP Long (%)", min_value=0.0, max_value=200.0, value=5.0, step=0.1,
    help="Long icin giris fiyata gore % kar al."
)
tp_short_pct = st.sidebar.number_input(
    "TP Short (%)", min_value=0.0, max_value=200.0, value=5.0, step=0.1,
    help="Short icin giris fiyata gore % kar al."
)
position_size = st.sidebar.number_input(
    "Islem Buyuklugu (USDT)", min_value=0.0, value=1000.0, step=100.0,
    help="Her islem icin kullanilacak notyonel tutar (USDT)."
)

# -------------------- Build df_view --------------------
df_view = df_full.copy()
if use_last_3m:
    cutoff = df_full["timestamp"].max() - timedelta(days=40)
    df_view = df_full[df_full["timestamp"] >= cutoff].reset_index(drop=True)

# MA hesapla (görünür veri üzerinde)
df_view["MA"] = df_view["close"].rolling(int(ma_window), min_periods=int(ma_window)).mean()
threshold_ratio = signal_threshold_pct / 100.0

signals = []
period_ids = []
period_types = []
current_period = 0
pending_long = False
pending_short = False
current_regime = None

for _, row in df_view.iterrows():
    ma_val = row["MA"]
    close_val = row["close"]

    if pd.isna(ma_val) or ma_val == 0:
        signals.append("")
        period_ids.append(current_period)
        period_types.append(current_regime)
        continue

    rel_diff = (close_val - ma_val) / ma_val

    if rel_diff <= -threshold_ratio:
        pending_long = True
    if rel_diff >= threshold_ratio:
        pending_short = True

    signal = ""
    if pending_long and rel_diff > threshold_ratio:
        signal = "LONG"
        pending_long = False
        pending_short = True
        current_period += 1
        current_regime = "long"
    elif pending_short and rel_diff < -threshold_ratio:
        signal = "SHORT"
        pending_short = False
        pending_long = True
        current_period += 1
        current_regime = "short"

    signals.append(signal)
    period_ids.append(current_period)
    period_types.append(current_regime)

df_view["signal"] = pd.Series(signals, index=df_view.index)
df_view["period"] = pd.Series(period_ids, index=df_view.index, dtype="int64")
df_view["period_type"] = pd.Series(period_types, index=df_view.index, dtype="object")

long_cross = df_view["signal"] == "LONG"
short_cross = df_view["signal"] == "SHORT"

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
last_period_value = int(df_view["period"].iloc[-1])
num_periods = last_period_value + 1
validated_breakouts = int((df_view["signal"] != "").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Satır", f"{rows:,}")
c2.metric("Zaman Aralığı (saat)", f"{span_hours:,.0f}")
c3.metric("Dönem", f"{df_view['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {df_view['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
c4.metric(
    "Periyot Sayısı",
    f"{num_periods}",
    help=f"Doğrulanan kırılım sayısı: {validated_breakouts}"
)

# -------------------- Chart --------------------
st.subheader("Candlestick + MA")

fig = go.Figure()

bar_duration = df_view["timestamp"].diff().median()
if pd.isna(bar_duration) or bar_duration == pd.Timedelta(0):
    bar_duration = timedelta(hours=1)

# Backtest helpers: collect per-period extreme moves from signal
long_peak_gains = []  # percent increase from LONG signal close to period max high
short_trough_drops = []  # percent decrease from SHORT signal close to period min low

for _, period_df in df_view.groupby("period"):
    period_type_series = period_df.get("period_type")
    if period_type_series is None or period_type_series.empty:
        continue
    period_type = period_type_series.iloc[-1]
    if pd.isna(period_type) or period_type not in {"long", "short"}:
        continue
    fillcolor = "green" if period_type == "long" else "red"
    period_end = period_df["timestamp"].iloc[-1]
    fig.add_vrect(
        x0=period_df["timestamp"].iloc[0],
        x1=period_end + bar_duration,
        fillcolor=fillcolor,
        opacity=0.4,
        layer="below",
        line_width=0,
    )

    # Draw horizontal line at per-period extreme and compute move from signal
    period_start_ts = period_df["timestamp"].iloc[0]
    period_end_ts = period_end + bar_duration

    if period_type == "long":
        y_extreme = period_df["high"].max()
        # locate the LONG signal row within this period; fallback to first row
        sig_rows = period_df[period_df["signal"] == "LONG"]
        start_close = (sig_rows["close"].iloc[0] if not sig_rows.empty else period_df["close"].iloc[0])
        if pd.notna(start_close) and pd.notna(y_extreme) and start_close > 0:
            long_peak_gains.append((y_extreme / start_close - 1.0) * 100.0)
        fig.add_shape(
            type="line",
            x0=period_start_ts, x1=period_end_ts,
            y0=y_extreme, y1=y_extreme,
            xref="x", yref="y",
            line=dict(color="green", width=2, dash="dot"),
            layer="above",
        )
    elif period_type == "short":
        y_extreme = period_df["low"].min()
        sig_rows = period_df[period_df["signal"] == "SHORT"]
        start_close = (sig_rows["close"].iloc[0] if not sig_rows.empty else period_df["close"].iloc[0])
        if pd.notna(start_close) and pd.notna(y_extreme) and start_close > 0:
            short_trough_drops.append((1.0 - (y_extreme / start_close)) * 100.0)
        fig.add_shape(
            type="line",
            x0=period_start_ts, x1=period_end_ts,
            y0=y_extreme, y1=y_extreme,
            xref="x", yref="y",
            line=dict(color="red", width=2, dash="dot"),
            layer="above",
        )

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

fig.update_yaxes(title="Price", rangemode="normal", autorange=True, fixedrange=False)

fig.update_layout(
    dragmode="pan",
    xaxis_rangeslider_visible=True,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    uirevision="constant",
    autosize=True
)

# -------------------- Backtest Metrics --------------------
avg_long_gain = (sum(long_peak_gains) / len(long_peak_gains)) if long_peak_gains else None
avg_short_drop = (sum(short_trough_drops) / len(short_trough_drops)) if short_trough_drops else None

st.subheader("Backtest Metrikleri")
bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Ortalama Tepe Artış (%)", f"{avg_long_gain:.2f}" if avg_long_gain is not None else "-",
           help="LONG sinyal kapanışından periyot içi en yüksek fiyata yüzde artış ortalaması")
bc2.metric("Ortalama Dip Düşüş (%)", f"{avg_short_drop:.2f}" if avg_short_drop is not None else "-",
           help="SHORT sinyal kapanışından periyot içi en düşük fiyata yüzde düşüş ortalaması")
bc3.metric("LONG Periyot Sayısı", f"{len(long_peak_gains)}")
bc4.metric("SHORT Periyot Sayısı", f"{len(short_trough_drops)}")

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

st.plotly_chart(
    fig,
    use_container_width=False,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "responsive": True
    }
)

# ---- Tüm veri (Excel'deki tüm günler) için dinamik metrikler ----
roi_times = []
roi_values = []  # cumulative PnL in USDT
cum_pnl = 0.0

in_position = False
pos_side = None  # 'long' or 'short'
entry_price = None
stop_price = None
tp_price = None

for i, row in df_view.iterrows():
    ts = row["timestamp"]
    high = row.get("high")
    low = row.get("low")
    close = row.get("close")
    ma_val = row.get("MA")

    if not in_position:
        sig = row.get("signal")
        if sig == "LONG" and pd.notna(ma_val) and ma_val > 0 and close > 0:
            in_position = True
            pos_side = "long"
            entry_price = float(close)
            stop_price = float(ma_val) * (1.0 - stop_threshold_pct / 100.0)
            tp_price = entry_price * (1.0 + tp_long_pct / 100.0)
        elif sig == "SHORT" and pd.notna(ma_val) and ma_val > 0 and close > 0:
            in_position = True
            pos_side = "short"
            entry_price = float(close)
            stop_price = float(ma_val) * (1.0 + stop_threshold_pct / 100.0)
            tp_price = entry_price * (1.0 - tp_short_pct / 100.0)

    if in_position and pd.notna(high) and pd.notna(low):
        exit_hit = None
        exit_price = None
        if pos_side == "long":
            if low <= stop_price:
                exit_hit = "stop"; exit_price = float(stop_price)
            elif high >= tp_price:
                exit_hit = "tp"; exit_price = float(tp_price)
            if exit_hit:
                qty = position_size / entry_price if entry_price else 0.0
                trade_pnl = (exit_price - entry_price) * qty
                cum_pnl += trade_pnl
                in_position = False; pos_side = None; entry_price = stop_price = tp_price = None
        elif pos_side == "short":
            if high >= stop_price:
                exit_hit = "stop"; exit_price = float(stop_price)
            elif low <= tp_price:
                exit_hit = "tp"; exit_price = float(tp_price)
            if exit_hit:
                qty = position_size / entry_price if entry_price else 0.0
                trade_pnl = (entry_price - exit_price) * qty
                cum_pnl += trade_pnl
                in_position = False; pos_side = None; entry_price = stop_price = tp_price = None

    roi_times.append(ts)
    roi_values.append(cum_pnl)

roi_fig = go.Figure()
roi_fig.add_hline(y=0, line_width=1, opacity=0.5)
roi_fig.add_trace(go.Scatter(x=roi_times, y=roi_values, mode="lines", name="ROI (PnL)", line=dict(color="#2a6fdb")))
roi_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False, yaxis_title="PnL (USDT)")
st.subheader("ROI (Kumulatif PnL)")
st.plotly_chart(roi_fig, use_container_width=False, config={"displayModeBar": True, "responsive": True})

df_full_calc = df_full.copy()
df_full_calc["MA"] = df_full_calc["close"].rolling(int(ma_window), min_periods=int(ma_window)).mean()

# Tam veride sinyal/periyotları aynı eşi̇klerle hesapla
signals_f, period_ids_f, period_types_f = [], [], []
current_period_f, pending_long_f, pending_short_f, current_regime_f = 0, False, False, None
for _, row in df_full_calc.iterrows():
    ma_val, close_val = row["MA"], row["close"]
    if pd.isna(ma_val) or ma_val == 0:
        signals_f.append(""); period_ids_f.append(current_period_f); period_types_f.append(current_regime_f); continue
    rel_diff = (close_val - ma_val) / ma_val
    if rel_diff <= -threshold_ratio: pending_long_f = True
    if rel_diff >= threshold_ratio: pending_short_f = True
    signal = ""
    if pending_long_f and rel_diff > threshold_ratio:
        signal = "LONG"; pending_long_f = False; pending_short_f = True; current_period_f += 1; current_regime_f = "long"
    elif pending_short_f and rel_diff < -threshold_ratio:
        signal = "SHORT"; pending_short_f = False; pending_long_f = True; current_period_f += 1; current_regime_f = "short"
    signals_f.append(signal); period_ids_f.append(current_period_f); period_types_f.append(current_regime_f)

df_full_calc["signal"] = pd.Series(signals_f, index=df_full_calc.index)
df_full_calc["period"] = pd.Series(period_ids_f, index=df_full_calc.index, dtype="int64")
df_full_calc["period_type"] = pd.Series(period_types_f, index=df_full_calc.index, dtype="object")

long_peak_gains_full, short_trough_drops_full = [], []
for _, pdf in df_full_calc.groupby("period"):
    pts = pdf.get("period_type")
    if pts is None or pts.empty: continue
    ptype = pts.iloc[-1]
    if pd.isna(ptype) or ptype not in {"long", "short"}: continue
    if ptype == "long":
        y_ext = pdf["high"].max(); srows = pdf[pdf["signal"] == "LONG"]
        start_close = (srows["close"].iloc[0] if not srows.empty else pdf["close"].iloc[0])
        if pd.notna(start_close) and pd.notna(y_ext) and start_close > 0:
            long_peak_gains_full.append((y_ext / start_close - 1.0) * 100.0)
    else:
        y_ext = pdf["low"].min(); srows = pdf[pdf["signal"] == "SHORT"]
        start_close = (srows["close"].iloc[0] if not srows.empty else pdf["close"].iloc[0])
        if pd.notna(start_close) and pd.notna(y_ext) and start_close > 0:
            short_trough_drops_full.append((1.0 - (y_ext / start_close)) * 100.0)

avg_long_gain_full = (sum(long_peak_gains_full) / len(long_peak_gains_full)) if long_peak_gains_full else None
avg_short_drop_full = (sum(short_trough_drops_full) / len(short_trough_drops_full)) if short_trough_drops_full else None

b2c1, b2c2, b2c3, b2c4 = st.columns(4)
b2c1.metric("Tüm Veri Ort. Tepe Artış (%)", f"{avg_long_gain_full:.2f}" if avg_long_gain_full is not None else "-",
           help="Tüm veri LONG sinyal kapanışından periyot içi en yüksek fiyata yüzde artış ortalaması")
b2c2.metric("Tüm Veri Ort. Dip Düşüş (%)", f"{avg_short_drop_full:.2f}" if avg_short_drop_full is not None else "-",
           help="Tüm veri SHORT sinyal kapanışından periyot içi en düşük fiyata yüzde düşüş ortalaması")
b2c3.metric("Tüm Veri LONG Periyot", f"{len(long_peak_gains_full)}")
b2c4.metric("Tüm Veri SHORT Periyot", f"{len(short_trough_drops_full)}")

# -------------------- Data Table --------------------
st.subheader("Veri Tablosu")
show_cols = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "MA",
    "signal",
    "period",
    "period_type",
]
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
