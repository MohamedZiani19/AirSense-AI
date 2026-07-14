"""
app.py — AirSense AI Streamlit Dashboard

Run with:
    streamlit run app.py

Expects Notebooks 1-8 to have already been run, so that
data/processed/airsense_cube.csv and the models/ folder are populated.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import app_utils as utils


def html(raw: str) -> str:
    """Strips leading whitespace from every line of a multi-line HTML
    string before it's handed to st.markdown().

    Why this is needed: Streamlit's markdown parser treats any line
    indented 4+ spaces as a code block (standard Markdown behavior) — so
    HTML written with Python's natural code indentation renders as literal
    text instead of actual HTML, even with unsafe_allow_html=True. This
    strips that indentation so the parser sees plain HTML instead.
    """
    return "\n".join(line.strip() for line in raw.strip("\n").split("\n"))

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AirSense AI", page_icon="🌫️", layout="wide")

# --- Design tokens: "mission control" dark theme -----------------------
# Grounded in the actual subject — a live atmospheric-monitoring readout —
# rather than the telecom reference's literal colors/copy. Chrome uses a
# cyan/purple instrument palette; the EPA AQI colors stay untouched as the
# one place color carries real semantic meaning.
BG_DEEP = "#0A0E17"
BG_PANEL = "#10161F"
BG_CARD = "#141B26"
BORDER = "#212B38"
TEXT_PRIMARY = "#E7ECF2"
TEXT_MUTED = "#6B7A8D"
ACCENT_CYAN = "#35D2E0"
ACCENT_PURPLE = "#8B7CF6"
ACCENT_GREEN = "#34D399"
ACCENT_AMBER = "#F5B94D"
ACCENT_RED = "#F2545B"

METRIC_COLORS = {
    "PM2.5": ACCENT_CYAN, "PM10": ACCENT_PURPLE, "NO2": ACCENT_AMBER,
    "O3": ACCENT_GREEN, "Temperature": "#FF8A65", "Wind Speed": "#4FC3F7",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'IBM Plex Mono', monospace; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; color: {TEXT_PRIMARY}; }}

[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background-color: {BG_DEEP}; }}
section[data-testid="stSidebar"] {{ background-color: {BG_PANEL}; border-right: 1px solid {BORDER}; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; color: {TEXT_PRIMARY}; }}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; letter-spacing: 0.05em; font-size: 0.75rem; text-transform: uppercase; }}

.asai-topbar {{
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(90deg, {BG_PANEL} 0%, {BG_CARD} 100%);
    border: 1px solid {BORDER}; border-radius: 10px;
    padding: 14px 22px; margin-bottom: 18px;
}}
.asai-topbar h1 {{ margin: 0; font-size: 1.35rem; color: {TEXT_PRIMARY}; }}
.asai-topbar .sub {{ color: {TEXT_MUTED}; font-size: 0.78rem; letter-spacing: 0.04em; }}
.asai-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {BG_DEEP}; border: 1px solid {BORDER}; border-radius: 20px;
    padding: 5px 12px; font-size: 0.72rem; color: {TEXT_MUTED}; margin-left: 8px;
}}
.asai-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}

.asai-panel {{
    background-color: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 18px; margin-bottom: 16px;
}}
.asai-panel h4 {{
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: {TEXT_MUTED}; margin: 0 0 12px 0; font-weight: 500;
}}

.asai-card {{
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 4px 4px 0 4px; margin-bottom: 4px;
}}
.asai-card-label {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: {TEXT_MUTED}; padding: 8px 12px 0 12px;
}}
.asai-card-value {{
    font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600;
    color: {TEXT_PRIMARY}; padding: 0 12px;
}}
.asai-card-delta-up {{ color: {ACCENT_RED}; font-size: 0.75rem; padding: 0 12px 8px 12px; }}
.asai-card-delta-down {{ color: {ACCENT_GREEN}; font-size: 0.75rem; padding: 0 12px 8px 12px; }}

.asai-event {{ display: flex; gap: 10px; padding: 6px 0; border-bottom: 1px solid {BORDER}; font-size: 0.8rem; }}
.asai-event-time {{ color: {TEXT_MUTED}; min-width: 60px; }}
.asai-badge {{ font-size: 0.68rem; font-weight: 600; padding: 1px 6px; border-radius: 3px; min-width: 42px; text-align: center; height: fit-content; }}
.asai-badge-info {{ background: {ACCENT_CYAN}22; color: {ACCENT_CYAN}; }}
.asai-badge-warn {{ background: {ACCENT_AMBER}22; color: {ACCENT_AMBER}; }}
.asai-badge-alert {{ background: {ACCENT_RED}22; color: {ACCENT_RED}; }}
.asai-event-msg {{ color: {TEXT_PRIMARY}; }}

.asai-aqi-banner {{
    border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;
    display: flex; justify-content: space-between; align-items: center;
}}

table.asai-rank {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
table.asai-rank th {{
    text-align: left; color: {TEXT_MUTED}; font-weight: 500; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 8px 10px;
    border-bottom: 1px solid {BORDER};
}}
table.asai-rank td {{ padding: 9px 10px; border-bottom: 1px solid {BORDER}; color: {TEXT_PRIMARY}; }}
.asai-rank-badge {{
    display: inline-block; width: 20px; height: 20px; border-radius: 5px;
    background: {BG_CARD}; color: {TEXT_MUTED}; text-align: center; line-height: 20px; font-size: 0.72rem;
}}
.asai-bar-track {{ background: {BG_DEEP}; border-radius: 3px; height: 6px; width: 100px; overflow: hidden; }}
.asai-bar-fill {{ height: 100%; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATHS = utils.get_project_paths(PROJECT_ROOT)


@st.cache_resource
def load_everything():
    config = utils.load_deployment_config(PATHS["config_path"])
    model = utils.load_best_model(config, PATHS)
    cube = utils.load_cube(PATHS["cube_path"])
    comparison = utils.load_comparison_table(PATHS["comparison_path"])
    return config, model, cube, comparison


try:
    config, model, cube, comparison = load_everything()
except FileNotFoundError as e:
    st.error(
        "Required files not found. Make sure Notebooks 1 through 8 have been "
        "run first, so that `data/processed/airsense_cube.csv` and the "
        "`models/` folder are populated.\n\n"
        f"Missing: {e}"
    )
    st.stop()


@st.cache_data
def get_forecast(_model, _config, _cube):
    return utils.generate_forecast(_model, _config, _cube)


forecast_times, forecast_values = get_forecast(model, config, cube)

AQI_RANK = {"Good": 0, "Moderate": 1, "Unhealthy for Sensitive Groups": 2,
            "Unhealthy": 3, "Very Unhealthy": 4, "Hazardous": 5}


# ---------------------------------------------------------------------------
# Helpers specific to this dashboard's visuals
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def glow_sparkline(series_times, series_values, color, height=90):
    """A small glowing area sparkline, styled after mission-control style
    telemetry cards: a soft multi-layer glow behind a crisp line, gradient
    fill underneath, axes hidden entirely."""
    fig = go.Figure()
    # Glow layers: same line, increasing width, decreasing opacity
    for width, opacity in [(10, 0.04), (6, 0.08), (3, 0.18)]:
        fig.add_trace(go.Scatter(
            x=series_times, y=series_values, mode="lines",
            line=dict(color=color, width=width), opacity=opacity,
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=series_times, y=series_values, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def metric_card(col, label, value_str, delta_str, delta_is_bad, times, values, color):
    with col:
        delta_class = "asai-card-delta-up" if delta_is_bad else "asai-card-delta-down"
        arrow = "▲" if delta_str.strip().startswith("+") else "▼"
        st.markdown(
            html(f"""<div class="asai-card">
                <div class="asai-card-label">{label}</div>
                <div class="asai-card-value">{value_str}
                    <span class="{delta_class}" style="font-size:0.7rem;">{arrow} {delta_str}</span>
                </div>
            </div>"""),
            unsafe_allow_html=True,
        )
        st.plotly_chart(glow_sparkline(times, values, color), use_container_width=True,
                         config={"displayModeBar": False}, key=f"spark_{label}")


def build_event_log(cube: pd.DataFrame, lookback_hours: int = 24 * 7) -> list[dict]:
    """Generates a real event log from actual data: AQI category
    transitions and statistical PM2.5 spikes over the lookback window —
    not simulated/fake events."""
    window = cube.tail(lookback_hours).reset_index(drop=True)
    events = []

    categories = window["PM2.5"].apply(lambda v: utils.get_aqi_category(v)["category"])
    for i in range(1, len(window)):
        if categories[i] != categories[i - 1]:
            old_rank, new_rank = AQI_RANK[categories[i - 1]], AQI_RANK[categories[i]]
            if new_rank > old_rank:
                level = "ALERT" if new_rank >= 3 else "WARN"
                msg = f"AQI worsened: {categories[i-1]} → {categories[i]}"
            else:
                level = "INFO"
                msg = f"AQI improved: {categories[i-1]} → {categories[i]}"
            events.append({"time": window["time"][i], "level": level, "msg": msg})

    mean, std = window["PM2.5"].mean(), window["PM2.5"].std()
    spikes = window[window["PM2.5"] > mean + 2.5 * std]
    for _, row in spikes.iterrows():
        events.append({
            "time": row["time"], "level": "ALERT",
            "msg": f"PM2.5 spike detected: {row['PM2.5']:.1f} μg/m³",
        })

    events.sort(key=lambda e: e["time"], reverse=True)
    return events[:8]


def render_event_log(events: list[dict]):
    badge_class = {"INFO": "asai-badge-info", "WARN": "asai-badge-warn", "ALERT": "asai-badge-alert"}
    if not events:
        st.markdown(f"<span style='color:{TEXT_MUTED};font-size:0.85rem;'>No notable events in this window.</span>",
                     unsafe_allow_html=True)
        return
    for e in events:
        st.markdown(
            html(f"""<div class="asai-event">
                <span class="asai-event-time">{e['time']:%m-%d %H:%M}</span>
                <span class="asai-badge {badge_class[e['level']]}">{e['level']}</span>
                <span class="asai-event-msg">{e['msg']}</span>
            </div>"""),
            unsafe_allow_html=True,
        )


def render_ranked_hours(cube: pd.DataFrame, lookback_hours: int = 24 * 7, top_n: int = 5):
    window = cube.tail(lookback_hours).copy()
    window = window.sort_values("PM2.5", ascending=False).head(top_n).reset_index(drop=True)
    max_pm25 = cube["PM2.5"].max()

    rows_html = ""
    for i, row in window.iterrows():
        aqi = utils.get_aqi_category(row["PM2.5"])
        score = row["PM2.5"] / max_pm25
        rows_html += html(f"""
        <tr>
            <td><span class="asai-rank-badge">{i+1}</span></td>
            <td>{row['time']:%Y-%m-%d %H:%M}</td>
            <td style="color:{aqi['color']};font-weight:600;">{row['PM2.5']:.1f} μg/m³</td>
            <td>{aqi['category']}</td>
            <td>{row['Wind Speed']:.1f} km/h</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div class="asai-bar-track"><div class="asai-bar-fill" style="width:{score*100:.0f}%;background:{aqi['color']};"></div></div>
                    <span>{score:.2f}</span>
                </div>
            </td>
        </tr>""")

    st.markdown(
        html(f"""<table class="asai-rank">
            <tr><th>Rank</th><th>Time</th><th>PM2.5</th><th>Category</th><th>Wind Speed</th><th>Severity Score</th></tr>
            {rows_html}
        </table>"""),
        unsafe_allow_html=True,
    )


def dark_line_chart(x_series_list, y_series_list, names, colors, y_label="", height=380):
    fig = go.Figure()
    for x, y, name, color in zip(x_series_list, y_series_list, names, colors):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers" if len(x) < 40 else "lines",
                                  name=name, line=dict(color=color, width=2), marker=dict(size=4)))
    fig.update_layout(
        height=height, paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUTED, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=TEXT_PRIMARY)),
        xaxis=dict(gridcolor=BORDER, showline=False), yaxis=dict(gridcolor=BORDER, title=y_label, showline=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

current_pm25 = cube["PM2.5"].iloc[-1]
current_time = cube["time"].iloc[-1]
current_aqi = utils.get_aqi_category(current_pm25)

st.markdown(
    html(f"""
    <div class="asai-topbar">
        <div>
            <h1>🌫️ AirSense AI</h1>
            <div class="sub">PM2.5 FORECASTING · NEW YORK CITY</div>
        </div>
        <div>
            <span class="asai-pill"><span class="asai-dot" style="background:{ACCENT_GREEN};"></span>DATA LIVE</span>
            <span class="asai-pill"><span class="asai-dot" style="background:{ACCENT_PURPLE};"></span>MODEL: {config['model_type']}</span>
            <span class="asai-pill">LAST READING: {current_time:%Y-%m-%d %H:%M}</span>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

tabs = st.tabs(["📊 Dashboard", "📈 Forecast", "🧪 Environmental Signals", "🏆 Model Performance"])

# ---------------------------------------------------------------------------
# Tab 1 — Dashboard (mission-control layout)
# ---------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown('<div class="asai-panel"><h4>Forecast Controls</h4>', unsafe_allow_html=True)
        horizon = st.radio("Horizon", ["6h", "12h", "24h"], index=2, horizontal=True, label_visibility="collapsed")
        horizon_n = {"6h": 6, "12h": 12, "24h": 24}[horizon]

        predicted_next = forecast_values[0]
        peak = max(forecast_values[:horizon_n])
        r2 = comparison.loc[config["model_type"], "R2"] if config["model_type"] in comparison.index else np.nan
        mae = comparison.loc[config["model_type"], "MAE"] if config["model_type"] in comparison.index else np.nan

        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='asai-card-label'>Next Hour</div><div class='asai-card-value' style='font-size:1.1rem;color:{ACCENT_CYAN};'>{predicted_next:.1f} μg/m³</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='asai-card-label'>{horizon} Peak</div><div class='asai-card-value' style='font-size:1.1rem;color:{ACCENT_AMBER};'>{peak:.1f} μg/m³</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        c3.markdown(f"<div class='asai-card-label'>Model R²</div><div class='asai-card-value' style='font-size:1.1rem;color:{ACCENT_GREEN};'>{r2:.3f}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='asai-card-label'>Model MAE</div><div class='asai-card-value' style='font-size:1.1rem;color:{ACCENT_PURPLE};'>{mae:.2f}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            html(f"""<div class="asai-aqi-banner" style="background:{current_aqi['color']}18;border:1px solid {current_aqi['color']}55;">
                <div>
                    <div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;">Current AQI</div>
                    <div style="color:{current_aqi['color']};font-size:1.15rem;font-weight:600;">{current_aqi['category']}</div>
                </div>
                <div style="color:{current_aqi['color']};font-size:1.6rem;font-weight:700;">{current_pm25:.1f}</div>
            </div>"""),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="asai-panel"><h4>Air Quality Event Log</h4>', unsafe_allow_html=True)
        events = build_event_log(cube)
        render_event_log(events)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"<h4 style='color:{TEXT_MUTED};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.06em;margin:4px 0 10px 0;'>Live Analytics — Rolling 7 Days</h4>", unsafe_allow_html=True)

    recent = cube.tail(24 * 7)
    metrics_row1 = st.columns(3)
    metrics_row2 = st.columns(3)

    def delta_pct(col):
        prev, curr = recent[col].iloc[-25], recent[col].iloc[-1]
        if prev == 0:
            return 0.0
        return (curr - prev) / abs(prev) * 100

    metric_specs = [
        ("PM2.5", "μg/m³"), ("PM10", "μg/m³"), ("NO2", "μg/m³"),
        ("O3", "μg/m³"), ("Temperature", "°C"), ("Wind Speed", "km/h"),
    ]
    for i, (col_name, unit) in enumerate(metric_specs):
        d = delta_pct(col_name)
        target_col = metrics_row1[i] if i < 3 else metrics_row2[i - 3]
        metric_card(
            target_col, col_name, f"{recent[col_name].iloc[-1]:.1f} {unit}",
            f"{d:+.1f}%", delta_is_bad=(col_name in ("PM2.5", "PM10", "NO2", "O3") and d > 0),
            times=recent["time"], values=recent[col_name], color=METRIC_COLORS.get(col_name, ACCENT_CYAN),
        )

    st.markdown(f"<h4 style='color:{TEXT_MUTED};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.06em;margin:20px 0 10px 0;'>Worst Hours — Last 7 Days</h4>", unsafe_allow_html=True)
    st.markdown('<div class="asai-panel">', unsafe_allow_html=True)
    render_ranked_hours(cube)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2 — Forecast
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"<span style='color:{TEXT_MUTED};font-size:0.85rem;'>⚠️ Weather held at last known values beyond hour 1 — see Notebook 8 for details.</span>", unsafe_allow_html=True)
    recent_history = cube.tail(72)
    fig = dark_line_chart(
        [recent_history["time"], forecast_times],
        [recent_history["PM2.5"], forecast_values],
        ["Recent actual", "24h forecast"],
        [TEXT_PRIMARY, ACCENT_RED],
        y_label="PM2.5 (μg/m³)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"<h4 style='color:{TEXT_MUTED};font-size:0.85rem;text-transform:uppercase;margin-top:16px;'>Recent Weather</h4>", unsafe_allow_html=True)
    weather_cols = ["Temperature", "Relative Humidity", "Wind Speed", "Cloud Cover"]
    wfig = dark_line_chart(
        [recent_history["time"]] * len(weather_cols),
        [recent_history[c] for c in weather_cols],
        weather_cols, [ACCENT_CYAN, ACCENT_PURPLE, ACCENT_AMBER, ACCENT_GREEN],
    )
    st.plotly_chart(wfig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Tab 3 — Environmental Signals
# ---------------------------------------------------------------------------
with tabs[2]:
    signal_choice = st.multiselect(
        "Select variables to plot",
        options=["PM2.5", "PM10", "NO2", "O3", "Temperature", "Relative Humidity",
                 "Wind Speed", "Cloud Cover", "Precipitation"],
        default=["PM2.5", "PM10"],
    )
    palette = [ACCENT_CYAN, ACCENT_PURPLE, ACCENT_AMBER, ACCENT_GREEN, "#FF8A65", "#4FC3F7", "#F2545B", TEXT_PRIMARY]
    if signal_choice:
        fig = dark_line_chart(
            [cube["time"]] * len(signal_choice), [cube[c] for c in signal_choice],
            signal_choice, palette[:len(signal_choice)],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"<h4 style='color:{TEXT_MUTED};font-size:0.85rem;text-transform:uppercase;margin-top:10px;'>Correlation Heatmap</h4>", unsafe_allow_html=True)
    numeric_cols = ["PM2.5", "PM10", "NO2", "O3", "Temperature", "Relative Humidity",
                     "Wind Speed", "Wind Direction", "Precipitation", "Cloud Cover"]
    numeric_cols = [c for c in numeric_cols if c in cube.columns]
    corr = cube[numeric_cols].corr()

    heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    heat.update_layout(
        height=450, paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUTED, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Tab 4 — Model Performance
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown(
        f"<span style='color:{TEXT_MUTED};font-size:0.85rem;'>⚠️ SARIMA's score reflects a single blind 30-day-ahead "
        "forecast — see Notebook 6's rolling re-forecast demo for a fairer comparison.</span>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        comparison.style.background_gradient(subset=["MAE", "RMSE", "MAPE"], cmap="Reds")
                          .background_gradient(subset=["R2"], cmap="Greens"),
        use_container_width=True,
    )

    bar_colors = [ACCENT_CYAN, ACCENT_PURPLE, ACCENT_AMBER, ACCENT_GREEN, "#FF8A65"]
    fig = go.Figure()
    for i, metric in enumerate(["MAE", "RMSE", "MAPE", "R2"]):
        sorted_vals = comparison[metric].sort_values(ascending=(metric != "R2"))
        fig.add_trace(go.Bar(x=sorted_vals.index, y=sorted_vals.values, name=metric,
                              marker_color=bar_colors[i], visible=(i == 0)))

    buttons = [
        dict(label=m, method="update",
             args=[{"visible": [j == i for j in range(4)]}, {"title": m}])
        for i, m in enumerate(["MAE", "RMSE", "MAPE", "R2"])
    ]
    fig.update_layout(
        height=380, paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUTED, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.15, buttons=buttons,
                           font=dict(color=BG_DEEP))],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        html(f"""<div class="asai-panel" style="text-align:center;">
            <span style="color:{TEXT_MUTED};">WINNING MODEL</span><br>
            <span style="color:{ACCENT_GREEN};font-size:1.4rem;font-weight:700;">{config['model_type']}</span>
        </div>"""),
        unsafe_allow_html=True,
    )
