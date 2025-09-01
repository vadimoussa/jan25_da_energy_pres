import os, io, base64, zipfile, re

# Headless-safe Matplotlib backend (required on Streamlit Cloud)
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import requests
from pandas.api.types import is_numeric_dtype

# =========================
# App title & navigation
# =========================
st.title("Electricity Production and Consumption in France")
st.sidebar.title("Table of contents")
pages = ["Exploration", "DataVizualization", "Modelling", "Report (PDF)"]
page = st.sidebar.radio("Go to", pages)

# =========================
# Data loading helpers
# =========================
SEPARATORS = [",", ";", "\t", "|"]

def _try_read_csv(bio_or_path, sep):
    try:
        return pd.read_csv(
            bio_or_path,
            sep=sep,
            low_memory=False,
            on_bad_lines="skip",
            engine="python"
        )
    except Exception:
        return None

def _read_csv_try_seps_from_bytes(buf_bytes):
    bio = io.BytesIO(buf_bytes)
    best = None
    for s in SEPARATORS:
        bio.seek(0)
        df = _try_read_csv(bio, s)
        if df is None:
            continue
        if best is None or df.shape[1] > best.shape[1]:
            best = df
        if df.shape[1] >= 5:
            break
    return best if best is not None else pd.DataFrame()

def _normalize_url(url: str) -> str:
    u = url.strip()
    # Dropbox: convert sharing page → direct file
    if "dropbox.com" in u and "dl.dropboxusercontent.com" not in u:
        u = u.replace("www.dropbox.com", "dl.dropboxusercontent.com")
        u = re.sub(r"\?dl=0($|&)", "?dl=1\\1", u)
        if "dl=1" not in u:
            u += ("&" if "?" in u else "?") + "dl=1"
    # Google Drive: file link -> export download
    m = re.search(r"drive.google.com/(?:file/d/|open\?id=)([^/?&]+)", u)
    if m:
        file_id = m.group(1)
        u = f"https://drive.google.com/uc?export=download&id={file_id}"
    return u

def _fetch_url_bytes(url: str):
    """
    Returns: (ok: bool, raw: bytes|None, meta: dict)
    meta keys: status, content_type, length, final_url, error (optional)
    """
    meta = {"status": None, "content_type": None, "length": None, "final_url": None, "error": None}
    try:
        fixed = _normalize_url(url)
        headers = {"User-Agent": "Mozilla/5.0 (Streamlit-DataLoader)"}
        r = requests.get(fixed, headers=headers, timeout=120, allow_redirects=True)
        meta["status"] = r.status_code
        meta["content_type"] = r.headers.get("content-type", "")
        meta["length"] = r.headers.get("content-length", None)
        meta["final_url"] = r.url
        if r.status_code >= 400:
            meta["error"] = f"HTTP {r.status_code}"
            return False, None, meta
        return True, r.content, meta
    except Exception as e:
        meta["error"] = str(e)
        return False, None, meta

def _parse_bytes_to_df(raw: bytes, url_hint: str = "") -> pd.DataFrame:
    """Try ZIP first, then CSV."""
    if not raw:
        return pd.DataFrame()

    # Try ZIP (by header or extension)
    is_zip_ext = url_hint.lower().endswith(".zip")
    looks_like_zip = raw.startswith(b"PK\x03\x04")
    if is_zip_ext or looks_like_zip:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    return pd.DataFrame()
                # pick the first CSV (you can rename so the intended one is first)
                data = zf.read(csvs[0])
                return _read_csv_try_seps_from_bytes(data)
        except Exception:
            pass  # fall through to CSV attempt

    # Try CSV regardless of content-type (some hosts mislabel)
    try:
        return _read_csv_try_seps_from_bytes(raw)
    except Exception:
        return pd.DataFrame()

def _read_csv_smart(uploaded_file=None, path=None, url=None):
    # 1) Uploaded file (CSV/ZIP)
    if uploaded_file is not None:
        name = (uploaded_file.name or "").lower()
        raw = uploaded_file.read() or b""
        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not csvs:
                        return pd.DataFrame()
                    data = zf.read(csvs[0])
                    return _read_csv_try_seps_from_bytes(data)
            except Exception:
                return pd.DataFrame()
        # assume CSV otherwise
        return _read_csv_try_seps_from_bytes(raw)

    # 2) Remote URL (CSV/ZIP)
    if url:
        ok, raw, meta = _fetch_url_bytes(url)
        st.session_state["_last_fetch_meta"] = meta  # store for sidebar debug
        if not ok or raw is None:
            return pd.DataFrame()
        return _parse_bytes_to_df(raw, url_hint=meta.get("final_url") or url)

    # 3) Local path (CSV/ZIP)
    if path and os.path.exists(path):
        try:
            pl = path.lower()
            if pl.endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not csvs:
                        return pd.DataFrame()
                    data = zf.read(csvs[0])
                    return _read_csv_try_seps_from_bytes(data)
            with open(path, "rb") as f:
                return _read_csv_try_seps_from_bytes(f.read())
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()

@st.cache_data
def load_data(uploaded_file=None, path=None, url=None):
    return _read_csv_smart(uploaded_file, path, url)

# =========================
# Sidebar data source
# =========================
st.sidebar.subheader("Data source")
uploaded_csv = st.sidebar.file_uploader("Upload CSV or ZIP (≤200 MB)", type=["csv", "zip"])
custom_path = st.sidebar.text_input("…or enter a local CSV/ZIP path (local runs)", value="")
remote_url  = st.sidebar.text_input(
    "…or enter a direct remote URL (CSV/ZIP)",
    value="",
    help="Examples: GitHub Releases direct asset, Dropbox (dl.dropboxusercontent.com), Google Drive export link."
)

# Optional: reuse a local constant if available
try:
    import importlib
    project_mod = importlib.import_module("project_sl")
    default_path = getattr(project_mod, "data_path", "")
    if (not uploaded_csv) and (not custom_path) and (not remote_url) and default_path:
        custom_path = default_path
except Exception:
    pass

df_raw = load_data(uploaded_csv, custom_path, remote_url)

# Debug panel for last fetch (only if URL used)
if remote_url:
    meta = st.session_state.get("_last_fetch_meta")
    with st.sidebar.expander("Remote fetch debug", expanded=False):
        if meta:
            st.write({k: v for k, v in meta.items() if v is not None})
        else:
            st.write("No fetch attempted or cached.")

# Gentle guidance if URL provided but no data parsed
if remote_url and df_raw.empty:
    st.sidebar.error(
        "Could not load data from the provided URL. "
        "Make sure it downloads a .csv or .zip directly (no HTML page)."
    )

if not df_raw.empty and df_raw.shape[1] == 1:
    st.sidebar.warning(
        "Data loaded as a single column. The parser tried ',', ';', tab, and '|'. "
        "If columns still look wrong, provide a ZIP containing a clean CSV."
    )

# =========================
# Data preparation
# =========================
RENAME_FR_EN = {
    'Code INSEE région': 'INSEE_region_code',
    'Nature': 'Nature',
    'Date': 'Date',
    'Région': 'Region',
    'Heure': 'Time',
    'Date - Heure': 'Date-Time',
    'Consommation (MW)': 'Consumption',
    'Thermique (MW)': 'Thermal',
    'Nucléaire (MW)': 'Nuclear',
    'Eolien (MW)': 'Wind',
    'Solaire (MW)': 'Solar',
    'Hydraulique (MW)': 'Hydraulic',
    'Pompage (MW)': 'Pumped_storage',
    'Bioénergies (MW)': 'Biomass',
    'Ech. physiques (MW)': 'Physical_exchanges',
    'Stockage batterie (MW)': 'Battery_storage',
    'Déstockage batterie (MW)': 'Battery_destock',
    'Eolien terrestre (MW)': 'Wind_onshore',
    'Eolien offshore (MW)': 'Wind_offshore',
}
MAIN_SOURCES = ['Thermal','Nuclear','Wind','Solar','Hydraulic','Biomass']
MWH_COLUMNS = ['Consumption','Thermal','Nuclear','Wind','Solar','Hydraulic','Biomass','Pumped_storage','Physical_exchanges']

def _drop_all_zero_cols(d, names=('Battery_storage','Battery_destock','Wind_onshore','Wind_offshore')):
    d = d.copy()
    for c in names:
        if c in d.columns:
            vals = pd.to_numeric(d[c], errors='coerce')
            if (vals.fillna(0) == 0).all():
                d = d.drop(columns=[c])
    return d

def preprocess(df):
    if df.empty:
        return df

    # Standardize columns (French → English)
    df = df.rename(RENAME_FR_EN, axis=1)
    if 'Date - Heure' in df.columns and 'Date-Time' not in df.columns:
        df = df.rename(columns={'Date - Heure': 'Date-Time'})

    # Keep definitive data if present
    if 'Nature' in df.columns:
        m = df['Nature'].astype(str).str.contains("Données définitives|Definitive data", case=False, regex=True)
        if m.any():
            df = df[m].copy()
        df['Nature'] = df['Nature'].replace('Données définitives', 'Definitive data')

    # Normalize Date-Time (remove timezone suffix if any)
    if 'Date-Time' in df.columns:
        dt_str = df['Date-Time'].astype(str)
        dt_str = np.where(pd.Series(dt_str).str.match(r'.*\+\d{2}:\d{2}$', na=False),
                          pd.Series(dt_str).str[:-6], dt_str)
        df['Date-Time'] = pd.to_datetime(dt_str, errors='coerce')

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Calendar fields
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.day_name()
    if 'Date-Time' in df.columns and df['Date-Time'].notna().any():
        df['Hour'] = df['Date-Time'].dt.hour
        # Use string YearMonth to avoid Period dtype issues
        if 'YearMonth' not in df.columns:
            df['YearMonth'] = df['Date-Time'].dt.strftime('%Y-%m')

    # Numeric coercion
    for c in MAIN_SOURCES + ['Consumption','Physical_exchanges','Pumped_storage']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Totals & balances
    if all(c in df.columns for c in MAIN_SOURCES):
        df['ProdTotal'] = df[MAIN_SOURCES].sum(axis=1)
        if 'Physical_exchanges' in df.columns:
            df['ProdTotal_PhyEx'] = df[MAIN_SOURCES + ['Physical_exchanges']].sum(axis=1)
        else:
            df['ProdTotal_PhyEx'] = df['ProdTotal']

    if 'Consumption' in df.columns and 'ProdTotal' in df.columns:
        renew_cols = [c for c in ['Wind','Solar','Hydraulic','Biomass'] if c in df.columns]
        df['Diff_ProdTotal'] = df['ProdTotal'] - df['Consumption']
        df['Renewables'] = df[renew_cols].sum(axis=1) if renew_cols else np.nan
        df['LowCO2'] = df[['Nuclear'] + renew_cols].sum(axis=1) if 'Nuclear' in df.columns and renew_cols else np.nan

    df = _drop_all_zero_cols(df)
    return df

df = preprocess(df_raw)

required_any = ['Date-Time', 'Consumption']
if not df.empty and (not set(required_any).issubset(df.columns)):
    st.sidebar.error(
        "The dataset loaded but required columns are missing. "
        "Expected at least 'Date-Time' and 'Consumption'."
    )

if df.empty:
    st.info("No data loaded yet. Upload a CSV/ZIP, paste a direct Remote URL (CSV/ZIP), or enter a valid local path.")
else:
    with st.sidebar.expander("Columns detected", expanded=False):
        st.write(list(df.columns))

# =========================
# Page 1 — Exploration
# =========================
if page == pages[0] and not df.empty:
    with st.expander("Project Overview", expanded=True):
        st.markdown(
            "- **Context**: France’s power system combines low-CO₂ baseload with variable renewables.\n"
            "- **Dataset**: RTE eco2mix regional definitive data (half-hourly).\n"
            "- **Focus**: Consumption vs production, energy mix, grid balance, and a simple forecasting demo.\n"
        )

    st.write("### Presentation of data")
    st.dataframe(df.head(10))
    st.write("**Shape (rows, columns):**", df.shape)

    if st.checkbox("Show numeric summary (describe)"):
        st.dataframe(df.describe(include=[np.number]))

    if st.checkbox("Show missing values (per column)"):
        st.dataframe(df.isna().sum())

    preview_col = st.selectbox("Preview a column (first 25 values)", options=df.columns)
    st.write(pd.DataFrame({preview_col: df[preview_col].head(25)}))

    with st.expander("Conclusions (summary)"):
        st.markdown(
            "- Distinct daily/weekly/seasonal patterns in consumption.\n"
            "- Nuclear + renewables form the low-CO₂ core; hydro contributes flexibility; wind/solar are variable.\n"
            "- Regional net balances vary between importing and exporting areas.\n"
            "- Simple tree-based regressors with time features and short lags capture core demand patterns.\n"
        )

    with st.expander("References"):
        st.markdown(
            "- RTE Open Data — eco2mix regional definitive dataset  \n"
            "  https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/  \n"
            "- Streamlit documentation — https://docs.streamlit.io/\n"
        )

# =========================
# Page 2 — DataVizualization
# =========================
if page == pages[1] and not df.empty and set(required_any).issubset(df.columns):
    st.write("### DataVizualization")

    try:
        convert_to_mwh = st.checkbox("Convert MW (half-hourly) to MWh (×0.5) for energy totals", value=False)
        if convert_to_mwh:
            df_plot = df.copy()
            for c in MWH_COLUMNS:
                if c in df_plot.columns:
                    df_plot[c] = pd.to_numeric(df_plot[c], errors='coerce') * 0.5
            unit_note = " (aggregations in MWh/TWh)"
        else:
            df_plot = df.copy()
            unit_note = ""

        # Region filter
        if 'Region' in df_plot.columns:
            regions = sorted([str(x) for x in df_plot['Region'].dropna().unique()])
            chosen_regions = st.multiselect("Filter by region", options=regions, default=regions[:1] if regions else [])
            if chosen_regions:
                df_plot = df_plot[df_plot['Region'].astype(str).isin(chosen_regions)]

        # Ensure calendar fields
        if 'Date-Time' in df_plot.columns:
            if 'Month' not in df_plot.columns:
                df_plot['Month'] = pd.to_datetime(df_plot['Date-Time'], errors='coerce').dt.month
            if 'Hour' not in df_plot.columns:
                df_plot['Hour'] = pd.to_datetime(df_plot['Date-Time'], errors='coerce').dt.hour
            if 'Weekday' not in df_plot.columns:
                df_plot['Weekday'] = pd.to_datetime(df_plot['Date-Time'], errors='coerce').dt.day_name()

        def numeric_columns(d):
            return [c for c in d.columns if is_numeric_dtype(d[c])]

        numeric_cols = numeric_columns(df_plot)

        tab1, tab2, tab3, tab4 = st.tabs([
            "General plots",
            "Monthly seasonality",
            "Weekly / Hourly by Region",
            "Grid balance & mix"
        ])

        # --- General plots ---
        with tab1:
            cat_cols = [c for c in df_plot.columns
                        if (df_plot[c].dtype == 'object') or (getattr(df_plot[c].dtype, "name", "") == 'category')]
            if cat_cols:
                chosen_cat = st.selectbox("Countplot — choose a categorical column", options=cat_cols, key="cat1")
                fig = plt.figure()
                sns.countplot(x=chosen_cat, data=df_plot)
                plt.title(f"Distribution of {chosen_cat}")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            if numeric_cols:
                chosen_num = st.selectbox("Histogram — choose a numeric column", options=numeric_cols, key="num1")
                bins = st.slider("Number of bins", 5, 100, 30, key="bins1")
                fig2 = plt.figure()
                sns.histplot(df_plot[chosen_num].dropna(), bins=bins)
                plt.title(f"Distribution of {chosen_num}")
                st.pyplot(fig2)

            if 'Date-Time' in df_plot.columns and numeric_cols:
                st.write("**Time series line plot**")
                y_col = st.selectbox("Y-axis numeric column", options=numeric_cols, key="ts_y1")
                ts = df_plot[['Date-Time', y_col]].dropna().sort_values('Date-Time')
                fig3 = plt.figure()
                plt.plot(ts['Date-Time'], ts[y_col])
                plt.title(f"{y_col} over time{unit_note}")
                plt.xlabel("Date-Time")
                plt.ylabel(y_col)
                plt.xticks(rotation=25, ha="right")
                st.pyplot(fig3)
                st.caption("Shows daily, weekly, and seasonal structure in the series.")

        # --- Monthly seasonality ---
        with tab2:
            st.write("#### Monthly seasonality (averages by month)")
            if 'Month' in df_plot.columns:
                preferred = [c for c in ['Consumption','ProdTotal'] if c in df_plot.columns]
                y_options = preferred + [c for c in numeric_cols if c not in preferred]
                if y_options:
                    y_metric = st.selectbox("Metric", options=y_options, index=0, key="mon_metric")
                    agg = df_plot.groupby('Month')[y_metric].mean().reindex(range(1,13))

                    fig_line = plt.figure()
                    plt.plot(agg.index, agg.values, marker="o")
                    plt.title(f"Average {y_metric} by Month{unit_note}")
                    plt.xlabel("Month (1–12)")
                    plt.ylabel(f"Avg {y_metric}")
                    plt.xticks(range(1,13))
                    st.pyplot(fig_line)

                    fig_bar = plt.figure()
                    plt.bar(agg.index, agg.values)
                    plt.title(f"Average {y_metric} by Month{unit_note}")
                    plt.xlabel("Month (1–12)")
                    plt.ylabel(f"Avg {y_metric}")
                    plt.xticks(range(1,13))
                    st.pyplot(fig_bar)
                    st.caption("Higher winter demand and summer troughs are visible.")

            sources = [c for c in ['Nuclear','Wind','Solar','Hydraulic','Thermal','Biomass'] if c in df_plot.columns]
            if 'Month' in df_plot.columns and sources:
                st.write("#### Monthly energy mix (share over selection)")
                monthly = df_plot.groupby('Month')[sources].mean().reindex(range(1,13))
                as_share = st.checkbox("Show as shares (0–100%)", value=True, key="mix_share")
                plot_data = monthly.div(monthly.sum(axis=1), axis=0)*100 if as_share else monthly

                fig_mix = plt.figure()
                bottom = np.zeros(len(plot_data))
                for col in plot_data.columns:
                    plt.bar(plot_data.index, plot_data[col].values, bottom=bottom, label=col)
                    bottom += plot_data[col].values
                ttl = "Monthly Energy Mix (share %)" if as_share else f"Monthly Energy Mix (avg level{unit_note})"
                plt.title(ttl)
                plt.xlabel("Month (1–12)")
                plt.legend(loc='upper right', ncol=2)
                st.pyplot(fig_mix)
                st.caption("Nuclear provides stable low-CO₂ base; wind/solar vary; hydro adds flexibility.")

        # --- Weekly / Hourly by Region ---
        with tab3:
            st.write("#### Weekly and Hourly demand profiles")
            if {'Weekday','Hour'}.issubset(df_plot.columns):
                ordered_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                try:
                    df_plot['Weekday'] = pd.Categorical(df_plot['Weekday'], categories=ordered_days, ordered=True)
                except Exception:
                    pass

                y_options = [c for c in ['Consumption','ProdTotal'] if c in df_plot.columns] or numeric_cols
                if y_options:
                    y_metric = st.selectbox("Metric", options=y_options, key="prof_metric")

                    if 'Region' in df_plot.columns:
                        region_for_profile = st.selectbox(
                            "Region (optional)", options=["<All>"] + sorted(df_plot['Region'].astype(str).unique().tolist()), index=0
                        )
                        dprof = df_plot if region_for_profile == "<All>" else df_plot[df_plot['Region'].astype(str) == region_for_profile]
                    else:
                        dprof = df_plot

                    wk = dprof.groupby('Weekday')[y_metric].mean()
                    fig_wk = plt.figure()
                    plt.plot(wk.index, wk.values, marker="o")
                    plt.title(f"Average {y_metric} by Weekday{unit_note}")
                    plt.xlabel("Weekday"); plt.ylabel(f"Avg {y_metric}")
                    plt.xticks(rotation=25, ha="right")
                    st.pyplot(fig_wk)

                    hr = dprof.groupby('Hour')[y_metric].mean().reindex(range(24))
                    fig_hr = plt.figure()
                    plt.plot(hr.index, hr.values, marker="o")
                    plt.title(f"Average {y_metric} by Hour{unit_note}")
                    plt.xlabel("Hour (0–23)"); plt.ylabel(f"Avg {y_metric}")
                    plt.xticks(range(0,24,2))
                    st.pyplot(fig_hr)

                    pivot = dprof.pivot_table(index='Weekday', columns='Hour', values=y_metric, aggfunc='mean').reindex(ordered_days)
                    fig_hm = plt.figure()
                    sns.heatmap(pivot, annot=False)
                    plt.title(f"{y_metric} — Weekday × Hour{unit_note}")
                    plt.xlabel("Hour"); plt.ylabel("Weekday")
                    st.pyplot(fig_hm)
                    st.caption("Profiles reflect commuting and evening peaks on weekdays.")
            else:
                st.info("Weekday/Hour not available. Ensure 'Date-Time' exists and is parsed.")

        # --- Grid balance & mix ---
        with tab4:
            if {'Year'}.issubset(df_plot.columns) and all(c in df_plot.columns for c in MAIN_SOURCES) and 'Consumption' in df_plot.columns:
                st.write("**Yearly Energy Production (by source) vs Consumption (TWh)**")
                denom = 1_000_000.0 if convert_to_mwh else 1e6
                yearly_prod = (df_plot.groupby('Year')[MAIN_SOURCES].sum() / denom)
                yearly_cons = (df_plot.groupby('Year')['Consumption'].sum() / denom)
                years = yearly_prod.index.tolist()

                colors = px.colors.qualitative.Set3
                fig = go.Figure()
                for i, col in enumerate(MAIN_SOURCES):
                    fig.add_trace(go.Bar(x=years, y=yearly_prod[col], name=col, marker_color=colors[i % len(colors)]))
                fig.add_trace(go.Scatter(x=years, y=yearly_cons, mode='lines+markers', name='Consumption', line=dict(width=3)))
                fig.update_layout(barmode='stack', template='plotly_white', height=520,
                                  xaxis_title='Year', yaxis_title='Energy (TWh)')
                st.plotly_chart(fig, use_container_width=True)
                st.caption("ProdTotal = Thermal + Nuclear + Wind + Solar + Hydraulic + Biomass.")

            if {'Region','Consumption','ProdTotal'}.issubset(df_plot.columns):
                st.write("**Net Energy Balance by Region**  \n(Consumption − ProdTotal; positive = deficit)")
                bal = df_plot.groupby('Region')[['Consumption','ProdTotal']].sum()
                bal['NetBalance'] = bal['Consumption'] - bal['ProdTotal']
                bal = bal.sort_values('NetBalance', ascending=True)
                fig_barh = go.Figure(go.Bar(x=bal['NetBalance'], y=bal.index, orientation='h'))
                fig_barh.update_layout(template='plotly_white', height=600,
                                       xaxis_title=f'Net Balance ({"MWh" if convert_to_mwh else "MW half-hour sums"})',
                                       yaxis_title='Region')
                st.plotly_chart(fig_barh, use_container_width=True)

            mix_cols = [c for c in ['Nuclear','Wind','Solar','Hydraulic','Thermal','Biomass'] if c in df_plot.columns]
            if mix_cols:
                st.write("**Energy mix (share over selection)**")
                mix_series = df_plot[mix_cols].select_dtypes(include=[np.number]).sum()
                if mix_series.sum() > 0:
                    fig_pie = go.Figure(go.Pie(labels=mix_series.index, values=mix_series.values, hole=0.35))
                    fig_pie.update_layout(template='plotly_white', height=420)
                    st.plotly_chart(fig_pie, use_container_width=True)
                st.caption("Low-CO₂ = Nuclear + Renewables (Wind, Solar, Hydraulic, Biomass).")

    except Exception:
        st.error("Visualization error. Please verify your data source and try again.")

# =========================
# Page 3 — Modelling
# =========================
if page == pages[2] and not df.empty and set(required_any).issubset(df.columns):
    st.write("### Modelling (Consumption forecasting demo)")

    if not {'Date-Time','Consumption'}.issubset(df.columns):
        st.warning("Required columns missing: Date-Time and Consumption.")
    else:
        model_choice = st.selectbox("Model", ["Decision Tree Regressor", "Random Forest Regressor"])
        add_time_feats = st.checkbox("Add time features (hour, dayofweek, month)", value=True)
        add_basic_lags = st.checkbox("Add basic lags (Consumption t-1, t-24)", value=True)
        test_size_pct = st.slider("Test size (chronological split)", 10, 40, 20, step=5)

        work = df[['Date-Time','Consumption','Region'] + [c for c in MAIN_SOURCES if c in df.columns]].copy()
        work = work.dropna(subset=['Consumption']).sort_values('Date-Time')

        if add_time_feats:
            work['hour'] = work['Date-Time'].dt.hour
            work['dayofweek'] = work['Date-Time'].dt.dayofweek
            work['month'] = work['Date-Time'].dt.month
            work['is_weekend'] = work['dayofweek'].isin([5,6]).astype(int)

        if add_basic_lags:
            work['cons_lag1'] = work['Consumption'].shift(1)
            work['cons_lag24'] = work['Consumption'].shift(24)

        if 'Region' in work.columns:
            work['Region'] = work['Region'].astype(str)
            work = pd.get_dummies(work, columns=['Region'], dummy_na=True)

        work = work.dropna()

        split_idx = int(len(work) * (1 - test_size_pct/100.0))
        train_df = work.iloc[:split_idx].copy()
        test_df  = work.iloc[split_idx:].copy()

        y_train = train_df['Consumption'].values
        y_test  = test_df['Consumption'].values
        X_train = train_df.drop(columns=['Consumption','Date-Time']).values
        X_test  = test_df.drop(columns=['Consumption','Date-Time']).values

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        if model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_leaf=5)
        else:
            model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        from sklearn.metrics import r2_score, mean_absolute_error
        preds = model.predict(X_test)
        c1, c2 = st.columns(2)
        with c1: st.metric("R²", f"{r2_score(y_test, preds):.3f}")
        with c2: st.metric("MAE (MW)", f"{mean_absolute_error(y_test, preds):,.0f}")

        st.write("**Actual vs Predicted Consumption (test period)**")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=test_df['Date-Time'], y=y_test, mode='lines', name='Actual'))
        fig_cmp.add_trace(go.Scatter(x=test_df['Date-Time'], y=preds,  mode='lines', name='Predicted'))
        fig_cmp.update_layout(template='plotly_white', height=450, xaxis_title="Date-Time", yaxis_title="Consumption (MW)")
        st.plotly_chart(fig_cmp, use_container_width=True)

# =========================
# Page 4 — Report (PDF)
# =========================
if page == pages[3]:
    st.write("### Report (PDF)")
    st.write("Upload a PDF to view it in the app and provide a download link.")
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if pdf_file is not None:
        pdf_bytes = pdf_file.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf")
