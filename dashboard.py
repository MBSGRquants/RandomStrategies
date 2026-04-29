import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

OUTPUT_DIR = r"U:\Gruppo Esperia Comune\Quant Team\Sviluppo\RandomStrategies\output"
TRADING_DAYS = 252

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_navs = pd.read_csv(
        f"{OUTPUT_DIR}/mean_navs.csv", index_col=0, parse_dates=True
    )
    summary = pd.read_csv(f"{OUTPUT_DIR}/summary_table.csv")
    try:
        bench_pct = pd.read_csv(f"{OUTPUT_DIR}/benchmark_percentile.csv")
    except FileNotFoundError:
        bench_pct = pd.DataFrame(columns=["N", "freq", "bench_percentile"])
    return mean_navs, summary, bench_pct


def _nav_metrics(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    n_days = len(nav)
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    perf_ann = (1 + total_ret) ** (TRADING_DAYS / n_days) - 1
    vol_ann = rets.std() * np.sqrt(TRADING_DAYS)
    perf_vol = perf_ann / vol_ann if vol_ann != 0 else np.nan
    max_dd = ((nav / nav.cummax()) - 1).min()
    dd_ret = abs(max_dd) / perf_ann if perf_ann != 0 else np.nan
    dd_vol = abs(max_dd) / vol_ann if vol_ann != 0 else np.nan
    return {
        "perf_ann": perf_ann,
        "vol_ann": vol_ann,
        "perf/vol": perf_vol,
        "max_dd": max_dd,
        "dd/ret": dd_ret,
        "dd/vol": dd_vol,
    }


def build_table(
    mean_navs: pd.DataFrame,
    summary: pd.DataFrame,
    bench_pct: pd.DataFrame,
    selected_combos: list[str],
) -> pd.DataFrame:
    rows = []

    bench_m = _nav_metrics(mean_navs["bench_EW"])
    rows.append({"N": "—", "freq": "Benchmark EW", "weighting": "—", **bench_m, "bench_%ile vs sims": np.nan})

    for combo in selected_combos:
        n_str, freq, w = combo.split(" | ")
        N = int(n_str.replace("N=", ""))
        col_name = f"N{N}_{freq}_{w}"
        nav = mean_navs[col_name]
        pct_row = bench_pct[
            (bench_pct["N"] == N) & (bench_pct["freq"] == freq) & (bench_pct["weighting"] == w)
        ] if "weighting" in bench_pct.columns else bench_pct[
            (bench_pct["N"] == N) & (bench_pct["freq"] == freq)
        ]
        pct_val = pct_row["bench_percentile"].iloc[0] if not pct_row.empty else np.nan
        rows.append({"N": N, "freq": freq, "weighting": w, **_nav_metrics(nav), "bench_%ile vs sims": pct_val})

    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    pct_cols = ["perf_ann", "vol_ann", "max_dd"]
    ratio_cols = ["perf/vol", "dd/ret", "dd/vol"]
    out = df.copy()
    for c in pct_cols:
        out[c] = out[c].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for c in ratio_cols:
        out[c] = out[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    out["bench_%ile vs sims"] = out["bench_%ile vs sims"].apply(
        lambda x: f"{x:.1f}°" if pd.notna(x) else "—"
    )
    return out


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Random Strategies", layout="wide")
st.title("Random Strategies — Impact Analysis")

mean_navs, summary, bench_pct = load_data()

all_n = sorted(summary["N"].unique().tolist())
all_freq = summary["freq"].unique().tolist() if "freq" in summary.columns else []
all_w = sorted(summary["weighting"].unique().tolist()) if "weighting" in summary.columns else ["equal"]

sel_col_n, sel_col_freq, sel_col_w, sel_col_spacer = st.columns([1, 2, 2, 3])

with sel_col_n:
    st.markdown("**N titoli**")
    chosen_n = [n for n in all_n if st.checkbox(f"N = {n}", value=True, key=f"n_{n}")]

with sel_col_freq:
    st.markdown("**Frequenza**")
    chosen_freq = [f for f in all_freq if st.checkbox(f.capitalize(), value=True, key=f"freq_{f}")]

with sel_col_w:
    st.markdown("**Weighting**")
    chosen_w = [w for w in all_w if st.checkbox(w.capitalize(), value=True, key=f"w_{w}")]

selected = [
    f"N={n} | {f} | {w}"
    for n in chosen_n
    for f in chosen_freq
    for w in chosen_w
    if not summary[(summary["N"] == n) & (summary["freq"] == f) & (summary["weighting"] == w)].empty
]

if not selected:
    st.warning("Seleziona almeno un N, una frequenza e un weighting.")
    st.stop()

col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Metriche di performance")
    table_df = build_table(mean_navs, summary, bench_pct, selected)
    st.dataframe(_fmt(table_df), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("NAV medie")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=mean_navs.index,
        y=mean_navs["bench_EW"],
        name="Benchmark EW",
        line=dict(color="red", width=2, dash="dash"),
    ))

    for i, combo in enumerate(selected):
        n_str, freq, w = combo.split(" | ")
        N = int(n_str.replace("N=", ""))
        col_name = f"N{N}_{freq}_{w}"
        fig.add_trace(go.Scatter(
            x=mean_navs.index,
            y=mean_navs[col_name],
            name=combo,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="NAV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)
