"""Streamlit dashboard: run the sweetpotato pipeline and explore its results.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import capstone_pipeline as cp  # noqa: E402
from src.etl_pipeline import run_etl  # noqa: E402

RAW_PHENOTYPES = ROOT / "data" / "raw" / "raw_phenotypes.csv"
RAW_GENOTYPES = ROOT / "data" / "raw" / "raw_genotypes.csv"
PHENOTYPES = ROOT / "data" / "processed" / "phenotype_clean.csv"
GENOTYPES = ROOT / "data" / "processed" / "genotype_clean.csv"
ETL_REPORT = ROOT / "data" / "processed" / "etl_report.json"
OUTPUT_DIR = ROOT / "outputs"
COMPARISON = OUTPUT_DIR / "model_comparison_all_targets.csv"
RANKED = OUTPUT_DIR / "ranked_clones_multi_index.csv"
SENSITIVITY = OUTPUT_DIR / "model_sensitivity_by_seed_and_folds.csv"
BENCHMARKS = OUTPUT_DIR / "benchmarks" / "learnable" / "model_comparison_all_datasets.csv"

TARGET_NAMES = {
    "market_index": "Market Index", "dri": "DRI", "eqi": "EQI",
    "parental_value": "Parental Value", "ayi": "AYI",
}
MODEL_NAMES = {"elastic_net": "Elastic Net", "ridge": "Ridge", "mlp_baseline": "MLP"}
CROP_NAMES = {"commonbean": "Common bean", "durum_wheat": "Durum wheat", "maize": "Maize",
              "rice210": "Rice", "soybean": "Soybean"}
SENSITIVITY_SEEDS = (42, 123, 2026)
SENSITIVITY_FOLDS = (5, 7, 10)

# Palette: first three categorical slots (validated all-pairs, light mode), one per
# model in fixed order; text always in ink tokens, never in series colors.
MODEL_COLORS = {"Elastic Net": "#2a78d6", "Ridge": "#eb6834", "MLP": "#1baf7a"}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#ebe9e4"
SURFACE = "#fcfcfb"
BLUE_LIGHT = "#cde2fb"
DIVERGING = [[0.0, "#b8322f"], [0.25, "#ec9a98"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]

st.set_page_config(page_title="Sweetpotato Genomic Prediction", page_icon="🍠", layout="wide")


# --- Helpers ------------------------------------------------------------------

def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load(path: Path) -> pd.DataFrame | None:
    return read_csv(str(path), _mtime(path)) if path.exists() else None


@st.cache_data(show_spinner="Ranking candidates…")
def top_candidates(target: str, top_n: int, seed: int, data_mtime: float) -> pd.DataFrame:
    """Top unphenotyped candidates for one target, ranked separately by each model."""
    ranked = cp.top_candidates_by_model(PHENOTYPES, GENOTYPES, target=target, top_n=top_n, seed=seed)
    ranked["model"] = ranked["model_family"].map(MODEL_NAMES)
    return ranked


def label_models(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        Model=frame["model_family"].map(MODEL_NAMES),
        Target=frame["target"].map(lambda t: TARGET_NAMES.get(t, t)),
    )


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    """Shared chart look: recessive grid and axes, thin rounded bars, ink text."""
    fig.update_layout(
        template="none", height=height, showlegend=legend,
        font=dict(family="Source Sans Pro, Source Sans 3, sans-serif", size=13, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=40 if legend else 12, b=8),
        barcornerradius=4, bargap=0.3, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title=None, font=dict(color=INK_2, size=13)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=13)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", tickfont=dict(color=INK_2),
                     title_font=dict(color=INK_2), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, ticks="",
                     tickfont=dict(color=INK_3), title_font=dict(color=INK_2), automargin=True)
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def model_dot(model: str) -> str:
    return f"<span style='color:{MODEL_COLORS[model]};font-size:1.1em'>●</span> {model}"


# --- Pipeline runs ------------------------------------------------------------

def run_analysis(seed: int) -> None:
    with st.status("Running the pipeline…", expanded=True) as status:
        st.write("Cleaning raw data (ETL)…")
        run_etl(RAW_PHENOTYPES, RAW_GENOTYPES, PHENOTYPES, GENOTYPES)
        st.write("Cross-validating Elastic Net, Ridge and MLP for every index…")
        _, outcomes = cp.compare_model_families_across_targets(
            PHENOTYPES, GENOTYPES, OUTPUT_DIR, seed=seed, return_outcomes=True,
        )
        st.write("Predicting and ranking every genotyped clone…")
        cp.run_capstone_analysis(PHENOTYPES, GENOTYPES, OUTPUT_DIR, seed=seed, outcomes=outcomes)
        status.update(label=f"Analysis finished (seed {seed})", state="complete", expanded=False)


def run_sensitivity() -> None:
    phenotype, genotype = cp.load_inputs(PHENOTYPES, GENOTYPES)
    markers = [column for column in genotype.columns if column != "clone_id"]
    labeled = phenotype.merge(genotype, on="clone_id", how="inner").set_index("clone_id")
    jobs = [(folds, seed, target, family)
            for folds in SENSITIVITY_FOLDS for seed in SENSITIVITY_SEEDS
            for target in cp.TARGET_COLUMNS for family in cp.MODEL_FAMILIES]
    progress = st.progress(0.0, text="Running seed and fold sensitivity…")
    rows = []
    for done, (folds, seed, target, family) in enumerate(jobs, start=1):
        outcome = cp._fit_and_evaluate_target(
            target, labeled[markers], labeled[cp.TARGET_SOURCE_COLUMNS[target]],
            model_family=family, seed=seed, n_cv_folds=folds,
        )
        rows.append({"target": target, "model_family": family, "seed": seed, "n_cv_folds": folds,
                     "cv_r2": outcome["cv_r2"], "cv_rmse": outcome["cv_rmse"], "cv_mae": outcome["cv_mae"]})
        progress.progress(done / len(jobs), text=f"Sensitivity: {done}/{len(jobs)} fits")
    results = pd.DataFrame(rows)
    results.to_csv(SENSITIVITY, index=False)
    (results.groupby(["target", "model_family", "n_cv_folds"], as_index=False)
     .agg(seed_runs=("seed", "nunique"), cv_r2_mean=("cv_r2", "mean"), cv_r2_sd=("cv_r2", "std"),
          cv_r2_min=("cv_r2", "min"), cv_r2_max=("cv_r2", "max"),
          cv_rmse_mean=("cv_rmse", "mean"), cv_mae_mean=("cv_mae", "mean"))
     .to_csv(OUTPUT_DIR / "model_sensitivity_summary.csv", index=False))
    progress.empty()


# --- Sidebar ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🍠 Pipeline")
    with st.container(border=True):
        st.markdown("**Analysis**")
        seed = st.number_input("Random seed", min_value=0, value=cp.DEFAULT_SEED, step=1)
        if st.button("Run analysis", type="primary", width="stretch", icon=":material/play_arrow:"):
            run_analysis(int(seed))
        st.caption("ETL, model comparison and rankings · about 30 s")
    with st.container(border=True):
        st.markdown("**Sensitivity**")
        if st.button("Run sensitivity", width="stretch", icon=":material/tune:",
                     disabled=not PHENOTYPES.exists()):
            run_sensitivity()
        st.caption(f"Seeds {', '.join(map(str, SENSITIVITY_SEEDS))} × "
                   f"{', '.join(map(str, SENSITIVITY_FOLDS))} folds · 135 fits · about 1 min")
    st.caption("Both runs overwrite data/processed/ and the CSVs in outputs/, as the notebooks do.")

# --- Header -------------------------------------------------------------------

st.title("Sweetpotato Genomic Prediction")

if not (PHENOTYPES.exists() and COMPARISON.exists() and RANKED.exists()):
    st.warning("No results yet. Click **Run analysis** in the sidebar.", icon=":material/info:")
    st.stop()

phenotypes = load(PHENOTYPES)
comparison = label_models(load(COMPARISON))
ranked = load(RANKED)
targets = [t for t in cp.TARGET_COLUMNS if t in phenotypes.columns]
target_labels = [TARGET_NAMES[t] for t in targets]
run_seed = int(comparison["seed"].iloc[0])
run_folds = int(comparison["cv_folds"].iloc[0])
data_mtime = _mtime(PHENOTYPES) + _mtime(GENOTYPES)
report = json.loads(ETL_REPORT.read_text(encoding="utf-8")) if ETL_REPORT.exists() else None

st.caption(
    f"{len(targets)} breeding indices · {len(phenotypes)} labeled clones · "
    f"{(report['genotype']['final_row_count'] if report else len(ranked)):,} genotyped clones · "
    f"{run_folds}-fold CV · seed {run_seed}"
)
st.info("Simulated data: phenotypes are partly synthetic and genotypes are simulated, with signal planted "
        "only in the labeled clones. Accuracy shows recovery of that signal; candidate rankings cannot be "
        "validated.", icon=":material/science:")

tab_data, tab_models, tab_candidates, tab_sensitivity, tab_external = st.tabs([
    ":material/dataset: Data", ":material/monitoring: Model performance",
    ":material/format_list_numbered: Candidates", ":material/tune: Sensitivity",
    ":material/public: External benchmarks",
])

# --- Data ---------------------------------------------------------------------

with tab_data:
    if report:
        cols = st.columns(4)
        cols[0].metric("Labeled clones", report["phenotype"]["final_row_count"],
                       help=f"{report['phenotype']['rows_in']} raw phenotype rows before de-duplication",
                       border=True)
        cols[1].metric("Genotyped clones", f"{report['genotype']['final_row_count']:,}",
                       help=f"{report['genotype']['exact_duplicate_rows_dropped']} duplicate rows removed",
                       border=True)
        cols[2].metric("SNP markers", report["genotype"]["marker_count"], border=True)
        missing = report["genotype"]["missing_marker_calls"]
        total = report["genotype"]["final_row_count"] * report["genotype"]["marker_count"]
        cols[3].metric("Missing marker calls", f"{missing:,}", help=f"{missing / total:.2%} of all calls",
                       border=True)

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Index distributions**")
        fig = go.Figure()
        for target in targets:
            fig.add_trace(go.Box(
                y=phenotypes[target], name=TARGET_NAMES[target], boxpoints="outliers",
                marker=dict(color="#2a78d6", size=8, line=dict(color=SURFACE, width=2)),
                line=dict(color="#2a78d6", width=2), fillcolor=BLUE_LIGHT,
                hovertemplate="%{y:.3f}<extra>%{x}</extra>",
            ))
        style(fig, height=380, legend=False)
        fig.update_yaxes(title="Index value (0–1)")
        chart(fig)
    with right, st.container(border=True):
        st.markdown("**Spearman correlations**")
        rho = phenotypes[targets].corr(method="spearman").to_numpy()
        lower = np.where(np.tril(np.ones_like(rho, dtype=bool)), rho, np.nan)
        fig = go.Figure(go.Heatmap(
            z=lower, x=target_labels, y=target_labels, zmin=-1, zmax=1, colorscale=DIVERGING,
            xgap=2, ygap=2, hoverongaps=False, colorbar=dict(thickness=10, outlinewidth=0, len=0.8),
            hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
        ))
        for i in range(len(targets)):
            for j in range(i + 1):
                fig.add_annotation(x=target_labels[j], y=target_labels[i], text=f"{rho[i, j]:.2f}",
                                   showarrow=False,
                                   font=dict(size=13, color="white" if abs(rho[i, j]) > 0.6 else INK))
        style(fig, height=380, legend=False)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK_2))
        chart(fig)

    with st.expander("Cleaned phenotype table"):
        st.dataframe(phenotypes, hide_index=True)

# --- Model performance --------------------------------------------------------

with tab_models:
    best = comparison.loc[comparison.groupby("target")["cv_r2"].idxmax()].set_index("target")
    cols = st.columns(len(targets))
    for col, target in zip(cols, targets):
        with col, st.container(border=True):
            st.caption(f"Best for {TARGET_NAMES[target]}")
            st.markdown(f"<span style='font-size:1.9rem;font-weight:600;color:{INK}'>"
                        f"{best.loc[target, 'cv_r2']:.3f}</span>", unsafe_allow_html=True)
            st.markdown(model_dot(best.loc[target, "Model"]) + " · R²", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Cross-validated R² by index and model**")
        fig = go.Figure()
        for model, color in MODEL_COLORS.items():
            part = comparison[comparison["Model"] == model].set_index("Target").reindex(target_labels)
            fig.add_trace(go.Bar(
                x=target_labels, y=part["cv_r2"], name=model, marker=dict(color=color, line=dict(width=0)),
                customdata=np.stack([part["cv_rmse"], part["cv_mae"]], axis=-1),
                hovertemplate=f"<b>%{{x}}</b> · {model}<br>R² %{{y:.3f}}<br>RMSE %{{customdata[0]:.3f}}"
                              "<br>MAE %{customdata[1]:.3f}<extra></extra>",
            ))
        style(fig, height=380)
        fig.update_yaxes(title="Out-of-fold R²", range=[0, 1], dtick=0.2)
        chart(fig)

    table = (comparison[["Target", "Model", "cv_r2", "cv_rmse", "cv_mae"]]
             .assign(Best=lambda d: d.groupby("Target")["cv_r2"].transform("max").eq(d["cv_r2"]).map({True: "★", False: ""}))
             .sort_values(["Target", "cv_r2"], ascending=[True, False]))
    with st.expander("Metrics table"):
        st.dataframe(
            table, hide_index=True,
            column_config={
                "cv_r2": st.column_config.ProgressColumn("R²", min_value=0.0, max_value=1.0, format="%.3f"),
                "cv_rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                "cv_mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                "Best": st.column_config.TextColumn("Best", width="small"),
            },
        )

    with st.container(border=True):
        head, pick = st.columns([3, 1])
        head.markdown("**Observed vs cross-validated prediction** · Elastic Net (primary model)")
        target = pick.selectbox("Index", targets, format_func=TARGET_NAMES.get, key="scatter_target",
                                label_visibility="collapsed")
        training = ranked[ranked["is_training_clone"]]
        observed, predicted = training[f"observed_{target}"], training[f"cv_predicted_{target}"]
        low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        pad = (high - low) * 0.05
        fig = go.Figure([
            go.Scatter(x=[low - pad, high + pad], y=[low - pad, high + pad], mode="lines", hoverinfo="skip",
                       line=dict(color=INK_3, width=1.5, dash="dash"), name="Perfect prediction"),
            go.Scatter(x=observed, y=predicted, mode="markers", text=training["clone_id"], name="Labeled clone",
                       marker=dict(color="#2a78d6", size=9, line=dict(color=SURFACE, width=2)),
                       hovertemplate="<b>%{text}</b><br>Observed %{x:.3f}<br>Predicted %{y:.3f}<extra></extra>"),
        ])
        style(fig, height=420, legend=False)
        fig.update_xaxes(title=f"Observed {TARGET_NAMES[target]}", range=[low - pad, high + pad],
                         showgrid=True, gridcolor=GRID, constrain="domain")
        fig.update_yaxes(title="Cross-validated prediction", range=[low - pad, high + pad],
                         scaleanchor="x", scaleratio=1)
        plot, stats_col = st.columns([3, 1])
        with plot:
            chart(fig)
        en = comparison[(comparison["target"] == target) & (comparison["model_family"] == "elastic_net")].iloc[0]
        stats_col.metric("CV R²", f"{en['cv_r2']:.3f}", border=True)
        stats_col.metric("Pearson r", f"{np.corrcoef(observed, predicted)[0, 1]:.3f}", border=True)
        stats_col.metric("RMSE", f"{en['cv_rmse']:.3f}", border=True)

# --- Candidates ---------------------------------------------------------------

with tab_candidates:
    cols = st.columns([2, 2, 1])
    target = cols[0].selectbox("Rank by", targets, format_func=TARGET_NAMES.get, key="rank_target")
    top_n = cols[1].select_slider("Shortlist size", options=[5, 10, 20, 30, 50], value=10)
    candidates = top_candidates(target, top_n, run_seed, data_mtime)
    counts = candidates.groupby("clone_id")["model"].nunique()
    shared = set(counts[counts >= 2].index)
    with cols[2]:
        st.write("")
        st.download_button("Download CSV", candidates.to_csv(index=False).encode("utf-8"),
                           file_name=f"top{top_n}_candidates_{target}.csv", mime="text/csv",
                           icon=":material/download:", width="stretch")

    rmse = comparison[comparison["target"] == target].set_index("Model")["cv_rmse"]
    model_cols = st.columns(3)
    for col, model in zip(model_cols, MODEL_COLORS):
        part = candidates[candidates["model"] == model]
        spread = part["predicted_value"].max() - part["predicted_value"].min()
        with col, st.container(border=True):
            st.markdown(f"**{model_dot(model)}**", unsafe_allow_html=True)
            st.dataframe(
                part.assign(Shared=part["clone_id"].isin(shared).map({True: "★", False: ""}))
                [["rank", "clone_id", "predicted_value", "Shared"]],
                hide_index=True, height=min(42 + 35 * len(part), 430),
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "clone_id": st.column_config.TextColumn("Clone"),
                    "predicted_value": st.column_config.NumberColumn("Predicted", format="%.3f"),
                    "Shared": st.column_config.TextColumn("★", width="small",
                                                          help="Also in another model's shortlist"),
                },
            )
            reliable = spread >= rmse[model]
            st.caption(
                f"{':material/check_circle:' if reliable else ':material/warning:'} "
                f"Spread {spread:.3f} vs error {rmse[model]:.3f} · "
                + ("order is meaningful" if reliable else "order within the list is not reliable")
            )

    st.caption(f"★ chosen by two or more models: {', '.join(sorted(shared)) if shared else 'none'}. "
               "When a shortlist's spread is smaller than the model's CV RMSE, treat it as a pool "
               "rather than a ranked order.")

# --- Sensitivity --------------------------------------------------------------

with tab_sensitivity:
    sensitivity = load(SENSITIVITY)
    if sensitivity is None:
        st.warning("No sensitivity results yet. Click **Run sensitivity** in the sidebar.",
                   icon=":material/info:")
    else:
        summary = (label_models(sensitivity)
                   .groupby(["Target", "Model", "n_cv_folds"], as_index=False)["cv_r2"]
                   .agg(mean="mean", sd="std", low="min", high="max"))
        with st.container(border=True):
            st.markdown("**Cross-validated R² across seeds, by outer fold count** · mean ± SD")
            fig = px.line(summary, x="n_cv_folds", y="mean", color="Model", facet_col="Target",
                          markers=True, error_y="sd", color_discrete_map=MODEL_COLORS,
                          category_orders={"Target": target_labels, "Model": list(MODEL_COLORS)},
                          custom_data=["Model", "sd", "low", "high"])
            fig.update_traces(line=dict(width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                              hovertemplate="<b>%{customdata[0]}</b> · %{x} folds<br>Mean R² %{y:.3f} "
                                            "± %{customdata[1]:.3f}<br>Range %{customdata[2]:.3f}–"
                                            "%{customdata[3]:.3f}<extra></extra>")
            fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>",
                                                       font=dict(color=INK, size=13)))
            style(fig, height=380)
            fig.update_layout(margin=dict(t=72), legend=dict(y=1.16))
            fig.update_xaxes(title=None, tickvals=list(SENSITIVITY_FOLDS), showgrid=False)
            fig.update_yaxes(title=None)
            fig.update_yaxes(title="Mean R²", col=1)
            chart(fig)
            st.caption(f"Seeds {', '.join(map(str, sorted(sensitivity['seed'].unique())))}. "
                       "Ridge's final fit does not depend on the seed.")
        with st.expander("Sensitivity table"):
            st.dataframe(summary.rename(columns={"n_cv_folds": "Folds", "mean": "Mean R²", "sd": "SD",
                                                 "low": "Min", "high": "Max"}).round(3), hide_index=True)

# --- External benchmarks ------------------------------------------------------

with tab_external:
    benchmarks = load(BENCHMARKS)
    if benchmarks is None:
        st.warning("No external benchmark results. Run notebooks 01 and 02_external_data_validation.",
                   icon=":material/info:")
    else:
        benchmarks = benchmarks.assign(Model=benchmarks["model_family"].map(MODEL_NAMES),
                                       Crop=benchmarks["dataset"].map(lambda d: CROP_NAMES.get(d, d)))
        n_traits = benchmarks[["dataset", "target"]].drop_duplicates().shape[0]
        wins = (benchmarks.sort_values("cv_r2", ascending=False)
                .drop_duplicates(["dataset", "target"])["Model"].value_counts())
        means = benchmarks.groupby("Model")["cv_r2"].mean()
        cols = st.columns(3)
        for col, model in zip(cols, MODEL_COLORS):
            with col, st.container(border=True):
                st.markdown(model_dot(model), unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:1.9rem;font-weight:600;color:{INK}'>"
                            f"{means.get(model, np.nan):.3f}</span>", unsafe_allow_html=True)
                st.caption(f"Mean R² · best on {int(wins.get(model, 0))} of {n_traits} traits")

        crop = benchmarks.groupby(["Crop", "Model"], as_index=False)["cv_r2"].mean()
        order = (crop[crop["Model"] == "Ridge"].sort_values("cv_r2", ascending=False)["Crop"].tolist())
        with st.container(border=True):
            st.markdown("**Mean cross-validated R² by crop**")
            fig = go.Figure()
            for model, color in MODEL_COLORS.items():
                part = crop[crop["Model"] == model].set_index("Crop").reindex(order)
                fig.add_trace(go.Bar(x=order, y=part["cv_r2"], name=model, marker=dict(color=color),
                                     hovertemplate=f"<b>%{{x}}</b> · {model}<br>Mean R² %{{y:.3f}}<extra></extra>"))
            style(fig, height=380)
            fig.update_yaxes(title="Mean out-of-fold R²", zeroline=True, zerolinecolor=INK_3, zerolinewidth=1)
            chart(fig)
            st.caption("Each crop averages its traits with equal weight. Negative R² means worse than "
                       "predicting the trait mean.")
        with st.expander("Per-trait results"):
            st.dataframe(
                benchmarks[["Crop", "target", "Model", "n_samples", "n_markers", "cv_r2", "cv_rmse", "cv_mae"]],
                hide_index=True,
                column_config={
                    "target": "Trait", "n_samples": "Samples", "n_markers": "Markers",
                    "cv_r2": st.column_config.NumberColumn("R²", format="%.3f"),
                    "cv_rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                    "cv_mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                },
            )
