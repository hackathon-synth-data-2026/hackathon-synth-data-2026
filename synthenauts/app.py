"""Upload page for the general synthesizer (anysynth.py).

Run from the repo root (Streamlit is fetched on the fly; the project lockfile is untouched):
    uv run --with streamlit --with openpyxl streamlit run synthenauts/app.py
"""
import hashlib
import html
import importlib
import io
import json
import os
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402
import statmatch  # noqa: E402
import anysynth as A  # noqa: E402

# Streamlit re-runs this page on every click but keeps imported modules from its start. Reload our
# generator modules when one of their files has changed, so an edited generator is always the one used.
_stamp = tuple(Path(m.__file__).stat().st_mtime for m in (synth, statmatch, A))
if getattr(A, "_loaded_stamp", None) != _stamp:
    for _m in (synth, statmatch, A):  # dependencies first
        importlib.reload(_m)
    A._loaded_stamp = _stamp

EXAMPLE = Path(__file__).resolve().parents[1] / "data/real_train.parquet"
EXAMPLE_CACHE = Path(__file__).resolve().parent / "example_cache"  # optional pre-computed results, used if the folder exists
LOCAL_CACHE = Path(__file__).resolve().parents[1] / "results/app_cache"  # results made on this machine
# Public demo (e.g. behind a tunnel for a QR code): never show rows of the real tables.
PUBLIC_DEMO = os.environ.get("SYNTH_PUBLIC_DEMO") == "1"
# The ICU example with default settings shows our hackathon submission and its official evaluator scores.
SUBMISSION = Path(__file__).resolve().parent / "submissions/synthenauts_submission.parquet"
SUBMISSION_BOARD = Path(__file__).resolve().parent / "submissions/leaderboard_v4.csv"
SUBMISSION_SEED = 12
# target, seed, rows, holdout, identifier columns, jitter columns, fast mode, competition mode
EXAMPLE_DEFAULTS = ("mortality", 0, 0, 0.2, (), (), False, False)

st.set_page_config(page_title="Synthetic data generator", page_icon="🧬", layout="wide")

# Categorical slots 1-2 of the reference palette (light / dark steps): real = blue-grey, synthetic = mint.
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
REAL_C, SYN_C = ("#87a7ba", "#91e4c1") if DARK else ("#68869a", "#91e4c1")
TEXT_C = "#efefef" if DARK else "#191919"
SURFACE_C = "#0e1117" if DARK else "#ffffff"
# Diverging pair for "synthetic minus real": blue = weaker in synthetic, red = stronger, grey = equal.
DIV_RANGE = ["#338ced", "#383838", "#eb6263"] if DARK else ["#1b64be", "#efefef", "#ec4647"]


@st.cache_data(show_spinner=False)
def load(data, name):
    return A.read_table(io.BytesIO(data), name)


@st.cache_data(show_spinner=False)
def example_hash():
    return hashlib.sha256(EXAMPLE.read_bytes()).hexdigest() if EXAMPLE.exists() else None


@st.cache_data(show_spinner=False)
def submission_result():
    """Our hackathon submission (v4, best of 30 seeds) with the scores the official evaluator gave it on the
    hidden test patients. Rules and column types are worked out from the real training table, and the rule
    checks are run on the submitted file itself."""
    syn = pd.read_parquet(SUBMISSION)
    board = pd.read_csv(SUBMISSION_BOARD)
    row = board[board.seed == SUBMISSION_SEED].iloc[0]
    real = pd.read_parquet(EXAMPLE)
    schema = A.detect_schema(real)
    enc = A.encode(real, schema)
    feats = [c for c in enc.columns if c != "mortality"]
    order, equal = A.find_rules(enc, [c for c in feats if schema[c]["kind"] == "numeric"])
    crules = A.find_category_rules(enc, schema, feats)
    s = A.encode(syn, schema)
    rep = dict(
        score_basis="official", target="mortality", task="binary", seed=SUBMISSION_SEED, temperature=0.3,
        rows_in=len(real), rows_out=len(syn), fit_rows=len(real), holdout_rows=0,
        overall=float(row.points), utility=float(row.utility), fidelity=float(row.fidelity),
        privacy=float(row.privacy),
        official=dict(ap_tstr=float(row.ap_tstr), ap_trtr=float(row.ap_trtr), near_copy=float(row.priv_near_copy),
                      membership=float(row.priv_membership), copy_rate=float(row.priv_copy_rate),
                      mean_points=float(board.points.mean()), seeds=int(len(board))),
        fidelity_parts=dict(marginals=float(row.fid_marginals), correlation=float(row.fid_correlation),
                            dependency=float(row.fid_dependency), missingness=float(row.fid_missingness)),
        columns={c: {k: v for k, v in i.items() if k != "levels"} for c, i in schema.items()},
        rules=[f"{a} <= {b}" for a, b in A.reduce_rules(order)] + [f"{a} == {b}" for a, b in equal],
        rule_violations_before_repair=None, rule_violations=A.rule_violations(s, order),
        category_rules=[A.describe_category_rule(schema, r) for r in crules],
        category_violations_before_repair=None, category_violations=A.category_violations(s, crules),
        swap_blocks=None, warnings=[], seconds=0.0)
    return syn, rep


def official_summary(rep):
    o, f = rep["official"], rep["fidelity_parts"]
    return "\n".join([
        "Our hackathon submission, scored by the official evaluator",
        f"overall   {rep['overall']:.2f} / 100   (about 94.4 is the most any method can reach)",
        f"utility   {rep['utility']:.3f}   average precision on the hidden test patients: "
        f"{o['ap_tstr']:.3f} trained on synthetic vs {o['ap_trtr']:.3f} trained on real",
        f"fidelity  {rep['fidelity']:.3f}   marginals {f['marginals']:.3f}, correlation {f['correlation']:.3f}, "
        f"dependency {f['dependency']:.3f}, missingness {f['missingness']:.3f}",
        f"privacy   {rep['privacy']:.3f}   1 - worst of: exact copies {o['copy_rate']:.3f}, near copies "
        f"{o['near_copy']:.3f}, membership inference {o['membership']:.3f}",
        f"rules     {len(rep['rules'])} order and {len(rep['category_rules'])} category rules; rows breaking one: "
        f"{rep['rule_violations']:.1%} and {rep['category_violations']:.1%}"])


def repair_line(before, after):
    return (f"Rows breaking one: {after:.1%} in this table" if before is None
            else f"Rows breaking one: {before:.1%} before repair, {after:.1%} after")


def to_bytes(df, kind):
    buf = io.BytesIO()
    if kind == "parquet":
        df.to_parquet(buf, index=False)
    else:
        df.to_csv(buf, index=False)
    return buf.getvalue()


def comparison(real, syn, col, kind):
    """Share of rows per bin (numbers, dates) or per value (categories), real vs synthetic."""
    r, s = real[col].dropna(), syn[col].dropna()
    if kind in ("numeric", "datetime"):
        conv = (lambda x: x.astype("int64").to_numpy(dtype=float)) if kind == "datetime" else \
            (lambda x: pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float))
        rv, sv = conv(pd.to_datetime(r) if kind == "datetime" else r), conv(pd.to_datetime(s) if kind == "datetime" else s)
        edges = np.histogram_bin_edges(np.concatenate([rv, sv]), bins=20)
        mids = (edges[:-1] + edges[1:]) / 2
        label = pd.to_datetime(mids).strftime("%Y-%m-%d") if kind == "datetime" else [f"{m:.4g}" for m in mids]
        rows = [{"bin": i, "value": label[i], "source": src, "share": h[i] / max(h.sum(), 1)}
                for src, h in (("real", np.histogram(rv, edges)[0]), ("synthetic", np.histogram(sv, edges)[0]))
                for i in range(len(mids))]
    else:
        rs, ss = r.astype(str).value_counts(normalize=True), s.astype(str).value_counts(normalize=True)
        cats = rs.add(ss, fill_value=0).sort_values(ascending=False).index[:30]
        rows = [{"bin": i, "value": c, "source": src, "share": float(v.get(c, 0.0))}
                for src, v in (("real", rs), ("synthetic", ss)) for i, c in enumerate(cats)]
    return pd.DataFrame(rows)


def chart(tab, kind):
    color = alt.Color("source:N", scale=alt.Scale(domain=["real", "synthetic"], range=[REAL_C, SYN_C]),
                      legend=alt.Legend(title=None, orient="top"))
    tip = [alt.Tooltip("value:N", title="value"), alt.Tooltip("source:N"),
           alt.Tooltip("share:Q", format=".1%")]
    x_sort = alt.EncodingSortField("bin")
    y = alt.Y("share:Q", axis=alt.Axis(format="%", title="share of rows", grid=True, gridOpacity=0.3))
    if kind in ("numeric", "datetime"):
        base = alt.Chart(tab).encode(x=alt.X("value:N", sort=x_sort, title=None, axis=alt.Axis(labelAngle=-45)),
                                     y=y, color=color, tooltip=tip)
        return base.mark_line(strokeWidth=2) + base.mark_point(size=70, filled=True, opacity=1)
    return alt.Chart(tab).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("value:N", sort=x_sort, title=None, axis=alt.Axis(labelAngle=-45)), xOffset="source:N",
        y=y, color=color, tooltip=tip)


def column_scores(real, syn, kinds):
    """Per-column similarity: 1 - KS for numbers and dates, 1 - total variation for categories."""
    from scipy.stats import ks_2samp
    out = []
    for c, k in kinds.items():
        if k == "id":
            continue
        r, s = real[c].dropna(), syn[c].dropna()
        if not len(r) or not len(s):
            continue
        if k in ("numeric", "datetime"):
            f = (lambda x: pd.to_datetime(x).astype("int64")) if k == "datetime" else (lambda x: pd.to_numeric(x, errors="coerce"))
            sim = 1 - ks_2samp(f(r).dropna(), f(s).dropna()).statistic
        else:
            rs, ss = r.astype(str).value_counts(normalize=True), s.astype(str).value_counts(normalize=True)
            sim = 1 - 0.5 * rs.sub(ss, fill_value=0).abs().sum()
        out.append({"column": c, "type": k, "similarity": sim,
                    "missing real": real[c].isna().mean(), "missing synthetic": syn[c].isna().mean()})
    return pd.DataFrame(out).sort_values("similarity")


SCALE_KINDS = ["ours", "best achievable"]


def score_scale(rep):
    """One row per score: ours next to the best score a synthetic table can honestly reach on that test,
    on a 0-1 scale (overall / 100). The best is: as useful as real data (utility 1), as close to the real
    rows as a fresh sample of real rows (fidelity), and no sign of copying (privacy 1)."""
    rows = []

    def add(metric, kind, value, shown):
        rows.append(dict(metric=metric, kind=kind, value=float(value), shown=shown))

    if "overall" in rep:
        add("Overall", "ours", rep["overall"] / 100, f"{rep['overall']:.1f}")
        add("Overall", "best achievable", rep["overall_real"] / 100, f"{rep['overall_real']:.1f}")
    if "utility" in rep:
        add("Utility", "ours", rep["utility"], f"{rep['utility']:.3f}")
        add("Utility", "best achievable", 1.0, "1.000 (as useful as real data)")
    if "fidelity_real" in rep:
        add("Fidelity", "ours", rep["fidelity"], f"{rep['fidelity']:.3f}")
        add("Fidelity", "best achievable", rep["fidelity_real"],
            f"{rep['fidelity_real']:.3f} (what a fresh sample of real rows scores)")
    if "privacy" in rep:
        add("Privacy", "ours", rep["privacy"], f"{rep['privacy']:.3f}")
        add("Privacy", "best achievable", 1.0, "1.000 (no sign of copying the training rows)")
    return pd.DataFrame(rows)


def score_scale_chart(d):
    order = [m for m in ("Overall", "Utility", "Fidelity", "Privacy") if m in set(d.metric)]
    y = alt.Y("metric:N", sort=order, title=None, axis=alt.Axis(domain=False, ticks=False, labelFontSize=13,
                                                                labelPadding=10))
    x = alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None,
              axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f", grid=False))
    color = alt.Color("kind:N", scale=alt.Scale(domain=SCALE_KINDS, range=[SYN_C, REAL_C]),
                      legend=alt.Legend(title=None, orient="top", symbolType="circle"))
    tip = [alt.Tooltip("metric:N"), alt.Tooltip("kind:N", title="table"), alt.Tooltip("shown:N", title="score")]
    track = alt.Chart(pd.DataFrame({"metric": order, "lo": 0.0, "hi": 1.0})).mark_rule(
        strokeWidth=8, opacity=0.12, color=TEXT_C, tooltip=False,  # background only: no hover box
    ).encode(y=y, x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 1])), x2="hi:Q")
    refs = alt.Chart(d).transform_filter(alt.datum.kind != "ours").mark_tick(thickness=4, size=40).encode(
        y=y, x=x, color=color, tooltip=tip)
    ours = alt.Chart(d).transform_filter(alt.datum.kind == "ours")
    dot = ours.mark_point(filled=True, size=170, opacity=1, stroke=SURFACE_C, strokeWidth=2).encode(
        y=y, x=x, color=color, tooltip=tip)
    label = ours.mark_text(dy=-31, fontWeight="bold", fontSize=13, color=TEXT_C).encode(
        y=y, x=x, text="shown:N", tooltip=tip)
    return (track + refs + dot + label).properties(height=76 * len(order))


def numeric_frame(f, cols, kinds):
    """Numbers, dates (as time stamps) and true/false (as 0/1) for rank correlations."""
    out = {}
    for c in cols:
        if kinds[c] == "datetime":
            v = pd.to_datetime(f[c], errors="coerce", format="mixed")
            if getattr(v.dt, "tz", None) is not None:
                v = v.dt.tz_localize(None)
            out[c] = (v - pd.Timestamp("1970-01-01")).dt.total_seconds()  # missing dates stay missing
        elif kinds[c] == "bool":
            out[c] = f[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        else:
            out[c] = pd.to_numeric(f[c], errors="coerce")
    return pd.DataFrame(out).astype(float)


def heatmap_chart(cr, cs, show):
    cells = [dict(a=a, b=b, real=cr.loc[a, b], synthetic=cs.loc[a, b], diff=cs.loc[a, b] - cr.loc[a, b])
             for a in show for b in show if a != b]
    d = pd.DataFrame(cells).dropna()
    return alt.Chart(d).mark_rect(stroke=SURFACE_C, strokeWidth=2).encode(
        x=alt.X("a:N", sort=show, title=None, axis=alt.Axis(labelAngle=-45, labelLimit=140)),
        y=alt.Y("b:N", sort=show, title=None, axis=alt.Axis(labelLimit=140)),
        color=alt.Color("diff:Q", title="synthetic − real",
                        scale=alt.Scale(domain=[-0.5, 0, 0.5], range=DIV_RANGE, clamp=True)),
        tooltip=[alt.Tooltip("a:N", title="column"), alt.Tooltip("b:N", title="with"),
                 alt.Tooltip("real:Q", format=".2f", title="real r"),
                 alt.Tooltip("synthetic:Q", format=".2f", title="synthetic r"),
                 alt.Tooltip("diff:Q", format="+.2f", title="difference")],
    ).properties(height=max(300, 22 * len(show)))


def scatter_charts(R, S, x, y, n=1500, seed=0):
    """Real and synthetic scatter as two separate charts on the same axes, so the page can place them
    side by side on a wide screen and one above the other on a phone."""
    both = pd.concat([R[[x, y]], S[[x, y]]]).dropna()
    xd = [float(both[x].min()), float(both[x].max())] if len(both) else None
    yd = [float(both[y].min()), float(both[y].max())] if len(both) else None
    parts = []
    for name, f, colr in (("Real", R, REAL_C), ("Synthetic", S, SYN_C)):
        pts = f[[x, y]].dropna()
        pts = pts.sample(min(n, len(pts)), random_state=seed) if len(pts) else pts
        r = f[[x, y]].corr("spearman").iloc[0, 1]
        parts.append(alt.Chart(pts).mark_circle(size=22, opacity=0.4, color=colr).encode(
            x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False, domain=xd, nice=True), title=x),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False, domain=yd, nice=True), title=y),
            tooltip=[alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q")],
        ).properties(title=f"{name} (rank correlation {r:.2f})", height=320))
    return parts


# ---------------------------------------------------------------------------

MINT = "#91e4c1" if DARK else "#91e4c1"
STRIPE = "#20372f" if DARK else "#eaf8f1"


def data_table(frame, **kwargs):
    """Tint every second column, including tables inside expanders."""
    styled = frame.style.apply(
        lambda col: [f"background-color: {STRIPE}" if frame.columns.get_loc(col.name) % 2 else ""
                     for _ in col], axis=0)
    return st.dataframe(styled, **kwargs)


def design():
    st.html("""
    <style>
    .stApp {--font: 'Aptos', 'Segoe UI', sans-serif;
        font-family: 'Aptos', 'Segoe UI', sans-serif;}
    .stApp :is(h1, h2, h3, h4, h5, h6, p, label, input, textarea, select, button, summary),
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stDataFrame"],
    [data-testid="stPopoverBody"] {
        font-family: 'Aptos', 'Segoe UI', sans-serif;}
    .stApp {background: var(--background-color);}
    .block-container {max-width: 1180px; padding-top: 0 !important; padding-bottom: 4rem;}
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;}
    h1 {font-size: clamp(2.5rem, 5vw, 4.5rem) !important; letter-spacing: -.055em;
        line-height: 1.06 !important; font-weight: 650 !important;}
    h2, h3 {letter-spacing: -.025em;}
    .page-header {background: transparent; color: #ffffff; padding: 0; box-sizing: border-box;
        height: 100px; display: flex; align-items: center; justify-content: center;
        margin-bottom: 1.5rem; position: relative; isolation: isolate;}
    .page-header::before {content: ''; position: absolute; inset-block: 0; left: 50%;
        transform: translateX(-50%); width: 100vw; z-index: -1;
        background: MINT;
        pointer-events: none;}
    .page-header h1 {color: #ffffff !important; font-size: clamp(1.75rem, 4vw, 3rem) !important;
        font-family: 'Aptos', 'Segoe UI', sans-serif; font-style: normal;
        letter-spacing: -.035em; font-weight: 700 !important; line-height: 1.1 !important;
        margin: 0; padding: 0 .3rem; width: 100%;
        text-align: center; transform: none;
        background: linear-gradient(165deg, #ffffff 8%, #ffffffcc 32%, #ffffff55 45%,
            #ffffffee 48%, #ffffff99 54%, #ffffff44 72%, #ffffffdd 100%);
        background-clip: text; -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; -webkit-text-stroke: .6px #ffffffe0;
        text-shadow: 0 1px 1px #ffffff55, 0 3px 6px #287b6430;
        filter: drop-shadow(0 1px 0 #ffffff77);}
    .page-header .header-slash {display: inline-block; margin: 0 .3em;
        font-weight: 300; opacity: .65;}
    .page-header .header-section {font-weight: 900; letter-spacing: .025em;}
    [data-testid="stBaseButton-secondary"]:not(:disabled),
    [data-testid="stBaseButton-primary"]:not(:disabled),
    [data-testid="stBaseButton-primaryFormSubmit"]:not(:disabled) {
        background: MINT !important; background-image: none !important;
        border-color: MINT !important; color: ON_MINT !important;}
    .stApp button p, .stApp button [data-testid="stIconMaterial"] {color: inherit;}
    .st-key-back_to_create button {border: none; background: transparent;
        color: FILE_TEXT; padding-left: 0;}
    .st-key-explore_result button, .st-key-back_generation button,
    .st-key-patient_previous button, .st-key-patient_next button {
        border: none !important; background: transparent !important;
        box-shadow: none !important; padding-left: 0;}
    .st-key-score_notes [data-testid="stPopoverButton"] {
        border: none !important; background: transparent !important;
        color: FILE_TEXT; min-width: 2.75rem; min-height: 2.75rem;}
    .st-key-score_notes [data-testid="stPopoverButton"] > svg {display: none;}
    :is(.st-key-back_to_create, .st-key-explore_result, .st-key-back_generation,
        .st-key-patient_previous, .st-key-patient_next, .st-key-change_table,
        .st-key-score_notes) button:not(:disabled) {
        color: NAV_TEXT !important; background: transparent !important;
        border-color: transparent !important;}
    .st-key-patient_record {border-top: 4px solid MINT !important; border-radius: 18px;}
    .patient-badge {display: inline-block; color: #287b64; background: #91e4c133;
        padding: .35rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 650;}
    .patient-field {display: flex; justify-content: space-between; gap: 1rem;
        border-bottom: 1px solid #51aa8822; padding: .65rem 0; font-size: .95rem;}
    .patient-field span {opacity: .7;}
    .patient-field strong {text-align: right; overflow-wrap: anywhere;}
    .intro {font-size: 1.15rem; opacity: .7; max-width: 620px; line-height: 1.7; margin: 1.2rem 0 2rem;}
    .steps {display: flex; width: 100%; margin: 0 0 1rem; isolation: isolate;}
    .steps span {flex: 1; min-width: 0; display: flex; align-items: center;
        justify-content: center; min-height: 76px; font-size: clamp(.85rem, 1.5vw, 1.1rem);
        font-weight: 550; background: PANEL; color: inherit;
        border-radius: 14px;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 360 76' preserveAspectRatio='none'%3E%3Cpath d='M20 0H320Q329 0 334 8L354 30Q362 38 354 46L334 68Q329 76 320 76H20Q10 76 16 68L32 46Q40 38 32 30L16 8Q10 0 20 0Z'/%3E%3C/svg%3E") center / 100% 100% no-repeat;
        padding: .7rem 1.3rem .7rem 1.8rem; margin-left: -24px;}
    .steps span:first-child {margin-left: 0; padding-left: 1rem;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 360 76' preserveAspectRatio='none'%3E%3Cpath d='M14 0H320Q329 0 334 8L354 30Q362 38 354 46L334 68Q329 76 320 76H14Q0 76 0 62V14Q0 0 14 0Z'/%3E%3C/svg%3E");}
    .steps .active {background: MINT; color: ON_MINT;}
    .st-key-workflow button {flex: 1; min-width: 0; display: flex; align-items: center;
        justify-content: center; min-height: 76px; font-size: clamp(.85rem, 1.5vw, 1.1rem);
        font-weight: 550; background: PANEL; color: inherit;
        border-radius: 14px;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 360 76' preserveAspectRatio='none'%3E%3Cpath d='M20 0H320Q329 0 334 8L354 30Q362 38 354 46L334 68Q329 76 320 76H20Q10 76 16 68L32 46Q40 38 32 30L16 8Q10 0 20 0Z'/%3E%3C/svg%3E") center / 100% 100% no-repeat;
        padding: .7rem 1.3rem .7rem 1.8rem; margin-left: -18px; width: calc(100% + 18px);}
    .st-key-workflow [data-testid="stColumn"]:first-child button {margin-left: 0; width: 100%; padding-left: 1rem;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 360 76' preserveAspectRatio='none'%3E%3Cpath d='M14 0H320Q329 0 334 8L354 30Q362 38 354 46L334 68Q329 76 320 76H14Q0 76 0 62V14Q0 0 14 0Z'/%3E%3C/svg%3E");}
    .st-key-workflow {margin-bottom: 1rem;}
    .st-key-workflow button[data-testid], .st-key-workflow button[data-testid]:not(:disabled),
    .st-key-workflow button[data-testid]:disabled {
        border: none !important; background: PANEL !important; color: NAV_TEXT !important;
        opacity: 1 !important; box-shadow: none !important;}
    .st-key-workflow button p {font-size: 1.1rem; font-weight: 550;}
    .st-key-workflow .st-key-active_step button[data-testid] {
        background: MINT !important; color: ON_MINT !important;}
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
        background: MINT !important; color: ON_MINT !important; border-color: MINT !important;
        min-height: 5rem; border-radius: 14px; font-size: 1.25rem; font-weight: 650;}
    [data-testid="stBaseButton-primary"] p {font-size: 1.25rem;}
    [data-testid="stFileUploader"] {border: 2px dashed #51aa8888; border-radius: 22px; padding: 1.2rem;}
    [data-testid="stFileUploaderDropzone"] {background: transparent; display: flex !important;
        flex-direction: column !important; align-items: center !important;
        justify-content: center !important; text-align: center; min-height: 300px; gap: 1.5rem;}
    [data-testid="stFileUploaderDropzoneInstructions"] {display: flex;
        flex-direction: column; align-items: center; margin: 0 !important; text-align: center;}
    [data-testid="stFileUploaderDropzone"] > :is(span, div):has(> button) {
        width: 100% !important; max-width: none !important; min-width: 0;
        flex: 0 0 auto !important; box-sizing: border-box;
        align-self: stretch !important; display: flex !important; justify-content: center;}
    [data-testid="stFileUploaderDropzone"] button {
        background: MINT !important; color: ON_MINT !important; border: 0 !important;
        display: flex !important; align-self: center !important; justify-content: center;
        margin: 0 auto !important; min-height: 5rem; width: calc(100% - 1rem) !important; max-width: none !important;
        border-radius: 16px; padding: 1.2rem 2.5rem;}
    [data-testid="stFileUploaderDropzone"] button p {font-size: 1.35rem !important; font-weight: 650;}
    [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {
        font-size: 2rem !important; width: 2rem !important; height: 2rem !important;}
    /* "Limit 200MB per file • CSV, TSV, …" wraps inside the box instead of running out of it on phones */
    [data-testid="stFileUploaderDropzoneInstructions"] {max-width: 100%; min-width: 0; box-sizing: border-box;}
    [data-testid="stFileUploaderDropzoneInstructions"] * {white-space: normal !important;
        overflow-wrap: anywhere; max-width: 100%; min-width: 0;}
    @media (max-width: 640px) {
        /* phones: the four steps stay one compact row instead of stacking into a tall column */
        .st-key-workflow [data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: 0 !important;}
        .st-key-workflow [data-testid="stColumn"] {width: 25% !important; min-width: 0 !important;
            flex: 1 1 0 !important;}
        .st-key-workflow button {min-height: 56px; padding: .35rem .4rem .35rem 1rem;
            margin-left: -10px; width: calc(100% + 10px);}
        .st-key-workflow [data-testid="stColumn"]:first-child button {padding-left: .45rem;}
        .st-key-workflow button p {font-size: .74rem !important; line-height: 1.15;}
        [data-testid="stFileUploader"] {padding: .8rem;}
        [data-testid="stFileUploaderDropzone"] {min-height: 220px; gap: 1rem; padding: .5rem !important;}
        [data-testid="stFileUploaderDropzone"] button {padding: 1rem 1.2rem; min-height: 4rem;}
        [data-testid="stFileUploaderDropzone"] button p {font-size: 1.1rem !important;}}
    [data-testid="stMetric"] {background: PANEL; border: 1px solid #51aa8844;
        border-top: 4px solid MINT; border-radius: 16px; padding: 1.2rem; min-height: 132px;}
    [data-testid="stMetricValue"] {color: MINT;}
    .score-cards {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem;}
    .score-card {background: PANEL; border: 1px solid #51aa8844;
        border-top: 4px solid MINT; border-radius: 16px; padding: 1.2rem; min-width: 0;}
    .score-card-text {font: 600 1.4rem/1.3 'Aptos', 'Segoe UI', sans-serif;
        margin: 0; padding: 0; font-variant-numeric: lining-nums;}
    .score-card-value {color: MINT; margin-top: .3rem;}
    /* info tooltips on the score cards: hover (desktop) or tap (phones; cursor:pointer lets iOS react to taps).
       The tooltip is placed against the whole card row, so it opens below it at full width and never runs
       off a phone screen. */
    .score-cards {position: relative;}
    .score-card-head {display: flex; align-items: center; justify-content: space-between; gap: .5rem;}
    .info {display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
        width: 1.6rem; height: 1.6rem; border-radius: 50%; border: 1.5px solid currentColor;
        font: italic 700 .9rem/1 Georgia, serif; opacity: .5; cursor: pointer; user-select: none;}
    .info:hover, .info:focus {opacity: 1; outline: none;}
    .info .tip {display: none; position: absolute; left: 0; right: 0; top: calc(100% + .6rem); z-index: 30;
        background: TIP_BG; color: TIP_TEXT; font: 400 .95rem/1.5 'Aptos', 'Segoe UI', sans-serif;
        font-style: normal; text-align: left; padding: .85rem 1rem; border-radius: 14px;
        box-shadow: 0 10px 28px #0000002e;}
    .info .tip b {font-weight: 700;}
    .info:hover .tip, .info:focus .tip {display: block;}
    @media (max-width: 700px) {.score-cards {grid-template-columns: repeat(2, minmax(0, 1fr));}}
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        font-size: 1.75rem !important; font-weight: 600 !important; line-height: 1.3;}
    [data-testid="stProgressBar"] [role="progressbar"] > div {background-color: MINT;}
    [data-testid="stExpander"] {border-radius: 14px;}
    .st-key-file_summary .file-summary-text {font-size: 1.2rem; color: FILE_TEXT !important; opacity: 1;}
    .st-key-change_table button {min-width: 2.75rem; min-height: 2.75rem;
        border: none !important; border-radius: 10px;
        box-shadow: none !important; background: transparent !important;
        color: NAV_TEXT !important;}
    .st-key-change_table button p {font-size: .875rem; color: inherit;}
    .st-key-change_table button [data-testid="stIconMaterial"] {
        font-size: 1.2rem !important; width: 1.2rem !important; height: 1.2rem !important;
        min-width: 1.2rem; flex-shrink: 0; display: inline-block; color: transparent !important;
        background-color: NAV_TEXT !important; opacity: 1 !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 8h16m-4-4 4 4-4 4M20 16H4m4-4-4 4 4 4'/%3E%3C/svg%3E") center / contain no-repeat;}
    .loading-scene {position: fixed; inset: 0; z-index: 2147483647;
        background: LOADING_SURFACE; color: LOADING_TEXT; opacity: 1;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 2rem; overflow-y: auto;}
    .loading-progress {width: min(480px, 85vw); margin-top: 1.5rem;}
    .loading-track {height: 8px; background: #51aa8833; border-radius: 999px; overflow: hidden;}
    .loading-fill {height: 100%; background: MINT; transition: width .3s ease;}
    .loading-label {margin-top: 1rem; font-size: .9rem;}
    body:has(.loading-scene) {overflow: hidden;}
    .orbit {position: relative; width: 130px; height: 130px; margin: 1rem auto 2rem;
        border: 2px solid #51aa8844; border-radius: 50%;}
    .orbit:before {content: ''; position: absolute; inset: 12px; border: 7px solid #51aa8833;
        border-top-color: MINT; border-right-color: MINT; border-radius: 50%; animation: spin 1.8s linear infinite;}
    .orbit:after {content: '✦'; position: absolute; inset: 0; display: grid; place-items: center;
        color: MINT; font-size: 3rem; animation: breathe 2s ease-in-out infinite;}
    @keyframes spin {to {transform: rotate(360deg);}}
    @keyframes breathe {50% {transform: scale(.8); opacity: .5;}}
    @media (prefers-reduced-motion: reduce) {.orbit:before, .orbit:after {animation: none;}}
    </style>
    """.replace("ON_MINT", "#123c2d" if DARK else "#123c2d")
       .replace("MINT", MINT).replace("PANEL", "#1c3028" if DARK else "#f0faf5")
       .replace("LOADING_SURFACE", SURFACE_C).replace("LOADING_TEXT", TEXT_C)
       .replace("FILE_TEXT", "#b5b9bf" if DARK else "#70757d").replace("NAV_TEXT", TEXT_C)
       .replace("PAGE_SURFACE", SURFACE_C)
       .replace("TIP_BG", "#f0faf5" if DARK else "#14213D").replace("TIP_TEXT", "#14213D" if DARK else "#ffffff"))


def metric_triangle(rep):
    """Three independent metric axes, from 0% at the centre to 100% at each vertex."""
    center = np.array([320.0, 298.0])
    vertices = np.array([[320.0, 90.0], [139.87, 402.0], [500.13, 402.0]])
    metrics = [("Utility", "utility"), ("Fidelity", "fidelity"), ("Privacy", "privacy")]
    values = [rep.get(key) for _, key in metrics]
    values = [float(v) if v is not None and np.isfinite(v) else None for v in values]
    accent = "#91e4c1" if DARK else "#37866e"

    def points(coords):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)

    labels = [(320, 38), (140, 444), (500, 444)]
    summary = "; ".join(f"{label}: {value:.1%}" if value is not None else f"{label}: not scored"
                        for (label, _), value in zip(metrics, values))
    parts = [f'<svg width="640" height="500" viewBox="0 0 640 500" role="img" aria-label="{html.escape(summary)}" '
             'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;'
             f'margin:0 auto;font-family:Aptos,Segoe UI,sans-serif;color:{TEXT_C}">',
             f'<title>{html.escape(summary)}</title>']
    for level in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<polygon points="{points(center + (vertices - center) * level)}" '
                     f'fill="none" stroke="{accent}" stroke-opacity=".25" stroke-width="1"/>')
    for x, y in vertices:
        parts.append(f'<line x1="320" y1="298" x2="{x}" y2="{y}" '
                     f'stroke="{accent}" stroke-opacity=".25"/>')
    available = [(i, value) for i, value in enumerate(values) if value is not None]
    coords = [(center + (vertices[i] - center) * np.clip(value, 0, 1)) for i, value in available]
    if len(coords) == 3:
        parts.append(f'<polygon points="{points(coords)}" fill="{MINT}" fill-opacity=".4" '
                     f'stroke="{accent}" stroke-width="3" stroke-linejoin="round"/>')
    for x, y in coords:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{accent}"/>')
    for ((label, _), value), (x, y) in zip(zip(metrics, values), labels):
        shown = f"{value:.1%}" if value is not None else "Not scored"
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" fill="currentColor" '
                     f'font-size="14" font-weight="600">{label}</text>'
                     f'<text x="{x}" y="{y + 29}" text-anchor="middle" fill="{accent}" '
                     f'font-size="14" font-weight="600">{shown}</text>')
    parts.append('</svg>')
    return "".join(parts)


def patient_record(syn):
    """A German-language record view of one synthetic row; no invented clinical fields."""
    if syn.empty:
        return
    index = min(max(int(st.session_state.get("patient_index", 0)), 0), len(syn) - 1)
    st.session_state["patient_index"] = index

    def change_patient(offset):
        st.session_state["patient_index"] = min(max(st.session_state["patient_index"] + offset, 0), len(syn) - 1)

    aliases = {
        "age": "Alter", "gender": "Geschlecht", "sex": "Geschlecht", "height": "Körpergröße",
        "weight": "Körpergewicht", "recordid": "Datensatz-ID", "bmi": "BMI",
        "hr": "Herzfrequenz", "resprate": "Atemfrequenz", "temp": "Körpertemperatur",
        "temperature": "Körpertemperatur", "spo2": "Sauerstoffsättigung", "sao2": "Sauerstoffsättigung",
        "sysabp": "Systolischer Blutdruck", "diasabp": "Diastolischer Blutdruck",
        "nisysabp": "Systolischer Blutdruck (nichtinvasiv)", "nidiasabp": "Diastolischer Blutdruck (nichtinvasiv)",
        "map": "Mittlerer arterieller Druck", "nimap": "Mittlerer arterieller Druck (nichtinvasiv)",
        "gcs": "Glasgow Coma Scale", "glucose": "Glukose", "creatinine": "Kreatinin",
        "hct": "Hämatokrit", "wbc": "Leukozyten", "platelets": "Thrombozyten",
        "na": "Natrium", "k": "Kalium", "bilirubin": "Bilirubin", "albumin": "Albumin",
        "bun": "Harnstoff-Stickstoff", "lactate": "Laktat", "ph": "pH-Wert",
        "paco2": "Kohlendioxidpartialdruck", "pao2": "Sauerstoffpartialdruck",
        "hco3": "Bikarbonat", "alt": "ALT", "ast": "AST", "alp": "Alkalische Phosphatase",
        "cholesterol": "Cholesterin", "mg": "Magnesium", "mortality": "Mortalität (Datensatzwert)",
    }
    suffixes = {"first": "erster Wert", "last": "letzter Wert", "lowest": "Minimum",
                "highest": "Maximum", "median": "Median"}
    groups = {"Stammdaten": [], "Vitalparameter": [], "Laborwerte": [], "Weitere Angaben": []}
    demographics = {"age", "gender", "sex", "height", "weight", "recordid", "bmi"}
    vitals = {"hr", "resprate", "temp", "temperature", "spo2", "sao2", "sysabp", "diasabp",
              "nisysabp", "nidiasabp", "map", "nimap", "gcs"}
    for column in syn.columns:
        base, _, suffix = str(column).lower().partition("_")
        group = "Stammdaten" if base in demographics else "Vitalparameter" if base in vitals else \
                "Laborwerte" if base in aliases and base != "mortality" else "Weitere Angaben"
        label = aliases.get(base, str(column))
        if base in aliases and suffix:
            label += f" · {suffixes.get(suffix, suffix)}"
        value = syn.iloc[index][column]
        if pd.api.types.is_scalar(value) and pd.isna(value):
            continue
        elif base in ("gender", "sex") and isinstance(value, (bool, np.bool_)):
            shown = "Männlich" if value else "Weiblich"
        elif base in ("gender", "sex") and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            shown = "Männlich" if value.strip().lower() == "true" else "Weiblich"
        elif isinstance(value, (float, np.floating)):
            shown = f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".")
        else:
            shown = str(value)
        groups[group].append((label, shown, str(column), suffix))

    def fields(items):
        st.html(''.join(f'<div class="patient-field" title="{html.escape(original, quote=True)}">'
                        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
                        for label, value, original, _ in items))

    with st.container(key="patient_record", border=True):
        st.html('<span class="patient-badge">SYNTHETISCHER DATENSATZ</span>')
        st.subheader(f"Patientenakte · SYN-{index + 1:05d}")
        st.caption("Patientenübersicht · Angaben aus den generierten Daten · Einheiten wie in der Quelldatei")
        main, clinical = st.columns(2, gap="large")
        with main:
            st.markdown("**Stammdaten**")
            if groups["Stammdaten"]:
                fields(groups["Stammdaten"][:6])
            else:
                st.caption("Keine Stammdaten im Datensatz vorhanden.")
        with clinical:
            for group in ("Vitalparameter", "Laborwerte"):
                selected = [item for item in groups[group] if item[3] in ("", "first")][:4]
                if selected:
                    st.markdown(f"**{group}**")
                    fields(selected)
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.button("Previous patient", icon=":material/arrow_back:", disabled=index == 0,
                      on_click=change_patient, args=(-1,), key="patient_previous")
            st.caption(f"Patient {index + 1} of {len(syn):,}", width="content")
            st.button("Next patient", icon=":material/arrow_forward:", disabled=index == len(syn) - 1,
                      on_click=change_patient, args=(1,), key="patient_next")


def metric_tips(rep):
    """What each score means, with this run's numbers (shown in the score cards' info tooltips)."""
    if rep.get("score_basis") == "official":
        o, f = rep["official"], rep["fidelity_parts"]
        return {
            "Overall": ("Official hackathon score of our submission: utility, fidelity and privacy averaged and "
                        "scaled to 100 by the organisers' evaluator. About 94.4 is the most any method can reach."),
            "Utility": (f"A model trained on this synthetic table reaches {o['ap_tstr'] / o['ap_trtr']:.0%} of the "
                        f"average precision of the same model trained on the real training data, both tested on "
                        f"hidden real test patients the generator never saw."),
            "Fidelity": (f"How closely the synthetic table matches the real training data: distributions "
                         f"{f['marginals']:.3f}, correlations {f['correlation']:.3f}, dependencies "
                         f"{f['dependency']:.3f}, missing values {f['missingness']:.3f}."),
            "Privacy": (f"Official privacy test against hidden real test patients: 1 minus the worst risk found "
                        f"(exact copies {o['copy_rate']:.0%}, near copies {o['near_copy']:.0%}, membership "
                        f"inference {o['membership']:.0%}). About 0.83 is the most any method can reach, because "
                        f"some real patients have too few demographics recorded."),
        }
    tips = {}
    if "overall" in rep:
        tips["Overall"] = (f"Mean of utility, fidelity and privacy. Compare it with the best achievable value, "
                           f"{rep['overall_real']:.1f}, not with 100: a table as useful as real data, as close to "
                           f"the real rows as a fresh sample of real rows, and with no sign of copying.")
    if "utility" in rep:
        lo, hi = rep["utility_interval"]
        tips["Utility"] = (f"A model trained on this synthetic data performs at {rep['utility_ratio']:.0%} of one "
                           f"trained on real data (could be {lo:.0%} to {hi:.0%}), both tested on real rows the "
                           f"generator never saw. 1.000 means exactly as useful as real data; performing above 100% "
                           f"counts against the score too, because the outcome would look easier to predict than it is.")
    elif "fidelity_real" in rep:
        tips["Utility"] = "Utility needs a main outcome that the real data itself can predict."
    if "fidelity" in rep:
        if rep.get("fidelity_basis") == "unseen rows":
            tips["Fidelity"] = (f"How closely distributions, correlations, dependencies and missing values match real "
                                f"rows the generator never saw. Best achievable: {rep['fidelity_real']:.3f}, what a "
                                f"fresh sample of real rows scores. Measured on the training rows it would read "
                                f"{rep['fidelity_train']:.3f}, which flatters it.")
        else:
            tips["Fidelity"] = ("How closely distributions, correlations, dependencies and missing values match the "
                                "real rows. Measured on the rows the generator learned from, which flatters it.")
    if "privacy" in rep:
        groups = ", ".join(f"{n}: {g['closer']:.0%}" for n, g in rep.get("privacy_groups", {}).items())
        tips["Privacy"] = (f"Share of synthetic rows that sit closer to a training row than to a real row the generator "
                           f"never saw ({groups}). 50% means no sign of copying and scores 1.000. It means no sign of "
                           f"copying, not that the data is anonymous against every attack. Exact copies of training "
                           f"rows: {rep['exact_copies']:.1%}.")
    return tips


def result_page_content(name, syn, rep):
    st.download_button("Download CSV ↓", to_bytes(syn, "csv"),
                       f"{Path(name).stem}_synthetic.csv", "text/csv", type="primary", width="stretch")
    st.divider()
    st.subheader("Your data, evaluated.")
    official = rep.get("score_basis") == "official"  # details are in the ⓘ tooltips and the Report tab
    st.caption(f"{len(syn):,} synthetic rows are ready" + (" · official hackathon score" if official else "") + ".")
    metrics = [("Overall", f"{rep['overall']:.1f} / 100" if "overall" in rep else "Not scored")]
    metrics += [(label, f"{rep[key]:.3f}" if key in rep else "Not scored")
                for label, key in (("Utility", "utility"), ("Fidelity", "fidelity"), ("Privacy", "privacy"))]
    tips = metric_tips(rep)

    def info(label):  # ⓘ with a tooltip: hover on desktop, tap on phones; opens below the card row
        text = tips.get(label)
        if not text:
            return ""
        return (f'<span class="info" tabindex="0" role="button" aria-label="About {label}">i'
                f'<span class="tip" role="tooltip"><b>{label}</b><br>{html.escape(text)}</span></span>')

    st.html('<div class="score-cards">' + ''.join(
        f'<div class="score-card"><div class="score-card-head"><div class="score-card-text">{label}</div>'
        f'{info(label)}</div>'
        f'<div class="score-card-text score-card-value">{value}</div></div>'
        for label, value in metrics) + '</div>')
    st.image(metric_triangle(rep), width="stretch")
    warnings = rep.get("warnings", [])
    has_fidelity_note = "fidelity_real" in rep and rep["fidelity_real"] - rep.get("fidelity_floor", 0) < 0.1
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="score_notes"):
        st.caption("Centre: 0% · Each vertex: 100% · Utility, fidelity and privacy scores.", width="stretch")
        if warnings or has_fidelity_note:
            with st.popover("", icon=":material/warning:", help="Warnings & notes", type="tertiary", key="score_warnings"):
                st.markdown("**Warnings & notes**")
                for w in warnings:
                    st.warning(w, icon="⚠️")
                if has_fidelity_note:
                    st.info("Most of the information in this table sits in the individual columns rather than in the links "
                            "between them, so fidelity here mainly checks the distribution of each column.", icon="ℹ️")
        with st.popover("", icon=":material/help_outline:", help="How to read these scores", type="tertiary", key="score_help"):
            st.markdown("**How to read these scores**")
            st.markdown(
                "- **Compare with the *best achievable* values, not with 100.** They are the most a synthetic table can "
                "honestly score on these tests: as useful as real data, as close to the real rows as a fresh sample of "
                "real rows, and no sign of copying.\n"
                "- **Utility 1.000** means a model trained on the synthetic rows performs exactly as well (100%) as one "
                "trained on the real rows. Performing above 100% is not better: it means the outcome looks easier to "
                "predict than it is, so it lowers the score just like performing below 100%.\n"
                "- **Fidelity on unseen rows** is the honest number. Fidelity on the training rows (the hackathon's "
                "definition) rewards copying.\n"
                "- **Privacy 1.000 means no sign of copying**, not that the data is anonymous. The nearest-row test checks "
                "whether synthetic rows copy or sit unusually close to the real rows the generator learned from. It does "
                "not cover every possible attack. (Real records themselves offer no privacy at all, which is why the "
                "reference is called *best achievable*, not *real data*.)")
    patient_record(syn)
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="large"):
        if st.button("Back to generation", icon=":material/arrow_back:", key="back_generation"):
            st.session_state["editing_generation"] = True
            st.rerun()
        if st.button("Explore data, charts & report", icon=":material/arrow_forward:", key="explore_result"):
            st.switch_page(explore_page_ref)

def page_header():
    st.html('<header class="page-header"><h1>Synthenauts</h1></header>')


def workflow_steps(active):
    """The steps are also the navigation: every step you can reach is clickable, from any page.
    Upload starts again with a new table, Generate opens the settings for the current table,
    Download shows the results, Explore opens the charts. The current step stays highlighted."""
    has_source = "source" in st.session_state
    has_result = "result" in st.session_state
    reachable = {1: has_source or has_result, 2: has_source, 3: has_result, 4: has_result}
    with st.container(key="workflow"):
        columns = st.columns(4, gap=None)
        for i, (column, label) in enumerate(zip(columns, ("01 / Upload", "02 / Generate", "03 / Download", "04 / Explore")), 1):
            with column, st.container(key="active_step" if i == active else f"workflow_step_{i}"):
                if st.button(label, key=f"step_{i}", disabled=i == active or not reachable[i], width="stretch"):
                    if i == 1:  # same as "Change table"
                        for field in ("source", "result", "result_context", "editing_generation"):
                            st.session_state.pop(field, None)
                        st.switch_page(create_page_ref)
                    elif i == 2:
                        st.session_state["editing_generation"] = True
                        st.switch_page(create_page_ref)
                    elif i == 3:
                        st.session_state["editing_generation"] = False
                        st.switch_page(create_page_ref)
                    else:
                        st.switch_page(explore_page_ref)


def create_page():
    page_header()
    # A chart tooltip shown on the Explore page can be left on screen after switching pages; this page has
    # no such charts, so hide any leftover one here.
    st.html("<style>#vg-tooltip-element, .vg-tooltip {display: none !important;}</style>")
    ready = "result" in st.session_state and not st.session_state.get("editing_generation", False)
    active = 3 if ready else (2 if st.session_state.get("source") else 1)
    with st.container():
        workflow_steps(active)
        if "source" not in st.session_state:
            up = st.file_uploader("Upload your table", label_visibility="collapsed",
                                  type=["csv", "tsv", "txt", "parquet", "xlsx", "xls", "json", "jsonl"])
            st.caption("Drag & drop your file above, or browse files. CSV, Excel, Parquet, TSV or JSON.")
            if up is not None:
                try:
                    frame = load(up.getvalue(), up.name)
                    if frame.empty or not len(frame.columns):
                        raise ValueError("Please upload a table with at least one row and one column.")
                    st.session_state["source"] = (up.name, up.getvalue())
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not read this table: {exc}")
            if EXAMPLE.exists() and st.button("Try the ICU example ↗", width="stretch"):
                st.session_state["source"] = (EXAMPLE.name, EXAMPLE.read_bytes())
                st.rerun()
            st.caption("Public demo: files are processed on our presenting laptop for this session; only the "
                       "synthetic result is kept, to make repeats instant." if PUBLIC_DEMO
                       else "Your data stays on this machine.")
            return

        name, data = st.session_state["source"]
        df = load(data, name)
        data_hash = hashlib.sha256(data).hexdigest()
        current = st.session_state.get("result")
        if current is not None and not st.session_state.get("editing_generation", False):
            _, syn, rep = current
            result_page_content(name, syn, rep)
            return
        previous = current[0] if current is not None and current[0][0] == data_hash else None
        action = st.empty()
        choices = ["(none)"] + list(df.columns)
        suggested = next((c for c in df.columns if str(c).lower() in
                          ("mortality", "target", "outcome", "label", "class", "stroke")), None)
        if suggested is None:
            candidates = [c for c in df.columns if 2 <= df[c].nunique() <= 20
                          and not str(c).lower().endswith("id")]
            suggested = candidates[-1] if candidates else None
        if previous is not None:
            suggested = previous[1]
        target_choice = st.selectbox("Main outcome", choices,
                                    index=choices.index(suggested) if suggested is not None else 0,
                                    key=f"target_{data_hash}",
                                    help="The result this data will be used to study, for example mortality. We keep its "
                                         "link to the other columns realistic, and test it: a model trained on the "
                                         "synthetic data must predict it on real rows. Nothing is predicted about real "
                                         "patients. Choose (none) if your table has no outcome.")
        target = None if target_choice == "(none)" else target_choice
        with st.expander("Advanced settings · ready to use defaults", expanded=False):
            seed = st.number_input("Seed", 0, 10_000, previous[2] if previous else 0)
            rows = st.number_input("Synthetic rows (0 = same as fitted rows)", 0, 10_000_000, previous[3] if previous else 0)
            holdout = st.slider("Held out for the report", 0.0, 0.5, previous[4] if previous else 0.2, 0.05,
                                help="Unseen rows for evaluation. Set to 0 for fidelity only.")
            ids = st.multiselect("Identifier columns (auto-detected otherwise)", list(df.columns), default=list(previous[5]) if previous else [])
            jitter = st.multiselect("Quasi-identifiers to jitter",
                                    [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], default=list(previous[6]) if previous else [])
            fast = st.toggle("Fast mode", value=previous[7] if previous else False, help="Fewer passes: quicker, slightly lower fidelity.")
            competition = st.toggle("Competition mode", value=previous[8] if previous else False, disabled=target is None,
                                    help="Redraws the outcome with less noise; may overstate model performance.")
            refresh = st.toggle("Ignore saved results")
        key = (data_hash, target, seed, rows, holdout, tuple(ids), tuple(jitter), fast, competition)
        go = action.button("Generate synthetic data ↗", type="primary", width="stretch")

        with st.container(key="file_summary", horizontal=True, horizontal_alignment="distribute",
                          vertical_alignment="center", gap="small"):
            st.markdown(f'<span class="file-summary-text">{html.escape(name)} · {len(df):,} rows · '
                        f'{len(df.columns)} columns</span>', unsafe_allow_html=True, width="content")
            if st.button("Change table", icon=":material/swap_horiz:",
                         help="Change table", key="change_table"):
                for field in ("source", "result", "result_context", "editing_generation"):
                    st.session_state.pop(field, None)
                st.rerun()

    if go:
        animation = action
        started = time.time()

        def on_progress(frac, label):
            frac = min(max(frac, 0.0), 1.0)
            spent = time.time() - started
            left = f" · about {spent / frac * (1 - frac):.0f}s left" if 0.05 < frac < 1 else ""
            animation.html(
                '<div class="loading-scene" role="status" aria-live="polite" aria-busy="true">'
                '<div class="orbit" aria-hidden="true"></div>'
                '<h2>Generating your synthetic data.</h2>'
                '<p>Learning your patterns. Creating fresh rows.</p>'
                '<div class="loading-progress">'
                f'<div class="loading-track" role="progressbar" aria-label="Generation progress" '
                f'aria-valuenow="{frac * 100:.0f}" aria-valuemin="0" aria-valuemax="100">'
                f'<div class="loading-fill" style="width:{frac * 100:.2f}%"></div></div>'
                f'<div class="loading-label">{html.escape(str(label))} · {frac:.0%}{left}</div>'
                '</div></div>')

        on_progress(0.0, "Starting …")

        settings = (target, int(seed), int(rows), float(holdout), tuple(ids), tuple(jitter), bool(fast), bool(competition))
        try:
            if (data_hash == example_hash() and settings == EXAMPLE_DEFAULTS and not refresh
                    and SUBMISSION.exists() and SUBMISSION_BOARD.exists()):
                syn, rep = submission_result()  # the ICU example as-is: our submission, officially scored
            else:
                syn, rep, saved = A.cached_synthesize(
                    df, data_hash, target=target, seed=seed, rows=rows or None, holdout=holdout, id_cols=ids,
                    jitter_cols=jitter, corr_iters=3 if fast else 10, stat_iters=300_000 if fast else None,
                    temperature=A.COMPETITION_TEMPERATURE if competition and target else 1.0,
                    cache_dirs=[EXAMPLE_CACHE, LOCAL_CACHE], write_dir=LOCAL_CACHE, refresh=refresh,
                    log=lambda *_: None, progress=on_progress)
        except Exception as exc:
            animation.empty()
            st.error(f"Generation failed: {exc}")
            return
        st.session_state["result"] = (key, syn, rep)
        st.session_state["result_context"] = (name, df, target)
        st.session_state["editing_generation"] = False
        st.session_state["patient_index"] = 0
        st.rerun()

def details_page():
    page_header()
    workflow_steps(4)
    if st.button("Back to results" if "result" in st.session_state else "Back to Create",
                 icon=":material/arrow_back:", key="back_to_create"):
        st.session_state["editing_generation"] = False
        st.switch_page(create_page_ref)
    st.caption("Explore the data, compare distributions, and see which relationships survived.")
    if "result" not in st.session_state:
        st.info("Generate your synthetic table first, then come back to explore it.")
        st.page_link(create_page_ref, label="Create synthetic data →")
        return
    _, syn, rep = st.session_state["result"]
    name, df, target = st.session_state["result_context"]
    if not PUBLIC_DEMO:  # the public demo never shows rows of the real table
        with st.expander("Original table"):
            data_table(df.head(50), width="stretch", height=220)
    st.download_button("Download Parquet", to_bytes(syn, "parquet"),
                       f"{Path(name).stem}_synthetic.parquet", width="content")
    kinds = {c: i["kind"] for c, i in rep["columns"].items()}
    t_data, t_cmp, t_rel, t_rules, t_types, t_raw = st.tabs(["Synthetic data", "Compare columns", "Relationships",
                                                              "Rules learned", "Column types", "Report"])
    with t_data:
        data_table(syn.head(500), width="stretch", height=360)
        st.caption(f"{len(syn):,} synthetic rows; showing the first 500.")

    with t_cmp:
        scores = column_scores(df, syn, kinds)
        left, right = st.columns([3, 2])
        with right:
            st.markdown("**Per-column similarity** (worst first)")
            data_table(scores, hide_index=True, width="stretch", height=420, column_config={
                "similarity": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                "missing real": st.column_config.NumberColumn(format="percent"),
                "missing synthetic": st.column_config.NumberColumn(format="percent")})
        with left:
            col = st.selectbox("Column", scores["column"].tolist() or [c for c in kinds if kinds[c] != "id"])
            tab = comparison(df, syn, col, kinds[col])
            st.altair_chart(chart(tab, kinds[col]).properties(height=360), width="stretch", theme="streamlit")
            with st.expander("As a table"):
                data_table(tab.pivot(index=["bin", "value"], columns="source", values="share").reset_index(level=0, drop=True),
                             width="stretch", column_config={
                                 "real": st.column_config.NumberColumn(format="percent"),
                                 "synthetic": st.column_config.NumberColumn(format="percent")})

    with t_rel:
        numc = [c for c in kinds if kinds[c] in ("numeric", "bool", "datetime")]
        if len(numc) < 2:
            st.info("Relationships need at least two number, date or true/false columns.")
        else:
            R, Sy = numeric_frame(df, numc, kinds), numeric_frame(syn, numc, kinds)
            cr, cs = R.corr("spearman"), Sy.corr("spearman")
            off = ~np.eye(len(numc), dtype=bool)
            gap = float(np.nanmean(np.abs((cs - cr).to_numpy())[off]))
            st.markdown("Synthetic data is only useful if columns stay linked the way they are in reality, for example "
                        "older patients having higher blood pressure. This tab checks those links: the heatmap for every "
                        "pair at once, the scatter for one pair in detail.")
            st.markdown(f"#### Heatmap: which links changed\n"
                        f"On average a link differs by **{gap:.3f}** between synthetic and real data, on a scale from "
                        f"−1 to 1 (0 = identical; below 0.05 is very close).")
            with st.expander("What is this heatmap and how do I use it?"):
                st.markdown(
                    "- **One square per pair of columns.** The colour compares how strongly the two columns move "
                    "together in the synthetic data with the real data.\n"
                    "- **Strength is the rank correlation**, from −1 to 1: how consistently one column goes up (1) or "
                    "down (−1) when the other goes up. 0 means no link.\n"
                    "- **Grey:** the same strength as in the real data. **Blue:** the link is weaker in the synthetic "
                    "data. **Red:** stronger. The deeper the colour, the bigger the difference (±0.5 or more is the "
                    "deepest).\n"
                    "- **How to use it:** pick the columns that matter for your analysis in the box below, look for "
                    "deeply coloured squares, and hover one for the exact values. The table under the heatmap lists the "
                    "biggest differences; check those pairs in the scatter plot further down.")
            strength = cr.abs().where(off).sum().sort_values(ascending=False)
            show = st.multiselect("Columns in the heatmap (default: the 12 most connected)", numc,
                                  default=strength.index[:12].tolist())
            if len(show) >= 2:
                st.altair_chart(heatmap_chart(cr, cs, show), width="stretch", theme="streamlit")
                st.caption("Grey: same strength as in the real data · blue: weaker in the synthetic data · red: "
                           "stronger. Hover a square for the exact values.")
                pairs = pd.DataFrame([dict(column=a, **{"with": b}, real=cr.loc[a, b], synthetic=cs.loc[a, b])
                                      for i, a in enumerate(show) for b in show[i + 1:]])
                pairs["difference"] = pairs.synthetic - pairs.real
                with st.expander("Largest differences as a table"):
                    data_table(pairs.reindex(pairs.difference.abs().sort_values(ascending=False).index).head(25),
                                 hide_index=True, width="stretch",
                                 column_config={k: st.column_config.NumberColumn(format="%.2f")
                                                for k in ("real", "synthetic", "difference")})
            st.divider()
            cont = [c for c in numc if R[c].nunique() > 10]
            if len(cont) >= 2:
                st.markdown("#### Scatter: one pair of columns in detail")
                with st.expander("What is this scatter plot and how do I use it?"):
                    st.markdown(
                        "- **Each dot is one row** (a patient). Left: real data. Right: synthetic data. Both use the "
                        "same axes, so the two clouds can be compared directly.\n"
                        "- **Similar clouds** mean the relationship between the two columns survived, including curved "
                        "patterns, clusters and outliers that a single correlation number in the heatmap cannot show.\n"
                        "- **How to use it:** it opens on the most strongly linked pair. Choose any two columns, for "
                        "example a pair the heatmap shows in blue or red. The title of each plot gives its rank "
                        "correlation.")
                sub = cr.loc[cont, cont].abs().where(~np.eye(len(cont), dtype=bool))
                a0, b0 = np.unravel_index(np.nanargmax(sub.to_numpy()), sub.shape)  # strongest real pair first
                c1, c2 = st.columns(2)
                x = c1.selectbox("Horizontal axis", cont, index=int(a0))
                y = c2.selectbox("Vertical axis", cont, index=int(b0))
                if x != y:
                    real_plot, syn_plot = scatter_charts(R, Sy, x, y)
                    if PUBLIC_DEMO:  # no real rows leave the laptop: synthetic points only, real data as one number
                        st.altair_chart(syn_plot, width="stretch", theme="streamlit")
                        st.caption(f"Public demo: real rows are not shown. In the real data these two columns have a "
                                   f"rank correlation of {R[[x, y]].corr('spearman').iloc[0, 1]:.2f}; the synthetic "
                                   f"points above should show the same trend.")
                    else:
                        for place, plot in zip(st.columns(2), (real_plot, syn_plot)):  # stack on phones
                            place.altair_chart(plot, width="stretch", theme="streamlit")
                        st.caption("Up to 1,500 random rows from each table. Matching shapes mean the relationship "
                                   "between the two columns survived.")

    with t_rules:
        st.markdown("Rules that hold in **every** real row and would rarely hold by chance are learned from the data "
                    "and enforced on the synthetic rows. A table with one value per concept (age, BMI, glucose) has "
                    "few order rules; category rules are more common there.")
        crules = rep.get("category_rules", [])
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Order rules (A ≤ B): {len(rep['rules'])}**  \n"
                        + repair_line(rep["rule_violations_before_repair"], rep["rule_violations"]))
            if rep["rules"]:
                data_table(pd.DataFrame({"rule": rep["rules"]}), hide_index=True, width="stretch", height=320)
            blocks = (f" Linked columns move as one block during statistic matching ({rep['swap_blocks']} blocks)."
                      if rep.get("swap_blocks") else "")
            st.caption("For example lowest ≤ highest, or admission ≤ discharge. Rules implied by two others are "
                       "not listed." + blocks)
        with right:
            st.markdown(f"**Category rules: {len(crules)}**  \n"
                        + repair_line(rep.get("category_violations_before_repair", 0), rep.get("category_violations", 0)))
            if crules:
                data_table(pd.DataFrame({"rule": crules}), hide_index=True, width="stretch", height=320)
            st.caption("For example *if work_type = children then ever_married = No*, or *... then age ≤ 16*. "
                       "A number outside the real range of its group is moved to the edge of that range.")

    with t_types:
        data_table(pd.DataFrame([{"column": c, "detected type": i["kind"], "stored as": i["dtype"],
                                    "missing": i["missing"], "rare values pooled": i.get("pooled", 0)}
                                   for c, i in rep["columns"].items()]),
                     hide_index=True, width="stretch",
                     column_config={"missing": st.column_config.NumberColumn(format="percent")})
        st.caption("Identifier columns get fresh values. Text columns with more than 50 distinct values have values "
                   f"seen fewer than {A.RARE} times pooled into '{A.OTHER}', so rare values cannot point to a person.")

    with t_raw:
        st.download_button("Download report (JSON)", json.dumps(rep, indent=2, default=str),
                           f"{Path(name).stem}_synthetic.report.json", "application/json")
        st.code(official_summary(rep) if rep.get("score_basis") == "official" else A.summary(rep), language=None)

design()
create_page_ref = st.Page(create_page, title="Create", icon=":material/auto_awesome:", default=True)
explore_page_ref = st.Page(details_page, title="Explore", icon=":material/analytics:")
st.navigation([create_page_ref, explore_page_ref], position="hidden").run()
