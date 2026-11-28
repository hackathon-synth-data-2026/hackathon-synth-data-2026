"""Synthesize any tabular dataset with our ICU pipeline, generalised.

Give it a table (CSV, TSV, Parquet, Excel or JSON) and, optionally, the column you want to
predict. It returns a synthetic table with the same columns and types, and a quality report.

What is learned from the data instead of written for the ICU table (see synth.py):
  - Column types: identifiers, numbers, dates, true/false, categories. Identifier and
    free-text columns are replaced with fresh tokens, never copied.
  - Consistency rules: every "column A <= column B" that holds in all real rows, and would
    often fail if the two columns were unrelated, becomes a rule. Rows that break a rule are
    fixed by swapping the two values. On the ICU table this finds lowest <= first <= highest
    and diastolic <= systolic by itself.
  - Swap blocks for statistic matching: columns linked by a rule or by an identical missing
    pattern move together, so the rules keep holding.
  - Category rules: "if work_type = children then ever_married = No" or "... then age <= 16",
    learned with the same chance test and enforced on the final rows.
  - Relabelling works for binary, multi-class and numeric targets. By default the target keeps
    the real amount of noise; --competition sharpens it (higher scores, less faithful).
  - Jitter on quasi-identifiers is opt-in (--jitter), since they depend on the data.

The report is built to be honest:
  - 20% of rows are held out. The generator never sees them; fidelity, utility and privacy are
    measured on them, so copying training rows cannot raise a score. --holdout 0 fits on every
    row, and the report then only has fidelity on the training rows (flagged as flattering).
  - Every score comes with what real data and shuffled columns (no relationships) get on the
    same test, because the scale does not start at 0: shuffled data scores around 65.
  - Utility is the uncapped ratio "model trained on synthetic / model trained on real", with a
    90% range. Above 1 means the outcome looks more predictable than it is; that is penalised.
  - Privacy takes the worse of two distance tests: all columns, and quasi-identifiers only.
  - Warnings explain anything that makes a number flattering or unreliable.

Usage (from the repo root):
    uv run python synthenauts/anysynth.py data.csv --target outcome --out synthetic.csv
    uv run python synthenauts/anysynth.py data.parquet --out synthetic.parquet   # no target
    uv run python synthenauts/anysynth.py data.csv --target y --holdout 0 --rows 10000 --seed 3
"""
import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth as S  # noqa: E402

warnings.filterwarnings("ignore")

ID_NAME = re.compile(r"(id|uuid|guid|key|ssn|mrn)$|^id[_\W]", re.I)
MAX_CATEGORIES = 50     # text columns with more distinct values get rare values pooled
RARE = 5                # categories seen fewer times are pooled into "(other)" (privacy)
OTHER = "(other)"


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------

def read_table(src, name=None):
    """src: path or file-like (e.g. an upload). name: file name used to pick the format."""
    name = str(name or src).lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(src)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(src)
    if name.endswith(".jsonl"):
        return pd.read_json(src, lines=True)
    if name.endswith(".json"):
        return pd.read_json(src)
    if name.endswith((".tsv", ".tab")):
        return pd.read_csv(src, sep="\t")
    if name.endswith(".csv"):
        return pd.read_csv(src)
    return pd.read_csv(src, sep=None, engine="python")  # sniff the delimiter


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", date_format="iso")
    elif ext == ".jsonl":
        df.to_json(path, orient="records", lines=True, date_format="iso")
    else:
        df.to_csv(path, index=False, sep="\t" if ext in (".tsv", ".tab") else ",")


# ---------------------------------------------------------------------------
# column types and encoding
# ---------------------------------------------------------------------------

def detect_schema(df, id_cols=()):
    """Return {column: info}; info['kind'] is id, numeric, datetime, bool or category."""
    schema, n = {}, len(df)
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        info = {"dtype": str(s.dtype), "missing": float(s.isna().mean())}
        if c in id_cols:
            info["kind"] = "id"
        elif pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            info["kind"] = "bool"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(kind="datetime", tz=str(s.dt.tz) if s.dt.tz is not None else None)
        elif pd.api.types.is_numeric_dtype(s):
            v = nn.to_numpy(dtype=float)
            integer = len(nn) > 0 and np.all(np.mod(v, 1) == 0)
            unique = integer and nn.is_unique and len(nn) == n and n > 20
            if unique and (ID_NAME.search(str(c)) or np.all(np.diff(np.sort(v)) == 1)):
                info["kind"] = "id"  # named like an identifier, or a running row number
            else:
                info["kind"] = "numeric"
                info["integer"] = bool(integer)
        else:
            text = nn.astype(str)
            as_num = pd.to_numeric(text, errors="coerce")
            numeric = len(nn) > 0 and as_num.notna().mean() > 0.98
            as_date = (pd.to_datetime(text, errors="coerce", format="mixed")
                       if not numeric and len(nn) and text.str.contains(r"\d[-/.:]\d", regex=True).mean() > 0.9
                       else None)
            if numeric:
                info.update(kind="numeric", integer=bool(np.all(np.mod(as_num.dropna(), 1) == 0)),
                            parse="number")
            elif as_date is not None and as_date.notna().mean() > 0.98:
                info.update(kind="datetime", tz=None, parse="date")
            elif len(nn) > 20 and nn.is_unique:
                info["kind"] = "id"  # every value unique: an identifier or free text, never copied
            else:
                counts = text.value_counts()
                info["kind"] = "category"
                info["pooled"] = int((counts < RARE).sum()) if len(counts) > MAX_CATEGORIES else 0
        schema[c] = info
    return schema


def encode(df, schema):
    """All modelled columns as floats (categories as integer codes, dates as nanoseconds)."""
    out = {}
    for c, info in schema.items():
        k, s = info["kind"], df[c]
        if k == "id":
            continue
        if k == "numeric":
            out[c] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        elif k == "bool":
            out[c] = s.map({True: 1.0, False: 0.0}).to_numpy(dtype=float, na_value=np.nan)
        elif k == "datetime":
            d = pd.to_datetime(s, errors="coerce", format="mixed") if info.get("parse") else s
            if getattr(d.dt, "tz", None) is not None:
                d = d.dt.tz_convert("UTC").dt.tz_localize(None)
            v = d.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
            v[d.isna().to_numpy()] = np.nan
            out[c] = v
        else:
            text = s.astype("string")
            if "levels" not in info:
                counts = text.value_counts()
                if info.get("pooled"):
                    counts = counts[counts >= RARE]
                info["levels"] = sorted(counts.index.tolist(), key=str) + ([OTHER] if info.get("pooled") else [])
            pos = {v: i for i, v in enumerate(info["levels"])}
            other = pos.get(OTHER, np.nan)
            out[c] = np.array([np.nan if pd.isna(v) else pos.get(v, other) for v in text], dtype=float)
    return pd.DataFrame(out, index=df.index)


def decode(syn, schema, real):
    """Back to the original columns, order and dtypes; identifiers are fresh."""
    out = {}
    n = len(syn)
    for c, info in schema.items():
        k = info["kind"]
        if k == "id":
            s = real[c]
            if pd.api.types.is_numeric_dtype(s):
                start = int(np.nanmax(s.to_numpy(dtype=float))) + 1 if s.notna().any() else 1
                out[c] = pd.Series(np.arange(start, start + n), dtype="int64")
            else:
                out[c] = pd.Series([f"SYN{i:07d}" for i in range(1, n + 1)], dtype=s.dtype)
            continue
        v = syn[c].to_numpy(dtype=float)
        miss = np.isnan(v)
        if k == "numeric":
            if info.get("integer"):
                v = np.round(v)
            if info.get("parse") == "number":
                col = pd.Series(v)
            elif info["integer"] and pd.api.types.is_integer_dtype(info["dtype"].lower()):
                col = pd.Series(v).astype("Int64" if miss.any() or info["dtype"][0] == "I" else info["dtype"])
            else:
                col = pd.Series(v).astype(info["dtype"] if "float" in info["dtype"].lower() else float)
        elif k == "bool":
            col = pd.Series(v).map({1.0: True, 0.0: False})
            col = col.astype("boolean" if miss.any() or info["dtype"] == "boolean" else bool)
        elif k == "datetime":
            col = pd.Series(pd.to_datetime(np.where(miss, 0, v).astype("int64")))
            col[miss] = pd.NaT
            if info.get("tz"):
                col = col.dt.tz_localize("UTC").dt.tz_convert(info["tz"])
        else:
            levels = np.array(info["levels"], dtype=object)
            col = pd.Series(np.where(miss, None, levels[np.clip(np.nan_to_num(v).astype(int), 0, len(levels) - 1)]))
            if info["dtype"] == "category":
                col = col.astype("category")
            elif info["dtype"] in ("string", "str"):
                col = col.astype("string")
        out[c] = col.reset_index(drop=True)
    return pd.DataFrame(out)[list(schema)]


# ---------------------------------------------------------------------------
# learned consistency rules
# ---------------------------------------------------------------------------

def find_rules(real, cols, min_support=30, min_chance=0.01, alpha=1e-3):
    """Pairs (a, b) with a <= b in every real row where both are present.

    A rule is kept only if it is unlikely to hold by accident: unrelated columns with the
    same values would break it in a share `chance` of row pairs, so seeing no break in `both`
    rows has probability (1 - chance) ** both, which must be below `alpha`. This drops
    accidents of scale ("age <= income") and needs more evidence on small tables.
    Pairs that are always equal are returned separately.
    """
    X = real[cols].to_numpy(dtype=float)
    obs = ~np.isnan(X)
    p = len(cols)
    sorted_obs = [np.sort(X[obs[:, j], j]) for j in range(p)]
    le = np.zeros((p, p), dtype=bool)
    with np.errstate(invalid="ignore"):
        for a in range(p):
            viol = (X[:, [a]] > X).sum(0)
            both = (obs[:, [a]] & obs).sum(0)
            for b in np.flatnonzero((viol == 0) & (both >= min_support)):
                if b == a:
                    continue
                sa, sb = sorted_obs[a], sorted_obs[b]
                chance = np.searchsorted(sb, sa, "left").sum() / (len(sa) * len(sb))
                le[a, b] = chance >= min_chance and both[b] * np.log1p(-min(chance, 1 - 1e-12)) <= np.log(alpha)
    eq = [(cols[a], cols[b]) for a in range(p) for b in range(a + 1, p) if le[a, b] and le[b, a]]
    rules = [(cols[a], cols[b]) for a in range(p) for b in range(p) if le[a, b] and not le[b, a]]
    return rules, eq


def reduce_rules(rules):
    """Drop rules implied by two others (a<=b, b<=c => a<=c), for display."""
    succ = {}
    for a, b in rules:
        succ.setdefault(a, set()).add(b)
    return [(a, b) for a, b in rules if not any(b in succ.get(m, ()) for m in succ.get(a, ()) if m != b)]


def repair_rules(syn, rules, eq, passes=100):
    """Fix broken rules by swapping the two values in that row (keeps the row's values)."""
    syn = syn.copy()
    for a, b in eq:
        both = syn[a].notna() & syn[b].notna()
        syn.loc[both, b] = syn.loc[both, a]
    if not rules:
        return syn
    cols = sorted({c for r in rules for c in r})
    pos = {c: i for i, c in enumerate(cols)}
    X = syn[cols].to_numpy(dtype=float)
    pairs = [(pos[a], pos[b]) for a, b in rules]
    with np.errstate(invalid="ignore"):
        for _ in range(passes):
            changed = 0
            for a, b in pairs:
                bad = X[:, a] > X[:, b]
                if bad.any():
                    X[bad, a], X[bad, b] = X[bad, b], X[bad, a].copy()
                    changed += bad.sum()
            if not changed:
                break
        else:  # only reachable with cyclic rules: clip instead
            for a, b in pairs:
                bad = X[:, a] > X[:, b]
                X[bad, b] = X[bad, a]
    for c in cols:
        syn[c] = X[:, pos[c]]
    return syn


def rule_violations(df, rules):
    bad = np.zeros(len(df), dtype=bool)
    with np.errstate(invalid="ignore"):
        for a, b in rules:
            bad |= (df[a] > df[b]).fillna(False).to_numpy()
    return float(bad.mean()) if len(df) else 0.0


def swap_units(real, cols, links):
    """Groups of columns that statistic matching must swap together."""
    parent = {c: c for c in cols}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a, b in links:
        parent[find(a)] = find(b)
    masks = {}
    for c in cols:  # identical missing pattern = recorded together
        m = real[c].isna().to_numpy()
        if 0 < m.sum() < len(m):
            masks.setdefault(m.tobytes(), []).append(c)
    for group in masks.values():
        for c in group[1:]:
            parent[find(c)] = find(group[0])
    units = {}
    for c in cols:
        units.setdefault(find(c), []).append(c)
    return list(units.values())


def find_category_rules(real, schema, cols, min_support=20, alpha=1e-3, max_levels=10):
    """Rules tied to a category value, holding in every real row, for example
         work_type = children  ->  ever_married = No        ("eq")
         work_type = children  ->  age <= 16                ("le"; "ge" for a lower bound)
    A is a category, a true/false column or a number with at most `max_levels` values; each level
    needs `min_support` rows. As for find_rules, a rule is kept only if it would hold by chance with
    probability below `alpha`: share(B = b) ** n for "eq", share(X <= bound) ** n for a bound.
    Returns tuples (A, a, kind, B, value) on the encoded table."""
    nunique = {c: real[c].nunique() for c in cols}
    levelled = [c for c in cols if schema[c]["kind"] in ("category", "bool")
                or (schema[c]["kind"] == "numeric" and 1 < nunique[c] <= max_levels)]
    ranged = [c for c in cols if schema[c]["kind"] in ("numeric", "datetime") and nunique[c] > max_levels]
    other = {c: schema[c]["levels"].index(OTHER) if OTHER in schema[c].get("levels", []) else None for c in cols}
    rules = []
    for a in levelled:
        for lv, n_a in real[a].value_counts().items():
            if n_a < min_support or lv == other[a]:
                continue
            g = real[real[a] == lv]
            for b in levelled:
                if b == a:
                    continue
                vals = g[b].dropna().unique()
                n = g[b].notna().sum()
                if len(vals) == 1 and n >= min_support and vals[0] != other[b]:
                    share = (real[b].dropna() == vals[0]).mean()
                    if share < 1 and n * np.log(share) <= np.log(alpha):
                        rules.append((a, lv, "eq", b, vals[0]))
            for x in ranged:
                v, allv = g[x].dropna(), real[x].dropna()
                if len(v) < min_support:
                    continue
                hi, lo = v.max(), v.min()
                below, above = (allv <= hi).mean(), (allv >= lo).mean()
                if hi < allv.max() and len(v) * np.log(below) <= np.log(alpha):
                    rules.append((a, lv, "le", x, hi))
                if lo > allv.min() and len(v) * np.log(above) <= np.log(alpha):
                    rules.append((a, lv, "ge", x, lo))
    return rules


def _broken(syn, rule):
    a, lv, kind, b, val = rule
    with np.errstate(invalid="ignore"):
        m = (syn[a] == lv).to_numpy()
        x = syn[b].to_numpy(dtype=float)
        if kind == "eq":
            return m & ~np.isnan(x) & (x != val)
        return m & ((x > val) if kind == "le" else (x < val))


def repair_category_rules(syn, crules, rng=None):
    """Implied values are set ("eq"); numbers outside the real range of their group are moved to
    the nearest value seen in that group (the bound itself).

    Mirror rules (CCU -> not CSRU, CSRU -> not CCU) are broken by the same rows. Such a row keeps
    one of the two at random, in proportion to how common each is, so no value is always preferred
    and the shares stay close to the real ones."""
    syn = syn.copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    eqs = [r for r in crules if r[2] == "eq"]
    for rule in crules:
        bad = _broken(syn, rule)
        if not bad.any():
            continue
        a, lv, kind, b, val = rule
        if kind == "eq":
            for m in eqs:
                if m[0] == b and m[3] == a:
                    both = bad & _broken(syn, m)
                    if both.any():
                        wa, wb = (syn[a] == lv).mean(), (syn[b] == m[1]).mean()
                        keep_b = both & (rng.random(len(syn)) < wb / (wa + wb))
                        syn.loc[keep_b, a] = m[4]
                        bad &= ~keep_b
        syn.loc[bad, b] = val
    return syn


def category_violations(df, crules):
    bad = np.zeros(len(df), dtype=bool)
    for rule in crules:
        bad |= _broken(df, rule)
    return float(bad.mean()) if len(df) else 0.0


def _show(schema, col, v):
    """A stored value as the user wrote it (category label, True/False, date or number)."""
    k = schema[col]["kind"]
    if k == "category":
        return str(schema[col]["levels"][int(v)])
    if k == "bool":
        return str(bool(v))
    if k == "datetime":
        return str(pd.Timestamp(int(v)))
    return f"{v:g}"


def describe_category_rule(schema, rule):
    a, lv, kind, b, val = rule
    sign = {"eq": "=", "le": "<=", "ge": ">="}[kind]
    return f"if {a} = {_show(schema, a, lv)} then {b} {sign} {_show(schema, b, val)}"


# ---------------------------------------------------------------------------
# target-aware steps
# ---------------------------------------------------------------------------

def target_type(real_f, target, schema):
    y = real_f[target].dropna()
    if schema[target]["kind"] == "category":
        return "binary" if y.nunique() <= 2 else "multiclass"
    return "binary" if y.nunique() <= 2 else "regression"


def _xgb(task, seed=0, **kw):
    from xgboost import XGBClassifier, XGBRegressor
    params = dict(S.LABEL_MODEL, random_state=seed, **kw)
    params.pop("eval_metric", None)
    return XGBRegressor(**params) if task == "regression" else XGBClassifier(**params)


def column_order(real_f, target, task, cols):
    """Target first, then features by importance for predicting it (if there is a target)."""
    if not target:
        return cols
    feats = [c for c in cols if c != target]
    keep = real_f[target].notna()
    y = real_f.loc[keep, target]
    if task != "regression":
        y = pd.factorize(y, sort=True)[0]
    model = _xgb(task, n_estimators=150).fit(real_f.loc[keep, feats], y)
    imp = pd.Series(model.feature_importances_, index=feats)
    return [target] + imp.sort_values(ascending=False, kind="stable").index.tolist()


def relabel_target(syn, real_f, target, task, rng, temperature=0.3, seed=0):
    """Redraw the target from a model fitted on real rows, with less noise than real labels.

    Binary: exact real class balance, top-k of logit + temperature * logistic noise (synth.py).
    Multi-class: Gumbel-max sampling from p ** (1 / temperature).
    Numeric: prediction + temperature * a real out-of-fold residual, mapped onto real values.
    """
    feats = [c for c in syn.columns if c != target]
    keep = real_f[target].notna().to_numpy()
    Xr, yr = real_f.loc[keep, feats], real_f.loc[keep, target].to_numpy()
    rows = syn[target].notna().to_numpy() if not keep.all() else np.ones(len(syn), dtype=bool)
    Xs = syn.loc[rows, feats]
    out = syn[target].to_numpy(dtype=float).copy()
    if task == "binary":
        levels = np.unique(yr)
        y01 = (yr == levels[-1]).astype(int)
        model = _xgb(task, seed).fit(Xr, y01)
        new = S.relabel(Xs, model, feats, y01.mean(), rng, temperature)
        out[rows] = np.where(new == 1, levels[-1], levels[0])
    elif task == "multiclass":
        levels, codes = np.unique(yr, return_inverse=True)
        model = _xgb(task, seed).fit(Xr, codes)
        logp = np.log(np.clip(model.predict_proba(Xs), 1e-9, 1)) / max(temperature, 1e-3)
        out[rows] = levels[np.argmax(logp + rng.gumbel(size=logp.shape), axis=1)]
    else:
        from sklearn.model_selection import cross_val_predict
        model = _xgb(task, seed)
        resid = yr - cross_val_predict(model, Xr, yr, cv=5)
        model.fit(Xr, yr)
        z = model.predict(Xs) + temperature * rng.choice(resid, rows.sum())
        ranks = np.argsort(np.argsort(z + rng.random(len(z)) * 1e-9))
        ref = np.sort(yr)
        out[rows] = ref[np.minimum(((ranks + 0.5) / len(z) * len(ref)).astype(int), len(ref) - 1)]
    syn[target] = out
    return syn


# ---------------------------------------------------------------------------
# quality report
# ---------------------------------------------------------------------------

def _eval_frame(f, schema):
    """Numbers, dates and true/false as floats; categories as text (the evaluator's split)."""
    out = f.copy()
    for c in f.columns:
        if schema[c]["kind"] == "category":
            levels = np.array(schema[c]["levels"], dtype=object)
            v = f[c].to_numpy(dtype=float)
            out[c] = pd.Series(np.where(np.isnan(v), None, levels[np.nan_to_num(v).astype(int)]),
                               index=f.index, dtype=object)
    return out


def _distance_space(frames, schema, ref):
    """Gower-like coordinates: ranks in [0, 1] for ordered columns, one-hot halves for categories.
    Manhattan distance / number of columns is then the mean per-column difference."""
    blocks = [[] for _ in frames]
    for c in ref.columns:
        if schema[c]["kind"] == "category":
            levels = np.unique(np.concatenate([f[c].dropna().to_numpy() for f in frames]))
            for k, f in enumerate(frames):
                v = f[c].to_numpy(dtype=float)
                blocks[k].append(0.5 * (v[:, None] == levels[None, :]))
                blocks[k].append(0.5 * np.isnan(v)[:, None])
        else:
            r = np.sort(ref[c].dropna().to_numpy())
            for k, f in enumerate(frames):
                v = f[c].to_numpy(dtype=float)
                q = np.searchsorted(r, v) / max(len(r), 1)
                m = np.isnan(v)
                blocks[k].append(np.where(m, 0.5, q)[:, None])
                blocks[k].append(m[:, None].astype(float))
    return [np.hstack(b) for b in blocks]


QI_WORDS = {"age", "sex", "gender", "height", "weight", "bmi", "zip", "zipcode", "postcode", "postal", "race",
            "ethnicity", "ethnic", "marital", "married", "birth", "birthdate", "dob"}
SUMMARY_WORDS = {"first", "last", "lowest", "highest", "median", "mean", "min", "max", "n", "count", "std"}


def quasi_identifiers(cols, extra=()):
    """Columns that together can point to a person (age, gender, height, postcode, ...), as in the
    hackathon's 'demographics' privacy group. Summaries such as Weight_first are left out."""
    out = []
    for c in cols:
        tokens = set(re.split(r"[^a-z0-9]+", str(c).lower())) - {""}
        if c in extra or (tokens & QI_WORDS and not tokens & SUMMARY_WORDS):
            out.append(c)
    return out


def _fidelity(real_f, syn_f, schema):
    """The hackathon's fidelity metric (and its parts) between two encoded tables."""
    from hackathon_synth_data_2026.fidelity import fidelity_score, mi_matrix
    cols = list(real_f.columns)
    r, s = _eval_frame(real_f, schema), _eval_frame(syn_f[cols], schema)
    num = [c for c in cols if schema[c]["kind"] != "category"]
    cat = [c for c in cols if c not in num]
    return fidelity_score(r, s, cols, num, cat, mi_matrix(r[cols]), mi_matrix(s[cols]))


def _closer_share(train_f, hold_f, syn_f, schema, cols, rng):
    """Share of synthetic rows nearer to a training row than to an unseen real row.
    New real patients give 50%; copies of training rows push it towards 100%. Ties count half."""
    from sklearn.neighbors import NearestNeighbors
    m = min(len(hold_f), len(train_f))
    tr = train_f.iloc[rng.choice(len(train_f), m, replace=False)][cols]
    sy = syn_f.iloc[rng.choice(len(syn_f), min(len(syn_f), 5000), replace=False)][cols]
    A, H, Z = _distance_space([tr, hold_f[cols], sy], schema, train_f[cols])
    nn = lambda ref: NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="brute").fit(ref)
    d_tr, d_ho = nn(A).kneighbors(Z)[0][:, 0], nn(H).kneighbors(Z)[0][:, 0]
    return float(np.mean(d_tr < d_ho) + 0.5 * np.mean(d_tr == d_ho))


def _privacy(train_f, hold_f, table, schema, groups, rng):
    """Worst group score, reduced by the share of exact copies of training rows."""
    key = lambda f: pd.util.hash_pandas_object(f[list(train_f.columns)].round(9), index=False)
    copies = float(key(table).isin(set(key(train_f))).mean())
    per = {}
    for name, cols in groups.items():
        share = _closer_share(train_f, hold_f, table, schema, cols, rng)
        per[name] = dict(columns=cols, closer=share, score=min(1.0, 2 * (1 - share)))
    return min(g["score"] for g in per.values()) * (1 - copies), per, copies


def _predictions(f, feats, target, task, test):
    """Train one fixed model on table f, predict the held-out real rows."""
    from xgboost import XGBClassifier, XGBRegressor
    f = f[f[target].notna()]
    if task == "regression":
        m = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        return m.fit(f[feats], f[target]).predict(test[feats])
    present = np.unique(f[target])
    if len(present) < 2:  # one class only: the best it can do is a constant guess
        fill = present[0] if len(present) else np.nan
        return np.zeros(len(test)) if task == "binary" else np.full(len(test), fill)
    m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    m.fit(f[feats], np.searchsorted(present, f[target].to_numpy()))
    if task == "binary":
        return m.predict_proba(test[feats])[:, -1]
    return present[m.predict(test[feats])]


def _score(task, y, pred, pos):
    from sklearn.metrics import average_precision_score, f1_score, r2_score
    if task == "binary":
        t = y == pos
        return float(average_precision_score(t, pred)) if t.any() else np.nan
    if task == "multiclass":
        return float(f1_score(y, pred, average="macro"))
    return float(r2_score(y, pred))


def _ratio_score(r):
    """1 when synthetic data is exactly as predictive as real data. Over-prediction (the outcome looks
    more predictable than it is) is penalised like under-prediction: 1.25x and 0.8x both score 0.8."""
    return float(min(r, 1 / r)) if r and r > 0 and np.isfinite(r) else 0.0


def evaluate(train_f, hold_f, syn_f, schema, target, task, seed=0, qi_extra=(), n_boot=300, progress=None):
    """Score the synthetic table, and two references scored the same way:
      real   the training rows themselves, i.e. what real data gets (the ceiling)
      floor  every training column shuffled on its own: right distributions, no relationships
    With held-out rows, fidelity is measured on them, so copying the training rows cannot raise it."""
    rep = {}
    rng = np.random.default_rng(seed)
    cols = list(train_f.columns)
    floor_f = train_f.apply(lambda c: pd.Series(rng.permutation(c.to_numpy()))).iloc[:len(syn_f)]
    step = progress or (lambda f: None)
    fid_train, parts_train = _fidelity(train_f, syn_f, schema)
    step(0.15)
    rep["fidelity_train"] = fid_train
    if hold_f is None or not len(hold_f):
        rep.update(fidelity=fid_train, fidelity_parts=parts_train, fidelity_basis="training rows")
        return rep
    fid, parts = _fidelity(hold_f, syn_f, schema)
    step(0.3)
    rep.update(fidelity=fid, fidelity_parts=parts, fidelity_basis="unseen rows",
               fidelity_real=_fidelity(hold_f, train_f, schema)[0],
               fidelity_floor=_fidelity(hold_f, floor_f, schema)[0])

    step(0.55)
    if target:
        feats = [c for c in cols if c != target]
        ht = hold_f[hold_f[target].notna()].reset_index(drop=True)
        y = ht[target].to_numpy()
        pos = train_f[target].value_counts().idxmin()  # binary: the rarer class is the event of interest
        preds = {k: _predictions(t, feats, target, task, ht) for k, t in
                 (("real", train_f), ("synthetic", syn_f), ("floor", floor_f))}
        sc = {k: _score(task, y, p, pos) for k, p in preds.items()}
        rep["utility_metric"] = {"binary": "average precision", "multiclass": "macro F1",
                                 "regression": "R2"}[task] + " on held-out real rows"
        rep["utility_real"], rep["utility_synthetic"] = sc["real"], sc["synthetic"]
        if task == "binary":
            from sklearn.metrics import roc_auc_score
            t = y == pos
            rep["test_positives"] = int(t.sum())
            if 0 < t.sum() < len(t):
                rep["auc_real"] = float(roc_auc_score(t, preds["real"]))
                rep["auc_synthetic"] = float(roc_auc_score(t, preds["synthetic"]))
        if sc["real"] > 0:
            ratio = sc["synthetic"] / sc["real"]
            boots = []
            for _ in range(n_boot):  # how much the ratio moves if the test rows had come out differently
                i = rng.integers(0, len(y), len(y))
                a, b = _score(task, y[i], preds["synthetic"][i], pos), _score(task, y[i], preds["real"][i], pos)
                if np.isfinite(a) and np.isfinite(b) and b > 0:
                    boots.append(a / b)
            rep["utility_ratio"] = float(ratio)
            rep["utility_interval"] = [float(v) for v in np.percentile(boots, [5, 95])] if boots else [ratio, ratio]
            rep["utility"] = _ratio_score(ratio)
            rep["utility_floor_ratio"] = float(sc["floor"] / sc["real"])
            rep["utility_floor"] = _ratio_score(sc["floor"] / sc["real"])

    step(0.7)
    groups = {"all columns": cols}
    qi = [c for c in quasi_identifiers(cols, qi_extra) if c != target]
    if 2 <= len(qi) < len(cols):
        groups["quasi-identifiers"] = qi
    rep["privacy"], rep["privacy_groups"], rep["exact_copies"] = _privacy(train_f, hold_f, syn_f, schema, groups, rng)
    rep["closer_to_train"] = rep["privacy_groups"]["all columns"]["closer"]
    step(0.9)
    rep["privacy_floor"] = _privacy(train_f, hold_f, floor_f, schema, groups, rng)[0]
    step(1.0)

    has_u = "utility" in rep  # no target, or a target even real data cannot predict: two parts only
    rep["overall"] = 100 * float(np.mean(([rep["utility"]] if has_u else []) + [rep["fidelity"], rep["privacy"]]))
    rep["overall_real"] = 100 * float(np.mean(([1.0] if has_u else []) + [rep["fidelity_real"], 1.0]))
    rep["overall_floor"] = 100 * float(np.mean(([rep["utility_floor"]] if has_u else [])
                                               + [rep["fidelity_floor"], rep["privacy_floor"]]))
    # Copies of the training rows: as useful and faithful as real data, but no privacy at all.
    rep["overall_copy"] = 100 * float(np.mean(([1.0] if has_u else []) + [rep["fidelity_real"], 0.0]))
    return rep


def honesty_warnings(rep, temperature):
    """Plain-language notes on anything that makes a number flattering or unreliable."""
    w, target = [], rep.get("target")
    if target and temperature is not None and temperature < 1:
        w.append(f"Competition mode is on: '{target}' was redrawn with less noise than in the real data "
                 f"(temperature {temperature}). Models trained on this data will look more accurate than "
                 "they would on real data.")
    if not rep.get("holdout_rows"):
        w.append("No rows were held out, so utility and privacy were not measured, and fidelity was measured "
                 "on the same rows the generator learned from, which flatters it.")
    elif rep["holdout_rows"] < 200:
        w.append(f"Only {rep['holdout_rows']} rows were held out for testing, so every score is rough.")
    if "utility_ratio" in rep:
        lo, hi = rep["utility_interval"]
        if lo > 1:
            w.append(f"A model trained on the synthetic data performs at {rep['utility_ratio']:.0%} of one trained "
                     f"on real data: the synthetic data makes '{target}' look more predictable than it is.")
        elif hi < 1:
            w.append(f"A model trained on the synthetic data performs at only {rep['utility_ratio']:.0%} of one "
                     f"trained on real data: part of the link between the columns and '{target}' is lost.")
        npos = rep.get("test_positives")
        if npos is not None and npos < 20:
            w.append(f"Only {npos} positive cases among the test rows: the utility score is close to meaningless.")
        elif hi - lo > 0.3:
            few = f" (only {npos} positive cases among the test rows)" if npos is not None else ""
            w.append(f"The utility result is uncertain{few}: a model trained on this data could perform anywhere "
                     f"from {lo:.0%} to {hi:.0%} of one trained on real data.")
    elif target and rep.get("holdout_rows") and rep.get("utility_real", 1) <= 0:
        w.append(f"Even the real data cannot predict '{target}' on held-out rows, so utility is not scored.")
    return w


def notes(rep):
    """Neutral context for the reader (not problems with the synthetic data)."""
    out = []
    if "fidelity_real" in rep and rep["fidelity_real"] - rep["fidelity_floor"] < 0.1:
        out.append(f"Most of the information in this table sits in the individual columns rather than in the "
                   f"links between them: shuffled columns already reach a fidelity of {rep['fidelity_floor']:.3f}, "
                   f"against {rep['fidelity_real']:.3f} for real data. Fidelity therefore mainly checks the "
                   "distribution of each column.")
    return out


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

COMPETITION_TEMPERATURE = 0.3  # sharper labels: higher hackathon scores, less faithful outcome


def synthesize(df, target=None, seed=0, rows=None, holdout=0.2, id_cols=(), jitter_cols=(),
               jitter=0.25, min_leaf=10, corr_iters=10, stat_iters=None, temperature=1.0,
               rules=True, report=True, log=print, progress=None):
    """Return (synthetic DataFrame, report dict). df is the uploaded table, unchanged.

    temperature 1.0 redraws the target with the real amount of noise (faithful). Lower values
    (COMPETITION_TEMPERATURE) make the outcome easier to predict than in reality."""
    t0 = time.time()
    log = log or (lambda *a: None)
    rng = np.random.default_rng(seed)
    if target is not None and target not in df.columns:
        raise ValueError(f"target column {target!r} not found")
    schema = detect_schema(df, set(id_cols) - {target})
    if target and schema[target]["kind"] == "id":
        schema[target]["kind"] = "category"
    real_all = encode(df, schema)
    cols = list(real_all.columns)
    if len(cols) < 2:
        raise ValueError("need at least two non-identifier columns")
    task = target_type(real_all, target, schema) if target else None

    # Holdout: the generator never sees these rows; the report uses them.
    idx = rng.permutation(len(df))
    n_hold = int(round(holdout * len(df))) if holdout and len(df) >= 50 else 0
    if n_hold and task in ("binary", "multiclass"):
        from sklearn.model_selection import train_test_split
        y = real_all[target].fillna(-1)
        tr_i, ho_i = train_test_split(np.arange(len(df)), test_size=n_hold, random_state=seed,
                                      stratify=y if y.value_counts().min() >= 2 else None)
    else:
        ho_i, tr_i = idx[:n_hold], idx[n_hold:]
    train, hold = real_all.iloc[np.sort(tr_i)].reset_index(drop=True), real_all.iloc[np.sort(ho_i)].reset_index(drop=True)
    n = rows or len(train)
    kinds = {c: schema[c]["kind"] for c in cols}
    ordered = [c for c in cols if kinds[c] in ("numeric", "datetime", "bool") and c != target]
    numeric = [c for c in cols if kinds[c] in ("numeric", "datetime") and c != target]
    cats = {c for c in cols if kinds[c] == "category"}
    log(f"{len(df)} rows x {len(df.columns)} columns; fitting on {len(train)}, holding out {len(hold)}")
    tick = progress or (lambda f, label: None)
    part = lambda lo, hi, label: (lambda f: tick(lo + (hi - lo) * f, label))  # share of the whole run
    tick(0.01, "Preparing")

    # 1. sequential CART
    order = column_order(train, target, task, cols)
    syn = S.cart_synthesize(train, n, order, rng, min_leaf=min_leaf, label_first=False, seed=seed,
                            cat_cols=cats | ({target} if task in ("binary", "multiclass") else set()),
                            progress=part(0.02, 0.12, "Building rows column by column"))
    syn = syn[cols]
    log(f"CART done ({time.time() - t0:.0f}s)")
    # 2. opt-in jitter of quasi-identifiers
    for c in jitter_cols:
        if c in numeric and jitter > 0:
            q75, q25 = np.nanpercentile(train[c], [75, 25])
            syn[c] = syn[c] + rng.normal(0, jitter * (q75 - q25), n)
    # 3. correlation restoration and 4. marginal matching (ordered columns only)
    # Columns tied by an "if A then B" rule (e.g. CCU, CSRU, SICU: at most one is true) come out of
    # CART consistent; the polishing steps below keep them together instead of moving each alone.
    feats = [c for c in cols if c != target]
    crules = find_category_rules(train, schema, feats) if rules else []
    eq_links = [(r[0], r[3]) for r in crules if r[2] == "eq"]
    linked = {c for pair in eq_links for c in pair}
    varying = [c for c in ordered if train[c].nunique() > 1 and train[c].notna().sum() >= 3]
    if corr_iters and len(varying) > 1:
        syn = S.restore_correlations(syn, train, varying, rng, 1.0, corr_iters, fixed=linked,
                                     progress=part(0.12, 0.34, "Restoring correlations"))
    tick(0.34, "Learning and enforcing rules")
    syn = S.match_marginals(syn, train, [c for c in ordered if c not in linked], rng)
    # 5. learned consistency rules
    found, equal = find_rules(train, numeric) if rules and len(numeric) > 1 else ([], [])
    before = rule_violations(syn, found)
    syn = repair_rules(syn, found, equal)
    log(f"{len(found)} rules ({len(reduce_rules(found))} after removing implied ones); "
        f"{before:.1%} of rows broke one before repair")
    # 6. statistic matching
    units = swap_units(train, feats, found + equal + eq_links)
    iters = stat_iters if stat_iters is not None else int(min(3_000_000, 12 * n * len(units)))
    if iters and len(units) > 1 and len(feats) <= 300:
        from statmatch import match_statistics
        syn = match_statistics(syn, train, feats, iters, seed, units=units, corr_cols=ordered,
                               progress=part(0.36, 0.8, "Polishing: matching correlations and dependencies"))
        log(f"statistic matching: {iters:,} swap attempts over {len(units)} blocks ({time.time() - t0:.0f}s)")
    tick(0.8, "Enforcing category rules")
    # 7. category rules ("if work_type = children then age <= 16"), repaired last so nothing undoes them.
    #    Fixing one rule type can touch the other, so both are repaired until neither is broken.
    cat_before = category_violations(syn, crules)
    for _ in range(5):
        syn = repair_category_rules(syn, crules, rng)
        syn = repair_rules(syn, found, equal)
        if not category_violations(syn, crules) and not rule_violations(syn, found):
            break
    log(f"{len(crules)} category rules; {cat_before:.1%} of rows broke one before repair")
    # 8. relabelling (the target is in no rule, so this cannot break one)
    if target and temperature is not None and temperature >= 0:
        tick(0.81, f"Redrawing '{target}'")
        syn = relabel_target(syn, train, target, task, rng, temperature, seed)
    out = decode(syn, schema, df)

    rep = dict(rows_in=len(df), rows_out=len(out), fit_rows=len(train), holdout_rows=len(hold),
               target=target, task=task, seed=seed, temperature=temperature if target else None,
               columns={c: {k: v for k, v in i.items() if k != "levels"} for c, i in schema.items()},
               rules=[f"{a} <= {b}" for a, b in reduce_rules(found)] + [f"{a} == {b}" for a, b in equal],
               rule_violations_before_repair=before, rule_violations=rule_violations(syn, found),
               category_rules=[describe_category_rule(schema, r) for r in crules],
               category_violations_before_repair=cat_before, category_violations=category_violations(syn, crules),
               swap_blocks=len(units))
    if report:
        log("scoring (synthetic data, plus real and shuffled data as references) ...")
        rep.update(evaluate(train, hold if len(hold) else None, syn, schema, target, task, seed,
                            qi_extra=tuple(jitter_cols),
                            progress=part(0.82, 1.0, "Scoring against real and shuffled data")))
    rep["warnings"] = honesty_warnings(rep, temperature)
    rep["notes"] = notes(rep)
    tick(1.0, "Done")
    rep["seconds"] = time.time() - t0
    return out, rep


# ---------------------------------------------------------------------------
# saved results: the same table with the same settings loads instantly
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
CODE_FILES = [HERE / "anysynth.py", HERE / "statmatch.py", HERE / "synth.py",
              HERE.parent / "src/hackathon_synth_data_2026/fidelity.py"]


def code_version():
    """Fingerprint of the code that makes and scores the data; a code change invalidates saved results.
    Comments, docstrings and formatting are ignored (the parsed code is hashed), so editing them does not."""
    import ast
    h = hashlib.sha256()
    for f in CODE_FILES:
        if not f.exists():
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and body
                    and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        h.update(ast.dump(tree).encode())
    return h.hexdigest()[:12]


def _plain(v):
    """numpy values to plain Python, so the report can be stored as JSON."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def cached_synthesize(df, data_hash, target=None, seed=0, rows=None, holdout=0.2, id_cols=(), jitter_cols=(),
                      corr_iters=10, stat_iters=None, temperature=1.0, cache_dirs=(), write_dir=None,
                      refresh=False, log=print, progress=None):
    """synthesize(), but a finished result is saved and reused for the same file, settings and code.
    Returns (synthetic DataFrame, report, loaded_from_cache)."""
    opts = dict(target=target, seed=int(seed), rows=int(rows) if rows else None, holdout=float(holdout),
                id_cols=sorted(id_cols), jitter_cols=sorted(jitter_cols), corr_iters=int(corr_iters),
                stat_iters=stat_iters, temperature=temperature)
    code = code_version()
    key = hashlib.sha256(json.dumps(dict(data=data_hash, opts=opts, code=code),
                                    sort_keys=True).encode()).hexdigest()[:20]
    if not refresh:
        for d in cache_dirs:
            p = Path(d) / key
            if (p / "synthetic.parquet").exists() and (p / "report.json").exists():
                rep = json.loads((p / "report.json").read_text())
                return pd.read_parquet(p / "synthetic.parquet"), rep, True
        # Same file and settings saved by an earlier version of the code? Say why it runs again.
        for d in cache_dirs:
            for f in Path(d).glob("*/report.json") if Path(d).exists() else []:
                old = json.loads(f.read_text())
                if old.get("data_hash") == data_hash and old.get("options") == _plain(opts) and old.get("code") != code:
                    (log or (lambda *a: None))(
                        f"A saved result for these settings exists from an earlier version of the generator "
                        f"(made {old.get('saved_at')}, it took {old.get('seconds', 0):.0f}s). The code has changed "
                        "since, so it runs again; the new result is saved for next time.")
                    break
    syn, rep = synthesize(df, log=log, progress=progress, **opts)
    rep["saved_at"] = time.strftime("%Y-%m-%d %H:%M")
    if write_dir:
        p = Path(write_dir) / key
        p.mkdir(parents=True, exist_ok=True)
        syn.to_parquet(p / "synthetic.parquet", index=False)
        (p / "report.json").write_text(json.dumps(_plain(dict(rep, options=opts, data_hash=data_hash, code=code)),
                                                  indent=1))
    return syn, rep, False


def summary(rep):
    """Short human-readable report."""
    lines = []
    if "overall" in rep:
        have = [k for k in ("utility", "fidelity", "privacy") if k in rep]
        lines.append(f"overall {rep['overall']:.1f} / 100  (mean of {', '.join(have)}); on the same tests real data "
                     f"scores {rep['overall_real']:.1f}, shuffled columns (useless) {rep['overall_floor']:.1f} and "
                     f"copies of the training rows (not private) {rep['overall_copy']:.1f}")
    if "utility" in rep:
        lo, hi = rep["utility_interval"]
        lines.append(f"  utility   {rep['utility']:.3f}  a model trained on synthetic data performs at "
                     f"{rep['utility_ratio']:.0%} of one trained on real data (90% range {lo:.0%}-{hi:.0%}; "
                     f"shuffled {rep['utility_floor_ratio']:.0%}); {rep['utility_metric']}: "
                     f"{rep['utility_synthetic']:.3f} vs {rep['utility_real']:.3f}")
    if "fidelity" in rep:
        parts = ", ".join(f"{k} {v:.3f}" for k, v in rep["fidelity_parts"].items())
        if rep["fidelity_basis"] == "unseen rows":
            lines.append(f"  fidelity  {rep['fidelity']:.3f}  on unseen rows; real data {rep['fidelity_real']:.3f}, "
                         f"shuffled {rep['fidelity_floor']:.3f}; on training rows {rep['fidelity_train']:.3f}")
        else:
            lines.append(f"  fidelity  {rep['fidelity']:.3f}  on the rows it learned from (flattering)")
        lines.append(f"            ({parts})")
    if "privacy" in rep:
        groups = "; ".join(f"{name} {g['closer']:.1%}" for name, g in rep["privacy_groups"].items())
        lines.append(f"  privacy   {rep['privacy']:.3f}  synthetic rows closer to training than to unseen rows: "
                     f"{groups} (new real patients: 50%); exact copies {rep['exact_copies']:.1%}")
    for w in rep.get("warnings", []):
        lines.append(f"! {w}")
    for nt in rep.get("notes", []):
        lines.append(f"note: {nt}")
    kinds = pd.Series({c: i["kind"] for c, i in rep["columns"].items()}).value_counts()
    lines.append("columns: " + ", ".join(f"{v} {k}" for k, v in kinds.items()))
    lines.append(f"order rules learned (A <= B): {len(rep['rules'])}; rows breaking one: "
                 f"{rep['rule_violations_before_repair']:.1%} before repair, {rep['rule_violations']:.1%} after")
    for r in rep["rules"][:6]:
        lines.append(f"  {r}")
    if len(rep["rules"]) > 6:
        lines.append(f"  ... {len(rep['rules']) - 6} more in the report file")
    crules = rep.get("category_rules", [])
    lines.append(f"category rules learned (if A = a then ...): {len(crules)}; rows breaking one: "
                 f"{rep['category_violations_before_repair']:.1%} before repair, {rep['category_violations']:.1%} after")
    for r in crules[:6]:
        lines.append(f"  {r}")
    if len(crules) > 6:
        lines.append(f"  ... {len(crules) - 6} more in the report file")
    pooled = [c for c, i in rep["columns"].items() if i.get("pooled")]
    if pooled:
        lines.append(f"rare categories pooled into '{OTHER}': {', '.join(pooled)}")
    ids = [c for c, i in rep["columns"].items() if i["kind"] == "id"]
    if ids:
        lines.append(f"replaced with fresh identifiers: {', '.join(ids)}")
    lines.append(f"{rep['seconds']:.0f}s")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="CSV, TSV, Parquet, Excel or JSON file")
    ap.add_argument("--target", help="column to predict (optional)")
    ap.add_argument("--out", help="output file; the format follows the extension (default: <data>_synthetic.csv)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, help="synthetic rows (default: as many as the fitted rows)")
    ap.add_argument("--holdout", type=float, default=0.2, help="share held out for the report (0 = use all rows)")
    ap.add_argument("--id", default="", help="comma-separated identifier columns (auto-detected otherwise)")
    ap.add_argument("--jitter", default="", help="comma-separated quasi-identifiers to jitter (e.g. Age,Height)")
    ap.add_argument("--jitter-size", type=float, default=0.25, help="jitter in IQR units")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="target relabelling noise: 1 = as noisy as the real data (default); negative = off")
    ap.add_argument("--competition", action="store_true",
                    help=f"temperature {COMPETITION_TEMPERATURE}: higher hackathon scores, but the outcome "
                         "becomes easier to predict than in reality")
    ap.add_argument("--stat-iters", type=int, help="statistic-matching swap attempts (default scales with size)")
    ap.add_argument("--fast", action="store_true", help="fewer passes: about 3x quicker, slightly lower fidelity")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--cache", metavar="DIR", help="save the result in DIR and reuse it for the same file, settings "
                    "and code")
    args = ap.parse_args()

    split = lambda s: [c.strip() for c in s.split(",") if c.strip()]
    df = read_table(args.data)
    opts = dict(target=args.target, seed=args.seed, rows=args.rows, holdout=args.holdout,
                id_cols=split(args.id), jitter_cols=split(args.jitter),
                temperature=COMPETITION_TEMPERATURE if args.competition
                else None if args.temperature < 0 else args.temperature,
                corr_iters=3 if args.fast else 10,
                stat_iters=300_000 if args.fast and args.stat_iters is None else args.stat_iters)
    if args.cache:
        data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
        syn, rep, saved = cached_synthesize(df, data_hash, cache_dirs=[args.cache], write_dir=args.cache, **opts)
        if saved:
            print(f"loaded a saved result from {args.cache} (made {rep.get('saved_at')})")
    else:
        syn, rep = synthesize(df, jitter=args.jitter_size, report=not args.no_report, **opts)
    out = Path(args.out or Path(args.data).with_name(Path(args.data).stem + "_synthetic.csv"))
    write_table(syn, out)
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(rep, indent=2, default=str))
    print(summary(rep))
    print(f"wrote {out} {syn.shape} and {report_path.name}")


if __name__ == "__main__":
    main()
