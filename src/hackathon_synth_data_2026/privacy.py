#!/usr/bin/env python3
"""
Score wird immer ausgegeben beim runnen, wenn man die genaue JSON Datei,  Zusammenfassungen der Seeds speichern will muss man --output angeben. Die JSON Datei enthält alle Details, die CSVs sind für die schnelle Übersicht.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEPENDENCY_ERROR = None
try:
    import numpy as np
    import pandas as pd
except ImportError:
    DEPENDENCY_ERROR = 'Python-Abhängigkeiten fehlen; requirements.txt installieren'

BINARY = ['Gender', 'CCU', 'CSRU', 'SICU', 'MechVentLast8Hour', 'In-hospital_death']
DEFAULTS = dict(near_threshold=.05, threshold_source='Provisorischer Startwert, nicht kalibriert',
                categorical_columns=BINARY, min_common_fraction=.25, min_common_features=3,
                min_targets=30, bootstrap_repeats=200, block_rows=64)
SEEDS = [42, 2026, 31415]


class AuditError(ValueError):
    """Kontrollierte Fehlertexte enthalten keine individuellen Werte."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def unit(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1 and math.isfinite(value)


def config_from(path=None, threshold=None, overrides=None):
    config = dict(DEFAULTS)
    if path is not None:
        custom = json.loads(path.read_text(encoding='utf-8'))
        require(isinstance(custom, dict) and set(custom) <= set(DEFAULTS), 'Unbekannte/ungültige Konfigurationsfelder')
        config.update(custom)
    if overrides is not None:
        require(
            isinstance(overrides, dict)
            and set(overrides) <= set(DEFAULTS),
            "Unknown/invalid privacy configuration fields",
        )
        config.update(overrides)

    if threshold is not None:
        config.update(near_threshold=threshold, threshold_source='CLI-Vorgabe; extern zu begründen')
    require(unit(config['near_threshold']) and config['near_threshold'] < 1, 'Nähe-Schwelle muss in [0,1) liegen')
    require(unit(config['min_common_fraction']) and config['min_common_fraction'] > 0, 'Ungültiger gemeinsamer Merkmalsanteil')
    for key, minimum in [('min_common_features', 1), ('min_targets', 2), ('bootstrap_repeats', 100), ('block_rows', 1)]:
        require(type(config[key]) is int and config[key] >= minimum, f'Ungültige Konfiguration: {key}')
    cats = config['categorical_columns']
    require(isinstance(cats, list) and all(isinstance(c, str) for c in cats)
            and len(cats) == len(set(cats)) and 'recordid' not in cats, 'Ungültige kategoriale Spalten')
    require(isinstance(config['threshold_source'], str) and bool(config['threshold_source'].strip()), 'Schwellenherkunft fehlt')
    return config


def row_keys(frame):
    return [tuple(None if pd.isna(v) else v for v in row)
            for row in frame.itertuples(index=False, name=None)]


def copy_audit(train, synthetic):
    keys, reference = row_keys(synthetic), set(row_keys(train))
    count = sum(k in reference for k in keys)
    return dict(rows=len(keys), copied_rows=count, copy_rate=count / len(keys),
                duplicate_rows_excess=len(keys) - len(set(keys)),
                duplicate_rows_all=int(synthetic.duplicated(keep=False).sum()))


def read_profile(path, name, data_specs=None):
    """Read a complete feature-and-target table."""
    spec = data_specs or {}
    identifier = spec.get("identifier_var", "recordid")
    target = spec.get("target_var", "In-hospital_death")

    require(
        identifier is not None,
        "Privacy evaluation requires an identifier column",
    )

    path = Path(path)

    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        separator = (
            "\t"
            if path.suffix.lower() in (".tsv", ".txt")
            else ","
        )

        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as stream:
            header = next(
                csv.reader(stream, delimiter=separator),
                [],
            )

        require(
            header
            and all(header)
            and len(header) == len(set(header)),
            f"{name}: empty or duplicate column names",
        )

        frame = pd.read_csv(
            path,
            sep=separator,
            dtype={identifier: "string"},
        )

    require(
        frame.columns.is_unique
        and all(
            isinstance(column, str) and column
            for column in frame.columns
        ),
        f"{name}: empty or duplicate column names",
    )
    require(
        identifier in frame,
        f"{name}: missing identifier column {identifier!r}",
    )
    require(
        target in frame,
        f"{name}: missing target column {target!r}",
    )
    require(
        frame[target].notna().all()
        and frame[target].isin([0, 1]).all(),
        f"{name}: target must contain nonmissing 0/1 values",
    )

    # Preserve the existing scoring code's internal column names.
    require(
        identifier == "recordid" or "recordid" not in frame,
        f"{name}: conflicting identifier columns",
    )
    require(
        target == "In-hospital_death"
        or "In-hospital_death" not in frame,
        f"{name}: conflicting target columns",
    )

    split = spec.get("split_var")
    require(
        split not in (identifier, target),
        "Split column must differ from ID and target",
    )

    if split and split in frame:
        frame = frame.drop(columns=split)

    frame = frame.rename(
        columns={
            identifier: "recordid",
            target: "In-hospital_death",
        }
    )
    frame["recordid"] = frame["recordid"].astype("string")

    return frame


def load_data(paths, config, data_specs=None):
    raw, data, checks = {}, {}, {}

    for name, path in paths.items():
        raw[name] = read_profile(
            path,
            name,
            data_specs,
        )
    columns = list(raw['train'].columns)
    require('recordid' in columns and len(columns) > 1, 'recordid oder Profilmerkmale fehlen')
    cats = set(config['categorical_columns']) | set(BINARY)
    for name, frame in raw.items():
        require(len(frame) > 0 and set(frame.columns) == set(columns), f'{name}: leere Tabelle/Schemaabweichung')
        ids = frame['recordid']
        require(not ids.dropna().str.lower().isin(['inf', '+inf', '-inf', 'infinity', '-infinity']).any(), f'{name}: unendliche ID')
        if name != 'synthetic':
            require(ids.notna().all() and ids.is_unique, f'{name}: fehlende/doppelte Original-IDs')
        out = frame[[c for c in columns if c != 'recordid']].copy()
        for col in out:
            if col not in cats or col in BINARY:
                try:
                    out[col] = pd.to_numeric(out[col], errors='raise').astype(float)
                except (ValueError, TypeError):
                    raise AuditError(f'{name}: nichtnumerische Werte in {col}; Kategorien explizit konfigurieren') from None
                require(not np.isinf(out[col].to_numpy()).any(), f'{name}: unendliche Werte in {col}')
            elif out[col].dropna().astype(str).str.lower().isin(['inf', '-inf', '+inf', 'infinity', '-infinity']).any():
                raise AuditError(f'{name}: unendliche Kategorie in {col}')
            if col in ['SAPS-I', 'SOFA']:
                out[col] = out[col].replace(-1, np.nan)
            if col in BINARY:
                require(set(out[col].dropna()) <= {0., 1.}, f'{name}: ungültige binäre Kategorie in {col}')
            if col in cats:
                out[col] = out[col].map(lambda x: str(x) if pd.notna(x) else np.nan).astype(object)
        data[name] = out.reset_index(drop=True)
        checks[name] = dict(rows=len(out), missing_by_column=out.isna().sum().to_dict(),
                            missing_ids=int(ids.isna().sum()), duplicate_ids_excess=int(ids.duplicated().sum()))
    train_ids, control_ids = set(raw['train']['recordid']), set(raw['control']['recordid'])
    require(not train_ids & control_ids, 'Original-IDs überschneiden sich zwischen Training und Kontrolle')
    checks['shared_distinct_profiles_train_control'] = len(set(row_keys(data['train'])) & set(row_keys(data['control'])))
    checks['synthetic_id_collisions'] = dict(train=int(raw['synthetic']['recordid'].isin(train_ids).sum()),
                                            control=int(raw['synthetic']['recordid'].isin(control_ids).sum()))
    checks['unseen_category_counts'] = {
        name: {col: len(set(frame[col].dropna()) - set(data['train'][col].dropna())) for col in frame if col in cats}
        for name, frame in data.items() if name != 'train'}
    checks['split_note'] = 'Profilüberschneidungen sind Prüfhinweise; Personenunabhängigkeit extern bestätigen.'
    return data, checks


class MixedDistance:
    """Gleichgewichtete begrenzte Mischdistanz; ausschließlich auf train angepasst."""
    def __init__(self, train, config):
        self.config = config
        self.columns = [c for c in train if c != 'recordid' and train[c].nunique(dropna=True) > 1]
        require(bool(self.columns), 'Keine variablen beobachteten Distanzmerkmale')
        self.categorical = set(config['categorical_columns']) | set(BINARY)
        self.scales = {}
        for col in self.columns:
            if col not in self.categorical:
                values = train[col].dropna()
                scale = float(values.quantile(.75) - values.quantile(.25))
                if scale == 0:
                    scale = float(values.max() - values.min())
                require(math.isfinite(scale) and scale > 0, f'Ungültige Trainingsskala für {col}')
                self.scales[col] = scale
        self.required_common = max(min(config['min_common_features'], len(self.columns)),
                                   math.ceil(config['min_common_fraction'] * len(self.columns)))
        self.description = dict(active_columns=self.columns, excluded_columns=[c for c in train if c not in self.columns],
                                required_common_features=self.required_common, weighting='equal_per_active_column',
                                scaling='training_IQR; training_range_if_IQR_zero',
                                missing='both missing: excluded; one missing: cost 1; common-observation minimum')

    def pairwise(self, query, reference):
        """Inf bezeichnet unzureichend beobachtete Paare und wird nie exportiert."""
        output = np.empty((len(query), len(reference)), dtype=float)
        for start in range(0, len(query), self.config['block_rows']):
            block = query.iloc[start:start + self.config['block_rows']]
            total = np.zeros((len(block), len(reference)))
            observed = np.zeros_like(total)
            common = np.zeros_like(total)
            for col in self.columns:
                a, b = block[col].to_numpy()[:, None], reference[col].to_numpy()[None, :]
                ma, mb = pd.notna(a), pd.notna(b)
                both, union = ma & mb, ma | mb
                if col in self.categorical:
                    delta = (a != b).astype(float)
                else:
                    with np.errstate(over='ignore', invalid='ignore'):
                        delta = np.minimum(np.abs(a.astype(float) - b.astype(float)) / self.scales[col], 1.)
                total += np.where(both, delta, (ma != mb).astype(float))
                observed += union
                common += both
            valid = common >= self.required_common
            distances = np.full_like(total, np.inf)
            np.divide(total, observed, out=distances, where=valid)
            output[start:start + len(block)] = distances
        return output


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise AuditError(message)


def make_parser(required_paths=False):
    parser = Parser(description=__doc__)
    names = dict(train='train_data.csv', control='eval_data.csv', synthetic='synthetic_data.csv')
    candidates = [Path.cwd(), Path.cwd() / 'synthetic-data-generation', Path(__file__).resolve().parent.parent / 'synthetic-data-generation']
    base = next((p for p in candidates if all((p / f).is_file() for f in names.values())), Path.cwd())
    for k, f in names.items():
        parser.add_argument('--' + k, type=Path, required=required_paths, default=None if required_paths else base / f)
    parser.add_argument('--output', type=Path, help='Ausgabebasis; pro Start ein neuer Zeitstempelordner')
    parser.add_argument('--budget', '--attacks', type=int, default=500, dest='budget')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--near-threshold', type=float)
    return parser


def setup(args):
    require(DEPENDENCY_ERROR is None, DEPENDENCY_ERROR or '')
    require(type(args.budget) is int and args.budget > 0 and bool(args.seeds)
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in args.seeds)
            and len(args.seeds) == len(set(args.seeds)), 'Ungültiges Budget/Seeds')
    config = config_from(args.config, args.near_threshold, getattr(args, "privacy_options", None),)
    paths = {k: getattr(args, k) for k in ['train', 'control', 'synthetic']}
    packages = ['numpy', 'pandas']
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    platform=platform.platform(), versions=versions, configuration=config,
                    budget=args.budget, seeds=args.seeds,
                    paths={k: str(p.resolve()) if p is not None else None for k, p in paths.items()},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None,
                    scope='Table release; fixed two-sided nearest-neighbor membership statistic; no Anonymeter')
    return config, paths, metadata


def new_folder(base=None, prefix='privacy_standalone'):
    base = base or Path(__file__).resolve().parent / 'results'
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'), dir=base))



GROUPS = {
    'full': None,
    'demographics': ['Age', 'Gender', 'Height', 'Weight'],
    'clinical_first': ['HR_first', 'GCS_first', 'Glucose_first', 'Creatinine_first', 'Temp_first'],
}
FAMILIES = ('near_copy', 'membership')


def failure(exc):
    return str(exc) if isinstance(exc, AuditError) else f'{type(exc).__name__}: Prüfung fehlgeschlagen'


def coverage(planned, evaluated, **extra):
    return dict(planned=planned, evaluated=evaluated, uncomparable=planned - evaluated,
                excluded=0, evaluated_rate=evaluated / planned if planned else 0.,
                uncomparable_rate=(planned - evaluated) / planned if planned else 0.,
                excluded_rate=0., **extra)


def check_distances(values):
    a = np.asarray(values, dtype=float)
    require(a.ndim == 1 and len(a) > 0, 'Leere/ungültige Distanzliste')
    require(bool(((np.isfinite(a) & (a >= 0) & (a <= 1)) | np.isposinf(a)).all()),
              'Ungültige Distanz; nur +Inf bezeichnet ein unbekanntes Ergebnis')
    return a


def transformed_bounds(low, high):
    require(unit(low) and unit(high) and low <= high, 'Ungültige AUC-Grenzen')
    return [0. if low <= .5 <= high else min(abs(2 * low - 1), abs(2 * high - 1)),
            max(abs(2 * low - 1), abs(2 * high - 1))]


def rank_wins(a, b):
    """Anzahl Trainingssiege + halbe Ties, größere Scores bedeuten Training."""
    ordered = np.sort(b)
    left = np.searchsorted(ordered, a, side='left')
    right = np.searchsorted(ordered, a, side='right')
    return float(np.sum(left + .5 * (right - left)))


def membership_test(train_distances, control_distances, seed, repeats):
    train, control = map(check_distances, (train_distances, control_distances))
    a, b = -train[np.isfinite(train)], -control[np.isfinite(control)]
    total, known = len(train) * len(control), len(a) * len(b)
    wins = rank_wins(a, b)
    auc_bounds = [wins / total, (wins + total - known) / total]
    bounds = transformed_bounds(*auc_bounds)
    complete = known == total
    known_auc, ci, ties = None, None, None
    if known:
        known_auc = wins / known
        ordered = np.sort(b)
        ties = int(np.sum(np.searchsorted(ordered, a, 'right') - np.searchsorted(ordered, a, 'left')))
    if len(a) >= 2 and len(b) >= 2 and repeats > 0:
        rng = np.random.default_rng([seed, 404])
        draws = [rank_wins(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)) / known
                 for _ in range(repeats)]
        low, high = map(float, np.quantile(draws, [.025, .975]))
        ci = dict(auc=[low, high], indicator=transformed_bounds(low, high),
                  nominal_level=.95, method='stratified percentile bootstrap',
                  scope='known targets only; conditional on this synthetic table and preprocessing',
                  contains_chance=low <= .5 <= high)
    return dict(status='measured' if complete else 'coverage_bounded',
                measured_indicator=bounds[0] if complete else None,
                coverage_bounds=bounds, auc_coverage_bounds=auc_bounds,
                known_subset_auc=known_auc, known_pair_ties=ties,
                constant_known_scores=bool(np.ptp(np.r_[a, b]) == 0) if known else None,
                statistical_ci=ci,
                coverage=dict(train=coverage(len(train), len(a)), control=coverage(len(control), len(b)),
                              rank_pairs=coverage(total, known)),
                reason='' if complete else 'Ziele ohne zulässigen Nachbarn bleiben als unbekannte Rangpaare im Nenner')


def near_test(train_distances, control_distances, threshold):
    train, control = map(check_distances, (train_distances, control_distances))
    require(len(train) == len(control) and unit(threshold) and threshold < 1, 'Ungültige Near-copy-Eingaben')
    n = len(train)
    kt, kc = np.isfinite(train), np.isfinite(control)
    ht, hc = int((train <= threshold).sum()), int((control <= threshold).sum())
    ut, uc = n - int(kt.sum()), n - int(kc.sum())
    tl, th, cl, ch = ht / n, (ht + ut) / n, hc / n, (hc + uc) / n
    # f(t,c) steigt mit t und fällt mit  c=1 ist strukturell undefiniert;
    # jede mit der Beobachtung vereinbare solche Vervollständigung bekommt [0,1].
    if ch == 1:
        bounds = [0., 1.]
    else:
        bounds = [max(0., (tl - ch) / (1 - ch)), max(0., (th - cl) / (1 - cl))]
    complete = ut == uc == 0 and hc < n
    return dict(status='measured' if complete else 'coverage_bounded',
                measured_indicator=bounds[0] if complete else None, coverage_bounds=bounds,
                train_rate_bounds=[tl, th], control_rate_bounds=[cl, ch],
                close_train=ht, close_control=hc, not_close_train=int(kt.sum()) - ht,
                not_close_control=int(kc.sum()) - hc,
                control_rate_one_possible=ch == 1, statistical_ci=None,
                coverage=dict(train_neighbors=coverage(n, int(kt.sum())),
                              control_neighbors=coverage(n, int(kc.sum())),
                              jointly_known=coverage(n, int((kt & kc).sum()))),
                reason=('Kontrollanteil 1 möglich: normierter Zusatzindikator nicht überall definiert' if ch == 1
                        else '' if complete else 'Zeilen ohne zulässigen Nachbarn: close/not-close unbekannt'))


def error_test(exc, family=None, n=None, synthetic_rows=None):
    # Ein Ausführungsfehler ist keine festgestellte Nichtvergleichbarkeit.
    planned = None
    if n is not None and synthetic_rows is not None:
        sizes = ({'train_neighbors': synthetic_rows, 'control_neighbors': synthetic_rows,
                  'jointly_known': synthetic_rows} if family == 'near_copy' else
                 {'train': n, 'control': n, 'rank_pairs': n * n})
        planned = {k: dict(planned=v, evaluated=None, uncomparable=None, excluded=0,
                           evaluated_rate=None, uncomparable_rate=None, excluded_rate=0.,
                           execution_failed=True) for k, v in sizes.items()}
    return dict(status='error', measured_indicator=None, coverage_bounds=None, coverage=planned, reason=failure(exc))


def run_seed(data, metric, budget, seed, config, group):
    started = time.perf_counter()
    n = min(budget, len(data['train']), len(data['control']))
    row = dict(group=group, seed=seed, targets_per_group=n, synthetic_rows=len(data['synthetic']), tests={})
    # Keine Vorauswahl geeigneter Personen: unzureichende Ziele bleiben im Nenner.
    try:
        require(n >= config['min_targets'], 'Zu wenige echte Ziele für das feste Mindestbudget')
        rng = np.random.default_rng(seed)
        targets = [data[k].iloc[rng.choice(len(data[k]), n, replace=False)] for k in ('train', 'control')]
        eligibility = {}
        for name, frame in zip(('train', 'control'), targets):
            eligible = int((frame[metric.columns].notna().sum(axis=1) >= metric.required_common).sum())
            eligibility[name] = dict(sampled=n, eligible_by_observation_count=eligible,
                                     insufficient_observation_count=n - eligible, excluded=0,
                                     not_sampled=len(data[name]) - n)
        eligible_syn = int((data['synthetic'][metric.columns].notna().sum(axis=1) >= metric.required_common).sum())
        eligibility['synthetic'] = dict(sampled=len(data['synthetic']), eligible_by_observation_count=eligible_syn,
                                        insufficient_observation_count=len(data['synthetic']) - eligible_syn, excluded=0)
        row['eligibility'] = eligibility
        distances = [metric.pairwise(target, data['synthetic']) for target in targets]
        row['pair_coverage'] = {name: coverage(int(d.size), int(np.isfinite(d).sum()))
                                for name, d in zip(('train', 'control'), distances)}
    except Exception as exc:
        row['tests'] = {family: error_test(exc, family, n, len(data['synthetic'])) for family in FAMILIES}
    else:
        # Getrennte Fehlergrenzen: ein Fehler im Indikator blockiert den anderen nicht.
        for family in FAMILIES:
            try:
                if family == 'near_copy':
                    test = near_test(*(d.min(axis=0) for d in distances), config['near_threshold'])
                else:
                    test = membership_test(*(d.min(axis=1) for d in distances), seed, config['bootstrap_repeats'])
                row['tests'][family] = test
            except Exception as exc:
                row['tests'][family] = error_test(exc, family, n, len(data['synthetic']))
    row['seconds'] = time.perf_counter() - started
    return row


def aggregate(copy_rate, runs, seeds, fatal=None):
    reasons = [fatal] if fatal else []
    expected = {(g, s) for g in GROUPS for s in seeds}
    actual = [(r.get('group'), r.get('seed')) for r in runs]
    structurally_complete = bool(seeds) and len(seeds) == len(set(seeds)) and len(actual) == len(expected) and set(actual) == expected
    if not structurally_complete:
        reasons.append('Pflichtgruppen/Seeds fehlen oder sind doppelt')
    if not unit(copy_rate):
        reasons.append('Kopienprüfung fehlt oder ist ungültig')
    scenarios, components, limits = {}, {'copy_rate': copy_rate if unit(copy_rate) else None}, {}
    all_measured = True
    for family in FAMILIES:
        scenarios[family] = {}
        for group in GROUPS:
            group_rows = [r for r in runs if r.get('group') == group]
            bounds = []
            for row in group_rows:
                test = row.get('tests', {}).get(family, {})
                bound = test.get('coverage_bounds')
                valid = (test.get('status') in ('measured', 'coverage_bounded') and isinstance(bound, list)
                         and len(bound) == 2 and all(unit(v) for v in bound) and bound[0] <= bound[1])
                if not valid:
                    reasons.append(f"{group}/{row.get('seed')}/{family}: {test.get('reason') or 'ungültiger Pflichtlauf'}")
                else:
                    bounds.append(bound)
                all_measured &= test.get('status') == 'measured'
            if structurally_complete and len(bounds) == len(seeds):
                scenarios[family][group] = [sum(b[i] for b in bounds) / len(seeds) for i in (0, 1)]
            else:
                scenarios[family][group] = None
        if all(v is not None for v in scenarios[family].values()):
            limits[family] = [max(v[i] for v in scenarios[family].values()) for i in (0, 1)]
            components[family] = limits[family][1]
        else:
            limits[family], components[family] = None, None
    technical = not reasons
    known_zero = unit(copy_rate) and copy_rate == 1.
    if known_zero:
        score, score_bounds, basis = 0., [0., 0.], 'determined_by_full_copy'
    elif technical:
        score = 1 - max(components.values())
        score_bounds = [score, 1 - max(copy_rate, *(b[0] for b in limits.values()))]
        basis = 'measured' if all_measured else 'conservative_coverage_bound'
    else:
        score, score_bounds, basis = 0., None, 'error_fallback'
    require(unit(score), 'Interner Scorefehler')
    return dict(privacy_score=score, status='experimentell/nicht kalibriert' if technical or known_zero else 'ungültig',
                score_basis=basis, technically_complete=technical, coverage_complete=technical and all_measured,
                used_conservative_bounds=basis == 'conservative_coverage_bound', fallback_score=basis == 'error_fallback',
                full_copy_determines_zero=known_zero, valid_for_ranking=False,
                components=components, component_coverage_bounds=limits, scenario_mean_coverage_bounds=scenarios,
                score_coverage_bounds=score_bounds, reasons=reasons,
                rule='1 - max(copy_rate, max_group(mean_seed(near_upper)), max_group(mean_seed(membership_upper)))',
                note='Experimentelle Wettbewerbsheuristik. Coverage-Grenzen sind keine Konfidenzintervalle oder Datenschutzgarantie.')


def evaluate(data, config, budget, seeds, progress=True):
    copies, copy_error = None, None
    try:
        copies = copy_audit(data['train'], data['synthetic'])
        copies.update(status='measured', coverage=coverage(len(data['synthetic']), len(data['synthetic'])))
    except Exception as exc:
        copy_error = failure(exc)
    runs, preprocessing = [], {}
    for group, preset in GROUPS.items():
        columns = list(data['train'].columns) if preset is None else preset
        try:
            require(set(columns) <= set(data['train'].columns), f'{group}: Pflichtmerkmale fehlen')
            metric = MixedDistance(data['train'][columns], config)
            preprocessing[group] = dict(metric.description, requested_columns=columns,
                                       small_active_group=len(metric.columns) <= config['min_common_features'])
        except Exception as exc:
            preprocessing[group] = dict(requested_columns=columns, error=failure(exc))
            n = min(budget, len(data['train']), len(data['control']))
            runs.extend(dict(group=group, seed=seed, targets_per_group=n, synthetic_rows=len(data['synthetic']),
                             tests={f: error_test(exc, f, n, len(data['synthetic'])) for f in FAMILIES}) for seed in seeds)
            continue
        for seed in seeds:
            row = run_seed(data, metric, budget, seed, config, group)
            runs.append(row)
            if progress:
                statuses = ', '.join(f'{f}={t["status"]}' for f, t in row['tests'].items())
                print(f'{group}/{seed}: {statuses} ({row["seconds"]:.1f}s)', flush=True)
    return dict(summary=aggregate(copies['copy_rate'] if copies else None, runs, seeds, copy_error),
                copies=copies, runs=runs, preprocessing=preprocessing)


def write_csv(path, rows, default_fields):
    fields = list(dict.fromkeys(default_fields + [k for row in rows for k in row]))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def write_result(folder, result, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'privacy.json').write_text(json.dumps(dict(result, metadata=metadata), ensure_ascii=False,
                                                   indent=2, allow_nan=False), encoding='utf-8')
    s = result['summary']
    write_csv(folder / 'summary.csv', [dict(s, **s['components'])], ['privacy_score', 'status'])
    rows = [dict(group=r['group'], seed=r['seed'], test=family,
                 targets_per_group=r.get('targets_per_group'), seconds=r.get('seconds'),
                 eligibility=r.get('eligibility'), pair_coverage=r.get('pair_coverage'), **test)
            for r in result['runs'] for family, test in r['tests'].items()]
    write_csv(folder / 'seed_runs.csv', rows, ['group', 'seed', 'test', 'status'])


def source_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def evaluate_privacy(
    *,
    train,
    control,
    synthetic,
    budget=500,
    seeds=None,
    config_path=None,
    near_threshold=None,
    output=None,
    progress=False,
    privacy_options=None,
    data_specs=None,
    ):  
    """Öffentliche Schnittstelle für das zentrale Evaluationsskript.

    train/control/synthetic: CSV-Pfade (str oder pathlib.Path), keine DataFrames.
    train=Org, control=zurückgehaltene Testdaten, synthetic=Synth.
    output: optionale Ausgabebasis; None bedeutet keine Dateien schreiben.
    Rückgabe: dict mit summary, copies, runs, preprocessing (bei erfolgreichem
    Einlesen), metadata und output_folder. Der Score steht unter
    result['summary']['privacy_score']. Vor Nutzung technically_complete prüfen.
    Abgefangene technische Fehler liefern Status/Gründe und Fehlerersatz 0.
    Es gibt keine Konsolenausgabe, solange progress=False bleibt.
    """
    started = time.perf_counter()
    planned_seeds = list(SEEDS)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version())
    result = dict(summary=aggregate(None, [], planned_seeds), copies=None, runs=[])
    folder = None
    try:
        candidate_seeds = list(SEEDS) if seeds is None else list(seeds)
        args = argparse.Namespace(
            train=Path(train),
            control=Path(control),
            synthetic=Path(synthetic),
            budget=budget,
            seeds=candidate_seeds,
            config=(
                Path(config_path)
                if config_path is not None
                else None
            ),
            near_threshold=near_threshold,
            privacy_options=privacy_options,
        )
        config, paths, metadata = setup(args)
        planned_seeds = args.seeds
        metadata['source_sha256'] = source_hashes()
        metadata['effective_config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        metadata['input_sha256'] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
        metadata["data_specs"] = data_specs
        data, metadata["data_checks"] = load_data(paths, config, data_specs,)
        result = evaluate(data, config, budget, planned_seeds, progress=progress)
    except Exception as exc:
        metadata['fatal_error'] = failure(exc)
        result['summary'] = aggregate(None, [], planned_seeds, failure(exc))
    metadata['seconds'] = time.perf_counter() - started
    if output is not None:
        try:
            folder = new_folder(Path(output))
            write_result(folder, result, metadata)
        except Exception as exc:
            metadata['export_error'] = f'{type(exc).__name__}: Export fehlgeschlagen'
            result['summary'] = aggregate(None, [], planned_seeds, metadata['export_error'])
            folder = None
    return dict(result, metadata=metadata, output_folder=str(folder.resolve()) if folder else None)


def run_privacy_eval(
    config,
    synthetic_data_path,
    *,
    config_dir=None,
):
    """Run privacy evaluation using the shared configuration."""
    base = (
        Path(config_dir)
        if config_dir is not None
        else Path.cwd()
    )

    def resolve(value):
        path = Path(value).expanduser()
        return (
            path
            if path.is_absolute()
            else (base / path).resolve()
        )

    train = config.get("real_train_path")
    control = config.get("real_test_path")

    require(
        bool(train) == bool(control),
        "Set both real_train_path and real_test_path, or neither",
    )

    if train:
        train = resolve(train)
        control = resolve(control)
    else:
        directory = resolve(config["real_data_path"])
        train = directory / "real_train.parquet"
        control = directory / "real_test.parquet"

    settings = config.get("privacy", {})

    require(
        isinstance(settings, dict),
        "privacy must be a JSON object",
    )
    require(
        set(settings)
        <= {"budget", "seeds", "config", "progress"},
        "Unknown privacy settings",
    )

    metric_config = dict(settings.get("config", {}))
    spec = config["data_specs"]

    # Keep the original ICU binary-column handling.
    # Explicitly configure any additional categorical features.
    categories = metric_config.get(
        "categorical_columns",
        spec.get("categorical_vars", BINARY),
    )

    require(
        isinstance(categories, list),
        "categorical_columns must be a list",
    )

    aliases = {
        spec.get("target_var"): "In-hospital_death",
        spec.get("identifier_var"): "recordid",
    }

    metric_config["categorical_columns"] = list(
        dict.fromkeys(
            aliases.get(column, column)
            for column in categories
        )
    )

    result = evaluate_privacy(
        train=train,
        control=control,
        synthetic=Path(synthetic_data_path),
        budget=settings.get("budget", 500),
        seeds=settings.get("seeds"),
        privacy_options=metric_config,
        data_specs=spec,
        progress=settings.get("progress", False),
    )

    metrics = (
        resolve(config["evaluation_output_path"])
    )
    metrics.mkdir(parents=True, exist_ok=True)

    details = metrics / "privacy" / "metrics"

    # Preserve the original detailed JSON and CSV exports.
    write_result(
        details,
        result,
        result["metadata"],
    )

    result["output_folder"] = str(details.resolve())

    (metrics / "privacy.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    summary = result["summary"]

    print(
        f"Privacy score: {summary['privacy_score']:.6f}"
    )
    print(f"Score basis: {summary['score_basis']}")
    print(
        "Technically complete: "
        f"{summary['technically_complete']}"
    )

    # The original evaluator catches technical errors and returns a
    # fallback zero. Keep that result in the report, but stop the
    # combined evaluation rather than presenting it as a valid score.
    if not summary["technically_complete"]:
        raise AuditError(
            "Privacy evaluation is not technically complete; "
            "see metrics/privacy.json"
        )

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate privacy using the shared configuration."
    )
    parser.add_argument(
        "experiment_config",
        type=Path,
        help="Path to config.json",
    )
    parser.add_argument(
        "--synth",
        type=Path,
        help="Override synthetic_data_path",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Override evaluation_output_path",
    )
    args = parser.parse_args(argv)

    config_path = args.experiment_config.resolve()

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    if args.synth is not None:
        synthetic_path = args.synth.expanduser().resolve()
    elif config.get("synthetic_data_path"):
        synthetic_path = Path(
            config["synthetic_data_path"]
        ).expanduser()

        if not synthetic_path.is_absolute():
            synthetic_path = config_path.parent / synthetic_path
    else:
        parser.error(
            "Pass --synth or set synthetic_data_path in config.json"
        )

    if args.output_path is not None:
        config["evaluation_output_path"] = str(
            args.output_path.expanduser().resolve()
        )

    try:
        run_privacy_eval(
            config,
            synthetic_path,
            config_dir=config_path.parent,
        )
    except AuditError as exc:
        print(f"Privacy evaluation failed: {exc}")
        return 2

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

