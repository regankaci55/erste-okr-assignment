"""OKR + Jira preparation pipeline.

Raw exports in, clean tables and a data-quality log out. Run from the project root:

    .venv/bin/python src/pipeline.py

Principles
  * Business rules live in config/settings.yaml, not here.
  * No record is deleted. Repairs go in separate, flagged columns beside the raw values.
  * Every count and every factual claim in the log is derived from the current data.
  * Blocking problems stop the run before anything is written. Others become warnings.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REQUIRED_STATUS_GROUPS = {"Done", "On track", "At risk", "Not started"}
DONE, NOT_STARTED = "Done", "Not started"


class PipelineError(Exception):
    """A problem that must stop the run. The message says what to fix."""


def load_settings(path=SETTINGS_FILE):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_STATUS_GROUPS - set(settings["status_groups"])
    if missing:
        raise PipelineError(f"settings.yaml: status_groups is missing {sorted(missing)}.")
    return settings


def status_lookup(settings):
    return {raw.strip().lower(): group for group, raws in settings["status_groups"].items() for raw in raws}


def month_lookup(settings):
    return {name.lower(): int(number) for number, names in settings["months"].items() for name in names}


def find_source(settings, key):
    raw_dir = Path(settings["files"]["raw_dir"])
    raw_dir = raw_dir if raw_dir.is_absolute() else ROOT / raw_dir
    pattern = settings["files"][key]
    matches = sorted(raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise PipelineError(f"No file matching '{pattern}' in {raw_dir}. Check 'files' in settings.yaml.")
    return matches[0]


def is_blank(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip()


def parse_date(value, months):
    """Read 31/Dez./2026, 04/Juni/2026, 31.03.2026 and ISO dates. Unreadable values return NaT."""
    if is_blank(value):
        return pd.NaT
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})/([^\W\d_]+)\.?/(\d{4})", text)
    if match:
        month = months.get(match.group(2).lower())
        return pd.Timestamp(int(match.group(3)), month, int(match.group(1))) if month else pd.NaT
    match = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    if match:
        return pd.Timestamp(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return pd.to_datetime(text, errors="coerce")
    return pd.NaT


def date_format(value):
    if is_blank(value):
        return "blank"
    text = str(value).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return "ISO"
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", text):
        return "dd.mm.yyyy"
    return "text month"


def parse_dates(frame, columns, months, source):
    """Parse date columns. A filled value that cannot be read stops the run."""
    for raw_col, new_col in columns.items():
        frame[new_col] = frame[raw_col].map(lambda v: parse_date(v, months))
        bad = frame.loc[frame[new_col].isna() & ~frame[raw_col].map(is_blank), raw_col]
        if not bad.empty:
            raise PipelineError(
                f"{source}: {len(bad)} unreadable value(s) in '{raw_col}', e.g. {sorted(set(bad))[:3]}. "
                "For a new month name, add it under 'months' in settings.yaml.")
    return frame


def check_columns(found, expected, source):
    missing = [c for c in expected if c not in found]
    if missing:
        extra = [c for c in found if isinstance(c, str) and c not in expected]
        raise PipelineError(f"{source}: expected column(s) {missing} not found. Unrecognised columns: {extra}. "
                            "If one was renamed, update the column map in settings.yaml.")


def map_status(series, lookup, source):
    clean = series.fillna("").str.strip().str.lower()
    unknown = sorted(set(clean) - set(lookup))
    if unknown:
        raise PipelineError(f"{source}: unknown status value(s) {unknown}. Add each one under "
                            "'status_groups' in settings.yaml so it is counted deliberately.")
    return clean.map(lookup)


def load_okr(settings):
    path = find_source(settings, "okr_pattern")
    raw = pd.read_excel(path, header=None, dtype=str)
    source = path.name
    marker = settings["okr_columns"]["header_marker"]
    header_rows = raw.index[raw[0].fillna("").str.strip() == marker]
    if len(header_rows) == 0:
        raise PipelineError(f"{source}: no header row whose first cell is '{marker}'. "
                            "Check okr_columns.header_marker in settings.yaml.")
    header_row = header_rows[0]
    headers = raw.iloc[header_row].tolist()
    first_named = next((i for i in range(1, len(headers)) if isinstance(headers[i], str) and headers[i].strip()), None)
    if first_named is None:
        raise PipelineError(f"{source}: the header row has no named columns.")
    indent_cols = [f"indent_{i}" for i in range(first_named)]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = indent_cols + [str(h).strip() for h in headers[first_named:]]
    check_columns(df.columns, settings["okr_columns"]["map"], source)
    df = df[df[indent_cols].notna().any(axis=1)].copy()
    df["source_row"] = df.index + 1
    df["indent_level"] = df[indent_cols].notna().values.argmax(axis=1)
    label = df[indent_cols].ffill(axis=1).iloc[:, -1]
    key_regex = settings["keys"]["okr_key_regex"]
    df["okr_key"] = label.str.extract(key_regex)[0]
    if df["okr_key"].isna().any():
        rows = df.loc[df["okr_key"].isna(), "source_row"].tolist()[:5]
        raise PipelineError(f"{source}: no OKR key found on row(s) {rows}. Check keys.okr_key_regex in settings.yaml.")
    df["okr_title"] = label.str.replace(key_regex, "", regex=True).str.strip()
    df = df.drop(columns=indent_cols).rename(columns=settings["okr_columns"]["map"]).reset_index(drop=True)

    report_cfg = settings["report"]
    if report_cfg.get("report_date"):
        report_date = pd.Timestamp(report_cfg["report_date"])
    else:
        months = month_lookup(settings)
        banner = " ".join(str(v) for v in raw.iloc[:header_row].values.ravel() if not is_blank(v))
        match = re.search(report_cfg["report_date_regex"], banner)
        report_date = parse_date(match.group(1), months) if match else pd.NaT
        if pd.isna(report_date):
            raise PipelineError(f"{source}: could not read a report date from the title '{banner[:80]}'. "
                                "Set report.report_date in settings.yaml.")
    return df, report_date, source


def clean_okr(df, report_date, settings):
    limits, months = settings["thresholds"], month_lookup(settings)
    out = df.copy()
    out["status_group"] = map_status(out["status_raw"], status_lookup(settings), "OKR export")
    out["status"] = out["status_raw"].str.strip().str.title()
    out["cycle"] = pd.to_numeric(out["cycle_raw"], errors="coerce")
    if out["cycle"].isna().any():
        raise PipelineError(f"OKR export: unreadable cycle on {int(out['cycle'].isna().sum())} row(s), "
                            f"e.g. {out.loc[out['cycle'].isna(), 'okr_key'].tolist()[:3]}.")
    out["cycle"] = out["cycle"].astype(int)
    out["grade"] = pd.to_numeric(out["grade_raw"], errors="coerce").round(1)
    out["grade_over_max"] = out["grade"] > limits["grade_max"]
    out["weight"] = pd.to_numeric(out["weight_raw"], errors="coerce")
    out = parse_dates(out, {"start_raw": "start_date", "due_raw": "due_date", "last_update_raw": "last_update"},
                      months, "OKR export")
    out["days_since_update"] = (report_date - out["last_update"]).dt.days
    out["update_after_report"] = out["last_update"] > report_date
    out["is_stale"] = out["days_since_update"] > limits["stale_days"]
    out["multi_group"] = out["group_raw"].fillna("").str.contains(settings["groups"]["separator"], regex=False)
    out["cycle_not_started"] = out["start_date"] > report_date
    out["due_before_start"] = out["due_date"] < out["start_date"]
    out["parent_key"] = out["parent_raw"].str.extract(settings["keys"]["okr_key_regex"])[0]
    return out


def indent_parents(okr):
    """Row order plus indentation gives each row's visual parent in the export."""
    stack, result = {}, []
    for level, key in zip(okr["indent_level"], okr["okr_key"]):
        stack = {k: v for k, v in stack.items() if k < level}
        result.append(stack[max(stack)] if stack else None)
        stack[level] = key
    return pd.Series(result, index=okr.index, dtype=object)


def one_record_per_okr(okr):
    """Keep the latest row per key. Earlier versions are returned for the log, not discarded."""
    keep = okr.sort_values("last_update", ascending=False, kind="stable").drop_duplicates("okr_key").index
    return okr.loc[sorted(keep)].copy(), okr.drop(keep).copy()


def repair_hierarchy(okr, settings):
    top, placeholder = settings["tiers"]["top_level"], settings["keys"]["okr_placeholder"]
    keys = set(okr["okr_key"])
    valid = okr["parent_key"].isin(keys)
    okr["parent_status"] = np.select(
        [okr["okr_type"] == top, okr["parent_key"] == placeholder, okr["parent_key"].isna(), valid],
        ["top level", "placeholder", "blank", "valid"], default="unknown key")
    okr["parent_key_final"] = okr["parent_key"].where(valid)
    broken = okr["parent_status"].isin(["placeholder", "blank", "unknown key"])
    can_infer = broken & okr["indent_parent"].isin(keys)
    okr.loc[can_infer, "parent_key_final"] = okr.loc[can_infer, "indent_parent"]
    okr["parent_inferred"] = can_infer
    okr["parent_unresolved"] = broken & ~can_infer
    agree = int((okr.loc[valid, "indent_parent"] == okr.loc[valid, "parent_key"]).sum())
    return okr, agree, int(valid.sum())


def add_paths(okr):
    lookup = okr.set_index("okr_key")["parent_key_final"].to_dict()

    def path(key):
        chain, seen = [key], {key}
        while pd.notna(lookup.get(chain[-1])):
            parent = lookup[chain[-1]]
            if parent in seen:
                return None
            chain.append(parent)
            seen.add(parent)
        return " > ".join(reversed(chain))

    okr["hierarchy_path"] = okr["okr_key"].map(path)
    okr["hierarchy_depth"] = okr["hierarchy_path"].str.count(">")
    okr["top_objective"] = okr["hierarchy_path"].str.split(" > ").str[0]
    return okr


def group_bridge(okr, settings):
    """One row per OKR and group, for team views only. Team rows do not sum to the total."""
    cfg = settings["groups"]
    bridge = okr[["okr_key", "group_raw", "multi_group"]].copy()
    bridge["group"] = bridge["group_raw"].fillna(cfg["unassigned_label"]).str.split(cfg["separator"], regex=False)
    bridge = bridge.explode("group")
    bridge["group"] = bridge["group"].str.strip()
    return bridge[["okr_key", "group", "multi_group"]]


def clean_jira(okr, report_date, settings):
    path = find_source(settings, "jira_pattern")
    j = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    raw_rows = len(j)
    check_columns(j.columns, settings["jira_columns"]["map"], path.name)
    j = j.rename(columns=settings["jira_columns"]["map"])
    j["source_row"] = j.index + 2
    keys, top = settings["keys"], settings["tiers"]["top_level"]
    months = month_lookup(settings)
    j = parse_dates(j, {"created_raw": "created", "updated_raw": "updated", "target_start_raw": "target_start",
                        "target_end_raw": "target_end", "resolution_raw": "resolution"}, months, path.name)
    j["story_points"] = pd.to_numeric(j["story_points_raw"], errors="coerce")
    j["status_group"] = map_status(j["status_raw"], status_lookup(settings), path.name)
    j["status"] = j["status_raw"].str.strip().str.title()
    if j["okr_key"].dropna().duplicated().any():
        dupes = j.loc[j["okr_key"].duplicated(keep=False) & j["okr_key"].notna(), "okr_key"].unique().tolist()[:3]
        raise PipelineError(f"{path.name}: one OKR key on several Jira rows, e.g. {dupes}. "
                            "The one-to-one link no longer holds, so the model needs reviewing.")

    # An issue key should be unique. Repair collisions only, and only if the key pairing is reliable.
    implied = keys["jira_prefix"] + j["okr_key"].str.replace(keys["okr_prefix"], "", regex=False)
    j["issue_key_collision"] = j["issue_key"].duplicated(keep=False)
    clean_links = j["okr_key"].notna() & ~j["issue_key_collision"]
    pairing_share = float((implied[clean_links] == j.loc[clean_links, "issue_key"]).mean()) if clean_links.any() else 0.0
    j["issue_key_final"] = j["issue_key"]
    j["issue_key_repaired"] = False
    if pairing_share >= settings["thresholds"]["min_pairing_share"]:
        repair = j["issue_key_collision"] & j["okr_key"].notna() & (implied != j["issue_key"]) & ~implied.isin(j["issue_key"])
        j.loc[repair, "issue_key_final"] = implied[repair]
        j.loc[repair, "issue_key_repaired"] = True
    j["issue_key_unresolved"] = j["issue_key_final"].duplicated(keep=False)

    linked = j[["okr_key"]].merge(
        okr[["okr_key", "status_raw", "group_raw", "cycle", "due_date", "start_date", "parent_key_final"]].rename(
            columns={"okr_key": "match_key", "status_raw": "okr_status", "group_raw": "okr_group", "cycle": "okr_cycle",
                     "due_date": "okr_due", "start_date": "okr_start", "parent_key_final": "okr_parent"}),
        how="left", left_on="okr_key", right_on="match_key")
    for column in ["okr_status", "okr_group", "okr_cycle", "okr_due", "okr_parent"]:
        j[column] = linked[column].values
    j["matched"] = linked["match_key"].notna().values
    j["okr_key_not_in_okr"] = j["okr_key"].notna() & ~j["matched"]
    j["is_unaligned"] = j["okr_key"].isna()
    j["group_final"] = j["group_raw"].fillna(j["okr_group"]).fillna(settings["groups"]["unassigned_label"])
    j["group_backfilled"] = j["group_raw"].isna() & j["okr_group"].notna()
    clean = lambda s: s.fillna("").str.strip().str.lower()
    j["status_conflict"] = j["matched"] & (clean(j["status_raw"]) != clean(j["okr_status"]))
    j["cycle"] = j["okr_cycle"].fillna(j["target_start"].dt.year).astype(int)

    final_keys = set(j["issue_key_final"])
    tier_keys = {keys["jira_prefix"] + k.replace(keys["okr_prefix"], "", 1) for k in okr.loc[okr["okr_type"] == top, "okr_key"]}
    j["parent_status"] = np.select(
        [j["parent_key"].isna(), j["parent_key"] == keys["jira_placeholder"], j["parent_key"].isin(final_keys), j["parent_key"].isin(tier_keys)],
        ["blank", "placeholder", "valid", "OKR tool only"], default="unknown key")
    j["end_before_start"] = j["target_end"] < j["target_start"]
    j["target_end_differs_from_okr"] = j["matched"] & j["target_end"].notna() & (j["target_end"] != j["okr_due"])
    j["target_end_final"] = pd.to_datetime(j["okr_due"].where(j["matched"])).fillna(j["target_end"])
    j["target_end_backfilled"] = j["target_end"].isna() & j["target_end_final"].notna()
    j["final_end_before_start"] = j["target_end_final"] < j["target_start"]
    j["updated_before_created"] = j["updated"] < j["created"]
    j["created_after_report"] = j["created"] > report_date
    j["updated_after_report"] = j["updated"] > report_date
    j["resolved_after_report"] = j["resolution"] > report_date
    j["done_no_resolution"] = (j["status_group"] == DONE) & j["resolution"].isna()
    j["is_overdue"] = (j["target_end_final"] < report_date) & (j["status_group"] != DONE) & ~j["final_end_before_start"]
    return j, raw_rows, path.name, pairing_share


def reconcile(okr, jira, settings):
    """One row per OKR and per unlinked Jira issue, so every record has a named outcome."""
    top = settings["tiers"]["top_level"]
    linked = okr[["okr_key", "okr_type", "cycle", "group_raw"]].merge(
        jira.loc[jira["matched"], ["okr_key", "issue_key_final", "status_conflict", "issue_key_repaired"]],
        how="left", on="okr_key")
    missing = linked["issue_key_final"].isna()
    repaired = linked["issue_key_repaired"].eq(True)
    linked["match_category"] = np.select([missing, repaired], ["OKR only", "Matched (issue key repaired)"], default="Matched")
    linked["explanation"] = np.select(
        [missing & (linked["okr_type"] == top), missing, repaired],
        [f"Expected: the '{top}' tier is not tracked in Jira",
         "Unexpected: this OKR tier normally has a Jira issue; the owner should check",
         "Issue key was shared with other rows; the link holds because the OKR key is unique"],
        default="Linked on OKR key")
    unlinked = jira.loc[~jira["matched"], ["issue_key_final", "issue_type", "cycle", "group_final", "okr_key", "labels", "parent_key"]].rename(
        columns={"issue_type": "okr_type", "group_final": "group_raw"})
    unlinked["match_category"] = "Jira only"
    unlinked["explanation"] = np.where(
        unlinked["okr_key"].notna(), "OKR key is not in the OKR export",
        "No OKR key (label: " + unlinked["labels"].fillna("none") + "); " +
        np.where(unlinked["parent_key"].isna(), "no parent either, so a link cannot be inferred", "has a parent; review manually"))
    return pd.concat([linked, unlinked.drop(columns=["labels", "parent_key"])], ignore_index=True)


def metric_checks(okr, settings):
    types = settings["metric_types"]
    kr = okr[okr["okr_type"] == settings["tiers"]["key_result"]].copy()
    num = lambda s: pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")
    start, current, end = num(kr["metric_start"]), num(kr["metric_current"]), num(kr["metric_end"])
    span = end - start
    implied = ((current - start) / span.where(span != 0) * 100).clip(0, 100)
    gap = (implied - kr["grade"]).abs()
    has_pct = kr[["metric_start", "metric_current", "metric_end"]].astype(str).apply(lambda c: c.str.contains("%"))
    return {
        "kr_total": len(kr),
        "missing_current": int(current.isna().sum()),
        "zero_range": int((span == 0).sum()),
        "decrease_wrong_way": int(((kr["metric_type"] == types["decrease"]) & (end > start)).sum()),
        "increase_wrong_way": int(((kr["metric_type"] == types["increase"]) & (end < start)).sum()),
        "boolean_non_binary": int(((kr["metric_type"] == types["boolean"]) & end.notna() & ~end.isin([0, 1, 100])).sum()),
        "mixed_units": int(has_pct.sum(axis=1).between(1, 2).sum()),
        "comparable": int(gap.notna().sum()),
        "gap_over_20": int((gap > 20).sum()),
        "corr": None if gap.notna().sum() < 3 else round(float(implied.corr(kr["grade"])), 2),
    }


def grade_sensitivity(okr, bridge, settings):
    """Average grade with and without grades above the scale, per cycle, scope and group."""
    kr = settings["tiers"]["key_result"]
    limits = settings["thresholds"]
    teams = bridge.merge(okr, on="okr_key")
    data = pd.concat([
        okr[okr["okr_type"] == kr].assign(group="All groups", scope="Key results"),
        teams[teams["okr_type"] == kr].assign(scope="Key results"),
        okr.assign(group="All groups", scope="All OKR levels"),
        teams.assign(scope="All OKR levels"),
    ], ignore_index=True)
    rows = []
    for (scope, cycle, group), g in data.groupby(["scope", "cycle", "group"]):
        over = g[g["grade_over_max"]]
        rows.append({"scope": scope, "cycle": cycle, "group": group, "n_all": len(g), "n_over_scale": len(over),
                     "avg_with": round(g["grade"].mean(), 2),
                     "avg_without": round(g.loc[~g["grade_over_max"], "grade"].mean(), 2),
                     "over_scale_keys": ", ".join(over["okr_key"]),
                     "over_scale_values": ", ".join(str(v) for v in sorted(over["grade"].unique()))})
    s = pd.DataFrame(rows)
    s["shift"] = (s["avg_with"] - s["avg_without"]).round(2)
    s["material"] = s["shift"].abs() >= limits["material_points"]
    teams_only = s["group"] != "All groups"
    for column, source in [("rank_with", "avg_with"), ("rank_without", "avg_without")]:
        s[column] = s[teams_only].groupby(["scope", "cycle"])[source].rank(ascending=False, method="min")
    s["rank_changed"] = s["rank_with"].notna() & (s["rank_with"] != s["rank_without"])

    def explain(row):
        if not row["material"]:
            return ""
        note = (f"Removing {row['n_over_scale']} grade(s) above {limits['grade_max']} ({row['over_scale_keys']}) "
                f"moves the average from {row['avg_with']} to {row['avg_without']}.")
        if row["rank_changed"]:
            note += f" Rank moves from {int(row['rank_with'])} to {int(row['rank_without'])}."
        return note

    s["explanation"] = s.apply(explain, axis=1)
    return s


def examples(values, n=3):
    return ", ".join(str(v) for v in list(values)[:n])


def counts_text(series, n=5):
    return ", ".join(f"{k}: {v}" for k, v in Counter(series.dropna()).most_common(n)) or "none"


def by_cycle(frame):
    counts = frame.groupby("cycle").size()
    return ", ".join(f"{cycle}: {count}" for cycle, count in counts.items()) or "none"


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def dq_log(okr, duplicates, jira, metrics, agree, valid, report_date, settings):
    """One row per check. Plain-language columns for readers, technical columns for analysts.

    Every count and claim is computed here, so the text cannot outlive the data it describes.
    """
    limits = settings["thresholds"]
    top = settings["tiers"]["top_level"]
    keys = settings["keys"]
    O, J = len(okr), len(jira)
    log = []

    def add(check, priority, problem, source, count, population, sample, found, why, did, detail, treatment):
        none = "Not found in this run."
        log.append({"check": check, "priority": priority, "problem": problem, "source": source, "count": int(count),
                    "population": int(population),
                    "what_we_found": found if count else none, "why_it_matters": why if count else "",
                    "what_we_did": did, "examples": sample if count else "",
                    "technical_detail": detail if count else none, "technical_treatment": treatment})

    # --- High: these change a headline number or the hierarchy ---
    future = okr[okr["cycle_not_started"]]
    progressing = future[future["status_group"] != NOT_STARTED]
    later = ", ".join(map(str, sorted(progressing["cycle"].unique()))) or "later cycles"
    add("future_cycle_progress", "High", "Next year's OKRs already show progress", "OKR", len(progressing), len(future),
        examples(progressing["okr_key"]),
        f"{len(progressing)} of {len(future)} OKRs for {later} start after the export date, yet are marked Done, Blocked, In Progress and so on.",
        "Mixed in with this year, they would distort the picture of current delivery.",
        f"The report opens on {report_date.year} and shows later years separately, as plans. Ask the OKR owner what these statuses mean.",
        f"Rows start after the {report_date:%d %b %Y} export yet show progress statuses ({counts_text(progressing['status'])}). "
        f"Cycles affected: {by_cycle(progressing)}.",
        f"Default the report to cycle {report_date.year} and show later cycles separately. Keep all rows.")

    late = jira[jira["resolved_after_report"] | jira["updated_after_report"] | jira["created_after_report"]]
    late_current = late[late["cycle"] == report_date.year]
    only_resolution = bool(len(late_current)) and bool(late_current["resolved_after_report"].all()
                                                       and not late_current["updated_after_report"].any()
                                                       and not late_current["created_after_report"].any())
    add("jira_dates_after_export", "High", "Jira dates in the future", "Jira", len(late), J,
        examples(late["issue_key_final"]),
        f"{len(late)} issues were created, updated or finished after the date the file was exported, which cannot have happened yet.",
        "We cannot measure how long work takes or when it was finished.",
        "Kept and flagged. No 'time to finish' measure is built. "
        + (f"In {report_date.year} only the finish date is affected; ask whether it is filled in automatically with the planned end date."
           if only_resolution else "Ask the Jira owner how these dates are filled in."),
        f"{int(jira['resolved_after_report'].sum())} of {int(jira['resolution'].notna().sum())} resolution dates, "
        f"{int(jira['updated_after_report'].sum())} updated and {int(jira['created_after_report'].sum())} created dates are after the export. "
        f"Affected rows by cycle: {by_cycle(late)}.",
        "Flag and keep; build no cycle-time KPIs.")

    back_order = jira[jira["updated_before_created"]]
    linked = jira[jira["matched"]].merge(okr[["okr_key", "last_update"]], on="okr_key")
    same_update = int((linked["updated"].dt.normalize() == linked["last_update"]).sum())
    affected = sorted(int(c) for c in back_order["cycle"].unique())
    clean_cycles = sorted(set(okr["cycle"]) - set(affected))
    shared = len(linked) > 0 and same_update / len(linked) >= 0.99
    clean_text = ", ".join(map(str, clean_cycles)) or "no year"
    add("updated_before_created", "High", "Issues updated before they were created", "Jira", len(back_order), J,
        examples(back_order["issue_key_final"]),
        f"{len(back_order)} issues, all in {', '.join(map(str, affected)) or 'none'}, have a last-updated date earlier than their created date.",
        "The update dates cannot be trusted for those years."
        + (" Jira's updated date is also a copy of the OKR file's, so it cannot be used to double-check it." if shared else ""),
        f"'Not updated in {limits['stale_days']}+ days' is shown only for {clean_text}. Ask how the export produces these dates.",
        f"By cycle: {by_cycle(back_order)}. The Jira updated date equals the OKR last-update date on {same_update} of {len(linked)} matched rows.",
        f"Publish freshness only for cycles without this problem ({clean_text})" + (" and label it as one source." if shared else "."))

    over = okr[okr["grade_over_max"]]
    values = sorted(over["grade"].unique())
    value_text = f"exactly {values[0]:g}" if len(values) == 1 else (f"between {values[0]:g} and {values[-1]:g}" if values else "")
    not_done = over[over["status_group"] != DONE]
    add("grade_over_scale", "High", f"Grades above {limits['grade_max']}", "OKR", len(over), O,
        examples(over["okr_key"]),
        f"{len(over)} grades are {value_text}"
        + (f", and {len(not_done)} of them are on work that is not finished." if len(not_done) else ", all on finished work."),
        f"If {limits['grade_max']} is the top of the scale these are probably errors, and they push averages up.",
        f"Kept. The main average leaves them out and the average with them is shown beside it. Ask the OKR owner whether grades above {limits['grade_max']} are allowed.",
        f"All {len(over)} are {value_text}. {len(not_done)} of {len(over)} are not marked Done ({counts_text(not_done['status'])}).",
        "Report averages with and without them, and explain the exclusion wherever the shift is material.")

    done_low = okr[(okr["status_group"] == DONE) & (okr["grade"] < limits["low_grade"])]
    start_high = okr[(okr["status_group"] == NOT_STARTED) & (okr["grade"] >= limits["low_grade"])]
    means = okr.groupby("status_group")["grade"].mean().round(1)
    done_mean, start_mean = means.get(DONE, float("nan")), means.get(NOT_STARTED, float("nan"))
    add("status_grade_conflict", "High", "Status and grade disagree", "OKR", len(done_low) + len(start_high), O,
        examples(pd.concat([done_low, start_high])["okr_key"]),
        f"{len(done_low)} finished OKRs have a grade under {limits['low_grade']}, and {len(start_high)} OKRs that have not started have {limits['low_grade']} or more.",
        f"Finished OKRs average {done_mean} and not-started ones {start_mean}, "
        + ("so a grade does not show how far along the work is." if done_mean <= start_mean else "so a grade only partly shows how far along the work is."),
        "Both are shown and never combined. Grades are labelled 'indicative'. The owner should define what a grade means.",
        f"Average grade by status group: Done {done_mean}, not started {start_mean}.",
        "Show both signals; list conflicts for owner review; define whether Done requires a minimum grade.")

    m = metrics
    wrong_way = m["decrease_wrong_way"] + m["increase_wrong_way"]
    add("kr_metric_vs_grade", "High", "Grades do not match the Key Result numbers", "OKR", m["gap_over_20"], m["comparable"],
        f"Correlation of implied progress with grade: {m['corr']}",
        f"Each Key Result has a start, current and target value. For {m['gap_over_20']} of {m['comparable']}, "
        "the progress those numbers imply is more than 20 points away from the grade.",
        "Grades cannot be checked or recalculated from the numbers"
        + (f". Also, {wrong_way} have a target that points the wrong way, such as a goal to reduce something with a target above where it started." if wrong_way else "."),
        "The grade is used as given, with a note that it cannot be checked. The OKR tool should validate these fields when they are entered.",
        f"Correlation {m['corr']}. Also: {m['zero_range']} with start equal to target, {m['decrease_wrong_way']} Decrease with a target above the start, "
        f"{m['increase_wrong_way']} Increase with a target below the start, {m['boolean_non_binary']} Boolean with a non-binary target, "
        f"{m['mixed_units']} mixing % and plain numbers.",
        "Keep the reported grade as the measure and add entry validation to the OKR tool.")

    okr_bad = okr[okr["parent_status"].isin(["placeholder", "blank", "unknown key"])]
    jira_bad = jira[jira["parent_status"].isin(["placeholder", "unknown key"])]
    unresolved = int(okr["parent_unresolved"].sum())
    add("broken_parents", "High", "Missing parent links", "OKR + Jira", len(okr_bad) + len(jira_bad), O + J,
        examples(okr_bad["okr_key"]),
        f"{len(okr_bad)} OKRs and {len(jira_bad)} Jira issues have a blank or dummy parent.",
        "Without a parent, an item cannot be counted under the objective it belongs to.",
        "OKR parents were filled in from the file's indentation, which matched "
        + (f"every one of the {valid} links we could check. " if agree == valid else f"{agree} of the {valid} links we could check. ")
        + "They are marked 'inferred' for owners to confirm"
        + (f"; {unresolved} could not be filled." if unresolved else ".") + " Jira parents are not used in the report.",
        f"OKR: {counts_text(okr_bad['parent_status'])}. Jira: {counts_text(jira_bad['parent_status'])}. {unresolved} OKR links could not be inferred.",
        f"Infer from indentation (agrees with {agree} of {valid} valid links); store in a separate flagged column.")

    add("duplicate_okr_rows", "High", "The same OKR listed twice", "OKR", len(duplicates), O + len(duplicates),
        examples(duplicates["okr_key"]) if len(duplicates) else "",
        f"{len(duplicates)} extra rows repeat an OKR that is already in the file.",
        "Those OKRs would be counted twice in every number.",
        "The latest row is kept. The others are saved to okr_duplicates.csv for review.",
        "Duplicate OKR keys.", "Keep the most recently updated row.")

    # --- Medium: these affect some views ---
    separator = settings["groups"]["separator"]
    multi_jira = jira["group_raw"].fillna("").str.contains(separator, regex=False)
    combined = okr.loc[okr["multi_group"], "group_raw"]
    add("multi_group", "Medium", "Two teams in one cell", "OKR + Jira", int(okr["multi_group"].sum() + multi_jira.sum()), O + J,
        examples(okr.loc[okr["multi_group"], "okr_key"]),
        f"{int(okr['multi_group'].sum())} OKRs and {int(multi_jira.sum())} Jira issues name more than one team"
        + (f", such as '{combined.iloc[0]}'." if len(combined) else "."),
        "Read as written, this looks like an extra team, and the real teams lose these items.",
        "Each item is shown under every team it names. Totals still count it once, so team figures do not add up to the total.",
        f"OKR: {int(okr['multi_group'].sum())}, Jira: {int(multi_jira.sum())}.",
        "Split with a bridge table; totals count each OKR once.")

    collided = jira[jira["issue_key_collision"]]
    collisions = collided.groupby("issue_key").size()
    repaired = set(jira.loc[jira["issue_key_repaired"], "issue_key_final"])
    add("issue_key_collision", "Medium", "One Jira key used for different issues", "Jira", len(collided), J,
        examples(collisions.index),
        f"{plural(len(collisions), 'key')} {'is' if len(collisions) == 1 else 'are'} used on {len(collided)} different issues.",
        "Linking the files by this key would attach work to the wrong goal.",
        f"The files are linked by the OKR key instead, which is unique. {int(jira['issue_key_repaired'].sum())} keys were corrected "
        f"using the usual pattern ({keys['jira_prefix']}n belongs to {keys['okr_prefix']}n), and flagged.",
        f"{plural(len(collisions), 'key')} shared across {len(collided)} rows with different OKR keys. {int(jira['issue_key_repaired'].sum())} repaired, "
        f"{int(jira['issue_key_unresolved'].sum())} unresolved. {int(jira['parent_key'].isin(repaired).sum())} child links depend on a repaired key.",
        f"Restore the issue key from the pairing only while it holds on at least {limits['min_pairing_share']:.0%} of issues.")

    conflicts = jira[jira["status_conflict"]]
    pairs = ("Jira '" + conflicts["status"] + "' vs OKR '" + conflicts["okr_status"].fillna("").str.strip() + "'")
    top_pair = Counter(pairs).most_common(1)
    add("status_conflict", "Medium", "Jira and the OKR file disagree on status", "OKR + Jira", len(conflicts), int(jira["matched"].sum()),
        examples(conflicts["okr_key"]),
        f"{len(conflicts)} of {int(jira['matched'].sum())} linked items" + (f", most often {top_pair[0][0]}." if top_pair else "."),
        "The share of finished work depends on which file you ask.",
        "The report uses the OKR status. The disagreements are listed for owners to fix.",
        f"Most common: {counts_text(pairs, 3)}.",
        "Use the OKR status for OKR KPIs; track the conflict rate on every refresh.")

    no_res = jira[jira["done_no_resolution"]]
    add("done_no_resolution", "Medium", "Finished, but no finish date", "Jira", len(no_res), int((jira["status_group"] == DONE).sum()),
        examples(no_res["issue_key_final"]),
        f"{len(no_res)} of {int((jira['status_group'] == DONE).sum())} finished Jira issues have no finish date.",
        "We cannot tell when that work was completed.",
        "Still counted as Done, and flagged. No date is made up.",
        "Status Done with a blank resolution date.", "Keep as Done; do not impute a date.")

    okr_ends = okr[okr["due_before_start"]]
    jira_ends = jira[jira["end_before_start"]]
    differs = jira[jira["target_end_differs_from_okr"]]
    add("end_before_start", "Medium", "End date before start date", "OKR + Jira", len(okr_ends) + len(jira_ends), O + J,
        examples(jira_ends["issue_key_final"]),
        f"{len(jira_ends)} Jira issues and {len(okr_ends)} OKRs end before they start.",
        "Overdue counts and durations would be wrong.",
        "Where Jira's end date is blank or differs, the OKR due date is used, and flagged. Items that still end before they start are left out of overdue counts only.",
        f"Jira differs from the OKR due date on {len(differs)} rows and is blank on {int(jira['target_end_backfilled'].sum())} more.",
        "Use the OKR due date where Jira is blank or differs; exclude dates that stay impossible from overdue counts.")

    unaligned = jira[jira["is_unaligned"]]
    add("unaligned_work", "Medium", "Jira work with no OKR", "Jira", len(unaligned), J,
        examples(unaligned["issue_key_final"]),
        f"{len(unaligned)} issues have no link to any OKR"
        + (f"; all are {unaligned['issue_type'].iloc[0].lower()}s labelled '{unaligned['labels'].iloc[0]}'."
           if len(unaligned) and unaligned["issue_type"].nunique() == 1 and unaligned["labels"].nunique() == 1
           else f". Types: {counts_text(unaligned['issue_type'])}. Labels: {counts_text(unaligned['labels'])}."),
        "This work is invisible in OKR reporting, so teams look less busy than they are, and nobody can see whether it serves a goal.",
        "Shown as its own group, not forced into a match. It is one of the report's five KPIs.",
        f"Types: {counts_text(unaligned['issue_type'])}. Labels: {counts_text(unaligned['labels'])}.",
        "Keep as an explicit Unaligned category; report its share by team.")

    bad_ref = jira[jira["okr_key_not_in_okr"]]
    add("unknown_okr_reference", "Medium", "Jira points to an OKR that is not in the file", "Jira", len(bad_ref), J,
        examples(bad_ref["okr_key"]) if len(bad_ref) else "",
        f"{len(bad_ref)} issues name an OKR key that does not exist in the OKR file.",
        "The work looks linked but counts towards nothing.",
        "Reported as 'Jira only' with this reason. The owner should correct the key.",
        "OKR key not found in the OKR export.", "Report as Jira only; owner corrects the key.")

    okr_formats = pd.concat([okr[c] for c in ["start_raw", "due_raw", "last_update_raw"]]).map(date_format)
    jira_formats = pd.concat([jira[c] for c in ["target_start_raw", "target_end_raw"]]).map(date_format)
    names = sorted({m.lower() for c in ["start_raw", "due_raw", "last_update_raw"] for v in okr[c].dropna() for m in re.findall(r"[^\W\d_]+", str(v))})
    odd = int((okr_formats == "dd.mm.yyyy").sum() + jira_formats.isin(["dd.mm.yyyy", "text month"]).sum())
    odd_example = okr.loc[okr["start_raw"].map(date_format) == "dd.mm.yyyy", "start_raw"]
    add("mixed_date_formats", "Medium", "Dates written in different ways", "OKR + Jira", odd, O + J,
        examples(odd_example),
        f"{odd} dates are written differently from the rest of their column"
        + (f", such as {odd_example.iloc[0]}" if len(odd_example) else "")
        + f". Month names also vary: {', '.join(n.title() for n in names[:6])}.",
        "A standard date reader can silently blank these dates or mix up day and month.",
        "One reader handles every format. If a date cannot be read, the run stops and names it.",
        f"OKR formats: {counts_text(okr_formats)}. Jira target dates: {counts_text(jira_formats)}.",
        "Single parser for every format; stop the run on an unreadable filled date.")

    # --- Low: tidy-ups and known gaps ---
    tier_only = okr[~okr["okr_key"].isin(jira.loc[jira["matched"], "okr_key"])]
    expected = tier_only[tier_only["okr_type"] == top]
    others = len(tier_only) - len(expected)
    add("okr_not_in_jira", "Low", "OKRs with no Jira issue", "OKR", len(tier_only), O,
        examples(tier_only["okr_key"]),
        f"{len(tier_only)} OKRs have no Jira issue; " + ("all of them" if others == 0 else f"{len(expected)} of them") + " are top-level company objectives.",
        "Expected for the top level, which is not tracked in Jira." + (f" The other {others} should have one." if others else ""),
        "Reported as 'OKR only'." + (" The others are listed for review." if others else ""),
        f"{len(expected)} are the '{top}' tier, {others} are other tiers.",
        f"Treat the '{top}' tier as expected; review any other tier.")

    no_owner = okr[okr["owner"].isna()]
    add("missing_owner", "Low", "No owner", "OKR + Jira", len(no_owner) + int(jira["assignee"].isna().sum()), O + J,
        examples(no_owner["okr_key"]),
        f"{len(no_owner)} OKRs and {int(jira['assignee'].isna().sum())} Jira issues have no owner.",
        "Nobody is responsible for acting on them.",
        "Shown as 'Unassigned' so the team lead can pick them up.",
        f"OKR: {len(no_owner)}, Jira: {int(jira['assignee'].isna().sum())}.", "Show as Unassigned; route to the group lead.")

    blank_group = jira[jira["group_raw"].isna()]
    add("blank_group", "Low", "No team on a Jira issue", "Jira", len(blank_group), J,
        examples(blank_group["okr_key"]),
        f"{len(blank_group)} Jira issues have no team.",
        "They would drop out of every team view.",
        "The team is copied from the linked OKR, and flagged.",
        "Blank Group column.", "Backfill from the linked OKR, flagged.")

    all_status = pd.concat([okr["status_raw"], jira["status_raw"]]).dropna()
    untidy = all_status[all_status != all_status.str.strip()]
    add("status_text_variants", "Low", "The same status written differently", "OKR + Jira", len(untidy), O + J,
        examples(sorted({repr(v) for v in untidy})),
        f"{len(untidy)} rows have extra spaces or different capitals, such as {examples(sorted({repr(v) for v in untidy}), 1)}.",
        "One status would show up as two in the charts.",
        "Spaces and capitals are tidied up. The original text is kept.",
        "Case and spacing variants.", "Trim and normalise; keep the raw value.")

    weights = int(okr["weight"].notna().sum())
    add("sparse_fields", "Low", "Mostly empty columns", "OKR", int(okr["weight"].isna().sum()), O,
        f"Weight filled on {weights}, comment date on {int(okr['last_comment_raw'].notna().sum())}, hierarchy grade on {int(okr['hierarchy_grade_raw'].notna().sum())}",
        f"Weight is filled in on only {weights} of {O} OKRs, and two other columns are completely empty.",
        "A weighted average would rest on too few weights to be fair.",
        "Simple averages are used, and the report says so. Weights on Key Results should be made mandatory.",
        f"Weight filled on {weights}, comment date on {int(okr['last_comment_raw'].notna().sum())}, hierarchy grade on {int(okr['hierarchy_grade_raw'].notna().sum())}.",
        "Use simple averages and state the limitation.")

    frame = pd.DataFrame(log)
    frame.insert(0, "id", [f"DQ-{i:02d}" for i in range(1, len(frame) + 1)])
    frame.insert(7, "pct", (frame["count"] / frame["population"].replace(0, np.nan) * 100).round(1))
    return frame


def run(settings=None, write=True):
    settings = settings or load_settings()
    raw_okr, report_date, okr_source = load_okr(settings)
    okr = clean_okr(raw_okr, report_date, settings)
    okr["indent_parent"] = indent_parents(okr)
    okr, duplicates = one_record_per_okr(okr)
    okr, agree, valid = repair_hierarchy(okr, settings)
    okr = add_paths(okr)
    bridge = group_bridge(okr, settings)
    jira, jira_rows, jira_source, pairing_share = clean_jira(okr, report_date, settings)
    recon = reconcile(okr, jira, settings)
    metrics = metric_checks(okr, settings)
    sensitivity = grade_sensitivity(okr, bridge, settings)
    log = dq_log(okr, duplicates, jira, metrics, agree, valid, report_date, settings)

    # Blocking: wrong totals would follow. These stop the run before anything is written.
    blocking = {
        "okr_rows_preserved": len(okr) + len(duplicates) == len(raw_okr),
        "okr_one_record_per_key": bool(okr["okr_key"].is_unique),
        "jira_rows_preserved": len(jira) == jira_rows,
        "reconciliation_covers_all": len(recon) == len(okr) + int((~jira["matched"]).sum()),
        "report_cycle_in_data": report_date.year in set(okr["cycle"]) or report_date.year < okr["cycle"].min(),
    }
    failed = [name for name, passed in blocking.items() if not passed]
    if failed:
        raise PipelineError(f"Validation failed, nothing was written: {failed}")

    # Warnings: reported, because a few imperfect rows should not stop the whole report.
    warnings = []
    if okr["parent_unresolved"].any():
        warnings.append(f"{int(okr['parent_unresolved'].sum())} OKR parent link(s) could not be inferred.")
    if okr["hierarchy_path"].isna().any():
        warnings.append(f"{int(okr['hierarchy_path'].isna().sum())} OKR(s) sit in a circular hierarchy.")
    if jira["issue_key_unresolved"].any():
        warnings.append(f"{int(jira['issue_key_unresolved'].sum())} Jira row(s) still share an issue key.")
    if pairing_share < settings["thresholds"]["min_pairing_share"]:
        warnings.append(f"Key pairing holds on only {pairing_share:.0%} of issues, so issue keys were not repaired.")
    if okr["grade"].isna().any():
        warnings.append(f"{int(okr['grade'].isna().sum())} OKR(s) have no readable grade and are left out of averages.")

    order_problem = {int(c) for c in jira.loc[jira["updated_before_created"], "cycle"].unique()}
    summary = {
        "report_date": str(report_date.date()),
        "report_cycle": int(report_date.year),
        "cycles": sorted(int(c) for c in okr["cycle"].unique()),
        "freshness_reliable_cycles": sorted(int(c) for c in set(okr["cycle"]) - order_problem),
        "sources": {"okr": okr_source, "jira": jira_source},
        "okr_rows": len(okr), "jira_rows": len(jira), "okr_duplicates": len(duplicates),
        "blocking_checks": blocking, "warnings": warnings,
        "indent_agreement": f"{agree}/{valid}",
        "key_pairing_share": round(pairing_share, 3),
        "match_categories": recon["match_category"].value_counts().to_dict(),
        "metric_checks": metrics,
    }

    if write:
        out = Path(settings["files"]["output_dir"])
        out = out if out.is_absolute() else ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        okr.to_csv(out / "okr_clean.csv", index=False)
        duplicates.to_csv(out / "okr_duplicates.csv", index=False)
        bridge.to_csv(out / "okr_group_bridge.csv", index=False)
        jira.to_csv(out / "jira_clean.csv", index=False)
        recon.to_csv(out / "reconciliation.csv", index=False)
        sensitivity.to_csv(out / "grade_sensitivity.csv", index=False)
        log.to_csv(out / "data_quality_log.csv", index=False)
        log.to_excel(out / "data_quality_log.xlsx", index=False, sheet_name="DQ log")
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"okr": okr, "duplicates": duplicates, "bridge": bridge, "jira": jira, "recon": recon,
            "sensitivity": sensitivity, "log": log, "summary": summary}


def main():
    try:
        result = run()
    except PipelineError as error:
        print(f"RUN STOPPED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
