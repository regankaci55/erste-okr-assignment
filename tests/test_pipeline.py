"""Tests that feed the pipeline changed data and check it adapts or stops clearly."""
import copy
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pipeline as p  # noqa: E402

RAW = ROOT / "data" / "raw"
OKR_NAME, JIRA_NAME = "Candidate_OKR_Raw_Export.xlsx", "Candidate_Jira_Issues.csv"


@pytest.fixture
def workspace(tmp_path):
    """A disposable copy of the raw data, with settings pointing at it."""
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(RAW / OKR_NAME, raw / OKR_NAME)
    shutil.copy(RAW / JIRA_NAME, raw / JIRA_NAME)
    settings = copy.deepcopy(p.load_settings())
    settings["files"]["raw_dir"] = str(raw)
    settings["files"]["output_dir"] = str(tmp_path / "out")
    return settings, raw


def edit_okr(raw, change):
    path = raw / OKR_NAME
    frame = pd.read_excel(path, header=None, dtype=str)
    header = frame.index[frame[0] == "OKR"][0]
    change(frame, header, {name: i for i, name in enumerate(frame.iloc[header]) if isinstance(name, str)})
    frame.to_excel(path, header=False, index=False)


def edit_jira(raw, change):
    path = raw / JIRA_NAME
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    frame = change(frame) if change else frame
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def entry(result, check):
    log = result["log"]
    return log[log["check"] == check].iloc[0]


# --- Current data: the invariants the report relies on ---

def test_no_records_lost_and_one_row_per_okr():
    result = p.run(write=False)
    assert len(result["okr"]) == 327 and result["okr"]["okr_key"].is_unique
    assert len(result["jira"]) == 343 and result["jira"]["issue_key_final"].is_unique


def test_every_record_has_a_match_outcome():
    result = p.run(write=False)
    assert result["recon"]["match_category"].value_counts().to_dict() == {
        "Matched": 308, "Jira only": 32, "OKR only": 16, "Matched (issue key repaired)": 3}


def test_hierarchy_fully_connected_and_inferences_flagged():
    okr = p.run(write=False)["okr"]
    assert okr["parent_inferred"].sum() == 7 and not okr["parent_unresolved"].any()
    assert (okr.loc[okr["okr_type"] != "ERSTE GROUP OBJ.", "hierarchy_depth"] > 0).all()


def test_grade_averages_match_an_independent_calculation():
    result = p.run(write=False)
    okr, s = result["okr"], result["sensitivity"]
    grades = okr[(okr["okr_type"] == "Key result") & (okr["cycle"] == 2026)]["grade"]
    row = s[(s["scope"] == "Key results") & (s["cycle"] == 2026) & (s["group"] == "All groups")].iloc[0]
    assert row["avg_with"] == round(sum(grades) / len(grades), 2)
    kept = [g for g in grades if g <= 100]
    assert row["avg_without"] == round(sum(kept) / len(kept), 2)


@pytest.mark.parametrize("raw,expected", [
    ("31/Dez./2026", "2026-12-31"), ("04/Juni/2026", "2026-06-04"), ("24/März/2026", "2026-03-24"),
    ("31.03.2026", "2026-03-31"), ("2026-01-15", "2026-01-15"), ("2026-01-15T09:00:00", "2026-01-15 09:00:00"),
])
def test_date_formats(raw, expected):
    assert p.parse_date(raw, p.month_lookup(p.load_settings())) == pd.Timestamp(expected)


# --- Changed data: claims must follow the data ---

def test_text_follows_changed_grades(workspace):
    settings, raw = workspace
    def change(frame, header, col):
        rows = frame.index[frame[col["Grade"]] == "110"]
        frame.loc[rows[0], col["Grade"]] = "105"
    edit_okr(raw, change)
    text = entry(p.run(settings, write=False), "grade_over_scale")["what_we_found"]
    assert "exactly" not in text and "105" in text and "110" in text


def test_no_grades_over_scale_reports_not_found(workspace):
    settings, raw = workspace
    def change(frame, header, col):
        frame.loc[frame[col["Grade"]] == "110", col["Grade"]] = "90"
    edit_okr(raw, change)
    result = p.run(settings, write=False)
    row = entry(result, "grade_over_scale")
    assert row["count"] == 0 and row["what_we_found"] == "Not found in this run."
    assert (result["sensitivity"]["shift"] == 0).all()


def test_rolling_every_year_forward_leaves_no_old_years(workspace):
    settings, raw = workspace
    shift = lambda text: re.sub(r"\b20(25|26|27)\b", lambda m: str(int(m.group(0)) + 2), text)
    frame = pd.read_excel(raw / OKR_NAME, header=None, dtype=str).map(lambda v: shift(v) if isinstance(v, str) else v)
    frame.to_excel(raw / OKR_NAME, header=False, index=False)
    path = raw / JIRA_NAME
    path.write_text(shift(path.read_text(encoding="utf-8-sig")), encoding="utf-8")
    result = p.run(settings, write=False)
    assert result["summary"]["report_cycle"] == 2028 and result["summary"]["cycles"] == [2028, 2029]
    assert result["summary"]["freshness_reliable_cycles"] == [2028]
    columns = ["problem", "examples", "what_we_found", "why_it_matters", "what_we_did", "technical_detail", "technical_treatment"]
    text = " ".join(result["log"][columns].astype(str).values.ravel())
    assert not re.search(r"\b202[5-7]\b", text)
    assert "cycle 2028" in text


def test_collision_text_follows_data(workspace):
    settings, raw = workspace
    def change(frame):
        frame.loc[frame["OKRKey"] == "X-1050", "IssueKey"] = "DEMO-1051"
        return frame
    edit_jira(raw, change)
    result = p.run(settings, write=False)
    assert "2 keys" in entry(result, "issue_key_collision")["what_we_found"]
    assert result["jira"]["issue_key_final"].is_unique


def test_unreliable_key_pairing_is_not_used_to_repair(workspace):
    settings, raw = workspace
    settings["thresholds"]["min_pairing_share"] = 1.01
    result = p.run(settings, write=False)
    assert not result["jira"]["issue_key_repaired"].any()
    assert any("pairing" in w for w in result["summary"]["warnings"])


def test_settings_change_results(workspace):
    settings, _ = workspace
    settings["thresholds"]["stale_days"] = 10000
    settings["thresholds"]["grade_max"] = 120
    result = p.run(settings, write=False)
    assert not result["okr"]["is_stale"].any()
    assert entry(result, "grade_over_scale")["count"] == 0


def test_new_file_name_found_by_pattern(workspace):
    settings, raw = workspace
    (raw / JIRA_NAME).rename(raw / "Candidate_Jira_Issues_2026-12-01.csv")
    assert p.run(settings, write=False)["summary"]["sources"]["jira"] == "Candidate_Jira_Issues_2026-12-01.csv"


def test_duplicate_okr_row_kept_aside_and_logged(workspace):
    settings, raw = workspace
    def change(frame, header, col):
        frame.loc[len(frame)] = frame.iloc[header + 2].tolist()
    edit_okr(raw, change)
    result = p.run(settings, write=False)
    assert result["okr"]["okr_key"].is_unique and len(result["duplicates"]) == 1
    assert entry(result, "duplicate_okr_rows")["count"] == 1


# --- Changed structure: stop with a message that says what to fix ---

def test_renamed_column_stops_with_explanation(workspace):
    settings, raw = workspace
    edit_jira(raw, lambda f: f.rename(columns={"OKRKey": "OKR Link"}))
    with pytest.raises(p.PipelineError, match=r"OKRKey.*OKR Link"):
        p.run(settings, write=False)


def test_renamed_column_works_after_settings_update(workspace):
    settings, raw = workspace
    edit_jira(raw, lambda f: f.rename(columns={"OKRKey": "OKR Link"}))
    settings["jira_columns"]["map"]["OKR Link"] = settings["jira_columns"]["map"].pop("OKRKey")
    assert len(p.run(settings, write=False)["jira"]) == 343


def test_reordered_and_extra_columns_are_fine(workspace):
    settings, raw = workspace
    def change(frame):
        frame = frame[list(frame.columns[::-1])].copy()
        frame["New field"] = "x"
        return frame
    edit_jira(raw, change)
    assert len(p.run(settings, write=False)["jira"]) == 343


def test_new_status_stops_with_explanation(workspace):
    settings, raw = workspace
    def change(frame):
        frame.loc[0, "Status"] = "On Hold"
        return frame
    edit_jira(raw, change)
    with pytest.raises(p.PipelineError, match=r"on hold.*status_groups"):
        p.run(settings, write=False)


def test_new_month_name_stops_then_works_once_added(workspace):
    settings, raw = workspace
    def change(frame, header, col):
        frame.loc[header + 1, col["Due date"]] = "31/Xyz./2026"
    edit_okr(raw, change)
    with pytest.raises(p.PipelineError, match=r"Xyz.*months"):
        p.run(settings, write=False)
    settings["months"][12].append("xyz")
    assert len(p.run(settings, write=False)["okr"]) == 327


def test_missing_file_stops_with_explanation(workspace):
    settings, raw = workspace
    (raw / JIRA_NAME).unlink()
    with pytest.raises(p.PipelineError, match="No file matching"):
        p.run(settings, write=False)


def test_failed_run_leaves_published_files_untouched(workspace):
    settings, raw = workspace
    out = Path(settings["files"]["output_dir"])
    out.mkdir()
    marker = out / "okr_clean.csv"
    marker.write_text("previous good version")
    edit_jira(raw, lambda f: f.drop(columns=["Status"]))
    with pytest.raises(p.PipelineError):
        p.run(settings)
    assert marker.read_text() == "previous good version"
