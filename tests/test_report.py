"""Checks on the report: filters are complete and its figures match the clean data."""
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import build_report as b  # noqa: E402
import pipeline as p  # noqa: E402
import preparation as prep  # noqa: E402


@pytest.fixture(scope="module")
def report():
    settings = p.load_settings()
    return b.Report(p.run(settings, write=False), settings)


def test_every_filter_combination_exists(report):
    views = report.payload()["views"]
    assert len(views) == len(report.cycles) * (len(report.teams) + 1) * 2
    assert "2026|Risk Technology|without" in views and "2027|All teams|with" in views


def test_combined_group_is_not_a_team(report):
    assert not any("," in team for team in report.teams) and len(report.teams) == 8


def test_totals_count_each_okr_once(report):
    assert len(report.okr_slice(2026, b.ALL)) == 165
    assert report.okr_slice(2026, b.ALL)["okr_key"].is_unique


def test_shared_okr_appears_in_both_teams(report):
    for team in ["Digital Channels", "Data Platforms"]:
        assert "X-1001" in set(report.okr_slice(2026, team)["okr_key"])


def test_headline_tile_follows_grade_toggle(report):
    assert ">46.5<" in report.kpis(2026, b.ALL, "without") and ">47.7<" in report.kpis(2026, b.ALL, "with")
    assert ">47.7<" in report.kpis(2026, "Risk Technology", "without") and ">56.6<" in report.kpis(2026, "Risk Technology", "with")


def test_freshness_hidden_only_where_dates_are_unreliable(report):
    views = report.payload()["views"]
    assert views["2026|All teams|without"]["fresh"] is not None
    assert views["2027|All teams|without"]["fresh"] is None
    assert "Not shown" in views["2027|All teams|without"]["kpis"]


def test_tree_contains_every_okr_in_the_cycle(report):
    html = report.tree(2026, b.ALL)
    expected = set(report.okr_slice(2026, b.ALL)["okr_key"])
    assert set(re.findall(r"<b>(X-\d+)</b>", html)) == expected


def test_team_tree_keeps_own_rows_and_marks_context(report):
    html = report.tree(2026, "Digital Channels")
    found = set(re.findall(r"<b>(X-\d+)</b>", html))
    assert set(report.okr_slice(2026, "Digital Channels")["okr_key"]) <= found
    assert "node context" in html


def test_insights_agree_with_data(report):
    o = report.okr_slice(2026, b.ALL)
    first, second, third = report.insights()
    assert first[0].startswith("37% of 2026 OKRs") and "61 of 165" in first[1]
    risk = o[o["status_group"] == "At risk"]
    assert second[0].startswith(f"{int(risk['is_stale'].sum())} of the {len(risk)} OKRs at risk")
    assert third[0].startswith("17 Jira issues")


def test_build_writes_a_self_contained_page(tmp_path):
    settings = p.load_settings()
    settings["files"]["output_dir"] = str(tmp_path / "data")
    path, _ = b.build(settings, tmp_path, pdf=False)
    page = path.read_text(encoding="utf-8")
    assert "__" not in re.sub(r"<script.*?</script>", "", page, flags=re.S).replace("__proto__", "")
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?:', page)
    data = re.search(r'id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(data.replace("<\\/", "</"))["defaults"]["cycle"] == "2026"


def test_charts_stay_inside_cards_when_a_filter_is_revisited(tmp_path):
    """Going to a team and back must not collapse the cards under the charts (needs Chrome)."""
    chrome = b.find_chrome()
    if not chrome:
        pytest.skip("Chrome not installed")
    b.build(p.load_settings(), tmp_path, pdf=False)
    page = next(tmp_path.glob("*.html"))
    probe = """<script>setTimeout(function(){
      var t=document.getElementById('team'),first=t.options[0].value,second=t.options[1].value;
      function go(v){t.value=v;t.dispatchEvent(new Event('change'))}
      go(second);setTimeout(function(){go(first);setTimeout(function(){
        var bad=['c-status','c-grade','c-fresh','c-adhoc'].filter(function(i){
          var e=document.getElementById(i),s=e.querySelector('.main-svg');
          var c=e.closest('.card').getBoundingClientRect();return c.height<200||s.getBoundingClientRect().bottom>c.bottom;});
        var o=document.createElement('pre');o.id='probe';o.textContent='overflow:'+bad.join(',');
        document.body.appendChild(o);},500)},500)},500);</script></body></html>"""
    page.write_text(page.read_text(encoding="utf-8").replace("</body></html>", probe), encoding="utf-8")
    dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--window-size=1400,1800",
                          "--virtual-time-budget=20000", "--dump-dom", page.as_uri() + "#tab=overview"],
                         capture_output=True, text=True, timeout=180).stdout
    assert '<pre id="probe">overflow:</pre>' in dom


@pytest.fixture(scope="module")
def evidence():
    settings = p.load_settings()
    return prep.profile(p.run(settings, write=False), settings)


def test_preparation_is_the_first_tab_and_covers_every_step(report):
    page = report.render()
    assert page.index('data-tab="prep"') < page.index('data-tab="overview"')
    assert all(f'id="step-{n}"' in page for n in range(1, 12))
    assert set(json.loads(re.search(r'id="data" type="application/json">(.*?)</script>', page, re.S).group(1).replace("<\\/", "</"))["prep"]) == \
        {"p-okr-blank", "p-jira-blank", "p-status", "p-grade"}


def test_status_evidence_accounts_for_every_row(evidence):
    rows = evidence["status_rows"]
    assert sum(r["okr"] for r in rows) == evidence["okr_rows"] and sum(r["jira"] for r in rows) == evidence["jira_rows"]
    assert (evidence["labels_raw"], evidence["labels_clean"], evidence["labels_group"]) == (10, 9, 4)


def test_status_what_if_matches_the_kpi(evidence):
    alignment = next(m for m in evidence["moves"] if m[0] == "Alignment")
    assert alignment[1:] == ["Not started", 17, "37%", "27%"]


def test_grade_bins_cover_every_okr_with_no_empty_bin_at_the_end(evidence):
    assert evidence["bins"].sum() == evidence["okr_rows"] and evidence["bins"].iloc[-1] > 0


def test_default_parser_evidence_is_measured():
    frame = pd.DataFrame({"x": ["31.03.2026", "01/Dez./2026", None]})
    assert prep.default_parser_misses(frame["x"]) >= 1


def test_preparation_text_follows_changed_data(tmp_path):
    """No grades above the scale and no shared teams: the text must say so, not repeat old findings."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ["Candidate_OKR_Raw_Export.xlsx", "Candidate_Jira_Issues.csv"]:
        shutil.copy(ROOT / "data" / "raw" / name, raw / name)
    path = raw / "Candidate_OKR_Raw_Export.xlsx"
    frame = pd.read_excel(path, header=None, dtype=str)
    header = frame.index[frame[0] == "OKR"][0]
    columns = {name: i for i, name in enumerate(frame.iloc[header]) if isinstance(name, str)}
    grades = pd.to_numeric(frame.iloc[header + 1:, columns["Grade"]], errors="coerce")
    frame.iloc[header + 1:, columns["Grade"]] = grades.clip(upper=100).astype(str)
    frame.to_excel(path, header=False, index=False)
    settings = copy.deepcopy(p.load_settings())
    settings["files"]["raw_dir"] = str(raw)
    settings["files"]["output_dir"] = str(tmp_path / "out")
    page, _ = prep.render(prep.profile(p.run(settings, write=False), settings))
    assert "No grade is above 100" in page and "all exactly 110" not in page


@pytest.mark.parametrize("width", [1300, 500])
def test_tabs_stay_reachable_on_every_page(tmp_path, width):
    """After opening any tab and scrolling to the bottom, the tabs must still be on screen (needs Chrome)."""
    chrome = b.find_chrome()
    if not chrome:
        pytest.skip("Chrome not installed")
    b.build(p.load_settings(), tmp_path, pdf=False)
    page = next(tmp_path.glob("*.html"))
    probe = """<script>setTimeout(function(){
      var tabs=[].slice.call(document.querySelectorAll('nav button')),nav=document.querySelector('nav'),out=[];
      var tops=tabs.map(function(t){t.click();return Math.round(nav.getBoundingClientRect().top+scrollY)});
      if(new Set(tops).size!==1)out.push('tabs move:'+tops.join(','));
      tabs.forEach(function(t){t.click();window.scrollTo(0,document.body.scrollHeight);
        var r=nav.getBoundingClientRect();if(r.top<0||r.bottom>innerHeight||r.height<20)out.push(t.dataset.tab+' lost');
        var on=['overview','hierarchy'].indexOf(t.dataset.tab)>=0;
        if(document.getElementById('team').disabled===on)out.push(t.dataset.tab+' filter state');
        if(on!==document.getElementById('filter-note').hidden)out.push(t.dataset.tab+' note');});
      var o=document.createElement('pre');o.id='probe';o.textContent='result:'+out.join(';');document.body.appendChild(o);
      },800);</script></body></html>"""
    page.write_text(page.read_text(encoding="utf-8").replace("</body></html>", probe), encoding="utf-8")
    dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--window-size={width},900",
                          "--virtual-time-budget=15000", "--dump-dom", page.as_uri()],
                         capture_output=True, text=True, timeout=180).stdout
    assert '<pre id="probe">result:</pre>' in dom


def test_automation_idea_answers_the_three_questions(report):
    page = report.render()
    start = page.index('id="automation"')
    section = page[start:page.index("</section>", start)]
    for heading in ["Which steps to automate", "How the data stays accurate", "What happens if something goes wrong"]:
        assert heading in section
    assert f"{len(report.summary['blocking_checks'])} checks" in section and "__" not in section
