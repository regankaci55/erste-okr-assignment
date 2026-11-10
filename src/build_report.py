"""Build the management report: runs the pipeline, then writes HTML and PDF to reports/.

    .venv/bin/python src/build_report.py

Every number and sentence is computed here in Python. Each filter combination is calculated
in advance, so the page's small script only swaps between finished views.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import pipeline as p
import preparation
from report_common import (ALL, BASE, BLUE, GRID, INK, MUTED, SECONDARY, SERIOUS, STATUS_COLORS, STATUS_ORDER, STATUS_TEXT,
                           SURFACE, base_layout, esc, fig_json, pct, table)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODES = {"without": "Without", "with": "With"}
FADED = 0.25
CHROME_PATHS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]


def opacities(teams, selected):
    return [1.0 if selected in (ALL, team) else FADED for team in teams]


class Report:
    def __init__(self, result, settings):
        self.settings = settings
        self.result = result
        self.limits = settings["thresholds"]
        self.grade_max = self.limits["grade_max"]
        self.kr = settings["tiers"]["key_result"]
        self.okr = result["okr"]
        self.jira = result["jira"]
        self.log = result["log"]
        self.recon = result["recon"]
        self.summary = result["summary"]
        self.sensitivity = result["sensitivity"]
        self.report_date = pd.Timestamp(self.summary["report_date"])
        self.report_cycle = self.summary["report_cycle"]
        self.cycles = self.summary["cycles"]
        self.teams_okr = result["bridge"].merge(self.okr, on="okr_key")
        cfg = settings["groups"]
        jira_teams = self.jira.assign(group=self.jira["group_final"].str.split(cfg["separator"], regex=False)).explode("group")
        jira_teams["group"] = jira_teams["group"].str.strip()
        self.teams_jira = jira_teams
        self.teams = sorted(set(self.teams_okr["group"]) | set(self.teams_jira["group"]))
        self.without_label = f"Without grades above {self.grade_max}"
        self.with_label = f"With grades above {self.grade_max}"

    # --- Slices -----------------------------------------------------------------
    def okr_slice(self, cycle, team):
        if team == ALL:
            return self.okr[self.okr["cycle"] == cycle]
        return self.teams_okr[(self.teams_okr["cycle"] == cycle) & (self.teams_okr["group"] == team)]

    def jira_slice(self, cycle, team):
        if team == ALL:
            return self.jira[self.jira["cycle"] == cycle]
        return self.teams_jira[(self.teams_jira["cycle"] == cycle) & (self.teams_jira["group"] == team)]

    def grade_row(self, cycle, team):
        group = "All groups" if team == ALL else team
        s = self.sensitivity
        rows = s[(s["scope"] == "Key results") & (s["cycle"] == cycle) & (s["group"] == group)]
        return rows.iloc[0] if len(rows) else None

    def freshness_ok(self, cycle):
        return cycle in self.summary["freshness_reliable_cycles"]

    # --- KPIs -------------------------------------------------------------------
    def labels(self, group):
        raws = [r.title() for r in self.settings["status_groups"][group]]
        return ", ".join(raws[:-1]) + " or " + raws[-1] if len(raws) > 1 else raws[0]

    def tile(self, label, value, note, tone=""):
        return f'<div class="tile {tone}"><div class="tile-label">{esc(label)}</div><div class="tile-value">{esc(value)}</div><div class="tile-note">{esc(note)}</div></div>'

    def kpis(self, cycle, team, mode):
        o, j = self.okr_slice(cycle, team), self.jira_slice(cycle, team)
        n = len(o)
        row = self.grade_row(cycle, team)
        if row is None:
            grade = self.tile("Average Key Result grade", "n/a", "No Key Results in this view")
        else:
            main, other = (row["avg_without"], row["avg_with"]) if mode == "without" else (row["avg_with"], row["avg_without"])
            kept = row["n_all"] - row["n_over_scale"] if mode == "without" else row["n_all"]
            note = (f"Out of {self.grade_max}, leaving out grades above {self.grade_max} (with them: {other:.1f}). {kept} Key Results."
                    if mode == "without" else
                    f"Out of {self.grade_max}, including grades above {self.grade_max} (without them: {other:.1f}). {kept} Key Results.")
            grade = self.tile("Average Key Result grade", f"{main:.1f}", note)
        risk = int((o["status_group"] == "At risk").sum())
        idle = int((o["status_group"] == "Not started").sum())
        stale = int(o["is_stale"].sum())
        loose = int(j["is_unaligned"].sum())
        days = self.limits["stale_days"]
        stale_tile = (self.tile(f"Not updated in {days}+ days", pct(stale, n), f"{stale} of {n} OKRs had no grade update in the {days} days before the export.")
                      if self.freshness_ok(cycle) else
                      self.tile(f"Not updated in {days}+ days", "Not shown", "The update dates for this year are unreliable (see Data quality)."))
        return "".join([
            grade,
            self.tile("OKRs at risk", pct(risk, n), f"{risk} of {n} OKRs are {self.labels('At risk')}."),
            self.tile("OKRs not started", pct(idle, n), f"{idle} of {n} OKRs are {self.labels('Not started')}."),
            stale_tile,
            self.tile("Jira issues with no OKR", pct(loose, len(j)),
                      f"{loose} of {len(j)} work items are not linked to any goal. {int(j['is_overdue'].sum())} Jira items in this view are overdue."),
        ])

    # --- Charts -----------------------------------------------------------------
    def status_chart(self, cycle, team):
        data = self.teams_okr[self.teams_okr["cycle"] == cycle]
        counts = data.groupby(["group", "status_group"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
        share = counts.div(counts.sum(axis=1), axis=0).sort_values("At risk")
        counts = counts.loc[share.index]
        fig = go.Figure(layout=base_layout(350))
        for status in STATUS_ORDER:
            fig.add_bar(
                y=list(share.index), x=(share[status] * 100).round(1).tolist(), name=status, orientation="h",
                marker=dict(color=STATUS_COLORS[status], line=dict(color=SURFACE, width=2), opacity=opacities(share.index, team)),
                text=[f"{v * 100:.0f}%" if v >= 0.1 and team in (ALL, g) else "" for g, v in share[status].items()], textposition="inside", insidetextanchor="middle",
                textangle=0, textfont=dict(color=STATUS_TEXT[status], size=12),
                customdata=counts[status].tolist(), hovertemplate="%{y}<br>" + status + ": %{x:.0f}% (%{customdata} OKRs)<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.3)
        fig.update_xaxes(range=[0, 100], ticksuffix="%", title=None)
        rows = [[g] + [int(counts.loc[g, s]) for s in STATUS_ORDER] + [int(counts.loc[g].sum())] for g in reversed(list(share.index))]
        return fig_json(fig), table(["Team"] + STATUS_ORDER + ["OKRs"], rows, numeric=range(1, 6))

    def grade_chart(self, cycle, team, mode):
        s = self.sensitivity
        rows = s[(s["scope"] == "Key results") & (s["cycle"] == cycle) & (s["group"] != "All groups")].copy()
        main_col, other_col = ("avg_without", "avg_with") if mode == "without" else ("avg_with", "avg_without")
        main_name, other_name = (self.without_label, self.with_label) if mode == "without" else (self.with_label, self.without_label)
        rows = rows.sort_values(main_col)
        names = rows["group"].tolist()
        labels = [f"{g} ({n})" for g, n in zip(names, rows["n_all"])]
        moved = rows["shift"].abs() > 0
        fig = go.Figure(layout=base_layout(350))
        xs, ys = [], []
        for label, a, b in zip(pd.Series(labels)[moved.values], rows.loc[moved, main_col], rows.loc[moved, other_col]):
            xs += [a, b, None]
            ys += [label, label, None]
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=BASE, width=2), showlegend=False, hoverinfo="skip")
        positions = ["middle left" if o > m else "middle right" for m, o in zip(rows[main_col], rows[other_col])]
        fig.add_scatter(x=rows[main_col].tolist(), y=labels, mode="markers+text", name=main_name, text=[f"{v:.1f}" for v in rows[main_col]],
                        textposition=positions, textfont=dict(color=INK, size=12),
                        marker=dict(color=BLUE, size=12, line=dict(color=SURFACE, width=2), opacity=opacities(names, team)),
                        hovertemplate="%{y}<br>" + main_name + ": %{x:.1f}<extra></extra>")
        if moved.any():
            other = rows[moved.values]
            other_labels = [l for l, flag in zip(labels, moved) if flag]
            other_pos = ["middle right" if o > m else "middle left" for m, o in zip(other[main_col], other[other_col])]
            fig.add_scatter(x=other[other_col].tolist(), y=other_labels, mode="markers+text", name=other_name,
                            text=[f"{v:.1f}" for v in other[other_col]], textposition=other_pos, textfont=dict(color=SECONDARY, size=12),
                            marker=dict(color=SURFACE, size=12, line=dict(color=BLUE, width=2), opacity=opacities(other["group"], team)),
                            hovertemplate="%{y}<br>" + other_name + ": %{x:.1f}<extra></extra>")
        overall = self.grade_row(cycle, ALL)
        reference = overall[main_col]
        fig.add_shape(type="line", x0=reference, x1=reference, yref="paper", y0=0, y1=1, line=dict(color=MUTED, width=1), layer="below")
        fig.add_annotation(x=reference, y=0, yref="paper", yanchor="bottom", xanchor="left", xshift=5, showarrow=False,
                           text=f"All teams {reference:.1f}", font=dict(color=MUTED, size=12))
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(categoryorder="array", categoryarray=labels)
        ranked = rows.sort_values(main_col, ascending=False)
        tbl = table(["Team", "Key Results", "Above scale", self.without_label, self.with_label, "Rank without", "Rank with"],
                    [[r["group"], int(r["n_all"]), int(r["n_over_scale"]), f"{r['avg_without']:.2f}", f"{r['avg_with']:.2f}",
                      int(r["rank_without"]), int(r["rank_with"])] for _, r in ranked.iterrows()], numeric=range(1, 7))
        return fig_json(fig), tbl

    def grade_note(self, cycle):
        s = self.sensitivity
        rows = s[(s["scope"] == "Key results") & (s["cycle"] == cycle)]
        changed = rows[(rows["group"] != "All groups") & rows["material"]]
        overall = rows[rows["group"] == "All groups"].iloc[0]
        if overall["n_over_scale"] == 0 and changed.empty:
            return f"No Key Result in this cycle has a grade above {self.grade_max}, so both averages are the same."
        entry = self.log[self.log["check"] == "grade_over_scale"].iloc[0]
        reason = f'{entry["what_we_found"]} {entry["why_it_matters"]}'
        parts = [f"The main figure leaves out grades above {self.grade_max}. Across all teams that moves the average by {abs(overall['shift']):.1f} points, "
                 f"from {overall['avg_with']:.1f} to {overall['avg_without']:.1f}."]
        for _, r in changed.iterrows():
            parts.append(f"It matters most for {r['group']}: leaving out {r['over_scale_keys']} moves it from {r['avg_with']:.1f} to {r['avg_without']:.1f}, "
                         f"and from place {int(r['rank_with'])} to place {int(r['rank_without'])}.")
        parts.append(f"Why: {reason} Nothing is deleted; switch 'Grade average' above to see the other figure.")
        return " ".join(parts)

    def fresh_chart(self, cycle, team):
        if not self.freshness_ok(cycle):
            return None, ""
        data = self.teams_okr[self.teams_okr["cycle"] == cycle]
        grouped = data.groupby("group")["is_stale"].agg(["mean", "sum", "size"]).sort_values("mean")
        fig = go.Figure(layout=base_layout(320, margin=dict(l=8, r=36, t=12, b=36), showlegend=False))
        fig.add_bar(y=list(grouped.index), x=(grouped["mean"] * 100).round(1).tolist(), orientation="h",
                    marker=dict(color=BLUE, line=dict(color=SURFACE, width=2), opacity=opacities(grouped.index, team)),
                    text=[f"{v * 100:.0f}%" for v in grouped["mean"]], textposition="outside", cliponaxis=False,
                    textfont=dict(color=INK, size=12), customdata=list(zip(grouped["sum"].astype(int), grouped["size"])),
                    hovertemplate="%{y}<br>%{x:.0f}% stale (%{customdata[0]} of %{customdata[1]})<extra></extra>")
        fig.update_layout(bargap=0.4)
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        tbl = table(["Team", "Stale OKRs", "OKRs", "Share"],
                    [[g, int(r["sum"]), int(r["size"]), pct(r["sum"], r["size"])] for g, r in grouped[::-1].iterrows()], numeric=(1, 2, 3))
        return fig_json(fig), tbl

    def adhoc_chart(self, cycle, team):
        data = self.teams_jira[self.teams_jira["cycle"] == cycle]
        loose = data[data["is_unaligned"]]
        order = pd.DataFrame({"total": data.groupby("group").size()})
        order["overdue"] = loose[loose["is_overdue"]].groupby("group").size()
        order["other"] = loose[~loose["is_overdue"]].groupby("group").size()
        order = order.fillna(0).astype(int)
        order["loose"] = order["overdue"] + order["other"]
        order = order.sort_values(["loose", "overdue"])
        fig = go.Figure(layout=base_layout(320))
        for column, name, color in [("overdue", "Overdue", SERIOUS), ("other", "Not overdue", BLUE)]:
            fig.add_bar(y=list(order.index), x=order[column].tolist(), name=name, orientation="h",
                        marker=dict(color=color, line=dict(color=SURFACE, width=2), opacity=opacities(order.index, team)),
                        hovertemplate="%{y}<br>" + name + ": %{x}<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.4)
        fig.update_xaxes(dtick=1, rangemode="tozero", title=None)
        tbl = table(["Team", "Issues with no OKR", "Overdue", "All Jira issues", "Share"],
                    [[g, int(r["loose"]), int(r["overdue"]), int(r["total"]), pct(r["loose"], r["total"])] for g, r in order[::-1].iterrows()],
                    numeric=(1, 2, 3, 4))
        return fig_json(fig), tbl

    # --- Hierarchy --------------------------------------------------------------
    def tree(self, cycle, team):
        data = self.okr[self.okr["cycle"] == cycle]
        conflicts = set(self.jira.loc[self.jira["status_conflict"], "okr_key"])
        children = {}
        for key, parent in zip(data["okr_key"], data["parent_key_final"]):
            children.setdefault(parent if pd.notna(parent) else None, []).append(key)
        rows = data.set_index("okr_key")
        sep = self.settings["groups"]["separator"]
        team_of = lambda k: [t.strip() for t in str(rows.loc[k, "group_raw"]).split(sep)]

        def descendants(key):
            found = []
            for child in children.get(key, []):
                found += [child] + descendants(child)
            return found

        def matches(key):
            return team == ALL or team in team_of(key) or any(team in team_of(d) for d in descendants(key))

        def flags(row, key):
            found = []
            if row["parent_inferred"]:
                found.append("Parent filled in by us")
            if row["grade_over_max"]:
                found.append(f"Grade above {self.grade_max}")
            if key in conflicts:
                found.append("Jira shows a different status")
            if pd.isna(row["owner"]):
                found.append("No owner")
            return "".join(f'<span class="flag">{esc(f)}</span>' for f in found)

        def line(key):
            r = rows.loc[key]
            status = r["status_group"]
            own = team == ALL or team in team_of(key)
            sub = descendants(key)
            at_risk = sum(rows.loc[d, "status_group"] == "At risk" for d in sub)
            rollup = f'<span class="rollup">{at_risk} of {len(sub)} below are at risk</span>' if sub else ""
            days = f"{int(r['days_since_update'])}d ago" if self.freshness_ok(cycle) else "n/a"
            return (f'<span class="node {"" if own else "context"}"><span class="n-title"><b>{esc(key)}</b> {esc(r["okr_title"])}{rollup}{flags(r, key)}</span>'
                    f'<span class="n-type">{esc(r["okr_type"])}</span>'
                    f'<span class="n-status"><i style="background:{STATUS_COLORS[status]}"></i>{esc(r["status"])}</span>'
                    f'<span class="n-grade">{r["grade"]:.1f}</span><span class="n-owner">{esc(r["owner"]) or "Unassigned"}</span><span class="n-days">{days}</span></span>')

        def render(key):
            kids = [k for k in children.get(key, []) if matches(k)]
            if not kids:
                return f'<div class="leaf">{line(key)}</div>'
            return f'<details><summary>{line(key)}</summary><div class="kids">{"".join(render(k) for k in kids)}</div></details>'

        roots = [k for k in children.get(None, []) if matches(k)]
        return "".join(render(k) for k in roots) or '<p class="note">No OKRs for this team in this cycle.</p>'

    # --- Written content --------------------------------------------------------
    def insights(self):
        cycle, limits = self.report_cycle, self.limits
        o = self.okr_slice(cycle, ALL)
        j = self.jira_slice(cycle, ALL)
        t = self.teams_okr[self.teams_okr["cycle"] == cycle]
        n = len(o)
        start = pd.Timestamp(cycle, 1, 1)
        elapsed = (self.report_date - start).days / ((pd.Timestamp(cycle + 1, 1, 1) - start).days)
        idle = o[o["status_group"] == "Not started"]
        done = o[o["status_group"] == "Done"]
        idle_team = t.groupby("group")["status_group"].apply(lambda s: (s == "Not started").mean()).sort_values()
        risk = o[o["status_group"] == "At risk"]
        blocked = o[o["status"] == "Blocked"]
        risk_team = t.groupby("group")["status_group"].apply(lambda s: (s == "At risk").mean()).sort_values()
        stale_team = t.groupby("group")["is_stale"].mean().sort_values()
        loose = j[j["is_unaligned"]]
        overdue = j[j["is_overdue"]]
        overdue_loose = int(overdue["is_unaligned"].sum())
        linked_ending = j[(~j["is_unaligned"]) & (j["target_end_final"] >= self.report_date)]
        loose_team = self.teams_jira[(self.teams_jira["cycle"] == cycle) & self.teams_jira["is_unaligned"]].groupby("group").size().sort_values()
        return [
            (f"{pct(len(idle), n)} of {cycle} OKRs have not started, and {elapsed:.0%} of the year has gone.",
             f"{len(idle)} of {n} OKRs are {self.labels('Not started')}. Only {len(done)} ({pct(len(done), n)}) are finished. "
             f"{idle_team.index[-1]} has the most not started ({idle_team.iloc[-1]:.0%} of its OKRs).",
             "Group leads should decide this month, for each one, whether to re-plan it, shrink it or close it, so the year-end result is not a surprise."),
            (f"{len(risk[risk['is_stale']])} of the {len(risk)} OKRs at risk have not been updated in {limits['stale_days']}+ days.",
             f"That includes {int(blocked['is_stale'].sum())} of the {len(blocked)} that are Blocked. {risk_team.index[-1]} has the most at risk ({risk_team.iloc[-1]:.0%}), "
             f"and {stale_team.index[-1]} the most not updated ({stale_team.iloc[-1]:.0%})."
             + (f" {int(risk['owner'].isna().sum())} OKRs at risk have no owner." if risk["owner"].isna().any() else ""),
             "Start with the Blocked ones nobody has updated: give each an owner, and ask for a monthly update on anything at risk."),
            (f"{len(loose)} Jira issues ({pct(len(loose), len(j))}) are not linked to any OKR, and {overdue_loose} of them are overdue.",
             (f"All {len(overdue)} overdue issues this year are in this group. " if overdue_loose == len(overdue) else f"{overdue_loose} of the {len(overdue)} overdue issues this year are in this group. ")
             + f"{loose_team.index[-1]} has the most ({int(loose_team.iloc[-1])}). Note: {len(linked_ending)} linked issues are not due yet, so they cannot be overdue yet.",
             "Team leads should go through each one: link it to an OKR, schedule it, or close it. Then agree how much time unplanned work should take."),
        ]

    def caution(self):
        o = self.okr
        means = o.groupby("status_group")["grade"].mean()
        m = self.summary["metric_checks"]
        s = self.sensitivity
        sizes = s[(s["scope"] == "Key results") & (s["cycle"] == self.report_cycle) & (s["group"] != "All groups")]["n_all"]
        parts = []
        if means["Done"] <= means["Not started"]:
            parts.append(f"Use grades with care: finished OKRs average a lower grade ({means['Done']:.1f}) than ones not started ({means['Not started']:.1f}).")
        if m["corr"] is not None and abs(m["corr"]) < 0.3:
            parts.append("Grades also do not match the Key Results' own start, current and target numbers.")
        parts.append(f"Each team has only {sizes.min()}–{sizes.max()} Key Results, so one record can change its position"
                     + (". Do not rank teams on grades until a grading rule is agreed." if parts else ". Read team comparisons with care."))
        return " ".join(parts)

    def limitations(self):
        items = []
        j, o = self.jira, self.okr
        linked = j[j["matched"]].merge(o[["okr_key", "last_update"]], on="okr_key")
        same = int((linked["updated"].dt.normalize() == linked["last_update"]).sum())
        reliable = ", ".join(map(str, self.summary["freshness_reliable_cycles"])) or "none"
        items.append(f"Whether work is being updated rests on one source: Jira's updated date matches the OKR file's on {same} of {len(linked)} linked items. It is shown only for {reliable}, where the dates make sense.")
        resolved, future = int(j["resolution"].notna().sum()), int(j["resolved_after_report"].sum())
        items.append(f"How long work takes: {future} of {resolved} finish dates in Jira are after the export date, so this cannot be measured.")
        loose_blank, other_blank = j.loc[j["is_unaligned"], "story_points"].isna().mean(), j.loc[~j["is_unaligned"], "story_points"].isna().mean()
        items.append(f"How much effort work takes: story points (effort estimates) are missing on {other_blank:.0%} of linked issues but {loose_blank:.0%} of unlinked ones, so totals would not compare fairly.")
        items.append("Whether things are improving: this is one snapshot, so it shows where things stand, not the trend.")
        return "".join(f"<li>{esc(i)}</li>" for i in items)

    def dq_table(self):
        rows = self.log[self.log["count"] > 0]
        body = ""
        for _, r in rows.iterrows():
            body += (f'<tr><td><span class="prio {esc(r["priority"]).lower()}">{esc(r["priority"])}</span></td>'
                     f'<td><b>{esc(r["problem"])}</b><div class="found">{esc(r["what_we_found"])}</div></td>'
                     f'<td class="num nowrap">{r["count"]:,}<div class="sub">of {r["population"]:,}</div></td>'
                     f'<td>{esc(r["why_it_matters"])}</td><td>{esc(r["what_we_did"])}</td></tr>')
        return ('<div class="table-wrap"><table class="dq"><thead><tr><th>Priority</th><th>Problem</th><th class="num">How many</th>'
                f'<th>Why it matters</th><th>What we did</th></tr></thead><tbody>{body}</tbody></table></div>')

    def sensitivity_table(self):
        s = self.sensitivity
        rows = s[s["material"] & ((s["group"] == "All groups") | (s["scope"] == "Key results"))].sort_values(["cycle", "scope", "shift"], ascending=[True, False, False])
        return table(["Year", "Level", "Team", "Average with", "Average without", "Change", "Place with", "Place without", "Grades left out"],
                     [[r["cycle"], r["scope"], "All teams" if r["group"] == "All groups" else r["group"], f"{r['avg_with']:.2f}", f"{r['avg_without']:.2f}", f"-{r['shift']:.2f}",
                       "" if pd.isna(r["rank_with"]) else int(r["rank_with"]), "" if pd.isna(r["rank_without"]) else int(r["rank_without"]),
                       r["over_scale_keys"]] for _, r in rows.iterrows()], numeric=(3, 4, 5, 6, 7))

    def recon_table(self):
        grouped = self.recon.groupby(["match_category", "explanation"]).size().reset_index(name="n").sort_values("n", ascending=False)
        return table(["Result", "Reason", "Records"], grouped[["match_category", "explanation", "n"]].values.tolist(), numeric=(2,))

    def status_table(self):
        return table(["Group in this report", "Statuses in the files"],
                     [[group, ", ".join(s.title() for s in raws)] for group, raws in self.settings["status_groups"].items()])

    def glossary(self):
        goals = self.okr[self.okr["okr_type"] != self.kr]
        levels = goals.groupby("okr_type")["indent_level"].median().sort_values()
        order = " → ".join(" / ".join(sorted(levels[levels == lvl].index)) for lvl in sorted(levels.unique()))
        groups = "".join(f"<li><b>{esc(g)}:</b> {esc(self.labels(g))}</li>" for g in self.settings["status_groups"])
        days = self.limits["stale_days"]
        items = [
            ("OKR", "Objectives and Key Results. An objective is a goal; its Key Results are the measurable targets that show whether the goal is being met."),
            ("Levels", f"Goals sit in a tree, from the top: {order}. Each level supports the one above it. Key Results sit under the goals they measure."),
            ("Cycle", "The year an OKR belongs to."),
            ("Grade", f"A score from 0 to {self.grade_max} for how far an OKR has got. That {self.grade_max} is the top is our assumption; the files do not say."),
            ("Not updated", f"No grade update in the {days} days before the export, so nobody has reported progress recently."),
            ("Jira issue", "A work item, such as a task. Most are linked to an OKR; those that are not show work the goals do not cover."),
            ("Overdue", "Its due date has passed and it is not finished."),
        ]
        terms = "".join(f"<dt>{esc(t)}</dt><dd>{esc(d)}</dd>" for t, d in items)
        return (f'<details class="card glossary"><summary><b>What the terms mean</b> <span class="sub2">New to OKRs? Start here.</span></summary>'
                f'<dl>{terms}<dt>Status groups</dt><dd>The files use many status labels. This report groups them:<ul>{groups}</ul></dd></dl></details>')

    def payload(self):
        views, trees, notes = {}, {}, {}
        for cycle in self.cycles:
            notes[str(cycle)] = {
                "grade": self.grade_note(cycle),
                "cycle": "" if not (self.okr.loc[self.okr["cycle"] == cycle, "cycle_not_started"].all()) else
                f"{cycle} has not started yet (the files were exported on {self.report_date:%d %b %Y}), so read these figures as plans, not results.",
            }
            for team in [ALL] + self.teams:
                trees[f"{cycle}|{team}"] = self.tree(cycle, team)
                status, status_tbl = self.status_chart(cycle, team)
                fresh, fresh_tbl = self.fresh_chart(cycle, team)
                adhoc, adhoc_tbl = self.adhoc_chart(cycle, team)
                for mode in MODES:
                    grade, grade_tbl = self.grade_chart(cycle, team, mode)
                    views[f"{cycle}|{team}|{mode}"] = {
                        "kpis": self.kpis(cycle, team, mode), "status": status, "statusTable": status_tbl,
                        "grade": grade, "gradeTable": grade_tbl, "fresh": fresh, "freshTable": fresh_tbl,
                        "adhoc": adhoc, "adhocTable": adhoc_tbl}
        return {"views": views, "trees": trees, "notes": notes, "prep": {},
                "defaults": {"cycle": str(self.report_cycle), "team": ALL, "mode": "without"}}

    def render(self):
        s = self.summary
        prep_html, prep_figures = preparation.render(preparation.profile(self.result, self.settings))
        insight_html = "".join(
            f'<li><h3>{esc(a)}</h3><p>{esc(b)}</p><p class="action"><b>Action:</b> {esc(c)}</p></li>' for a, b, c in self.insights())
        options = lambda values, chosen: "".join(f'<option value="{esc(v)}"{" selected" if str(v) == str(chosen) else ""}>{esc(v)}</option>' for v in values)
        fills = {
            "__DATE__": f"{self.report_date:%d %B %Y}",
            "__CYCLE_OPTIONS__": options(self.cycles, self.report_cycle),
            "__TEAM_OPTIONS__": options([ALL] + self.teams, ALL),
            "__WITHOUT__": esc(self.without_label), "__WITH__": esc(self.with_label),
            "__REPORT_CYCLE__": str(self.report_cycle),
            "__INSIGHTS__": insight_html,
            "__GLOSSARY__": self.glossary(),
            "__CAUTION__": esc(self.caution()),
            "__LIMITATIONS__": self.limitations(),
            "__DQ_TABLE__": self.dq_table(),
            "__DQ_SHOWN__": str(int((self.log["count"] > 0).sum())),
            "__SENSITIVITY__": self.sensitivity_table(),
            "__RECON__": self.recon_table(),
            "__STATUS_MAP__": self.status_table(),
            "__GRADE_MAX__": str(self.grade_max),
            "__MATERIAL__": f"{self.limits['material_points']:g}",
            "__STALE__": str(self.limits["stale_days"]),
            "__OKR_ROWS__": f"{s['okr_rows']:,}", "__JIRA_ROWS__": f"{s['jira_rows']:,}",
            "__SOURCES__": esc(f"{s['sources']['okr']} and {s['sources']['jira']}"),
            "__AGREEMENT__": s["indent_agreement"],
            "__AGREEMENT_TEXT__": (lambda a, v: f"every one of the {v}" if a == v else f"{a} of the {v}")(*s["indent_agreement"].split("/")),
            "__WEIGHTS__": str(int(self.okr["weight"].notna().sum())),
            "__PREP__": prep_html,
            "__STOP_CHECKS__": str(len(s["blocking_checks"])),
            "__DQ_CHECKS__": str(len(self.log)),
            "__AUTOMATION_TABLE__": table(["Part", "Status"], [
                ["Rules in one settings file", "Built in this project"],
                ["Checks that stop the run before anything is written", "Built in this project"],
                ["Last good output kept when a run fails", "Built in this project"],
                ["Data-quality log rebuilt on every run", "Built in this project"],
                ["Tests with deliberately changed files", "Built in this project"],
                ["Running on a schedule and fetching files automatically", "To add"],
                ["Alerts to owners when a run stops or a file is late", "To add"],
                ["Keeping each run's inputs and comparing with the last run", "To add"],
            ]),
            "__PLOTLY__": get_plotlyjs(),
            "__DATA__": json.dumps({**self.payload(), "prep": prep_figures}, separators=(",", ":")).replace("</", "<\\/"),
        }
        page = TEMPLATE
        for key, value in fills.items():
            page = page.replace(key, value)
        return page


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OKR Delivery Report</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--grid:#e1e0d9;--base:#c3c2b7;--blue:#2a78d6;--crit:#d03b3b;--warn:#fab219;}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:24px 16px 56px}
header h1{font-size:26px;line-height:1.2;margin:0 0 4px;font-weight:650}
header p{margin:0;color:var(--ink2)}
.topbar{position:sticky;top:0;z-index:5;background:var(--plane);padding:8px 0 10px;margin:8px 0 12px}
.controls{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:end;padding:8px 14px;background:var(--surface);border:1px solid var(--grid);border-radius:8px}
.controls.off label.ctl{opacity:.55}.filter-note{font-size:13px;color:var(--ink2);align-self:center}
.step{scroll-margin-top:150px}
label.ctl{display:flex;flex-direction:column;font-size:12px;color:var(--ink2);gap:3px}
select{font:inherit;color:var(--ink);padding:6px 8px;border:1px solid var(--base);border-radius:6px;background:#fff;min-width:150px;max-width:100%}
nav{display:flex;gap:4px;flex-wrap:wrap;margin:0 0 8px;border-bottom:1px solid var(--grid)}
nav button{font:inherit;color:var(--ink2);padding:8px 12px;background:none;border:0;border-bottom:2px solid transparent;cursor:pointer}
nav button[aria-selected=true]{color:var(--ink);border-color:var(--blue);font-weight:600}
.panel[hidden],.filter-note[hidden]{display:none}
.banner{padding:10px 14px;border-left:4px solid var(--warn);background:#fff8e6;border-radius:4px;margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:0 0 20px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:14px}
.tile-label{font-size:13px;color:var(--ink2)}.tile-value{font-size:32px;font-weight:650;line-height:1.25}.tile-note{font-size:13px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:16px;margin:0 0 16px;min-width:0}
.card h2{font-size:17px;margin:0 0 2px}.card>p.sub2,.sub2{color:var(--ink2);margin:0 0 10px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}.grid .card{margin:0}.grid{margin:0 0 16px}
.insights{list-style:none;counter-reset:i;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}
.insights li{counter-increment:i;padding-top:8px;border-top:2px solid var(--blue)}
.insights h3{font-size:15px;margin:0 0 6px;line-height:1.35}.insights p{margin:0 0 8px;font-size:14px;color:var(--ink2)}.insights p.action{color:var(--ink)}
.caution{margin:14px 0 0;padding:10px 14px;border-left:4px solid var(--warn);background:#fff8e6;border-radius:4px;font-size:14px}
.note{font-size:14px;color:var(--ink2);margin:10px 0 0}
details.tv{margin-top:8px;font-size:13px}details.tv>summary{cursor:pointer;color:var(--ink2)}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 0}
th,td{padding:7px 8px;text-align:left;vertical-align:top;border-bottom:1px solid var(--grid)}th{font-weight:600;color:var(--ink2);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.nowrap{white-space:nowrap}
.dq td{min-width:80px;font-size:14px}.dq td:nth-child(2){min-width:280px}.dq td:nth-child(4),.dq td:nth-child(5){min-width:220px}.dq .found{color:var(--ink2);margin-top:3px}
.glossary{padding:12px 16px}.glossary summary{cursor:pointer}.glossary dl{display:grid;grid-template-columns:130px 1fr;gap:6px 16px;margin:12px 0 0;font-size:14px}
.glossary dt{font-weight:600}.glossary dd{margin:0;color:var(--ink2)}.glossary ul{margin:4px 0 0;padding-left:18px}
@media (max-width:640px){.glossary dl{grid-template-columns:1fr;gap:2px}.glossary dd{margin-bottom:6px}}
.flow{list-style:none;counter-reset:f;display:grid;grid-template-columns:repeat(5,1fr);gap:22px;margin:16px 0 10px;padding:0}
.flow li{counter-increment:f;position:relative;background:#fff;border:1px solid var(--base);border-radius:8px;padding:10px 12px;font-size:13px;color:var(--ink2)}
.flow li b{display:block;color:var(--ink);font-size:14px;margin-bottom:2px}.flow li b::before{content:counter(f) ". ";color:var(--blue)}
.flow li.gate{border-color:var(--warn);background:#fff8e6}
.flow li+li::before{content:"→";position:absolute;left:-18px;top:50%;transform:translateY(-50%);color:var(--muted);font-size:16px}
.flow-fail{margin:0 0 8px;padding:10px 14px;border-left:4px solid var(--crit);background:#fdf0f0;border-radius:4px;font-size:14px}
@media (max-width:900px){.flow{grid-template-columns:1fr;gap:20px}.flow li+li::before{content:"↓";left:50%;top:-19px;transform:none}}
.legend-prio{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin:0 0 6px}
.sub{color:var(--muted);font-size:12px;font-weight:400;overflow-wrap:anywhere}
.prio{display:inline-block;font-size:12px;font-weight:600;padding:1px 7px;border-radius:10px;border:1px solid var(--base)}
.prio.high{background:#d03b3b;color:#fff;border-color:#d03b3b}.prio.medium{background:#fff3d6;border-color:#fab219}.prio.low{background:#f0efec}
.tree-head,.node{display:grid;grid-template-columns:minmax(260px,1fr) 130px 110px 52px 120px 70px;gap:10px;align-items:baseline}
.tree-scroll{overflow-x:auto}.tree-inner{min-width:820px}
.tree-head{font-size:12px;font-weight:600;color:var(--ink2);padding:6px 8px 6px 26px;border-bottom:1px solid var(--grid)}
.tree details>summary{list-style-position:outside;cursor:pointer;padding:6px 8px 6px 4px;margin-left:18px;border-bottom:1px solid #efeeea}
.tree details>summary:hover{background:#f4f3ef}.tree .leaf{padding:6px 8px 6px 4px;margin-left:18px;border-bottom:1px solid #efeeea}
.tree .kids{margin-left:16px}.node{font-size:13px}.node.context{opacity:.55}
.n-title b{font-weight:600}.n-grade,.n-days{font-variant-numeric:tabular-nums;text-align:right}.n-type,.n-owner{color:var(--ink2)}
.n-status i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.rollup{margin-left:8px;color:var(--muted);font-size:12px}.flag{display:inline-block;margin-left:6px;font-size:11px;padding:0 6px;border:1px solid var(--warn);border-radius:9px;background:#fff8e6;white-space:nowrap}
.method h3{font-size:15px;margin:18px 0 4px}.method h3:first-of-type{margin-top:0}.method p,.method li{font-size:14px;color:var(--ink2)}.method ul{padding-left:20px;margin:6px 0}
.step-no{margin:0;font-size:12px;font-weight:600;color:var(--blue);letter-spacing:.02em}
.step h3{font-size:14px;margin:12px 0 2px}.evidence{margin:6px 0 4px}.prep-chart{width:100%}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px 24px}.two>div{min-width:0}
.decision{display:grid;grid-template-columns:170px 1fr;gap:8px 16px;margin:14px 0 0;padding:12px 0 0;border-top:1px solid var(--grid);font-size:14px}
.decision dt{font-weight:600}.decision dd{margin:0;color:var(--ink2)}
.meaning{margin:6px 0 0;padding-left:18px;font-size:14px;color:var(--ink2)}.meaning li{margin:0 0 8px}.meaning b{color:var(--ink)}
.step-links{margin:8px 0 0;padding-left:22px;columns:3 200px;font-size:14px}.step-links a{color:var(--blue)}
footer{color:var(--muted);font-size:12px;margin-top:20px}
@media (max-width:640px){.decision{grid-template-columns:1fr;gap:2px 0}.decision dd{margin-bottom:8px}.topbar{display:contents}nav{position:sticky;top:0;z-index:5;background:var(--plane)}.controls{margin:0 0 12px}.step{scroll-margin-top:90px}.tile-value{font-size:28px}label.ctl,select{width:100%}}
@media print{
 @page{size:1240px 1754px;margin:24px}
 body{background:#fff}.wrap{max-width:none;padding:0}.topbar,.controls,nav{display:none}.panel[hidden]{display:block}
 .card,.tile,.insights li,tr,.tree details,.decision,.two{break-inside:avoid}.step-links{display:none}.panel{break-before:page}.panel:first-of-type{break-before:auto}
 .flow{grid-template-columns:repeat(5,1fr)!important;gap:22px!important}
 .flow li+li::before{content:"→"!important;left:-18px!important;top:50%!important;transform:translateY(-50%)!important}
 details.tv{display:none}.tree-inner{min-width:0}
}
</style></head><body><div class="wrap">
<header><h1>OKR Delivery Report</h1><p>Practice data. OKR and Jira exports dated __DATE__.</p></header>
<div class="topbar">
<nav role="tablist">
 <button role="tab" data-tab="prep" aria-selected="true">Data preparation</button>
 <button role="tab" data-tab="overview" aria-selected="false">Overview</button>
 <button role="tab" data-tab="hierarchy" aria-selected="false">Hierarchy</button>
 <button role="tab" data-tab="quality" aria-selected="false">Data quality</button>
 <button role="tab" data-tab="method" aria-selected="false">Method and assumptions</button>
 <button role="tab" data-tab="automation" aria-selected="false">Automation idea</button>
</nav>
<div class="controls" id="controls">
 <label class="ctl">Cycle (year)<select id="cycle">__CYCLE_OPTIONS__</select></label>
 <label class="ctl">Team<select id="team">__TEAM_OPTIONS__</select></label>
 <label class="ctl">Grade average<select id="mode"><option value="without">__WITHOUT__</option><option value="with">__WITH__</option></select></label>
 <span class="filter-note" id="filter-note" hidden>Filters apply to Overview and Hierarchy. This page always covers all the data.</span>
</div>
</div>

<section class="panel" id="prep">__PREP__</section>

<section class="panel" id="overview">
 <div class="banner" id="cycle-banner" hidden></div>
 __GLOSSARY__
 <div class="tiles" id="kpis"></div>
 <div class="card"><h2>What to act on in __REPORT_CYCLE__</h2><p class="sub2">Three findings for all teams this year, each with a suggested action. They do not change with the filters.</p>
  <ol class="insights">__INSIGHTS__</ol><p class="caution">__CAUTION__</p></div>
 <div class="grid">
  <div class="card"><h2>Status mix by team</h2><p class="sub2">How each team's OKRs split by status. Teams with the most at risk are at the top.</p><div id="c-status" role="img" aria-label="Stacked bars of OKR status share by team"></div><details class="tv"><summary>Table view</summary><div id="t-status"></div></details></div>
  <div class="card"><h2>Average Key Result grade by team</h2><p class="sub2">Out of __GRADE_MAX__. Number of Key Results in brackets. Filled dot: the average chosen in the filter. Hollow dot: the other one.</p><div id="c-grade" role="img" aria-label="Dot plot of average grade by team, with and without grades above the scale"></div><details class="tv"><summary>Table view</summary><div id="t-grade"></div></details></div>
  <div class="card"><h2>OKRs with no update in __STALE__+ days</h2><p class="sub2">Share of each team's OKRs with no grade update in the __STALE__ days before the export. Lower is better.</p><div id="c-fresh" role="img" aria-label="Bars of stale OKR share by team"></div><p class="note" id="fresh-note" hidden>Not shown for this year: its update dates are unreliable (see Data quality).</p><details class="tv" id="tv-fresh"><summary>Table view</summary><div id="t-fresh"></div></details></div>
  <div class="card"><h2>Jira issues with no OKR link</h2><p class="sub2">Work items per team that are not linked to any goal. Orange: past their due date.</p><div id="c-adhoc" role="img" aria-label="Stacked bars of unlinked Jira issues by team"></div><details class="tv"><summary>Table view</summary><div id="t-adhoc"></div></details></div>
 </div>
 <div class="card"><h2>Why two grade averages?</h2><p class="note" id="grade-note"></p></div>
</section>

<section class="panel" id="hierarchy">
 <div class="card"><h2>OKR hierarchy</h2><p class="sub2">Every OKR with the goals and Key Results beneath it. Click a row to open it. When a team is selected, grey rows belong to other teams and are shown only to keep the tree whole. Yellow tags point to a data problem.</p>
 <div class="tree-scroll"><div class="tree-inner"><div class="tree-head"><span>OKR</span><span>Level</span><span>Status</span><span style="text-align:right">Grade</span><span>Owner</span><span style="text-align:right">Updated</span></div><div class="tree" id="tree"></div></div></div></div>
</section>

<section class="panel" id="quality">
 <div class="card"><h2>Data-quality log</h2><p class="sub2">__DQ_SHOWN__ problems found in the two files, most serious first. Nothing was deleted: every problem is flagged, fixed in a separate column, or explained. Numbers cover all years.</p><p class="legend-prio"><span><span class="prio high">High</span> changes a headline number or the hierarchy</span><span><span class="prio medium">Medium</span> affects some views</span><span><span class="prio low">Low</span> tidy-up or known gap</span></p>__DQ_TABLE__</div>
</section>

<section class="panel" id="method">
 <div class="card method"><h2>Method and assumptions</h2>
  <h3>How the average grade is worked out</h3>
  <ul><li><b>Only Key Results are averaged.</b> They are the measurable targets. The goals above them summarise the same work, so including them would count it twice.</li>
  <li><b>Every Key Result counts the same.</b> Only __WEIGHTS__ of __OKR_ROWS__ OKRs have a weight, which is too few to use.</li>
  <li><b>Grades above __GRADE_MAX__ are left out of the main number</b> and shown beside it. That __GRADE_MAX__ is the top of the scale is our assumption; the files do not say.</li>
  <li><b>An OKR shared by two teams counts for both.</b> So team figures do not add up to the total, which counts each OKR once.</li>
  <li><b>The Contribute column is not used,</b> because the files do not explain what it means.</li></ul>
  <h3>Where leaving out grades above __GRADE_MAX__ changes the result</h3>
  <p>Every change of __MATERIAL__ point or more. The grades are still in the data.</p>__SENSITIVITY__
  <h3>How statuses are grouped</h3>__STATUS_MAP__
  <h3>How the two files were matched</h3>
  <p>Each Jira issue is matched to its OKR by the OKR key, which is unique. Every record ends up in one of these groups.</p>__RECON__
  <h3>Parent links</h3>
  <p>Where an OKR's parent was blank or a dummy value, we used the file's indentation instead. It agreed with __AGREEMENT_TEXT__ parent links we could check. These are tagged for owners to confirm.</p>
  <h3>What this report cannot tell you</h3>
  <ul>__LIMITATIONS__</ul>
  <h3>Source and tools</h3>
  <p>__OKR_ROWS__ OKRs and __JIRA_ROWS__ Jira issues from __SOURCES__. Built in Python (pandas and Plotly). The business rules are in config/settings.yaml. AI use: Claude Code (Anthropic) helped with data profiling, code and drafting.</p>
 </div>
</section>
<section class="panel" id="automation">
 <div class="card method"><h2>Automation idea</h2>
  <p>If both exports are updated regularly, the report can refresh itself. Nobody should need to copy, clean or paste anything by hand, and a bad file should never reach a manager.</p>
  <ol class="flow" aria-label="Flow of an automatic refresh">
   <li><b>New exports arrive</b><span>Saved to a shared folder, or pulled from Jira on a schedule.</span></li>
   <li class="gate"><b>Check the files</b><span>Columns, statuses and dates are compared with the rules file.</span></li>
   <li><b>Clean and link</b><span>The same steps as the Data preparation tab, every time.</span></li>
   <li class="gate"><b>Check the totals</b><span>__STOP_CHECKS__ checks, such as "no record lost" and "one row per OKR".</span></li>
   <li><b>Publish</b><span>Report, PDF and data-quality log are updated; owners get a list of new problems.</span></li>
  </ol>
  <p class="flow-fail"><b>If a check fails:</b> stop, publish nothing, keep the last good report online, and alert the owner with what to fix.</p>

  <h3>Which steps to automate</h3>
  <ul><li><b>Collecting the files</b>, so there is no manual download and upload.</li>
  <li><b>Cleaning, linking and checking</b>, so the same rules are applied the same way every time.</li>
  <li><b>Rebuilding the report and the data-quality log</b>, and sending owners the problems that concern them.</li></ul>

  <h3>How the data stays accurate</h3>
  <ul><li><b>One rules file.</b> Status groups, thresholds and column names live in one place (config/settings.yaml), so every run uses the same rules and a change is made once.</li>
  <li><b>The run checks itself.</b> If any of the __STOP_CHECKS__ total checks fails, nothing is published.</li>
  <li><b>Problems stay visible.</b> The __DQ_CHECKS__ data-quality checks run every time. Comparing their counts with the last run would show when something gets worse, such as more grades above __GRADE_MAX__.</li>
  <li><b>Rule changes are tested first,</b> including against deliberately broken files.</li></ul>

  <h3>What happens if something goes wrong</h3>
  <ul><li><b>A file has changed</b> (a renamed column, a new status or an unknown date format): the run stops and names the problem. The last good report stays online. Someone updates the rules file and runs it again.</li>
  <li><b>A few rows are wrong</b> (a missing parent, an impossible date): the report is still published, the rows are flagged in the data-quality log, and owners get a list to fix at the source.</li>
  <li><b>A file is missing or late:</b> the report always shows the export date, so readers can see how old it is. If no new file arrives on time, the owner is alerted.</li>
  <li><b>A wrong number is found after publishing:</b> each run keeps its input files, so the earlier version can be restored and the difference explained.</li></ul>

  <h3>What exists today and what would be added</h3>
  __AUTOMATION_TABLE__
 </div>
</section>

</div>
<script>__PLOTLY__</script>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
 var D=JSON.parse(document.getElementById('data').textContent);
 var cycle=document.getElementById('cycle'),team=document.getElementById('team'),mode=document.getElementById('mode');
 var config={displayModeBar:false,responsive:true};
 var $=function(id){return document.getElementById(id)};
 function plot(id,fig){var el=$(id);if(!fig){Plotly.purge(el);el.innerHTML='';el.style.display='none';el.style.height='';return;}
  // Plotly edits the objects it is given, and views are revisited, so each draw gets its own copy.
  // The explicit height keeps the card sized around the chart.
  var f=JSON.parse(JSON.stringify(fig));el.style.display='';el.style.height=f.layout.height+'px';Plotly.react(el,f.data,f.layout,config);}
 function update(){
  var v=D.views[cycle.value+'|'+team.value+'|'+mode.value],n=D.notes[cycle.value];
  $('kpis').innerHTML=v.kpis;
  plot('c-status',v.status);plot('c-grade',v.grade);plot('c-fresh',v.fresh);plot('c-adhoc',v.adhoc);
  $('t-status').innerHTML=v.statusTable;$('t-grade').innerHTML=v.gradeTable;$('t-fresh').innerHTML=v.freshTable;$('t-adhoc').innerHTML=v.adhocTable;
  $('fresh-note').hidden=!!v.fresh;$('tv-fresh').hidden=!v.fresh;
  $('grade-note').textContent=n.grade;
  $('cycle-banner').hidden=!n.cycle;$('cycle-banner').textContent=n.cycle;
  $('tree').innerHTML=D.trees[cycle.value+'|'+team.value];
 }
 function show(name){
  document.querySelectorAll('.panel').forEach(function(p){p.hidden=p.id!==name});
  document.querySelectorAll('nav button').forEach(function(b){b.setAttribute('aria-selected',b.dataset.tab===name)});
  var on=name==='overview'||name==='hierarchy';
  [cycle,team,mode].forEach(function(c){c.disabled=!on});
  $('controls').classList.toggle('off',!on);$('filter-note').hidden=on;
  window.dispatchEvent(new Event('resize'));
 }
 [cycle,team,mode].forEach(function(c){c.addEventListener('change',update)});
 document.querySelectorAll('nav button').forEach(function(b){b.addEventListener('click',function(){show(b.dataset.tab);window.scrollTo(0,0)})});
 var h=new URLSearchParams(location.hash.slice(1));
 if(h.get('cycle'))cycle.value=h.get('cycle');if(h.get('team'))team.value=h.get('team');if(h.get('mode'))mode.value=h.get('mode');
 update();Object.keys(D.prep).forEach(function(id){plot(id,D.prep[id])});show(h.get('tab')||'prep');
 if(h.get('open'))document.querySelectorAll('#tree details').forEach(function(d,i){if(i<+h.get('open'))d.open=true});
})();
</script></body></html>
"""


def find_chrome():
    return next((c for c in CHROME_PATHS if Path(c).exists()), shutil.which("chromium") or shutil.which("google-chrome"))


def export_pdf(html_path, pdf_path):
    chrome = find_chrome()
    if not chrome:
        return False
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--window-size=1192,1400", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=180)
    return pdf_path.exists()


def build(settings=None, out_dir=REPORTS, pdf=True):
    settings = settings or p.load_settings()
    result = p.run(settings)
    page = Report(result, settings).render()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "okr_report.html"
    html_path.write_text(page, encoding="utf-8")
    made_pdf = export_pdf(html_path, out_dir / "okr_report.pdf") if pdf else False
    return html_path, made_pdf


def main():
    try:
        html_path, made_pdf = build()
    except p.PipelineError as error:
        print(f"RUN STOPPED: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {html_path} ({html_path.stat().st_size / 1e6:.1f} MB)")
    print("PDF written" if made_pdf else "PDF skipped: no Chrome or Edge found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
