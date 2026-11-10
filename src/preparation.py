"""The data-preparation walkthrough: what the raw files showed and why each decision was made.

profile() measures the evidence from the raw columns and the pipeline's results.
render() turns it into the report's first tab. Every figure is computed on each run, and sentences
that interpret a figure check it first, so a new export cannot leave an old conclusion on the page.
"""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pipeline as p
from report_common import BASE, BLUE, GRID, INK, MUTED, SERIOUS, STATUS_COLORS, STATUS_ORDER, base_layout, esc, fig_json, pct, table

FORMATS = ["text month", "dd.mm.yyyy", "ISO", "blank"]
SMALL_CELL = 5  # A team and status combination with fewer OKRs than this is too small to compare.


def clean_label(value):
    return str(value).strip().lower()


def default_parser_misses(series):
    """How many filled dates pandas' standard parser would silently blank."""
    filled = series[~series.map(p.is_blank)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(pd.to_datetime(filled, errors="coerce").isna().sum())


def blank_shares(frame, column_map):
    return pd.Series({source: frame[name].map(p.is_blank).mean() for source, name in column_map.items()})


def date_rows(frame, columns, file_name):
    rows = []
    for label, raw_col in columns.items():
        formats = frame[raw_col].map(p.date_format).value_counts()
        rows.append([file_name, label] + [int(formats.get(f, 0)) for f in FORMATS] + [default_parser_misses(frame[raw_col])])
    return rows


def profile(result, settings):
    okr, jira, bridge, recon, log = result["okr"], result["jira"], result["bridge"], result["recon"], result["log"]
    summary, sens = result["summary"], result["sensitivity"]
    limits, tiers = settings["thresholds"], settings["tiers"]
    report_cycle = summary["report_cycle"]
    report_date = pd.Timestamp(summary["report_date"])
    groups_by_status = p.status_lookup(settings)
    counts = log.set_index("check")["count"].to_dict()
    kr = okr[okr["okr_type"] == tiers["key_result"]]

    # The sheet as it arrives: a title block first, then indentation columns.
    sheet = pd.read_excel(p.find_source(settings, "okr_pattern"), header=None, dtype=str)
    marker = settings["okr_columns"]["header_marker"]
    header_row = sheet.index[sheet[0].fillna("").str.strip() == marker][0]
    headers = sheet.iloc[header_row].tolist()
    indent_count = next(i for i in range(1, len(headers)) if isinstance(headers[i], str) and headers[i].strip())
    title_rows = [[i + 1, sheet.iloc[i].dropna().iloc[0]] for i in range(header_row) if sheet.iloc[i].notna().any()]
    sample = sheet.iloc[header_row + 1:header_row + 6, :indent_count]
    indent_rows = [[row + 1] + [("" if p.is_blank(v) else str(v)[:34] + ("…" if len(str(v)) > 34 else "")) for v in sample.loc[row]] for row in sample.index]

    # Statuses, one row per label exactly as written, including text variants.
    both = pd.concat([okr["status_raw"].rename("label").to_frame().assign(file="OKR"),
                      jira["status_raw"].rename("label").to_frame().assign(file="Jira")])
    status_rows = []
    for label, part in both.groupby("label"):
        key = clean_label(label)
        status_rows.append({
            "label": label, "key": key, "group": groups_by_status[key],
            "okr": int((part["file"] == "OKR").sum()), "jira": int((part["file"] == "Jira").sum()),
            "avg_grade": okr.loc[okr["status_raw"] == label, "grade"].mean(),
            "resolution": jira.loc[jira["status_raw"] == label, "resolution"].notna().mean(),
        })
    status_rows.sort(key=lambda r: (STATUS_ORDER.index(r["group"]), -(r["okr"] + r["jira"])))
    variants = sorted({r["label"] for r in status_rows if r["label"] != r["label"].strip().title()})
    done_rows = jira[jira["status_group"] == p.DONE]
    other_rows = jira[jira["status_group"] != p.DONE]

    # Would team comparisons survive if the raw labels were kept?
    current = bridge.merge(okr[["okr_key", "cycle", "status", "status_group"]], on="okr_key")
    current = current[current["cycle"] == report_cycle]

    def small_share(column):
        cells = current.groupby(["group", column]).size().unstack(fill_value=0).values.ravel()
        return float((cells < SMALL_CELL).mean()), float(np.median(cells))

    # How much does each grouping choice move the headline figures?
    in_cycle = okr[okr["cycle"] == report_cycle]
    total = len(in_cycle)
    keys = in_cycle["status_raw"].map(clean_label)
    moves = []
    for group, raws in settings["status_groups"].items():
        if len(raws) < 2:
            continue
        group_n = int((in_cycle["status_group"] == group).sum())
        for raw in raws:
            n = int((keys == clean_label(raw)).sum())
            moves.append([raw.title(), group, n, pct(group_n, total), pct(group_n - n, total)])

    # Grades.
    edges = list(range(0, int(np.ceil(okr["grade"].max() / 10)) * 10 + 1, 10))
    bins = pd.cut(okr["grade"], edges, right=True, include_lowest=True).value_counts().sort_index()
    over = okr[okr["grade_over_max"]]
    means = okr.groupby("status_group")["grade"].mean()
    kr_shift = sens[(sens["scope"] == "Key results") & (sens["group"] == "All groups") & (sens["cycle"] == report_cycle)].iloc[0]
    biggest = sens[(sens["scope"] == "Key results") & (sens["group"] != "All groups")].sort_values("shift", key=np.abs, ascending=False).iloc[0]

    # Dates by cycle, to see which time measures hold.
    cycle_rows = []
    for cycle in summary["cycles"]:
        c_okr, c_jira = okr[okr["cycle"] == cycle], jira[jira["cycle"] == cycle]
        cycle_rows.append([cycle, len(c_okr), int(c_okr["cycle_not_started"].sum()), len(c_jira),
                           int(c_jira["updated_before_created"].sum()), int(c_jira["created_after_report"].sum()),
                           "Yes" if cycle in summary["freshness_reliable_cycles"] else "No"])
    linked = jira[jira["matched"]].merge(okr[["okr_key", "last_update"]], on="okr_key")

    points = jira.groupby("issue_type")["story_points"].apply(lambda s: s.isna().mean())
    unlinked = jira[~jira["matched"]]
    okr_only = recon[recon["match_category"] == "OKR only"]
    parents = okr["parent_status"].value_counts()

    return {
        "summary": summary, "limits": limits, "tiers": tiers, "report_cycle": report_cycle, "report_date": report_date,
        "counts": counts, "settings": settings,
        "okr_rows": len(okr), "jira_rows": len(jira), "okr_columns": len(settings["okr_columns"]["map"]),
        "jira_columns": len(settings["jira_columns"]["map"]), "sheet_rows": len(sheet),
        "okr_blanks": blank_shares(okr, {k: v for k, v in settings["okr_columns"]["map"].items()}),
        "jira_blanks": blank_shares(jira, {k: v for k, v in settings["jira_columns"]["map"].items()}),
        "metric_filled": int(okr["metric_type"].notna().sum()),
        "metric_on_kr": int(kr["metric_type"].notna().sum()), "kr_rows": len(kr),
        "weights": int(okr["weight"].notna().sum()), "points_low": float(points.min()), "points_high": float(points.max()),
        "points_low_type": points.idxmin(), "points_high_type": points.idxmax(),
        "header_row": int(header_row) + 1, "indent_count": indent_count, "title_rows": title_rows, "indent_rows": indent_rows,
        "uniqueness": [
            ["OKR file", "OKR key", len(okr), int(okr["okr_key"].nunique()), len(okr) - int(okr["okr_key"].nunique())],
            ["Jira file", "Issue key", len(jira), int(jira["issue_key"].nunique()), len(jira) - int(jira["issue_key"].nunique())],
            ["Jira file", "OKR key (where filled)", int(jira["okr_key"].notna().sum()), int(jira["okr_key"].nunique()),
             int(jira["okr_key"].notna().sum()) - int(jira["okr_key"].nunique())],
        ],
        "duplicates": summary["okr_duplicates"], "collisions": jira.loc[jira["issue_key_collision"], ["issue_key", "okr_key", "issue_key_final"]].values.tolist(),
        "pairing": summary["key_pairing_share"],
        "date_rows": date_rows(okr, {"Start date": "start_raw", "Due date": "due_raw", "Last grade update": "last_update_raw"}, "OKR")
        + date_rows(jira, {"Created": "created_raw", "Updated": "updated_raw", "Target start": "target_start_raw",
                           "Target end": "target_end_raw", "Resolution": "resolution_raw"}, "Jira"),
        "month_names": sorted({str(v).split("/")[1] for c in ["start_raw", "due_raw", "last_update_raw"] for v in okr[c].dropna() if str(v).count("/") == 2}),
        "status_rows": status_rows, "variants": variants, "labels_raw": len(status_rows),
        "labels_clean": len({r["key"] for r in status_rows}), "labels_group": len({r["group"] for r in status_rows}),
        "done_resolved": float(done_rows["resolution"].notna().mean()), "other_resolved": float(other_rows["resolution"].notna().mean()),
        "status_agree": int((jira["matched"] & ~jira["status_conflict"]).sum()), "status_linked": int(jira["matched"].sum()),
        "small_raw": small_share("status"), "small_group": small_share("status_group"),
        "group_means": means, "moves": moves, "total_in_cycle": total,
        "bins": bins, "edges": edges, "grade_min": float(okr["grade"].min()), "grade_median": float(okr["grade"].median()),
        "over_count": len(over), "over_values": sorted(over["grade"].unique().tolist()),
        "over_not_done": int((over["status_group"] != p.DONE).sum()), "kr_with": float(kr_shift["avg_with"]), "kr_without": float(kr_shift["avg_without"]),
        "kr_n": int(kr_shift["n_all"]), "kr_n_without": int(kr_shift["n_all"] - kr_shift["n_over_scale"]), "biggest": biggest, "metrics": summary["metric_checks"],
        "tier_levels": pd.crosstab(okr["okr_type"], okr["indent_level"]), "parents": parents,
        "agreement": summary["indent_agreement"], "inferred": okr.loc[okr["parent_inferred"], "okr_key"].tolist(),
        "unresolved": int(okr["parent_unresolved"].sum()), "depth": int(okr["hierarchy_depth"].max()) + 1,
        "group_values": okr["group_raw"].value_counts(), "multi_rows": int(okr["multi_group"].sum()),
        "jira_multi_rows": int(jira["group_raw"].fillna("").str.contains(settings["groups"]["separator"], regex=False).sum()),
        "bridge_rows": len(bridge), "teams": int(bridge["group"].nunique()), "group_blank": int(jira["group_raw"].isna().sum()),
        "group_backfilled": int(jira["group_backfilled"].sum()),
        "okr_key_filled": int(jira["okr_key"].notna().sum()), "okr_key_found": int(jira["matched"].sum()),
        "recon": recon.groupby(["match_category", "explanation"]).size().reset_index(name="n").sort_values("n", ascending=False),
        "recon_rows": len(recon), "unlinked_types": unlinked["issue_type"].value_counts().to_dict(),
        "unlinked_labels": unlinked["labels"].fillna("none").value_counts().to_dict(), "unlinked_n": len(unlinked),
        "okr_only_types": okr_only["okr_type"].value_counts().to_dict(), "okr_only_n": len(okr_only),
        "cycle_rows": cycle_rows, "same_update": int((linked["updated"].dt.normalize() == linked["last_update"]).sum()), "same_total": len(linked),
        "end_differs": int(jira["target_end_differs_from_okr"].sum()), "end_blank": int(jira["target_end_backfilled"].sum()),
        "jira_cycle_filled": int(jira["okr_cycle"].isna().sum()),
    }


def counts_text(counts):
    return ", ".join(f"{name} ({n})" for name, n in counts.items()) or "none"


def blank_chart(shares):
    shares = shares[shares > 0].sort_values()
    colors = [BASE if v < 0.5 else SERIOUS for v in shares.values]
    fig = go.Figure(go.Bar(x=shares.values * 100, y=shares.index, orientation="h", marker_color=colors, marker_line_width=0,
                           text=[f"{v:.0%}" for v in shares.values], textposition="outside", textfont_color=INK, cliponaxis=False,
                           hovertemplate="%{y}: %{x:.0f}% blank<extra></extra>"))
    fig.update_layout(**base_layout(max(150, 26 * len(shares) + 56), bargap=0.3, margin=dict(l=8, r=44, t=8, b=32)))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    return fig_json(fig)


def status_chart(rows):
    fig = go.Figure()
    order = [r["label"] for r in reversed(rows)]
    labels = [repr(l) if l != l.strip() else l for l in order]
    shown = dict(zip(order, labels))
    for group in STATUS_ORDER:
        part = [r for r in rows if r["group"] == group]
        fig.add_bar(y=[shown[r["label"]] for r in part], x=[r["okr"] + r["jira"] for r in part], name=group, orientation="h",
                    marker_color=STATUS_COLORS[group], marker_line_width=0, text=[r["okr"] + r["jira"] for r in part], textposition="outside",
                    textfont_color=INK, cliponaxis=False, hovertemplate="%{y}: %{x} rows across both files<extra>" + group + "</extra>")
    fig.update_layout(**base_layout(26 * len(rows) + 80, bargap=0.3, margin=dict(l=8, r=36, t=36, b=32)))
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    return fig_json(fig)


def grade_chart(bins, edges, grade_max):
    labels = [f"{low}–{high}" for low, high in zip(edges[:-1], edges[1:])]
    colors = [BLUE if high <= grade_max else SERIOUS for high in edges[1:]]
    fig = go.Figure(go.Bar(x=labels, y=bins.values, marker_color=colors, marker_line_width=0, text=bins.values, textposition="outside",
                           textfont_color=INK, cliponaxis=False, hovertemplate="Grade %{x}: %{y} OKRs<extra></extra>"))
    layout = base_layout(270, bargap=0.12, margin=dict(l=8, r=16, t=24, b=44))
    layout["xaxis"] = dict(type="category", showgrid=False, linecolor=BASE, tickfont_color=MUTED, fixedrange=True, title=dict(text="Grade", font_color=MUTED))
    layout["yaxis"] = dict(visible=False, fixedrange=True)
    fig.update_layout(**layout)
    return fig_json(fig)


def step(number, title, question, evidence, found, decision, why, effect):
    rows = "".join(f"<dt>{label}</dt><dd>{esc(text)}</dd>" for label, text in
                   [("What we found", found), ("What we decided", decision), ("Why", why), ("Result", effect)])
    return (f'<article class="card step" id="step-{number}"><p class="step-no">Step {number}</p><h2>{esc(title)}</h2>'
            f'<p class="sub2">{esc(question)}</p><div class="evidence">{evidence}</div><dl class="decision">{rows}</dl></article>')


def chart_slot(chart_id, label):
    return f'<div id="{chart_id}" class="prep-chart" role="img" aria-label="{esc(label)}"></div>'


def share(value):
    return f"{value:.0%}"


def render(e):
    limits, tiers = e["limits"], e["tiers"]
    grade_max, cycle, counts = limits["grade_max"], e["report_cycle"], e["counts"]
    steps, figures = [], {}
    fig_label = lambda name: f"{name} blank share by column"

    # 1. What arrived
    figures["p-okr-blank"] = blank_chart(e["okr_blanks"])
    figures["p-jira-blank"] = blank_chart(e["jira_blanks"])
    empty = [k for k, v in e["okr_blanks"].items() if v == 1]
    steps.append(step(1, "What did we receive, and which columns can be used?",
        "Before any cleaning: how big is each file, and how complete is each column?",
        '<div class="two"><div><h3>OKR file: blank share by column</h3>' + chart_slot("p-okr-blank", fig_label("OKR")) +
        '</div><div><h3>Jira file: blank share by column</h3>' + chart_slot("p-jira-blank", fig_label("Jira")) +
        '</div></div><p class="note">Columns with no blanks are not shown. Orange means at least half the column is blank.</p>',
        f"{e['okr_rows']} OKRs and {e['jira_rows']} Jira issues. {' and '.join(empty) or 'No column'} "
        f"{'are' if len(empty) != 1 else 'is'} completely empty. Weight is filled on only {e['weights']} of {e['okr_rows']} OKRs. "
        f"Metric fields are filled on {e['metric_on_kr']} of {e['kr_rows']} Key Results and on {e['metric_filled'] - e['metric_on_kr']} other rows, "
        f"so their blanks are by design. Story points are blank on {share(e['points_low'])} of {e['points_low_type']} rows and {share(e['points_high'])} of {e['points_high_type']} rows.",
        "Use simple averages rather than weighted ones. Do not build KPIs on the empty columns or on story points.",
        "A weighted score would rest on too few weights to defend. Comparing story points would overstate some issue types, because blanks are not spread evenly across them. "
        "A blank is not always a defect, so each was checked before being logged.",
        "All records stay in. The KPIs use only fields that are well filled, or blank for an understood reason.")
    )

    # 2. Reading the sheet
    level_heads = ["Sheet row"] + [f"Column {i + 1}" for i in range(e["indent_count"])]
    steps.append(step(2, "Where does the OKR table actually start?",
        "A standard file read assumes the first row holds the column names. Does it?",
        '<div class="two"><div><h3>Rows above the table</h3>' + table(["Sheet row", "First filled cell"], e["title_rows"]) +
        '</div><div><h3>First rows of the table</h3>' + table(level_heads, e["indent_rows"]) + '</div></div>',
        f"The column names are on sheet row {e['header_row']}, under a title block. The export date is only in the title. "
        f"The OKR key and title sit together in one of {e['indent_count']} columns, and which column shows how deep the row is in the hierarchy.",
        f"Find the header by its marker cell ('{e['settings']['okr_columns']['header_marker']}'). Read the report date from the title. "
        "Split the key from the title, and keep the column position as an indent level. Record each row's source row number.",
        "A fixed 'skip N rows' breaks as soon as the title block changes. Dropping the extra columns would lose the indentation, which later repairs the hierarchy (step 7). "
        "Reading the date from the title means a new export needs no code change.",
        f"{e['okr_rows']} OKR records, each traceable to its sheet row, and a report date of {e['report_date']:%d %B %Y} that drives the default cycle and every 'overdue' and 'stale' figure.")
    )

    # 3. Grain
    collisions = [[a, b, c] for a, b, c in e["collisions"]]
    collision_text = (f"The Jira issue key is not unique: {e['uniqueness'][1][4]} repeats, on {len(collisions)} rows." if collisions else "The Jira issue key is unique.")
    steps.append(step(3, "What is one record, and which key identifies it?",
        "The brief asks for one record per OKR, so we tested each candidate key for repeats.",
        table(["File", "Candidate key", "Filled rows", "Distinct values", "Repeats"], e["uniqueness"], numeric=(2, 3, 4)) +
        ('<h3>The repeated issue key</h3>' + table(["Issue key in file", "OKR key on that row", "Issue key after repair"], collisions) if collisions else ""),
        f"The OKR key has {e['uniqueness'][0][4]} repeats, so the file already holds one row per OKR. {collision_text} "
        f"The pairing DEMO-n to X-n holds on {share(e['pairing'])} of issues that have no collision.",
        "Use the OKR key as the identifier and for the join. Repair a repeated issue key only from that pairing, in a separate column, with a flag. "
        "If an OKR key ever repeats, keep the latest row and save the others to okr_duplicates.csv.",
        f"A naive drop-duplicates on issue key would have deleted {max(len(collisions) - 1, 0)} real issues. The repair runs only if the pairing holds on at least {share(limits['min_pairing_share'])} of issues, so it switches itself off if the pattern breaks.",
        f"{e['okr_rows']} OKR records from {e['okr_rows']} rows ({e['duplicates']} duplicates), and every Jira issue kept with a unique final key.")
    )

    # 4. Dates
    unread = sum(r[-1] for r in e["date_rows"])
    steps.append(step(4, "Can the dates be read as they are?",
        "We counted the date formats in each column, then tried pandas' default date parser on them.",
        table(["File", "Column", "Text month", "dd.mm.yyyy", "ISO", "Blank", "Default parser loses"], e["date_rows"], numeric=(2, 3, 4, 5, 6)),
        f"Each file mixes formats. Month names include {', '.join(e['month_names'][:8])}, so the export depends on the language of whoever downloaded it. "
        f"pandas' default date parser would have silently lost {unread} filled dates.",
        "Read all text as text first, then parse with one parser that knows each format. Month names live in the settings. A filled date it cannot read stops the run and names the value.",
        "Silent blanks are the worst outcome here: an unread due date just disappears from the overdue count, with no error. "
        "A stopped run is annoying, but a wrong number that nobody notices is worse. Formats are matched by pattern, so a day can never be read as a month.",
        "No filled date was lost across either file, and each original value is kept beside the parsed one.")
    )

    # 5. Statuses
    figures["p-status"] = status_chart(e["status_rows"])
    status_table = table(["Label as written", "Report group", "OKR rows", "Jira rows", "Avg grade (OKR)", "Jira rows with a resolution date"],
                         [[repr(r["label"]) if r["label"] != r["label"].strip() else r["label"], r["group"], r["okr"], r["jira"],
                           f"{r['avg_grade']:.1f}", share(r["resolution"])] for r in e["status_rows"]], numeric=(2, 3, 4, 5))
    means = e["group_means"]
    grade_test = (f"they cannot confirm the grouping, since Done averages {means[p.DONE]:.1f} and Not started {means[p.NOT_STARTED]:.1f}, so a higher grade does not mean further along."
                  if means[p.DONE] <= means[p.NOT_STARTED] else f"Done averages {means[p.DONE]:.1f} against {means[p.NOT_STARTED]:.1f} for Not started, which fits the grouping but does not prove it.")
    notes = e["settings"].get("status_group_notes", {})
    meaning = "".join(f"<li><b>{esc(g)}:</b> {esc(notes.get(g, ''))}</li>" for g in STATUS_ORDER)
    steps.append(step(5, "Which statuses mean the same thing?",
        f"The files use {e['labels_raw']} different labels. A chart with that many slices is unreadable, so we asked which ones can honestly be combined.",
        '<div class="two"><div><h3>Each label, coloured by the group it joined</h3>' + chart_slot("p-status", "Bars of rows per raw status label, coloured by report group") +
        '</div><div><h3>Meaning of each group</h3><ul class="meaning">' + meaning + '</ul></div></div>'
        '<h3>Testing the labels against the data</h3>' + status_table +
        f'<h3>What if a label were grouped differently? (cycle {cycle}, {e["total_in_cycle"]} OKRs)</h3>' +
        table(["Label", "Now in group", f"OKRs", "Group share now", "Group share if it moved out"], e["moves"], numeric=(2, 3, 4)),
        f"Four tests. Text: {', '.join(repr(v) for v in e['variants'])} differs from another label only in case or spacing, which takes {e['labels_raw']} labels to {e['labels_clean']}. "
        f"Dates: {share(e['done_resolved'])} of Done Jira rows have a resolution date against {share(e['other_resolved'])} of all others, so Done truly is separate. "
        f"Grades: {grade_test} Team size: with raw labels {share(e['small_raw'][0])} of team and status combinations have fewer than {SMALL_CELL} OKRs; with groups it is {share(e['small_group'][0])}.",
        f"Trim and fix the case first, then map the {e['labels_clean']} labels to {e['labels_group']} groups in the settings. Keep the original label in every row. "
        "Apply one mapping to both files. A new label stops the run until someone assigns it to a group.",
        f"Only the text variants are certain. The rest is a judgement based on what each word means, because grades do not separate the statuses, and the table above shows what each choice costs. "
        f"The two files agree on status for {e['status_agree']} of {e['status_linked']} linked rows, which is why one mapping serves both. "
        "We rejected keeping every label (too thin per team), and guessing a label's meaning from grades (they do not follow status).",
        "The status KPIs and chart use the four groups. The table above is the part to confirm with the OKR owner, since moving one label can change a headline figure.")
    )

    # 6. Grades
    figures["p-grade"] = grade_chart(e["bins"], e["edges"], grade_max)
    values = e["over_values"]
    value_text = (f"exactly {values[0]:g}" if len(values) == 1 else f"between {values[0]:g} and {values[-1]:g}") if values else ""
    big = e["biggest"]
    m = e["metrics"]
    over_text = (f"{e['over_count']} grades sit above {grade_max}, all {value_text}, and {e['over_not_done']} of them are on work not marked Done. " if e["over_count"] else f"No grade is above {grade_max}. ")
    steps.append(step(6, f"Are the grades on a 0–{grade_max} scale, and can they be averaged?",
        "We looked at the full spread of grades, and at whether they follow the Key Result's own progress figures.",
        '<h3>How grades are spread across all OKRs</h3>' + chart_slot("p-grade", "Histogram of OKR grades, with grades above the scale in orange"),
        f"Grades run from {e['grade_min']:g} to {max(values) if values else grade_max:g}, with a median of {e['grade_median']:g}. {over_text}"
        f"For Key Results, comparing the grade with the progress implied by the start, current and target values gives a correlation of {m['corr']}, and {m['gap_over_20']} of {m['comparable']} differ by more than 20 points.",
        f"Treat {grade_max} as the top of the scale, and say clearly that this is an assumption. Keep the grades above it in the data, flag them, and show the average both ways with 'without' as the main number. "
        "Average Key Results only. Warn that grades should not be used to rank teams yet.",
        f"Deleting those rows would hide the issue, and capping them at {grade_max} would invent a value. Using them as they are would let a single record lift a small team. "
        f"Key Results only, because the higher tiers roll up the same work, and including them would count it more than once.",
        f"Cycle {cycle} Key Result average: {e['kr_without']:.1f} without ({e['kr_n_without']} Key Results), {e['kr_with']:.1f} with ({e['kr_n']}). The largest team effect is {big['group']} in {int(big['cycle'])}: {big['avg_with']:.1f} to {big['avg_without']:.1f}.")
    )

    # 7. Hierarchy
    tiers_table = e["tier_levels"].reset_index()
    tier_rows = tiers_table.values.tolist()
    parent_rows = [[k, int(v)] for k, v in e["parents"].items()]
    broken = sum(int(e["parents"].get(k, 0)) for k in ["placeholder", "blank", "unknown key"])
    kr_levels = e["tier_levels"].loc[tiers["key_result"]]
    kr_levels = [str(i) for i, v in kr_levels.items() if v > 0]
    steps.append(step(7, "Does each OKR know its parent?",
        "The parent-child structure must survive. We checked the Parent column, then looked for a second source to repair gaps.",
        '<div class="two"><div><h3>Tier by indent level</h3>' + table(["Tier"] + [f"Level {c}" for c in e["tier_levels"].columns], tier_rows, numeric=tuple(range(1, len(e["tier_levels"].columns) + 1))) +
        '</div><div><h3>State of the Parent column</h3>' + table(["Parent value", "OKRs"], parent_rows, numeric=(1,)) + '</div></div>',
        f"{broken} OKRs have a blank or placeholder parent ({settings_placeholder(e)}). The indentation from step 2 gives a second opinion, and it agrees on {e['agreement']} valid parent links. "
        f"Key Results sit at indent level{'s' if len(kr_levels) != 1 else ''} {' and '.join(kr_levels)}" + (", so a tier name does not fix a row's depth." if len(kr_levels) > 1 else "."),
        f"Take the parent from the column where valid. Otherwise take it from the indentation, in a separate column, flagged for the owner to confirm. '{tiers['top_level']}' rows have no parent by design.",
        f"The two sources agree on {e['agreement']} links we can check, which is why the indentation is trusted where the column is empty. Dropping those rows would have cut off whole branches beneath them. "
        "A rule based on tier name alone was rejected, because the tier does not always fix the depth.",
        f"{len(e['inferred'])} parents inferred ({', '.join(e['inferred'][:4])}{'…' if len(e['inferred']) > 4 else ''}), {e['unresolved']} unresolved, {e['depth']} levels in the final tree.")
    )

    # 8. Teams
    separator = e["settings"]["groups"]["separator"]
    group_rows = [[k, int(v), "Shared by two or more teams" if separator in k else "One team"] for k, v in e["group_values"].items()]
    steps.append(step(8, "What counts as a team?",
        "The brief wants a team breakdown, so we listed every value in the Group column.",
        table(["Group value as written", "OKRs", "Reads as"], group_rows, numeric=(1,)),
        f"The column has {len(group_rows)} values but only {e['teams']} real teams: {e['multi_rows']} OKRs and {e['jira_multi_rows']} Jira issues list several teams in one cell. "
        f"{e['group_blank']} Jira issues have no group.",
        f"Build a bridge table with one row per OKR and team, used for team views only. Totals are always counted from the OKR table. Blank Jira groups are filled from the linked OKR, and flagged.",
        f"Grouping on the raw column would show a fake extra team. Giving a shared OKR to one team would be arbitrary. Splitting it into every team and then summing would double count it, so team rows are not added up to a total.",
        f"{e['okr_rows']} OKRs become {e['bridge_rows']} team rows, and the totals still count {e['okr_rows']}. {e['group_backfilled']} Jira groups filled from the linked OKR.")
    )

    # 9. Join
    recon_rows = e["recon"][["match_category", "explanation", "n"]].values.tolist()
    expected_note = (f"All {e['okr_only_n']} OKR-only rows are the '{tiers['top_level']}' tier" if set(e["okr_only_types"]) == {tiers["top_level"]}
                     else f"OKR-only rows by tier: {counts_text(e['okr_only_types'])}")
    steps.append(step(9, "How do the two files connect, and what will not match?",
        "We joined on the key tested in step 3, then gave every unmatched record a reason.",
        table(["Outcome", "Reason", "Records"], recon_rows, numeric=(2,)),
        f"{e['okr_key_filled']} Jira rows carry an OKR key, and {e['okr_key_found']} of those keys exist in the OKR file. {expected_note}. "
        f"The {e['unlinked_n']} Jira-only rows by type: {counts_text(e['unlinked_types'])}; by label: {counts_text(e['unlinked_labels'])}.",
        "Join on the OKR key, one to one. Keep unmatched records on both sides as named categories and do not force a match. Report unlinked Jira work as its own KPI.",
        "Joining on issue key would break on the repeated keys, and matching on titles would be guesswork. An inner join would have hidden exactly the records the brief asks us to explain. "
        "Unlinked work is a finding in its own right: work the strategy cannot see.",
        f"{e['recon_rows']} reconciled records, equal to {e['okr_rows']} OKRs plus {e['unlinked_n']} Jira-only rows. The run stops if these ever fail to add up.")
    )

    # 10. Trust in time
    steps.append(step(10, "Which dates can we trust for time-based KPIs?",
        f"The export is dated {e['report_date']:%d %B %Y}. We checked each cycle's dates for sequences that cannot be true.",
        table(["Cycle", "OKRs", "Start after export", "Jira issues", "Updated before created", "Created after export", "Freshness shown"], e["cycle_rows"], numeric=(1, 2, 3, 4, 5)),
        f"{counts.get('jira_dates_after_export', 0)} Jira rows carry dates after the export, and {counts.get('updated_before_created', 0)} were updated before they were created. "
        f"{counts.get('future_cycle_progress', 0)} OKRs in a cycle that has not started already show progress. "
        f"The Jira updated date equals the OKR last-update date on {e['same_update']} of {e['same_total']} linked rows, so those are one source and not two.",
        f"Default the report to cycle {cycle} and label later cycles as planning data. Show freshness only for cycles without impossible dates. Take a linked issue's cycle and due date from its OKR, which fills {e['end_blank']} blank and overrides {e['end_differs']} differing Jira target dates, flagged. "
        "Build no cycle-time KPI.",
        f"Blending cycles would mix delivery with plans. Cycle time would be the natural Jira KPI, but it needs dates we have just shown to be unreliable. We first drafted freshness from both files as if they confirmed each other, and dropped that when the dates turned out to be identical.",
        f"The {limits['stale_days']}-day freshness KPI is shown for {', '.join(map(str, e['summary']['freshness_reliable_cycles'])) or 'no cycle'} and labelled as one source. Impossible dates are excluded from overdue counts only; the rows stay.")
    )

    # 11. Final checks
    labels = {"okr_rows_preserved": "Every OKR row is either in the clean table or saved as a duplicate",
              "okr_one_record_per_key": "One record per OKR key",
              "jira_rows_preserved": "Every Jira row is kept",
              "reconciliation_covers_all": "Reconciliation covers every record",
              "report_cycle_in_data": "The report's cycle exists in the data"}
    check_rows = [[labels.get(k, k), "Passed" if v else "Failed"] for k, v in e["summary"]["blocking_checks"].items()]
    warning_text = "; ".join(e["summary"]["warnings"]) or "none"
    steps.append(step(11, "How do we know nothing went wrong?",
        "Every run ends with checks. A failure stops it before any file is overwritten.",
        table(["Check", "This run"], check_rows),
        f"All {len(check_rows)} checks {'passed' if all(r[1] == 'Passed' for r in check_rows) else 'did not pass'}. Warnings: {warning_text}.",
        "Separate problems into two kinds. Those that would make totals wrong stop the run. Those that affect a few rows become warnings and log entries.",
        "Stopping on every imperfection would mean the report never runs, while never stopping would let wrong totals reach a manager. Every issue found along the way is counted in the Data quality tab.",
        f"{e['okr_rows']} OKRs and {e['jira_rows']} Jira issues in, and the same records out: no record was deleted. Clean tables and the log are in data/processed.")
    )

    links = "".join(f'<li><a href="#step-{i + 1}">{esc(title)}</a></li>' for i, title in enumerate(
        ["What arrived", "Where the table starts", "What one record is", "Reading dates", "Grouping statuses", "Grade scale",
         "Parent links", "Teams", "Joining the files", "Trusting dates", "Final checks"]))
    intro = (f'<div class="card"><h2>How the raw exports became the report</h2><p class="sub2">The exploration comes first, and each step shows the evidence, the decision and what it changed. '
             f'All figures are recalculated from the raw files on every run and cover all cycles unless a step says otherwise.</p>'
             f'<ol class="step-links">{links}</ol></div>')
    return intro + "".join(steps), figures


def settings_placeholder(e):
    return f"blank {int(e['parents'].get('blank', 0))}, placeholder {e['settings']['keys']['okr_placeholder']} {int(e['parents'].get('placeholder', 0))}, unknown key {int(e['parents'].get('unknown key', 0))}"
