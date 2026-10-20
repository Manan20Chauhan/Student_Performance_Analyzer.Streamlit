"""
Student Performance Analysis System
===================================
A Streamlit dashboard that loads student marks from a CSV file, cleans and
validates the data, calculates grades / ranks, draws charts and exports
CSV + PDF reports.

Run locally:   streamlit run app.py
Tech stack:    Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, ReportLab
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")  # headless backend (required on Streamlit Cloud)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_TITLE = "Student Performance Analysis System"
SAMPLE_PATH = Path(__file__).parent / "sample_students.csv"
MAX_MARKS = 100  # every subject is marked out of 100

# Grade scale: (minimum percentage, grade). Checked from the top down.
GRADE_SCALE = [(90, "A+"), (80, "A"), (70, "B"), (60, "C"), (40, "D"), (0, "F")]
GRADE_ORDER = [g for _, g in GRADE_SCALE]

TEAL = "#1F6F78"
PASS_COLOR = "#2A9D8F"
FAIL_COLOR = "#D1495B"
AMBER = "#E9A23B"
GRADE_COLORS = {
    "A+": "#14533C",
    "A": "#2A9D8F",
    "B": "#7BBF8E",
    "C": "#E9A23B",
    "D": "#E07B39",
    "F": "#D1495B",
}

MISSING_STRATEGIES = {
    "Fill with subject median": "median",
    "Fill with 0 (treat as absent)": "zero",
    "Drop the student": "drop",
}
ALL_SUBJECTS = "All subjects (overall)"

sns.set_theme(style="whitegrid", font_scale=0.95)


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes, trying common encodings."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode the file: {last_error}")


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def template_csv() -> bytes:
    """A tiny example file users can download and fill in."""
    example = pd.DataFrame(
        {
            "Student_ID": ["S001", "S002", "S003"],
            "Name": ["Student One", "Student Two", "Student Three"],
            "Maths": [78, 56, 91],
            "Science": [65, 48, 88],
            "English": [72, 60, 79],
        }
    )
    return example.to_csv(index=False).encode("utf-8")


_ID_PATTERNS = (r"(^|[_\s\-])(id|roll|reg|enrol)", r"[a-z]ID$")


def _looks_like_id(col: str) -> bool:
    return any(re.search(p, col, flags=re.I if i == 0 else 0) for i, p in enumerate(_ID_PATTERNS))


def detect_columns(raw: pd.DataFrame) -> tuple[str | None, str | None, list[str]]:
    """Guess (id_col, name_col, subject_cols) from the raw dataframe."""
    cols = list(raw.columns)
    name_col = next((c for c in cols if "name" in c.lower()), None)
    id_col = next((c for c in cols if c != name_col and _looks_like_id(c)), None)

    subjects: list[str] = []
    for c in cols:
        if c in (id_col, name_col):
            continue
        non_null = raw[c].dropna()
        if non_null.empty:
            continue
        numeric_share = pd.to_numeric(non_null, errors="coerce").notna().mean()
        if numeric_share >= 0.5:
            subjects.append(c)

    if name_col is None:  # fall back to the first text column that is not a subject
        name_col = next((c for c in cols if c not in subjects and c != id_col), None)
    return id_col, name_col, subjects


# --------------------------------------------------------------------------- #
# 2. Data cleaning & validation
# --------------------------------------------------------------------------- #
def clean_data(
    raw: pd.DataFrame,
    id_col: str | None,
    name_col: str | None,
    subjects: list[str],
    missing_strategy: str = "median",
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Clean the raw marks table.

    Returns (clean_df, logs). The clean dataframe always has the columns
    ``Student_ID`` and ``Name`` followed by the subject columns.
    Each log entry is a (level, message) tuple, level in {success, info, warning, error}.
    """
    logs: list[tuple[str, str]] = []
    keep = [c for c in [id_col, name_col, *subjects] if c]
    df = raw[keep].copy()
    logs.append(("info", f"Loaded {len(df)} rows and {len(raw.columns)} columns; "
                         f"analysing {len(subjects)} subject column(s)."))

    # Tidy text: trim spaces and turn blanks into real missing values.
    for c in (id_col, name_col):
        if c:
            df[c] = df[c].astype("string").str.strip().replace("", pd.NA)

    # --- fully empty rows
    empty = df[keep].isna().all(axis=1)
    if empty.any():
        df = df.loc[~empty]
        logs.append(("warning", f"Removed {int(empty.sum())} completely empty row(s)."))

    # --- exact duplicate rows
    dup_rows = df.duplicated(keep="first")
    if dup_rows.any():
        names = df.loc[dup_rows, name_col].astype(str).tolist() if name_col else []
        extra = f" ({', '.join(names[:5])})" if names else ""
        logs.append(("warning", f"Removed {int(dup_rows.sum())} exact duplicate row(s){extra}."))
        df = df.loc[~dup_rows]

    # --- duplicate student IDs (different data, same ID)
    if id_col:
        dup_ids = df[id_col].notna() & df.duplicated(subset=[id_col], keep="first")
        if dup_ids.any():
            ids = df.loc[dup_ids, id_col].astype(str).tolist()
            logs.append(("warning", f"Removed {int(dup_ids.sum())} row(s) that reused an existing "
                                    f"Student ID (kept the first): {', '.join(ids[:8])}."))
            df = df.loc[~dup_ids]

    df = df.reset_index(drop=True)

    # --- standard ID / Name columns
    df["Student_ID"] = df[id_col] if id_col else pd.Series(
        [f"ROW{i + 1:03d}" for i in range(len(df))], dtype="string")
    df["Name"] = df[name_col] if name_col else df["Student_ID"]
    for i in df.index[df["Name"].isna()]:
        df.loc[i, "Name"] = f"Student {i + 1}"
        logs.append(("warning", f"Row {i + 2}: missing name, labelled 'Student {i + 1}'."))
    for i in df.index[df["Student_ID"].isna()]:
        df.loc[i, "Student_ID"] = f"ROW{i + 1:03d}"
    df["Student_ID"] = df["Student_ID"].astype(str)
    df["Name"] = df["Name"].astype(str)
    df = df[["Student_ID", "Name", *subjects]]

    # --- numeric conversion
    for s in subjects:
        original = df[s]
        converted = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & converted.isna()
        if bad.any():
            samples = ", ".join(f"{df.loc[i, 'Name']} = '{original[i]}'" for i in df.index[bad][:4])
            logs.append(("warning", f"{s}: {int(bad.sum())} non-numeric value(s) treated as missing ({samples})."))
        df[s] = converted.astype(float)

    # --- range check 0..100
    for s in subjects:
        out = df[s].notna() & ((df[s] < 0) | (df[s] > MAX_MARKS))
        if out.any():
            samples = ", ".join(f"{df.loc[i, 'Name']} = {df.loc[i, s]:g}" for i in df.index[out][:4])
            logs.append(("error", f"{s}: {int(out.sum())} mark(s) outside 0-{MAX_MARKS} set to missing ({samples})."))
            df.loc[out, s] = np.nan

    # --- subjects with no usable data at all
    dead = [s for s in subjects if df[s].isna().all()]
    if dead:
        df = df.drop(columns=dead)
        subjects = [s for s in subjects if s not in dead]
        logs.append(("error", f"Dropped subject column(s) with no valid marks: {', '.join(dead)}."))

    # --- students with no marks at all
    no_marks = df[subjects].isna().all(axis=1) if subjects else pd.Series(False, index=df.index)
    if no_marks.any():
        logs.append(("warning", f"Removed {int(no_marks.sum())} student(s) with no valid marks: "
                                f"{', '.join(df.loc[no_marks, 'Name'].head(5))}."))
        df = df.loc[~no_marks]

    # --- missing marks
    missing_total = int(df[subjects].isna().sum().sum()) if subjects else 0
    if missing_total:
        if missing_strategy == "median":
            for s in subjects:
                df[s] = df[s].fillna(df[s].median())
            logs.append(("info", f"Filled {missing_total} missing mark(s) with the subject median."))
        elif missing_strategy == "zero":
            df[subjects] = df[subjects].fillna(0)
            logs.append(("info", f"Filled {missing_total} missing mark(s) with 0."))
        else:
            has_gap = df[subjects].isna().any(axis=1)
            logs.append(("info", f"Dropped {int(has_gap.sum())} student(s) with missing marks."))
            df = df.loc[~has_gap]

    df = df.reset_index(drop=True)
    if len(df) == 0 or not subjects:
        logs.append(("error", "No usable records left after cleaning. Check the column selection."))
    elif not any(level in ("warning", "error") for level, _ in logs):
        logs.append(("success", "No data quality problems found."))
    logs.append(("success", f"Cleaning finished: {len(df)} valid student record(s) ready."))
    return df, logs


# --------------------------------------------------------------------------- #
# 3. Grades, results and ranking
# --------------------------------------------------------------------------- #
def assign_grades(pct: pd.Series) -> pd.Series:
    conditions = [pct >= cut for cut, _ in GRADE_SCALE]
    choices = [g for _, g in GRADE_SCALE]
    return pd.Series(np.select(conditions, choices, default="F"), index=pct.index)


def compute_results(df: pd.DataFrame, subjects: list[str], pass_mark: float) -> pd.DataFrame:
    """Add Total, Percentage, Grade, Result, Rank and Percentile columns."""
    res = df.copy()
    res["Total"] = res[subjects].sum(axis=1)
    res["Percentage"] = (res["Total"] / (len(subjects) * MAX_MARKS) * 100).round(2)

    failed = res[subjects].lt(pass_mark)
    res["Failed Subjects"] = failed.apply(lambda row: ", ".join(row.index[row]), axis=1).replace("", "-")
    res["Result"] = np.where(failed.any(axis=1), "Fail", "Pass")

    # Grade follows the percentage, but a student who fails any subject gets F.
    res["Grade"] = assign_grades(res["Percentage"])
    res.loc[res["Result"] == "Fail", "Grade"] = "F"

    res["Rank"] = res["Total"].rank(method="min", ascending=False).astype(int)
    res["Percentile"] = (res["Total"].rank(method="average", pct=True) * 100).round(1)
    cols = ["Rank", "Student_ID", "Name", *subjects, "Total", "Percentage",
            "Grade", "Result", "Failed Subjects", "Percentile"]
    return res[cols].sort_values(["Rank", "Name"]).reset_index(drop=True)


def subject_statistics(res: pd.DataFrame, subjects: list[str], pass_mark: float) -> pd.DataFrame:
    rows = []
    for s in subjects:
        col = res[s]
        top = col.max()
        toppers = res.loc[col == top, "Name"].tolist()
        rows.append(
            {
                "Subject": s,
                "Class Average": round(col.mean(), 2),
                "Median": round(col.median(), 2),
                "Std Dev": round(col.std(ddof=1), 2) if len(col) > 1 else 0.0,
                "Highest": top,
                "Lowest": col.min(),
                "Pass %": round((col >= pass_mark).mean() * 100, 1),
                "Topper": ", ".join(toppers),
            }
        )
    return pd.DataFrame(rows)


def overall_statistics(res: pd.DataFrame) -> dict:
    pct = res["Percentage"]
    passed = int((res["Result"] == "Pass").sum())
    return {
        "students": int(len(res)),
        "mean": round(float(pct.mean()), 2),
        "median": round(float(pct.median()), 2),
        "std": round(float(pct.std(ddof=1)), 2) if len(pct) > 1 else 0.0,
        "highest": round(float(pct.max()), 2),
        "lowest": round(float(pct.min()), 2),
        "pass_pct": round(passed / len(res) * 100, 1),
        "passed": passed,
        "failed": int(len(res) - passed),
    }


def style_table(df: pd.DataFrame):
    """Colour the Result / Grade columns (works on pandas 1.x - 3.x)."""

    def colour(value):
        if value == "Pass":
            return f"color: {PASS_COLOR}; font-weight: 600"
        if value in ("Fail", "F"):
            return f"color: {FAIL_COLOR}; font-weight: 600"
        return ""

    styler = df.style
    apply_cells = getattr(styler, "map", None) or getattr(styler, "applymap")
    target = [c for c in ("Result", "Grade") if c in df.columns]
    return apply_cells(colour, subset=target) if target else styler


def insights(res: pd.DataFrame, stats: pd.DataFrame, overall: dict) -> list[str]:
    """Short plain-English observations for the overview and the report."""
    best = stats.loc[stats["Class Average"].idxmax()]
    worst = stats.loc[stats["Class Average"].idxmin()]
    hardest_pass = stats.loc[stats["Pass %"].idxmin()]
    top = res.iloc[0]
    out = [
        f"{top['Name']} ranks first with {top['Percentage']:.2f}% ({top['Grade']}).",
        f"Strongest subject: {best['Subject']} (class average {best['Class Average']:.1f}).",
        f"Weakest subject: {worst['Subject']} (class average {worst['Class Average']:.1f}).",
        f"Lowest pass rate: {hardest_pass['Subject']} at {hardest_pass['Pass %']:.1f}%.",
        f"{overall['passed']} of {overall['students']} students passed every subject "
        f"({overall['pass_pct']:.1f}%).",
    ]
    return out


# --------------------------------------------------------------------------- #
# 4. Charts (Matplotlib + Seaborn)
# --------------------------------------------------------------------------- #
def fig_subject_average(stats: pd.DataFrame, pass_mark: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = stats.sort_values("Class Average", ascending=False)
    sns.barplot(data=order, x="Subject", y="Class Average", hue="Subject",
                palette=sns.light_palette(TEAL, n_colors=len(order) + 2)[::-1][:len(order)],
                legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
    ax.axhline(pass_mark, color=FAIL_COLOR, ls="--", lw=1.2, label=f"Pass mark ({pass_mark:g})")
    ax.set_ylim(0, 105)
    ax.set_xlabel("")
    ax.set_ylabel("Class average")
    ax.set_title("Subject-wise class average")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_pass_fail_pie(overall: dict):
    sizes = [overall["passed"], overall["failed"]]
    labels = ["Pass", "Fail"]
    palette = [PASS_COLOR, FAIL_COLOR]
    keep = [i for i, v in enumerate(sizes) if v > 0]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.pie([sizes[i] for i in keep], labels=[labels[i] for i in keep],
           colors=[palette[i] for i in keep], autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 10})
    ax.set_title("Pass / fail ratio")
    fig.tight_layout()
    return fig


def fig_histogram(res: pd.DataFrame, column: str, pass_mark: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = min(10, max(4, len(res) // 2))
    sns.histplot(res[column], bins=bins, kde=len(res) > 3, color=TEAL, edgecolor="white", ax=ax)
    mean = res[column].mean()
    ax.axvline(mean, color=AMBER, lw=2, label=f"Mean = {mean:.1f}")
    if column != "Percentage":
        ax.axvline(pass_mark, color=FAIL_COLOR, ls="--", lw=1.2, label=f"Pass mark ({pass_mark:g})")
    ax.set_xlabel("Percentage" if column == "Percentage" else f"{column} marks")
    ax.set_ylabel("Students")
    ax.set_title(f"Score distribution: {'overall percentage' if column == 'Percentage' else column}")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_boxplot(res: pd.DataFrame, subjects: list[str], pass_mark: float):
    long = res[subjects].melt(var_name="Subject", value_name="Marks")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=long, x="Subject", y="Marks", hue="Subject", palette="crest",
                legend=False, ax=ax, fliersize=4)
    ax.axhline(pass_mark, color=FAIL_COLOR, ls="--", lw=1.2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("")
    ax.set_title("Marks spread by subject")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_grade_distribution(res: pd.DataFrame):
    counts = res["Grade"].value_counts().reindex(GRADE_ORDER, fill_value=0).reset_index()
    counts.columns = ["Grade", "Students"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(data=counts, x="Grade", y="Students", hue="Grade", palette=GRADE_COLORS,
                legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, padding=2, fontsize=9)
    ax.set_title("Grade distribution")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


def fig_correlation(res: pd.DataFrame, subjects: list[str]):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(res[subjects].corr(), annot=True, fmt=".2f", cmap="crest", vmin=-1, vmax=1,
                linewidths=1, linecolor="white", ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("How subjects move together (correlation)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return buf.getvalue()


def show(fig) -> None:
    """Render a Matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Reports (text + PDF)
# --------------------------------------------------------------------------- #
def build_text_report(res, stats, overall, subjects, pass_mark, source_name) -> str:
    lines = [
        f"# {APP_TITLE}: summary report",
        f"Generated: {datetime.now():%d %b %Y, %H:%M}  |  Source: {source_name}",
        "",
        "## Class overview",
        f"- Students analysed: {overall['students']}",
        f"- Subjects: {', '.join(subjects)}",
        f"- Pass mark (per subject): {pass_mark:g}",
        f"- Average percentage: {overall['mean']}%  |  Median: {overall['median']}%  |  Std dev: {overall['std']}",
        f"- Highest / lowest percentage: {overall['highest']}% / {overall['lowest']}%",
        f"- Pass percentage: {overall['pass_pct']}% ({overall['passed']} passed, {overall['failed']} failed)",
        "",
        "## Subject-wise statistics",
    ]
    for _, r in stats.iterrows():
        lines.append(
            f"- {r['Subject']}: average {r['Class Average']}, median {r['Median']}, "
            f"std dev {r['Std Dev']}, pass {r['Pass %']}%, topper {r['Topper']} ({r['Highest']:g})"
        )
    lines += ["", "## Top 5 students"]
    for _, r in res.head(5).iterrows():
        lines.append(f"{int(r['Rank'])}. {r['Name']}: {r['Percentage']}% ({r['Grade']})")
    lines += ["", "## Observations"] + [f"- {t}" for t in insights(res, stats, overall)]
    return "\n".join(lines)


def build_pdf(res, stats, overall, subjects, pass_mark, source_name, logs) -> bytes:
    """Create a formatted multi-page PDF report with ReportLab."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=APP_TITLE, author="Student Performance Analysis System")
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=base["Title"], textColor=colors.HexColor(TEAL), fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], textColor=colors.HexColor(TEAL), spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)
    usable_w = A4[0] - 3.2 * cm

    def table(data, col_widths=None, zebra=True, font=8.5):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D3D2")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if zebra:
            for i in range(1, len(data)):
                if i % 2 == 0:
                    style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F1F6F5")))
        t.setStyle(TableStyle(style))
        return t

    def picture(fig, width):
        png = fig_to_png(fig)
        plt.close(fig)
        reader_buf = io.BytesIO(png)
        w, h = plt.imread(io.BytesIO(png)).shape[1::-1]
        return Image(reader_buf, width=width, height=width * h / w)

    story = [
        Paragraph(APP_TITLE, h1),
        Paragraph(f"Performance report &nbsp;|&nbsp; Source: {escape(str(source_name))} &nbsp;|&nbsp; "
                  f"Generated {datetime.now():%d %b %Y, %H:%M}", small),
        Spacer(1, 8),
        Paragraph("Class overview", h2),
    ]

    kpis = [
        ["Students", "Average %", "Median %", "Std dev", "Highest %", "Pass %"],
        [overall["students"], overall["mean"], overall["median"], overall["std"],
         overall["highest"], f"{overall['pass_pct']}%"],
    ]
    story.append(table(kpis, zebra=False, font=10))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Subjects: {escape(', '.join(subjects))}. A student passes only when every subject is at or "
        f"above {pass_mark:g}. Grades: A+ (90+), A (80+), B (70+), C (60+), D (40+), F (below 40 or any "
        f"subject failed).", body))

    story.append(Paragraph("Observations", h2))
    for line in insights(res, stats, overall):
        story.append(Paragraph(f"&bull; {escape(line)}", body))

    story.append(Paragraph("Subject-wise statistics", h2))
    srows = [["Subject", "Average", "Median", "Std dev", "Highest", "Lowest", "Pass %", "Topper"]]
    for _, r in stats.iterrows():
        srows.append([r["Subject"], r["Class Average"], r["Median"], r["Std Dev"],
                      f"{r['Highest']:g}", f"{r['Lowest']:g}", f"{r['Pass %']}%", r["Topper"][:28]])
    story.append(table(srows, col_widths=[3.2 * cm, 1.8 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm,
                                          1.6 * cm, usable_w - 13.4 * cm]))

    story.append(Paragraph("Top 10 students", h2))
    trows = [["Rank", "ID", "Name", "Total", "%", "Grade", "Result"]]
    for _, r in res.head(10).iterrows():
        trows.append([int(r["Rank"]), r["Student_ID"], r["Name"], f"{r['Total']:g}",
                      f"{r['Percentage']:.2f}", r["Grade"], r["Result"]])
    story.append(table(trows))

    story.append(PageBreak())
    story.append(Paragraph("Charts", h2))
    half = usable_w * 0.48
    row1 = Table([[picture(fig_subject_average(stats, pass_mark), half),
                   picture(fig_pass_fail_pie(overall), half * 0.85)]], colWidths=[usable_w * 0.55, usable_w * 0.45])
    row1.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(row1)
    story.append(Spacer(1, 6))
    row2 = Table([[picture(fig_histogram(res, "Percentage", pass_mark), half),
                   picture(fig_grade_distribution(res), half)]], colWidths=[usable_w * 0.5, usable_w * 0.5])
    row2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(row2)
    story.append(Spacer(1, 6))
    story.append(picture(fig_boxplot(res, subjects, pass_mark), usable_w * 0.7))

    story.append(PageBreak())
    story.append(Paragraph("Complete results", h2))
    rrows = [["Rank", "ID", "Name", *[s.replace(" ", "\n") for s in subjects], "Total", "%", "Grade", "Result"]]
    for _, r in res.iterrows():
        rrows.append([int(r["Rank"]), r["Student_ID"], r["Name"][:20], *[f"{r[s]:g}" for s in subjects],
                      f"{r['Total']:g}", f"{r['Percentage']:.1f}", r["Grade"], r["Result"]])
    story.append(table(rrows, font=7.5 if len(subjects) > 5 else 8.5))

    warn = [m for lvl, m in logs if lvl in ("warning", "error")]
    story.append(Paragraph("Data quality notes", h2))
    if warn:
        for m in warn:
            story.append(Paragraph(f"&bull; {escape(m)}", body))
    else:
        story.append(Paragraph("No data quality problems were found in the source file.", body))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.6 * cm, 0.9 * cm, APP_TITLE)
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# 6. Streamlit UI
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1250px; }}
[data-testid="stMetric"] {{
    background: #F1F6F5; border-left: 4px solid {TEAL};
    padding: 0.7rem 0.9rem; border-radius: 6px;
}}
[data-testid="stMetricLabel"] p {{ font-size: 0.82rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; overflow-x: auto; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.45rem 0.9rem; white-space: nowrap; }}
.rank-card {{
    border: 1px solid #D5E2E0; border-radius: 8px; padding: 0.8rem 1rem; height: 100%;
}}
.rank-card b {{ color: {TEAL}; }}
@media (max-width: 640px) {{
    .block-container {{ padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.2rem; }}
    h1 {{ font-size: 1.6rem !important; }}
}}
</style>
"""


def render_logs(logs: list[tuple[str, str]]) -> None:
    for level, message in logs:
        {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}[level](message)


def sidebar() -> dict | None:
    """Draw the sidebar and return the chosen configuration (or None if no data yet)."""
    st.sidebar.title("Controls")
    source = st.sidebar.radio("Data source", ["Sample dataset", "Upload CSV"], horizontal=False)

    raw, source_name = None, ""
    if source == "Sample dataset":
        if SAMPLE_PATH.exists():
            raw, source_name = load_sample(), "sample_students.csv"
            st.sidebar.caption("The sample file has a few deliberate errors (a duplicate, a missing "
                               "value, out-of-range and text marks) so you can see the cleaning log.")
        else:
            st.sidebar.error("sample_students.csv was not found next to app.py.")
    else:
        up = st.sidebar.file_uploader("Upload a .csv file", type=["csv"])
        st.sidebar.download_button("Download CSV template", template_csv(), "marks_template.csv",
                                   "text/csv", help="One row per student, one column per subject.")
        if up is not None:
            try:
                raw, source_name = read_csv_bytes(up.getvalue()), up.name
            except Exception as exc:  # unreadable / empty / not a CSV
                st.sidebar.error(f"Could not read this file: {exc}")
        else:
            st.sidebar.info("Upload a CSV with a name column and one numeric column per subject.")

    if raw is None:
        return None
    if raw.empty or len(raw.columns) < 2:
        st.sidebar.error("The file needs at least a name column and one subject column.")
        return None

    id_guess, name_guess, subj_guess = detect_columns(raw)
    sig = f"{source_name}-{'|'.join(raw.columns)}"
    options = list(raw.columns)
    none_opt = "(none)"

    with st.sidebar.expander("Column mapping", expanded=False):
        name_col = st.selectbox("Name column", options,
                                index=options.index(name_guess) if name_guess in options else 0,
                                key=f"name-{sig}")
        id_choices = [none_opt] + [c for c in options if c != name_col]
        id_col = st.selectbox("Student ID column", id_choices,
                              index=id_choices.index(id_guess) if id_guess in id_choices else 0,
                              key=f"id-{sig}")
        sub_options = [c for c in options if c not in (name_col, id_col)]
        subjects = st.multiselect("Subject columns", sub_options,
                                  default=[s for s in subj_guess if s in sub_options],
                                  key=f"sub-{sig}")

    st.sidebar.subheader("Rules")
    pass_mark = st.sidebar.slider("Pass mark per subject", 0, 100, 40, 1)
    strategy_label = st.sidebar.selectbox("Missing marks", list(MISSING_STRATEGIES))
    with st.sidebar.expander("Grading scale"):
        st.table(pd.DataFrame({"Grade": ["A+", "A", "B", "C", "D", "F"],
                               "Percentage": ["90 and above", "80 - 89.99", "70 - 79.99",
                                              "60 - 69.99", "40 - 59.99", "Below 40 / any subject failed"]}
                              ).set_index("Grade"))

    return {
        "raw": raw, "source_name": source_name, "id_col": None if id_col == none_opt else id_col,
        "name_col": name_col, "subjects": subjects, "pass_mark": pass_mark,
        "strategy": MISSING_STRATEGIES[strategy_label],
    }


def tab_overview(res, stats, overall, subjects):
    c = st.columns(3)
    c[0].metric("Students", overall["students"])
    c[1].metric("Class average", f"{overall['mean']}%")
    c[2].metric("Pass percentage", f"{overall['pass_pct']}%")
    c = st.columns(3)
    c[0].metric("Median", f"{overall['median']}%")
    c[1].metric("Std deviation", overall["std"])
    c[2].metric("Highest score", f"{overall['highest']}%")

    st.subheader("Top performers")
    top = res.head(3)
    cols = st.columns(len(top))
    for col, (_, r) in zip(cols, top.iterrows()):
        col.markdown(
            f"<div class='rank-card'>Rank <b>{int(r['Rank'])}</b><br><b style='font-size:1.1rem'>"
            f"{escape(r['Name'])}</b><br>{r['Percentage']:.2f}% &nbsp;|&nbsp; Grade {r['Grade']}</div>",
            unsafe_allow_html=True)

    st.subheader("What stands out")
    for line in insights(res, stats, overall):
        st.markdown(f"- {line}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Subject averages**")
        st.bar_chart(stats.set_index("Subject")["Class Average"], color=TEAL)
    with right:
        st.markdown("**Grade distribution**")
        counts = res["Grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
        st.bar_chart(counts, color=AMBER)


def tab_data(raw, clean, logs, subjects, res):
    cleaned_rows = len(clean)
    n_warn = sum(1 for lvl, _ in logs if lvl in ("warning", "error"))
    c = st.columns(3)
    c[0].metric("Rows in file", len(raw))
    c[1].metric("Valid records", cleaned_rows)
    c[2].metric("Issues found", n_warn)

    st.subheader("Cleaning log")
    render_logs(logs)

    with st.expander("Raw data as uploaded", expanded=False):
        st.dataframe(raw, hide_index=True)
    st.subheader("Cleaned data")
    st.dataframe(clean, hide_index=True)
    st.download_button("Download cleaned data (CSV)", clean.to_csv(index=False).encode("utf-8"),
                       "cleaned_students.csv", "text/csv")


def tab_stats(res, stats, overall, subjects):
    st.subheader("Subject-wise statistics")
    st.dataframe(stats, hide_index=True)
    st.caption("Std Dev uses the sample formula (n - 1). Pass % counts students at or above the pass mark.")

    st.subheader("Subject toppers")
    cols = st.columns(min(3, len(stats)))
    for i, (_, r) in enumerate(stats.iterrows()):
        cols[i % len(cols)].metric(r["Subject"], f"{r['Highest']:g}", r["Topper"], delta_color="off")

    st.subheader("Overall percentage")
    summary = pd.DataFrame(
        {"Measure": ["Mean", "Median", "Standard deviation", "Highest", "Lowest", "Pass percentage"],
         "Value": [overall["mean"], overall["median"], overall["std"], overall["highest"],
                   overall["lowest"], overall["pass_pct"]]})
    st.dataframe(summary, hide_index=True)


def tab_charts(res, stats, overall, subjects, pass_mark):
    a, b = st.columns(2)
    with a:
        show(fig_subject_average(stats, pass_mark))
    with b:
        show(fig_pass_fail_pie(overall))

    choice = st.selectbox("Distribution for", ["Percentage", *subjects], key="hist-choice")
    a, b = st.columns(2)
    with a:
        show(fig_histogram(res, choice, pass_mark))
    with b:
        show(fig_boxplot(res, subjects, pass_mark))

    a, b = st.columns(2)
    with a:
        show(fig_grade_distribution(res))
    with b:
        if len(subjects) >= 2 and len(res) >= 3:
            show(fig_correlation(res, subjects))
        else:
            st.info("The correlation heatmap needs at least two subjects and three students.")

    st.markdown("**Pass rate by subject (Streamlit chart)**")
    st.line_chart(stats.set_index("Subject")["Pass %"], color=PASS_COLOR)


def tab_ranking(res, subjects):
    st.subheader("Rank list")
    f1, f2 = st.columns([2, 1])
    query = f1.text_input("Search by name or ID", placeholder="e.g. Aarav or S004")
    subject = f2.selectbox("Subject", [ALL_SUBJECTS, *subjects])

    g1, g2, g3 = st.columns(3)
    grades = g1.multiselect("Grade", GRADE_ORDER)
    result = g2.selectbox("Result", ["All", "Pass", "Fail"])
    min_pct = g3.slider("Minimum percentage", 0, 100, 0)

    view = res.copy()
    if subject != ALL_SUBJECTS:
        view[f"{subject} Rank"] = view[subject].rank(method="min", ascending=False).astype(int)
    if query.strip():
        q = query.strip().lower()
        view = view[view["Name"].str.lower().str.contains(q, regex=False)
                    | view["Student_ID"].str.lower().str.contains(q, regex=False)]
    if grades:
        view = view[view["Grade"].isin(grades)]
    if result != "All":
        view = view[view["Result"] == result]
    view = view[view["Percentage"] >= min_pct]

    sort_options = ["Overall rank", "Percentage", "Percentile", "Name"]
    if subject != ALL_SUBJECTS:
        sort_options.insert(1, f"{subject} marks")
    s1, s2 = st.columns([2, 1])
    sort_by = s1.selectbox("Sort by", sort_options,
                           index=1 if subject != ALL_SUBJECTS else 0)
    best_first = s2.toggle("Best first", value=True, help="Rank 1 at the top, highest marks first, names A to Z.")

    if sort_by == "Overall rank":
        view = view.sort_values(["Rank", "Name"], ascending=[best_first, True])
    elif sort_by == "Name":
        view = view.sort_values("Name", ascending=best_first)
    else:
        key = f"{subject}" if sort_by.endswith("marks") else sort_by
        view = view.sort_values(key, ascending=not best_first)

    if subject != ALL_SUBJECTS:
        lead = ["Rank", "Student_ID", "Name", subject, f"{subject} Rank"]
        view = view[lead + [c for c in view.columns if c not in lead]]

    st.caption(f"Showing {len(view)} of {len(res)} students.")
    if view.empty:
        st.warning("No students match these filters. Clear the search or relax the filters.")
    else:
        st.dataframe(style_table(view), hide_index=True)
        st.download_button("Download this view (CSV)", view.to_csv(index=False).encode("utf-8"),
                           "filtered_ranking.csv", "text/csv")

    st.subheader("Student profile")
    labels = (res["Name"] + " (" + res["Student_ID"] + ")").tolist()
    pick = st.selectbox("Choose a student", labels, key="profile-pick")
    row = res.iloc[labels.index(pick)]
    m = st.columns(4)
    m[0].metric("Rank", f"{int(row['Rank'])} of {len(res)}")
    m[1].metric("Percentage", f"{row['Percentage']:.2f}%")
    m[2].metric("Grade", row["Grade"])
    m[3].metric("Percentile", f"{row['Percentile']:.1f}")
    compare = pd.DataFrame({"Student": [row[s] for s in subjects],
                            "Class average": [res[s].mean() for s in subjects]}, index=subjects)
    st.bar_chart(compare, color=[TEAL, AMBER])
    if row["Result"] == "Fail":
        st.warning(f"Needs attention in: {row['Failed Subjects']}")


def tab_reports(res, stats, overall, subjects, pass_mark, source_name, logs):
    st.subheader("Export results")
    st.download_button("Download processed results (CSV)", res.to_csv(index=False).encode("utf-8"),
                       "student_results.csv", "text/csv")

    st.subheader("PDF report")
    st.write("A formatted report with class overview, subject statistics, top students, charts, "
             "the full result list and data-quality notes.")
    sig = (source_name, len(res), float(res["Total"].sum()), pass_mark)
    if st.button("Generate PDF report"):
        with st.spinner("Building the PDF..."):
            st.session_state["pdf_bytes"] = build_pdf(res, stats, overall, subjects, pass_mark,
                                                      source_name, logs)
            st.session_state["pdf_sig"] = sig
    if st.session_state.get("pdf_sig") == sig and "pdf_bytes" in st.session_state:
        st.success("Your PDF is ready.")
        st.download_button("Download PDF report", st.session_state["pdf_bytes"],
                           "student_performance_report.pdf", "application/pdf")

    st.subheader("Summary report")
    text = build_text_report(res, stats, overall, subjects, pass_mark, source_name)
    st.markdown(text)
    st.download_button("Download summary (Markdown)", text.encode("utf-8"),
                       "summary_report.md", "text/markdown")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption("Upload marks, clean the data, and get grades, ranks, charts and reports.")

    cfg = sidebar()
    if cfg is None:
        st.info("Pick the sample dataset or upload a CSV from the sidebar to begin. "
                "On a phone, open the sidebar with the arrow at the top left.")
        return
    if not cfg["subjects"]:
        st.warning("Choose at least one subject column under Column mapping in the sidebar.")
        return

    clean, logs = clean_data(cfg["raw"], cfg["id_col"], cfg["name_col"], cfg["subjects"], cfg["strategy"])
    subjects = [s for s in cfg["subjects"] if s in clean.columns]
    if clean.empty or not subjects:
        render_logs(logs)
        return

    res = compute_results(clean, subjects, cfg["pass_mark"])
    stats = subject_statistics(res, subjects, cfg["pass_mark"])
    overall = overall_statistics(res)

    tabs = st.tabs(["Overview", "Data & Cleaning", "Statistics", "Charts", "Ranking & Search", "Reports"])
    with tabs[0]:
        tab_overview(res, stats, overall, subjects)
    with tabs[1]:
        tab_data(cfg["raw"], clean, logs, subjects, res)
    with tabs[2]:
        tab_stats(res, stats, overall, subjects)
    with tabs[3]:
        tab_charts(res, stats, overall, subjects, cfg["pass_mark"])
    with tabs[4]:
        tab_ranking(res, subjects)
    with tabs[5]:
        tab_reports(res, stats, overall, subjects, cfg["pass_mark"], cfg["source_name"], logs)


if __name__ == "__main__":
    main()
