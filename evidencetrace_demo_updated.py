
import re
import html
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pandas as pd
import numpy as np
import gradio as gr
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# EvidenceTrace — citation/evidence traceability prototype
# ============================================================

# ============================================================
# Audit trail / activity log
# ============================================================
# SQLite is used because it is dependency-free and provides a
# real application-level audit trail. No uploaded PDF contents
# are stored; only metadata and analysis outcomes are recorded.
#
# For a public deployment, set EVIDENCETRACE_ADMIN_PIN as an
# environment secret. The fallback PIN is intended only for the
# research/demo environment.
# ============================================================

AUDIT_DB = Path(
    os.getenv("EVIDENCETRACE_AUDIT_DB", "evidencetrace_audit.db")
)
ADMIN_PIN = os.getenv("EVIDENCETRACE_ADMIN_PIN", "1234")


def db_connect():
    conn = sqlite3.connect(
        AUDIT_DB,
        check_same_thread=False,
        timeout=10,
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_audit_db():
    conn = db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                timestamp_utc TEXT NOT NULL,
                session_id TEXT,
                event_type TEXT NOT NULL,
                file_name TEXT,
                file_size_bytes INTEGER,
                pages INTEGER,
                claims INTEGER,
                citations INTEGER,
                references_count INTEGER,
                resolved_citations INTEGER,
                unresolved_citations INTEGER,
                supported_claims INTEGER,
                weak_claims INTEGER,
                unsupported_claims INTEGER,
                high_findings INTEGER,
                medium_findings INTEGER,
                publication_gate TEXT,
                integrity_score INTEGER,
                report_generated INTEGER DEFAULT 0,
                error_message TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def audit_write(
    *,
    event_type,
    session_id="",
    file_name="",
    file_size_bytes=None,
    pages=None,
    claims=None,
    citations=None,
    references_count=None,
    resolved_citations=None,
    unresolved_citations=None,
    supported_claims=None,
    weak_claims=None,
    unsupported_claims=None,
    high_findings=None,
    medium_findings=None,
    publication_gate="",
    integrity_score=None,
    report_generated=False,
    error_message="",
):
    init_audit_db()
    conn = db_connect()
    try:
        conn.execute(
            """
            INSERT INTO audit_events (
                event_id, timestamp_utc, session_id, event_type,
                file_name, file_size_bytes, pages, claims, citations,
                references_count, resolved_citations, unresolved_citations,
                supported_claims, weak_claims, unsupported_claims,
                high_findings, medium_findings, publication_gate,
                integrity_score, report_generated, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                session_id,
                event_type,
                file_name,
                file_size_bytes,
                pages,
                claims,
                citations,
                references_count,
                resolved_citations,
                unresolved_citations,
                supported_claims,
                weak_claims,
                unsupported_claims,
                high_findings,
                medium_findings,
                publication_gate,
                integrity_score,
                1 if report_generated else 0,
                error_message,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_audit_dataframe(limit=500):
    init_audit_db()
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT
                id, event_id, timestamp_utc, session_id, event_type,
                file_name, pages, claims, citations, references_count,
                resolved_citations, unresolved_citations,
                supported_claims, weak_claims, unsupported_claims,
                high_findings, medium_findings, publication_gate,
                integrity_score, report_generated, error_message
            FROM audit_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()

    columns = [
        "id", "event_id", "timestamp_utc", "session_id", "event_type",
        "file_name", "pages", "claims", "citations", "references_count",
        "resolved_citations", "unresolved_citations",
        "supported_claims", "weak_claims", "unsupported_claims",
        "high_findings", "medium_findings", "publication_gate",
        "integrity_score", "report_generated", "error_message",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame([dict(row) for row in rows], columns=columns)


def clear_audit_log(pin):
    if str(pin or "").strip() != ADMIN_PIN:
        return "❌ Incorrect administrator PIN.", load_audit_dataframe()

    init_audit_db()
    conn = db_connect()
    try:
        conn.execute("DELETE FROM audit_events")
        conn.commit()
    finally:
        conn.close()

    audit_write(event_type="ADMIN_LOG_CLEARED", session_id="ADMIN")
    return "✅ Audit log cleared.", load_audit_dataframe()


init_audit_db()

# This is a research prototype. Semantic similarity is only a
# screening signal; it does NOT prove factual correctness.
# Human verification remains mandatory.
# ============================================================

REFERENCE_HEADINGS = re.compile(
    r"^\s*(references?|reference\s+list|reference\s+register|"
    r"bibliography|works\s+cited|sources?)\s*:?\s*$",
    re.IGNORECASE,
)

CITATION_RE = re.compile(r"\[(\d{1,4}(?:\s*[-,;]\s*\d{1,4})*)\]")
NUMBERED_REF_RE = re.compile(
    r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)\-:])\s+(.+?)\s*$"
)

HIGH_RISK_PATTERNS = [
    r"\bzero\s+(?:residual\s+)?(?:cybersecurity\s+)?risk\b",
    r"\bzero\s+risk\b",
    r"\bfully\s+compliant\b",
    r"\bfully\s+effective\b",
    r"\bcompletely\s+(?:eliminated|mitigated|secure|protected)\b",
    r"\balways\b",
    r"\bnever\b",
    r"\b100%\b",
    r"\bno\s+risk\b",
    r"\bno\s+cybersecurity\s+risk\b",
]

CLAIM_TERMS = re.compile(
    r"\b("
    r"maintains?|implemented?|enforced?|performed?|"
    r"reviewed?|monitored?|protected?|encrypted?|"
    r"tested?|approved?|documented?|tracked?|"
    r"effective|compliant|secure|eliminated|"
    r"ensured?|prevents?|detects?|supports?|"
    r"requires?|subject to|concludes?"
    r")\b",
    re.IGNORECASE,
)


def extract_pdf(path):
    doc = pymupdf.open(path)
    pages = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append({"page": page_no, "text": text})
    doc.close()
    return pages


def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def expand_citation_group(group):
    """
    Expand [1], [1,2], [1-3] into integer citation IDs.
    """
    ids = []
    for part in re.split(r"\s*[,;]\s*", group):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"\d+\s*-\s*\d+", part):
            a, b = [int(x) for x in re.split(r"\s*-\s*", part)]
            if a <= b and b - a <= 1000:
                ids.extend(range(a, b + 1))
        elif part.isdigit():
            ids.append(int(part))
    return sorted(set(ids))


def extract_references(pages):
    """
    Finds common reference-section headings and supports:
      [1] Reference...
      1. Reference...
      1) Reference...
      1 - Reference...
    Also joins continuation lines to the current reference.
    """
    in_refs = False
    refs = {}
    current_num = None
    current_text = []

    def flush():
        nonlocal current_num, current_text
        if current_num is not None:
            txt = normalize_space(" ".join(current_text))
            if txt:
                refs[current_num] = txt
        current_num = None
        current_text = []

    for p in pages:
        for raw_line in p["text"].splitlines():
            line = raw_line.strip()

            if not in_refs and REFERENCE_HEADINGS.match(line):
                in_refs = True
                continue

            if not in_refs or not line:
                continue

            m = NUMBERED_REF_RE.match(line)
            if m:
                flush()
                num = int(m.group(1) or m.group(2))
                text = m.group(3)
                current_num = num
                current_text = [text]
            else:
                # Ignore obvious page/footer noise.
                if re.fullmatch(r"Page\s+\d+(?:\s+of\s+\d+)?", line, re.I):
                    continue
                if current_num is not None:
                    current_text.append(line)

    flush()
    return dict(sorted(refs.items()))


def extract_citations(pages):
    citations = []
    for p in pages:
        for match in CITATION_RE.finditer(p["text"]):
            ids = expand_citation_group(match.group(1))
            for ref_id in ids:
                citations.append(
                    {
                        "page": p["page"],
                        "ref_id": ref_id,
                        "raw": match.group(0),
                    }
                )
    return citations


def extract_claims(pages):
    """
    Lightweight claim detector with citation-aware sentence handling.

    Important fix:
    Citations such as:
        "... periodic management review. [1] [2]"
    were previously split into two sentence fragments because the
    sentence splitter saw the period before [1]. This caused the
    claim to lose its citation IDs.

    We temporarily replace citation tokens with placeholders before
    sentence splitting, then restore them. This keeps the citation
    attached to the claim.
    """
    claims = []
    claim_id = 0

    for p in pages:
        text = normalize_space(p["text"])

        # Protect citation tokens from sentence splitting.
        saved_citations = []

        def protect(match):
            idx = len(saved_citations)
            saved_citations.append(match.group(0))
            return f"__CITATION_{idx}__"

        protected = CITATION_RE.sub(protect, text)

        # Split into sentences while citations remain embedded in the
        # sentence that precedes them.
        sentences = re.split(r"(?<=[.!?])\s+(?!(?:__CITATION_\d+__))", protected)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 35:
                continue

            # Restore citations.
            def restore(match):
                idx = int(match.group(1))
                return saved_citations[idx]

            sentence = re.sub(r"__CITATION_(\d+)__", restore, sentence)

            cited_ids = []
            for m in CITATION_RE.finditer(sentence):
                cited_ids.extend(expand_citation_group(m.group(1)))

            # Remove citation tokens only for claim-language analysis.
            clean = CITATION_RE.sub("", sentence).strip()

            if not CLAIM_TERMS.search(clean):
                continue

            claim_id += 1

            claims.append(
                {
                    "claim_id": claim_id,
                    "page": p["page"],
                    "claim": clean,
                    "citation_ids": sorted(set(cited_ids)),
                }
            )

    return claims


def build_reference_texts(refs):
    if not refs:
        return [], []
    ids = sorted(refs)
    texts = [refs[i] for i in ids]
    return ids, texts


def semantic_match(claim, ref_texts, threshold_supported=0.18, threshold_weak=0.08):
    """
    TF-IDF similarity is used only as a screening signal.
    """
    if not ref_texts:
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform([claim] + ref_texts)
    except ValueError:
        return []

    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    ranked = np.argsort(scores)[::-1]

    return [(int(i), float(scores[i])) for i in ranked]


def high_risk_matches(text):
    hits = []
    for pattern in HIGH_RISK_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.I):
            hits.append(m.group(0))
    return sorted(set(hits), key=str.lower)


def analyze(path):
    pages = extract_pdf(path)
    refs = extract_references(pages)
    citations = extract_citations(pages)
    claims = extract_claims(pages)

    ref_ids, ref_texts = build_reference_texts(refs)
    ref_index = {rid: idx for idx, rid in enumerate(ref_ids)}

    findings = []

    # --------------------------------------------------------
    # Citation-level checks
    # --------------------------------------------------------
    for c in citations:
        rid = c["ref_id"]

        if rid not in refs:
            findings.append(
                {
                    "severity": "HIGH",
                    "type": "Unresolved reference",
                    "page": c["page"],
                    "claim_id": "",
                    "citation": f"[{rid}]",
                    "finding": f"Citation [{rid}] has no matching reference entry.",
                    "recommendation": "Verify the citation and add/correct the reference before publication.",
                }
            )

    # --------------------------------------------------------
    # Claim-level checks
    # --------------------------------------------------------
    supported_count = 0
    weak_count = 0
    unsupported_count = 0
    high_risk_count = 0

    for c in claims:
        claim = c["claim"]
        cited_ids = c["citation_ids"]
        risks = high_risk_matches(claim)

        if risks:
            high_risk_count += 1
            findings.append(
                {
                    "severity": "HIGH",
                    "type": "High-risk language",
                    "page": c["page"],
                    "claim_id": c["claim_id"],
                    "citation": ", ".join(f"[{x}]" for x in cited_ids) if cited_ids else "",
                    "finding": "Potentially over-strong assurance/security wording: "
                               + ", ".join(risks),
                    "recommendation": "Require human review and ensure the conclusion is proportionate to tested evidence.",
                }
            )

        if not cited_ids:
            unsupported_count += 1
            findings.append(
                {
                    "severity": "MEDIUM",
                    "type": "Missing citation",
                    "page": c["page"],
                    "claim_id": c["claim_id"],
                    "citation": "",
                    "finding": "Evidence-bearing claim detected without an explicit numeric citation.",
                    "recommendation": "Add a traceable source/evidence reference or document the basis for the claim.",
                }
            )
            continue

        resolved_ids = [rid for rid in cited_ids if rid in refs]

        if not resolved_ids:
            unsupported_count += 1
            # Unresolved-reference findings already exist above.
            continue

        # Evaluate each resolved citation against the claim.
        best_score = 0.0
        best_ref = None

        for rid in resolved_ids:
            idx = ref_index[rid]
            ranked = semantic_match(claim, [ref_texts[idx]])
            score = ranked[0][1] if ranked else 0.0

            if score > best_score:
                best_score = score
                best_ref = rid

        if best_score >= 0.18:
            supported_count += 1
            status = "Supported (screening signal)"
        elif best_score >= 0.08:
            weak_count += 1
            status = "Weak semantic support"
            findings.append(
                {
                    "severity": "MEDIUM",
                    "type": "Weak evidence-to-claim relationship",
                    "page": c["page"],
                    "claim_id": c["claim_id"],
                    "citation": ", ".join(f"[{x}]" for x in cited_ids),
                    "finding": f"Best citation [{best_ref}] has low semantic similarity "
                               f"({best_score:.3f}) to the claim.",
                    "recommendation": "Human reviewer should verify whether the source actually supports the exact assertion.",
                }
            )
        else:
            unsupported_count += 1
            findings.append(
                {
                    "severity": "HIGH",
                    "type": "Potential citation mismatch",
                    "page": c["page"],
                    "claim_id": c["claim_id"],
                    "citation": ", ".join(f"[{x}]" for x in cited_ids),
                    "finding": f"No strong semantic support detected; best match was "
                               f"[{best_ref}] with score {best_score:.3f}.",
                    "recommendation": "Verify source-to-claim alignment before publication.",
                }
            )

    # --------------------------------------------------------
    # Citation integrity metrics
    # --------------------------------------------------------
    total_citations = len(citations)
    unresolved = sum(1 for c in citations if c["ref_id"] not in refs)
    resolved = total_citations - unresolved

    if total_citations:
        resolution_rate = round(100 * resolved / total_citations, 1)
    else:
        resolution_rate = 0.0

    total_claims = len(claims)
    if total_claims:
        support_rate = round(100 * supported_count / total_claims, 1)
    else:
        support_rate = 0.0

    # Publication gate:
    # BLOCK if high-severity findings exist; REVIEW if only medium findings.
    high_findings = sum(1 for f in findings if f["severity"] == "HIGH")
    medium_findings = sum(1 for f in findings if f["severity"] == "MEDIUM")

    if high_findings > 0:
        gate = "BLOCK"
    elif medium_findings > 0:
        gate = "REVIEW"
    else:
        gate = "PASS"

    # A transparent score, not a legal/compliance conclusion.
    score = 100
    score -= min(45, unresolved * 4)
    score -= min(30, high_findings * 5)
    score -= min(20, medium_findings * 1.5)
    score = max(0, round(score))

    # --------------------------------------------------------
    # Findings table
    # --------------------------------------------------------
    df = pd.DataFrame(
        findings,
        columns=[
            "severity",
            "type",
            "page",
            "claim_id",
            "citation",
            "finding",
            "recommendation",
        ],
    )

    # --------------------------------------------------------
    # Reference table
    # --------------------------------------------------------
    ref_rows = [
        {"reference_id": rid, "reference": refs[rid]}
        for rid in sorted(refs)
    ]
    ref_df = pd.DataFrame(ref_rows)

    # --------------------------------------------------------
    # Claim table
    # --------------------------------------------------------
    claim_rows = []
    for c in claims:
        claim_rows.append(
            {
                "claim_id": c["claim_id"],
                "page": c["page"],
                "claim": c["claim"],
                "citations": ", ".join(f"[{x}]" for x in c["citation_ids"])
                if c["citation_ids"] else "",
            }
        )
    claim_df = pd.DataFrame(claim_rows)

    summary_html = f"""
    <div style="font-family:Arial,sans-serif">
      <h1>EvidenceTrace — Publication Gate</h1>
      <h2 style="font-size:28px">{"🔴" if gate=="BLOCK" else "🟠" if gate=="REVIEW" else "🟢"} {gate}</h2>

      <p><b>{high_findings} high-severity evidence issue(s) detected.</b></p>

      <table style="border-collapse:collapse;width:100%;max-width:700px">
        <tr><th style="text-align:left;border:1px solid #bbb;padding:8px">Metric</th>
            <th style="text-align:left;border:1px solid #bbb;padding:8px">Result</th></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Pages</td>
            <td style="border:1px solid #bbb;padding:8px">{len(pages)}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Claims detected</td>
            <td style="border:1px solid #bbb;padding:8px">{total_claims}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">References extracted</td>
            <td style="border:1px solid #bbb;padding:8px">{len(refs)}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Citations analyzed</td>
            <td style="border:1px solid #bbb;padding:8px">{total_citations}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Resolved citations</td>
            <td style="border:1px solid #bbb;padding:8px">{resolved}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Unresolved citations</td>
            <td style="border:1px solid #bbb;padding:8px">{unresolved}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Supported claims (screening signal)</td>
            <td style="border:1px solid #bbb;padding:8px">{supported_count}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Weak-support claims</td>
            <td style="border:1px solid #bbb;padding:8px">{weak_count}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Potentially unsupported/mismatched claims</td>
            <td style="border:1px solid #bbb;padding:8px">{unsupported_count}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">High-risk findings</td>
            <td style="border:1px solid #bbb;padding:8px">{high_findings}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Review findings</td>
            <td style="border:1px solid #bbb;padding:8px">{medium_findings}</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Citation resolution rate</td>
            <td style="border:1px solid #bbb;padding:8px">{resolution_rate}%</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px">Claim support rate</td>
            <td style="border:1px solid #bbb;padding:8px">{support_rate}%</td></tr>
        <tr><td style="border:1px solid #bbb;padding:8px"><b>Citation integrity screening score</b></td>
            <td style="border:1px solid #bbb;padding:8px"><b>{score}/100</b></td></tr>
      </table>

      <h3>GRC interpretation</h3>
      <p>EvidenceTrace flags potential:</p>
      <ul>
        <li>Evidence traceability failures</li>
        <li>Unsupported cybersecurity claims</li>
        <li>Citation/reference mismatches</li>
        <li>Weak evidence-to-claim relationships</li>
        <li>Over-strong compliance/security assertions</li>
      </ul>
      <p><b>Human verification remains mandatory.</b></p>
      <p style="color:#666">
      Note: TF-IDF similarity is a screening mechanism. A high similarity score does not
      establish that a source is factually correct or sufficient for an assurance conclusion.
      </p>
    </div>
    """

    return summary_html, df, ref_df, claim_df


def create_result_report(
    pdf_name,
    gate,
    pages_count,
    total_claims,
    references_count,
    total_citations,
    resolved,
    unresolved,
    supported_count,
    weak_count,
    unsupported_count,
    high_findings,
    medium_findings,
    resolution_rate,
    support_rate,
    score,
    findings_df,
):
    """
    Create a professional one-page PDF executive result report.
    The report is generated automatically after document analysis.
    """

    import tempfile
    import os

    fd, report_path = tempfile.mkstemp(
        prefix="EvidenceTrace_Result_",
        suffix=".pdf"
    )
    os.close(fd)

    # Professional cybersecurity/GRC palette.
    NAVY = colors.HexColor("#102A43")
    BLUE = colors.HexColor("#1F5A8A")
    TEAL = colors.HexColor("#0F766E")
    LIGHT_BLUE = colors.HexColor("#EAF2F8")
    LIGHT_TEAL = colors.HexColor("#E8F5F3")
    LIGHT_RED = colors.HexColor("#FDECEC")
    RED = colors.HexColor("#B42318")
    AMBER = colors.HexColor("#B54708")
    GREEN = colors.HexColor("#087443")
    GREY = colors.HexColor("#52606D")
    LIGHT_GREY = colors.HexColor("#F5F7FA")
    BORDER = colors.HexColor("#D9E2EC")

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=13 * mm,
        leftMargin=13 * mm,
        topMargin=11 * mm,
        bottomMargin=10 * mm,
        title="EvidenceTrace — Analysis Result",
        author="EvidenceTrace",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#D9EAF7"),
        alignment=TA_CENTER,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=11,
        textColor=NAVY,
        spaceBefore=4,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=GREY,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=GREY,
    )

    finding_style = ParagraphStyle(
        "Finding",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.7,
        leading=8.1,
        textColor=NAVY,
    )

    story = []

    # ---------------- Header ----------------
    gate_upper = str(gate).upper()
    if gate_upper == "BLOCK":
        gate_color = RED
        gate_bg = LIGHT_RED
        gate_text = "PUBLICATION BLOCK"
    elif gate_upper == "REVIEW":
        gate_color = AMBER
        gate_bg = colors.HexColor("#FFF4E5")
        gate_text = "HUMAN REVIEW REQUIRED"
    else:
        gate_color = GREEN
        gate_bg = LIGHT_TEAL
        gate_text = "PUBLICATION PASS"

    header = Table(
        [
            [
                Paragraph("EvidenceTrace", title_style),
                Paragraph("AI-Assisted Cybersecurity Assurance", subtitle_style),
            ]
        ],
        colWidths=[70 * mm, 100 * mm],
        rowHeights=[17 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(header)
    story.append(Spacer(1, 3 * mm))

    filename = os.path.basename(pdf_name or "Uploaded document")
    story.append(
        Paragraph(
            f"<b>Analysis report:</b> {html.escape(filename)}",
            small_style,
        )
    )
    story.append(Spacer(1, 2 * mm))

    # ---------------- Gate banner ----------------
    gate_banner = Table(
        [[
            Paragraph(
                f"<font size='11'><b>{html.escape(gate_text)}</b></font>"
                f"<br/><font size='7.5'>EvidenceTrace publication decision: "
                f"<b>{html.escape(gate_upper)}</b></font>",
                ParagraphStyle(
                    "GateText",
                    parent=body_style,
                    textColor=gate_color,
                    leading=12,
                ),
            ),
            Paragraph(
                f"<font size='20'><b>{score}</b></font>"
                f"<br/><font size='6.5'>Integrity<br/>screening /100</font>",
                ParagraphStyle(
                    "Score",
                    parent=body_style,
                    textColor=gate_color,
                    alignment=TA_CENTER,
                    leading=9,
                ),
            ),
        ]],
        colWidths=[145 * mm, 25 * mm],
    )
    gate_banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), gate_bg),
                ("BOX", (0, 0), (-1, -1), 0.8, gate_color),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(gate_banner)
    story.append(Spacer(1, 2.5 * mm))

    # ---------------- KPI grid ----------------
    kpis = [
        ("Pages", pages_count),
        ("Claims", total_claims),
        ("References", references_count),
        ("Citations", total_citations),
        ("Resolved", resolved),
        ("Unresolved", unresolved),
        ("High-risk", high_findings),
        ("Review", medium_findings),
    ]

    kpi_cells = []
    for label, value in kpis:
        kpi_cells.append(
            Paragraph(
                f"<font size='13'><b>{value}</b></font><br/>"
                f"<font size='6.5'>{label}</font>",
                ParagraphStyle(
                    "KPI",
                    parent=body_style,
                    alignment=TA_CENTER,
                    textColor=NAVY,
                    leading=9,
                ),
            )
        )

    kpi_table = Table(
        [kpi_cells[:4], kpi_cells[4:]],
        colWidths=[42.5 * mm] * 4,
        rowHeights=[13 * mm, 13 * mm],
    )
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 2.5 * mm))

    # ---------------- Quality metrics ----------------
    metrics = [
        ["Quality indicator", "Result", "Interpretation"],
        ["Citation resolution", f"{resolution_rate}%", "References successfully resolved"],
        ["Claim support", f"{support_rate}%", "Claims with strong screening support"],
        ["High-severity issues", str(high_findings), "Require correction / human verification"],
    ]

    metric_table = Table(
        [
            [Paragraph(f"<b>{x}</b>", small_style) for x in metrics[0]]
        ] +
        [
            [
                Paragraph(str(row[0]), small_style),
                Paragraph(str(row[1]), ParagraphStyle(
                    "MetricValue", parent=small_style,
                    fontName="Helvetica-Bold", textColor=NAVY
                )),
                Paragraph(str(row[2]), small_style),
            ]
            for row in metrics[1:]
        ],
        colWidths=[55 * mm, 30 * mm, 85 * mm],
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 2.5 * mm))

    # ---------------- Key findings ----------------
    story.append(Paragraph("Key findings requiring attention", section_style))

    key_findings = []
    if findings_df is not None and not findings_df.empty:
        # Prioritize HIGH, then MEDIUM; limit to keep the report one page.
        priority = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        temp = findings_df.copy()
        temp["_priority"] = temp["severity"].map(priority).fillna(3)
        temp = temp.sort_values(["_priority", "page"]).head(6)

        for _, row in temp.iterrows():
            sev = str(row.get("severity", ""))
            typ = str(row.get("type", ""))
            page = str(row.get("page", ""))
            citation = str(row.get("citation", "") or "—")
            finding = str(row.get("finding", ""))

            sev_color = RED if sev == "HIGH" else AMBER
            key_findings.append(
                [
                    Paragraph(
                        f"<font color='{sev_color.hexval()}'><b>{html.escape(sev)}</b></font>",
                        finding_style,
                    ),
                    Paragraph(
                        f"<b>{html.escape(typ)}</b><br/>"
                        f"Page {html.escape(page)} · Citation {html.escape(citation)}<br/>"
                        f"{html.escape(finding)}",
                        finding_style,
                    ),
                ]
            )
    else:
        key_findings.append(
            [Paragraph("—", finding_style), Paragraph("No findings detected.", finding_style)]
        )

    findings_table = Table(
        key_findings,
        colWidths=[20 * mm, 150 * mm],
    )
    findings_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(findings_table)
    story.append(Spacer(1, 2.5 * mm))

    # ---------------- GRC interpretation ----------------
    story.append(Paragraph("GRC interpretation", section_style))
    interpretation = (
        "EvidenceTrace identified potential evidence-traceability failures, "
        "unsupported or weakly supported cybersecurity claims, citation/reference "
        "mismatches, and over-strong assurance language. "
        "<b>Human verification remains mandatory before publication.</b>"
    )
    story.append(
        Table(
            [[Paragraph(interpretation, body_style)]],
            colWidths=[170 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.5, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
    )
    story.append(Spacer(1, 2.5 * mm))

    # ---------------- Methodology note ----------------
    story.append(
        Paragraph(
            "<b>Methodology note:</b> Citation resolution and TF-IDF semantic similarity "
            "are screening signals, not proof of factual correctness. The report is intended "
            "for quality-control and GRC review, not as an automated audit opinion.",
            small_style,
        )
    )

    # Footer on every page
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.line(13 * mm, 7 * mm, A4[0] - 13 * mm, 7 * mm)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(GREY)
        canvas.drawString(
            13 * mm, 4 * mm,
            "EvidenceTrace | Cybersecurity Assurance Documentation Quality Control"
        )
        canvas.drawRightString(
            A4[0] - 13 * mm, 4 * mm,
            "Confidential • Human review required"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    return report_path



def run_analysis(file_obj, request: gr.Request = None):
    if file_obj is None:
        return (
            "<h2>Please upload a PDF.</h2>",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
        )

    session_id = ""
    try:
        session_id = getattr(request, "session_hash", "") or ""
    except Exception:
        session_id = ""

    # Use a fresh anonymous session identifier when the hosting
    # environment does not expose a Gradio session hash.
    if not session_id:
        session_id = f"anon-{uuid.uuid4().hex[:12]}"

    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        file_name = Path(path).name
        file_size = Path(path).stat().st_size if Path(path).exists() else None

        summary_html, findings_df, ref_df, claim_df = analyze(path)

        pages = extract_pdf(path)
        refs = extract_references(pages)
        citations = extract_citations(pages)
        claims_data = extract_claims(pages)

        ref_ids, ref_texts = build_reference_texts(refs)
        unresolved = sum(1 for c in citations if c["ref_id"] not in refs)
        resolved = len(citations) - unresolved

        supported_count = 0
        weak_count = 0
        unsupported_count = 0

        for c in claims_data:
            cited_ids = c["citation_ids"]

            if not cited_ids:
                unsupported_count += 1
                continue

            resolved_ids = [rid for rid in cited_ids if rid in refs]
            if not resolved_ids:
                unsupported_count += 1
                continue

            best_score = 0.0

            for rid in resolved_ids:
                idx = ref_ids.index(rid)
                ranked = semantic_match(
                    c["claim"],
                    [ref_texts[idx]],
                )

                if ranked:
                    best_score = max(
                        best_score,
                        ranked[0][1],
                    )

            if best_score >= 0.18:
                supported_count += 1
            elif best_score >= 0.08:
                weak_count += 1
            else:
                unsupported_count += 1

        high_findings = (
            int((findings_df["severity"] == "HIGH").sum())
            if not findings_df.empty
            and "severity" in findings_df.columns
            else 0
        )

        medium_findings = (
            int((findings_df["severity"] == "MEDIUM").sum())
            if not findings_df.empty
            and "severity" in findings_df.columns
            else 0
        )

        total_claims = len(claims_data)
        total_citations = len(citations)

        resolution_rate = (
            round(100 * resolved / total_citations, 1)
            if total_citations
            else 0.0
        )

        support_rate = (
            round(100 * supported_count / total_claims, 1)
            if total_claims
            else 0.0
        )

        score = 100
        score -= min(45, unresolved * 4)
        score -= min(30, high_findings * 5)
        score -= min(20, medium_findings * 1.5)
        score = max(0, round(score))

        gate = (
            "BLOCK"
            if high_findings > 0
            else "REVIEW"
            if medium_findings > 0
            else "PASS"
        )

        report_path = create_result_report(
            pdf_name=path,
            gate=gate,
            pages_count=len(pages),
            total_claims=total_claims,
            references_count=len(refs),
            total_citations=total_citations,
            resolved=resolved,
            unresolved=unresolved,
            supported_count=supported_count,
            weak_count=weak_count,
            unsupported_count=unsupported_count,
            high_findings=high_findings,
            medium_findings=medium_findings,
            resolution_rate=resolution_rate,
            support_rate=support_rate,
            score=score,
            findings_df=findings_df,
        )

        # -----------------------------------------------
        # Application-level audit trail
        # -----------------------------------------------
        audit_write(
            event_type="ANALYSIS",
            session_id=session_id,
            file_name=file_name,
            file_size_bytes=file_size,
            pages=len(pages),
            claims=total_claims,
            citations=total_citations,
            references_count=len(refs),
            resolved_citations=resolved,
            unresolved_citations=unresolved,
            supported_claims=supported_count,
            weak_claims=weak_count,
            unsupported_claims=unsupported_count,
            high_findings=high_findings,
            medium_findings=medium_findings,
            publication_gate=gate,
            integrity_score=score,
            report_generated=True,
        )

        audit_write(
            event_type="REPORT_GENERATED",
            session_id=session_id,
            file_name=file_name,
            pages=len(pages),
            publication_gate=gate,
            integrity_score=score,
            report_generated=True,
        )

        return (
            summary_html,
            findings_df,
            ref_df,
            claim_df,
            report_path,
        )

    except Exception as exc:
        try:
            audit_write(
                event_type="ANALYSIS_ERROR",
                session_id=session_id,
                file_name=Path(
                    file_obj if isinstance(file_obj, str)
                    else getattr(file_obj, "name", "unknown.pdf")
                ).name,
                error_message=str(exc),
            )
        except Exception:
            pass

        return (
            f"<h2>Analysis error</h2><pre>{html.escape(str(exc))}</pre>",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
        )


def show_audit_log(pin, limit=500):
    if str(pin or "").strip() != ADMIN_PIN:
        return (
            "🔒 Enter the correct administrator PIN to view the audit trail.",
            pd.DataFrame(),
        )

    df = load_audit_dataframe(limit=int(limit))

    if df.empty:
        return (
            "✅ Administrator access granted. No audit events recorded yet.",
            df,
        )

    analyses = df[df["event_type"] == "ANALYSIS"]
    blocked = int((analyses["publication_gate"] == "BLOCK").sum())
    review = int((analyses["publication_gate"] == "REVIEW").sum())
    passed = int((analyses["publication_gate"] == "PASS").sum())

    overview = f"""
### Audit Trail

**Access:** ✅ Administrator

| Activity | Count |
|---|---:|
| Total audit events | {len(df)} |
| Document analyses | {len(analyses)} |
| Publication BLOCK | {blocked} |
| Publication REVIEW | {review} |
| Publication PASS | {passed} |

The detailed event log is shown below. Uploaded document contents are **not stored** by EvidenceTrace.
"""

    return overview, df


# ============================================================
# Professional Gradio UI
# ============================================================

# ============================================================

CUSTOM_CSS = """
/* EvidenceTrace enterprise presentation layer */
footer {
    display: none !important;
}

.gradio-footer,
footer,
[data-testid="gradio-footer"] {
    display: none !important;
}

:root {
    --navy: #0b1f33;
    --navy-2: #12324a;
    --teal: #0f766e;
    --teal-2: #149f93;
    --blue: #2563eb;
    --ink: #172b3a;
    --muted: #607483;
    --line: #dbe5ea;
    --surface: #ffffff;
    --surface-2: #f5f8fa;
    --success: #087443;
    --success-bg: #e9f7ef;
    --warning: #b45309;
    --warning-bg: #fff4df;
    --danger: #b42318;
    --danger-bg: #fdeceb;
}

.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 0 26px 28px !important;
    background: #f4f7f9 !important;
}

body {
    background: #f4f7f9 !important;
}

#et-header {
    background: linear-gradient(120deg, var(--navy), var(--navy-2));
    border-radius: 18px;
    padding: 24px 28px;
    margin: 8px 0 18px;
    box-shadow: 0 10px 30px rgba(11,31,51,.12);
}

#et-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

#et-mark {
    width: 48px;
    height: 48px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #8ce7d4, #3db9a8);
    color: #073b38;
    font-weight: 900;
    font-size: 22px;
}

#et-title {
    color: #fff;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -.04em;
    line-height: 1.05;
}

#et-subtitle {
    color: #b8ccd8;
    font-size: 12px;
    margin-top: 4px;
}

#et-tag {
    margin-left: auto;
    border: 1px solid rgba(255,255,255,.2);
    color: #d8edf3;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 700;
}

#et-header-line {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 15px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,.12);
}

#et-header-line span {
    display: inline-block;
    color: #b9cdd8;
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 999px;
    padding: 5px 9px;
    font-size: 9px;
    letter-spacing: .08em;
    font-weight: 800;
}

#upload-panel .wrap,
#download-panel .wrap {
    color: var(--ink);
}

.gradio-container input[type="file"] {
    border-color: var(--line) !important;
}

.gradio-container button {
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}

.gradio-container button:hover {
    transform: translateY(-1px);
}


.section-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 16px 18px;
    box-shadow: 0 6px 18px rgba(16,42,67,.05);
}

.section-title {
    color: var(--ink);
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
}

#upload-panel {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 6px 18px rgba(16,42,67,.05);
}

#upload-panel label {
    font-weight: 700 !important;
    color: var(--ink) !important;
}

#analyze-btn {
    background: linear-gradient(135deg, var(--teal), var(--teal-2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 48px !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 20px rgba(15,118,110,.22) !important;
}

#analyze-btn:hover {
    filter: brightness(1.05);
}

#download-panel {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 12px;
    box-shadow: 0 6px 18px rgba(16,42,67,.05);
}

#summary-panel {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 3px 4px;
    min-height: 345px;
    box-shadow: 0 6px 18px rgba(16,42,67,.05);
}

#summary-panel h1 {
    color: var(--ink) !important;
    font-size: 24px !important;
    margin-bottom: 6px !important;
}

#summary-panel h2 {
    color: var(--ink) !important;
    margin-top: 6px !important;
}

#summary-panel table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 10px;
}

#summary-panel th {
    background: #eaf2f8 !important;
    color: var(--navy) !important;
    font-weight: 800 !important;
}

#summary-panel td,
#summary-panel th {
    padding: 9px 10px !important;
    border-bottom: 1px solid var(--line);
    font-size: 12px !important;
}

#summary-panel tr:last-child td {
    border-bottom: none !important;
}

#findings-panel,
#refs-panel,
#claims-panel {
    background: #fff !important;
    border: 1px solid var(--line) !important;
    border-radius: 15px !important;
    box-shadow: 0 6px 18px rgba(16,42,67,.05) !important;
}

.gradio-tabs {
    border: none !important;
}

.gradio-tabs > .tab-nav {
    border-bottom: 1px solid var(--line) !important;
    gap: 4px !important;
}

.gradio-tabs > .tab-nav button {
    color: var(--muted) !important;
    font-weight: 700 !important;
    border-radius: 8px 8px 0 0 !important;
}

.gradio-tabs > .tab-nav button.selected {
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}


#et-audit-badge {
    background: #eef5f7;
    border: 1px solid #d5e4e8;
    color: #274c5b;
    border-radius: 10px;
    padding: 9px 11px;
    font-size: 10px;
    line-height: 1.5;
}

#research-note {
    background: linear-gradient(135deg, #eef8f6, #f5fafb);
    border: 1px solid #cce8e2;
    border-radius: 14px;
    padding: 14px 16px;
    color: var(--ink);
    font-size: 11px;
    line-height: 1.6;
}

#footer-note {
    text-align: center;
    color: #738592;
    font-size: 10px;
    padding: 14px 0 4px;
}

button, input, textarea, select {
    font-family: Inter, Arial, sans-serif !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 0 12px 18px !important;
    }
    #et-tag {
        display: none;
    }
}
"""

HEADER_HTML = """
<div id="et-header">
  <div id="et-brand">
    <div id="et-mark">ET</div>
    <div>
      <div id="et-title">EvidenceTrace</div>
      <div id="et-subtitle">AI-assisted cybersecurity assurance &amp; evidence quality control</div>
    </div>
    <div id="et-tag">GRC&nbsp;&nbsp;•&nbsp;&nbsp;ASSURANCE&nbsp;&nbsp;•&nbsp;&nbsp;CYBER RISK</div>
  </div>
  <div id="et-header-line">
    <span>PRE-PUBLICATION QUALITY CONTROL</span>
    <span>HUMAN-IN-THE-LOOP</span>
    <span>EVIDENCE-TRACEABILITY</span>
  </div>
</div>
"""

RESEARCH_NOTE = """
<div id="research-note">
<div style="font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#0f766e;font-weight:800;margin-bottom:5px;">
Evidence integrity workflow
</div>
<b>Claim → Citation → Reference → Semantic screening → GRC risk → Publication gate</b><br>
<span style="color:#607483;">
EvidenceTrace is a research prototype for surfacing potentially unsupported or weakly supported cybersecurity assurance claims.
Similarity scores are screening signals, not proof of factual correctness. Final publication decisions require qualified human review.
</span>
</div>
"""

with gr.Blocks(
    title="EvidenceTrace | Cybersecurity Assurance",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="md",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=5, elem_id="upload-panel"):
            gr.Markdown(
                """
<div class="section-title">Analyze an assurance document</div>
<div class="section-subtitle">
Upload a cybersecurity, audit, compliance, policy or assurance PDF.
The prototype extracts claims and references locally, then performs evidence-quality screening.
</div>
                """
            )
            pdf = gr.File(
                label="PDF document",
                file_types=[".pdf"],
                type="filepath",
            )
            analyze_btn = gr.Button(
                "Analyze Document",
                elem_id="analyze-btn",
                variant="primary",
            )

        with gr.Column(scale=2, elem_id="download-panel"):
            gr.Markdown(
                """
<div class="section-title">Publication output</div>
<div class="section-subtitle">
A one-page executive result report is generated after analysis.
</div>
                """
            )
            report_file = gr.File(
                label="Result report",
                interactive=False,
            )
            gr.Markdown(
                """
<div class="section-subtitle">
<b>Report includes:</b><br>
• Publication Gate<br>
• Integrity score<br>
• Citation metrics<br>
• Key findings<br>
• GRC interpretation<br>
• Methodology note
</div>
                """
            )

    gr.HTML(RESEARCH_NOTE)

    gr.Markdown("## Analysis result")

    summary = gr.HTML(
        "<div class='section-card'><div class='section-title'>Ready for analysis</div>"
        "<div class='section-subtitle'>Upload a PDF and select <b>Analyze Document</b>.</div></div>",
        elem_id="summary-panel",
    )

    with gr.Tabs():
        with gr.Tab("EvidenceTrace Findings"):
            findings = gr.Dataframe(
                label="Findings requiring attention",
                interactive=False,
                wrap=True,
                elem_id="findings-panel",
            )

        with gr.Tab("Extracted References"):
            refs = gr.Dataframe(
                label="Reference Register",
                interactive=False,
                wrap=True,
                elem_id="refs-panel",
            )

        with gr.Tab("Detected Claims"):
            claims = gr.Dataframe(
                label="Detected Evidence-Bearing Claims",
                interactive=False,
                wrap=True,
                elem_id="claims-panel",
            )


    with gr.Tabs():
        with gr.Tab("Audit Trail"):
            gr.Markdown(
                """
### Administrator audit trail

Use the administrator PIN to view application-level activity logs.

**Logged metadata:** timestamp, anonymous session ID, file name, analysis metrics,
publication gate, integrity score, report-generation event, and errors.

**Not stored:** uploaded PDF contents.
"""
            )

            with gr.Row():
                admin_pin = gr.Textbox(
                    label="Administrator PIN",
                    type="password",
                    placeholder="Enter admin PIN",
                )
                audit_limit = gr.Number(
                    label="Maximum events",
                    value=500,
                    precision=0,
                )

            with gr.Row():
                view_logs_btn = gr.Button(
                    "View Audit Log",
                    variant="primary",
                )
                clear_logs_btn = gr.Button(
                    "Clear Audit Log",
                    variant="stop",
                )

            audit_status = gr.Markdown(
                "🔒 Audit trail is administrator-restricted."
            )

            audit_table = gr.Dataframe(
                label="Application Audit Events",
                interactive=False,
                wrap=True,
            )

            audit_download = gr.File(
                label="Audit CSV export",
                interactive=False,
            )

            def export_audit(pin, limit=500):
                if str(pin or "").strip() != ADMIN_PIN:
                    return "❌ Incorrect administrator PIN.", None

                df = load_audit_dataframe(limit=int(limit))
                export_path = Path(
                    f"EvidenceTrace_Audit_Log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
                )
                df.to_csv(export_path, index=False)
                return f"✅ Export ready: {len(df)} event(s).", str(export_path)

            def view_audit(pin, limit=500):
                status, df = show_audit_log(pin, limit)
                return status, df

            def clear_audit(pin):
                status, df = clear_audit_log(pin)
                return status, df

            view_logs_btn.click(
                fn=view_audit,
                inputs=[admin_pin, audit_limit],
                outputs=[audit_status, audit_table],
            )

            clear_logs_btn.click(
                fn=clear_audit,
                inputs=admin_pin,
                outputs=[audit_status, audit_table],
            )

            gr.Button("Export Audit CSV").click(
                fn=export_audit,
                inputs=[admin_pin, audit_limit],
                outputs=[audit_status, audit_download],
            )

    gr.HTML(
        """
<div id="footer-note">
  <div style="font-weight:800;color:#102a43;letter-spacing:.02em;">EvidenceTrace</div>
  <div style="margin-top:3px;">
    Cybersecurity Assurance • Evidence Integrity • GRC Quality Control • Audit Trail
  </div>
  <div style="margin-top:5px;font-size:9px;">
    Research Prototype &nbsp;|&nbsp; Activity metadata only &nbsp;|&nbsp; Human professional judgment remains mandatory
  </div>
</div>
        """
    )

    analyze_btn.click(
        fn=run_analysis,
        inputs=pdf,
        outputs=[summary, findings, refs, claims, report_file],
    )


if __name__ == "__main__":
    demo.launch(share=True)
