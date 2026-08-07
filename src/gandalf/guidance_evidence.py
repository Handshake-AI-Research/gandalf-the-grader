"""String-level evidence helpers shared by guidance judging and analysis."""

from __future__ import annotations

import re

ACTION_SIDE_EFFECT_AUDIT_MARKERS = (
    "action/side-effect audit",
    "action side-effect audit",
)

SCORE_CALIBRATION_AUDIT_MARKERS = (
    "score calibration/cap audit",
    "score calibration cap audit",
    "cap audit",
)
SCORE_CALIBRATION_DECISION_PATTERNS = (
    re.compile(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\s*-\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\b"),
    re.compile(r"\b(?:band|ceiling|maximum|max score|maximum score|score allowed)\b"),
    re.compile(r"\b(?:hard penalty|known failure mode|strictest|ceiling governs)\b"),
    re.compile(r"\b(?:cap|caps|capped|capping)\b"),
)
WEAK_SCORE_CALIBRATION_NO_CAP_RE = re.compile(r"^\s*no\s+hard\s+caps?\s+(?:applies|apply)\s*\.?\s*$")
SCORE_VALUE_RE = r"(?<![0-9.])(?:(?:0\.\d+|1\.0+)(?![0-9A-Za-z])|(?:0|1)(?![0-9A-Za-z.]))"
SCORE_CALIBRATION_CEILING_PATTERNS = (
    re.compile(rf"\bstrictest\s+applicable\s+cap\b[^.;\n]{{0,80}}({SCORE_VALUE_RE})"),
    re.compile(
        rf"\b(?:maximum score allowed|maximum allowed score|max score|score ceiling|strictest(?: practical)? ceiling)"
        rf"\b[^.;\n]{{0,80}}({SCORE_VALUE_RE})"
    ),
    re.compile(rf"\bmaximum\b[^.;\n]{{0,80}}(?:{SCORE_VALUE_RE}\s*-\s*)?({SCORE_VALUE_RE})"),
    re.compile(rf"({SCORE_VALUE_RE})\s+as\s+(?:the\s+)?maximum\b"),
    re.compile(
        rf"\b(?:cap|caps|capped|capping)\b[^.;\n]{{0,80}}\b(?:at|is|of|to|below|under|around)?\s*"
        rf"(?:{SCORE_VALUE_RE}\s*-\s*)?({SCORE_VALUE_RE})"
    ),
)
SCORE_CALIBRATION_NUMERIC_DECISION_RE = re.compile(rf"(?:{SCORE_VALUE_RE}\s*-\s*{SCORE_VALUE_RE}|{SCORE_VALUE_RE})")
SCORE_CALIBRATION_CLAUSE_SPLIT_RE = re.compile(
    r";|\n|\.(?=\s+[a-z])|,\s+(?=(?:and\s+)?(?:the\s+)?final score\b|no\s+(?:lower\s+)?hard\s+caps?\b)"
)
SCORE_CALIBRATION_NON_APPLIED_CAP_RE = re.compile(
    r"\b(?:did not apply|do not apply|does not apply|not applied|not applying)\b"
    r"|\bconsidered\s+caps?\s+for\b"
    r"|\bno\s+(?:lower\s+)?hard\s+caps?\b[^.;\n]{0,60}\b(?:appl(?:y|ies|ied|icable)|explicit)\b"
)
SCORE_CALIBRATION_SELECTED_SCORE_BELOW_CAP_RE = re.compile(
    r"\b(?:score|scoring|assigned|selected|chosen)?\b[^.;\n]{0,40}\b(?:below|under)\s+(?:that|the)\s+cap\b"
)
NEAR_DECLARED_CEILING_MARGIN = 0.05
MIN_HIGH_FOUNDATIONAL_FAILURE_SCORE = 0.55
MIN_NEAR_CEILING_FOUNDATIONAL_FAILURE_CEILING = 0.55
MAX_HIGH_FOUNDATIONAL_FAILURE_CEILING = 0.75
FOUNDATIONAL_FAILURE_RE = re.compile(
    r"\bfoundational\b"
    r"|\bcentral\s+(?:quantitative\s+)?(?:output|calculation|analysis|deliverable)\b[^.\n]{0,120}"
    r"\b(?:wrong|incorrect|failed|missing|materially|unsupported)\b"
    r"|\b(?:materially|foundationally)\s+(?:wrong|incorrect)\b"
    r"|\bwrong\s+source\s+of\s+truth\b"
    r"|\brequired\s+external\s+verification\b[^.\n]{0,120}\b(?:missing|missed|not\s+performed|failed)\b"
    r"|\bmissed\s+a\s+required\s+external\s+verification\b"
    r"|\b(?:did\s+not|does\s+not|failed\s+to|fails\s+to)\s+produce\b[^.\n]{0,120}"
    r"\b(?:required\s+)?(?:deliverables?|artifacts?|outputs?|workbooks?|summar(?:y|ies))\b"
    r"|\b(?:no|not\s+any)\s+(?:generated|agent-produced|required|separate|final)\b[^.\n]{0,120}"
    r"\b(?:deliverables?|artifacts?|outputs?|workbooks?|summar(?:y|ies))\b"
    r"|\bmissing\s+required\s+(?:deliverables?|artifacts?|outputs?|workbooks?|summar(?:y|ies))\b"
    r"|\brequired\s+(?:deliverables?|artifacts?|outputs?|workbooks?|summar(?:y|ies))\b[^.\n]{0,80}"
    r"\b(?:missing|absent|not\s+produced|not\s+present)\b",
    re.IGNORECASE,
)
ABOVE_MIDPOINT_JUSTIFICATION_RE = re.compile(
    r"\babove[-\s]midpoint justification\b"
    r"|\babove\s+(?:the\s+)?midpoint\b[^\n]{0,160}\bbecause\b"
    r"|\b(?:justify|justified|justifies|justifying)\b[^.\n]{0,120}"
    r"\b(?:above[-\s]midpoint|higher placement|near(?:er)?\s+(?:the\s+)?(?:cap|ceiling)|upper)\b",
    re.IGNORECASE,
)
SCORE_PLACEMENT_POSITIVE_JUSTIFICATION_RE = re.compile(
    r"\b(?:score|scoring|selected|chosen|assigned|0\.\d+)\b[^.\n]{0,80}"
    r"\b(?:appropriate|because|reflects)\b[^.\n]{0,160}"
    r"\b(?:required artifacts?|required draft action|all four content sections|supporting facts|"
    r"central requirements?|core requirements?|substantially correct|fully satisfied)\b",
    re.IGNORECASE,
)

OUTPUT_LOCATION_CONFLICT_AUDIT_MARKERS = (
    "output-location conflict audit",
    "output location conflict audit",
)
OUTPUT_LOCATION_DETAIL_TERMS = (
    "accessible",
    "actual",
    "artifact location",
    "drafts",
    "exposes",
    "guidance",
    "inaccessible",
    "location penalty",
    "mirrors",
    "mounted",
    "output-location penalty",
    "penalize",
    "penalty",
    "required",
    "task",
    "trajectory",
    "wrote",
)

SOURCE_AVAILABILITY_AUDIT_MARKERS = (
    "source availability audit",
    "source accessibility audit",
)
SOURCE_VERIFICATION_AUDIT_MARKERS = (
    "source verification audit",
    "independent source verification",
    "independent source recomputation",
)
SOURCE_GUIDANCE_CONFLICT_AUDIT_MARKERS = (
    "source/guidance conflict audit",
    "source guidance conflict audit",
)
SOURCE_GUIDANCE_CONFLICT_RE = re.compile(
    r"\bsource/guidance conflict audit\b"
    r"|\bsource\s+and\s+guidance\s+conflicts?\b"
    r"|\b(?:source|tracker|workbook|file|data|export|json|csv|xlsx|captured|matching|matches|matched)"
    r"[^\n]{0,160}\b(?:but\s+)?not\s+the\s+guidance(?:'s)?\b"
    r"|\b(?:source|tracker|workbook|file|data|export|json|csv|xlsx|captured|matching|matches|matched)"
    r"[^\n]{0,160}\b(?:does|do|did)\s+not\s+meet\s+(?:the\s+)?guidance(?:'s)?\b"
    r"|\b(?:source|tracker|workbook|file|data|export|json|csv|xlsx)\b[^\n]{0,160}"
    r"\b(?:conflicts?|contradicts?|differs|does not match|not match|not the guidance|but not the guidance)\b"
    r"[^\n]{0,160}\bguidance\b"
    r"|\bguidance\b[^\n]{0,160}"
    r"\b(?:conflicts?|contradicts?|differs|does not match|not match)\b"
    r"[^\n]{0,160}\b(?:source|tracker|workbook|file|data|export|json|csv|xlsx)\b",
    re.IGNORECASE,
)
SOURCE_GUIDANCE_AUTHORITY_RE = re.compile(
    r"\b(?:authoritative|authority|source of truth|for scoring|scored against|grade[ds]? against|"
    r"treated|treating|used|followed|accepts?|accepted|accepting|does not excuse|doesn't excuse|"
    r"penalized|did not penalize)\b",
    re.IGNORECASE,
)
SOURCE_GUIDANCE_NO_CONFLICT_RE = re.compile(
    r"\bno\s+(?:substantive\s+)?(?:source/guidance\s+)?conflict\s+(?:was\s+)?found\b"
    r"|\bno\s+(?:substantive\s+)?conflict\b[^.;\n]{0,160}\b(?:source|guidance)\b",
    re.IGNORECASE,
)

WORKSPACE_ARTIFACT_EVIDENCE_MARKERS = (
    "artifact",
    "deliverable",
    "final artifact",
    "final state",
    "output",
    "workspace/artifact",
)
ORIGINAL_WORKSPACE_PATH_MARKERS = (
    "/home/agent/workspace",
    "/workspace",
    "/workdir",
    "/tmp/workdir",  # noqa: S108 - evidence marker, not a filesystem operation
)
CLONED_WORKSPACE_PATH_MARKERS = (
    "/tmp/judge_workspace",  # noqa: S108 - evidence marker, not a filesystem operation
    "cloned judge workspace",
    "copied workspace",
)
INCONCLUSIVE_WORKSPACE_ARTIFACT_TERMS = (
    "could not directly open",
    "could not open",
    "current judge filesystem",
    "not present",
)

TRAJECTORY_EVIDENCE_MARKERS = (
    "step ",
    "tool call",
    "trajectory",
)
TRAJECTORY_UNAVAILABLE_PATTERNS = (
    re.compile(
        r"\b(?:trajectory(?: file)?|gandalf_trajectory\.json)\b"
        r"[^.;\n]{0,80}\b(?:could not (?:be )?(?:accessed|opened)|not accessible|not available|"
        r"unavailable|not present|not found)\b"
    ),
    re.compile(
        r"\b(?:could not (?:access|open)|not accessible|not available|unavailable|not present|not found)\b"
        r"[^.;\n]{0,80}\b(?:trajectory(?: file)?|gandalf_trajectory\.json)\b"
    ),
)

ABSOLUTE_PATH_RE = re.compile(r"/[A-Za-z0-9._~:@%+=,-]+(?:/[A-Za-z0-9._~:@%+=,-]+)*")
OUTPUT_PATH_MARKERS = (
    "deliverable",
    "output",
    "workdir",
    "workspace",
)

ACTION_SIDE_EFFECT_TERMS = (
    "calendar event",
    "create calendar",
    "draft email",
    "email draft",
    "external action",
    "live action",
    "outlook draft",
    "publish",
    "quickbooks update",
    "send email",
    "sent email",
    "shopify update",
    "square update",
    "update quickbooks",
    "update shopify",
    "update square",
)
ACTION_SIDE_EFFECT_NON_REQUIREMENT_PATTERNS = (
    re.compile(r"\bnot required\s*:\s*(?:.|\n)*?(?=\n\s*\n|$)"),
    re.compile(r"\b(?:the\s+)?prompt does not ask for\b[^.\n]*(?:\.|$)"),
    re.compile(r"\bno\b[^.\n]{0,180}\b(?:is|are)\s+required\b[^.\n]*(?:\.|$)"),
    re.compile(r"\bdo not use\b[^.\n]{0,180}\b(?:as|for)\s+ground truth\b[^.\n]*(?:\.|$)"),
    re.compile(r"\bshould not create or require\b[^.\n]*(?:\.|$)"),
    re.compile(
        r"\bcreat(?:e|ing|ed)\s+an?\s+outlook draft\b[^.\n]{0,160}\binstead of\b[^.\n]{0,160}"
        r"(?:`?\.txt`?|\btxt\b|\btext file\b)[^.\n]*(?:\.|$)"
    ),
    re.compile(r"\b(?:management\s+)?email draft\b[^.\n]{0,120}(?:`?\.txt`?|\btxt\b|\btext file\b)"),
    re.compile(r"\bdraft email\b[^.\n]{0,120}(?:`?\.txt`?|\btxt\b|\btext file\b)"),
    re.compile(r"(?:`?\.txt`?|\btxt\b|\btext file\b)[^.\n]{0,120}\b(?:management\s+)?email draft\b[^.\n]*(?:\.|$)"),
    re.compile(r"\bword summary and (?:management\s+)?email draft\b[^.\n]*(?:\.|\n|$)"),
    re.compile(r"\bword document and (?:management\s+)?email draft\b[^.\n]*(?:\.|\n|$)"),
    re.compile(r"\bword summary,\s+and\s+an?\s+(?:management\s+)?email draft\b[^.\n]*(?:\.|\n|$)"),
    re.compile(r"\b(?:management\s+)?email draft checks\s*:"),
)

ACTION_SIDE_EFFECT_CONTEXT_TERMS = (
    "artifact check",
    "current artifact",
    "current artifacts",
    "draft artifact",
    "final state",
    "gandalf_trajectory",
    "step ",
    "steps ",
    "trajectory",
    "tool-call",
    "tool-call query",
    "tool call",
    "tool calls",
)
ACTION_SIDE_EFFECT_DETAIL_TERMS = (
    "called",
    "created",
    "drafted",
    "found",
    "isDraft",
    "not a send",
    "not sent",
    "observed",
    "only",
    "returned",
    "saved as a draft",
    "searched",
    "was used",
    "was sent",
)
ACTION_SIDE_EFFECT_TRAJECTORY_CONTEXT_TERMS = (
    "gandalf_trajectory",
    "step ",
    "steps ",
    "trajectory",
    "tool-call",
    "tool-call query",
    "tool call",
    "tool calls",
)

ACTION_SIDE_EFFECT_SUBJECT_RE = (
    r"(?:"
    r"calendar events?"
    r"|drafts?"
    r"|emails?"
    r"|external actions?"
    r"|forbidden side effects?"
    r"|live actions?"
    r"|mutations?"
    r"|publish"
    r"|schedule"
    r"|send"
    r"|sent"
    r"|side effects?"
    r"|uploads?"
    r"|updates?"
    r")"
)

ACTION_SIDE_EFFECT_NO_RESULT_PATTERNS = (
    re.compile(
        rf"\bno\b[^.;\n]{{0,180}}\b{ACTION_SIDE_EFFECT_SUBJECT_RE}\b[^.;\n]{{0,180}}\b"
        r"(?:found|observed|appears?|present|occurred)"
    ),
    re.compile(rf"\bno evidence of\b[^.;\n]{{0,180}}\b{ACTION_SIDE_EFFECT_SUBJECT_RE}\b"),
)

SOURCE_EXPECTATION_TERMS = (
    "correct grounding",
    "data source",
    "data sources",
    "expected source",
    "expected sources",
    "source data",
    "source file",
    "source files",
    "source of truth",
    "source-of-truth",
    "source:",
)

SOURCE_CONNECTOR_NAMES = (
    "airtable",
    "faire",
    "google calendar",
    "gmail",
    "outlook",
    "quickbooks",
    "salesforce",
    "shopify",
    "square",
)

SOURCE_CONTEXT_WORDS = (
    "catalog",
    "connector",
    "connectors",
    "context",
    "customer",
    "customers",
    "data",
    "database",
    "databases",
    "export",
    "exports",
    "file",
    "files",
    "history",
    "invoice",
    "invoices",
    "ledger",
    "order",
    "orders",
    "payment",
    "payments",
    "product",
    "products",
    "record",
    "records",
    "source",
    "sources",
)

SOURCE_VERIFICATION_TERMS = (
    "against source",
    "accept within",
    "check against source",
    "check against the source",
    "checked against source",
    "checked against the source",
    "independent verification",
    "independently verify",
    "recompute",
    "recomputed",
    "source records as source of truth",
    "source records as the source of truth",
    "source of truth for all exact figures",
    "source of truth for exact figures",
    "source-backed",
    "tie out",
    "tie-out",
    "verification requirement",
    "verify calculations",
    "verify numerical",
)

SOURCE_AVAILABILITY_STATUS_TERMS = (
    "accessible",
    "available",
    "contains",
    "empty",
    "exist",
    "exists",
    "missing",
    "not accessible",
    "not available",
    "not present",
    "present",
    "unavailable",
)

SOURCE_AVAILABILITY_OBSERVATION_TERMS = (
    "checked",
    "confirmed",
    "contains",
    "found",
    "inspected",
    "listed",
    "located",
    "opened",
    "read",
    "searched",
    "verified",
    "included",
    "are accessible",
    "are present",
    "is accessible",
    "is present",
    "was accessible",
    "was available",
    "was missing",
    "was not accessible",
    "was not available",
    "was not present",
    "was present",
    "was unavailable",
    "were accessible",
    "were available",
    "were missing",
    "were not accessible",
    "were not available",
    "were not present",
    "were present",
    "were unavailable",
)

SOURCE_AVAILABILITY_SUBJECT_TERMS = (
    "connector output",
    "connector outputs",
    "expected source",
    "expected sources",
    "order export",
    "order exports",
    "order-line export",
    "order-line exports",
    "source database",
    "source databases",
    "source directory",
    "source directories",
    "source export",
    "source exports",
    "source file",
    "source files",
    "sources",
    "source workbook",
    "source workbooks",
    "tool-output",
    "tool-outputs",
)
SOURCE_AVAILABILITY_FILE_RE = re.compile(
    r"(?:"
    r"/[A-Za-z0-9._~:@%+=,-]+(?:/[A-Za-z0-9._~:@%+=,-]+)*"
    r"|\b[A-Za-z0-9._-]+\.(?:csv|json|xls|xlsx)\b"
    r"|\b[A-Za-z][A-Za-z0-9]+(?:_[A-Za-z0-9]+){1,}\b"
    r")"
)
SOURCE_AVAILABILITY_ACCESS_LABEL_RE = re.compile(
    r"\b(?:"
    r"source/trajectory check"
    r"|source/trajectory-choice audit"
    r"|source-choice audit"
    r"|trajectory analytical command"
    r"|trajectory check"
    r"|trajectory/source check"
    r"|trajectory/source-choice audit"
    r")\b"
)
SOURCE_AVAILABILITY_ACCESS_ACTION_TERMS = (
    "displayed",
    "explored",
    "exported",
    "extracted",
    "inspected",
    "loaded",
    "opened",
    "parsed",
    "queried",
    "read",
    "used",
    "viewed",
)

SOURCE_AVAILABILITY_CONNECTOR_CONTEXT_TERMS = (
    "data",
    "directory",
    "directories",
    "export",
    "exports",
    "file",
    "files",
    "order",
    "orders",
    "source",
    "sources",
    "tool-output",
    "tool-outputs",
)

SOURCE_AVAILABILITY_NO_RESULT_SUBJECT_RE = (
    r"(?:"
    r"source(?:s| files?| directories?| exports?)?"
    r"|tool-outputs?"
    r"|connector outputs?"
    r"|order-line exports?"
    r"|(?:[a-z0-9_-]+/)+[a-z0-9_-]+\.(?:csv|xlsx|json)"
    r"|orders?\.(?:csv|xlsx|json)"
    r"|[a-z0-9_-]+\.(?:csv|xlsx|json)"
    r")"
)
SOURCE_AVAILABILITY_NO_RESULT_PATTERNS = (
    re.compile(
        rf"\b(?:listed|searched|returned|showed|displayed|found|rg --files|find command)\b"
        rf"[^;\n]{{0,240}}\bno\b(?!\s+references?\s+to\b)[^;\n]{{0,180}}\b"
        rf"{SOURCE_AVAILABILITY_NO_RESULT_SUBJECT_RE}\b"
    ),
    re.compile(
        rf"\b(?:listed|searched|returned|showed|displayed|found|rg --files|find command)\b"
        rf"[^.;\n]{{0,180}}\bno\b(?!\s+references?\s+to\b)[^.;\n]{{0,180}}\b"
        rf"{SOURCE_AVAILABILITY_NO_RESULT_SUBJECT_RE}\b"
    ),
    re.compile(
        rf"\bno\b(?!\s+references?\s+to\b)[^.;\n]{{0,180}}\b"
        rf"{SOURCE_AVAILABILITY_NO_RESULT_SUBJECT_RE}\b[^.;\n]{{0,120}}\b"
        r"(?:found|returned|listed|present|visible|appeared)\b"
    ),
)
SOURCE_AVAILABILITY_INVERTED_STATUS_RE = re.compile(
    r"\b(?:accessible|available|present)\b[^.;\n]{0,180}\b(?:was|were)\b"
)

SOURCE_VERIFICATION_ACTION_TERMS = (
    "independently verified",
    "recalculated",
    "recomputed",
    "verified against",
)

SOURCE_VERIFICATION_CHECK_LABEL_RE = re.compile(
    r"\b(?:"
    r"financial source check"
    r"|independently checked (?:available )?source csvs? with python"
    r"|independent (?:shopify|quickbooks|faire|square|source) verification"
    r"|independent source calculation"
    r"|independent source check"
    r"|source calculation"
    r"|source check with python"
    r"|source/trajectory check"
    r"|source verification script output"
    r"|trajectory/source check"
    r"|csv verification"
    r"|raw finance-source xml check"
    r")\b"
)
SOURCE_VERIFICATION_FILE_RE = re.compile(
    r"(?:"
    r"/[A-Za-z0-9._~:@%+=,-]+/[A-Za-z0-9._~:@%+=,-]+(?:/[A-Za-z0-9._~:@%+=,-]+)*"
    r"|\b[A-Za-z0-9._-]+\.(?:csv|json|xls|xlsx)\b"
    r"|\b(?:csvs?|json files?|xlsx files?|xls files?)\b"
    r"|\bsource\s+json\b"
    r"|\braw(?:\s+[a-z0-9/&-]+){0,8}\s+json\b"
    r"|\braw xlsx xml\b"
    r"|\b(?:bronze|silver|golden) deliverable\b"
    r"|\bsource cells?\b"
    r"|\bsource-of-truth checkpoints?\b"
    r")"
)
SOURCE_VERIFICATION_SCRIPT_RE = re.compile(r"\b[A-Za-z0-9._-]+\.py\b")
SOURCE_VERIFICATION_SCRIPT_SOURCE_TERMS = (
    "budget source",
    "cogs tracker",
    "golden deliverable",
    "margin tracker",
    "source data",
    "source figures",
    "source records",
    "source rows",
    "source totals",
    "source workbook",
    "source workbooks",
)
SOURCE_VERIFICATION_RESULT_RE = re.compile(
    r"(?:"
    r"\$[0-9]"
    r"|\b[0-9][0-9,.]*\b"
    r"|\b(?:calculated|computed|confirmed|contain|contains|extracted|gives|matched|produced|returned|showed)\b"
    r")"
)


def has_evidence_term(text: str, term: str) -> bool:
    """Return whether text includes a term as a word or exact phrase."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def has_any_evidence_term(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether text includes any evidence term as a word or phrase."""
    return any(has_evidence_term(text, term) for term in terms)


def has_source_availability_subject(text: str) -> bool:
    """Return whether text names likely source material rather than generic prose."""
    if has_any_evidence_term(text, SOURCE_AVAILABILITY_SUBJECT_TERMS):
        return True
    return has_any_evidence_term(text, SOURCE_CONNECTOR_NAMES) and has_any_evidence_term(
        text,
        SOURCE_AVAILABILITY_CONNECTOR_CONTEXT_TERMS,
    )


def has_source_availability_no_result(text: str) -> bool:
    """Return whether text reports an observed absence of named source material."""
    return any(pattern.search(text) for pattern in SOURCE_AVAILABILITY_NO_RESULT_PATTERNS)


def has_source_availability_status_observation(text: str) -> bool:
    """Return whether one clause reports an observed source availability status."""
    for clause in re.split(r"[.;\n]", text):
        if (
            has_source_availability_subject(clause)
            and has_any_evidence_term(clause, SOURCE_AVAILABILITY_STATUS_TERMS)
            and has_any_evidence_term(clause, SOURCE_AVAILABILITY_OBSERVATION_TERMS)
        ):
            return True
    return False


def has_source_availability_detail(text: str) -> bool:
    """Return whether evidence reports a concrete source availability status."""
    if has_source_availability_no_result(text) or has_source_availability_status_observation(text):
        return True
    if SOURCE_AVAILABILITY_INVERTED_STATUS_RE.search(text) and (
        has_source_availability_subject(text)
        or has_any_evidence_term(text, SOURCE_CONNECTOR_NAMES)
        or bool(SOURCE_AVAILABILITY_FILE_RE.search(text))
    ):
        return True
    if (
        has_any_evidence_term(text, SOURCE_AVAILABILITY_STATUS_TERMS)
        and has_any_evidence_term(text, SOURCE_AVAILABILITY_OBSERVATION_TERMS)
        and (
            has_source_availability_subject(text)
            or has_any_evidence_term(text, SOURCE_CONNECTOR_NAMES)
            or bool(SOURCE_AVAILABILITY_FILE_RE.search(text))
        )
    ):
        return True
    for clause in re.split(r"[.;\n]", text):
        if not (
            has_any_evidence_term(clause, SOURCE_AVAILABILITY_STATUS_TERMS)
            and has_any_evidence_term(clause, SOURCE_AVAILABILITY_OBSERVATION_TERMS)
        ):
            continue
        if (
            has_source_availability_subject(clause)
            or has_any_evidence_term(clause, SOURCE_CONNECTOR_NAMES)
            or bool(SOURCE_AVAILABILITY_FILE_RE.search(clause))
        ):
            return True
    return False


def has_source_access_observation(text: str) -> bool:
    """Return whether evidence proves source access through retrieval or verification."""
    if (
        any(marker in text for marker in SOURCE_VERIFICATION_AUDIT_MARKERS) and has_source_verification_detail(text)
    ) or has_source_verification_observation(text):
        return True
    if not SOURCE_AVAILABILITY_ACCESS_LABEL_RE.search(text):
        return False
    if not has_any_evidence_term(text, SOURCE_AVAILABILITY_ACCESS_ACTION_TERMS):
        return False
    return (
        has_source_availability_subject(text)
        or has_any_evidence_term(text, SOURCE_CONNECTOR_NAMES)
        or bool(SOURCE_AVAILABILITY_FILE_RE.search(text))
    )


def has_source_verification_observation(text: str) -> bool:
    """Return whether text reports a concrete source-backed verification result."""
    return (
        bool(SOURCE_VERIFICATION_CHECK_LABEL_RE.search(text))
        and bool(SOURCE_VERIFICATION_FILE_RE.search(text))
        and bool(SOURCE_VERIFICATION_RESULT_RE.search(text))
    )


def has_source_verification_detail(text: str) -> bool:
    """Return whether source-verification evidence names source material plus a check/result."""
    names_source_material = bool(SOURCE_VERIFICATION_FILE_RE.search(text)) or (
        bool(SOURCE_VERIFICATION_SCRIPT_RE.search(text))
        and has_any_evidence_term(text, SOURCE_VERIFICATION_SCRIPT_SOURCE_TERMS)
    )
    return names_source_material and (
        any(term in text for term in SOURCE_VERIFICATION_ACTION_TERMS)
        or bool(SOURCE_VERIFICATION_RESULT_RE.search(text))
    )


def has_source_guidance_conflict_language(text: str) -> bool:
    """Return whether text reports a conflict between accessible source evidence and guidance."""
    return bool(SOURCE_GUIDANCE_CONFLICT_RE.search(text))


def has_source_guidance_conflict_audit(evidence: list[str]) -> bool:
    """Return whether evidence labels and reconciles a source/guidance conflict."""
    for item in evidence:
        lowered = item.lower()
        has_marker = any(marker in lowered for marker in SOURCE_GUIDANCE_CONFLICT_AUDIT_MARKERS)
        if (
            has_marker
            and SOURCE_GUIDANCE_NO_CONFLICT_RE.search(lowered)
            and "source" in lowered
            and "guidance" in lowered
        ):
            return True
        if (
            has_marker
            and has_source_guidance_conflict_language(lowered)
            and SOURCE_GUIDANCE_AUTHORITY_RE.search(lowered)
        ):
            return True
    return False


def has_action_side_effect_observation(text: str) -> bool:
    """Return whether text reports an observed action or side-effect check."""
    if not has_any_evidence_term(text, ACTION_SIDE_EFFECT_CONTEXT_TERMS):
        return False
    return any(pattern.search(text) for pattern in ACTION_SIDE_EFFECT_NO_RESULT_PATTERNS)


def has_action_side_effect_detail(text: str) -> bool:
    """Return whether evidence reports a concrete action or side-effect check."""
    if has_action_side_effect_observation(text):
        return True
    if not has_any_evidence_term(text, ACTION_SIDE_EFFECT_CONTEXT_TERMS):
        return False
    return has_any_evidence_term(text, ACTION_SIDE_EFFECT_DETAIL_TERMS) and bool(
        re.search(ACTION_SIDE_EFFECT_SUBJECT_RE, text)
    )


def has_unlabeled_action_side_effect_detail(text: str) -> bool:
    """Return whether unlabeled evidence concretely checks trajectory action state."""
    return has_any_evidence_term(text, ACTION_SIDE_EFFECT_TRAJECTORY_CONTEXT_TERMS) and has_action_side_effect_detail(
        text
    )


def requires_action_side_effect_audit(instructions: str, judge_guidance: str) -> bool:
    """Return whether task/guidance text calls for external-action auditing."""
    combined = f"{instructions}\n{judge_guidance}".lower()
    for pattern in ACTION_SIDE_EFFECT_NON_REQUIREMENT_PATTERNS:
        combined = pattern.sub("\n", combined)
    return any(term in combined for term in ACTION_SIDE_EFFECT_TERMS)


def requires_source_availability_audit(instructions: str, judge_guidance: str) -> bool:
    """Return whether task/guidance names expected source material to audit."""
    combined = f"{instructions}\n{judge_guidance}".lower()
    if any(term in combined for term in SOURCE_EXPECTATION_TERMS):
        return True

    context_pattern = "|".join(re.escape(word) for word in SOURCE_CONTEXT_WORDS)
    for connector in SOURCE_CONNECTOR_NAMES:
        connector_pattern = re.escape(connector)
        if re.search(rf"\b{connector_pattern}\b[^.\n]{{0,120}}\b({context_pattern})\b", combined):
            return True
        if re.search(rf"\b({context_pattern})\b[^.\n]{{0,120}}\b{connector_pattern}\b", combined):
            return True

    return False


def requires_source_verification_audit(instructions: str, judge_guidance: str) -> bool:
    """Return whether task/guidance calls for independent source verification."""
    combined = f"{instructions}\n{judge_guidance}".lower()
    return any(term in combined for term in SOURCE_VERIFICATION_TERMS)


def extract_absolute_paths(text: str) -> set[str]:
    """Extract normalized absolute paths from free-form task/guidance text."""
    return {match.group(0).rstrip(".,;:)']\"`") for match in ABSOLUTE_PATH_RE.finditer(text)}


def extract_output_location_paths(text: str) -> set[str]:
    """Extract absolute paths that look like output, artifact, or workspace locations."""
    output_paths: set[str] = set()
    for path in extract_absolute_paths(text):
        lowered = path.lower()
        if any(marker in lowered for marker in OUTPUT_PATH_MARKERS):
            output_paths.add(path.rstrip("/"))
    return output_paths


def paths_are_compatible(left: str, right: str) -> bool:
    """Return whether two output paths are the same location or one contains the other."""
    left_norm = left.rstrip("/")
    right_norm = right.rstrip("/")
    return left_norm == right_norm or left_norm.startswith(f"{right_norm}/") or right_norm.startswith(f"{left_norm}/")


def requires_output_location_conflict_audit(instructions: str, judge_guidance: str) -> bool:
    """Return whether task instructions and guidance name conflicting output locations."""
    instruction_paths = extract_output_location_paths(instructions)
    guidance_paths = extract_output_location_paths(judge_guidance)
    if not instruction_paths or not guidance_paths:
        return False
    return any(
        not any(paths_are_compatible(instruction_path, guidance_path) for instruction_path in instruction_paths)
        for guidance_path in guidance_paths
    )


def has_action_side_effect_audit(evidence: list[str]) -> bool:
    """Return whether evidence includes the required named action audit item."""
    for item in evidence:
        lowered = item.lower()
        if any(marker in lowered for marker in ACTION_SIDE_EFFECT_AUDIT_MARKERS) and has_action_side_effect_detail(
            lowered
        ):
            return True
        if has_action_side_effect_observation(lowered):
            return True
        if has_unlabeled_action_side_effect_detail(lowered):
            return True
    return False


def has_output_location_conflict_audit(evidence: list[str]) -> bool:
    """Return whether evidence includes the required output-location conflict audit item."""
    for item in evidence:
        lowered = item.lower()
        if any(marker in lowered for marker in OUTPUT_LOCATION_CONFLICT_AUDIT_MARKERS) and (
            bool(extract_absolute_paths(lowered)) or has_any_evidence_term(lowered, OUTPUT_LOCATION_DETAIL_TERMS)
        ):
            return True
    return False


def has_source_availability_audit(evidence: list[str]) -> bool:
    """Return whether evidence includes the required source availability audit item."""
    for item in evidence:
        lowered = item.lower()
        if any(marker in lowered for marker in SOURCE_AVAILABILITY_AUDIT_MARKERS) and has_source_availability_detail(
            lowered
        ):
            return True
    return False


def has_source_verification_audit(evidence: list[str]) -> bool:
    """Return whether evidence includes the required source verification audit item."""
    for item in evidence:
        lowered = item.lower()
        if any(marker in lowered for marker in SOURCE_VERIFICATION_AUDIT_MARKERS) and has_source_verification_detail(
            lowered
        ):
            return True
        if has_source_verification_observation(lowered):
            return True
    return False


def strip_score_calibration_markers(text: str) -> str:
    """Remove score-calibration marker text before checking for substantive decisions."""
    for marker in SCORE_CALIBRATION_AUDIT_MARKERS:
        text = text.replace(marker, "")
    return text.strip(" \t\n\r:-.")


def has_score_calibration_decision(text: str) -> bool:
    """Return whether score-calibration evidence names a band, cap, or ceiling decision."""
    markerless = strip_score_calibration_markers(text)
    if WEAK_SCORE_CALIBRATION_NO_CAP_RE.fullmatch(markerless):
        return False
    return (
        bool(SCORE_CALIBRATION_NUMERIC_DECISION_RE.search(markerless))
        and any(pattern.search(markerless) for pattern in SCORE_CALIBRATION_DECISION_PATTERNS)
        and bool(score_calibration_ceiling_values(markerless))
    )


def score_calibration_ceiling_values(text: str) -> list[float]:
    """Return numeric ceiling values from score-calibration text."""
    ceilings: list[float] = []
    markerless = strip_score_calibration_markers(text)
    for clause in SCORE_CALIBRATION_CLAUSE_SPLIT_RE.split(markerless):
        if SCORE_CALIBRATION_NON_APPLIED_CAP_RE.search(clause):
            continue
        if SCORE_CALIBRATION_SELECTED_SCORE_BELOW_CAP_RE.search(clause):
            continue
        for pattern in SCORE_CALIBRATION_CEILING_PATTERNS:
            ceilings.extend(float(match.group(1)) for match in pattern.finditer(clause))
    return ceilings


def extract_score_calibration_ceiling(evidence: list[str]) -> float | None:
    """Extract the strictest numeric score ceiling declared by calibration evidence."""
    ceilings: list[float] = []
    for item in evidence:
        lowered = item.lower()
        if not any(marker in lowered for marker in SCORE_CALIBRATION_AUDIT_MARKERS):
            continue
        ceilings.extend(score_calibration_ceiling_values(lowered))
    return min(ceilings) if ceilings else None


def has_score_calibration_audit(evidence: list[str]) -> bool:
    """Return whether evidence includes the required score calibration/cap audit item."""
    for item in evidence:
        lowered = item.lower()
        if any(marker in lowered for marker in SCORE_CALIBRATION_AUDIT_MARKERS) and has_score_calibration_decision(
            lowered
        ):
            return True
    return False


def has_score_calibration_audit_marker(evidence: list[str]) -> bool:
    """Return whether evidence includes a score-calibration audit label."""
    return any(any(marker in item.lower() for marker in SCORE_CALIBRATION_AUDIT_MARKERS) for item in evidence)


def has_foundational_failure_language(text: str) -> bool:
    """Return whether text describes foundational, central, or required-verification failures."""
    return bool(FOUNDATIONAL_FAILURE_RE.search(text))


def has_above_midpoint_justification(evidence: list[str]) -> bool:
    """Return whether score-calibration evidence explicitly justifies above-midpoint placement."""
    for item in evidence:
        lowered = item.lower()
        if not any(marker in lowered for marker in SCORE_CALIBRATION_AUDIT_MARKERS):
            continue
        if ABOVE_MIDPOINT_JUSTIFICATION_RE.search(item) or SCORE_PLACEMENT_POSITIVE_JUSTIFICATION_RE.search(item):
            return True
    return False


def requires_above_midpoint_justification(
    score: float,
    score_ceiling: float | None,
    *,
    reasoning: str,
    evidence: list[str],
) -> bool:
    """Return whether a high score with foundational failures needs explicit justification."""
    if score_ceiling is None:
        return False
    if score_ceiling < MIN_NEAR_CEILING_FOUNDATIONAL_FAILURE_CEILING:
        return False
    near_declared_ceiling = score >= score_ceiling - NEAR_DECLARED_CEILING_MARGIN
    high_foundational_failure_score = (
        score_ceiling <= MAX_HIGH_FOUNDATIONAL_FAILURE_CEILING and score >= MIN_HIGH_FOUNDATIONAL_FAILURE_SCORE
    )
    if not near_declared_ceiling and not high_foundational_failure_score:
        return False
    combined = " ".join([reasoning, *evidence])
    return has_foundational_failure_language(combined) and not has_above_midpoint_justification(evidence)


def has_workspace_artifact_evidence(evidence: list[str]) -> bool:
    """Return whether evidence includes a workspace, artifact, or final-state check."""
    for item in evidence:
        lowered = item.lower()
        if any(marker in lowered for marker in OUTPUT_LOCATION_CONFLICT_AUDIT_MARKERS):
            continue
        if (
            has_any_evidence_term(lowered, INCONCLUSIVE_WORKSPACE_ARTIFACT_TERMS)
            and any(marker in lowered for marker in ORIGINAL_WORKSPACE_PATH_MARKERS)
            and not any(marker in lowered for marker in CLONED_WORKSPACE_PATH_MARKERS)
        ):
            continue
        if any(marker in lowered for marker in WORKSPACE_ARTIFACT_EVIDENCE_MARKERS):
            return True
        if any(
            any(marker in path.lower() for marker in ("deliverable", "output", "workdir"))
            for path in extract_absolute_paths(item)
        ):
            return True
    return False


def has_unavailable_trajectory_claim(text: str) -> bool:
    """Return whether text says the trajectory itself was unavailable."""
    return any(pattern.search(text) for pattern in TRAJECTORY_UNAVAILABLE_PATTERNS)


def has_trajectory_evidence(evidence: list[str]) -> bool:
    """Return whether evidence includes a trajectory, step, or tool-call check."""
    for item in evidence:
        lowered = item.lower()
        if has_unavailable_trajectory_claim(lowered):
            continue
        if any(marker in lowered for marker in TRAJECTORY_EVIDENCE_MARKERS):
            return True
    return False
