"""
LangGraph workflow for IEM BOM comparison.

Defines a self-correcting agent graph that:
1. Converts both PDFs (schematic + wiring diagram) to page images.
2. OCRs the first page of each to extract the Drawing Index.
3. Identifies BOM pages from the Drawing Index.
4. Extracts and concatenates BOM tables from all BOM pages.
5. Compares the two BOMs using a Bedrock Claude model.
6. Retries with self-correction if any step produces an error.

The graph uses tool-calling via MCP to perform all document processing,
and Bedrock Claude (via ChatBedrock) as the orchestration LLM.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Literal

from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .state import BOMComparisonState
from .tools import get_mcp_tools

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

TOOL_RESULT_PREVIEW_LEN = 2000


def _print_banner(text: str, color: str = CYAN) -> None:
    """Print a prominent section banner to stderr.

    Args:
        text: Banner text to display.
        color: ANSI color code for the banner.
    """
    width = max(len(text) + 4, 60)
    border = "═" * width
    print(f"\n{color}{BOLD}{border}{RESET}", file=sys.stderr)
    print(f"{color}{BOLD}  {text}{RESET}", file=sys.stderr)
    print(f"{color}{BOLD}{border}{RESET}\n", file=sys.stderr)


def _print_step(icon: str, label: str, detail: str = "", color: str = GREEN) -> None:
    """Print a single workflow step line to stderr.

    Args:
        icon: Emoji or symbol prefix.
        label: Short step label.
        detail: Optional detail text (may be multi-line).
        color: ANSI color code.
    """
    print(f"{color}{BOLD}{icon} {label}{RESET}", file=sys.stderr)
    if detail:
        for line in detail.splitlines():
            print(f"  {DIM}{line}{RESET}", file=sys.stderr)


def _truncate(text: str, max_len: int = TOOL_RESULT_PREVIEW_LEN) -> str:
    """Truncate text with an ellipsis indicator if it exceeds max_len.

    Args:
        text: Input string.
        max_len: Maximum character length before truncation.

    Returns:
        Truncated string with trailing indicator if shortened.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n  ... ({len(text) - max_len} more chars)"


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool call arguments for display, truncating long values.

    Args:
        args: Dictionary of tool call arguments.

    Returns:
        Formatted multi-line string of key=value pairs.
    """
    lines = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 120:
            v_str = v_str[:120] + "..."
        lines.append(f"    {k} = {v_str}")
    return "\n".join(lines)


def _detect_phase(text: str) -> str:
    """Detect which workflow phase the agent is actively starting.

    Only matches explicit phase start markers at the beginning of lines,
    not casual mentions of phase names in reasoning text.

    Args:
        text: The agent's reasoning text.

    Returns:
        Phase label string, or empty string if not detected.
    """
    import re

    for line in text.splitlines():
        line_s = line.strip().upper()

        m = re.match(r"^\*{0,2}PHASE\s+([A-H])\b", line_s)
        if m:
            phase_map = {
                "A": "PHASE A: SCHEMATIC",
                "B": "PHASE B: WIRING DIAGRAM",
                "C": "PHASE C: COMPARING BOMs",
                "D": "PHASE D: FINAL REPORT",
                "E": "PHASE E: SCHEMATIC 3L DIAGRAMS",
                "F": "PHASE F: WD 3L DIAGRAMS",
                "G": "PHASE G: CIRCUIT COMPARISON",
                "H": "PHASE H: WD ANALYSIS",
            }
            return phase_map.get(m.group(1), "")

        m = re.match(r"^\*{0,2}([A-H])(\d)[.:\s]", line_s)
        if m:
            phase_map = {
                "A": "PHASE A: SCHEMATIC",
                "B": "PHASE B: WIRING DIAGRAM",
                "C": "PHASE C: COMPARING BOMs",
                "D": "PHASE D: FINAL REPORT",
                "E": "PHASE E: SCHEMATIC 3L DIAGRAMS",
                "F": "PHASE F: WD 3L DIAGRAMS",
                "G": "PHASE G: CIRCUIT COMPARISON",
                "H": "PHASE H: WD ANALYSIS",
            }
            return phase_map.get(m.group(1), "")

    return ""


PROMPT_HEADER = """You are an expert electrical engineering document analyst working for IEM.
Your job is to review drawing packages by comparing an original schematic (the
source of truth) against a wiring diagram (WD, the edited copy) and find mistakes.

You have access to tools that can:
- Convert PDFs to page images (pdf_to_images) — renders at 300 DPI
- Read Drawing Index from image (read_drawing_index_from_image) — vision-based, no OCR
- Validate page is BOM (validate_page_is_bom) — confirms a page is actually a BOM table
- OCR page images to extract text (ocr_page_text) — only if needed as fallback
- Parse the Drawing Index from OCR text (parse_drawing_index) — regex fallback
- Extract BOM tables from BOM page images (extract_bom_table_from_page)
- Concatenate BOM CSV files (concat_bom_csvs)
- Compare BOMs with vision (compare_boms_with_vision)
- Compare BOM images direct (compare_bom_images_direct) — sends full BOM page images to Claude, no OCR
- Generate BOM HTML report (generate_html_report)
- Extract diagram pages (extract_diagram_pages)
- Analyze 3L diagram (analyze_3l_diagram) — confirms page type + extracts curly brace labels
- Validate WD page label (validate_wd_page_label) — checks uncropped page title block, crops if match
- Resolve continuations (resolve_continuations) — maps labels to _WD_ pages, crops them
- Compare single 3L pair (compare_single_3l_pair) — one SCH vs one WD + continuations
- Generate circuit HTML report (generate_circuit_html_report)
- Analyze single WD page (analyze_single_wd_page) — inspect one WD page for errors + wire numbers
- Check wire uniqueness across pages (check_wire_uniqueness_across_pages) — find duplicate wires
- Generate WD analysis report (generate_wd_analysis_report)

═══════════════════════════════════════════════════════════════════════
CRITICAL RULES:
1. Process ONE document at a time. Complete the SCHEMATIC fully before
   touching the Wiring Diagram.
2. Make ONE tool call at a time. Never batch multiple tool calls in
   parallel. Wait for each result, validate it, then proceed.
3. After EVERY tool result, print what you received and confirm it
   looks correct before moving on.
═══════════════════════════════════════════════════════════════════════

PAGE NUMBER MAPPING — CRITICAL:
  The Drawing Index lists page numbers (1, 2, 3, ...). The PDF page keys
  are page_001, page_002, etc. The mapping between them varies by document:
  - Some documents have a 1-page offset (Drawing Index page N = PDF page_{N-1})
  - Some documents have no offset (Drawing Index page N = PDF page_{N})
  - The schematic and WD may have DIFFERENT offsets.
  STRATEGY: For each page number N, try page_{N:03d} first. If that page
  doesn't contain the expected content, try page_{N-1:03d} as fallback.
  Do NOT assume the same offset for both documents."""

PROMPT_COMMON_EXTRACT = """
PHASE A: PROCESS THE SCHEMATIC (source of truth)
──────────────────────────────────────────────────
A1. Call pdf_to_images for the schematic PDF (leave output_dir empty, dpi=300).
    VALIDATE: Print page count.

A2. Call read_drawing_index_from_image with the schematic page_001 image path.
    This uses vision to read the Drawing Index directly from the image —
    no OCR needed. If page_001 has no Drawing Index, try page_002.
    VALIDATE: Print the bom_pages and three_line_pages lists.

PHASE B: PROCESS THE WIRING DIAGRAM (edited copy)
──────────────────────────────────────────────────
B1. Call pdf_to_images for the WD PDF (leave output_dir empty, dpi=300).
    VALIDATE: Print page count.

B2. Call read_drawing_index_from_image with the WD page_001 image path.
    If page_001 has no Drawing Index, try page_002.
    VALIDATE: Print the bom_pages and three_line_pages lists."""

PROMPT_BOM_PHASES = """
PHASE A4: IDENTIFY AND VALIDATE SCHEMATIC BOM PAGE IMAGES
──────────────────────────────────────────────────────────
A4. From the Drawing Index (A2), you know which page numbers are BOM pages.
    For EACH BOM page number N:
    a) Try page_{N:03d} first. Call validate_page_is_bom on that image.
    b) If is_bom=true → use this page. Print "page_{N:03d}: CONFIRMED BOM"
    c) If is_bom=false → try page_{N-1:03d}. Call validate_page_is_bom.
    d) If is_bom=true → use this page. Print "page_{N-1:03d}: CONFIRMED BOM"
    e) If neither is BOM → print warning and skip.
    Only collect CONFIRMED BOM page image paths.
    DO NOT crop — use the full uncropped page images.

PHASE B4: IDENTIFY AND VALIDATE WD BOM PAGE IMAGES
───────────────────────────────────────────────────
B4. Same as A4 but for the WD. The WD may have a DIFFERENT page offset.
    Validate each candidate with validate_page_is_bom before using it.

PHASE C: COMPARE BOMs (direct image comparison)
────────────────────────────────────────────────
C1. Call compare_bom_images_direct with:
    - schematic_bom_image_paths_json: JSON list of SCH BOM image paths
    - wd_bom_image_paths_json: JSON list of WD BOM image paths
    These are FULL uncropped page images. Claude reads the BOM tables
    directly from the images.
    VALIDATE: Print the assessment and issue count.

PHASE D: BOM HTML REPORT
─────────────────────────
D1. Call generate_html_report with the comparison JSON.
    For schematic_bom_pages_json and wd_bom_pages_json, pass JSON lists
    where each entry has: {"cropped_path": "<the image path>",
    "label": "BOM Page N (sheet_name)"}.
    Output: bom_comparison_report.html
D2. Print summary: issue counts + file path."""

PROMPT_CIRCUIT_PHASES = """
PHASE E: EXTRACT AND VALIDATE SCHEMATIC 3L DIAGRAMS

E0. Re-render the SCHEMATIC PDF at 300 DPI for higher resolution.
    Call pdf_to_images with the schematic PDF and dpi=300.
    Store this as schematic_pages_hd.

E1. From the SCHEMATIC Drawing Index (parsed in A3), list ALL entries
    where the sheet name contains "_3L_".

E2. Compute PDF page keys using -1 offset (page N -> page_{N-1:03d}).
    Call extract_diagram_pages with schematic_pages_hd, the offset
    page numbers, and dpi=300.

E3. For EACH cropped page, call analyze_3l_diagram.
    Only confirm is_3l_diagram. Ignore curly_brace_labels for schematic.
    Print: "Page X (sheet Y): is_3l_diagram=[true/false], section=[label]"

E4. Print a table of confirmed schematic 3L diagrams:
    | # | Sheet | Page | Section | Status |

PHASE F: EXTRACT AND VALIDATE WD 3L DIAGRAMS (with curly brace detection)

F0. Re-render the WD PDF at 300 DPI.
    Call pdf_to_images with the WD PDF and dpi=300.
    Store this as wd_pages_hd.

F1. From the WD Drawing Index (parsed in B3), list ALL entries
    where the sheet name contains "_3L_".

F2. Compute PDF page keys using -1 offset.
    Call extract_diagram_pages with wd_pages_hd, the offset page
    numbers, and dpi=300.

F3. For EACH cropped WD page (one at a time), call analyze_3l_diagram.
    For WD pages you need BOTH:
    - is_3l_diagram confirmation
    - curly_brace_labels (these are the circuit continuation references)
    Print: "Page X (sheet Y): is_3l_diagram=[true/false], section=[label]"
    Print: "  Curly brace labels: [list of labels found]"
    If false: "SKIP — detected as [type]"

F4. Print a table of confirmed WD 3L diagrams with their curly brace labels:
    | # | Sheet | Page | Section | Curly Brace Labels | Status |
    |---|-------|------|---------|--------------------|--------|
    | 1 | 11_3L_01 | 8 | SECTION 1 | STB-U/SEC.1 RIGHT PAN, FDISC-PTU/SEC.1 BACKPAN | CONFIRMED |
    | 2 | 11_3L_02 | 9 | SECTION 2 | FDISC-7/SEC.2 BACKPAN | CONFIRMED |

    STOP — 3L diagram extraction and validation complete for both documents.

PHASE G: RESOLVE WD CONTINUATIONS AND COMPARE (one pair at a time)
──────────────────────────────────────────────────────────────────
The assumption is confirmed: SCHEMATIC uses only 3L diagrams. The WIRING
DIAGRAM uses 3L diagrams PLUS curly brace references to _WD_ detail pages.

For EACH matched 3L pair (from the sheet-name mapping):

G1. RESOLVE WD CONTINUATION PAGES
    Take the curly_brace_labels from the WD 3L analysis (Phase F3).
    Call resolve_continuations with:
    - curly_brace_labels_json: the labels list
    - continuation_sheets_json: [] (empty — we use labels, not sheet refs)
    - drawing_index_json: the FULL WD Drawing Index entries (from B3)
    - document_pages_json: the WD page image map (from B1)
    The tool searches for _WD_ pages matching by section number.

    Print what was resolved:
    "  Label: STB-U/SEC.1 RIGHT PAN → matched 11_WD_02 (page 20)"
    "  Label: FDISC-PTU/SEC.1 BACKPAN → matched 11_WD_01 (page 19)"

G2. VALIDATE EACH RESOLVED WD PAGE (without cropping first)
    For each resolved continuation page from G1:
    Call validate_wd_page_label with:
    - page_image_path: the ORIGINAL uncropped page path (from pdf_to_images,
      use the page number from resolve_continuations to look up the key)
    - expected_reference: the curly brace label or matched description
    The tool reads the title block at the bottom-right of the uncropped page.
    If it matches → the tool crops it and returns cropped_path.
    If not → skip this page.

    Print: "  11_WD_02: label='SEC.1 - LEFT/RIGHT PAN WIRING' → MATCH, cropped"
    Print: "  11_WD_05: label='FRONT VIEW' → MISMATCH, skipping"

G3. COMPARE THIS PAIR
    Call compare_single_3l_pair with:
    - schematic_3l_path: the SCH 3L cropped image
    - wd_3l_path: the WD 3L cropped image
    - wd_continuation_paths_json: JSON list of validated+cropped WD pages
    - pair_label: e.g. "11_3L_01 (Section 1)"

    IMPORTANT: The comparison is NOT 1:1. The WD has MORE detail than the
    schematic. Only flag OBVIOUS errors:
    - Missing circuits that exist in SCH but not in WD
    - Wrong component ratings/values
    - Wrong wire labels or terminal numbers
    Do NOT flag extra information in the WD — it's expected to have more detail.

    Print the issues found for this pair.

G4. Move to the next pair. Repeat G1–G3.

G5. After ALL pairs are compared, build the combined JSON and call
    generate_circuit_html_report. The circuit_comparison_json MUST be
    a JSON string with this exact structure:
    {"pairs": [
      {"pair_label": "11_3L_01 (Section 1)",
       "schematic_sheet": "11_3L_01",
       "wd_sheet": "11_3L_01",
       "schematic_cropped_path": "/tmp/diagram_SCH_9_xxx.png",
       "wd_cropped_path": "/tmp/diagram_3L_5_xxx.png",
       "issues": [<issues from compare_single_3l_pair for this pair>],
       "wd_continuations": [{"label": "STB-U/SEC.1 RIGHT PAN (11_WD_02)",
                              "cropped_path": "/tmp/wd_validated_xxx.png"}, ...],
       "summary": "1-2 sentence summary for this pair"},
      ...one entry per pair...
    ]}
    IMPORTANT: Include the actual file paths for schematic_cropped_path
    and wd_cropped_path from the extract_diagram_pages results. These
    are needed to embed the correct images in the HTML report.
    Pass schematic_diagrams_json and wd_diagrams_json as the lists from E/F.
    Output: circuit_comparison_report.html

G6. Print final summary table:
    | Pair | SCH Sheet | WD Sheet | WD Pages Resolved | Issues Found |
    |------|-----------|----------|-------------------|--------------|
    | 1    | 11_3L_01  | 11_3L_01 | 11_WD_01, 11_WD_02 | 2          |
    | 2    | 11_3L_02  | 11_3L_02 | 11_WD_07, 11_WD_08 | 0          |
    Print file path to HTML report."""

PROMPT_WD_ANALYSIS = """
PHASE H: STANDALONE WD DIAGRAM ANALYSIS
────────────────────────────────────────
This mode inspects ALL _WD_ pages in the Wiring Diagram document for
circuit errors and wire number uniqueness. No schematic comparison.

H1. From the WD Drawing Index (parsed in B3), list ALL entries where
    the sheet name contains "_WD_". These are the wiring detail pages.
    They should be sequential. Note page numbers.

H2. Call pdf_to_images for the WD PDF at dpi=300 if not already done.

H3. Call extract_diagram_pages with the WD page map, the _WD_ page
    numbers (with -1 offset), and dpi=300.

H4. For EACH WD page (one at a time), call analyze_single_wd_page with:
    - wd_image_path: the cropped_path
    - page_label: the sheet name (e.g. "11_WD_01 - SEC.1 BACKPAN WIRING")
    Print the result: wire count, issues found.
    Collect all results into a list.

H5. After ALL pages are analyzed, call check_wire_uniqueness_across_pages
    with the collected results as a JSON list.
    This finds wire numbers that appear on multiple different pages.
    Print: total unique wires, duplicate count, duplicate details.

H6. Build the combined analysis JSON:
    {"per_page_results": [<results from H4>],
     "cross_page_duplicates": <result from H5>}
    Call generate_wd_analysis_report with:
    - analysis_json: the combined JSON
    - wd_pages_json: the diagram entries from H3
    - output_path: "wd_analysis_report.html"

H7. Print summary: total pages, total wires, duplicates, issues."""

PROMPT_FOOTER = """
REMEMBER: One tool call at a time. Schematic first, then WD, then compare."""


def _build_system_prompt(mode: str) -> str:
    """Compose the system prompt based on the review mode.

    Args:
        mode: One of "bom", "circuit", "both", or "wdanalysis".

    Returns:
        The assembled system prompt string.
    """
    parts = [PROMPT_HEADER, PROMPT_COMMON_EXTRACT]
    if mode in ("bom", "both"):
        parts.append(PROMPT_BOM_PHASES)
    if mode in ("circuit", "both"):
        parts.append(PROMPT_CIRCUIT_PHASES)
    if mode == "wdanalysis":
        parts.append(PROMPT_WD_ANALYSIS)
    parts.append(PROMPT_FOOTER)
    return "\n".join(parts)


def _build_llm(
    model_id: str = "us.anthropic.claude-sonnet-4-6",
    region: str = "us-east-1",
) -> ChatBedrock:
    """Construct the Bedrock Claude LLM with tool-calling support.

    Args:
        model_id: Bedrock model identifier.
        region: AWS region for the Bedrock endpoint.

    Returns:
        Configured ChatBedrock instance.
    """
    from botocore.config import Config as BotoConfig

    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={"max_tokens": 128000, "temperature": 0.0},
        config=BotoConfig(read_timeout=300, connect_timeout=30),
    )


def agent_node(state: BOMComparisonState) -> dict[str, Any]:
    """Invoke the LLM with the current message history and bound tools.

    The LLM decides which tool to call next (or produces a final answer)
    based on the conversation so far. Retries with exponential backoff
    on timeout errors.

    Args:
        state: Current workflow state including message history.

    Returns:
        Updated messages list with the LLM's response appended.
    """
    tools = get_mcp_tools()
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(tools)

    messages = state["messages"]
    mode = state.get("review_mode", "both")
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=_build_system_prompt(mode))] + list(messages)

    max_retries = 3
    base_delay = 10
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as exc:
            exc_str = str(exc).lower()
            if "timeout" in exc_str or "timed out" in exc_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Agent LLM call timed out (attempt %d/%d). "
                        "Retrying in %ds...",
                        attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
            raise


def should_continue(state: BOMComparisonState) -> Literal["tools", "self_correct", "end"]:
    """Route after the agent node: call tools, self-correct, or finish.

    Examines the last AI message to decide the next step:
    - If the LLM requested tool calls → route to the tools node.
    - If there's an error and retries remain → route to self-correction.
    - Otherwise → end the workflow.

    Args:
        state: Current workflow state.

    Returns:
        Name of the next node to execute.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    if state.get("error") and state.get("retry_count", 0) < MAX_RETRIES:
        return "self_correct"

    return "end"


def self_correct_node(state: BOMComparisonState) -> dict[str, Any]:
    """Inject a correction prompt when a previous step failed.

    Reads the error from state, increments the retry counter, and
    adds a human message asking the LLM to diagnose and fix the issue.

    Args:
        state: Current workflow state with an error field set.

    Returns:
        Updated state with correction message and bumped retry count.
    """
    error = state.get("error", "Unknown error")
    retry_count = state.get("retry_count", 0) + 1

    correction_msg = HumanMessage(
        content=(
            f"The previous step encountered an error:\n\n{error}\n\n"
            f"This is retry attempt {retry_count}/{MAX_RETRIES}. "
            "Please analyze what went wrong and try a different approach. "
            "If the OCR output was malformed, try re-extracting with adjusted parameters. "
            "If a tool call failed, check the arguments and retry."
        )
    )

    return {
        "messages": [correction_msg],
        "error": "",
        "retry_count": retry_count,
    }


def handle_tool_error(state: BOMComparisonState) -> dict[str, Any]:
    """Check for tool errors and trim oversized tool results.

    Scans the latest messages for ToolMessages that indicate failure
    and stores the error text so the router can trigger self-correction.
    Also trims very large tool results to prevent context bloat — the
    full CSV data is only needed by concat_bom_csvs, and the agent
    only needs the metadata (row_count, columns) for validation.

    Args:
        state: Current workflow state after tool execution.

    Returns:
        Updated state with error field populated (or cleared) and
        trimmed messages if any were oversized.
    """
    messages = list(state["messages"])
    errors: list[str] = []
    trimmed_messages: list[Any] = []

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "error" in content.lower() or "traceback" in content.lower():
                errors.append(content)
        elif isinstance(msg, AIMessage):
            break

    if errors:
        return {"error": "\n---\n".join(errors)}
    return {"error": ""}


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow for BOM comparison.

    Graph topology::

        [start]
           │
           ▼
        agent ──tool_calls──► tools ──► check_errors ──► agent
           │                                                │
           │ (no tool calls)                                │
           ▼                                                │
        should_continue                                     │
           │                                                │
           ├─ "self_correct" ──► self_correct ──────────► agent
           │
           └─ "end" ──► [END]

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    tools = get_mcp_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(BOMComparisonState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_errors", handle_tool_error)
    graph.add_node("self_correct", self_correct_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "self_correct": "self_correct",
            "end": END,
        },
    )

    graph.add_edge("tools", "check_errors")
    graph.add_edge("check_errors", "agent")
    graph.add_edge("self_correct", "agent")

    return graph.compile()


def run_review(
    schematic_pdf: str,
    wiring_diagram_pdf: str,
    mode: str = "both",
    verbose: bool = True,
    job_id: str = "",
) -> str:
    """Execute the IEM drawing review pipeline.

    Args:
        schematic_pdf: Path to the original schematic PDF.
        wiring_diagram_pdf: Path to the wiring diagram (WD) PDF.
        mode: Review mode — "bom", "circuit", "both", or "wdanalysis".
        verbose: If True, print step-by-step execution details to stderr.
        job_id: Optional job ID for progress tracking via JobTracker.

    Returns:
        The final report as a string.
    """
    from iem_bom_agent.job_tracker import tracker as job_tracker

    graph = build_graph()

    mode_labels = {
        "bom": "BOM comparison",
        "circuit": "Circuit diagram comparison",
        "both": "Full review (BOM + Circuit)",
        "wdanalysis": "Standalone WD diagram analysis",
    }

    mode_instructions = {
        "bom": (
            "Execute Phases A–D only (BOM comparison).\n"
            "Extract and compare BOM tables, then generate the HTML report."
        ),
        "circuit": (
            "Execute Phases A1–A3 (extract Schematic Drawing Index),\n"
            "B1–B3 (extract WD Drawing Index),\n"
            "then Phase E (validate Schematic 3L diagrams),\n"
            "Phase F (validate WD 3L diagrams with curly brace detection),\n"
            "and Phase G (resolve WD continuations and compare pairs)."
        ),
        "both": (
            "Execute ALL phases: A–D (BOM comparison), then E–G (circuit comparison).\n"
            "Process BOM first, then circuits."
        ),
        "wdanalysis": (
            "Execute Phases B1–B3 (extract WD Drawing Index only),\n"
            "then Phase H (standalone WD diagram analysis).\n"
            "Skip schematic processing entirely. Only analyze the WD document\n"
            "for circuit errors and wire number uniqueness."
        ),
    }

    initial_message = HumanMessage(
        content=(
            f"Review these two IEM drawing packages:\n\n"
            f"**Schematic (source of truth):** `{schematic_pdf}`\n"
            f"**Wiring Diagram (edited copy):** `{wiring_diagram_pdf}`\n\n"
            f"**Mode:** {mode_labels[mode]}\n\n"
            f"{mode_instructions[mode]}\n\n"
            f"Make ONE tool call at a time. After each tool result, validate it and "
            f"print what you see before proceeding. Never call tools in parallel."
        )
    )

    initial_state: dict[str, Any] = {
        "messages": [initial_message],
        "review_mode": mode,
        "schematic_pdf_path": schematic_pdf,
        "wiring_diagram_pdf_path": wiring_diagram_pdf,
        "schematic_pages": "",
        "wiring_diagram_pages": "",
        "schematic_drawing_index": "",
        "wiring_diagram_drawing_index": "",
        "schematic_bom_csv": "",
        "wiring_diagram_bom_csv": "",
        "comparison_report": "",
        "error": "",
        "current_step": "start",
        "retry_count": 0,
    }

    if verbose:
        _print_banner("IEM DRAWING REVIEW AGENT")
        _print_step("📄", "Schematic PDF:", schematic_pdf, BLUE)
        _print_step("📄", "Wiring Diagram PDF:", wiring_diagram_pdf, BLUE)
        _print_step("⚙️", "Mode:", mode_labels[mode], CYAN)
        _print_banner("STARTING WORKFLOW", YELLOW)

    step_counter = 0
    tool_call_counter = 0
    final_state = None
    t_start = time.time()

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            step_counter += 1
            elapsed = time.time() - t_start

            if node_name == "agent" and node_output.get("messages"):
                last_msg = node_output["messages"][-1]

                if isinstance(last_msg, AIMessage):
                    phase = _detect_phase(last_msg.content) if last_msg.content else ""

                    if job_id and phase:
                        phase_key = phase.split(":")[0].replace("PHASE ", "").strip()
                        job_tracker.update_phase(job_id, phase_key, step_counter)
                        job_tracker.add_log(job_id, {
                            "type": "phase",
                            "phase": phase,
                            "step": step_counter,
                            "elapsed": round(elapsed, 1),
                        })
                    elif job_id:
                        job_tracker.update_phase(job_id, "", step_counter)

                    usage = getattr(last_msg, "usage_metadata", None)
                    if job_id and usage:
                        job_tracker.add_tokens(
                            job_id,
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                        )

                    if job_id and last_msg.content:
                        summary = last_msg.content[:200].split("\n")[0]
                        job_tracker.add_log(job_id, {
                            "type": "reasoning",
                            "summary": summary,
                            "step": step_counter,
                            "elapsed": round(elapsed, 1),
                        })

                    if verbose and last_msg.content:
                        phase_label = f" [{phase}]" if phase else ""
                        _print_banner(
                            f"AGENT{phase_label}  (step {step_counter}, {elapsed:.1f}s)",
                            BLUE,
                        )
                        print(f"{BLUE}{last_msg.content}{RESET}\n", file=sys.stderr)

                    if last_msg.tool_calls:
                        tool_call_counter += len(last_msg.tool_calls)
                        if job_id:
                            for tc in last_msg.tool_calls:
                                job_tracker.add_tool_call(job_id)
                                job_tracker.add_log(job_id, {
                                    "type": "tool_call",
                                    "tool": tc["name"],
                                    "step": step_counter,
                                    "elapsed": round(elapsed, 1),
                                })
                        if verbose:
                            _print_step(
                                "🔧",
                                f"EXECUTING TOOL CALL (#{tool_call_counter}):",
                                color=MAGENTA,
                            )
                            for tc in last_msg.tool_calls:
                                print(
                                    f"  {MAGENTA}{BOLD}→ {tc['name']}{RESET}",
                                    file=sys.stderr,
                                )
                                print(
                                    f"{DIM}{_format_tool_args(tc['args'])}{RESET}",
                                    file=sys.stderr,
                                )
                            print(file=sys.stderr)
                    elif not last_msg.tool_calls:
                        final_state = node_output

            elif node_name == "tools":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, ToolMessage):
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        ok = "error" not in content.lower()
                        if job_id:
                            job_tracker.add_log(job_id, {
                                "type": "tool_result",
                                "tool": msg.name,
                                "success": ok,
                                "preview": content[:150],
                                "step": step_counter,
                                "elapsed": round(elapsed, 1),
                            })
                        if verbose:
                            status = "✅" if ok else "❌"
                            if step_counter == 1 or True:
                                _print_banner(
                                    f"TOOL RESULT  (step {step_counter}, {elapsed:.1f}s)", GREEN
                                )
                            _print_step(
                                status,
                                f"Tool: {msg.name}",
                                _truncate(content),
                                GREEN if ok else RED,
                            )
                            print(file=sys.stderr)

            elif node_name == "check_errors":
                err = node_output.get("error", "")
                if err:
                    if job_id:
                        job_tracker.add_log(job_id, {
                            "type": "error",
                            "message": err[:200],
                            "step": step_counter,
                            "elapsed": round(elapsed, 1),
                        })
                    if verbose:
                        _print_step("⚠️", "ERROR DETECTED:", _truncate(err, 300), RED)

            elif node_name == "self_correct":
                if job_id:
                    job_tracker.add_log(job_id, {
                        "type": "self_correct",
                        "retry": node_output.get("retry_count", 0),
                        "step": step_counter,
                        "elapsed": round(elapsed, 1),
                    })
                if verbose:
                    _print_banner(
                        f"SELF-CORRECTION  (step {step_counter}, {elapsed:.1f}s)", YELLOW
                    )
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, HumanMessage):
                            print(
                                f"{YELLOW}{_truncate(msg.content, 400)}{RESET}\n",
                                file=sys.stderr,
                            )
                    retry = node_output.get("retry_count", "?")
                    _print_step("🔄", f"Retry attempt: {retry}/{MAX_RETRIES}", color=YELLOW)

            logger.info("Node '%s' completed (step %d).", node_name, step_counter)

    elapsed_total = time.time() - t_start
    if verbose:
        _print_banner(
            f"WORKFLOW COMPLETE  ({step_counter} steps, {tool_call_counter} tool calls, {elapsed_total:.1f}s)",
            GREEN,
        )

    if final_state and final_state.get("messages"):
        return final_state["messages"][-1].content

    return "Workflow completed but no final summary was produced."
