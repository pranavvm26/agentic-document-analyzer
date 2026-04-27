"""
MCP Server for IEM Drawing Document Processing.

Exposes tools for PDF-to-image conversion, OCR via a local SGLang model,
BOM table extraction, and drawing index parsing. Designed to be consumed
by a LangGraph agent that orchestrates the full BOM comparison workflow.

Usage:
    python -m mcp_server.server          # stdio transport (default)
    python -m mcp_server.server --sse    # SSE transport on port 8000
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)

mcp = FastMCP("iem-drawing-tools")

OCR_BASE_URL = os.environ.get("OCR_BASE_URL", "http://localhost:8080/v1")
OCR_MODEL = os.environ.get("OCR_MODEL", "zai-org/GLM-OCR")

# Crop coordinates for BOM table slices (left half, right half) at 200 DPI.
# These are empirically determined from the IEM drawing template.
BOM_LEFT_CROP = {"left": 277, "top": 147, "right": 1488, "bottom": 2041}
BOM_RIGHT_CROP = {"left": 1558, "top": 135, "right": 2911, "bottom": 2052}

# Crop coordinates for the central content area (removes border/title block).
# At 200 DPI:
CENTRAL_CROP = {"left": 96, "top": 83, "right": 2983, "bottom": 2088}
# At 300 DPI (scaled 1.5x from 200 DPI values):
CENTRAL_CROP_300 = {"left": 144, "top": 125, "right": 4475, "bottom": 3132}


def _extract_table(html: str) -> pd.DataFrame:
    """Extract the first HTML table from OCR output into a DataFrame.

    The local OCR model returns HTML ``<table>`` markup. This function
    wraps it in a StringIO buffer so pandas treats it as in-memory
    content (not a file path), then parses and returns the first table.

    Args:
        html: Raw HTML string containing a ``<table>`` element.

    Returns:
        DataFrame of the first table found.

    Raises:
        ValueError: If no tables are found in the HTML.
    """
    tables = pd.read_html(StringIO(html))
    return tables[0]


def _crop_and_resize(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
    resize_pct: float = 1.0,
) -> Image.Image:
    """Crop a PIL image to the given pixel box and optionally resize.

    Args:
        image: Source PIL image.
        left: Left x pixel coordinate.
        top: Top y pixel coordinate.
        right: Right x pixel coordinate.
        bottom: Bottom y pixel coordinate.
        resize_pct: Scale factor applied after cropping (1.0 = no resize).

    Returns:
        Cropped (and optionally resized) PIL image.
    """
    cropped = image.crop((left, top, right, bottom))
    if resize_pct != 1.0:
        new_w = int(cropped.width * resize_pct)
        new_h = int(cropped.height * resize_pct)
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
    return cropped


def _run_ocr(
    image: Image.Image,
    prompt: str = "Text Recognition:",
    max_tokens: int = 8192,
) -> str:
    """Send a PIL image to the local SGLang OCR server and return the text.

    Saves the image to a temporary file and sends the file URI to the
    OCR server, avoiding base64 encoding overhead and maintaining the
    highest image fidelity.

    Args:
        image: PIL image to process.
        prompt: Instruction prompt for the vision model.
        max_tokens: Maximum generation length.

    Returns:
        Raw text output from the OCR model.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="ocr_input_", delete=False)
    image.save(tmp.name, format="PNG")

    client = OpenAI(base_url=OCR_BASE_URL, api_key="unused")
    response = client.chat.completions.create(
        model=OCR_MODEL,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{tmp.name}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.choices[0].message.content


@mcp.tool()
def pdf_to_images(pdf_path: str, output_dir: str = "", dpi: int = 300) -> str:
    """Convert a PDF into per-page PNG images.

    Splits every page of the PDF into a separate PNG rendered at the
    requested DPI. Images are written to a temporary directory unless
    an explicit output_dir is provided. Returns a JSON object mapping
    page keys (``page_001``, ``page_002``, …) to their file paths on
    disk, plus a ``_output_dir`` key with the directory used.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where page images will be saved.
                    If empty, a temporary directory is created automatically.
        dpi: Rendering resolution (default 200).

    Returns:
        JSON string of ``{page_key: image_path, _output_dir: dir}`` pairs.
    """
    pdf_path = Path(pdf_path)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix=f"iem_{pdf_path.stem}_"))

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: dict[str, str] = {"_output_dir": str(out_dir)}
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            key = f"page_{idx:03d}"
            out_path = out_dir / f"{pdf_path.stem}_{key}.png"
            pix.save(str(out_path))
            image_paths[key] = str(out_path)

    return json.dumps(image_paths)


@mcp.tool()
def ocr_page_text(image_path: str) -> str:
    """Run full-page OCR on a drawing page image.

    Crops the central content area (removing the border and title block)
    then sends the image to the local OCR model for text recognition.

    Args:
        image_path: Path to the page PNG image.

    Returns:
        Recognised text content of the page.
    """
    img = Image.open(image_path)
    cropped = _crop_and_resize(img, resize_pct=0.8, **CENTRAL_CROP)
    return _run_ocr(cropped, prompt="Text Recognition:", max_tokens=64000)


@mcp.tool()
def parse_drawing_index(ocr_text: str) -> str:
    """Extract the DRAWING INDEX table from OCR text of the legend page.

    Parses the OCR output to find the ``DRAWING INDEX:`` section and
    extracts entries using multiple regex strategies to handle varying
    OCR formatting. Returns a JSON object with all entries and a
    filtered list of BOM pages.

    Args:
        ocr_text: Raw OCR text from the legend / first page.

    Returns:
        JSON string with ``{"entries": [...], "bom_pages": [...]}``.
    """
    idx_match = re.search(r"DRAWING INDEX[:\s]*", ocr_text, re.IGNORECASE)
    if not idx_match:
        return json.dumps({"error": "DRAWING INDEX section not found in OCR text."})

    block = ocr_text[idx_match.end():]
    entries: list[dict[str, Any]] = []
    bom_pages: list[dict[str, Any]] = []

    patterns = [
        re.compile(
            r"(\d+)\s+(\d+_[\w_]+)\s+(.+?)\s+([A-Z]{1,4})\s+(\d+)"
        ),
        re.compile(
            r"(\d+)\s+([\w_]+)\s+(.+?)\s+(\w+)\s+(\d+)\s*$",
            re.MULTILINE,
        ),
        re.compile(
            r"(\d+)\s+([\w_]+-[\w_]+|[\w_]+)\s+((?:(?!\s[A-Z]{1,4}\s+\d+\s).)+)\s+([A-Z]{1,4})\s+(\d+)"
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(block):
            page_num = int(match.group(1))
            if any(e["page"] == page_num for e in entries):
                continue
            entry = {
                "page": page_num,
                "sheet": match.group(2).strip(),
                "description": match.group(3).strip(),
                "initials": match.group(4).strip(),
                "last_rev": int(match.group(5)),
            }
            entries.append(entry)

    entries.sort(key=lambda e: e["page"])
    bom_pages = [e for e in entries if "_BOM_" in e["sheet"].upper()]

    if not entries:
        bom_fallback = re.findall(
            r"(\d+)\s+\S*_BOM_\S*\s+BILL\s+OF\s+MATERIAL",
            block,
            re.IGNORECASE,
        )
        for page_str in bom_fallback:
            bom_pages.append({
                "page": int(page_str),
                "sheet": "BOM",
                "description": "BILL OF MATERIAL",
                "initials": "",
                "last_rev": 0,
            })

    return json.dumps({"entries": entries, "bom_pages": bom_pages}, indent=2)


@mcp.tool()
def read_drawing_index_from_image(
    page_image_path: str,
) -> str:
    """Read the Drawing Index directly from a page image using Bedrock Claude vision.

    Sends the page image to Claude and asks it to find and read the
    DRAWING INDEX table. Returns structured JSON with all entries and
    identified BOM pages. This is more reliable than OCR + regex parsing.

    Args:
        page_image_path: Path to the page image (uncropped or cropped).

    Returns:
        JSON with entries list and bom_pages list.
    """
    if not os.path.exists(page_image_path):
        return json.dumps({"error": "Image not found", "entries": [], "bom_pages": []})

    with open(page_image_path, "rb") as f:
        img_bytes = f.read()

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            "Look at this electrical drawing page. Find the DRAWING INDEX table.\n"
            "It lists all pages in the drawing package with columns like:\n"
            "PAGE, SHEET, DESCRIPTION, INT., LAST REV ON SHEET\n\n"
            "Extract ALL entries from the Drawing Index.\n"
            "Identify which entries are BOM pages (description contains\n"
            "'BILL OF MATERIAL' or sheet name contains '_BOM_').\n"
            "Also identify 3L pages (sheet contains '_3L_').\n\n"
            "Return a JSON object:\n"
            '{"entries": [{"page": 1, "sheet": "11_LEG_01", "description": "LEGEND"},\n'
            '             {"page": 5, "sheet": "11_BOM_01", "description": "BILL OF MATERIAL"}],\n'
            ' "bom_pages": [{"page": 5, "sheet": "11_BOM_01", "description": "BILL OF MATERIAL"}],\n'
            ' "three_line_pages": [{"page": 9, "sheet": "11_3L_01", "description": "THREE LINE DIAGRAM"}]}\n\n'
            "If no Drawing Index is found on this page, return:\n"
            '{"error": "No Drawing Index found", "entries": [], "bom_pages": [], "three_line_pages": []}\n'
            "Return ONLY the JSON object."
        )},
        {"image": {"format": "png", "source": {"bytes": img_bytes}}},
    ]

    result = _call_bedrock(content_blocks, max_tokens=8192)

    try:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    data.setdefault("entries", [])
    data.setdefault("bom_pages", [])
    data.setdefault("three_line_pages", [])
    return json.dumps(data, indent=2)


@mcp.tool()
def extract_bom_table_from_page(image_path: str) -> str:
    """Extract the Bill of Material table from a single BOM page image.

    Each BOM page contains two side-by-side tables (left half and right
    half). This tool crops each half, runs OCR with a table-recognition
    prompt, parses the resulting HTML tables into DataFrames, concatenates
    them, and saves the result to a temporary CSV file.

    The response contains only metadata and a preview — the full CSV is
    written to disk to avoid bloating the LLM conversation context.
    Pass the returned ``csv_path`` to ``concat_bom_csvs`` later.

    Args:
        image_path: Path to the BOM page PNG image.

    Returns:
        JSON object with ``csv_path`` (path to saved CSV file),
        ``csv_preview`` (first few rows), ``row_count``, ``columns``,
        and ``source`` (the input image path).
    """
    img = Image.open(image_path)
    frames: list[pd.DataFrame] = []
    slice_image_paths: list[str] = []

    for label, crop_box in [("left", BOM_LEFT_CROP), ("right", BOM_RIGHT_CROP)]:
        cropped = _crop_and_resize(img, resize_pct=0.99, **crop_box)

        slice_file = tempfile.NamedTemporaryFile(
            suffix=f"_{label}.png", prefix="bom_slice_", delete=False
        )
        cropped.save(slice_file.name, format="PNG")
        slice_image_paths.append(slice_file.name)

        html = _run_ocr(cropped, prompt="Table Recognition:", max_tokens=32000)
        try:
            df = _extract_table(html)
            frames.append(df)
        except ValueError:
            pass

    if not frames:
        return json.dumps({
            "csv_path": "",
            "csv_preview": "",
            "row_count": 0,
            "columns": [],
            "source": image_path,
            "slice_images": slice_image_paths,
            "error": "OCR returned no parseable HTML tables for this page.",
        })

    page_df = pd.concat(frames, ignore_index=True).dropna(how="all")
    full_csv = page_df.to_csv(index=False)

    csv_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="bom_page_", delete=False
    )
    csv_file.write(full_csv)
    csv_file.close()

    preview_lines = full_csv.splitlines()
    if len(preview_lines) > 6:
        preview = "\n".join(preview_lines[:6]) + f"\n... ({len(preview_lines) - 6} more rows)"
    else:
        preview = full_csv

    return json.dumps({
        "csv_path": csv_file.name,
        "csv_preview": preview,
        "row_count": len(page_df),
        "columns": list(page_df.columns),
        "source": image_path,
        "slice_images": slice_image_paths,
    })


@mcp.tool()
def concat_bom_csvs(csv_paths_json: str) -> str:
    """Concatenate multiple BOM CSV files into one unified table.

    Accepts a JSON list of file paths (from ``extract_bom_table_from_page``
    ``csv_path`` values), reads each CSV into a DataFrame, merges them,
    drops fully-empty rows, and returns the result as a Markdown table
    for LLM consumption.

    Args:
        csv_paths_json: JSON-encoded list of CSV file path strings.

    Returns:
        JSON object with ``markdown`` (Markdown table), ``row_count``,
        and ``columns``.
    """
    paths: list[str] = json.loads(csv_paths_json)
    frames: list[pd.DataFrame] = []
    for csv_path in paths:
        if not csv_path:
            continue
        try:
            df = pd.read_csv(csv_path)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return json.dumps({
            "markdown": "",
            "row_count": 0,
            "columns": [],
            "error": "No valid BOM data could be parsed from any page.",
        })

    combined = pd.concat(frames, ignore_index=True).dropna(how="all")
    full_markdown = combined.to_markdown(index=False)

    md_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="bom_merged_", delete=False
    )
    md_file.write(full_markdown)
    md_file.close()

    preview_lines = full_markdown.splitlines()
    if len(preview_lines) > 12:
        preview = "\n".join(preview_lines[:12]) + f"\n... ({len(preview_lines) - 12} more rows)"
    else:
        preview = full_markdown

    return json.dumps({
        "markdown_preview": preview,
        "markdown_path": md_file.name,
        "row_count": len(combined),
        "columns": list(combined.columns),
    })


def _image_to_b64(path: str) -> str:
    """Read an image file and return its base64-encoded PNG string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_bedrock(
    content_blocks: list[dict[str, Any]],
    max_tokens: int = 16384,
) -> str:
    """Call Bedrock Converse API with retry and exponential backoff.

    Args:
        content_blocks: List of content blocks for the user message.
        max_tokens: Maximum tokens to generate.

    Returns:
        The model's text response, or a JSON error string.
    """
    import boto3

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=boto3.session.Config(
            read_timeout=300,
            connect_timeout=30,
            retries={"max_attempts": 0},
        ),
    )

    max_retries = 4
    base_delay = 5
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId="us.anthropic.claude-sonnet-4-6",
                messages=[{"role": "user", "content": content_blocks}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                import time as _time
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Bedrock call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, max_retries, exc, delay,
                )
                _time.sleep(delay)

    return json.dumps({
        "error": f"Bedrock call failed after {max_retries} attempts: {last_error}",
    })


@mcp.tool()
def compare_boms_with_vision(
    schematic_bom_pages_json: str,
    wd_bom_pages_json: str,
) -> str:
    """Compare two BOMs in a 2-step process: table diff then image verification.

    Step 1 (Table Comparison): Sends both OCR-extracted Markdown tables to
    Claude to find candidate differences. This is fast and text-only.

    Step 2 (Image Verification): For each candidate issue from Step 1,
    sends the relevant BOM page images to Claude to visually confirm
    whether the difference is real or just OCR noise.

    Each bom_pages entry is a JSON list of objects with:
    - ``csv_path``: path to the per-page CSV file
    - ``slice_images``: list of image paths [left_half, right_half]
    - ``label``: page label like "BOM Page 1"

    Args:
        schematic_bom_pages_json: JSON list of per-page BOM data for schematic.
        wd_bom_pages_json: JSON list of per-page BOM data for WD.

    Returns:
        JSON string with the structured comparison report.
    """
    sch_pages: list[dict[str, Any]] = json.loads(schematic_bom_pages_json)
    wd_pages: list[dict[str, Any]] = json.loads(wd_bom_pages_json)

    sch_frames = []
    for page in sch_pages:
        if page.get("csv_path") and os.path.exists(page["csv_path"]):
            sch_frames.append(pd.read_csv(page["csv_path"]))
    wd_frames = []
    for page in wd_pages:
        if page.get("csv_path") and os.path.exists(page["csv_path"]):
            wd_frames.append(pd.read_csv(page["csv_path"]))

    sch_md = pd.concat(sch_frames, ignore_index=True).dropna(how="all").to_markdown(index=False) if sch_frames else ""
    wd_md = pd.concat(wd_frames, ignore_index=True).dropna(how="all").to_markdown(index=False) if wd_frames else ""

    logger.info("Step 1: Table-based comparison (text only)")
    step1_blocks: list[dict[str, Any]] = [
        {"text": (
            "STEP 1: TABLE-BASED BOM COMPARISON\n\n"
            "Compare these two BOM tables. The SCHEMATIC is the source of truth.\n"
            "The WIRING DIAGRAM (WD) is the edited copy.\n\n"
            "FOCUS ONLY ON THESE COLUMNS: SEQ, QTY, ITEM, MANUFACTURER.\n"
            "Ignore the DESCRIPTION column entirely — description differences\n"
            "are nuisance items and will be handled separately.\n\n"
            "Find candidate differences:\n"
            "- Rows present in one table but missing from the other\n"
            "- QTY differences for the same SEQ\n"
            "- ITEM part number differences for the same SEQ\n"
            "- MANUFACTURER differences for the same SEQ\n\n"
            "IMPORTANT: This is a preliminary scan. Some differences may be OCR\n"
            "extraction noise. Flag everything — we verify with images next.\n\n"
            "Also separately list any DESCRIPTION differences as nuisance items.\n\n"
            "Return a JSON object:\n"
            '{"candidates": [{"seq": N, "column": "...", '
            '"schematic_value": "...", "wd_value": "...", '
            '"suspected_type": "missing_row|value_diff", '
            '"note": "..."}], '
            '"nuisance": [{"seq": N, "schematic_desc_snippet": "...", '
            '"wd_desc_snippet": "...", "note": "..."}], '
            '"schematic_rows": N, "wd_rows": N}\n\n'
            f"══════ SCHEMATIC BOM ══════\n\n{sch_md}\n\n"
            f"══════ WIRING DIAGRAM BOM ══════\n\n{wd_md}\n\n"
            "Return ONLY the JSON object."
        )},
    ]

    step1_result = _call_bedrock(step1_blocks)
    logger.info("Step 1 complete. Parsing candidates.")

    try:
        step1_json_match = re.search(r"\{.*\}", step1_result, re.DOTALL)
        step1_data = json.loads(step1_json_match.group()) if step1_json_match else {}
    except (json.JSONDecodeError, AttributeError):
        step1_data = {}

    candidates = step1_data.get("candidates", [])
    nuisance = step1_data.get("nuisance", [])

    if not candidates:
        return json.dumps({
            "summary": {
                "schematic_rows": step1_data.get("schematic_rows", 0),
                "wd_rows": step1_data.get("wd_rows", 0),
                "matched": step1_data.get("schematic_rows", 0),
                "mismatched": 0,
                "missing_from_wd": 0,
                "extra_in_wd": 0,
            },
            "issues": [],
            "nuisance": nuisance,
            "assessment": (
                "No SEQ/QTY/ITEM/MANUFACTURER differences found. "
                "The WD BOM is consistent with the Schematic BOM."
            ),
        })

    logger.info("Step 2: Image verification for %d candidates", len(candidates))

    all_sch_images: list[str] = []
    all_wd_images: list[str] = []
    for page in sch_pages:
        all_sch_images.extend(page.get("slice_images", []))
    for page in wd_pages:
        all_wd_images.extend(page.get("slice_images", []))

    step2_blocks: list[dict[str, Any]] = [
        {"text": (
            "STEP 2: IMAGE VERIFICATION OF CANDIDATE DIFFERENCES\n\n"
            "Step 1 found candidate differences in SEQ, QTY, ITEM, or MANUFACTURER\n"
            "between a Schematic BOM and a Wiring Diagram BOM. Many of these are\n"
            "OCR extraction noise — NOT real errors.\n\n"
            "CRITICAL — FALSE POSITIVES ARE EXTREMELY COSTLY:\n"
            "Every false positive requires an engineer to manually review the item,\n"
            "costing significant time and money. Think carefully before flagging\n"
            "anything. If you cannot clearly see a difference in the images, it is\n"
            "NOT an issue. When in doubt, DISCARD.\n\n"
            "I am showing you the ACTUAL BOM page images from both documents.\n"
            "For each candidate, find the row in BOTH the schematic image AND the\n"
            "WD image. Read the value directly from each image.\n\n"
            "THE CRITICAL QUESTION FOR EACH CANDIDATE:\n"
            "Do the schematic image and WD image show THE SAME value for this cell?\n"
            "- If YES → DISCARD. It is OCR noise, NOT an issue.\n"
            "- If NO and you are CERTAIN → include as an issue.\n"
            "- If UNCLEAR / hard to read → DISCARD. Assume they match.\n\n"
            f"CANDIDATES TO VERIFY:\n{json.dumps(candidates, indent=2)}\n\n"
            "PRIORITY DEFINITIONS (only for CONFIRMED real differences):\n"
            "- HIGH: Entire row MISSING — present in one document, absent from other.\n"
            "- MEDIUM: Clear, unambiguous value mismatch — different quantity,\n"
            "  different vendor, different part number. Must be clearly readable.\n"
            "- LOW: Minor typo-level difference visible in images.\n"
            "- NUISANCE: Anything uncertain. When in doubt, use this.\n\n"
            "WHAT IS NOT AN ISSUE:\n"
            "- Both images show the same value but OCR read differently → DISCARD\n"
            "- Values you cannot clearly read in the image → DISCARD\n"
            "- Any difference where both images show the same value → DISCARD\n\n"
        )},
    ]

    step2_blocks.append({"text": "══════ SCHEMATIC BOM IMAGES ══════"})
    for i, page in enumerate(sch_pages):
        step2_blocks.append({"text": f"--- {page.get('label', f'Schematic Page {i+1}')} ---"})
        for img_path in page.get("slice_images", []):
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    step2_blocks.append({
                        "image": {"format": "png", "source": {"bytes": f.read()}},
                    })

    step2_blocks.append({"text": "\n══════ WIRING DIAGRAM BOM IMAGES ══════"})
    for i, page in enumerate(wd_pages):
        step2_blocks.append({"text": f"--- {page.get('label', f'WD Page {i+1}')} ---"})
        for img_path in page.get("slice_images", []):
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    step2_blocks.append({
                        "image": {"format": "png", "source": {"bytes": f.read()}},
                    })

    step2_blocks.append({"text": (
        "\n══════ VERIFICATION TASK ══════\n"
        "For each candidate:\n"
        "1. Find the row in the SCHEMATIC image. Read the value.\n"
        "2. Find the row in the WD image. Read the value.\n"
        "3. Are they the SAME? → DISCARD (OCR noise, not a real issue).\n"
        "4. Are they DIFFERENT? → Include as an issue with priority.\n\n"
        "Return a JSON object:\n"
        '{"summary": {"schematic_rows": N, "wd_rows": N, "matched": N, '
        '"mismatched": N, "missing_from_wd": N, "extra_in_wd": N}, '
        '"issues": [{"seq": N, "column": "...", "schematic_value": "...", '
        '"wd_value": "...", "priority": "HIGH|MEDIUM|LOW", '
        '"note": "what the images actually show"}], '
        '"discarded_ocr_noise": [{"seq": N, "column": "...", '
        '"reason": "both images show identical value: XYZ"}], '
        '"assessment": "1-2 sentence summary"}\n\n'
        "STRICT RULES:\n"
        "- If both images show the same value → MUST go in discarded_ocr_noise.\n"
        "- The issues list must ONLY contain differences visible in the images.\n"
        "- Do NOT include description differences anywhere in issues.\n"
        "- Assessment: 1-2 sentences, crisp, human-readable.\n"
        "Return ONLY the JSON object."
    )})

    step2_result = _call_bedrock(step2_blocks)
    logger.info("Step 2 complete.")

    try:
        step2_json_match = re.search(r"\{.*\}", step2_result, re.DOTALL)
        if step2_json_match:
            final_report = json.loads(step2_json_match.group())
            final_report["nuisance"] = nuisance
            return json.dumps(final_report, indent=2)
    except (json.JSONDecodeError, AttributeError):
        pass

    return step2_result


@mcp.tool()
def validate_page_is_bom(page_image_path: str) -> str:
    """Check if a page image is actually a Bill of Material page.

    Sends the uncropped page image to Bedrock Claude and asks it to
    confirm whether this page contains a BOM table. Returns true/false
    plus what the page actually contains.

    Args:
        page_image_path: Path to the uncropped page image.

    Returns:
        JSON with ``is_bom`` (bool), ``detected_type`` (str),
        and ``title_block_text`` (str).
    """
    if not os.path.exists(page_image_path):
        return json.dumps({"is_bom": False, "error": "Image not found"})

    with open(page_image_path, "rb") as f:
        img_bytes = f.read()

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            "Is this page a BILL OF MATERIAL (BOM) table?\n"
            "A BOM page has columns like SEQ, QTY, ITEM, MANUFACTURER, DESCRIPTION.\n"
            "Look at the page content and the title block at the bottom-right.\n\n"
            "Return a JSON object:\n"
            '{"is_bom": true, "detected_type": "BILL OF MATERIAL",\n'
            ' "title_block_text": "text from bottom-right title block"}\n'
            "Return ONLY the JSON object."
        )},
        {"image": {"format": "png", "source": {"bytes": img_bytes}}},
    ]

    result = _call_bedrock(content_blocks, max_tokens=1024)
    try:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    data.setdefault("is_bom", False)
    data.setdefault("detected_type", "UNKNOWN")
    return json.dumps(data, indent=2)


@mcp.tool()
def compare_bom_images_direct(
    schematic_bom_image_paths_json: str,
    wd_bom_image_paths_json: str,
) -> str:
    """Compare BOM pages by sending full page images directly to Bedrock Claude.

    No OCR, no slicing, no table extraction. Just sends the cropped BOM
    page images from both documents and asks Claude to compare them visually.

    Args:
        schematic_bom_image_paths_json: JSON list of cropped SCH BOM image paths.
        wd_bom_image_paths_json: JSON list of cropped WD BOM image paths.

    Returns:
        JSON comparison report with issues and assessment.
    """
    sch_paths: list[str] = json.loads(schematic_bom_image_paths_json)
    wd_paths: list[str] = json.loads(wd_bom_image_paths_json)

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            "BOM COMPARISON — DIRECT IMAGE ANALYSIS\n\n"
            "Compare the Bill of Material (BOM) pages between a SCHEMATIC\n"
            "(source of truth) and a WIRING DIAGRAM (WD, edited copy).\n\n"
            "I am sending you the full BOM page images from both documents.\n"
            "Read the tables directly from the images. Compare row by row.\n\n"
            "FOCUS ON: SEQ, QTY, ITEM, MANUFACTURER columns.\n"
            "DESCRIPTION differences are NUISANCE — list them separately.\n\n"
            "FALSE POSITIVES ARE EXTREMELY COSTLY. Only flag differences\n"
            "you can clearly read in both images. When in doubt, do not flag.\n\n"
            "PRIORITY:\n"
            "- HIGH: Entire row missing from one document\n"
            "- MEDIUM: Clear value mismatch in SEQ/QTY/ITEM/MANUFACTURER\n"
            "- LOW: Minor typo-level difference\n"
            "- Do NOT flag anything you cannot clearly read\n\n"
            "Return a JSON object:\n"
            '{"summary": {"schematic_rows": N, "wd_rows": N, "matched": N,\n'
            '  "mismatched": N, "missing_from_wd": N, "extra_in_wd": N},\n'
            ' "issues": [{"seq": N, "column": "...", "schematic_value": "...",\n'
            '   "wd_value": "...", "priority": "HIGH|MEDIUM|LOW",\n'
            '   "note": "..."}],\n'
            ' "nuisance": [{"seq": N, "schematic_desc_snippet": "...",\n'
            '   "wd_desc_snippet": "...", "note": "..."}],\n'
            ' "assessment": "1-2 sentence crisp summary"}\n\n'
        )},
    ]

    content_blocks.append({"text": "=== SCHEMATIC BOM PAGES (Source of Truth) ==="})
    for i, path in enumerate(sch_paths):
        if os.path.exists(path):
            content_blocks.append({"text": f"Schematic BOM Page {i + 1}:"})
            with open(path, "rb") as f:
                content_blocks.append({
                    "image": {"format": "png", "source": {"bytes": f.read()}},
                })

    content_blocks.append({"text": "\n=== WIRING DIAGRAM BOM PAGES (Edited Copy) ==="})
    for i, path in enumerate(wd_paths):
        if os.path.exists(path):
            content_blocks.append({"text": f"WD BOM Page {i + 1}:"})
            with open(path, "rb") as f:
                content_blocks.append({
                    "image": {"format": "png", "source": {"bytes": f.read()}},
                })

    content_blocks.append({"text": (
        "\nCompare the SCHEMATIC BOM pages against the WD BOM pages.\n"
        "Read directly from the images. Return ONLY the JSON object."
    )})

    return _call_bedrock(content_blocks, max_tokens=16384)


@mcp.tool()
def generate_html_report(
    comparison_json: str,
    schematic_bom_pages_json: str,
    wd_bom_pages_json: str,
    output_path: str = "bom_comparison_report.html",
) -> str:
    """Generate a clean HTML report for BOM comparison results.

    Deterministic image mapping from the page lists. Each BOM page
    shows its slice images side-by-side with the extracted table.
    """
    sch_pages: list[dict[str, Any]] = json.loads(schematic_bom_pages_json)
    wd_pages: list[dict[str, Any]] = json.loads(wd_bom_pages_json)
    try:
        report = json.loads(comparison_json)
    except json.JSONDecodeError:
        report = {"raw_text": comparison_json}

    issues_raw = report.get("issues", [])
    nuisance = report.get("nuisance", [])
    discarded = report.get("discarded_ocr_noise", [])
    summary = report.get("summary", {})
    assessment = report.get("assessment", report.get("raw_text", ""))

    issues = []
    for i in issues_raw:
        note = str(i.get("note", "")).lower()
        sch_val = str(i.get("schematic_value", "")).strip()
        wd_val = str(i.get("wd_value", "")).strip()
        if any(phrase in note for phrase in (
            "discard", "ocr noise", "no real diff", "no real mismatch",
            "matching wd", "matching wo", "no issue", "consistent",
            "confirmed to match", "images show the same",
        )):
            discarded.append({"seq": i.get("seq"), "column": i.get("column"), "reason": i.get("note", "")})
            continue
        if sch_val and wd_val and sch_val.upper().replace(" ", "") == wd_val.upper().replace(" ", ""):
            discarded.append({"seq": i.get("seq"), "column": i.get("column"), "reason": f"Values identical: {sch_val}"})
            continue
        issues.append(i)

    high_n = sum(1 for i in issues if i.get("priority", "").upper() == "HIGH")
    med_n = sum(1 for i in issues if i.get("priority", "").upper() == "MEDIUM")
    low_n = sum(1 for i in issues if i.get("priority", "").upper() in ("LOW", "NUISANCE"))
    disc_n = len(discarded)

    issue_rows = ""
    for i in issues:
        p = i.get("priority", "LOW").upper()
        co = {"HIGH": "#d32f2f", "MEDIUM": "#f57c00", "LOW": "#388e3c", "NUISANCE": "#9e9e9e"}.get(p, "#888")
        issue_rows += (
            f'<tr class="s{p[0].lower()}">'
            f'<td style="color:{co};font-weight:700">{p}</td>'
            f'<td>{i.get("seq", "")}</td>'
            f'<td>{i.get("column", "")}</td>'
            f'<td>{i.get("schematic_value", "")}</td>'
            f'<td>{i.get("wd_value", "")}</td>'
            f'<td>{i.get("note", "")}</td></tr>'
        )

    nui_rows = ""
    for n in nuisance:
        nui_rows += (
            f'<tr><td>{n.get("seq", "")}</td>'
            f'<td>{str(n.get("schematic_desc_snippet", ""))[:80]}</td>'
            f'<td>{str(n.get("wd_desc_snippet", ""))[:80]}</td>'
            f'<td>{n.get("note", "")}</td></tr>'
        )

    def _bom_section(pages: list[dict[str, Any]], doc: str) -> str:
        parts: list[str] = []
        for pg in pages:
            lbl = pg.get("label", "BOM Page")
            parts.append(f'<div class="bom-page"><h4>{doc}: {lbl}</h4>')

            cropped = pg.get("cropped_path", "")
            if cropped and os.path.exists(cropped):
                parts.append(f'<img src="data:image/png;base64,{_image_to_b64(cropped)}" class="slice-img" style="max-width:100%">')

            slices = pg.get("slice_images", [])
            if slices:
                parts.append('<div class="slice-row">')
                for ip in slices:
                    if os.path.exists(ip):
                        parts.append(f'<img src="data:image/png;base64,{_image_to_b64(ip)}" class="slice-img">')
                parts.append('</div>')

            cp = pg.get("csv_path", "")
            if cp and os.path.exists(cp):
                df = pd.read_csv(cp)
                parts.append(_markdown_to_html_table(df.to_markdown(index=False)))
            parts.append('</div>')
        return "\n".join(parts)

    sch_html = _bom_section(sch_pages, "Schematic")
    wd_html = _bom_section(wd_pages, "Wiring Diagram")

    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>IEM BOM Comparison Report</title><style>
*{{box-sizing:border-box}}body{{font-family:-apple-system,'Segoe UI',Arial,sans-serif;margin:0;padding:24px;color:#222;background:#f5f6fa}}
h1{{color:#1a5276;margin-bottom:4px}}h1+p{{color:#666;margin-top:0}}h2{{color:#2c3e50;margin-top:32px}}h3{{color:#34495e}}h4{{color:#555;margin:12px 0 8px}}
.sb{{display:flex;gap:12px;margin:20px 0}}.sc{{background:#fff;border-radius:8px;padding:16px 20px;text-align:center;flex:1;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.sc .n{{font-size:32px;font-weight:700}}.sc .l{{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.5px}}
.nh{{color:#d32f2f}}.nm{{color:#f57c00}}.no{{color:#388e3c}}
.as{{background:#e8f8f5;border:1px solid #1abc9c;border-radius:8px;padding:16px;margin:16px 0;font-size:15px}}
.card{{background:#fff;border-radius:10px;padding:24px;margin:20px 0;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
th{{background:#34495e;color:#fff;padding:8px 10px;text-align:left}}
td{{border:1px solid #e0e0e0;padding:8px 10px;vertical-align:top}}
tr:nth-child(even){{background:#fafafa}}tr.sh{{background:#fff5f5}}tr.sm{{background:#fff8e1}}tr.sn td{{color:#999}}
.doc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.bom-page{{margin:16px 0;padding:16px;background:#f8f9fa;border-radius:6px;border:1px solid #eee}}
.slice-row{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}.slice-img{{max-width:48%;border:1px solid #ddd;border-radius:4px}}
.nui{{background:#fafafa;border:1px solid #ddd;border-radius:8px;padding:16px;margin:20px 0}}
.disc{{background:#f0f0f0;border-radius:8px;padding:12px;margin:16px 0;font-size:12px;color:#888}}
</style></head><body>
<h1>IEM BOM Comparison Report</h1>
<p>Schematic vs Wiring Diagram \u2014 Bill of Material Review</p>

<div class="sb">
<div class="sc"><div class="n">{summary.get("schematic_rows","?")}</div><div class="l">SCH Rows</div></div>
<div class="sc"><div class="n">{summary.get("wd_rows","?")}</div><div class="l">WD Rows</div></div>
<div class="sc"><div class="n {"nh" if high_n else "no"}">{high_n}</div><div class="l">High</div></div>
<div class="sc"><div class="n {"nm" if med_n else "no"}">{med_n}</div><div class="l">Medium</div></div>
<div class="sc"><div class="n">{low_n}</div><div class="l">Low</div></div>
<div class="sc"><div class="n">{disc_n}</div><div class="l">Discarded (OCR)</div></div>
</div>

<div class="as"><strong>Assessment:</strong> {assessment}</div>

<div class="card">
<h2>Issues (SEQ / QTY / ITEM / MANUFACTURER)</h2>
<p style="color:#666;font-size:13px">Only confirmed differences visible in the source images. Description differences are listed separately as nuisance items.</p>
<table>
<thead><tr><th style="width:80px">Priority</th><th>SEQ</th><th>Column</th><th>SCH Value</th><th>WD Value</th><th>Note</th></tr></thead>
<tbody>{issue_rows if issue_rows else '<tr><td colspan="6">No issues found \u2014 BOMs are consistent.</td></tr>'}</tbody>
</table>
</div>

<div class="card">
<h2>BOM Source Images &amp; Tables</h2>
<div class="doc-grid">
<div><h3>Schematic (Source of Truth)</h3>{sch_html}</div>
<div><h3>Wiring Diagram (Edited Copy)</h3>{wd_html}</div>
</div>
</div>

<div class="nui">
<h2>Nuisance Items (Description Differences)</h2>
<p style="color:#666;font-size:13px">DESCRIPTION-only differences. Do not affect procurement. Listed for reference.</p>
<table>
<thead><tr><th>SEQ</th><th>SCH Description</th><th>WD Description</th><th>Note</th></tr></thead>
<tbody>{nui_rows if nui_rows else '<tr><td colspan="4">No description differences.</td></tr>'}</tbody>
</table>
</div>

{f'<div class="disc"><strong>Discarded OCR Noise ({disc_n} items):</strong> These candidates were confirmed as OCR extraction artifacts after image verification.</div>' if disc_n else ''}

</body></html>'''

    with open(output_path, "w") as f:
        f.write(html)
    return json.dumps({"output_path": os.path.abspath(output_path)})


def _markdown_to_html_table(md: str) -> str:
    """Convert a pandas-generated Markdown table to an HTML table string.

    Args:
        md: Markdown table string (pipe-delimited).

    Returns:
        HTML ``<table>`` string.
    """
    lines = [l.strip() for l in md.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return f"<pre>{md}</pre>"

    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    html = "<table><thead><tr>"
    for cell in header_cells:
        html += f"<th>{cell}</th>"
    html += "</tr></thead><tbody>"

    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        html += "<tr>"
        for cell in cells:
            html += f"<td>{cell}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    return html


@mcp.tool()
def extract_diagram_pages(
    pages_json: str,
    page_numbers: str,
    diagram_type: str = "3L",
    dpi: int = 200,
) -> str:
    """Crop and cache diagram pages from a PDF's page images.

    Takes the page image map (from pdf_to_images) and a list of page
    numbers, crops each to the central content area (removing borders
    and title block), and saves the cropped images to temp files.

    Uses crop coordinates scaled to the DPI of the source images:
    200 DPI → CENTRAL_CROP, 300 DPI → CENTRAL_CROP_300.

    Args:
        pages_json: JSON string of the page image map from pdf_to_images.
        page_numbers: JSON list of page numbers (ints) to extract.
        diagram_type: Label for the diagram type (e.g. "3L", "CB", "SCH").
        dpi: DPI of the source page images (200 or 300). Determines which
             crop coordinates to use.

    Returns:
        JSON list of objects with ``page_number``, ``original_path``,
        ``cropped_path``, and ``label``.
    """
    pages: dict[str, str] = json.loads(pages_json)
    numbers: list[int] = json.loads(page_numbers)
    crop_box = CENTRAL_CROP_300 if dpi >= 300 else CENTRAL_CROP

    results: list[dict[str, Any]] = []
    for num in numbers:
        key = f"page_{num:03d}"
        original_path = pages.get(key, "")
        if not original_path or not os.path.exists(original_path):
            results.append({
                "page_number": num,
                "original_path": original_path,
                "cropped_path": "",
                "label": f"{diagram_type} Page {num}",
                "error": f"Page image not found for {key}",
            })
            continue

        img = Image.open(original_path)
        cropped = _crop_and_resize(img, **crop_box, resize_pct=1.0)

        crop_file = tempfile.NamedTemporaryFile(
            suffix=".png", prefix=f"diagram_{diagram_type}_{num}_", delete=False
        )
        cropped.save(crop_file.name, format="PNG")

        results.append({
            "page_number": num,
            "original_path": original_path,
            "cropped_path": crop_file.name,
            "label": f"{diagram_type} Page {num}",
        })

    return json.dumps(results, indent=2)


@mcp.tool()
def analyze_3l_diagram(cropped_image_path: str) -> str:
    """Analyze a single cropped diagram page using Bedrock Claude vision.

    Determines:
    1. Whether this is actually a Three Line Diagram (not a 1L, BOM, etc.)
    2. All circuit continuation references — the curly brace labels like
       ``STB-U/SEC.1 RIGHT PAN``, ``FDISC-PTU/SEC.1 BACKPAN``, and any
       ``CONTINUED ON/TO 11_CB_01`` type references.

    Args:
        cropped_image_path: Path to the cropped diagram image.

    Returns:
        JSON with ``is_3l_diagram`` (bool), ``detected_type`` (str),
        ``curly_brace_labels`` (list of component/location names from
        curly braces), ``continuation_sheets`` (list of sheet references
        like "11_CB_01"), and ``section_label`` (e.g. "SECTION 1").
    """
    if not os.path.exists(cropped_image_path):
        return json.dumps({"is_3l_diagram": False, "error": "Image not found"})

    with open(cropped_image_path, "rb") as f:
        img_bytes = f.read()

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            "Analyze this electrical drawing page.\n\n"
            "1. DIAGRAM TYPE: Is this a Three Line Diagram (3L)? A 3L diagram\n"
            "   shows three-phase power distribution with bus bars, breakers,\n"
            "   and connections. It is NOT a one-line diagram, BOM, nameplate,\n"
            "   front view, or schematic.\n\n"
            "2. CURLY BRACE LABELS: Find ALL text labels that appear next to\n"
            "   curly braces (}) in the diagram. These indicate circuit\n"
            "   continuations to other wiring pages. Examples:\n"
            "   - 'STB-U/SEC.1 RIGHT PAN'\n"
            "   - 'FDISC-PTU/SEC.1 BACKPAN'\n"
            "   - 'FDISC-7/SEC.3 BACKPAN'\n"
            "   List every single one you can find.\n\n"
            "3. CONTINUATION SHEETS: Find any 'CONTINUED ON ...', 'CONTINUED TO ...',\n"
            "   or 'CONTINUE TO ...' references that point to specific sheet numbers\n"
            "   (like 11_3L_02, 11_WD_01, 11_WD_02). Focus especially on _WD_ sheets\n"
            "   as these are the wiring diagram detail pages.\n\n"
            "4. SECTION LABEL: What section label appears at the bottom of the page?\n"
            "   (e.g. 'SECTION 1', 'SECTION 2')\n\n"
            "Return a JSON object:\n"
            '{"is_3l_diagram": true, "detected_type": "THREE LINE DIAGRAM",\n'
            ' "curly_brace_labels": ["STB-U/SEC.1 RIGHT PAN", "FDISC-PTU/SEC.1 BACKPAN"],\n'
            ' "continuation_sheets": ["11_3L_02", "11_CB_01"],\n'
            ' "section_label": "SECTION 1"}\n\n'
            "Return ONLY the JSON object."
        )},
        {"image": {"format": "png", "source": {"bytes": img_bytes}}},
    ]

    result_text = _call_bedrock(content_blocks, max_tokens=4096)

    try:
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    data.setdefault("is_3l_diagram", False)
    data.setdefault("detected_type", "UNKNOWN")
    data.setdefault("curly_brace_labels", [])
    data.setdefault("continuation_sheets", [])
    data.setdefault("section_label", "")

    return json.dumps(data, indent=2)


@mcp.tool()
def validate_wd_page_label(
    page_image_path: str,
    expected_reference: str,
) -> str:
    """Check if a WD page's bottom-right label matches the expected reference.

    Sends the UNCROPPED page image to Bedrock Claude to read the title
    block label at the bottom-right corner. Compares it against the
    expected reference (e.g. "SEC.1 - LEFT/RIGHT PAN WIRING").

    If the label matches, the page is valid and should be cropped.
    If not, the page is wrong and should be skipped.

    Args:
        page_image_path: Path to the UNCROPPED page image.
        expected_reference: The expected title/description to match.

    Returns:
        JSON with ``matches`` (bool), ``detected_label`` (str),
        ``cropped_path`` (str, only if matches=true).
    """
    if not os.path.exists(page_image_path):
        return json.dumps({"matches": False, "error": "Image not found"})

    with open(page_image_path, "rb") as f:
        img_bytes = f.read()

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            "Look at the BOTTOM-RIGHT corner of this electrical drawing page.\n"
            "There is a title block with the sheet description/title.\n"
            "Read the title text and return it.\n\n"
            f"I expect it to contain something like: '{expected_reference}'\n\n"
            "Return a JSON object:\n"
            '{"detected_label": "the text you read from the title block",\n'
            ' "matches": true/false}\n'
            "Return ONLY the JSON object."
        )},
        {"image": {"format": "png", "source": {"bytes": img_bytes}}},
    ]

    result_text = _call_bedrock(content_blocks, max_tokens=1024)

    try:
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    data.setdefault("matches", False)
    data.setdefault("detected_label", "UNKNOWN")
    data["cropped_path"] = ""

    if data["matches"]:
        img = Image.open(page_image_path)
        use_300 = img.width > 3500
        crop_box = CENTRAL_CROP_300 if use_300 else CENTRAL_CROP
        cropped = _crop_and_resize(img, **crop_box, resize_pct=1.0)
        crop_file = tempfile.NamedTemporaryFile(
            suffix=".png", prefix="wd_validated_", delete=False
        )
        cropped.save(crop_file.name, format="PNG")
        data["cropped_path"] = crop_file.name

    return json.dumps(data, indent=2)


@mcp.tool()
def resolve_continuations(
    curly_brace_labels_json: str,
    continuation_sheets_json: str,
    drawing_index_json: str,
    document_pages_json: str,
) -> str:
    """Resolve continuation references to actual page images.

    Takes the curly brace labels and continuation sheet references from
    analyze_3l_diagram, searches the Drawing Index for matching pages,
    crops and caches them.

    For curly brace labels (e.g. "STB-U/SEC.1 RIGHT PAN"), searches
    the Drawing Index for WD (wiring diagram detail) pages whose
    description contains the component/section name.

    For continuation sheets (e.g. "11_WD_02"), does a direct sheet lookup.

    Args:
        curly_brace_labels_json: JSON list of curly brace label strings.
        continuation_sheets_json: JSON list of sheet reference strings.
        drawing_index_json: JSON string of Drawing Index entries.
        document_pages_json: JSON string of page image map from pdf_to_images.

    Returns:
        JSON list of resolved continuations, each with ``label``,
        ``matched_sheet``, ``page_number``, ``cropped_path``.
    """
    labels: list[str] = json.loads(curly_brace_labels_json)
    sheets: list[str] = json.loads(continuation_sheets_json)
    index_entries: list[dict[str, Any]] = json.loads(drawing_index_json)
    pages: dict[str, str] = json.loads(document_pages_json)

    resolved: list[dict[str, Any]] = []
    seen_pages: set[int] = set()

    for sheet_ref in sheets:
        for entry in index_entries:
            if entry.get("sheet", "").upper() == sheet_ref.upper():
                page_num = entry["page"]
                if page_num not in seen_pages:
                    cropped = _crop_page(page_num, pages)
                    resolved.append({
                        "label": f"Sheet {sheet_ref}",
                        "matched_sheet": entry["sheet"],
                        "matched_description": entry.get("description", ""),
                        "page_number": page_num,
                        "cropped_path": cropped,
                    })
                    seen_pages.add(page_num)
                break

    for label in labels:
        label_upper = label.upper()
        name_parts = label_upper.replace("/", " ").replace(".", " ").replace("-", " ").split()
        sec_match = re.search(r"SEC\.?\s*(\d+)", label_upper)
        section_num = sec_match.group(1) if sec_match else None

        for entry in index_entries:
            sheet = entry.get("sheet", "").upper()
            desc = entry.get("description", "").upper()

            if "_WD_" not in sheet:
                continue

            matched = False
            if section_num and f"SEC.{section_num}" in desc.replace(" ", ""):
                matched = True
            elif section_num and f"SEC {section_num}" in desc:
                matched = True
            elif any(part in desc for part in name_parts if len(part) > 3):
                matched = True

            if matched:
                page_num = entry["page"]
                if page_num not in seen_pages:
                    cropped = _crop_page(page_num, pages)
                    resolved.append({
                        "label": label,
                        "matched_sheet": entry["sheet"],
                        "matched_description": entry.get("description", ""),
                        "page_number": page_num,
                        "cropped_path": cropped,
                    })
                    seen_pages.add(page_num)

    return json.dumps(resolved, indent=2)


def _crop_page(page_num: int, pages: dict[str, str], dpi: int = 200) -> str:
    """Crop a page image by page number, trying -1 offset first.

    Args:
        page_num: Drawing Index page number.
        pages: Page image map from pdf_to_images.
        dpi: DPI of the source images (determines crop coordinates).

    Returns:
        Path to the cropped image, or empty string if not found.
    """
    crop_box = CENTRAL_CROP_300 if dpi >= 300 else CENTRAL_CROP
    for offset in [-1, 0]:
        key = f"page_{page_num + offset:03d}"
        img_path = pages.get(key, "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            cropped = _crop_and_resize(img, **crop_box, resize_pct=1.0)
            crop_file = tempfile.NamedTemporaryFile(
                suffix=".png", prefix=f"cont_p{page_num}_", delete=False
            )
            cropped.save(crop_file.name, format="PNG")
            return crop_file.name
    return ""


@mcp.tool()
def compare_single_3l_pair(
    schematic_3l_path: str,
    wd_3l_path: str,
    wd_continuation_paths_json: str,
    pair_label: str = "3L Diagram",
) -> str:
    """Compare one schematic 3L diagram against its WD counterpart + continuations.

    Sends one SCH 3L image, one WD 3L image, and all WD continuation
    images to Bedrock Claude for visual comparison. This is called once
    per 3L diagram pair.

    Args:
        schematic_3l_path: Path to the cropped schematic 3L diagram.
        wd_3l_path: Path to the cropped WD 3L diagram.
        wd_continuation_paths_json: JSON list of objects with "label" and
            "cropped_path" for each WD continuation page.
        pair_label: Human-readable label for this pair.

    Returns:
        JSON comparison report for this single pair.
    """
    continuations: list[dict[str, Any]] = json.loads(wd_continuation_paths_json)

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            f"CIRCUIT DIAGRAM COMPARISON: {pair_label}\n\n"
            "Compare the SCHEMATIC 3L diagram (source of truth) against the\n"
            "WIRING DIAGRAM 3L diagram and its WD detail pages.\n\n"
            "IMPORTANT — THIS IS NOT A 1:1 MAPPING:\n"
            "The WD has MORE detail than the schematic. The WD detail pages\n"
            "show wiring that is only summarized in the schematic. Extra\n"
            "information in the WD is EXPECTED and NORMAL.\n\n"
            "CRITICAL — FALSE POSITIVES ARE EXTREMELY COSTLY:\n"
            "Every false positive requires an engineer to manually review the\n"
            "flagged item, costing significant time and money. You MUST think\n"
            "carefully before flagging anything as MEDIUM or HIGH. If you are\n"
            "not confident a difference is a real circuit error, classify it\n"
            "as LOW or NUISANCE. Only flag something as MEDIUM/HIGH when you\n"
            "are certain it represents an actual circuit inconsistency that\n"
            "would cause incorrect fabrication or wiring.\n\n"
            "OCR AND PARSING LIMITATIONS:\n"
            "The images may have rendering artifacts, small text that is hard\n"
            "to read, or labels that appear slightly different due to image\n"
            "quality. These are NOT real errors. If a value looks like it\n"
            "COULD be the same but you cannot read it clearly, assume it\n"
            "matches and classify as NUISANCE at most.\n\n"
            "SEVERITY DEFINITIONS:\n"
            "- HIGH: A circuit or major component is completely MISSING from\n"
            "  the WD that exists in the SCH. Also: a wire number that appears\n"
            "  on two different circuits where it should be unique.\n"
            "- MEDIUM: A clear, unambiguous value mismatch that would cause\n"
            "  wrong fabrication — e.g. breaker frame size 4000AF vs 6000AF,\n"
            "  or bus rating 4000A vs 3000A. Also: duplicate wire numbers\n"
            "  used across different WD pages (same wire number on two pages\n"
            "  connecting to different circuits is an error).\n"
            "- LOW: Minor differences unlikely to cause fabrication errors.\n"
            "- NUISANCE: Observations, layout differences, uncertain items.\n\n"
            "WIRE NUMBER UNIQUENESS CHECK:\n"
            "Each wire number in the WD should be unique to its circuit.\n"
            "If you see the same wire number used on two different circuits\n"
            "or two different WD detail pages connecting to different points,\n"
            "flag it as MEDIUM or HIGH depending on severity.\n\n"
            "DO NOT FLAG:\n"
            "- Extra detail in WD not in SCH (normal — WD adds detail)\n"
            "- Different layout or arrangement (only content matters)\n"
            "- Label style differences (e.g. 'CB-U' vs 'CB-U(1)')\n"
            "- Values you cannot clearly read in the image\n\n"
            "Return a JSON object:\n"
            '{"issues": [{"description": "...", "severity": "HIGH|MEDIUM|LOW|NUISANCE",\n'
            '  "schematic_detail": "what SCH clearly shows",\n'
            '  "wd_detail": "what WD clearly shows or is missing",\n'
            '  "confidence": "HIGH|MEDIUM|LOW"}],\n'
            ' "summary": "1-2 sentence crisp assessment"}\n\n'
        )},
    ]

    content_blocks.append({"text": f"--- SCHEMATIC: {pair_label} ---"})
    if os.path.exists(schematic_3l_path):
        with open(schematic_3l_path, "rb") as f:
            content_blocks.append({
                "image": {"format": "png", "source": {"bytes": f.read()}},
            })

    content_blocks.append({"text": f"--- WIRING DIAGRAM: {pair_label} ---"})
    if os.path.exists(wd_3l_path):
        with open(wd_3l_path, "rb") as f:
            content_blocks.append({
                "image": {"format": "png", "source": {"bytes": f.read()}},
            })

    for cont in continuations:
        cont_path = cont.get("cropped_path", "")
        cont_label = cont.get("label", "continuation")
        if cont_path and os.path.exists(cont_path):
            content_blocks.append({"text": f"  ↳ WD Continuation: {cont_label}"})
            with open(cont_path, "rb") as f:
                content_blocks.append({
                    "image": {"format": "png", "source": {"bytes": f.read()}},
                })

    content_blocks.append({"text": (
        "Compare the schematic against the WD + continuations.\n"
        "Return ONLY the JSON object."
    )})

    return _call_bedrock(content_blocks, max_tokens=8192)


@mcp.tool()
def generate_circuit_html_report(
    circuit_comparison_json: str,
    schematic_diagrams_json: str,
    wd_diagrams_json: str,
    output_path: str = "circuit_comparison_report.html",
) -> str:
    """Generate a clean HTML report for circuit diagram comparison.

    Deterministic image mapping: pair N uses sch_list[N] and wd_list[N].
    Continuation images validated for existence before embedding.
    Returns debug log of all image paths and their status.
    """
    sch_list: list[dict[str, Any]] = json.loads(schematic_diagrams_json)
    wd_list: list[dict[str, Any]] = json.loads(wd_diagrams_json)
    try:
        report = json.loads(circuit_comparison_json)
    except json.JSONDecodeError:
        report = {}
    pairs = report.get("pairs", [])
    if not pairs:
        return json.dumps({"output_path": "", "error": "No 'pairs' in JSON."})

    high_n = sum(1 for p in pairs for i in p.get("issues", []) if i.get("severity", "").upper() == "HIGH")
    med_n = sum(1 for p in pairs for i in p.get("issues", []) if i.get("severity", "").upper() == "MEDIUM")
    low_n = sum(1 for p in pairs for i in p.get("issues", []) if i.get("severity", "").upper() in ("LOW", "NUISANCE"))

    debug: list[dict[str, Any]] = []
    sections: list[str] = []

    for idx, pair in enumerate(pairs):
        lbl = pair.get("pair_label", f"Pair {idx+1}")
        issues = pair.get("issues", [])
        conts = pair.get("wd_continuations", [])
        summ = pair.get("summary", "")

        sd = sch_list[idx] if idx < len(sch_list) else {}
        wd = wd_list[idx] if idx < len(wd_list) else {}
        sp, wp = sd.get("cropped_path", ""), wd.get("cropped_path", "")
        s_ok = bool(sp and os.path.exists(sp))
        w_ok = bool(wp and os.path.exists(wp))

        cd: list[dict] = []
        vc: list[tuple[str, str]] = []
        for c in conts:
            cp, cl = c.get("cropped_path", ""), c.get("label", "cont")
            ok = bool(cp and os.path.exists(cp))
            cd.append({"label": cl, "path": cp, "exists": ok})
            if ok:
                vc.append((cl, cp))

        debug.append({"pair": idx, "label": lbl, "sch": sp, "sch_ok": s_ok, "wd": wp, "wd_ok": w_ok, "conts": cd})

        si = f'<img src="data:image/png;base64,{_image_to_b64(sp)}" class="di">' if s_ok else '<div class="ni">No image</div>'
        wi = f'<img src="data:image/png;base64,{_image_to_b64(wp)}" class="di">' if w_ok else '<div class="ni">No image</div>'

        ch = ""
        if vc:
            ch = '<div class="cs"><h4>WD Detail Pages</h4>'
            for cl, cp in vc:
                ch += f'<div class="ci"><span class="cl">\u21b3 {cl}</span><img src="data:image/png;base64,{_image_to_b64(cp)}" class="di"></div>'
            ch += '</div>'

        has_high = any(i.get("severity", "").upper() == "HIGH" for i in issues)
        has_med = any(i.get("severity", "").upper() == "MEDIUM" for i in issues)
        badge = '<span class="bh">HIGH</span>' if has_high else '<span class="bm">MEDIUM</span>' if has_med else '<span class="bo">CLEAN</span>'

        ir = ""
        for i in issues:
            sv = i.get("severity", "LOW").upper()
            co = {"HIGH": "#d32f2f", "MEDIUM": "#f57c00", "LOW": "#388e3c", "NUISANCE": "#9e9e9e"}.get(sv, "#888")
            ir += f'<tr class="s{sv[0].lower()}"><td style="color:{co};font-weight:700">{sv}</td><td>{i.get("description", "")}</td><td>{i.get("schematic_detail", "")}</td><td>{i.get("wd_detail", "")}</td></tr>'

        sections.append(f'''<section class="ps">
<div class="ph"><h2>Pair {idx+1}: {lbl}</h2>{badge}</div>
{f'<p class="sm">{summ}</p>' if summ else ''}
<div class="dr"><div><h3>Schematic 3L</h3><p class="mt">Page {sd.get("page_number","?")} \u2014 {sd.get("label","")}</p>{si}</div>
<div><h3>Wiring Diagram 3L</h3><p class="mt">Page {wd.get("page_number","?")} \u2014 {wd.get("label","")}</p>{wi}</div></div>
{ch}
<div class="ib"><h3>Issues ({len(issues)})</h3>
<table><thead><tr><th style="width:80px">Severity</th><th>Description</th><th>SCH</th><th>WD</th></tr></thead>
<tbody>{ir if ir else "<tr><td colspan=4>No issues \u2014 diagrams consistent.</td></tr>"}</tbody></table></div></section>''')

    body = "\n".join(sections)
    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>IEM Circuit Comparison</title><style>
*{{box-sizing:border-box}}body{{font-family:-apple-system,'Segoe UI',Arial,sans-serif;margin:0;padding:24px;color:#222;background:#f5f6fa}}
h1{{color:#1a5276;margin-bottom:4px}}h1+p{{color:#666;margin-top:0}}h2{{color:#2c3e50;margin:0}}h3{{color:#34495e;margin-bottom:8px}}h4{{color:#555;margin:12px 0 8px}}
.sb{{display:flex;gap:12px;margin:20px 0}}.sc{{background:#fff;border-radius:8px;padding:16px 20px;text-align:center;flex:1;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.sc .n{{font-size:32px;font-weight:700}}.sc .l{{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.5px}}.nh{{color:#d32f2f}}.nm{{color:#f57c00}}.no{{color:#388e3c}}
.ps{{background:#fff;border-radius:10px;padding:24px;margin:24px 0;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.ph{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}.sm{{color:#555;font-style:italic;margin:4px 0 16px}}
.bh,.bm,.bo{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff}}.bh{{background:#d32f2f}}.bm{{background:#f57c00}}.bo{{background:#388e3c}}
.dr{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}.di{{width:100%;border:1px solid #ddd;border-radius:4px;margin-top:6px}}
.mt{{font-size:12px;color:#999;margin:2px 0 0}}.ni{{background:#fafafa;border:2px dashed #ddd;border-radius:4px;padding:40px;text-align:center;color:#aaa}}
.cs{{margin:16px 0;padding:16px;background:#f8f9fa;border-radius:6px;border:1px solid #eee}}.ci{{margin:10px 0}}.cl{{display:block;font-weight:600;color:#555;margin-bottom:4px}}
.ib{{margin-top:16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{background:#34495e;color:#fff;padding:8px 10px;text-align:left}}
td{{border:1px solid #e0e0e0;padding:8px 10px;vertical-align:top}}tr:nth-child(even){{background:#fafafa}}
tr.sh{{background:#fff5f5}}tr.sm{{background:#fff8e1}}tr.sn td{{color:#999}}
</style></head><body>
<h1>IEM Circuit Diagram Comparison</h1><p>Schematic vs Wiring Diagram \u2014 Three Line Diagram Review</p>
<div class="sb">
<div class="sc"><div class="n">{len(pairs)}</div><div class="l">Pairs</div></div>
<div class="sc"><div class="n {"nh" if high_n else "no"}">{high_n}</div><div class="l">High</div></div>
<div class="sc"><div class="n {"nm" if med_n else "no"}">{med_n}</div><div class="l">Medium</div></div>
<div class="sc"><div class="n">{low_n}</div><div class="l">Low/Nuisance</div></div></div>
{body}
</body></html>'''

    with open(output_path, "w") as f:
        f.write(html)

    return json.dumps({"output_path": os.path.abspath(output_path), "image_debug": debug}, indent=2)


@mcp.tool()
def analyze_single_wd_page(
    wd_image_path: str,
    page_label: str = "WD Page",
) -> str:
    """Analyze a single WD detail page for circuit errors and wire uniqueness.

    Sends the cropped WD page image to Bedrock Claude to inspect for:
    1. Circuit errors (open circuits, missing connections, wrong terminals)
    2. Duplicate wire numbers within this page
    3. Wire numbering consistency

    Args:
        wd_image_path: Path to the cropped WD page image.
        page_label: Human-readable label for this page.

    Returns:
        JSON with wire_numbers list, issues list, and page_label.
    """
    if not os.path.exists(wd_image_path):
        return json.dumps({"error": "Image not found", "page_label": page_label})

    with open(wd_image_path, "rb") as f:
        img_bytes = f.read()

    content_blocks: list[dict[str, Any]] = [
        {"text": (
            f"WIRING DIAGRAM ANALYSIS: {page_label}\n\n"
            "Analyze this wiring diagram page for errors.\n\n"
            "1. LIST ALL WIRE NUMBERS visible on this page. Wire numbers are\n"
            "   the labels on each wire/conductor (e.g. 1UA, 1UB, VGA1, etc.).\n"
            "   List every unique wire number you can read.\n\n"
            "2. CHECK FOR DUPLICATE WIRE NUMBERS on this page. If the same\n"
            "   wire number appears connecting to two DIFFERENT circuits or\n"
            "   terminals on this page, that is an error.\n\n"
            "3. CHECK FOR CIRCUIT ERRORS:\n"
            "   - Open circuits (wire goes nowhere)\n"
            "   - Missing connections (terminal with no wire)\n"
            "   - Wrong terminal assignments\n"
            "   - Crossed wires that should not cross\n\n"
            "FALSE POSITIVES ARE COSTLY. Only flag clear, visible errors.\n"
            "If you cannot clearly read a label, do NOT flag it.\n\n"
            "Return a JSON object:\n"
            '{"page_label": "...",\n'
            ' "wire_numbers": ["1UA", "1UB", "VGA1", ...],\n'
            ' "issues": [{"description": "...", "severity": "HIGH|MEDIUM|LOW",\n'
            '   "detail": "..."}],\n'
            ' "summary": "1 sentence"}\n'
            "Return ONLY the JSON object."
        )},
        {"image": {"format": "png", "source": {"bytes": img_bytes}}},
    ]

    result = _call_bedrock(content_blocks, max_tokens=8192)
    try:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    data.setdefault("page_label", page_label)
    data.setdefault("wire_numbers", [])
    data.setdefault("issues", [])
    return json.dumps(data, indent=2)


@mcp.tool()
def check_wire_uniqueness_across_pages(
    all_pages_json: str,
) -> str:
    """Check for duplicate wire numbers across multiple WD pages.

    Takes the combined results from analyze_single_wd_page calls and
    finds wire numbers that appear on more than one page. Duplicate
    wire numbers across different pages (connecting to different circuits)
    are flagged as errors.

    Args:
        all_pages_json: JSON list of results from analyze_single_wd_page.

    Returns:
        JSON with duplicate_wires list and per-page wire counts.
    """
    pages: list[dict[str, Any]] = json.loads(all_pages_json)

    wire_to_pages: dict[str, list[str]] = {}
    for pg in pages:
        label = pg.get("page_label", "unknown")
        for wire in pg.get("wire_numbers", []):
            w = wire.strip().upper()
            if w:
                wire_to_pages.setdefault(w, []).append(label)

    duplicates: list[dict[str, Any]] = []
    for wire, page_list in sorted(wire_to_pages.items()):
        if len(page_list) > 1:
            unique_pages = sorted(set(page_list))
            if len(unique_pages) > 1:
                duplicates.append({
                    "wire_number": wire,
                    "found_on_pages": unique_pages,
                    "count": len(unique_pages),
                    "severity": "MEDIUM" if len(unique_pages) == 2 else "HIGH",
                })

    page_stats = [
        {"page": pg.get("page_label", "?"), "wire_count": len(pg.get("wire_numbers", []))}
        for pg in pages
    ]

    return json.dumps({
        "total_unique_wires": len(wire_to_pages),
        "total_pages": len(pages),
        "duplicate_wires": duplicates,
        "duplicate_count": len(duplicates),
        "page_stats": page_stats,
    }, indent=2)


@mcp.tool()
def generate_wd_analysis_report(
    analysis_json: str,
    wd_pages_json: str,
    output_path: str = "wd_analysis_report.html",
) -> str:
    """Generate an HTML report for standalone WD diagram analysis.

    Args:
        analysis_json: JSON with per-page issues and cross-page duplicates.
        wd_pages_json: JSON list of WD page entries with cropped_path.
        output_path: Where to save the HTML file.

    Returns:
        JSON with the output file path.
    """
    wd_pages: list[dict[str, Any]] = json.loads(wd_pages_json)
    try:
        analysis = json.loads(analysis_json)
    except json.JSONDecodeError:
        analysis = {}

    per_page = analysis.get("per_page_results", [])
    cross_page = analysis.get("cross_page_duplicates", {})
    duplicates = cross_page.get("duplicate_wires", [])
    total_wires = cross_page.get("total_unique_wires", 0)
    dup_count = cross_page.get("duplicate_count", 0)

    all_issues: list[dict[str, Any]] = []
    for pg in per_page:
        for i in pg.get("issues", []):
            i["page"] = pg.get("page_label", "?")
            all_issues.append(i)
    for d in duplicates:
        all_issues.append({
            "page": ", ".join(d.get("found_on_pages", [])),
            "description": f"Wire '{d['wire_number']}' appears on {d['count']} different pages",
            "severity": d.get("severity", "MEDIUM"),
            "detail": f"Found on: {', '.join(d.get('found_on_pages', []))}",
        })

    high_n = sum(1 for i in all_issues if i.get("severity", "").upper() == "HIGH")
    med_n = sum(1 for i in all_issues if i.get("severity", "").upper() == "MEDIUM")

    issue_rows = ""
    for i in sorted(all_issues, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.get("severity", "LOW").upper(), 3)):
        sv = i.get("severity", "LOW").upper()
        co = {"HIGH": "#d32f2f", "MEDIUM": "#f57c00", "LOW": "#388e3c"}.get(sv, "#888")
        issue_rows += (
            f'<tr class="s{sv[0].lower()}">'
            f'<td style="color:{co};font-weight:700">{sv}</td>'
            f'<td>{i.get("page", "")}</td>'
            f'<td>{i.get("description", "")}</td>'
            f'<td>{i.get("detail", "")}</td></tr>'
        )

    page_sections = ""
    for idx, pg_result in enumerate(per_page):
        lbl = pg_result.get("page_label", f"WD Page {idx+1}")
        wires = pg_result.get("wire_numbers", [])
        pg_issues = pg_result.get("issues", [])
        img_path = ""
        if idx < len(wd_pages):
            img_path = wd_pages[idx].get("cropped_path", "")
        img_html = f'<img src="data:image/png;base64,{_image_to_b64(img_path)}" class="di">' if img_path and os.path.exists(img_path) else '<div class="ni">No image</div>'

        page_sections += f'''<div class="pg">
<h3>{lbl}</h3>
<div class="pg-grid"><div>{img_html}</div>
<div><p><strong>Wire numbers ({len(wires)}):</strong> {", ".join(wires[:30])}{" ..." if len(wires) > 30 else ""}</p>
<p><strong>Issues:</strong> {len(pg_issues)} found</p>
</div></div></div>'''

    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>IEM WD Analysis Report</title><style>
*{{box-sizing:border-box}}body{{font-family:-apple-system,'Segoe UI',Arial,sans-serif;margin:0;padding:24px;color:#222;background:#f5f6fa}}
h1{{color:#1a5276;margin-bottom:4px}}h1+p{{color:#666;margin-top:0}}h2{{color:#2c3e50;margin-top:28px}}h3{{color:#34495e;margin-bottom:8px}}
.sb{{display:flex;gap:12px;margin:20px 0}}.sc{{background:#fff;border-radius:8px;padding:16px 20px;text-align:center;flex:1;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.sc .n{{font-size:32px;font-weight:700}}.sc .l{{font-size:12px;color:#888;text-transform:uppercase}}.nh{{color:#d32f2f}}.nm{{color:#f57c00}}.no{{color:#388e3c}}
.card{{background:#fff;border-radius:10px;padding:24px;margin:20px 0;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}th{{background:#34495e;color:#fff;padding:8px 10px;text-align:left}}
td{{border:1px solid #e0e0e0;padding:8px 10px;vertical-align:top}}tr:nth-child(even){{background:#fafafa}}
tr.sh{{background:#fff5f5}}tr.sm{{background:#fff8e1}}
.pg{{background:#f8f9fa;border:1px solid #eee;border-radius:6px;padding:16px;margin:12px 0}}
.pg-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.di{{width:100%;border:1px solid #ddd;border-radius:4px}}
.ni{{background:#fafafa;border:2px dashed #ddd;border-radius:4px;padding:40px;text-align:center;color:#aaa}}
</style></head><body>
<h1>IEM Wiring Diagram Analysis Report</h1>
<p>Standalone WD inspection \u2014 circuit errors and wire number uniqueness</p>
<div class="sb">
<div class="sc"><div class="n">{len(per_page)}</div><div class="l">WD Pages</div></div>
<div class="sc"><div class="n">{total_wires}</div><div class="l">Unique Wires</div></div>
<div class="sc"><div class="n {"nh" if dup_count else "no"}">{dup_count}</div><div class="l">Cross-Page Duplicates</div></div>
<div class="sc"><div class="n {"nh" if high_n else "no"}">{high_n}</div><div class="l">High</div></div>
<div class="sc"><div class="n {"nm" if med_n else "no"}">{med_n}</div><div class="l">Medium</div></div>
</div>
<div class="card"><h2>All Issues</h2>
<table><thead><tr><th style="width:80px">Severity</th><th>Page</th><th>Description</th><th>Detail</th></tr></thead>
<tbody>{issue_rows if issue_rows else "<tr><td colspan=4>No issues found.</td></tr>"}</tbody></table></div>
<div class="card"><h2>WD Pages Analyzed</h2>{page_sections}</div>
</body></html>'''

    with open(output_path, "w") as f:
        f.write(html)
    return json.dumps({"output_path": os.path.abspath(output_path)})


def main() -> None:
    """Entry point for the MCP server."""
    transport = "sse" if "--sse" in sys.argv else "stdio"
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
