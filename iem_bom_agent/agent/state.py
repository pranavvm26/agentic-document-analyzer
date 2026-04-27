"""
Agent state schema for the IEM BOM comparison workflow.

Defines the TypedDict that flows through every node in the LangGraph,
carrying document paths, intermediate extraction results, and the
final comparison report.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class BOMComparisonState(MessagesState):
    """State that accumulates through the BOM comparison pipeline.

    Inherits ``messages`` from MessagesState for LLM conversation history
    and adds domain-specific fields for each processing stage.

    Attributes:
        review_mode: One of "bom", "circuit", or "both".
        schematic_pdf_path: File path to the original schematic PDF.
        wiring_diagram_pdf_path: File path to the wiring diagram PDF.
        schematic_pages: JSON map of page keys to image paths.
        wiring_diagram_pages: JSON map of page keys to image paths.
        schematic_drawing_index: Parsed drawing index JSON.
        wiring_diagram_drawing_index: Parsed drawing index JSON.
        schematic_bom_csv: Merged BOM CSV for the schematic.
        wiring_diagram_bom_csv: Merged BOM CSV for the wiring diagram.
        comparison_report: Final structured diff report JSON.
        error: Error message if any step fails.
        current_step: Tracks which pipeline stage is active.
        retry_count: Number of self-correction retries attempted.
    """

    review_mode: str
    schematic_pdf_path: str
    wiring_diagram_pdf_path: str
    schematic_pages: str
    wiring_diagram_pages: str
    schematic_drawing_index: str
    wiring_diagram_drawing_index: str
    schematic_bom_csv: str
    wiring_diagram_bom_csv: str
    comparison_report: str
    error: str
    current_step: str
    retry_count: int
