"""
Tool bridge: wraps MCP server functions as LangChain-compatible tools.

Instead of going through the MCP transport layer at runtime, this module
imports the MCP server's underlying Python functions directly and wraps
them with StructuredTool so LangGraph's ToolNode can invoke them.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import StructuredTool

from iem_bom_agent.mcp_server.server import (
    analyze_3l_diagram,
    analyze_single_wd_page,
    check_wire_uniqueness_across_pages,
    compare_bom_images_direct,
    compare_boms_with_vision,
    compare_single_3l_pair,
    concat_bom_csvs,
    extract_bom_table_from_page,
    extract_diagram_pages,
    generate_circuit_html_report,
    generate_html_report,
    generate_wd_analysis_report,
    ocr_page_text,
    parse_drawing_index,
    pdf_to_images,
    read_drawing_index_from_image,
    resolve_continuations,
    validate_page_is_bom,
    validate_wd_page_label,
)


@lru_cache(maxsize=1)
def get_mcp_tools() -> list[StructuredTool]:
    """Build and return the list of LangChain tools for the agent.

    Returns:
        List of StructuredTool instances ready for LLM binding.
    """
    raw_tools = [
        pdf_to_images,
        ocr_page_text,
        parse_drawing_index,
        read_drawing_index_from_image,
        validate_page_is_bom,
        extract_bom_table_from_page,
        concat_bom_csvs,
        compare_boms_with_vision,
        compare_bom_images_direct,
        generate_html_report,
        extract_diagram_pages,
        analyze_3l_diagram,
        validate_wd_page_label,
        resolve_continuations,
        compare_single_3l_pair,
        generate_circuit_html_report,
        analyze_single_wd_page,
        check_wire_uniqueness_across_pages,
        generate_wd_analysis_report,
    ]

    langchain_tools: list[StructuredTool] = []
    for mcp_func in raw_tools:
        fn = mcp_func.fn if hasattr(mcp_func, "fn") else mcp_func
        tool = StructuredTool.from_function(
            func=fn,
            name=fn.__name__,
            description=fn.__doc__ or "",
        )
        langchain_tools.append(tool)

    return langchain_tools
