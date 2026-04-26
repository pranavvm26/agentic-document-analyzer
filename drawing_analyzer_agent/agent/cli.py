"""
CLI entrypoint for the IEM Drawing Review Agent.

Usage:
    python -m iem_bom_agent.agent.cli \\
        --schematic docs-sample/124329-11_R3.pdf \\
        --wiring-diagram docs-sample/124329-11WD_R0.pdf \\
        --mode both

Modes:
    bom      - BOM comparison only (Phases A–D)
    circuit  - Circuit diagram comparison only (Phases A1–A3, B1–B3, E–G)
    both     - Full review: BOM then circuit (default)

Environment variables:
    AWS_REGION          - AWS region for Bedrock (default: us-east-1)
    OCR_BASE_URL        - Local OCR server URL (default: http://localhost:8080/v1)
    OCR_MODEL           - OCR model name (default: zai-org/GLM-OCR)
"""

from __future__ import annotations

import argparse
import logging
import sys

from .graph import run_review


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with schematic, wiring_diagram, mode, verbose.
    """
    parser = argparse.ArgumentParser(
        prog="iem-review-agent",
        description=(
            "LangGraph agent that reviews IEM drawing packages — "
            "compares BOM tables and/or circuit diagrams between "
            "a schematic and its wiring diagram."
        ),
    )
    parser.add_argument(
        "--schematic",
        required=True,
        help="Path to the original schematic PDF.",
    )
    parser.add_argument(
        "--wiring-diagram",
        required=True,
        help="Path to the wiring diagram (WD) PDF.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["bom", "circuit", "both", "wdanalysis"],
        default="both",
        help="Review mode: 'bom', 'circuit', 'both', or 'wdanalysis' (standalone WD inspection).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the review agent from the command line.

    Parses arguments, configures logging, executes the LangGraph
    workflow in the selected mode, and prints the final report to stdout.

    Args:
        argv: Optional argument list for testing.
    """
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    logger = logging.getLogger("iem_bom_agent")
    logger.info("Starting IEM review workflow")
    logger.info("  Schematic:       %s", args.schematic)
    logger.info("  Wiring Diagram:  %s", args.wiring_diagram)
    logger.info("  Mode:            %s", args.mode)

    try:
        report = run_review(
            schematic_pdf=args.schematic,
            wiring_diagram_pdf=args.wiring_diagram,
            mode=args.mode,
            verbose=args.verbose,
        )
        print(report)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Agent workflow failed with an unhandled error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
