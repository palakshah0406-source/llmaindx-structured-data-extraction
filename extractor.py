"""
Structured data extraction from natural language queries using LlamaIndex (Gemini).
Extracts: start_time, end_time, geo_location, constraints, goal, followup, category
"""

import sys
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    raise RuntimeError(
        "This project targets Python 3.12.x only. Recreate the venv with Python 3.12 "
        "(see .python-version and pyproject.toml), then pip install -r requirements.txt."
    )

from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ExtractedQuery(BaseModel):
    """Structured representation of a natural language query."""

    start_time: Optional[str] = Field(
        None,
        description=(
            "Start time or date mentioned in the query. "
            "Use ISO 8601 format if deterministic (e.g. '2026-05-01T09:00:00'), "
            "or a human-readable phrase (e.g. 'next Monday morning') when relative."
        ),
    )
    end_time: Optional[str] = Field(
        None,
        description=(
            "End time or date mentioned in the query. "
            "Same format rules as start_time."
        ),
    )
    geo_location: Optional[str] = Field(
        None,
        description=(
            "Geographic location referenced in the query "
            "(city, region, country, coordinates, or address)."
        ),
    )
    constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Explicit or implicit restrictions / requirements stated in the query "
            "(e.g. 'budget under $500', 'must be vegetarian', 'no weekends')."
        ),
    )
    goal: str = Field(
        ...,
        description="The primary objective or intent the user wants to achieve.",
    )
    followup: List[str] = Field(
        default_factory=list,
        description=(
            "Suggested follow-up questions or clarifications that would help "
            "fulfil the user's goal more precisely."
        ),
    )
    category: str = Field(
        ...,
        description=(
            "High-level category for the query. "
            "Examples: Travel, Event Planning, Research, Shopping, Health, "
            "Finance, Education, Entertainment, Productivity, Other."
        ),
    )


# ---------------------------------------------------------------------------
# Extraction helper
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a structured-data extraction assistant.
Given a user query, extract the relevant fields precisely.
Today's date is {today}.
If a field is not mentioned or cannot be reasonably inferred, leave it null / empty.
Always populate 'goal' and 'category'.
"""

USER_PROMPT = """\
Extract structured information from the following query:

Query: {query}
"""


GEMINI_MODEL = "gemini-2.5-flash"


def build_llm() -> Gemini:
    return Gemini(model=GEMINI_MODEL)


def extract(query: str, llm: Optional[Gemini] = None) -> ExtractedQuery:
    """Extract structured data from a natural language query."""
    if llm is None:
        llm = build_llm()

    sllm = llm.as_structured_llm(output_cls=ExtractedQuery)

    today = datetime.now().strftime("%Y-%m-%d")
    system_msg = ChatMessage.from_str(
        SYSTEM_PROMPT.format(today=today), role="system"
    )
    user_msg = ChatMessage.from_str(USER_PROMPT.format(query=query), role="user")

    response = sllm.chat([system_msg, user_msg])
    return response.raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: ExtractedQuery) -> None:
    print("\n--- Extracted Structure ---")
    print(f"  goal        : {result.goal}")
    print(f"  category    : {result.category}")
    print(f"  start_time  : {result.start_time or '—'}")
    print(f"  end_time    : {result.end_time or '—'}")
    print(f"  geo_location: {result.geo_location or '—'}")
    if result.constraints:
        for i, c in enumerate(result.constraints, 1):
            print(f"  constraint {i}: {c}")
    else:
        print("  constraints : —")
    if result.followup:
        for i, f in enumerate(result.followup, 1):
            print(f"  followup  {i}: {f}")
    else:
        print("  followup    : —")
    print()


DEMO_QUERIES = [
    "Find me a budget-friendly hotel in Tokyo for 3 nights starting next Friday, under $150/night, near public transport.",
    "I need to schedule a team meeting in New York between 9am and 12pm on Monday, lasting no more than 1 hour, for the Q2 planning discussion.",
    "What are the best vegetarian restaurants open on Sunday evening in Berlin with outdoor seating?",
    "Remind me to renew my car insurance before it expires on June 30th and compare quotes from at least 3 providers.",
]


if __name__ == "__main__":
    llm = build_llm()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}")
        result = extract(query, llm)
        _print_result(result)
    else:
        print("Running demo queries...\n")
        for query in DEMO_QUERIES:
            print(f"Query: {query}")
            result = extract(query, llm)
            _print_result(result)
