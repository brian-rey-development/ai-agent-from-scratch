"""Load .env and validate the OpenRouter key. Import this first in a notebook.

    from scratch_agents.env import PROJECT_ROOT, require
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_MARKER = "REPLACE_ME"
OPENROUTER_KEY = "OPENROUTER_API_KEY"

load_dotenv(PROJECT_ROOT / ".env")


def require(name: str) -> str:
    value = os.getenv(name, "")
    if not value or PLACEHOLDER_MARKER in value:
        raise RuntimeError(f"Set {name} in {PROJECT_ROOT / '.env'}")
    return value


def key_status() -> dict[str, bool]:
    names = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
             "TAVILY_API_KEY", "HF_TOKEN", "E2B_API_KEY")
    return {
        name: bool(os.getenv(name)) and PLACEHOLDER_MARKER not in os.getenv(name, "")
        for name in names
    }


require(OPENROUTER_KEY)
