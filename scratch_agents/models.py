"""Curated OpenRouter models for the book's agents.

Snapshot from the OpenRouter /models API on 2026-10-02. Every model supports
tool calling. Prices are USD per 1M tokens and change often, so re-check
https://openrouter.ai/models before a long run.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from scratch_agents.llm import LlmClient


class Model(StrEnum):
    GPT_6_LUNA = "openrouter/openai/gpt-6-luna"
    GEMINI_3_5_FLASH_LITE = "openrouter/google/gemini-3.5-flash-lite"
    GEMINI_3_8_FLASH = "openrouter/google/gemini-3.8-flash"
    CLAUDE_HAIKU_4_5 = "openrouter/anthropic/claude-haiku-4.5"
    CLAUDE_SONNET_5_5 = "openrouter/anthropic/claude-sonnet-5.5"
    NEMOTRON_3_5_LIGHTNING = "openrouter/nvidia/nemotron-3.5-lightning"
    MIMO_V2_6_FLASH = "openrouter/xiaomi/mimo-v2.6-flash"
    GLM_5_3_FLASH = "openrouter/z-ai/glm-5.3-flash"
    QWEN3_8_FLASH = "openrouter/qwen/qwen3.8-flash"
    DEEPSEEK_V4_1_FLASH = "openrouter/deepseek/deepseek-v4.1-flash"
    QWEN3_8_27B = "openrouter/qwen/qwen3.8-27b"
    DEEPSEEK_V4_PRO = "openrouter/deepseek/deepseek-v4-pro-0813"
    GLM_5_3 = "openrouter/z-ai/glm-5.3"


class Tier(StrEnum):
    BUDGET = "budget"
    BALANCED = "balanced"
    PREMIUM = "premium"


@dataclass(frozen=True)
class ModelSpec:
    tier: Tier
    open_weights: bool
    price_in: float
    price_out: float
    context: int
    params: dict[str, Any] = field(default_factory=dict)


LOW_EFFORT = {"reasoning_effort": "low"}

SPECS: dict[Model, ModelSpec] = {
    Model.GPT_6_LUNA: ModelSpec(Tier.BUDGET, False, 0.10, 0.50, 1_050_000, LOW_EFFORT),
    Model.GEMINI_3_5_FLASH_LITE: ModelSpec(Tier.BUDGET, False, 0.30, 2.50, 1_048_576, LOW_EFFORT),
    Model.GEMINI_3_8_FLASH: ModelSpec(Tier.BALANCED, False, 0.75, 3.75, 1_048_576, LOW_EFFORT),
    Model.CLAUDE_HAIKU_4_5: ModelSpec(Tier.BALANCED, False, 1.00, 5.00, 200_000),
    Model.CLAUDE_SONNET_5_5: ModelSpec(Tier.PREMIUM, False, 2.00, 10.00, 1_000_000),
    Model.NEMOTRON_3_5_LIGHTNING: ModelSpec(Tier.BUDGET, True, 0.06, 0.16, 262_144),
    Model.MIMO_V2_6_FLASH: ModelSpec(Tier.BUDGET, True, 0.14, 0.28, 1_050_000),
    Model.GLM_5_3_FLASH: ModelSpec(Tier.BUDGET, True, 0.15, 0.50, 1_048_576, LOW_EFFORT),
    Model.QWEN3_8_FLASH: ModelSpec(Tier.BUDGET, True, 0.15, 0.47, 1_000_000),
    Model.DEEPSEEK_V4_1_FLASH: ModelSpec(Tier.BALANCED, True, 0.30, 1.20, 1_048_576, LOW_EFFORT),
    Model.QWEN3_8_27B: ModelSpec(Tier.BALANCED, True, 0.42, 3.00, 1_000_000, LOW_EFFORT),
    Model.DEEPSEEK_V4_PRO: ModelSpec(Tier.PREMIUM, True, 0.66, 1.98, 1_048_576, LOW_EFFORT),
    Model.GLM_5_3: ModelSpec(Tier.PREMIUM, True, 1.40, 4.40, 1_048_576, LOW_EFFORT),
}

DEFAULT_MODEL = Model.GPT_6_LUNA


def get_client(model: Model = DEFAULT_MODEL, **overrides: Any) -> LlmClient:
    return LlmClient(model=model.value, **{**SPECS[model].params, **overrides})


def list_models(tier: Tier | None = None, open_weights: bool | None = None) -> list[Model]:
    return [
        model for model, spec in SPECS.items()
        if (tier is None or spec.tier == tier)
        and (open_weights is None or spec.open_weights == open_weights)
    ]
