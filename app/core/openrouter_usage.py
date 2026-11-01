"""Normalize reported OpenRouter costs, including upstream billing for BYOK."""


def openrouter_usage_properties(usage: dict) -> dict:
    router_cost = usage.get("cost")
    upstream_cost = (usage.get("cost_details") or {}).get("upstream_inference_cost")
    is_byok = usage.get("is_byok")
    # With BYOK, cost is the router fee (often zero), not the upstream provider bill.
    # For ordinary credit requests cost already includes inference; do not add it twice.
    total_cost = router_cost
    if is_byok and upstream_cost is not None:
        total_cost = upstream_cost + (router_cost or 0)
    return {
        key: value
        for key, value in {
            "$ai_total_cost_usd": total_cost,
            "openrouter_cost_usd": router_cost,
            "upstream_inference_cost_usd": upstream_cost,
            "is_byok": is_byok,
            "$ai_total_tokens": usage.get("total_tokens"),
            "$ai_reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "$ai_cache_read_input_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
        }.items()
        if value is not None
    }
