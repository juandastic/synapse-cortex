"""Keep stable persona and memory text ahead of per-turn context for caching."""

import re

RAG_HEADER = (
    "\n\n### RELEVANT EPISODIC MEMORY FOR THIS TURN ###\n"
    "(The following facts and entities were retrieved from the user's "
    "graph memory based on the current conversation context. "
    "Use them if relevant.)\n\n"
)
_CURRENT_TIME = re.compile(r"^Current date and time: [^\n]*$", re.MULTILINE)


def assemble_system_context(instructions: str, compilation: str | None = None) -> str:
    # Cortex adds this exact delimiter; retain the retrieved text verbatim.
    stable, separator, recalled = instructions.partition(RAG_HEADER)
    dates = _CURRENT_TIME.findall(stable)
    stable = _CURRENT_TIME.sub("", stable).strip()
    return "\n\n".join(
        part for part in (stable, compilation, *dates, (separator + recalled).strip()) if part
    )
