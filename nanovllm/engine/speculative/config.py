from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    max_draft_tokens: int = 4
    draft_model: str | None = None
