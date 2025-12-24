from __future__ import annotations

from abc import ABC, abstractmethod


class DecodingStrategy(ABC):
    @abstractmethod
    def step(self, engine):
        raise NotImplementedError


class DefaultDecodingStrategy(DecodingStrategy):
    def step(self, engine):
        seqs, is_prefill = engine.scheduler.schedule()
        token_ids = engine.model_runner.call("run", seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens


class SpeculativeDecodingStrategy(DecodingStrategy, ABC):
    @abstractmethod
    def draft(self, engine, seqs, is_prefill: bool):
        raise NotImplementedError

    @abstractmethod
    def verify(self, engine, seqs, draft_token_ids):
        raise NotImplementedError


class SingleDraftSingleTargetStrategy(SpeculativeDecodingStrategy):
    def __init__(self, config):
        self.config = config

    def step(self, engine):
        raise NotImplementedError("Speculative decoding strategy is not wired yet.")

    def draft(self, engine, seqs, is_prefill: bool):
        raise NotImplementedError("Draft model execution is not wired yet.")

    def verify(self, engine, seqs, draft_token_ids):
        raise NotImplementedError("Target verification is not wired yet.")
