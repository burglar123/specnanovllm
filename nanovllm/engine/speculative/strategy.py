from __future__ import annotations

from abc import ABC, abstractmethod

from nanovllm.engine.sequence import SequenceStatus


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
        seqs, is_prefill = engine.scheduler.schedule()
        if is_prefill or engine.draft_model_runner is None:
            token_ids = engine.target_model_runner.call("run", seqs, is_prefill)
            engine.scheduler.postprocess(seqs, token_ids)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
            return outputs, num_tokens

        block_manager = engine.scheduler.block_manager
        original_lengths = [len(seq) for seq in seqs]
        draft_token_ids = [[] for _ in seqs]
        for _ in range(self.config.max_draft_tokens):
            token_ids = engine.draft_model_runner.call("run", seqs, False)
            for seq, token_id, draft_tokens in zip(seqs, token_ids, draft_token_ids):
                block_manager.may_append(seq)
                seq.append_token(token_id)
                draft_tokens.append(token_id)

        verify_logits = engine.target_model_runner.call("run_verify", seqs, draft_token_ids)
        accepted_tokens = []
        total_accepted = 0
        for idx, (seq, draft_tokens, logits) in enumerate(zip(seqs, draft_token_ids, verify_logits)):
            if len(draft_tokens) == 0:
                accepted_tokens.append([])
                continue
            greedy_tokens = logits.argmax(-1).tolist()
            accepted_len = 0
            for greedy, draft_token in zip(greedy_tokens, draft_tokens):
                if greedy != draft_token:
                    break
                accepted_len += 1
            accepted_tokens.append(draft_tokens[:accepted_len])
            total_accepted += accepted_len
            if accepted_len < len(draft_tokens):
                new_len = original_lengths[idx] + accepted_len
                block_manager.truncate(seq, new_len)
                block_manager.may_append(seq)
                token_id = engine.target_model_runner.call("run", [seq], False)[0]
                seq.append_token(token_id)
                accepted_tokens[-1].append(token_id)
                total_accepted += 1

        eos_token = engine.scheduler.eos
        outputs = []
        for seq, original_len, tokens in zip(seqs, original_lengths, accepted_tokens):
            final_len = original_len + len(tokens)
            for idx, token_id in enumerate(tokens):
                if (not seq.ignore_eos and token_id == eos_token) or (
                    (original_len + idx + 1) - seq.num_prompt_tokens == seq.max_tokens
                ):
                    final_len = original_len + idx + 1
                    seq.truncate_to(final_len)
                    seq.status = SequenceStatus.FINISHED
                    block_manager.deallocate(seq)
                    engine.scheduler.running.remove(seq)
                    outputs.append((seq.seq_id, seq.completion_token_ids))
                    break
            else:
                if len(seq) != final_len:
                    block_manager.truncate(seq, final_len)
        return outputs, -total_accepted

    def draft(self, engine, seqs, is_prefill: bool):
        token_ids = engine.draft_model_runner.call("run", seqs, is_prefill)
        return token_ids

    def verify(self, engine, seqs, draft_token_ids):
        return engine.target_model_runner.call("run_verify", seqs, draft_token_ids)
