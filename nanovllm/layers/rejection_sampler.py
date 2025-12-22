import torch
from torch import nn

from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler


class RejectionSampler(nn.Module):

    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler
    
    def prepare_tensors(self, seqs: list[Sequence]):
        batch_size = len(seqs)
        draft_tokens = []
        num_draft_tokens = [0]
        max_num_sampled_tokens = 0
        
        for seq in seqs:
            draft_tokens.extend(seq.decode_draft_tokens)
            num_draft_tokens.append(seq.num_spec_tokens)
            max_num_sampled_tokens = max(seq.num_spec_tokens + 1, max_num_sampled_tokens)
        
        draft_tokens = torch.tensor(draft_tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        num_draft_tokens = torch.tensor(num_draft_tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        num_sampled_tokens = num_draft_tokens + 1
        num_sampled_tokens[0] = 0
        cum_num_draft_tokens = torch.cumsum(num_draft_tokens, dim=0)
        cum_num_sampled_tokens = torch.cumsum(num_sampled_tokens, dim=0)
        
        device = device=draft_tokens.device
        output_tokens = torch.zeros((batch_size, max_num_sampled_tokens), dtype=torch.int64, device=device)
        num_accepted_tokens = torch.zeros((batch_size, ), dtype=torch.int64, device=device)
        
        return (
            draft_tokens, cum_num_draft_tokens, cum_num_sampled_tokens,
            output_tokens, num_accepted_tokens
        )

    def rejection_sample(self, 
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        draft_tokens: torch.Tensor,
        cum_num_draft_tokens: torch.Tensor,
        cum_num_sampled_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ):
        
        # [num_tokens, ...]
        sample_tokens = self.sampler(logits, temperatures)
        # TODO: optimize the rejection sample process
        
        num_seq = cum_num_draft_tokens.shape[-1] - 1
        for i in range(num_seq):
            draft_tokens_start = cum_num_draft_tokens[i]
            draft_tokens_end = cum_num_draft_tokens[i + 1]
            sampled_tokens_start = cum_num_sampled_tokens[i]
            
            # num sample tokens = num draft tokens + 1, so it's safe here
            j = 0
            while draft_tokens_start + j < draft_tokens_end:
                if sample_tokens[draft_tokens_start + j] != draft_tokens[draft_tokens_start + j]:
                    break
                output_tokens[i][j] = draft_tokens[draft_tokens_start + j]
                j += 1
            
            output_tokens[i][j] = sample_tokens[sampled_tokens_start + j]
            num_accepted_tokens[i] = j + 1
        
        
        return output_tokens, num_accepted_tokens

    def forward(
        self,
        seqs: list[Sequence],
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> list[list[int]]:
        (
            draft_tokens, cum_num_draft_tokens, cum_num_sampled_tokens,
            output_tokens, num_accepted_tokens
        ) = self.prepare_tensors(seqs)
        
        output_tokens, num_accepted_tokens = self.rejection_sample(
            logits=logits,
            temperatures=temperatures,
            draft_tokens = draft_tokens,
            cum_num_draft_tokens=cum_num_draft_tokens,
            cum_num_sampled_tokens=cum_num_sampled_tokens,
            output_tokens=output_tokens,
            num_accepted_tokens=num_accepted_tokens
        )
        
        _, max_num_sampled_tokens = output_tokens.shape
        positions = torch.arange(max_num_sampled_tokens, dtype=torch.int64, device=num_accepted_tokens.device)
        accept_mask = positions.unsqueeze(0) < num_accepted_tokens.unsqueeze(1)
        accept_tokens = output_tokens[accept_mask]
        
        accepted_tokens_start_indices = torch.cumsum(num_accepted_tokens, dim=0) - num_accepted_tokens
        
        return accept_tokens, accepted_tokens_start_indices
        