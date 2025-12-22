from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
    
    def allocate_block(self, token_ids: list[int], h) -> tuple[Block, bool]:
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        cache_miss = False
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        
        return block, cache_miss        

    def can_allocate_prefill(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate_prefill(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            block, cache_miss = self.allocate_block(token_ids, h)
            if not cache_miss:
                seq.num_cached_prefill_tokens += self.block_size
            assert isinstance(block, Block)
            seq.block_table.append(block.block_id)
            h = block.hash    

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_prefill_tokens = 0
        seq.block_table.clear()

    def can_allocate_decode(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (seq.num_blocks - len(seq.block_table))

    def allocate_decode(self, seq: Sequence):
        num_decode_tokens = seq.num_decode_tokens
        block_table = seq.block_table
        last_block: Block = self.blocks[block_table[-1]]
        
        num_cached_blocks = len(block_table)
        num_total_blocks = seq.num_blocks
        
        # fill in the last cached block
        token_ids = seq.block(num_cached_blocks - 1)
        if last_block.hash == -1 and len(token_ids) == self.block_size:
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)      
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
                
        # allocate new blocks
        h = last_block.hash
        for i in range(num_cached_blocks, num_total_blocks):
            token_ids = seq.block(i)
            block, _ = self.allocate_block(token_ids, h)
            assert isinstance(block, Block)
            seq.block_table.append(block.block_id)
            h = block.hash
    
    def deallocate_decode(self, seq: Sequence):
        block_table = seq.block_table
        num_reject_blocks = len(block_table) - seq.num_blocks
        
        if num_reject_blocks > 0:
            for i in range(1, num_reject_blocks + 1):
                block_id = block_table[-i]
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
            del block_table[- num_reject_blocks: ]
        
        if seq.last_block_num_tokens > 0: # not full
            last_block = self.blocks[block_table[-1]]
            last_block.hash = -1
            last_block.token_ids = seq.block(seq.num_blocks -1)
