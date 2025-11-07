import re
from typing import Dict, List, Tuple, Iterable, Iterator

def train_bpe(input_path:str, vocab_size: int, special_tokens: list[str]) -> (Dict[str, bytes], List[Tuple[bytes, bytes]]):
        """
        Args:
        - input_path: str Path to a text file with BPE tokenizer training data.
        - vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        - special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
        
        Returns:
        - vocab: Dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        - merges: List[Tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
        """
        pass

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        pass

    # def pretokenize(self, text: str) -> str:
    #     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #     return re.findall(PAT, text) #use re.finditer instead

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        pass
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

# uv run pytest tests/test_tokenizer.py
        

        


