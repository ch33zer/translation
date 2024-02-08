from typing import List, Dict, Tuple, Optional
import collections
import re
from abc import abstractmethod
import pickle
from pathlib import Path

DEFAULT_BLOCK_SIZE = 520
BOS = "BOS"
BOS_I = 0
EOS = "EOS"
EOS_I = 1
PAD = "PAD"
PAD_I = 2
TOKEN_MAPPING = {BOS: BOS_I, EOS: EOS_I, PAD: PAD_I}
REVERSE_TOKEN_MAPPING = {v: k for k, v in TOKEN_MAPPING.items()}


class EncoderDecoder:
    @abstractmethod
    def encode(self, string: str) -> List[int]: ...
    @abstractmethod
    def decode(self, encoded: List[int]) -> str: ...
    @abstractmethod
    def vocab_size(self) -> int: ...

    def write(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def tryload(cls, filename: str) -> Optional['EncoderDecoder']:
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except OSError:
            return None

class JPEncoderDecoder(EncoderDecoder):
    def __init__(self, sentences: List[str], block_size: int = DEFAULT_BLOCK_SIZE):
        chars = sorted(list(set("".join(sentences))))
        self.stoi = {
            **TOKEN_MAPPING,
            **{ch: i + len(TOKEN_MAPPING) for i, ch in enumerate(chars)},
        }
        self.itos = {v: k for k, v in self.stoi.items()}
        self.block_size = block_size

    def encode(self, string: str) -> List[int]:
        enc = [self.stoi[BOS]] + [self.stoi[c] for i, c in enumerate(string)]
        enc += [self.stoi[EOS]]
        enc += [self.stoi[PAD]] * (self.block_size - len(enc))
        return enc

    def decode(self, encoded: List[int]) -> str:
        return "".join([self.itos[c] for c in encoded])

    def vocab_size(self) -> int:
        return len(self.stoi)


class LinkedListIter:
    def __init__(self, head):
        self.head = head

    def __next__(self):
        if not self.head:
            raise StopIteration
        curr = self.head
        self.head = self.head.next
        return curr


class LinkedList:
    def __init__(self, contents):
        self.contents = contents
        self.next = None
        self.cached_full = None

    @staticmethod
    def create(chunk):
        curr = None
        head = None
        for c in chunk:
            new = LinkedList(c)
            if curr:
                curr.next = new
            else:
                head = new
            curr = new
        head.cached_full = chunk
        return head

    def merge(self):
        assert self.next
        self.contents += self.next.contents
        self.next = self.next.next

    def __iter__(self):
        return LinkedListIter(self)

    def __repr__(self):
        return f"LL{{{self.contents}, {self.next}}}"


class ENEncoderDecoder(EncoderDecoder):
    PRE_TOKENIZATION_REGEX = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+|\s|\S")

    def __init__(self, sentences: List[str], merge_factor=1000, block_size: int = DEFAULT_BLOCK_SIZE):
        self.merges = self._seed_merges(sentences)
        self._bpe(sentences, merge_factor)
        self.merges_reversed = {i: seq for seq, i in self.merges.items()}
        self.block_size = block_size

    def _count_bigrams(
        self, chunks: List[LinkedList], weights: collections.Counter
    ) -> collections.Counter:
        counter: collections.Counter = collections.Counter()
        for chunk_i in range(len(chunks)):
            chunk = chunks[chunk_i]
            if not chunk.next:
                continue
            for c1, c2 in zip(chunk, chunk.next):
                counter[(c1.contents, c2.contents)] += weights[chunk.cached_full]
        return counter

    def _merge(
        self,
        chunks: List[LinkedList],
        bigram: Tuple[str, str],
        cnts: collections.Counter,
        weights: collections.Counter,
    ):
        for chunk in chunks:
            head = chunk
            prev = None
            while head:
                if not head.next:
                    break
                c1, c2 = head.contents, head.next.contents
                if (c1, c2) == bigram:
                    head.merge()
                    weight = weights[
                            chunk.cached_full
                        ]
                    cnts[bigram] -= weight
                    if prev:
                        cnts[(prev.contents, head.contents)] += weight
                        cnts[(prev.contents, c1)] -= weight
                    if head.next:
                        cnts[(head.contents, head.next.contents)] += weight
                        cnts[(c2, head.next.contents)] -= weight
                prev = head
                head = head.next

    def _seed_merges(self, sentences: List[str]) -> Dict[str, int]:
        out = dict(TOKEN_MAPPING)
        for sent in sentences:
            for c in sent:
                if c not in out:
                    out[c] = len(out)
        return out

    def _count_occurences(self, splits: List[List[str]]) -> collections.Counter:
        counter: collections.Counter = collections.Counter()
        for sent in splits:
            for chunk in sent:
                counter[chunk] += 1
        return counter

    def _bpe(self, sentences: List[str], vocab_size: int):
        splits = [re.findall(self.PRE_TOKENIZATION_REGEX, s) for s in sentences]
        weights = self._count_occurences(splits)
        chunks = list(weights.keys())
        lls: List[LinkedList] = []
        for chunk in chunks:
            lls.append(LinkedList.create(chunk))
        cnts = self._count_bigrams(lls, weights)
        for _ in range(vocab_size):
            top = cnts.most_common(1)
            # Nothing left to merge
            if not top or not top[0][1]:
                return
            top_tup = top[0][0]
            self.merges["".join(top_tup)] = len(self.merges)
            self._merge(lls, top_tup, cnts, weights)

    def encode(self, string: str) -> List[int]:
        chunks = re.findall(self.PRE_TOKENIZATION_REGEX, string)
        for i in range(len(chunks)):
            chunks[i] = LinkedList.create(chunks[i])
        out = []
        for chunk in chunks:
            head = chunk
            while head:
                if not head.next:
                    break
                c1, c2 = head.contents, head.next.contents
                if (c1 + c2) in self.merges:
                    head.merge()
                head = head.next
        out = [self.merges[BOS]]
        for chunk in chunks:
            for c in chunk:
                out.append(self.merges[c.contents])
        out += [self.merges[EOS]]
        out += [self.merges[PAD]] * (self.block_size - len(out))
        return out

    def decode(self, encoded: List[int]) -> str:
        return "".join(self.merges_reversed[id] for id in encoded)

    def vocab_size(self) -> int:
        return len(self.merges)
