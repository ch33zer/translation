from typing import List, Tuple
import csv
import io
import torch
from torch.utils.data import Dataset
import encode_decode


def load() -> List[Tuple[str, str]]:
    sentences = []
    with open("jp_en_sentences.tsv", "r", encoding="utf-8-sig") as f:
        raw = f.read()
        raw = raw.replace("”", '"')
        raw = raw.replace("“", '"')
        rd = csv.DictReader(io.StringIO(raw), delimiter="\t")
        for row in rd:
            sentences.append((row["jp"], row["en"]))
    return sentences


class SentencesDataset(Dataset):
    def __init__(
        self,
        en: List[str],
        en_enc: encode_decode.ENEncoderDecoder,
        jp: List[str],
        jp_enc: encode_decode.JPEncoderDecoder,
        device: torch.device,
        block_size=encode_decode.DEFAULT_BLOCK_SIZE
    ):
        assert len(en) == len(jp), f"Sizes must match! Found {len(en)} and {len(jp)}"
        self.en = en
        self.jp = jp
        self.en_enc = en_enc
        self.jp_enc = jp_enc
        self.device = device
        self.block_size = block_size
        too_long_en = 0
        too_long_jp = 0
        for en, jp in zip(self.en, self.jp):
            if len(en) > self.block_size:
                too_long_en += 1
            if len(jp) > self.block_size:
                too_long_jp += 1
        if too_long_en:
            print(f"WARNING: Found {too_long_en} English sentences longer than the block size!")
        if too_long_jp:
            print(f"WARNING: Found {too_long_jp} Japanese sentences longer than the block size!")


    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        raw_en = self.en[idx]
        raw_jp = self.jp[idx]
        # -2 because we will always add 2 chars: BOS and EOS. Usually for EN the byte pair encoding will make the encoded rep shorter so we don't need
        # to worry about this, but we do it for safety.
        clipped_en = raw_en[:self.block_size-2]
        clipped_jp = raw_jp[:self.block_size-2]
        en_encoded = self.en_enc.encode(clipped_en)
        jp_encoded = self.jp_enc.encode(clipped_jp)
        en_t = torch.tensor(en_encoded)
        jp_t = torch.tensor(jp_encoded)
        label = torch.zeros_like(en_t)
        label[:-1] = en_t[1:]
        label[-1] = encode_decode.PAD_I
        #print(idx)
        #print(" ", raw_en, len(raw_en), clipped_en, len(clipped_en), en_encoded, len(en_encoded), en_t, en_t.shape)
        #print(" ", raw_jp, len(raw_jp), clipped_jp, len(clipped_jp), jp_encoded, len(jp_encoded), jp_t, jp_t.shape)
        return en_t.to(self.device), jp_t.to(self.device), label.to(self.device)

