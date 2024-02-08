from typing import List
import model
import sys
import utils
import torch
import torch.nn.functional as F
import encode_decode


def generate_response(m: model.Model, jp: List[int], config: model.ModelConfig):
    jp_t = torch.tensor([jp], device=config.device)
    idx = torch.tensor([[encode_decode.BOS_I]], dtype=torch.long, device=config.device)
    # idx is (B, T) array of indices in the current context
    for _ in range(config.block_size):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -config.block_size :]
        # get the predictions
        logits, loss = m(jp_t, idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        # exit if we got an EOS token.
        if idx_next[0][0].item() == encode_decode.EOS_I:
            break

    return idx


def translate(
    m: model.Model,
    jp: str,
    config: model.ModelConfig,
    en_enc: encode_decode.ENEncoderDecoder,
    jp_enc: encode_decode.JPEncoderDecoder,
):
    encoded_jp = jp_enc.encode(jp)
    translated = en_enc.decode(generate_response(m, encoded_jp, config).tolist()[0])
    return translated


def main():
    device = utils.choose_device()
    print(f"Using {device}")

    config = model.ModelConfig.tryload("output/config")
    if not config:
        print("Failed to load config. Have you run train.py?")
        sys.exit(1)
    config.device = device
    print("Loaded config")
    jp_enc = encode_decode.JPEncoderDecoder.tryload("output/jp_enc")
    en_enc = encode_decode.ENEncoderDecoder.tryload("output/en_enc")
    if not jp_enc or not en_enc:
        print("Failed to load an encoder")
        sys.exit(1)

    m = model.Model(config).to(device)
    m.load_state_dict(torch.load("output/model.pt"))
    m.eval()

    while sent := input("Type japanese sentence:"):
        translated = translate(m, sent, config, en_enc, jp_enc)
        print(f"{sent} -> {translated}")


if __name__ == "__main__":
    main()
