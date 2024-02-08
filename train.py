import torch
import input
from tqdm import tqdm
import encode_decode
import model
import utils
import math

BLOCK_SIZE = 15 #encode_decode.DEFAULT_BLOCK_SIZE
TRAIN_FRAC = 0.9
SAMPLE_CAP = 3
NUM_EPOCHS = 50
N_EMB = 256
N_LAYER = 3
N_HEAD = 8
BATCH_SIZE = 8
LEARNING_RATE = 0.001

def train(model, data, opt):
    loss_total = 0
    cnt = 0
    model.train()
    with tqdm(data, unit="batch") as batch_iter:
      for en, jp, label in batch_iter:
          print(f"{en=}")
          print(f"{jp=}")
          print(f"{label=}")
          logits, loss = model(jp, en, label)

          loss_total += loss.item()
          cnt += 1

          opt.zero_grad()
          loss.backward()
          #nn.utils.clip_grad_norm_(model.parameters(), 1)

          opt.step()

    return loss_total /cnt

def validate(model, data):
    loss_total = 0
    cnt = 0
    model.eval()
    with tqdm(data, unit="batch") as batch_iter:
      for en, jp, label in batch_iter:
          logits, loss = model(jp, en, label)

          loss_total += loss.item()
          cnt += 1

    return loss_total /cnt


def main():
    device = utils.choose_device()
    print(f"Using {device}")
    sentences = input.load()
    print(f"Loaded {len(sentences)} sentences")
    jp = [sentence[0] for sentence in sentences]
    en = [sentence[1] for sentence in sentences]

    # Cache to avoid redoing bpe each run.
    jp_enc = encode_decode.JPEncoderDecoder.tryload("output/jp_enc")
    en_enc = encode_decode.ENEncoderDecoder.tryload("output/en_enc")
    if not jp_enc:
        jp_enc = encode_decode.JPEncoderDecoder(jp, block_size=BLOCK_SIZE)
        jp_enc.write("output/jp_enc")
        print("Cached jp encoder state")
    if not en_enc:
        en_enc = encode_decode.ENEncoderDecoder(en, block_size=BLOCK_SIZE)
        en_enc.write("output/en_enc")
        print("Cached en encoder state")

    n = SAMPLE_CAP if SAMPLE_CAP > 0 else len(en)
    train_n = int(n * TRAIN_FRAC)
    assert n > 0
    assert train_n > 0
    train_dataset = input.SentencesDataset(
        en[:train_n], en_enc, jp[:train_n], jp_enc, device, BLOCK_SIZE
    )
    test_dataset = input.SentencesDataset(
        en[train_n:n], en_enc, jp[train_n:n], jp_enc, device, BLOCK_SIZE
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    mc = model.ModelConfig(
        jp_enc.vocab_size(),
        en_enc.vocab_size(),
        device,
        n_emb=N_EMB,
        n_layer=N_LAYER,
        n_heads=N_HEAD,
        block_size=BLOCK_SIZE,
    )
    mc.write("output/config")

    m = model.Model(mc)
    md = m.to(device)

    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    opt = torch.optim.AdamW(md.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_losses = []

    best = math.inf
    for epoch in range(NUM_EPOCHS):
        print("Start epoch", epoch + 1)
        with utils.runtime() as f:
            train_loss = train(md,train_dataloader, opt)
            train_time = f.curr()
        with utils.runtime() as f:
            test_loss = validate(md, test_dataloader)
            test_time = f.curr()
        print(f"Finished epoch {epoch + 1}. Train time: {train_time} Train loss: {train_loss} Test time: {test_time} Test loss {test_loss}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if test_loss < best:
            torch.save(md.state_dict(), 'output/model.pt')



if __name__ == "__main__":
    main()
