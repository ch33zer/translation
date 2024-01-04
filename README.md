# Japanese to English translation

This  is a toy Japanese to English translater. It's based on the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). It's very early days and is not suitable for anything but experimentation.

## Runing

Just open [translation.ipynb](translation.ipynb) and run the steps. You will almost certainly need a GPU to train at all efficiently. There's a trained model stored in `model.pt` you can load if you don't want to full trained.

## Sample translations

```
input="私は一日に100ユーロ稼ぎます。"
translated='BOSI am to earn to earnest under a day.EOS'
expected='I make 100 euros per day.'

input="すぐに諦めて昼寝をするかも知れない。"
translated='BOSWe want to keep to sleep on your day after lunch.EOS'
expected='I may give up soon and just nap instead.'

input="そんなことは起きないでしょう。"
translated="BOSDon't get it up so that.EOS"
expected="That sort of thing won't happen."
```
As you can see there are significant errors and issues with the translations. We probably need:

* More training. This was only 100000 training steps with a batch size of 16. You'd need a lot more to truly understand the structure of the two languages.
* More optimized architecture. This is absically the dumbest least optimized way to run this model. Using more optimized transformer pieces (like those provided by pytorch) would be better
* More training data. The data came from [https://tatoeba.org/en/](https://tatoeba.org/en/) which is of unknown quality (and my Japanese is beginner so I can't even verify the quality).
