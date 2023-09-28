import torch
import torch.nn.functional as F
import numpy as np
words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
stoi["."] = 0
itos[0] = "."
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("Number of examples:", num)
W = torch.randn((27, 27), requires_grad=True)
for k in range(1000):
    xenc = F.one_hot(xs, num_classes=27).float()
    # forward pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.1*(W**2).mean()
    # backward pass
    W.grad = None
    loss.backward()
    # update
    W.data += -50*W.grad
print("Final Loss", loss.item())
for i in range(20):
    ix = 0
    out = []
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts/counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
