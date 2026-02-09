import numpy as np
from brain import NeuralNet

pairs = []
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "=>" in line:
            q, a = line.strip().split("=>")
            pairs.append((q.strip(), a.strip()))

assert len(pairs) > 0

vocab = sorted(set("".join(q for q, _ in pairs)))
char_to_ix = {c: i for i, c in enumerate(vocab)}

def encode(text):
    x = np.zeros((1, len(vocab)))
    for c in text:
        if c in char_to_ix:
            x[0, char_to_ix[c]] += 1
    return x

net = NeuralNet(len(vocab), 32, len(pairs))
lr = 0.01

for epoch in range(1000):
    for i, (q, a) in enumerate(pairs):
        x = encode(q)
        y = np.zeros((1, len(pairs)))
        y[0, i] = 1

        out, hidden = net.forward(x)

        error = out - y

        net.w2 -= lr * hidden.T @ error
        net.b2 -= lr * error

        hidden_error = error @ net.w2.T
        net.w1 -= lr * x.T @ hidden_error
        net.b1 -= lr * hidden_error

np.savez("model.npz", w1=net.w1, b1=net.b1, w2=net.w2, b2=net.b2)
print("Training complete.")
