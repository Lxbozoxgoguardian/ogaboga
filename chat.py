import numpy as np
from brain import NeuralNet

pairs = []
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "=>" in line:
            q, a = line.strip().split("=>")
            pairs.append((q.strip(), a.strip()))

vocab = sorted(set("".join(q for q, _ in pairs)))
char_to_ix = {c: i for i, c in enumerate(vocab)}

def encode(text):
    x = np.zeros((1, len(vocab)))
    for c in text:
        if c in char_to_ix:
            x[0, char_to_ix[c]] += 1
    return x

data = np.load("model.npz")
net = NeuralNet(len(vocab), 32, len(pairs))
net.w1 = data["w1"]
net.b1 = data["b1"]
net.w2 = data["w2"]
net.b2 = data["b2"]

print("Compiler-safe AI ready.")

while True:
    msg = input("You: ")
    if msg.lower() == "exit":
        break

    x = encode(msg)
    out, _ = net.forward(x)
    i = int(np.argmax(out))
    print("AI:", pairs[i][1])
