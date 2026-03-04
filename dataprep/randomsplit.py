import random

random.seed(42)

with open("data\mlm\mlm.txt") as f:
    lines = [l.strip() for l in f if l.strip()]

lines = list(set(lines))
random.shuffle(lines)

n = len(lines)
train = lines[:int(0.7*n)]
val = lines[int(0.7*n):int(0.85*n)]
test = lines[int(0.85*n):]

def write_lines(path, arr):
    with open(path, "w", encoding="utf-8") as f:
        for s in arr:
            f.write(s.strip() + "\n")

write_lines("data/mlm/train.txt", train)
write_lines("data/mlm/val.txt", val)
write_lines("data/mlm/test.txt", test)


print(len(train), len(val), len(test))