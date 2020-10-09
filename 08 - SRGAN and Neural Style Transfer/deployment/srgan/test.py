import torch
from super_resolution import Generator, ResidualBlock, UpsampleBLock

def lm(path):
    m = torch.load(path)
    m = m.eval()
    return m


k = lm('srgan.pt')
print(k)
