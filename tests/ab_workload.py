import torch, time
A = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda:0")
B = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda:0")
end = time.time() + 30
c = 0
while time.time() < end:
    _ = A @ B
    torch.cuda.synchronize()
    c += 1
print(f"{(c * 2 * 8192**3) / 30 / 1e12:.1f} tok/s")
