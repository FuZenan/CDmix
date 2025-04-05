import torch

# 修正了x1的创建方式，使其符合torch.tensor的参数要求


def my_cdist(x1, x2):

    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1),
                      x1,
                      x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30), x1_norm,x2_norm,x2_norm.transpose(-2, -1),x1.transpose(-2, -1)


x1 = torch.tensor([[1, 2, 3, 4], [2, 2, 2, 2], [5, 8, 0, 2], [1, 1, 1, 1]], dtype=torch.float32)
x2 = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=torch.float32)

try:
    result, x1_norm ,x2_norm ,p,q= my_cdist(x1, x2)
    print(x1_norm)
    print(x2_norm)
    print(p)
    print(q)
    print(result)

except ValueError as e:
    print(f"错误: {e}")

