from build.gdel3d import Del
import torch

N = 100000
d = Del(N)
points = torch.rand(N, 3)
tets = d.compute(points)
print(tets.min(), tets.max())
