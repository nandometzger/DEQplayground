import torch
import torch.nn as nn

class TanhFixedPointLayer(nn.Module):
    def __init__(self, out_features, tol = 1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter
  
    def forward(self, x):
        # initialize output z to be zero
        z = torch.zeros_like(x)
        self.iterations = 0

        # iterate until convergence
        while self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z - z_next)
            z = z_next
            self.iterations += 1
            if self.err < self.tol:
                break
        return z

layer = TanhFixedPointLayer(50, tol=1e-2)
X = torch.randn(10,50)
Z = layer(X)
print(f"Terminated after {layer.iterations} iterations with error {layer.err}")