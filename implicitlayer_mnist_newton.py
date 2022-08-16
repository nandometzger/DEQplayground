# import the MNIST dataset and data loaders
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
from torch.autograd import gradcheck
from tqdm import tqdm

from torchvision import datasets, transforms

mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol = 1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter
  
    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:]
                # z = z - torch.solve(g[:,:,None], J)[0][:,:,0]
                z = z - torch.linalg.solve(J,g[:,:,None])[:,:,0]
                self.iterations += 1
    
        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        # z.register_hook(lambda grad : torch.solve(grad[:,:,None], J.transpose(1,2))[0][:,:,0])
        z.register_hook(lambda grad : torch.linalg.solve(J.transpose(1,2), grad[:,:,None])[:,:,0])
        return z

# a generic function for running a single epoch (training or evaluation)
def epoch(loader, model, opt=None, monitor=None):
    total_loss, total_err, total_monitor = 0.,0.,0.
    model.eval() if opt is None else model.train()
    for X,y in tqdm(loader, leave=False):
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            if sum(torch.sum(torch.isnan(p.grad)) for p in model.parameters()) == 0:
              opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), total_monitor / len(loader)

layer = TanhNewtonImplicitLayer(5, tol=1e-10).double()
gradcheck(layer, torch.randn(3, 5, requires_grad=True, dtype=torch.double), check_undefined_grad=False)

torch.manual_seed(0)
mode = "DEQ"
if mode=="DEQ":
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 100),
                        TanhNewtonImplicitLayer(100, max_iter=40),
                        nn.Linear(100, 10) ).to(device)
    monitor = lambda x : x[2].iterations
else:
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 100), nn.Linear(100, 100), nn.Linear(100, 10) ).to(device)
    monitor = None
opt = optim.SGD(model.parameters(), lr=1e-1)

for i in range(10):
    if i == 5:
        opt.param_groups[0]["lr"] = 1e-2

    train_err, train_loss, train_fpiter = epoch(train_loader, model, opt, monitor=monitor)
    test_err, test_loss, test_fpiter = epoch(test_loader, model, monitor=monitor)
    print(f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, Newton Iters: {train_fpiter:.2f} | " +
          f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}, Newton Iters: {test_fpiter:.2f}")