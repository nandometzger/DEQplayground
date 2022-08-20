import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import tqdm

# CIFAR10 data loader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.autograd import gradcheck
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNetLayer(torch.nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = torch.nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = torch.nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups, n_channels)
        self.norm3 = torch.nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    Xall = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1],)[:, 1:n+1, 0]   # (bsz x n)
        # alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        Xall.append(X[:,k%m].view_as(x0))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res, torch.cat(Xall,0)


class DEQFixedPoint(torch.nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x, train=True):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res, zall = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = zall.clone().detach().requires_grad_()
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res,zall = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)

        # meta = {
        #     "forward_res": self.forward_res, 
        # }
        return z, zall

# run a very small network with double precision, iterating to high precision
f = ResNetLayer(2,2, num_groups=2).double()
deq = DEQFixedPoint(f, anderson, tol=1e-10, max_iter=50).double()
gradcheck(deq, torch.randn(1,2,3,3).double().requires_grad_(), eps=1e-5, atol=1e-3, check_undefined_grad=False)

f = ResNetLayer(64,128)
deq = DEQFixedPoint(f, anderson, tol=1e-4, max_iter=50, beta=2.0)
X = torch.randn(10,64,32,32)
out = deq(X)[0]
(out*torch.randn_like(out)).sum().backward()


# plt.figure(dpi=150)
# plt.semilogy(deq.forward_res)
# plt.semilogy(deq.backward_res)
# plt.legend(['Forward', 'Backward'])
# plt.xlabel("Iteration")
# plt.ylabel("Residual")


class TinyModel(torch.nn.Module):

    def __init__(self, f, chan):
        super(TinyModel, self).__init__()

        self.pre = model = torch.nn.Sequential(torch.nn.Conv2d(3,chan, kernel_size=3, bias=True, padding=1), torch.nn.BatchNorm2d(chan))
        self.deq = DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5)
        self.post = torch.nn.Sequential( torch.nn.ReLU(), torch.nn.BatchNorm2d(chan),
                      torch.nn.AvgPool2d(8,8), torch.nn.Flatten(),
                      torch.nn.Linear(chan*4*4,10))
    def forward(self, x, train=True):
        inshape = x.shape
        x = self.pre(x)
        if train:
            x = self.deq(x)[1]
            x = self.post(x)
            return x
        else:
            x = self.deq(x)[0]
            x = self.post(x)
            return x
torch.manual_seed(0)
chan = 24
f = ResNetLayer(chan, 64, kernel_size=3)
# model = torch.nn.Sequential(torch.nn.Conv2d(3,chan, kernel_size=3, bias=True, padding=1),
#                       torch.nn.BatchNorm2d(chan),
#                       DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
#                     #   torch.nn.ReLU(),
#                     #   torch.nn.Conv2d(chan,64, kernel_size=3, bias=True, padding=1),
#                     #   torch.nn.ReLU(),
#                     #   torch.nn.Conv2d(64,chan, kernel_size=3, bias=True, padding=1),
#                       torch.nn.ReLU(),
#                       torch.nn.BatchNorm2d(chan),
#                       torch.nn.AvgPool2d(8,8),
#                       torch.nn.Flatten(),
#                       torch.nn.Linear(chan*4*4,10)).to(device)

model = TinyModel(f,chan).to(device) 

# standard training or evaluation loop
def epoch(loader, model, opt=None, lr_scheduler=None, train=True):
    total_loss, total_err, total_iter, final_res = 0.,0.,0.,0.
    model.eval() if opt is None else model.train()
    for X,y in tqdm(loader):
        X,y = X.to(device), y.to(device)
        yp = model(X, train=True)
        if train:
            loss = nn.CrossEntropyLoss()(yp,y.repeat(yp.shape[0]//y.shape[0]))
        else:
            loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
                
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        total_iter += len(model[2].forward_res) *X.shape[0]
        final_res += model[2].forward_res[-1] * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset), total_iter/len(loader.dataset), final_res/len(loader.dataset)


cifar10_train = datasets.CIFAR10(".", train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar10_train, batch_size = 40, shuffle=True, num_workers=0)
test_loader = DataLoader(cifar10_test, batch_size = 40, shuffle=False, num_workers=0)


if __name__ == '__main__':
        
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

    max_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)

    for i in (range(50)):
        print(epoch(train_loader, model, opt, scheduler, train=True))
        print(epoch(test_loader, model)) 

