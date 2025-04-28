import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# ---------- CVaR (Superquantile) ------------
def superquantile(losses, alpha=0.8):
    q = torch.quantile(losses, alpha)
    tail = losses[losses >= q]
    return tail.mean() if tail.numel()>0 else q

# -------- Data Loaders ---------------------
def get_imbalanced_mnist(train=True, batch_size=32, imbalance_ratio=0.1, minority_classes=[0]):
    ds = datasets.MNIST(root='./data/MNIST', train=train, download=False,
                        transform=transforms.ToTensor())
    targets = ds.targets
    idxs = []
    for c in range(10):
        mask = (targets==c).nonzero(as_tuple=True)[0]
        if c in minority_classes:
            k = int(len(mask)*imbalance_ratio)
            sel = mask[torch.randperm(len(mask))[:k]]
            idxs.append(sel)
        else:
            idxs.append(mask)
    subset = Subset(ds, torch.cat(idxs))
    return DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=0, drop_last=False)

def get_celeba_loader(root='./data/celeba_data', batch_size=32):
    tf = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    celeba = datasets.CelebA(root=root, split='all', download=False, transform=tf)
    return DataLoader(celeba, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# ---------- VAE Model & Training ------------
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3  = nn.Linear(latent_dim, hidden_dim)
        self.fc4  = nn.Linear(hidden_dim, input_dim)
    def encode(self,x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)
    def reparam(self,mu,logvar):
        std = torch.exp(0.5*logvar); eps=torch.randn_like(std)
        return mu+eps*std
    def decode(self,z):
        h = F.relu(self.fc3(z)); return torch.sigmoid(self.fc4(h))
    def forward(self,x):
        mu,logvar=self.encode(x); z=self.reparam(mu,logvar)
        return self.decode(z), mu, logvar

def train_standard_vae(model, loader, optimizer, device):
    model.train(); total=0
    for data,_ in loader:
        x = data.view(-1,784).to(device)
        optimizer.zero_grad()
        recon,mu,logvar=model(x)
        recon_loss = F.binary_cross_entropy(recon,x,reduction='mean')*784
        kl = (-0.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(1)).mean()
        (recon_loss+kl).backward(); optimizer.step()
        total += (recon_loss+kl).item()
    return total/len(loader)

def eval_standard_vae(model, loader, device):
    model.eval(); total=0
    with torch.no_grad():
        for data,_ in loader:
            x = data.view(-1,784).to(device)
            recon,mu,logvar=model(x)
            recon_loss = F.binary_cross_entropy(recon,x,reduction='mean')*784
            kl = (-0.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(1)).mean()
            total += (recon_loss+kl).item()
    return total/len(loader)

def train_cvar_vae(model, loader, optimizer, device, alpha=0.8):
    model.train(); total=0
    for data,_ in loader:
        x = data.view(-1,784).to(device)
        optimizer.zero_grad()
        recon,mu,logvar=model(x)
        per = F.binary_cross_entropy(recon,x,reduction='none').mean(1)*784
        cvar = superquantile(per, alpha)
        kl = (-0.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(1)).mean()
        (cvar+kl).backward(); optimizer.step()
        total += (cvar+kl).item()
    return total/len(loader)

def eval_cvar_vae(model, loader, device, alpha=0.8):
    model.eval(); total=0
    with torch.no_grad():
        for data,_ in loader:
            x = data.view(-1,784).to(device)
            recon,mu,logvar=model(x)
            per = F.binary_cross_entropy(recon,x,reduction='none').mean(1)*784
            cvar = superquantile(per, alpha)
            kl = (-0.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(1)).mean()
            total += (cvar+kl).item()
    return total/len(loader)

# ---------- GAN Models & Training ------------
class Generator32(nn.Module):
    def __init__(self, latent_dim=100, base_ch=128, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_ch*2, 4,1,0,bias=False),
            nn.BatchNorm2d(base_ch*2), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch*2, base_ch,4,2,1,bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch, base_ch//2,4,2,1,bias=False),
            nn.BatchNorm2d(base_ch//2), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch//2, channels,4,2,1,bias=False),
            nn.Tanh()
        )
    def forward(self,z):
        return self.net(z.view(z.size(0),z.size(1),1,1))

class Discriminator32(nn.Module):
    def __init__(self, base_ch=128, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, base_ch//2,4,2,1,bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(base_ch//2, base_ch,4,2,1,bias=False),
            nn.BatchNorm2d(base_ch), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base_ch, base_ch*2,4,2,1,bias=False),
            nn.BatchNorm2d(base_ch*2), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base_ch*2,1,4,1,0,bias=False)
        )
    def forward(self,x):
        return self.net(x).view(-1)

def train_epoch_standard_gan(G,D,loader,optG,optD,device):
    G.train(); D.train()
    d_sum=g_sum=0
    for real,_ in loader:
        real=real.to(device); bs=real.size(0)
        # D
        optD.zero_grad()
        fake=G(torch.randn(bs,G.latent_dim,device=device)).detach()
        out_r=D(real); out_f=D(fake)
        loss_d=0.5*(F.binary_cross_entropy_with_logits(out_r,torch.ones_like(out_r))+
                    F.binary_cross_entropy_with_logits(out_f,torch.zeros_like(out_f)))
        loss_d.backward(); optD.step(); d_sum+=loss_d.item()
        # G
        optG.zero_grad()
        out= D(G(torch.randn(bs,G.latent_dim,device=device)))
        loss_g=F.binary_cross_entropy_with_logits(out,torch.ones_like(out))
        loss_g.backward(); optG.step(); g_sum+=loss_g.item()
    return d_sum/len(loader), g_sum/len(loader)



def train_epoch_cvar_gan(G, D, loader, optG, optD, device, alpha=0.8):
    """
    One epoch of CVaR‐GAN training: applies CVaR loss to both D and G.
    Returns:
        (mean_discriminator_loss, mean_generator_loss)
    """
    G.train()
    D.train()
    d_losses = []
    g_losses = []

    for real, _ in loader:
        real = real.to(device)
        bs   = real.size(0)

        # --------------------
        # Discriminator step
        # --------------------
        optD.zero_grad()
        # Draw fake samples
        z    = torch.randn(bs, G.latent_dim, device=device)
        fake = G(z).detach()
        # Compute logits
        out_r = D(real)
        out_f = D(fake)
        # Per‐sample BCE losses
        l_r = F.binary_cross_entropy_with_logits(
            out_r,
            torch.ones_like(out_r),
            reduction='none'
        )
        l_f = F.binary_cross_entropy_with_logits(
            out_f,
            torch.zeros_like(out_f),
            reduction='none'
        )
        # CVaR over the joint real+fake losses
        loss_d = superquantile(torch.cat([l_r, l_f], dim=0), alpha)
        loss_d.backward()
        optD.step()
        d_losses.append(loss_d.item())

        # --------------------
        # Generator step
        # --------------------
        optG.zero_grad()
        z2   = torch.randn(bs, G.latent_dim, device=device)
        out_g = D(G(z2))
        l_g   = F.binary_cross_entropy_with_logits(
            out_g,
            torch.ones_like(out_g),
            reduction='none'
        )
        loss_g = superquantile(l_g, alpha)
        loss_g.backward()
        optG.step()
        g_losses.append(loss_g.item())

    return np.mean(d_losses), np.mean(g_losses)