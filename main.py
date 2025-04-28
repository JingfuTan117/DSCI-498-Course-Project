import argparse
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from aux_1 import (
    get_imbalanced_mnist, get_celeba_loader,
    VAE, train_standard_vae, eval_standard_vae,
    train_cvar_vae, eval_cvar_vae,
    Generator32, Discriminator32,
    train_epoch_standard_gan, train_epoch_cvar_gan
)

def run_vae(alpha, epochs, device):
    # MNIST VAE experiment
    train_loader = get_imbalanced_mnist(train=True)
    test_loader  = get_imbalanced_mnist(train=False)
    model_std = VAE().to(device)
    opt_std   = torch.optim.Adam(model_std.parameters(), lr=1e-4)
    model_cvar = VAE().to(device)
    opt_cvar   = torch.optim.Adam(model_cvar.parameters(), lr=1e-4)

    train_std, test_std   = [], []
    train_cvar, test_cvar = [], []

    for ep in range(1, epochs+1):
        ts = train_standard_vae(model_std, train_loader, opt_std, device)
        vs = eval_standard_vae(   model_std, test_loader,  device)
        tc = train_cvar_vae(      model_cvar, train_loader, opt_cvar, device, alpha)
        vc = eval_cvar_vae(       model_cvar, test_loader,  device, alpha)

        train_std.append(ts);   test_std.append(vs)
        train_cvar.append(tc);  test_cvar.append(vc)

        print(f"[VAE] Ep{ep:03d} | Std T={ts:.2f} E={vs:.2f} | CVaR T={tc:.2f} E={vc:.2f}")

    # plot train/test losses and save
    plt.figure()
    plt.plot(train_std, label='Std Train')
    plt.plot(test_std,  label='Std Test')
    plt.plot(train_cvar, '--', label='CVaR Train')
    plt.plot(test_cvar, '--', label='CVaR Test')
    plt.legend()
    plt.title('VAE Train/Test Loss')
    plt.savefig('vae_loss_curves.png')
    plt.close()

    # plot loss distribution (reconstruction BCE per sample) and save
    std_losses, cvar_losses = [], []
    model_std.eval(); model_cvar.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            x = data.view(data.size(0), -1).to(device)
            recon_std, _, _ = model_std(x)
            recon_cvar, _, _ = model_cvar(x)
            per_std  = F.binary_cross_entropy(recon_std,  x, reduction='none').mean(dim=1).cpu().numpy()
            per_cvar = F.binary_cross_entropy(recon_cvar, x, reduction='none').mean(dim=1).cpu().numpy()
            std_losses.extend(per_std)
            cvar_losses.extend(per_cvar)
    std_losses  = np.array(std_losses)
    cvar_losses = np.array(cvar_losses)
    p90_std  = np.percentile(std_losses, 90)
    p90_cvar = np.percentile(cvar_losses, 90)

    plt.figure()
    plt.hist(std_losses,  bins=50, alpha=0.5, label='VAE Std')
    plt.hist(cvar_losses, bins=50, alpha=0.5, label='VAE CVaR')
    plt.axvline(p90_std,  color='C0', linestyle='--', label='Std 90th %ile')
    plt.axvline(p90_cvar, color='C1', linestyle='--', label='CVaR 90th %ile')
    plt.legend()
    plt.title('VAE Loss Distribution')
    plt.savefig('vae_loss_distribution.png')
    plt.close()

    # sample and save
    with torch.no_grad():
        z = torch.randn(64, model_std.latent_dim, device=device)
        imgs_std  = model_std.decode(z).cpu().view(-1,1,28,28)
        imgs_cvar = model_cvar.decode(z).cpu().view(-1,1,28,28)

    grid1 = vutils.make_grid(imgs_std,  nrow=8, normalize=True)
    grid2 = vutils.make_grid(imgs_cvar, nrow=8, normalize=True)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(grid1.permute(1,2,0))
    plt.title('VAE Std')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(grid2.permute(1,2,0))
    plt.title('VAE CVaR')
    plt.axis('off')
    plt.savefig('vae_samples.png')
    plt.close()



def run_gan(alpha, epochs, device):
    import numpy as np
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    # 1) Data: down‐sample 5k from CelebA, center‐crop 178→resize 32
    tf = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    full = datasets.CelebA(root='./data/celeba_data', split='all',
                           download=False, transform=tf)
    idxs   = torch.randperm(len(full))[:5000]
    subset = Subset(full, idxs)
    loader = DataLoader(subset, batch_size=32,
                        shuffle=True, num_workers=0, drop_last=True)

    # 2) Models & Opts
    G_std, D_std   = Generator32().to(device), Discriminator32().to(device)
    G_cvar, D_cvar = Generator32().to(device), Discriminator32().to(device)
    optG_s = torch.optim.Adam(G_std.parameters(),  lr=2e-4, betas=(0.5,0.999))
    optD_s = torch.optim.Adam(D_std.parameters(),  lr=2e-4, betas=(0.5,0.999))
    optG_c = torch.optim.Adam(G_cvar.parameters(), lr=2e-4, betas=(0.5,0.999))
    optD_c = torch.optim.Adam(D_cvar.parameters(), lr=2e-4, betas=(0.5,0.999))

    # 3) Train & record
    losses_std  = {'d':[], 'g':[]}
    losses_cvar = {'d':[], 'g':[]}
    for ep in range(1, epochs+1):
        ds, gs = train_epoch_standard_gan(G_std, D_std, loader,
                                          optG_s, optD_s, device)
        dc, gc = train_epoch_cvar_gan(    G_cvar, D_cvar, loader,
                                          optG_c, optD_c, device, alpha)
        losses_std['d'].append(ds);  losses_std['g'].append(gs)
        losses_cvar['d'].append(dc); losses_cvar['g'].append(gc)
        print(f"[GAN] Ep{ep:03d} | Std D={ds:.3f} G={gs:.3f} "
              f"| CVaR D={dc:.3f} G={gc:.3f}")

    # 4) Plot loss curves
    plt.figure()
    plt.plot(losses_std['d'],  label='D Std')
    plt.plot(losses_cvar['d'], label='D CVaR')
    plt.plot(losses_std['g'],  label='G Std')
    plt.plot(losses_cvar['g'], label='G CVaR')
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("GAN Training Curves")
    plt.legend(); plt.savefig('gan_loss_curves.png'); plt.close()

    # 5) Plot D‐loss distribution
    std_dist, cvar_dist = [], []
    D_std.eval(); D_cvar.eval(); G_std.eval(); G_cvar.eval()
    with torch.no_grad():
        for real, _ in loader:
            real = real.to(device)
            bs   = real.size(0)

            # vanilla
            out_r_s = D_std(real)
            out_f_s = D_std(G_std(torch.randn(bs, G_std.latent_dim,
                                              device=device)))
            l_r_s = F.binary_cross_entropy_with_logits(
                        out_r_s, torch.ones_like(out_r_s),  reduction='none')
            l_f_s = F.binary_cross_entropy_with_logits(
                        out_f_s, torch.zeros_like(out_f_s), reduction='none')
            std_dist.extend(torch.cat([l_r_s, l_f_s], dim=0).cpu().numpy())

            # CVaR
            out_r_c = D_cvar(real)
            out_f_c = D_cvar(G_cvar(torch.randn(bs, G_cvar.latent_dim,
                                                device=device)))
            l_r_c = F.binary_cross_entropy_with_logits(
                        out_r_c, torch.ones_like(out_r_c),  reduction='none')
            l_f_c = F.binary_cross_entropy_with_logits(
                        out_f_c, torch.zeros_like(out_f_c), reduction='none')
            cvar_dist.extend(torch.cat([l_r_c, l_f_c], dim=0).cpu().numpy())

    std_arr  = np.array(std_dist)
    cvar_arr = np.array(cvar_dist)
    p90_s    = np.percentile(std_arr, 90)
    p90_c    = np.percentile(cvar_arr, 90)

    plt.figure()
    plt.hist(std_arr,  bins=50, alpha=0.5, label='Std GAN')
    plt.hist(cvar_arr, bins=50, alpha=0.5, label='CVaR GAN')
    plt.axvline(p90_s, color='C0', linestyle='--', label='Std 90th')
    plt.axvline(p90_c, color='C1', linestyle='--', label='CVaR 90th')
    plt.xlabel("Discriminator Loss"); plt.ylabel("Count")
    plt.title("GAN Loss Distribution")
    plt.legend(); plt.savefig('gan_loss_distribution.png'); plt.close()

    # 6) Final sample grid
    with torch.no_grad():
        z = torch.randn(64, G_std.latent_dim, device=device)
        imgs_s = G_std(z).cpu()
        imgs_c = G_cvar(z).cpu()
    g1 = vutils.make_grid(imgs_s, nrow=8, normalize=True)
    g2 = vutils.make_grid(imgs_c, nrow=8, normalize=True)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(g1.permute(1,2,0)); plt.title('GAN Std'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(g2.permute(1,2,0)); plt.title('GAN CVaR'); plt.axis('off')
    plt.savefig('gan_samples.png'); plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["vae","gan"], help="model type")
    parser.add_argument("--dataset", choices=["mnist","celeba"], required=True)
    parser.add_argument("--alpha",   type=float, default=0.8, help="CVaR alpha")
    parser.add_argument("--epochs",  type=int,   default=20,   help="number of epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "vae":
        if args.dataset != "mnist":
            raise ValueError("VAE is only implemented on MNIST")
        run_vae(args.alpha, args.epochs, device)
    else:
        if args.dataset != "celeba":
            raise ValueError("GAN is only implemented on CelebA")
        run_gan(args.alpha, args.epochs, device)


