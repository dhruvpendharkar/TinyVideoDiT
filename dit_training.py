import torch
import torch.nn as nn
import tqdm.tqdm as tqdm
import torch.optim as optim

def beta_schedule(T, init_beta=1e-4, final_beta=1e-2):
    return torch.linspace(init_beta, final_beta, T)

def alpha_schedules(beta):
    alphas = 1. - beta
    alphas_cum = torch.cumprod(alphas, dim=0)
    alphas_cum_prev = torch.cat([torch.tensor([1.0], device=alphas.device), alphas_cum[:-1]])
    return alphas, alphas_cum, alphas_cum_prev

def unpatchify_video(patched_video, patch_size=(2, 8, 8), frames=16, resolution=(64, 64)):
    b, num_patches, pt, ph, pw, c = patched_video.shape
    fh, fw = resolution
    num_t = frames // pt
    num_h = fh // ph
    num_w = fw // pw

    resh = patched_video.view(b, num_t, num_h, num_w, pt, ph, pw, c)
    permuted = resh.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    unpatchified = permuted.view(b, num_t * pt, num_h * ph, num_w, pw, c)
    return unpatchified





def train_video_dit(model, train_dataloader, val_dataloader, num_epochs=10, T=100, lr=1e-4, device="mps"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    betas = beta_schedule(T)
    alphas, alphas_cum, _ = alpha_schedules(betas)
    alphas = alphas.to(device)
    alphas_cum = alphas_cum.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = batch.to(device)
            batch_size = batch.shape[0]
            t = torch.randint(0, T, (batch_size,), device=device)
            noise = torch.randn_like(batch, device=device)
            curr_alphas_cum = alphas_cum[t].view(batch_size, 1, 1, 1, 1)
            sqrt_term = torch.sqrt(curr_alphas_cum)
            sqrt_alt_term = torch.sqrt(1. - curr_alphas_cum)
            batch_t = sqrt_term * batch + sqrt_alt_term * noise
            pred_noise_patchified = model(batch_t, t)
            pred_noise = unpatchify_video(pred_noise_patchified)

            loss = loss_fn(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(train_dataloader)}")
    
    print("Finished training!")









