import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



def train_discriminator_ensemble(model, optimizer, expert_obs, policy_obs, lambda_gp=10.0, device="cpu", batch_size=512):
    model.train()
    total_size = expert_obs.shape[0]
    indices = np.random.permutation(total_size)
    num_batches = int(np.ceil(total_size / batch_size))

    loss_total = 0.0

    for i in range(num_batches):
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        expert_batch = torch.tensor(expert_obs[batch_indices], dtype=torch.float32, device=device)
        policy_batch = policy_obs[batch_indices].to(dtype=torch.float32, device=device)

        alpha = torch.rand(len(batch_indices), 1, 1, device=device)
        interpolated = alpha * expert_batch + (1 - alpha) * policy_batch
        interpolated.requires_grad_()
        with torch.backends.cudnn.flags(enabled=False):
            expert_preds = model(expert_batch)
            policy_preds = model(policy_batch)
            interpolated_preds = model(interpolated)

        batch_loss = 0.0
        for j in range(model.num_heads):
            d_expert = expert_preds[j]
            d_policy = policy_preds[j]
            d_interp = interpolated_preds[j]

            loss_expert = torch.mean(F.relu(1.0 - d_expert))
            loss_policy = torch.mean(F.relu(1.0 + d_policy))
            loss_head = loss_expert + loss_policy

            gradients = torch.autograd.grad(
                outputs=d_interp,
                inputs=interpolated,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradients = gradients.reshape(len(batch_indices), -1)
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            batch_loss += loss_head + lambda_gp * gp

        batch_loss /= model.num_heads
        batch_loss.backward()
        loss_total += batch_loss.item()

    optimizer.step()
    optimizer.zero_grad()
    return loss_total / num_batches


class EnsembleDiscriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_heads=16):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.gru = nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(self.num_heads)])

    def forward(self, obs):
        # with torch.backends.cudnn.flags(enabled=False):  # <- Disable CuDNN for double backward
        _, h_n = self.gru(obs)
        z = h_n.squeeze(0)
        shared = self.shared_mlp(z)
        outputs = [head(shared).squeeze(-1) for head in self.heads]
        return outputs
