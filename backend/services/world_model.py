"""Native action-conditioned latent GRU / Gaussian RSSM; no external trainer."""
import torch
from torch import nn
from torch.nn import functional as F
from backend.services.world_data import action_width


class WorldModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        h, z = c['hidden_dim'], c['latent_dim']
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(), nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, h), nn.SiLU())
        self.initial = nn.Linear(h + c['state_dim'], h)
        self.gru = nn.GRUCell(z + action_width(c), h)
        self.prior = nn.Linear(h, z * (2 if c['architecture'] == 'rssm' else 1))
        self.posterior = nn.Linear(h * 2, z * (2 if c['architecture'] == 'rssm' else 1))
        self.decoder = nn.Sequential(nn.Linear(h + z, 32 * 8 * 8), nn.SiLU(), nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(16, 3, 4, 2, 1))
        self.state_head = nn.Linear(h + z, c['state_dim']) if c['state_dim'] else None
        self.reward_head = nn.Linear(h + z, 1)
        self.done_head = nn.Linear(h + z, 1)

    def distribution(self, values):
        mean, raw = values.float().chunk(2, -1)
        return torch.distributions.Normal(mean, F.softplus(raw) + .1)

    def latent(self, values, sample=False):
        if self.config['architecture'] != 'rssm':
            return values, None
        dist = self.distribution(values)
        return dist.rsample() if sample else dist.mean, dist

    def observe(self, frame, state):
        encoded = self.encoder(frame)
        h = torch.tanh(self.initial(torch.cat([encoded, state], -1)))
        z, _ = self.latent(self.posterior(torch.cat([h, encoded], -1)))
        return h, z

    def decode(self, h, z):
        feat = torch.cat([h, z], -1)
        frame = torch.sigmoid(F.interpolate(self.decoder(feat), size=self.config['image_size'], mode='bilinear', align_corners=False))
        return frame, self.state_head(feat) if self.state_head else feat[:, :0], self.reward_head(feat).squeeze(-1), self.done_head(feat).squeeze(-1)

    def rollout(self, frame, state, actions):
        """Only initial observation is visible; future observations cannot enter this path."""
        h, z = self.observe(frame, state)
        outputs = []
        for a in actions.unbind(1):
            h = self.gru(torch.cat([z, a], -1), h)
            z, _ = self.latent(self.prior(h))
            outputs.append(self.decode(h, z))
        return tuple(torch.stack([row[i] for row in outputs], 1) for i in range(4))

    def objective(self, b):
        frames, states, rewards, dones = self.rollout(b['images'][:, 0], b['states'][:, 0], b['actions'])
        image_loss = F.mse_loss(frames.float(), b['images'][:, 1:].float())
        state_loss = F.mse_loss(states.float(), b['states'][:, 1:].float()) if self.state_head else image_loss * 0
        def masked(loss, mask):
            return (loss * mask).sum() / mask.sum().clamp_min(1)
        reward_loss = masked(F.smooth_l1_loss(rewards.float(), b['rewards'], reduction='none'), b['reward_mask'])
        done_loss = masked(F.binary_cross_entropy_with_logits(dones.float(), b['dones'], reduction='none'), b['done_mask'])
        reconstruction, kl = image_loss * 0, image_loss * 0
        if self.config['architecture'] == 'rssm':
            h, z = self.observe(b['images'][:, 0], b['states'][:, 0])
            for t, a in enumerate(b['actions'].unbind(1)):
                h = self.gru(torch.cat([z, a], -1), h)
                prior = self.distribution(self.prior(h))
                encoded = self.encoder(b['images'][:, t+1])
                z, post = self.latent(self.posterior(torch.cat([h, encoded], -1)), sample=self.training)
                # Free nats per sequence step; posterior is used only in this auxiliary objective.
                kl = kl + torch.distributions.kl_divergence(post, prior).sum(-1).clamp_min(self.config['free_nats']).mean()
                reconstruction = reconstruction + F.mse_loss(self.decode(h, z)[0].float(), b['images'][:, t+1].float())
            kl = kl / b['actions'].shape[1]
            reconstruction = reconstruction / b['actions'].shape[1]
        total = image_loss + state_loss + reward_loss + done_loss + reconstruction + self.config['kl_weight'] * kl
        return total, dict(image_loss=image_loss, state_loss=state_loss, reward_loss=reward_loss, done_loss=done_loss, reconstruction_loss=reconstruction, kl=kl)
