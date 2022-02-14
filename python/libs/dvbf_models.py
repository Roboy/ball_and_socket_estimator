import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from typing import Tuple, Dict


class DVBF(pl.LightningModule):
    def __init__(self, 
                n_frames: int,
                n_observations: int, 
                n_actions: int, 
                n_latents: int,
                n_outputs: int, 
                seq_len: int,
                batch_size: int,
                num_matrices: int = 16, 
                hidden_size: int = 128,
                n_initial_obs: int = 3,
                learning_rate: float = 5e-4,
                alpha: float = 2.0,
                beta: float = 5.0,
                annealing: float = 0.1,
                temperature: float = 5e-3):
        super().__init__()
        self.save_hyperparameters()

        self.n_frames = n_frames
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_latents = n_latents
        self.n_outputs = n_outputs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_initial_obs = n_initial_obs
        self.alpha = alpha
        self.beta = beta
        self.annealing = 1 / temperature
        self.temperature = temperature

        # Prepare networks
        
        self.initial_lstm = nn.LSTM(input_size=n_frames * n_observations, batch_first=True, hidden_size=hidden_size, dropout=0.1, bidirectional=True)
        self.initial_to_params = nn.Sequential(
            nn.Linear(in_features=2*hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2*n_latents)
        )
        self.w1_to_z1 = nn.Sequential(
            nn.Linear(in_features=n_latents, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=n_latents)
        )

        self.matrix_params = nn.Sequential(
            nn.Linear(in_features=n_latents+n_actions, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_matrices),
            nn.Softmax(dim=1)
        )

        shared_encoder = nn.Sequential(
            nn.Linear(in_features=n_observations, out_features=hidden_size),
            nn.ReLU()
        )

        self.encoder_models = []

        for _ in range(n_frames):
            self.encoder_models.append(
                nn.Sequential(
                    shared_encoder,
                    nn.Linear(in_features=hidden_size, out_features=2*n_latents)
                )
            )

        self.decoder_model = nn.Sequential(
            nn.Linear(in_features=n_latents, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2*n_frames*n_observations)
        )

        self.regressor_model = nn.Sequential(
            nn.Linear(in_features=n_latents, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2*n_outputs)
        )

        self.encoder_models = nn.ModuleList(self.encoder_models)

        # Prepare locally linear matrices

        scale = 1 / num_matrices / n_observations
        std_scale = 1 / num_matrices

        self.A = nn.Parameter(torch.randn([num_matrices, n_latents, n_latents]) * scale)

        if n_actions > 0:
            self.B = nn.Parameter(torch.randn([num_matrices, n_latents, n_actions]) * scale)

        self.posterior_std = nn.Parameter(
            torch.randn(num_matrices, n_latents, dtype=torch.float32) * std_scale
        )

        self.prior_std = nn.Parameter(
            torch.randn(num_matrices, n_latents, dtype=torch.float32) * std_scale
        )


    def get_initial_samples(self, x: torch.Tensor, predict: bool = False) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        output, (hidden, cell_states) = self.initial_lstm(x.reshape(-1, x.shape[1], self.n_frames * self.n_observations))
        w_params = self.initial_to_params(output[:, -1])
        mean, std = torch.split(w_params, split_size_or_sections=self.n_latents, dim=1)
        std = torch.exp(std) + 1e-5
        q_w = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
        w1 = q_w.sample()
        if not predict:
            z1 = self.w1_to_z1(w1)
        else:
            z1 = self.w1_to_z1(mean)
        return q_w, z1, w1

    def mix_matrices(self, z_t, u_t = None):
        if u_t is not None:
            alpha = self.matrix_params(torch.cat([z_t, u_t], dim=-1))
        else:
            alpha = self.matrix_params(z_t)

        M = self.A.shape[0]
        A = (alpha @ self.A.view(M, -1)).view(-1, self.n_latents, self.n_latents)
        posterior_var = alpha @ self.posterior_std
        posterior_var = posterior_var ** 2 + 1e-3

        prior_var = alpha @ self.prior_std
        prior_var = prior_var ** 2 + 1e-3

        if u_t is not None:
            B = (alpha @ self.B.view(M, -1)).view(-1, self.n_latents, self.n_actions)
            return A, B, posterior_var, prior_var
        else:
            return A, posterior_var, prior_var

    def sensor_fusion(self, means, vars):
        # Check dimension
        inv_vars = 1 / (vars + 1e-5)
        posterior_variance = 1 / inv_vars.sum(1)
        posterior_mean = posterior_variance * (inv_vars * means).sum(1)
        return torch.distributions.MultivariateNormal(posterior_mean, torch.diag_embed(torch.sqrt(posterior_variance)))

    def belief_update(self, z: torch.Tensor, u: torch.Tensor = None, x: torch.Tensor = None):

        # Transition propagation
        if u is not None:
            A, B, posterior_var, prior_var = self.mix_matrices(z, u)
            z = z + (A @ z.unsqueeze(-1) + B @ u.unsqueeze(-1)).squeeze(-1)
        else:
            A, posterior_var, prior_var = self.mix_matrices(z)
            z = z + (A @ z.unsqueeze(-1)).squeeze(-1)

        # Inverse measurement
        meas_means, meas_vars = [], []
        for i, encoder in enumerate(self.encoder_models):
            meas = encoder(x[:, i])
            meas_mean, meas_var = torch.split(meas, split_size_or_sections=self.n_latents, dim=1)
            meas_var = torch.exp(meas_var) + 1e-5

            meas_means.append(meas_mean)
            meas_vars.append(meas_var)

        # Sensor fusion
        all_means = torch.stack([z] + meas_means, 1) # concat z and meas_mean
        all_vars = torch.stack([posterior_var] + meas_vars, 1) # concat posterior_var and meas_var
        z_dist = self.sensor_fusion(all_means, all_vars)
        z = z_dist.rsample()

        return z, z_dist.mean, z_dist.variance, prior_var

    def filter(self, x: torch.Tensor, u: torch.Tensor = None):
        q_w, z_t, w_t = self.get_initial_samples(x[:, :self.n_initial_obs])
        z = [z_t]
        z_mean = [q_w.mean]
        z_var = [q_w.variance]
        prior_var = []

        for t in range(1, self.seq_len):
            if u is not None:
                z_t, z_t_mean, z_t_var, prior_t_var = self.belief_update(z=z_t, u=u[:, t - 1], x=x[:, t])
            else:
                z_t, z_t_mean, z_t_var, prior_t_var = self.belief_update(z=z_t, x=x[:, t])

            z.append(z_t)
            z_mean.append(z_t_mean)
            z_var.append(z_t_var)
            prior_var.append(prior_t_var)

        z = torch.stack(z, dim=1)
        z_mean = torch.stack(z_mean, dim=1)
        z_var = torch.stack(z_var, dim=1)
        prior_var = torch.stack(prior_var, dim=1)
        
        q_z = torch.distributions.MultivariateNormal(z_mean, torch.diag_embed(z_var))
        prior_z = torch.distributions.MultivariateNormal(
            loc=torch.cat((torch.zeros_like(z_mean[:, 0])[:, None], z_mean[:, :-1]), dim=1), 
            covariance_matrix=torch.eye(self.n_latents).to(
                torch.cat((torch.ones_like(prior_var[:, 0])[:, None], prior_var[:, :-1]), dim=1)
            )
        )
        return z, q_z, prior_z

    def reconstruct(self, z: torch.Tensor, return_dist=False):
        x_rec = self.decoder_model(z)
        mean, std = torch.split(x_rec, split_size_or_sections=self.n_frames * self.n_observations, dim=-1)
        std = torch.exp(std) + 1e-5
        p_x = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

        if return_dist:
            return p_x
        else:
            return p_x.sample()

    def regressing(self, z: torch.Tensor, return_dist=False):
        y_hat = self.regressor_model(z)
        mean, std = torch.split(y_hat, split_size_or_sections=self.n_outputs, dim=-1)
        std = torch.exp(std) + 1e-5
        p_y = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

        if return_dist:
            return p_y
        else:
            return p_y.sample()

    def forward(self, x: torch.Tensor, u: torch.Tensor=None):
        z, q_z, prior_z = self.filter(x, u)
        x = self.reconstruct(z, return_dist=False)
        return x

    def predict_initial(self, x: torch.Tensor):
        q_w, z_t, w_t = self.get_initial_samples(x[:, :self.n_initial_obs], predict=True)

        return z_t, q_w

    def predict_belief(self, z: torch.Tensor, u: torch.Tensor = None, x: torch.Tensor = None):

        if u is not None:
            z_t, z_t_mean, z_t_var, prior_t_var = self.belief_update(z=z, u=u, x=x)
        else:
            z_t, z_t_mean, z_t_var, prior_t_var = self.belief_update(z=z, x=x)

        p_y = self.regressing(z_t_mean, return_dist=True)
        y_mean = p_y.mean

        return y_mean, z_t_mean

    def predict(self, x: torch.Tensor, u: torch.Tensor = None):
        T = x.shape[1]

        outputs = []
        state, _ = self.predict_initial(x[:, :self.n_initial_obs])
        for t in range(T):
            output, state = self.predict_belief(state, u[:, t], x[:, t])
            # output, state = self.predict_belief(state, None, x[:, t])
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def inv_meas(self, x: torch.Tensor):
        # Inverse measurement
        meas_means, meas_vars = [], []
        for i, encoder in enumerate(self.encoder_models):
            meas = encoder(x[:, i])
            meas_mean, meas_var = torch.split(meas, split_size_or_sections=self.n_latents, dim=1)
            meas_var = torch.exp(meas_var) + 1e-5

            meas_means.append(meas_mean)
            meas_vars.append(meas_var)

        return meas_means, meas_vars

    def criterion(self, x, u, y, x_hat=None):
        # u = None
        x_hat = x if x_hat is None else x_hat

        z, q_z, prior_z = self.filter(x, u)

        p_x = self.reconstruct(z, return_dist=True)
        logprob_x = p_x.log_prob(x.view(-1, self.seq_len, self.n_frames * self.n_observations) + 1e-6)

        p_y = self.regressing(z, return_dist=True)
        logprob_y = p_y.log_prob(y + 1e-6)

        nllx = -logprob_x.mean()
        nlly = -logprob_y.mean()
        kl = torch.distributions.kl_divergence(q_z, prior_z).mean()
        
        loss = nllx + self.alpha * nlly + self.beta * self.annealing * kl

        self.annealing = min(self.annealing + (1.0 / self.temperature), 1.0)

        return loss, nllx, nlly, kl

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, u, y = batch
            x_hat = None
        elif len(batch) == 4:
            x, u, y, x_hat = batch
        loss, nllx, nlly, kl = self.criterion(x, u, y, x_hat)
        self.log_dict({"train_loss": loss, "train_nllx": nllx, "train_nlly": nlly, "train_kl": kl}, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, u, y = batch
            x_hat = None
        elif len(batch) == 4:
            x, u, y, x_hat = batch
        loss, nllx, nlly, kl = self.criterion(x, u, y, x_hat)
        self.log_dict({"val_loss": loss, "val_nllx": nllx, "val_nlly": nlly, "val_kl": kl}, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, u, y = batch
            x_hat = None
        elif len(batch) == 4:
            x, u, y, x_hat = batch
        loss, nllx, nlly, kl = self.criterion(x, u, y, x_hat)
        self.log_dict({"test_loss": loss, "test_nllx": nllx, "test_nlly": nlly, "test_kl": kl}, on_epoch=True, prog_bar=True, logger=True)
        return loss
