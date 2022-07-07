import torch
import torch.nn as nn
import pytorch_lightning as pl


class DVBF(pl.LightningModule):
    def __init__(self, 
                dim_x: Tuple, 
                dim_u: int, 
                dim_z: int, 
                dim_w: int, 
                num_matrices: int = 16, 
                hidden_size=128):
        super().__init__()

        self.dim_x = np.prod(dim_x).item()
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_u = dim_u
        self.initial_lstm = nn.LSTM(input_size=dim_x, batch_first=True, hidden_size=hidden_size, dropout=0.1, bidirectional=True)
        self.initial_to_params = nn.Sequential(
            nn.Linear(in_features=2*hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2*dim_w)
        )
        self.w1_to_z1 = nn.Sequential(
            nn.Linear(in_features=dim_w, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=dim_z)
        )
        self.w_params = nn.Sequential(
            nn.Linear(in_features=dim_z+dim_u+self.dim_x, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=2*dim_w),
        )

        self.v_params = nn.Sequential(
            nn.Linear(in_features=dim_z+dim_u, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_matrices),
            nn.Softmax()
        )

        self.observation_model = nn.Sequential(
            nn.Linear(in_features=dim_z, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim_x)
        )

        self.A = nn.Parameter(torch.randn([num_matrices, dim_z, dim_z]))
        self.B = nn.Parameter(torch.randn([num_matrices, dim_z, dim_u]))
        self.C = nn.Parameter(torch.randn([num_matrices, dim_z, dim_w]))


    def get_initial_samples(self, x: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        output, (hidden, cell_states) = self.initial_lstm(x)
        w_params = self.initial_to_params(output[:, -1])
        mean, std = torch.split(w_params, split_size_or_sections=self.dim_w, dim=1)
        std = torch.exp(std) + 1e-5
        q_w = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
        w1 = q_w.sample()
        z1 = self.w1_to_z1(w1)
        return q_w, z1, w1

    def mix_matrices(self, z_t, u_t):
        alpha = self.v_params(torch.cat([z_t, u_t], dim=-1))
        M = self.A.shape[0]
        A = (alpha @ self.A.view(M, -1)).view(-1, self.dim_z, self.dim_z)
        B = (alpha @ self.B.view(M, -1)).view(-1, self.dim_z, self.dim_u)
        C = (alpha @ self.C.view(M, -1)).view(-1, self.dim_z, self.dim_w)
        return A, B, C

    def sample_w(self, z_t, u_t, x_t=None):
        if x_t is not None:
            data = torch.cat([x_t, z_t, u_t], dim=1)
            w_params = self.w_params(data)
            mean, std = torch.split(w_params, split_size_or_sections=self.dim_w, dim=1)
            std = torch.exp(std) + 1e-2
        else:
            mean = torch.zeros((z_t.shape[0], self.dim_w)).to(z_t)
            std = torch.ones((z_t.shape[0], self.dim_w)).to(z_t)
        return torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

    def filter(self, x: torch.Tensor, u: torch.Tensor):
        num_obs = x.shape[1]
        N, T, _ = u.shape
        q_w, z_t, w_t = self.get_initial_samples(x)
        z = [z_t]
        w = [w_t]
        w_means = [q_w.mean]
        w_stds = [q_w.stddev]

        for t in range(1, T):
            u_t = u[:, t - 1]
            if t < num_obs:
                z_t, q_w, w_t = self.forward(z=z_t, u=u_t, x=x[:, t], return_q=True)
            else:
                z_t, q_w, w_t = self.forward(z=z_t, u=u_t, return_q=True)
            z.append(z_t)
            w.append(w_t)
            w_means.append(q_w.mean)
            w_stds.append(q_w.stddev)
        z = torch.stack(z, dim=1)
        w_means = torch.stack(w_means, dim=1)
        w_stds = torch.stack(w_stds, dim=1)
        w = torch.stack(w, dim=1)
        return z, dict(w_means=w_means, w_stds=w_stds, betas=w)

    def forward(self, z: torch.Tensor, u: torch.Tensor, x: torch.Tensor = None, return_q=False):
        q_w = self.sample_w(z, u, x)
        A, B, C = self.mix_matrices(z, u)
        w = q_w.sample()
        z = (A @ z.unsqueeze(-1) + B @ u.unsqueeze(-1) + C @ w.unsqueeze(-1)).squeeze(-1)
        if return_q:
            return z, q_w, w
        else:
            return z

    def reconstruct(self, z: torch.Tensor, return_dist=False):
        x_rec_mean = self.observation_model(z).view(-1, self.dim_x)
        p_x = torch.distributions.MultivariateNormal(x_rec_mean, torch.diag(torch.ones(self.dim_x)).to(x_rec_mean))
        if return_dist:
            return p_x
        else:
            return p_x.sample()


    def loss(self, x, u, c=1.0):
        z, info = self.filter(x, u)
        w_means, w_stds, w = info['w_means'], info['w_stds'], info['betas']
        p_x = self.reconstruct(z, return_dist=True)
        logprob_x = p_x.log_prob(x.view(-1, self.dim_x))
        w_mean, w_std = w_means.view(-1, self.dim_w), w_stds.view(-1, self.dim_w)
        q_w = torch.distributions.MultivariateNormal(w_mean, torch.diag_embed(w_std))
        prior_w = torch.distributions.MultivariateNormal(loc=torch.zeros_like(w_mean), covariance_matrix=torch.eye(self.dim_w).to(w_mean))
        loss = logprob_x.sum() - torch.distributions.kl_divergence(q_w, prior_w).sum()
        #loss = c * p_x.log_prob(x) - q_w.log_prob(w) + c * prior_w.log_prob(w)
        return -loss
