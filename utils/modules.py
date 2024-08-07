import torch
from torch import nn
from torch.distributions import Normal


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        # out += identity  # Skip connection
        return out

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(TwoLayerMLP, self).__init__()
        self.layer =  MLPLayer(
            input_dim,
            output_dim,
            dropout_prob=dropout_prob
        )
        self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = self.linear(self.layer(x))
        return out

class TwoLayerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(TwoLayerDecoder, self).__init__()
        self.layer =  MLPLayer(
            input_dim,
            input_dim,
            dropout_prob=dropout_prob
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(self.layer(x))
        return out

class MidFC(nn.Module):
    def __init__(self, dim, num_layers, dropout_prob=0.1):
        super(MidFC, self).__init__()
        self.layers = nn.Sequential(
            *[MLPLayer(dim, dim) for _ in range(num_layers)]
        )
        
    def forward(self, x):
        return self.layers(x)


class CustomDecoder(nn.Module):
    def __init__(self, hidden_size, input_dim, device, num_blocks=1):
        super(CustomDecoder, self).__init__()
        self.intermediate_dim = max(hidden_size, input_dim)
        self.initial_layer = nn.Sequential(
            nn.Linear(hidden_size, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(self.intermediate_dim) for _ in range(num_blocks)]
        )
        self.final_layer = nn.Linear(self.intermediate_dim, input_dim)
        self.to(device)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.final_layer(x)
        return x

class CustomVAEDecoder(nn.Module):
    def __init__(self, hidden_size, input_dim, device, reshape_postvae, space_dim, num_blocks=1, mlp_enc=False):
        super(CustomVAEDecoder, self).__init__()
        self.reshape_postvae = reshape_postvae
        self.space_dim = space_dim
        if self.reshape_postvae:
            gauss_dim = hidden_size
        else:
            gauss_dim = hidden_size*space_dim
        if mlp_enc:
            self.mean_encoder = TwoLayerMLP(gauss_dim, gauss_dim).to(device)
            self.var_encoder = TwoLayerMLP(gauss_dim, gauss_dim).to(device)
        else:
            self.mean_encoder = ResidualBlock(gauss_dim).to(device)
            self.var_encoder = ResidualBlock(gauss_dim).to(device)
        self.var_activation = torch.exp
        self.var_eps = 0.0001
        self.decoder = CustomDecoder(
            hidden_size=hidden_size*space_dim,
            input_dim=input_dim,
            device=device,
            num_blocks=num_blocks
        )
        self.to(device)

    def forward(self, x, temperature=1.0):
        mu = self.mean_encoder(x)
        var = self.var_encoder(x)
        var = self.var_activation(var) + self.var_eps
        dist = Normal(mu, temperature*(var.sqrt()))
        latents = dist.rsample()
        if self.reshape_postvae:
            batch_size, seq_len, feature_size = latents.shape # now tensor shape is batch_size x num_time_points*space_dim x feature_dim//space_dim
            latents_reshaped = latents.view(batch_size, seq_len//self.space_dim, self.space_dim, feature_size)
            latents_reshaped = latents_reshaped.view(batch_size, seq_len//self.space_dim, feature_size*self.space_dim)
            outputs = self.decoder(latents_reshaped)
        else:
            outputs = self.decoder(latents)
        return outputs, latents, dist


class CustomSCVIDecoder(nn.Module):
    def __init__(
        self, hidden_size, input_dim, device, num_blocks=1
    ):
        super().__init__()
        self.px_r = torch.nn.Parameter(torch.randn(input_dim))
        self.px_decoder = CustomDecoder(
            hidden_size=hidden_size,
            input_dim=input_dim,
            device=device,
            num_blocks=num_blocks
        )

        self.mean_encoder = MLPLayer(hidden_size, hidden_size).to(device)
        self.var_encoder = MLPLayer(hidden_size, hidden_size).to(device)

        self.library_mean_encoder = MLPLayer(hidden_size, 1).to(device)
        self.library_var_encoder = MLPLayer(hidden_size, 1).to(device)

        self.var_activation = torch.exp
        self.var_eps = 0.0001

        px_scale_activation = nn.Softmax(dim=-1)
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_size, input_dim),
            px_scale_activation,
        ).to(device)

        self.px_dropout_decoder = nn.Linear(hidden_size, input_dim).to(device)
        self.to(device)

    def forward(self, x, temperature=1.0):
        mu = self.mean_encoder(x)
        var = self.var_encoder(x)
        var = self.var_activation(var) + self.var_eps
        dist = Normal(mu, temperature*(var.sqrt()))
        z = dist.rsample()

        lib_mu = self.library_mean_encoder(x)
        lib_var = self.library_var_encoder(x)
        lib_var = self.var_activation(lib_var) + self.var_eps
        lib_dist = Normal(lib_mu, lib_var.sqrt())
        library = lib_dist.rsample()

        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r
        px_r = torch.exp(px_r)
        return px_scale, px_r, px_rate, px_dropout, z, dist
