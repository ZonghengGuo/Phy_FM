from models.transformer import NeuralTransformer, NormEMAVectorQuantizer, PatchEmbed
from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange
import torch.nn.functional as F


class ECGTokenizer(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 embed_dim,
                 n_embed=8192,
                 quantize_kmeans_init=True,
                 decay=0.99,
                 patch_size=300,
                 decoder_out_dim=100,
                 smooth_l1_loss = False):
        super().__init__()

        self.encoder = NeuralTransformer(**encoder_config)
        self.decoder = NeuralTransformer(**decoder_config)

        self.decoder_out_dim = decoder_out_dim
        self.patch_size = patch_size

        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)
        )
        # Predicts amplitude from decoder output
        self.decode_task_layer_amp = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        # Predicts phase from decoder output
        self.decode_task_layer_phase = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        # Initialize weights
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_amp.apply(self._init_weights)
        self.decode_task_layer_phase.apply(self._init_weights)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed,
            embedding_dim=embed_dim,
            beta=1.0,  # Beta for commitment loss (can be tuned)
            kmeans_init=quantize_kmeans_init,
            decay=decay,
        )
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def _init_weights(self, m):
        """Initializes weights for linear and LayerNorm layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def std_norm(self, x):
        """Applies z-score normalization across specified dimensions."""
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        # Add a small epsilon to avoid division by zero
        x = (x - mean) / (std + 1e-6)
        return x

    def calculate_rec_loss(self, rec, target):
        """Calculates the reconstruction loss."""
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def get_codebook_indices(self, x, **kwargs):
        # for LaBraM pre-training
        x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size)
        indices = self.get_tokens(x, **kwargs)['token']
        return indices

    def get_tokens(self, data, **kwargs):
        quantize, embed_ind, loss = self.encode(data)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')
        return output

    def encode(self, x):
        ''''
            Encode ECG signals to quantized.
        '''
        batch_size, n_channels, n_patches_a, n_points_t = x.shape
        encoder_features = self.encoder(x, return_patch_tokens=True) # [4, 360, 100]

        # project features to codebook dimension
        to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight)) # [4, 360, 200]

        # Transfer feature shape the same with quantizer
        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=n_channels, w=n_patches_a) # [4, 200, 12, 30]


        # quantize features
        quantize, loss, embed_ind = self.quantize(to_quantizer_features) # [4, 200, in_chan, 5], value, 1440
        # Todo: loss how to calculate

        return quantize, embed_ind, loss

    def decode(self, quantize):
        ''''
            Decode the quantized embedding to reconstruct amp and phase
        '''
        # quantize: [4, 100, 12, 30]
        decoder_features = self.decoder(quantize, return_patch_tokens=True) # [4, 360, 24]

        # reconstruct amp and phase
        rec_amp = self.decode_task_layer_amp(decoder_features)
        rec_phase = self.decode_task_layer_phase(decoder_features)

        return rec_amp, rec_phase

    def forward(self, x):
        # Reshape input to [B, N, A, T] where T is the patch size
        x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size)

        # --- Fourier Spectrum Calculation ---
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # Normalize amplitude and phase
        amplitude = self.std_norm(amplitude)
        phase = self.std_norm(phase)

        # --- Encoding and Quantization ---
        quantize, embed_ind, emb_loss = self.encode(x)

        # --- Decoding and Reconstruction ---
        x_rec_amp, x_rec_phase = self.decode(quantize)

        # --- Loss Calculation ---
        rec_loss_amp = self.calculate_rec_loss(x_rec_amp, amplitude)
        rec_loss_phase = self.calculate_rec_loss(x_rec_phase, phase)

        # Total loss combines embedding loss and reconstruction losses
        loss = emb_loss + rec_loss_amp + rec_loss_phase

        # --- Logging ---
        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss_amp'] = rec_loss_amp.detach().mean()
        log[f'{split}/rec_loss_phase'] = rec_loss_phase.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log



# --- Helper to get default parameters (adapted from LaBraM) ---
def get_model_default_params(ecg_size=3000, patch_size=100, in_chans=12):
    """
    Provides default parameters for NeuralTransformer, adaptable for ECG.

    Args:
        ecg_size (int): Total number of sample points in an ECG window.
        patch_size (int): Size of each patch (w).
        in_chans (int): Number of ECG channels (C).

    Returns:
        dict: A dictionary of parameters.
    """
    return dict(
        sig_size=ecg_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=0, # No classification head needed for tokenizer
        embed_dim=100 * (12 // in_chans), # Base model size
        depth=12,      # Base model depth
        num_heads=10,  # Base model heads
        mlp_ratio=4.,
        qkv_bias=True, # Use bias
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1, # Base model init
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True, # For VQNSP, we usually need patch tokens
        init_scale=0.001,
        out_chans=8 # For TemporalConv if used (Encoder uses it)
    )

if __name__ == '__main__':
    # --- Configuration ---
    ECG_SAMPLE_RATE = 300 # Hz
    ECG_DURATION = 5     # seconds
    ECG_SIZE = ECG_SAMPLE_RATE * ECG_DURATION # 3000 points
    PATCH_SIZE = 300 # ~1 second per patch
    NUM_CHANNELS = 12 # Standard 12-lead ECG

    CODEBOOK_SIZE = 8192
    CODEBOOK_DIM = 100


    # Get default configs and adapt decoder
    encoder_config = get_model_default_params(ECG_SIZE, PATCH_SIZE, NUM_CHANNELS)
    decoder_config = get_model_default_params(ECG_SIZE, PATCH_SIZE, NUM_CHANNELS)

    # --- Important Decoder Adaptations ---
    # Decoder input size is the number of patches, not raw signal length
    decoder_config['sig_size'] = encoder_config['sig_size'] // encoder_config['patch_size']
    # Decoder patch size is 1 because it processes one token embedding at a time
    decoder_config['patch_size'] = 1
    # Decoder input channels must match the codebook dimension
    decoder_config['in_chans'] = NUM_CHANNELS
    # Decoder can be smaller (e.g., fewer layers)
    decoder_config['depth'] = 3
    decoder_config['embed_dim'] = 33 * (12//NUM_CHANNELS) + 2
    # Decoder needs to use PatchEmbed instead of TemporalConv
    decoder_config['patch_embed'] = PatchEmbed(
        EEG_size=decoder_config['sig_size'],
        patch_size=decoder_config['patch_size'],
        in_chans=decoder_config['in_chans'],
        embed_dim=decoder_config['embed_dim']
    )


    # --- Model Instantiation ---
    model = ECGTokenizer(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_embed=CODEBOOK_SIZE,
        embed_dim=CODEBOOK_DIM,
        patch_size=PATCH_SIZE,
        decoder_out_dim=PATCH_SIZE # Predict spectrum for each patch
    )

    # --- Dummy Data ---
    # Batch size = 4, 12 channels, 1500 points
    # Batch size = 4, 6 channels, 3000 points
    dummy_ecg_data = torch.randn(4, NUM_CHANNELS, ECG_SIZE)

    print("Input ECG data shape:", dummy_ecg_data.shape)

    # --- Forward Pass ---
    print("\n--- Running Forward Pass ---")
    loss, log_data = model(dummy_ecg_data)
    print(f"Total Loss: {loss.item()}")
    print("Log Data:", log_data)

    # --- Get Codebook Indices ---
    print("\n--- Getting Codebook Indices ---")
    indices = model.get_codebook_indices(dummy_ecg_data)
    print("Codebook Indices Shape:", indices.shape)

    print("\n--- Model Architecture ---")
    print(model)
