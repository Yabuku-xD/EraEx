import json
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


class AudioTransformer(nn.Module):
    # Initialize class state.
    def __init__(self, d_model=768, nhead=8, num_layers=12):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        super().__init__()
        self.patch_embed = nn.Linear(128 * 16, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 400, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    # Handle forward.
    def forward(self, x, return_all_layers=False):
        """
        Run a forward pass through the model.
        
        This method implements the forward step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        patches = extract_patches(x)
        x = self.patch_embed(patches)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        if return_all_layers:
            hidden_states = []
            output = x
            for layer in self.transformer.layers:
                output = layer(output)
                hidden_states.append(output)
            return hidden_states
        return self.transformer(x)


# Extract patches.
def extract_patches(mel_spec, patch_size=(128, 16)):
    """
    Extract patches.
    
    This function implements the extract patches step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    batch, freq, time = mel_spec.shape
    patch_width = patch_size[1]
    num_patches = time // patch_width
    patches = []
    for i in range(num_patches):
        start = i * patch_width
        end = start + patch_width
        patch = mel_spec[:, :, start:end]
        patches.append(patch.reshape(batch, -1))
    return torch.stack(patches, dim=1)


# Compute mel spectrogram.
def compute_mel_spectrogram(audio_path, sr=24000, n_mels=128):
    """
    Compute mel spectrogram.
    
    This function implements the compute mel spectrogram step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not os.path.exists(audio_path):
        return np.zeros((128, 100))
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=30)
        spectrum = librosa.stft(y, n_fft=2048, hop_length=512)
        spectrum_power = np.abs(spectrum) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels)
        mel_spec = mel_basis @ spectrum_power
        return librosa.power_to_db(mel_spec, ref=np.max)
    except Exception as exc:
        print(f"Error processing {audio_path}: {exc}")
        return np.zeros((128, 100))


class ActivationSteering:
    # Initialize class state.
    def __init__(self, model):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.model = model
        self.steering_vectors = {}

    # Compute steering vectors.
    def compute_steering_vectors(self, prompt_set_A, prompt_set_B, attr_name="feature"):
        """
        Compute steering vectors.
        
        This method implements the compute steering vectors step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        layers = 12
        d_model = 768
        mean_a = [torch.zeros(d_model) for _ in range(layers)]
        mean_b = [torch.zeros(d_model) for _ in range(layers)]
        self.model.eval()

        count_a = 0
        with torch.no_grad():
            for audio_path in prompt_set_A:
                try:
                    mel = compute_mel_spectrogram(audio_path)
                    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
                    if mel_tensor.shape[2] < 16:
                        continue
                    hidden_states = self.model(mel_tensor, return_all_layers=True)
                    for layer_idx in range(layers):
                        mean_a[layer_idx] += hidden_states[layer_idx].mean(dim=1).squeeze(0)
                    count_a += 1
                except Exception as exc:
                    print(f"Error processing {audio_path}: {exc}")

        count_b = 0
        with torch.no_grad():
            for audio_path in prompt_set_B:
                try:
                    mel = compute_mel_spectrogram(audio_path)
                    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
                    if mel_tensor.shape[2] < 16:
                        continue
                    hidden_states = self.model(mel_tensor, return_all_layers=True)
                    for layer_idx in range(layers):
                        mean_b[layer_idx] += hidden_states[layer_idx].mean(dim=1).squeeze(0)
                    count_b += 1
                except Exception as exc:
                    print(f"Error processing {audio_path}: {exc}")

        if count_a > 0:
            for layer_idx in range(layers):
                mean_a[layer_idx] /= count_a
        if count_b > 0:
            for layer_idx in range(layers):
                mean_b[layer_idx] /= count_b

        vectors = [mean_a[layer_idx] - mean_b[layer_idx] for layer_idx in range(layers)]
        self.steering_vectors[attr_name] = vectors
        print(f"Computed steering vectors for '{attr_name}' using {count_a} vs {count_b} samples.")
        return vectors

    # Extract features.
    def extract_features(self, audio_path):
        """
        Extract features.
        
        This method implements the extract features step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        try:
            mel = compute_mel_spectrogram(audio_path)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
            if mel_tensor.shape[2] < 16:
                return {key: 0.0 for key in self.steering_vectors}
            device = next(self.model.parameters()).device
            mel_tensor = mel_tensor.to(device)
            with torch.no_grad():
                hidden_states = self.model(mel_tensor, return_all_layers=True)
            features = {}
            target_layer = 10 if len(hidden_states) > 10 else -1
            h = hidden_states[target_layer].mean(dim=1).squeeze(0)
            for attr, vectors in self.steering_vectors.items():
                if not vectors:
                    features[attr] = 0.0
                    continue
                delta = vectors[target_layer].to(device)
                norm_h = torch.norm(h)
                norm_delta = torch.norm(delta)
                if norm_h > 0 and norm_delta > 0:
                    score = torch.dot(h, delta) / (norm_h * norm_delta)
                    features[attr] = score.item()
                else:
                    features[attr] = 0.0
            return features
        except Exception:
            return {key: 0.0 for key in self.steering_vectors}

    # Load dummy vectors.
    def load_dummy_vectors(self):
        """
        Load dummy vectors.
        
        This method implements the load dummy vectors step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        layers = 12
        d_model = 768
        attrs = ["tempo", "energy", "brightness", "mood"]
        torch.manual_seed(42)
        for attr in attrs:
            self.steering_vectors[attr] = [torch.randn(d_model) for _ in range(layers)]


class AudioDownloader:
    # Initialize class state.
    def __init__(self, output_dir="data/audio_cache"):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ydlp_bin = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")

    # Internal helper to download with module.
    def _download_with_module(self, url, track_id):
        """
        Execute download with module.
        
        This method implements the download with module step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.output_dir / f"{track_id}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "max_filesize": 15 * 1024 * 1024,
            "match_filter": self._filter_duration,
            "socket_timeout": 5,
            "retries": 0,
            "fragment_retries": 0,
            "source_address": "0.0.0.0",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ),
            "extractor_args": {
                "youtube": {"player_client": ["android", "web"], "player_skip": ["configs", "js"]}
            },
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return self.get_audio_path(track_id)

    # Internal helper to download with binary.
    def _download_with_binary(self, url, track_id):
        """
        Execute download with binary.
        
        This method implements the download with binary step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self._ydlp_bin:
            return None
        outtmpl = str(self.output_dir / f"{track_id}.%(ext)s")
        cmd = [
            self._ydlp_bin,
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--socket-timeout",
            "5",
            "--retries",
            "0",
            "--fragment-retries",
            "0",
            "-f",
            "bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "128K",
            "-o",
            outtmpl,
            url,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return self.get_audio_path(track_id)

    # Handle download preview.
    def download_preview(self, query, track_id):
        """
        Execute download preview.
        
        This method implements the download preview step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        url = query if str(query).startswith("http") else f"ytsearch1:{query}"
        if yt_dlp is not None:
            try:
                path = self._download_with_module(url, track_id)
                if path:
                    return path
            except Exception:
                pass
        return self._download_with_binary(url, track_id)

    # Internal helper to filter duration.
    @staticmethod
    def _filter_duration(info, *, incomplete):
        """
        Execute filter duration.
        
        This method implements the filter duration step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        duration = info.get("duration")
        if duration and duration > 600:
            return "Video too long"
        return None

    # Get audio path.
    def get_audio_path(self, track_id):
        """
        Get audio path.
        
        This method implements the get audio path step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        for ext in ["mp3", "m4a", "webm", "wav", "opus"]:
            path = self.output_dir / f"{track_id}.{ext}"
            if path.exists():
                return str(path)
        return None