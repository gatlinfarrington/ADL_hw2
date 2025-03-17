import abc
import torch
from torch import nn
from pathlib import Path
import os

def load() -> torch.nn.Module:
    model_name = "PatchAutoEncoder"
    model_path = os.path.join(os.path.dirname(__file__), f"{model_name}.pth")
    print(f"Loading {model_name} from {model_path}")

    # Fix: Explicitly set `weights_only=False` to load full model
    model = torch.load(model_path, map_location="cpu", weights_only=False)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"{model_name} loaded successfully to {device}")
    return model

def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)

def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)

class PatchifyLinear(torch.nn.Module):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))

class UnpatchifyLinear(torch.nn.Module):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))

class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        pass

class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    class PatchEncoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.patch_size = patch_size
            self.patchify = PatchifyLinear(patch_size=patch_size, latent_dim=latent_dim)
            self.conv = nn.Conv2d(latent_dim, bottleneck, kernel_size=3, padding=1, bias=False)
            self.activation = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patchify(x)
            x = hwc_to_chw(x)
            x = self.conv(x)
            x = self.activation(x)
            return chw_to_hwc(x)

    class PatchDecoder(torch.nn.Module):
      def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
          super().__init__()
          self.patch_size = patch_size
          # Make sure padding is sufficient
          self.conv = nn.ConvTranspose2d(bottleneck, latent_dim, kernel_size=3, padding=1, bias=False)
          self.activation = nn.GELU()
          self.unpatchify = UnpatchifyLinear(patch_size=patch_size, latent_dim=latent_dim)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          # Preserve dimensions through the entire pipeline
          x = hwc_to_chw(x)
          x = self.conv(x)
          x = self.activation(x)
          x = chw_to_hwc(x)
          return self.unpatchify(x)

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

if __name__ == "__main__":
    model = PatchAutoEncoder(patch_size=5, latent_dim=128, bottleneck=128)
    x = torch.randn(2, 150, 100, 3)
    encoded = model.encode(x)
    print(f"Encoded shape: {encoded.shape}")
    reconstructed, _ = model(x)
    print(f"Reconstructed shape: {reconstructed.shape}")