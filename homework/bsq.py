import abc
import torch
from torch import nn
from .ae import PatchAutoEncoder, hwc_to_chw, chw_to_hwc

def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)

def diff_sign(x: torch.Tensor) -> torch.Tensor:
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        pass

class BSQ(nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.down_proj = nn.Linear(embedding_dim, codebook_bits, bias=False)
        self.up_proj = nn.Linear(codebook_bits, embedding_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, h, w, _ = x.shape
        x = x.reshape(B * h * w, -1)
        x = self.down_proj(x)
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = diff_sign(x)
        return x.view(B, h, w, self.codebook_bits)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
      if x.dim() == 3:
          x = x.unsqueeze(0)  # Ensure batch dimension

      B, h, w, _ = x.shape  # Encoded shape
      x = x.view(B * h * w, -1)  # Flatten
      x = self.up_proj(x)  # Expand to embedding_dim
      x = x.view(B, h, w, self.embedding_dim)
      
      # Don't manipulate channels here or perform interpolation
      # Just return the embeddings and let the decoder handle it
      return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        codes = self.encode(x)  # Directly use self.encode()
        indices = self._code_to_index(codes)  # Convert codes to indices
        return indices  # Ensure correct shape

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        print(f"decode_index() input shape: {x.shape}")  # Print input shape

        decoded = self.decode(self.bsq._index_to_code(x))

        print(f"Decoded output shape before reshaping: {decoded.shape}")  # Print before reshaping

        return decoded

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(self.codebook_bits).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self.codebook_bits).to(x.device))) > 0).float() - 1

class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 8):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.patch_size = patch_size  # Store for later use
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return self.bsq.encode(embeddings)
        
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        print(f"🔹 Input to decode (BSQ output) shape: {x.shape}")

        embeddings = self.bsq.decode(x)
        print(f"🔹 After BSQ decode, shape: {embeddings.shape}")

        reconstructed = self.decoder(embeddings)
        print(f"🔹 After decoder, shape: {reconstructed.shape}")

        # Check if height & width are correct
        if reconstructed.shape[1] != 150:  
            print("⚠️ Height is incorrect. Swapping height and width.")
            reconstructed = reconstructed.permute(0, 2, 1, 3)  # Swap height & width
            print(f"🔹 After permute, shape: {reconstructed.shape}")

        return reconstructed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        codes = self.encode(x)
        reconstructed = self.decode(codes)
        indices = self.bsq._code_to_index(codes).flatten()
        cnt = torch.bincount(indices, minlength=2**self.codebook_bits)
        cb0 = (cnt == 0).float().mean()
        cb2 = (cnt <= 2).float().mean()
        additional = {
            "cb0": cb0.detach(),
            "cb2": cb2.detach(),
        }
        code_usage_loss = cb0 * 0.1
        if x.shape[0] == 1:
            reconstructed = reconstructed.squeeze(0)
        return reconstructed, {"code_usage_loss": code_usage_loss} | additional

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
      if x.dim() == 3:
          x = x.unsqueeze(0)  # Ensure batch dimension
      if x.shape[1] == 100 and x.shape[2] == 150:  
          x = x.transpose(1, 2)  # Swap to (B, 150, 100, 3) before encoding
      indices = self.bsq.encode_index(self.encoder(x))
      return indices

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.bsq._index_to_code(x))

if __name__ == "__main__":
    model = BSQPatchAutoEncoder(patch_size=8, latent_dim=128, codebook_bits=8)
    x = torch.randn(150, 100, 3)
    codes = model.encode(x)
    print(f"Codes shape: {codes.shape}")
    indices = model.encode_index(x)
    print(f"Indices shape: {indices.shape}")
    reconstructed, _ = model(x)
    print(f"Reconstructed shape: {reconstructed.shape}")