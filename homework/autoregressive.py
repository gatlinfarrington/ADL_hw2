import abc
import torch
from torch import nn

def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)

class Autoregressive(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        pass

class AutoregressiveModel(nn.Module, Autoregressive):
    def __init__(self, d_latent: int = 256, n_tokens: int = 2**10):
        super().__init__()
        self.h = 30
        self.w = 20
        self.seq_len = self.h * self.w
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        self.embedding = nn.Embedding(n_tokens, d_latent)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_latent) * 0.01)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4, norm=nn.LayerNorm(d_latent))
        
        self.to_logits = nn.Linear(d_latent, n_tokens)
        nn.init.xavier_uniform_(self.to_logits.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        print(f"Input shape: {x.shape}")
        B = x.shape[0]
        # Handle training input [B, 1, h, w], [B, h, w], or generation input [B, t+1]
        if x.dim() == 4:  # [B, 1, h, w]
            x = x.squeeze(1)  # [B, h, w]
        if x.dim() == 3:  # [B, h, w]
            x = x.view(B, -1)  # [B, h*w]
        elif x.dim() == 2:  # [B, t+1]
            pass  # Already in the correct shape
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got {x.dim()}D tensor with shape {x.shape}")
        print(f"After flatten: {x.shape}")
        
        x = self.embedding(x)
        print(f"After embedding: {x.shape}")
        
        # Adjust positional embedding for variable sequence length
        seq_len = x.shape[1]
        if seq_len > self.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.seq_len}")
        x = x + self.pos_embedding[:, :seq_len]
        print(f"After pos embedding: {x.shape}")
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        print(f"Mask shape: {mask.shape}")
        
        x = self.transformer(x, mask=mask)
        print(f"After transformer: {x.shape}")
        logits = self.to_logits(x)
        print(f"Logits shape before view: {logits.shape}")
        if x.shape[1] == self.seq_len:
            logits = logits.view(B, self.h, self.w, self.n_tokens)
        print(f"Final logits shape: {logits.shape}")
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None, temperature: float = 0.7, top_k: int = 20) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        seq_len = h * w
        tokens = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        for t in range(seq_len):
            with torch.no_grad():
                logits, _ = self.forward(tokens[:, :t+1])
                logits_t = logits[:, -1] / temperature
                probs = nn.functional.softmax(logits_t, dim=-1)
                if probs.dim() > 2:
                    probs = probs.view(B, -1)
                # Clamp logits to prevent numerical issues
                logits_t = torch.clamp(logits_t, min=-100.0, max=100.0)
                probs = nn.functional.softmax(logits_t, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token_indices = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_indices)
                # Ensure tokens are within codebook range (0 to n_tokens-1)
                tokens[:, t] = torch.clamp(next_token.squeeze(-1), min=0, max=self.n_tokens - 1)
                if t % 100 == 0:
                    print(f"Token at position {t}: {tokens[:, t].cpu().numpy()}")
        print(f"Final tokens shape: {tokens.shape}")
        print(f"Final tokens unique values: {torch.unique(tokens).cpu().numpy()}")
        return tokens.view(B, h, w)

if __name__ == "__main__":
    model = AutoregressiveModel(d_latent=512, n_tokens=1024)
    x = torch.randint(0, 1024, (2, 30, 20))
    logits, _ = model(x)
    print(f"Logits shape: {logits.shape}")
    generated = model.generate(B=2)
    print(f"Generated shape: {generated.shape}")