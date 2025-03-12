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
        nn.init.xavier_uniform_(self.embedding.weight)  # Better initialization
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_latent) * 0.01)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_latent,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ) for _ in range(4)
        ])
        self.to_logits = nn.Linear(d_latent, n_tokens)
        nn.init.xavier_uniform_(self.to_logits.weight)  # Better initialization

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = x.shape[0]
        if x.dim() == 3:
            x = x.view(B, -1)
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        for layer in self.transformer:
            x = layer(x, mask)
        logits = self.to_logits(x)
        if x.shape[1] == self.seq_len:
            logits = logits.view(B, self.h, self.w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        seq_len = h * w
        tokens = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        for t in range(seq_len):
            with torch.no_grad():
                logits, _ = self.forward(tokens[:, :t+1])
                probs = nn.functional.softmax(logits[:, -1], dim=-1)
                if probs.dim() > 2:
                    probs = probs.view(B, -1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens[:, t] = next_token.squeeze(-1)
                print(f"Token at position {t}: {next_token.squeeze(-1).cpu().numpy()}")  # Debug
        print(f"Final tokens: {tokens.cpu().numpy()}")  # Debug
        return tokens.view(B, h, w)

if __name__ == "__main__":
    model = AutoregressiveModel(d_latent=256, n_tokens=1024)
    x = torch.randint(0, 1024, (2, 30, 20))
    logits, _ = model(x)
    print(f"Logits shape: {logits.shape}")
    generated = model.generate(B=2)
    print(f"Generated shape: {generated.shape}")