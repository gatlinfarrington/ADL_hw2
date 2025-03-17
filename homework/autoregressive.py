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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.to_logits = nn.Linear(d_latent, n_tokens)
        nn.init.xavier_uniform_(self.to_logits.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # print(f"Input shape: {x.shape}")
        B = x.shape[0]
        if x.dim() == 3:
            x = x.view(B, -1)
        # print(f"After flatten: {x.shape}")
        x = self.embedding(x)
        # print(f"After embedding: {x.shape}")
        x = x + self.pos_embedding[:, :x.shape[1]]
        # print(f"After pos embedding: {x.shape}")
        
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        # print(f"Mask shape: {mask.shape}")
        
        # Flatten spatial dimensions before passing to Transformer
        B, C, H, W, D = x.shape  # Expected shape: (batch, 1, height, width, embedding_dim)
        x = x.view(B, H * W, D)  # Reshape to (batch, sequence_length, embedding_dim)

        # Ensure mask shape matches sequence length
        seq_len = H * W
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)  # Causal mask

        x = self.transformer(x, mask=mask)

        # print(f"After transformer: {x.shape}")
        logits = self.to_logits(x)
        # print(f"Logits shape before view: {logits.shape}")
        if x.shape[1] == self.seq_len:
            logits = logits.view(B, self.h, self.w, self.n_tokens)
        # print(f"Final logits shape: {logits.shape}")
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
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
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token_indices = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_indices)
                tokens[:, t] = next_token.squeeze(-1)
                if t % 100 == 0:
                    print(f"Token at position {t}: {next_token.squeeze(-1).cpu().numpy()}")
        print(f"Final tokens shape: {tokens.shape}")
        print(f"Final tokens unique values: {torch.unique(tokens).cpu().numpy()}")
        return tokens.view(B, h, w)

if __name__ == "__main__":
    model = AutoregressiveModel(d_latent=256, n_tokens=1024)
    x = torch.randint(0, 1024, (2, 30, 20))
    logits, _ = model(x)
    print(f"Logits shape: {logits.shape}")
    generated = model.generate(B=2)
    print(f"Generated shape: {generated.shape}")