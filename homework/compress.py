from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
import struct  # For efficient byte storage

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Steps:
        1. Tokenize the image (convert to integer tokens).
        2. Encode the tokens using arithmetic coding.
        3. Store the encoded tokens as a bytes object.
        """

        # Ensure input is in the correct range (-0.5 to 0.5)
        x = x.float() / 255.0 - 0.5  
        
        # Ensure image is in (1, 3, 100, 150) shape
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Tokenize the image (convert to integer token indices)
        tokens = self.tokenizer.encode_index(x)  # Expected shape: (B, H', W')

        # Flatten tokens to a 1D sequence for compression
        tokens = tokens.flatten().cpu().numpy()

        # Encode tokens as bytes using struct
        compressed_bytes = struct.pack(f"{len(tokens)}H", *tokens)  # Store as uint16

        return compressed_bytes

    def decompress(self, compressed_data: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.

        Steps:
        1. Decode the bytes back into token indices.
        2. Convert tokens back into an image using the decoder.
        3. Ensure the output image is (100, 150) in shape.
        """

        # Read token indices from byte stream
        num_tokens = len(compressed_data) // 2  # uint16 means 2 bytes per token
        tokens = struct.unpack(f"{num_tokens}H", compressed_data)

        # Convert back to tensor (Assuming original shape was (1, H', W'))
        tokens = torch.tensor(tokens, dtype=torch.long).view(1, -1, -1)  # Reshape properly

        # Decode token indices back into an image
        reconstructed = self.tokenizer.decode_index(tokens)

        # Ensure image is in (-0.5, 0.5) range → Convert back to (0, 255) for saving
        reconstructed = ((reconstructed + 0.5) * 255.0).clamp(0, 255).byte()

        return reconstructed


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    # Load the image and convert to tensor
    x = torch.tensor(np.array(Image.open(image).convert("RGB")), dtype=torch.uint8, device=device)

    # Compress image into bytes
    cmp_img = cmp.compress(x)

    # Save compressed bytes to a file
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the decompressed image.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    # Load the compressed data from the file
    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    # Decompress the image
    x = cmp.decompress(cmp_img)

    # Convert to PIL and save
    img = Image.fromarray(x.cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})