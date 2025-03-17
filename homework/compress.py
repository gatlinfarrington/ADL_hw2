from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

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
        Uses arithmetic coding for token compression.
        """
        assert x.shape == (150, 100, 3), "Input image must be 150x100 pixels with 3 channels."
        
        # Convert image to tokens
        tokens = self.tokenizer.encode_index(x.float() / 255.0 - 0.5)  # (B, H, W)
        
        # Flatten tokens for efficient encoding
        tokens_flat = tokens.view(-1).cpu().numpy()  # Convert to NumPy array
        
        # Apply Run-Length Encoding (RLE)
        rle_encoded = self._run_length_encode(tokens_flat)
        
        # Use arithmetic coding for final compression
        compressed_bytes = self._arithmetic_encode(rle_encoded)

        return compressed_bytes

    def _run_length_encode(self, array):
        """Applies simple Run-Length Encoding (RLE) to reduce repeated values."""
        values = []
        counts = []
        prev = array[0]
        count = 1

        for i in range(1, len(array)):
            if array[i] == prev:
                count += 1
            else:
                values.append(prev)
                counts.append(count)
                prev = array[i]
                count = 1

        values.append(prev)
        counts.append(count)
        
        return np.array(list(zip(values, counts)), dtype=np.uint16)

    def _arithmetic_encode(self, rle_encoded):
        """Applies arithmetic coding to the run-length encoded data."""
        import io
        bitout = arithmeticcoding.BitOutputStream(io.BytesIO())

        # Initialize encoder
        freq_table = arithmeticcoding.FlatFrequencyTable(256)  # Adjust based on token range
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

        for value, count in rle_encoded:
            enc.write(freq_table, value)  # Encode token
            enc.write(freq_table, count)  # Encode count

        enc.finish()
        return bitout.stream.getvalue()

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        """
        # Decode from arithmetic coding
        rle_decoded = self._arithmetic_decode(x)
        
        # Reverse RLE to get original token sequence
        tokens_flat = self._run_length_decode(rle_decoded)

        # Reshape into (B, H, W)
        h, w = 150 // self.tokenizer.patch_size, 100 // self.tokenizer.patch_size
        tokens = torch.tensor(tokens_flat, dtype=torch.int64).view(1, h, w)

        # Decode tokens back into an image
        reconstructed_image = self.tokenizer.decode_index(tokens)

        return reconstructed_image

    def _run_length_decode(self, rle_encoded):
        """Decodes an RLE-encoded sequence back to the original array."""
        decoded = []
        for value, count in rle_encoded:
            decoded.extend([value] * count)
        return np.array(decoded, dtype=np.uint8)

    def _arithmetic_decode(self, compressed_bytes):
        """Applies arithmetic decoding to get back the RLE data."""
        import io
        bitin = arithmeticcoding.BitInputStream(io.BytesIO(compressed_bytes))

        freq_table = arithmeticcoding.FlatFrequencyTable(256)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

        decoded = []
        while True:
            try:
                value = dec.read(freq_table)
                count = dec.read(freq_table)
                decoded.append((value, count))
            except EOFError:
                break

        return np.array(decoded, dtype=np.uint16)


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

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
