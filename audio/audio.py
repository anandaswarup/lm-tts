"""
Interface for NeuCodec model to encode and decode audio data.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from neucodec import DistillNeuCodec


class NeucodecProcessor(nn.Module):
    """
    NeucodecProcessor provides an interface to the Neucodec model for audio encoding and decoding.
    """

    def __init__(self) -> None:
        super().__init__()

        # Define the input and output sample rates
        # Nucodec input sample rate is 16kHz, output sample rate is 24kHz, so we need to resample the input audio data
        # accordingly.
        self.in_sample_rate = 16000
        self.out_sample_rate = 24000

        # Initialize the Neucodec model; put it in evaluation mode since we won't be training it.
        self.model = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec")
        self.model.eval()

        # Place it on the available device (GPU if available, otherwise CPU).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model.to(device)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio data from a file and resample it to the input sample rate.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Resampled audio data as a tensor.
        """
        # Load the audio file
        audio, sample_rate = torchaudio.load(audio_path)

        # Make the audio mono if it's stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample the audio to the input sample rate if necessary
        if sample_rate != self.in_sample_rate:
            audio = T.Resample(orig_freq=sample_rate, new_freq=self.in_sample_rate)(
                audio
            )

        # Add a batch dimension to the audio tensor
        audio = audio[None, ...]

        return audio

    def _save_audio(self, audio_tensor: torch.Tensor, output_path: str) -> None:
        """
        Save audio data from a tensor to a wav file

        Args:
            audio_tensor (torch.Tensor): Audio data as a tensor.
            output_path (str): Path to save the output audio file.
        """
        # Remove the batch dimension
        audio_tensor = audio_tensor[0, :, :]

        # Write the audio tensor to a wav file
        torchaudio.save(output_path, audio_tensor.cpu(), self.out_sample_rate)
        print(f"Saved audio to: {output_path}")

    def encode(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode audio data using the Neucodec model.

        Args:
            audio_tensor (torch.Tensor): Input audio data as a tensor.

        Returns:
            torch.Tensor: Encoded audio representation.
        """
        with torch.no_grad():
            fsq_codes = self.model.encode_code(audio_tensor)  # type: ignore[operator]

        print(f"Encoded audio using FSQ codes with shape: {fsq_codes.shape}")

        return fsq_codes

    def decode(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode FSQ codes back into audio data using the Neucodec model.

        Args:
            fsq_codes (torch.Tensor): Encoded audio representation as FSQ codes.

        Returns:
            torch.Tensor: Decoded audio data as a tensor.
        """
        with torch.no_grad():
            decoded_audio = self.model.decode_code(fsq_codes)  # type: ignore[operator]

        print(f"Decoded audio with shape: {decoded_audio.shape}")

        return decoded_audio
