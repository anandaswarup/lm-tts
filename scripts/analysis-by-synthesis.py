"""
Analysis by synthesis script to encode a wav file into FSQ codes using the NeuCodec model, and then decode the FSQ
codes back into audio data, saving the output as a new wav file. This script demonstrates the encoding and decoding
process of the NeuCodec model, allowing for analysis of the synthesized audio compared to the original input audio.
"""

import argparse
import warnings

from audio.audio import NeucodecProcessor

warnings.filterwarnings("ignore")


def main():
    """
    Main method to execute the analysis by synthesis process. It parses command-line arguments for input and output
    paths, initializes the AudioProcessor, and performs encoding and decoding of the audio data.
    """
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Analysis by synthesis script to evaluate the NeuCodec model's encoding and decoding capabilities."
    )
    parser.add_argument(
        "--input_wav",
        type=str,
        required=True,
        help="Path to the input wav file to be processed.",
    )
    parser.add_argument(
        "--output_wav",
        type=str,
        required=True,
        help="Path to save the output wav file after decoding.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    input_wav_path = args.input_wav
    output_wav_path = args.output_wav

    # Initialize the NeucodecProcessor
    processor = NeucodecProcessor()

    # Load and preprocess the input audio
    audio_tensor = processor._load_audio(input_wav_path)
    print(f"Loaded audio from: {input_wav_path} with shape: {audio_tensor.shape}")

    # Encode the audio using the Neucodec model
    fsq_codes = processor.encode(audio_tensor)

    # Decode the FSQ codes back into audio data
    decoded_audio = processor.decode(fsq_codes)

    # Save the decoded audio to the specified output path
    processor._save_audio(decoded_audio, output_wav_path)


if __name__ == "__main__":
    main()
