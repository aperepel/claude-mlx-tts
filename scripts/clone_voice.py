#!/usr/bin/env python3
"""
Clone Voice - Pre-compute voice embeddings from WAV files.

Extracts voice conditionals from a WAV file and saves them as .safetensors
for efficient distribution and loading. Pre-computed embeddings are ~6x smaller
than the original WAV files and eliminate runtime extraction overhead.

Usage:
    python clone_voice.py input.wav my_voice
    python clone_voice.py input.wav my_voice --output-dir /custom/path
"""
import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import Conditionals

# Default output directory (assets/ in plugin root)
_SCRIPT_DIR = Path(__file__).parent
_PLUGIN_ROOT = _SCRIPT_DIR.parent
DEFAULT_ASSETS_DIR = _PLUGIN_ROOT / "assets"


def validate_input_file(path: str) -> Path:
    """
    Validate that input file exists and is a WAV file.

    Args:
        path: Path to the input WAV file.

    Returns:
        Path object for the validated file.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not a WAV file.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if input_path.suffix.lower() != ".wav":
        raise ValueError(f"Input must be a WAV file, got: {input_path.suffix}")

    return input_path


def validate_voice_name(name: str) -> str:
    """
    Validate and clean voice name.

    Args:
        name: Voice name to validate.

    Returns:
        Cleaned voice name.

    Raises:
        ValueError: If name is empty or contains invalid characters.
    """
    cleaned = name.strip()

    if not cleaned:
        raise ValueError("Voice name cannot be empty")

    if "/" in cleaned or "\\" in cleaned:
        raise ValueError("Voice name contains invalid path separators")

    return cleaned


def get_output_path(voice_name: str, output_dir: str | None) -> Path:
    """
    Generate output path for the safetensors file.

    Args:
        voice_name: Name of the voice.
        output_dir: Optional custom output directory.

    Returns:
        Path for the output safetensors file.
    """
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = DEFAULT_ASSETS_DIR

    return base_dir / f"{voice_name}.safetensors"


def flatten_conditionals(conds: "Conditionals") -> dict[str, mx.array]:
    """
    Flatten conditionals into a dict of arrays for safetensors saving.

    Args:
        conds: Conditionals object to flatten.

    Returns:
        Dict mapping key names to mlx arrays.
    """
    arrays = {}

    # T3 conditionals
    arrays["t3_speaker_emb"] = conds.t3.speaker_emb
    arrays["t3_cond_prompt_speech_tokens"] = conds.t3.cond_prompt_speech_tokens

    if conds.t3.clap_emb is not None:
        arrays["t3_clap_emb"] = conds.t3.clap_emb
    if conds.t3.cond_prompt_speech_emb is not None:
        arrays["t3_cond_prompt_speech_emb"] = conds.t3.cond_prompt_speech_emb
    if conds.t3.emotion_adv is not None:
        arrays["t3_emotion_adv"] = conds.t3.emotion_adv

    # Gen conditionals
    for key, value in conds.gen.items():
        arrays[f"gen_{key}"] = value

    return arrays


@overload
def extract_and_save_conditionals(
    model: Any,
    input_path: Path,
    output_path: Path,
    return_sizes: Literal[False] = False,
) -> Path: ...


@overload
def extract_and_save_conditionals(
    model: Any,
    input_path: Path,
    output_path: Path,
    return_sizes: Literal[True],
) -> tuple[Path, int, int]: ...


def extract_and_save_conditionals(
    model: Any,
    input_path: Path,
    output_path: Path,
    return_sizes: bool = False,
) -> Path | tuple[Path, int, int]:
    """
    Extract voice conditionals from WAV and save as safetensors.

    Args:
        model: Loaded TTS model with prepare_conditionals method.
        input_path: Path to input WAV file.
        output_path: Path for output safetensors file.
        return_sizes: If True, also return file sizes.

    Returns:
        Output path, or (output_path, input_size, output_size) if return_sizes=True.
    """
    # Get input file size
    input_size = input_path.stat().st_size

    # Prepare conditionals from WAV
    model.prepare_conditionals(str(input_path))
    conds = model._conds

    # Flatten and save
    arrays = flatten_conditionals(conds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path), arrays)

    # Get output file size
    output_size = output_path.stat().st_size

    if return_sizes:
        return output_path, input_size, output_size
    return output_path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "1.50 KB", "2.30 MB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: List of arguments (uses sys.argv if None).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Pre-compute voice embeddings from WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clone_voice.py voice.wav my_voice
    python clone_voice.py voice.wav daniel --output-dir ./voices
        """,
    )

    parser.add_argument(
        "input_file",
        help="Path to input WAV file (7-20 seconds recommended)",
    )
    parser.add_argument(
        "voice_name",
        help="Name for the cloned voice (used as output filename)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=f"Output directory (default: {DEFAULT_ASSETS_DIR})",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for clone_voice script.

    Args:
        args: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parsed = parse_args(args)

    try:
        # Validate inputs
        input_path = validate_input_file(parsed.input_file)
        voice_name = validate_voice_name(parsed.voice_name)
        output_path = get_output_path(voice_name, parsed.output_dir)

        # Check if output already exists
        if output_path.exists():
            print(f"Warning: Output file already exists: {output_path}")
            response = input("Overwrite? [y/N]: ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        # Load model
        print("Loading TTS model...")
        from mlx_tts_core import load_tts_model
        model = load_tts_model()

        # Extract and save
        print(f"Extracting voice conditionals from: {input_path}")
        result, input_size, output_size = extract_and_save_conditionals(
            model, input_path, output_path, return_sizes=True
        )

        # Report success
        ratio = input_size / output_size if output_size > 0 else 0
        print(f"\nSuccess! Voice embeddings saved to: {result}")
        print(f"  Input:  {format_file_size(input_size)}")
        print(f"  Output: {format_file_size(output_size)}")
        print(f"  Compression: {ratio:.1f}x smaller")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print("Error: mlx_audio not installed. Run: uv sync --extra mlx", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
