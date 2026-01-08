"""
Unit tests for clone_voice.py script.

Run with: uv run pytest tests/unit/test_clone_voice.py -v
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestValidateInputFile:
    """Tests for input file validation."""

    def test_nonexistent_file_raises_error(self):
        """Should raise FileNotFoundError for nonexistent input."""
        from clone_voice import validate_input_file

        with pytest.raises(FileNotFoundError, match="not found"):
            validate_input_file("/nonexistent/path/voice.wav")

    def test_non_wav_file_raises_error(self):
        """Should raise ValueError for non-WAV files."""
        from clone_voice import validate_input_file

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake mp3 content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="WAV"):
                validate_input_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_valid_wav_returns_path(self):
        """Should return Path for valid WAV file."""
        from clone_voice import validate_input_file

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)  # Minimal WAV-like header
            temp_path = f.name

        try:
            result = validate_input_file(temp_path)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(temp_path)

    def test_case_insensitive_extension(self):
        """Should accept .WAV extension (uppercase)."""
        from clone_voice import validate_input_file

        with tempfile.NamedTemporaryFile(suffix=".WAV", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            temp_path = f.name

        try:
            result = validate_input_file(temp_path)
            assert result.exists()
        finally:
            os.unlink(temp_path)


class TestValidateVoiceName:
    """Tests for voice name validation."""

    def test_empty_name_raises_error(self):
        """Should raise ValueError for empty voice name."""
        from clone_voice import validate_voice_name

        with pytest.raises(ValueError, match="empty"):
            validate_voice_name("")

    def test_name_with_slashes_raises_error(self):
        """Should raise ValueError for names with path separators."""
        from clone_voice import validate_voice_name

        with pytest.raises(ValueError, match="invalid"):
            validate_voice_name("path/to/voice")

    def test_valid_name_returns_cleaned(self):
        """Should return cleaned voice name."""
        from clone_voice import validate_voice_name

        result = validate_voice_name("my_voice")
        assert result == "my_voice"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        from clone_voice import validate_voice_name

        result = validate_voice_name("  my_voice  ")
        assert result == "my_voice"

    def test_name_with_spaces_allowed(self):
        """Should allow names with internal spaces."""
        from clone_voice import validate_voice_name

        result = validate_voice_name("My Voice Name")
        assert result == "My Voice Name"


class TestGetOutputPath:
    """Tests for output path generation."""

    def test_default_output_dir(self):
        """Should use assets/ by default."""
        from clone_voice import get_output_path

        result = get_output_path("test_voice", None)
        assert "assets" in str(result)
        assert result.name == "test_voice.safetensors"

    def test_custom_output_dir(self):
        """Should use custom output directory when provided."""
        from clone_voice import get_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_output_path("test_voice", tmpdir)
            assert str(result).startswith(tmpdir)
            assert result.name == "test_voice.safetensors"

    def test_output_has_safetensors_extension(self):
        """Should always have .safetensors extension."""
        from clone_voice import get_output_path

        result = get_output_path("my_voice", None)
        assert result.suffix == ".safetensors"


class TestExtractAndSaveConditionals:
    """Tests for conditionals extraction and saving."""

    def test_saves_to_correct_path(self):
        """Should save safetensors to specified path."""
        from clone_voice import extract_and_save_conditionals

        import mlx.core as mx

        mock_model = MagicMock()
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1]]),
            "prompt_token_len": mx.array([1]),
            "prompt_feat": mx.array([[[0.1]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5]]),
        }
        mock_model._conds = mock_conds
        mock_model.prepare_conditionals = MagicMock(return_value=None)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            input_path = Path(f.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.safetensors"

            try:
                result = extract_and_save_conditionals(mock_model, input_path, output_path)

                assert output_path.exists()
                assert result == output_path
                mock_model.prepare_conditionals.assert_called_once()
            finally:
                input_path.unlink(missing_ok=True)

    def test_returns_file_sizes(self):
        """Should return input and output file sizes."""
        from clone_voice import extract_and_save_conditionals

        import mlx.core as mx

        mock_model = MagicMock()
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1]]),
            "prompt_token_len": mx.array([1]),
            "prompt_feat": mx.array([[[0.1]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5]]),
        }
        mock_model._conds = mock_conds
        mock_model.prepare_conditionals = MagicMock(return_value=None)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 1000)  # ~1KB input
            input_path = Path(f.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.safetensors"

            try:
                result, input_size, output_size = extract_and_save_conditionals(
                    mock_model, input_path, output_path, return_sizes=True
                )

                assert input_size > 0
                assert output_size > 0
            finally:
                input_path.unlink(missing_ok=True)


class TestFlattenConditionals:
    """Tests for conditionals flattening logic."""

    def test_flattens_t3_fields(self):
        """Should flatten T3 conditionals with t3_ prefix."""
        from clone_voice import flatten_conditionals

        import mlx.core as mx

        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2]])
        mock_conds.t3.clap_emb = mx.array([[0.3]])
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {}

        result = flatten_conditionals(mock_conds)

        assert "t3_speaker_emb" in result
        assert "t3_cond_prompt_speech_tokens" in result
        assert "t3_clap_emb" in result

    def test_flattens_gen_dict(self):
        """Should flatten gen dict with gen_ prefix."""
        from clone_voice import flatten_conditionals

        import mlx.core as mx

        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1, 2]]),
            "embedding": mx.array([[0.5, 0.6]]),
        }

        result = flatten_conditionals(mock_conds)

        assert "gen_prompt_token" in result
        assert "gen_embedding" in result

    def test_skips_none_values(self):
        """Should skip None values in T3 conditionals."""
        from clone_voice import flatten_conditionals

        import mlx.core as mx

        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1]])
        mock_conds.t3.clap_emb = None  # Should be skipped
        mock_conds.t3.cond_prompt_speech_emb = None  # Should be skipped
        mock_conds.t3.emotion_adv = None  # Should be skipped
        mock_conds.gen = {}

        result = flatten_conditionals(mock_conds)

        assert "t3_clap_emb" not in result
        assert "t3_cond_prompt_speech_emb" not in result
        assert "t3_emotion_adv" not in result


class TestFormatFileSize:
    """Tests for file size formatting."""

    def test_bytes(self):
        """Should format bytes correctly."""
        from clone_voice import format_file_size

        assert format_file_size(500) == "500 B"

    def test_kilobytes(self):
        """Should format kilobytes correctly."""
        from clone_voice import format_file_size

        assert format_file_size(1536) == "1.50 KB"

    def test_megabytes(self):
        """Should format megabytes correctly."""
        from clone_voice import format_file_size

        assert format_file_size(1_500_000) == "1.43 MB"


class TestMainCLI:
    """Tests for CLI argument parsing and main function."""

    def test_requires_input_and_voice_name(self):
        """Should require both input file and voice name arguments."""
        from clone_voice import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

        with pytest.raises(SystemExit):
            parse_args(["input.wav"])

    def test_accepts_optional_output_dir(self):
        """Should accept optional --output-dir argument."""
        from clone_voice import parse_args

        args = parse_args(["input.wav", "my_voice", "--output-dir", "/tmp"])
        assert args.output_dir == "/tmp"

    def test_default_output_dir_is_none(self):
        """Should default output_dir to None (uses assets/)."""
        from clone_voice import parse_args

        args = parse_args(["input.wav", "my_voice"])
        assert args.output_dir is None
