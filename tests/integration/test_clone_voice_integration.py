"""
Integration tests for clone_voice.py script.

These tests require MLX hardware and the voice reference file.
Run with: uv run pytest tests/integration/test_clone_voice_integration.py -v
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestCloneVoiceIntegration:
    """Integration tests for clone_voice with real MLX model."""

    @pytest.fixture
    def default_voice_path(self):
        """Path to the default voice WAV file."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        plugin_root = scripts_dir.parent
        return plugin_root / "assets" / "default_voice.wav"

    def test_extract_conditionals_with_real_model(self, default_voice_path):
        """Test extracting conditionals from real WAV with real model."""
        from clone_voice import extract_and_save_conditionals, validate_input_file
        from mlx_audio.tts.utils import load_model

        # Validate input
        input_path = validate_input_file(str(default_voice_path))

        # Load model
        model = load_model("mlx-community/chatterbox-turbo-fp16")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_voice.safetensors"

            # Extract and save
            result, input_size, output_size = extract_and_save_conditionals(
                model, input_path, output_path, return_sizes=True
            )

            # Verify output
            assert result.exists()
            assert output_path.exists()
            assert input_size > 0
            assert output_size > 0

            # Verify compression ratio (should be ~6x smaller)
            ratio = input_size / output_size
            assert ratio > 3, f"Expected >3x compression, got {ratio:.1f}x"

    def test_output_is_loadable_safetensors(self, default_voice_path):
        """Test that output can be loaded as valid safetensors."""
        import mlx.core as mx
        from clone_voice import extract_and_save_conditionals, validate_input_file
        from mlx_audio.tts.utils import load_model

        input_path = validate_input_file(str(default_voice_path))
        model = load_model("mlx-community/chatterbox-turbo-fp16")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_voice.safetensors"

            extract_and_save_conditionals(model, input_path, output_path)

            # Load and verify structure
            loaded = mx.load(str(output_path))

            # Check required keys
            assert "t3_speaker_emb" in loaded
            assert "t3_cond_prompt_speech_tokens" in loaded

            # Check gen keys exist
            gen_keys = [k for k in loaded.keys() if k.startswith("gen_")]
            assert len(gen_keys) > 0, "Should have gen_ prefixed keys"

    def test_main_function_end_to_end(self, default_voice_path, monkeypatch):
        """Test main() function with real inputs (no interactive prompts)."""
        from clone_voice import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = tmpdir
            voice_name = "integration_test_voice"

            # Run main with arguments
            exit_code = main([
                str(default_voice_path),
                voice_name,
                "--output-dir", output_dir,
            ])

            assert exit_code == 0

            # Verify output file created
            expected_output = Path(output_dir) / f"{voice_name}.safetensors"
            assert expected_output.exists()

    def test_output_file_size_is_reasonable(self, default_voice_path):
        """Test that output file size is in expected range (~160KB)."""
        from clone_voice import extract_and_save_conditionals, validate_input_file
        from mlx_audio.tts.utils import load_model

        input_path = validate_input_file(str(default_voice_path))
        model = load_model("mlx-community/chatterbox-turbo-fp16")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_voice.safetensors"

            _, input_size, output_size = extract_and_save_conditionals(
                model, input_path, output_path, return_sizes=True
            )

            # Expected output ~160KB based on issue description
            # Allow range of 100KB - 300KB to account for model variations
            assert 100_000 < output_size < 300_000, (
                f"Output size {output_size} not in expected range 100KB-300KB"
            )
