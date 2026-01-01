"""
Unit tests for voice conditionals caching module.

Run with: uv run pytest tests/unit/test_voice_cache.py -v
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestCacheDir:
    """Tests for CACHE_DIR constant."""

    def test_cache_dir_is_in_plugin_cache(self):
        """CACHE_DIR should be under <plugin>/.cache/voice_conds."""
        from voice_cache import CACHE_DIR, _PLUGIN_ROOT

        expected = _PLUGIN_ROOT / ".cache" / "voice_conds"
        assert CACHE_DIR == expected

    def test_cache_dir_is_path_object(self):
        """CACHE_DIR should be a Path object."""
        from voice_cache import CACHE_DIR

        assert isinstance(CACHE_DIR, Path)


class TestGetCachePath:
    """Tests for get_cache_path function."""

    def test_returns_path_object(self):
        """get_cache_path should return a Path object."""
        from voice_cache import get_cache_path

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"test audio content")
            temp_path = f.name

        try:
            result = get_cache_path(temp_path)
            assert isinstance(result, Path)
        finally:
            os.unlink(temp_path)

    def test_uses_content_hash(self):
        """get_cache_path should use SHA256[:16] of file content."""
        from voice_cache import get_cache_path, CACHE_DIR

        content = b"test audio content for hashing"
        expected_hash = hashlib.sha256(content).hexdigest()[:16]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = get_cache_path(temp_path)
            assert expected_hash in str(result)
            assert result.parent == CACHE_DIR
            assert result.suffix == ".safetensors"
        finally:
            os.unlink(temp_path)

    def test_same_content_same_hash(self):
        """Files with same content should produce same cache path."""
        from voice_cache import get_cache_path

        content = b"identical content"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
            f1.write(content)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
            f2.write(content)
            path2 = f2.name

        try:
            result1 = get_cache_path(path1)
            result2 = get_cache_path(path2)
            assert result1 == result2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_different_content_different_hash(self):
        """Files with different content should produce different cache paths."""
        from voice_cache import get_cache_path

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
            f1.write(b"content A")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
            f2.write(b"content B")
            path2 = f2.name

        try:
            result1 = get_cache_path(path1)
            result2 = get_cache_path(path2)
            assert result1 != result2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_nonexistent_file_raises_error(self):
        """get_cache_path should raise FileNotFoundError for nonexistent file."""
        from voice_cache import get_cache_path

        with pytest.raises(FileNotFoundError):
            get_cache_path("/nonexistent/path/voice.wav")


class TestIsCacheValid:
    """Tests for is_cache_valid function."""

    def test_returns_false_when_no_cache(self):
        """is_cache_valid should return False when cache doesn't exist."""
        from voice_cache import is_cache_valid

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"test content no cache")
            temp_path = f.name

        try:
            result = is_cache_valid(temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)

    def test_returns_true_when_cache_exists(self):
        """is_cache_valid should return True when cache file exists."""
        from voice_cache import is_cache_valid, get_cache_path, CACHE_DIR

        content = b"cached audio content"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            cache_path = get_cache_path(temp_path)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Create dummy cache file
            cache_path.write_bytes(b"dummy cache data")

            result = is_cache_valid(temp_path)
            assert result is True

            # Cleanup
            cache_path.unlink(missing_ok=True)
        finally:
            os.unlink(temp_path)

    def test_nonexistent_voice_file_returns_false(self):
        """is_cache_valid should return False for nonexistent voice file."""
        from voice_cache import is_cache_valid

        result = is_cache_valid("/nonexistent/path/voice.wav")
        assert result is False


class TestSaveConditionals:
    """Tests for save_conditionals function."""

    def test_creates_cache_directory(self):
        """save_conditionals should create cache directory if needed."""
        from voice_cache import save_conditionals, get_cache_path, CACHE_DIR

        # Create mock conditionals with mlx arrays
        import mlx.core as mx
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2, 0.3]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2, 3]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1, 2]]),
            "prompt_token_len": mx.array([2]),
            "prompt_feat": mx.array([[[0.1, 0.2]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5, 0.6]]),
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"voice content")
            temp_path = f.name

        cache_path = None
        try:
            cache_path = get_cache_path(temp_path)  # Get before file deletion
            result = save_conditionals(mock_conds, temp_path)
            assert CACHE_DIR.exists()
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(temp_path)
            if cache_path:
                cache_path.unlink(missing_ok=True)

    def test_returns_cache_path(self):
        """save_conditionals should return the cache file path."""
        from voice_cache import save_conditionals, get_cache_path

        import mlx.core as mx
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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"voice content for save")
            temp_path = f.name

        result = None
        try:
            result = save_conditionals(mock_conds, temp_path)
            expected = get_cache_path(temp_path)
            assert result == expected
        finally:
            os.unlink(temp_path)
            if result:
                result.unlink(missing_ok=True)

    def test_saves_as_safetensors(self):
        """save_conditionals should save data in safetensors format."""
        from voice_cache import save_conditionals, get_cache_path

        import mlx.core as mx
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2, 0.3]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2, 3]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1, 2]]),
            "prompt_token_len": mx.array([2]),
            "prompt_feat": mx.array([[[0.1, 0.2]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5, 0.6]]),
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"voice for safetensors test")
            temp_path = f.name

        result = None
        try:
            result = save_conditionals(mock_conds, temp_path)

            # Verify it's a valid safetensors file by loading it
            loaded = mx.load(str(result))
            assert isinstance(loaded, dict)
            assert "t3_speaker_emb" in loaded
            assert "gen_prompt_token" in loaded
        finally:
            os.unlink(temp_path)
            if result:
                result.unlink(missing_ok=True)


class TestLoadConditionals:
    """Tests for load_conditionals function."""

    def test_returns_none_when_no_cache(self):
        """load_conditionals should return None when cache doesn't exist."""
        from voice_cache import load_conditionals

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"uncached voice content")
            temp_path = f.name

        try:
            result = load_conditionals(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)

    def test_loads_cached_conditionals(self):
        """load_conditionals should load and return cached Conditionals."""
        from voice_cache import load_conditionals, save_conditionals

        import mlx.core as mx
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2, 0.3]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2, 3]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1, 2]]),
            "prompt_token_len": mx.array([2]),
            "prompt_feat": mx.array([[[0.1, 0.2]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5, 0.6]]),
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"voice for load test")
            temp_path = f.name

        try:
            # Save first
            cache_path = save_conditionals(mock_conds, temp_path)

            # Load and verify
            result = load_conditionals(temp_path)
            assert result is not None
            assert hasattr(result, "t3")
            assert hasattr(result, "gen")

            # Cleanup
            cache_path.unlink(missing_ok=True)
        finally:
            os.unlink(temp_path)

    def test_returns_none_for_nonexistent_voice_file(self):
        """load_conditionals should return None for nonexistent voice file."""
        from voice_cache import load_conditionals

        result = load_conditionals("/nonexistent/path/voice.wav")
        assert result is None

    def test_loaded_conditionals_have_correct_data(self):
        """Loaded conditionals should have the same data as saved."""
        from voice_cache import load_conditionals, save_conditionals

        import mlx.core as mx
        import numpy as np

        original_speaker_emb = mx.array([[0.123, 0.456, 0.789]])
        original_tokens = mx.array([[10, 20, 30]])

        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = original_speaker_emb
        mock_conds.t3.cond_prompt_speech_tokens = original_tokens
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1, 2]]),
            "prompt_token_len": mx.array([2]),
            "prompt_feat": mx.array([[[0.1, 0.2]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5, 0.6]]),
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"voice for data test")
            temp_path = f.name

        try:
            cache_path = save_conditionals(mock_conds, temp_path)
            result = load_conditionals(temp_path)

            # Verify data integrity
            assert result is not None
            assert np.allclose(
                np.array(result.t3.speaker_emb),
                np.array(original_speaker_emb)
            )
            assert np.array_equal(
                np.array(result.t3.cond_prompt_speech_tokens),
                np.array(original_tokens)
            )

            cache_path.unlink(missing_ok=True)
        finally:
            os.unlink(temp_path)


class TestGetOrPrepareConditionals:
    """Tests for get_or_prepare_conditionals function."""

    def test_returns_cached_on_cache_hit(self):
        """get_or_prepare_conditionals should return cached conds on hit."""
        from voice_cache import get_or_prepare_conditionals

        mock_model = MagicMock()
        mock_cached = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"cached voice content")
            temp_path = f.name

        try:
            with patch("voice_cache.load_conditionals", return_value=mock_cached):
                result = get_or_prepare_conditionals(mock_model, temp_path)
                assert result == mock_cached
                mock_model.prepare_conditionals.assert_not_called()
        finally:
            os.unlink(temp_path)

    def test_prepares_and_caches_on_miss(self):
        """get_or_prepare_conditionals should prepare and cache on miss."""
        from voice_cache import get_or_prepare_conditionals

        import mlx.core as mx

        mock_model = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.t3 = MagicMock()
        mock_prepared.t3.speaker_emb = mx.array([[0.1]])
        mock_prepared.t3.cond_prompt_speech_tokens = mx.array([[1]])
        mock_prepared.t3.clap_emb = None
        mock_prepared.t3.cond_prompt_speech_emb = None
        mock_prepared.t3.emotion_adv = None
        mock_prepared.gen = {
            "prompt_token": mx.array([[1]]),
            "prompt_token_len": mx.array([1]),
            "prompt_feat": mx.array([[[0.1]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5]]),
        }

        # prepare_conditionals stores in model._conds
        mock_model._conds = mock_prepared
        mock_model.prepare_conditionals = MagicMock(return_value=None)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"uncached voice")
            temp_path = f.name

        try:
            with patch("voice_cache.load_conditionals", return_value=None):
                with patch("voice_cache.save_conditionals") as mock_save:
                    result = get_or_prepare_conditionals(mock_model, temp_path)

                    mock_model.prepare_conditionals.assert_called_once_with(temp_path)
                    mock_save.assert_called_once()
                    assert result == mock_prepared
        finally:
            os.unlink(temp_path)

    def test_logs_cache_hit(self, caplog):
        """get_or_prepare_conditionals should log on cache hit."""
        import logging
        from voice_cache import get_or_prepare_conditionals

        mock_model = MagicMock()
        mock_cached = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"logged cached voice")
            temp_path = f.name

        try:
            with patch("voice_cache.load_conditionals", return_value=mock_cached):
                with caplog.at_level(logging.INFO):
                    get_or_prepare_conditionals(mock_model, temp_path)
                    log_text = caplog.text.lower()
                    assert "cache hit" in log_text or "cached" in log_text or "loaded" in log_text
        finally:
            os.unlink(temp_path)

    def test_logs_cache_miss(self, caplog):
        """get_or_prepare_conditionals should log on cache miss."""
        import logging
        from voice_cache import get_or_prepare_conditionals

        import mlx.core as mx
        mock_model = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.t3 = MagicMock()
        mock_prepared.t3.speaker_emb = mx.array([[0.1]])
        mock_prepared.t3.cond_prompt_speech_tokens = mx.array([[1]])
        mock_prepared.t3.clap_emb = None
        mock_prepared.t3.cond_prompt_speech_emb = None
        mock_prepared.t3.emotion_adv = None
        mock_prepared.gen = {
            "prompt_token": mx.array([[1]]),
            "prompt_token_len": mx.array([1]),
            "prompt_feat": mx.array([[[0.1]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5]]),
        }
        mock_model._conds = mock_prepared
        mock_model.prepare_conditionals = MagicMock(return_value=None)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"logged uncached voice")
            temp_path = f.name

        try:
            with patch("voice_cache.load_conditionals", return_value=None):
                with patch("voice_cache.save_conditionals"):
                    with caplog.at_level(logging.INFO):
                        get_or_prepare_conditionals(mock_model, temp_path)
                        log_text = caplog.text.lower()
                        assert "miss" in log_text or "preparing" in log_text or "caching" in log_text
        finally:
            os.unlink(temp_path)


class TestCacheInvalidation:
    """Tests for cache invalidation when voice file changes."""

    def test_modified_voice_file_invalidates_cache(self):
        """When voice file content changes, cache should be invalid."""
        from voice_cache import get_cache_path, is_cache_valid, CACHE_DIR

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"original content")
            temp_path = f.name

        try:
            # Create cache for original content
            cache_path = get_cache_path(temp_path)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"cached data")

            assert is_cache_valid(temp_path) is True

            # Modify the file
            with open(temp_path, "wb") as f:
                f.write(b"modified content")

            # Now cache should be invalid (different hash = different path)
            new_cache_path = get_cache_path(temp_path)
            assert new_cache_path != cache_path
            assert is_cache_valid(temp_path) is False

            # Cleanup
            cache_path.unlink(missing_ok=True)
        finally:
            os.unlink(temp_path)


class TestVoiceCachePerformance:
    """Performance tests for voice caching."""

    def test_cached_load_under_10ms(self):
        """Loading conditionals from disk cache should be <10ms."""
        import time
        from voice_cache import load_conditionals, save_conditionals, get_cache_path

        import mlx.core as mx

        # Create realistic-sized mock conditionals
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.random.normal((1, 256))
        mock_conds.t3.cond_prompt_speech_tokens = mx.random.randint(0, 1000, (1, 512))
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.random.randint(0, 1000, (1, 256)),
            "prompt_token_len": mx.array([256]),
            "prompt_feat": mx.random.normal((1, 128, 80)),
            "prompt_feat_len": mx.array([128]),
            "embedding": mx.random.normal((1, 512)),
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"voice for perf test")
            temp_path = f.name

        cache_path = None
        try:
            # Save to cache first
            cache_path = save_conditionals(mock_conds, temp_path)

            # Warm up (first load may be slower due to imports)
            _ = load_conditionals(temp_path)

            # Time multiple loads and take median
            times = []
            for _ in range(5):
                start = time.perf_counter()
                result = load_conditionals(temp_path)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
                assert result is not None

            median_ms = sorted(times)[len(times) // 2]
            assert median_ms < 10, f"Median load time {median_ms:.2f}ms exceeds 10ms threshold"

        finally:
            os.unlink(temp_path)
            if cache_path:
                cache_path.unlink(missing_ok=True)


# =============================================================================
# Tests for load_conditionals_from_file (mlx-tts-36a)
# =============================================================================


class TestLoadConditionalsFromFile:
    """Tests for load_conditionals_from_file function."""

    def test_loads_safetensors_directly(self):
        """load_conditionals_from_file should load pre-computed conditionals from safetensors."""
        from voice_cache import load_conditionals_from_file, save_conditionals, CACHE_DIR

        import mlx.core as mx

        # Create a test safetensors file by saving mock conditionals
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1, 0.2, 0.3]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1, 2, 3]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {
            "prompt_token": mx.array([[1, 2]]),
            "prompt_token_len": mx.array([2]),
            "prompt_feat": mx.array([[[0.1, 0.2]]]),
            "prompt_feat_len": mx.array([1]),
            "embedding": mx.array([[0.5, 0.6]]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "test_voice.safetensors"

            # Save mock data to safetensors
            arrays = {}
            arrays["t3_speaker_emb"] = mock_conds.t3.speaker_emb
            arrays["t3_cond_prompt_speech_tokens"] = mock_conds.t3.cond_prompt_speech_tokens
            for key, value in mock_conds.gen.items():
                arrays[f"gen_{key}"] = value
            mx.save_safetensors(str(safetensors_path), arrays)

            # Load and verify
            result = load_conditionals_from_file(safetensors_path)

            assert result is not None
            assert hasattr(result, "t3")
            assert hasattr(result, "gen")

    def test_returns_conditionals_type(self):
        """load_conditionals_from_file should return a Conditionals object."""
        from voice_cache import load_conditionals_from_file

        import mlx.core as mx

        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "test_voice.safetensors"

            # Create minimal valid safetensors
            arrays = {
                "t3_speaker_emb": mx.array([[0.1]]),
                "t3_cond_prompt_speech_tokens": mx.array([[1]]),
                "gen_prompt_token": mx.array([[1]]),
                "gen_prompt_token_len": mx.array([1]),
                "gen_prompt_feat": mx.array([[[0.1]]]),
                "gen_prompt_feat_len": mx.array([1]),
                "gen_embedding": mx.array([[0.5]]),
            }
            mx.save_safetensors(str(safetensors_path), arrays)

            result = load_conditionals_from_file(safetensors_path)

            # Check it's a Conditionals namedtuple-like object
            from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import Conditionals
            assert isinstance(result, Conditionals)

    def test_raises_for_nonexistent_file(self):
        """load_conditionals_from_file should raise FileNotFoundError for missing file."""
        from voice_cache import load_conditionals_from_file

        with pytest.raises(FileNotFoundError):
            load_conditionals_from_file(Path("/nonexistent/path/voice.safetensors"))

    def test_preserves_array_values(self):
        """load_conditionals_from_file should preserve exact array values."""
        from voice_cache import load_conditionals_from_file

        import mlx.core as mx
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "test_voice.safetensors"

            original_emb = mx.array([[0.123, 0.456, 0.789]])
            original_tokens = mx.array([[10, 20, 30]])

            arrays = {
                "t3_speaker_emb": original_emb,
                "t3_cond_prompt_speech_tokens": original_tokens,
                "gen_prompt_token": mx.array([[1]]),
                "gen_prompt_token_len": mx.array([1]),
                "gen_prompt_feat": mx.array([[[0.1]]]),
                "gen_prompt_feat_len": mx.array([1]),
                "gen_embedding": mx.array([[0.5]]),
            }
            mx.save_safetensors(str(safetensors_path), arrays)

            result = load_conditionals_from_file(safetensors_path)

            assert np.allclose(np.array(result.t3.speaker_emb), np.array(original_emb))
            assert np.array_equal(np.array(result.t3.cond_prompt_speech_tokens), np.array(original_tokens))


# =============================================================================
# Tests for get_voice_conditionals (mlx-tts-36a)
# =============================================================================


class TestGetVoiceConditionals:
    """Tests for get_voice_conditionals unified entry point."""

    def test_prefers_safetensors_over_wav(self):
        """get_voice_conditionals should prefer .safetensors over .wav when both exist."""
        from voice_cache import get_voice_conditionals

        import mlx.core as mx

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            # Create both files
            wav_path = assets_dir / "test_voice.wav"
            wav_path.write_bytes(b"dummy wav content")

            safetensors_path = assets_dir / "test_voice.safetensors"
            arrays = {
                "t3_speaker_emb": mx.array([[0.1]]),
                "t3_cond_prompt_speech_tokens": mx.array([[1]]),
                "gen_prompt_token": mx.array([[1]]),
                "gen_prompt_token_len": mx.array([1]),
                "gen_prompt_feat": mx.array([[[0.1]]]),
                "gen_prompt_feat_len": mx.array([1]),
                "gen_embedding": mx.array([[0.5]]),
            }
            mx.save_safetensors(str(safetensors_path), arrays)

            with patch("voice_cache._PLUGIN_ROOT", Path(tmpdir)):
                result = get_voice_conditionals(mock_model, "test_voice")

            # Model should NOT have been called (loaded from safetensors)
            mock_model.prepare_conditionals.assert_not_called()
            assert result is not None

    def test_falls_back_to_wav_when_no_safetensors(self):
        """get_voice_conditionals should use wav when safetensors doesn't exist."""
        from voice_cache import get_voice_conditionals

        import mlx.core as mx

        mock_model = MagicMock()
        mock_conds = MagicMock()
        mock_conds.t3 = MagicMock()
        mock_conds.t3.speaker_emb = mx.array([[0.1]])
        mock_conds.t3.cond_prompt_speech_tokens = mx.array([[1]])
        mock_conds.t3.clap_emb = None
        mock_conds.t3.cond_prompt_speech_emb = None
        mock_conds.t3.emotion_adv = None
        mock_conds.gen = {"embedding": mx.array([[0.5]])}
        mock_model._conds = mock_conds

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            # Only wav file exists
            wav_path = assets_dir / "test_voice.wav"
            wav_path.write_bytes(b"dummy wav content")

            with patch("voice_cache._PLUGIN_ROOT", Path(tmpdir)):
                with patch("voice_cache.get_or_prepare_conditionals") as mock_get:
                    mock_get.return_value = mock_conds
                    result = get_voice_conditionals(mock_model, "test_voice")

            # Should have called get_or_prepare_conditionals for wav
            mock_get.assert_called_once()

    def test_raises_for_nonexistent_voice(self):
        """get_voice_conditionals should raise ValueError when neither format exists."""
        from voice_cache import get_voice_conditionals

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("voice_cache._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="No voice files found"):
                    get_voice_conditionals(mock_model, "nonexistent_voice")

    def test_loads_safetensors_only_voice(self):
        """get_voice_conditionals should work when only .safetensors exists."""
        from voice_cache import get_voice_conditionals

        import mlx.core as mx

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            # Only safetensors file exists (no wav)
            safetensors_path = assets_dir / "embedded_voice.safetensors"
            arrays = {
                "t3_speaker_emb": mx.array([[0.1]]),
                "t3_cond_prompt_speech_tokens": mx.array([[1]]),
                "gen_prompt_token": mx.array([[1]]),
                "gen_prompt_token_len": mx.array([1]),
                "gen_prompt_feat": mx.array([[[0.1]]]),
                "gen_prompt_feat_len": mx.array([1]),
                "gen_embedding": mx.array([[0.5]]),
            }
            mx.save_safetensors(str(safetensors_path), arrays)

            with patch("voice_cache._PLUGIN_ROOT", Path(tmpdir)):
                result = get_voice_conditionals(mock_model, "embedded_voice")

            assert result is not None
            mock_model.prepare_conditionals.assert_not_called()
