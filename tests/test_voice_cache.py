"""
Tests for voice conditionals caching module.

TDD: These tests are written FIRST before implementation.
Run with: uv run pytest tests/test_voice_cache.py -v
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


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

        try:
            result = save_conditionals(mock_conds, temp_path)
            expected = get_cache_path(temp_path)
            assert result == expected
        finally:
            os.unlink(temp_path)
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

        try:
            result = save_conditionals(mock_conds, temp_path)

            # Verify it's a valid safetensors file by loading it
            loaded = mx.load(str(result))
            assert "t3_speaker_emb" in loaded
            assert "gen_prompt_token" in loaded
        finally:
            os.unlink(temp_path)
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


class TestIntegration:
    """Integration tests for voice caching with real MLX model."""

    @pytest.mark.integration
    def test_full_cache_cycle_with_real_model(self):
        """Test full cache cycle: prepare -> save -> load with real model."""
        import time
        from voice_cache import (
            get_or_prepare_conditionals,
            is_cache_valid,
            get_cache_path,
            _PLUGIN_ROOT,
        )
        from mlx_audio.tts.utils import load_model

        # Use the default voice reference
        voice_ref = str(_PLUGIN_ROOT / "assets" / "default_voice.wav")

        # Load model
        model = load_model("mlx-community/chatterbox-turbo-fp16")

        # Clean any existing cache for this voice
        try:
            cache_path = get_cache_path(voice_ref)
            cache_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

        assert is_cache_valid(voice_ref) is False

        # First call - should be cache miss (slow)
        start = time.time()
        conds1 = get_or_prepare_conditionals(model, voice_ref)
        first_call_time = time.time() - start

        assert conds1 is not None
        assert is_cache_valid(voice_ref) is True

        # Second call - should be cache hit (fast)
        start = time.time()
        conds2 = get_or_prepare_conditionals(model, voice_ref)
        second_call_time = time.time() - start

        assert conds2 is not None

        # Cache hit should be significantly faster
        assert second_call_time < first_call_time * 0.5, (
            f"Cache hit ({second_call_time:.2f}s) should be <50% of miss ({first_call_time:.2f}s)"
        )

        # Cleanup
        cache_path = get_cache_path(voice_ref)
        cache_path.unlink(missing_ok=True)


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
            cache_path.unlink(missing_ok=True)
