"""
Integration tests for voice conditionals caching module.

These tests require MLX hardware and the voice reference file.
Run with: uv run pytest tests/integration/
"""
import os
import sys
import time
from pathlib import Path

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestVoiceCacheIntegration:
    """Integration tests for voice caching with real MLX model."""

    def test_full_cache_cycle_with_real_model(self):
        """Test full cache cycle: prepare -> save -> load with real model."""
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
        model = load_model(Path("mlx-community/chatterbox-turbo-fp16"))

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
