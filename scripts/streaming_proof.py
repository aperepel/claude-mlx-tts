#!/usr/bin/env python3
"""
Streaming Proof: Demonstrate true streaming behavior with timing analysis.

This script proves HTTP streaming works by showing:
1. TTFT (time to first audio) << total generation time
2. Playback starts BEFORE last chunk arrives
3. Network I/O and audio playback overlap in time

The key proof: if playback_start_time < last_chunk_received_time,
then we're truly streaming (playing audio while still receiving data).

Usage:
    uv run python scripts/streaming_proof.py "Your test text here"
    uv run python scripts/streaming_proof.py --compare "Text for A/B test"
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class StreamEvent:
    """A timestamped streaming event."""
    timestamp: float
    event: str
    details: str = ""
    samples: int = 0
    bytes_received: int = 0


class StreamingProof:
    """Collector for timestamped streaming events with analysis."""

    def __init__(self, verbose: bool = True):
        self.events: list[StreamEvent] = []
        self.start_time = time.perf_counter()
        self.lock = Lock()
        self.verbose = verbose

        # Counters
        self.total_bytes = 0
        self.total_samples = 0
        self.chunks_received = 0

    def log(self, event: str, details: str = "", samples: int = 0, bytes_recv: int = 0):
        """Log a timestamped event."""
        elapsed = time.perf_counter() - self.start_time
        with self.lock:
            self.total_bytes += bytes_recv
            self.total_samples += samples
            if bytes_recv > 0:
                self.chunks_received += 1
            self.events.append(StreamEvent(elapsed, event, details, samples, bytes_recv))
        if self.verbose:
            if samples > 0:
                details = f"{details} ({samples} samples)" if details else f"{samples} samples"
            if bytes_recv > 0:
                details = f"{details} ({bytes_recv} bytes)" if details else f"{bytes_recv} bytes"
            print(f"[t+{elapsed:7.3f}s] {event:24s} {details}")

    def analyze(self) -> dict:
        """Analyze timeline to prove streaming behavior."""
        def find_first(event_type: str) -> Optional[StreamEvent]:
            return next((e for e in self.events if e.event == event_type), None)

        def find_last(event_type: str) -> Optional[StreamEvent]:
            matches = [e for e in self.events if e.event == event_type]
            return matches[-1] if matches else None

        def find_first_any(event_types: list) -> Optional[StreamEvent]:
            return next((e for e in self.events if e.event in event_types), None)

        def find_last_any(event_types: list) -> Optional[StreamEvent]:
            matches = [e for e in self.events if e.event in event_types]
            return matches[-1] if matches else None

        # Support both HTTP mode (CHUNK_RECEIVED/AUDIO_QUEUED) and direct MLX mode (AUDIO_CHUNK_GENERATED)
        first_chunk = find_first_any(["HTTP_CHUNK_RECEIVED", "AUDIO_CHUNK_GENERATED"])
        first_audio = find_first_any(["AUDIO_QUEUED", "AUDIO_CHUNK_GENERATED"])
        playback_start = find_first("PLAYBACK_STARTED")
        last_chunk = find_last_any(["HTTP_CHUNK_RECEIVED", "AUDIO_CHUNK_GENERATED"])
        gen_complete = find_first("GENERATION_COMPLETE")
        playback_end = find_first("PLAYBACK_COMPLETE")

        # Count chunks from events (works for both HTTP and direct modes)
        chunk_events = [e for e in self.events if e.event in ["HTTP_CHUNK_RECEIVED", "AUDIO_CHUNK_GENERATED"]]
        chunk_count = len(chunk_events)
        total_samples_from_events = sum(e.samples for e in chunk_events)

        proof = {
            "ttft_seconds": first_audio.timestamp if first_audio else None,
            "playback_start_seconds": playback_start.timestamp if playback_start else None,
            "last_chunk_seconds": last_chunk.timestamp if last_chunk else None,
            "generation_complete_seconds": gen_complete.timestamp if gen_complete else None,
            "playback_end_seconds": playback_end.timestamp if playback_end else None,
            "total_chunks": chunk_count or self.chunks_received,
            "total_bytes": self.total_bytes,
            "total_samples": total_samples_from_events or self.total_samples,
        }

        # THE KEY PROOF: playback started BEFORE last chunk arrived
        if proof["playback_start_seconds"] and proof["last_chunk_seconds"]:
            proof["is_truly_streaming"] = proof["playback_start_seconds"] < proof["last_chunk_seconds"]
            proof["streaming_overlap_seconds"] = max(0, proof["last_chunk_seconds"] - proof["playback_start_seconds"])
        else:
            proof["is_truly_streaming"] = False
            proof["streaming_overlap_seconds"] = 0

        # Calculate audio duration
        sample_rate = 24000  # Chatterbox default
        if proof["total_samples"] > 0:
            proof["audio_duration_seconds"] = proof["total_samples"] / sample_rate
        else:
            proof["audio_duration_seconds"] = 0

        return proof

    def print_summary(self):
        """Print a formatted summary of the streaming proof."""
        proof = self.analyze()

        print("\n" + "=" * 60)
        print("STREAMING PROOF ANALYSIS")
        print("=" * 60)

        print(f"\n📊 Metrics:")
        print(f"   TTFT (time to first audio):     {proof['ttft_seconds']:.3f}s" if proof['ttft_seconds'] else "   TTFT: N/A")
        print(f"   Playback started at:            {proof['playback_start_seconds']:.3f}s" if proof['playback_start_seconds'] else "   Playback start: N/A")
        print(f"   Last chunk received at:         {proof['last_chunk_seconds']:.3f}s" if proof['last_chunk_seconds'] else "   Last chunk: N/A")
        print(f"   Generation complete at:         {proof['generation_complete_seconds']:.3f}s" if proof['generation_complete_seconds'] else "   Gen complete: N/A")
        print(f"   Playback complete at:           {proof['playback_end_seconds']:.3f}s" if proof['playback_end_seconds'] else "   Playback end: N/A")
        print(f"   Total chunks received:          {proof['total_chunks']}")
        print(f"   Total audio duration:           {proof['audio_duration_seconds']:.2f}s")

        print(f"\n🔍 Streaming Proof:")
        if proof["is_truly_streaming"]:
            print(f"   ✅ TRUE STREAMING CONFIRMED!")
            print(f"   ✅ Playback started {proof['streaming_overlap_seconds']:.2f}s before generation finished")
            print(f"   ✅ Audio was playing while chunks were still arriving")
        else:
            print(f"   ❌ Streaming NOT proven (playback started after all chunks received)")

        # Additional analysis
        if proof["ttft_seconds"] and proof["generation_complete_seconds"]:
            speedup = proof["generation_complete_seconds"] / proof["ttft_seconds"]
            print(f"\n⚡ Latency Improvement:")
            print(f"   Non-streaming would wait:       {proof['generation_complete_seconds']:.2f}s")
            print(f"   Streaming starts at:            {proof['ttft_seconds']:.2f}s")
            print(f"   Perceived latency reduction:    {speedup:.1f}x faster start")

        print("=" * 60 + "\n")

        return proof


def run_streaming_proof(text: str, voice: str = None) -> dict:
    """
    Run streaming TTS with full instrumentation to prove streaming behavior.

    Returns proof dict with timing analysis.
    """
    import requests
    from streaming_wav_parser import StreamingWavParser, WavParseError
    from mlx_server_utils import ensure_server_running, TTS_SERVER_HOST, TTS_SERVER_PORT, MLX_MODEL, TTSRequestError

    proof = StreamingProof(verbose=True)

    # Get audio processor and player
    try:
        from mlx_audio.tts.audio_player import AudioPlayer
    except ImportError:
        print("ERROR: mlx_audio not installed")
        return {}

    try:
        from audio_processor import create_processor
        import tts_config
    except ImportError:
        create_processor = None
        tts_config = None

    proof.log("REQUEST_INIT", f"Text: {text[:50]}...")

    # Ensure server is running
    ensure_server_running()
    proof.log("SERVER_READY")

    # Use our new streaming endpoint that passes stream=True to model
    url = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech/stream"
    payload = {
        "input": text,
        "model": MLX_MODEL,
        "streaming_interval": 0.5,  # yield every 0.5s of audio
    }
    if voice:
        payload["voice"] = voice

    parser = StreamingWavParser()
    player = None
    audio_processor = None
    sample_rate = 24000

    # Wrap AudioPlayer to capture playback start
    original_start_stream = AudioPlayer.start_stream
    def instrumented_start_stream(self):
        proof.log("PLAYBACK_STARTED", "AudioPlayer stream started")
        return original_start_stream(self)
    AudioPlayer.start_stream = instrumented_start_stream

    try:
        proof.log("HTTP_REQUEST_SENT")
        response = requests.post(url, json=payload, timeout=60, stream=True)

        if response.status_code != 200:
            raise TTSRequestError(f"Status {response.status_code}: {response.text}")

        proof.log("HTTP_RESPONSE_STARTED", f"Status {response.status_code}")

        first_audio_queued = False

        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue

            proof.log("HTTP_CHUNK_RECEIVED", bytes_recv=len(chunk))

            try:
                parser.feed(chunk)
            except WavParseError as e:
                raise TTSRequestError(f"WAV parse error: {e}")

            audio = parser.read_audio()
            if audio is not None and len(audio) > 0:
                if not first_audio_queued:
                    # First audio chunk - initialize player
                    if parser.header:
                        sample_rate = parser.header.sample_rate
                    player = AudioPlayer(sample_rate=sample_rate)

                    # Initialize audio processor
                    if create_processor and tts_config and parser.header:
                        compressor = tts_config.get_effective_compressor(voice)
                        limiter = tts_config.get_effective_limiter(voice)
                        audio_processor = create_processor(
                            sample_rate=sample_rate,
                            input_gain_db=compressor.get("input_gain_db"),
                            threshold_db=compressor.get("threshold_db"),
                            ratio=compressor.get("ratio"),
                            attack_ms=compressor.get("attack_ms"),
                            release_ms=compressor.get("release_ms"),
                            gain_db=compressor.get("gain_db"),
                            master_gain_db=compressor.get("master_gain_db"),
                            compressor_enabled=compressor.get("enabled"),
                            limiter_threshold_db=limiter.get("threshold_db"),
                            limiter_release_ms=limiter.get("release_ms"),
                            limiter_enabled=limiter.get("enabled"),
                        )
                    first_audio_queued = True

                # Apply compression
                if audio_processor:
                    audio = audio_processor(audio)

                proof.log("AUDIO_QUEUED", samples=len(audio))

                # Queue to player
                if player:
                    player.queue_audio(audio)

        proof.log("GENERATION_COMPLETE", f"All {proof.chunks_received} chunks received")

        # Wait for playback to finish
        if player:
            if not player.playing and player.buffered_samples() > 0:
                proof.log("FORCE_START", "Starting stream for short audio")
                player.start_stream()

            if player.playing:
                proof.log("WAITING_FOR_DRAIN")
                player.wait_for_drain()
                proof.log("PLAYBACK_COMPLETE")

        response.close()

    finally:
        # Restore original method
        AudioPlayer.start_stream = original_start_stream

    return proof


def run_non_streaming_baseline(text: str, voice: str = None) -> dict:
    """Run non-streaming TTS for comparison."""
    import time
    import io
    import requests
    import sounddevice as sd
    import soundfile as sf

    from mlx_server_utils import ensure_server_running, TTS_SERVER_HOST, TTS_SERVER_PORT, MLX_MODEL

    ensure_server_running()

    url = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech"
    payload = {"input": text, "model": MLX_MODEL}
    if voice:
        payload["voice"] = voice

    print("\n--- Non-Streaming Baseline ---")
    start = time.perf_counter()

    # Non-streaming: wait for full response
    response = requests.post(url, json=payload, timeout=60)
    gen_time = time.perf_counter() - start
    print(f"[t+{gen_time:.3f}s] Full response received ({len(response.content)} bytes)")

    # Parse and play
    audio_buffer = io.BytesIO(response.content)
    data, samplerate = sf.read(audio_buffer)
    data = data.astype(np.float32)

    print(f"[t+{time.perf_counter() - start:.3f}s] Starting playback...")
    sd.play(data, samplerate)
    sd.wait()
    total_time = time.perf_counter() - start
    print(f"[t+{total_time:.3f}s] Playback complete")

    return {
        "generation_time": gen_time,
        "total_time": total_time,
        "audio_duration": len(data) / samplerate,
    }


def run_direct_mlx_streaming_proof(text: str, voice: str = None) -> StreamingProof:
    """
    Run TRUE streaming proof using direct MLX API with stream=True.

    This bypasses the HTTP server and uses the model's native streaming,
    proving that MLX TTS can stream with fast TTFT when stream=True is used.
    """
    try:
        from mlx_audio.tts.audio_player import AudioPlayer
    except ImportError:
        print("ERROR: mlx_audio not installed")
        return StreamingProof()

    try:
        from audio_processor import create_processor
        import tts_config
    except ImportError:
        create_processor = None
        tts_config = None

    proof = StreamingProof(verbose=True)
    proof.log("DIRECT_MLX_INIT", f"Text: {text[:50]}...")

    # Load model directly
    from mlx_tts_core import get_model
    model = get_model()
    proof.log("MODEL_LOADED")

    # Pre-warm voice conditionals if needed
    if voice:
        try:
            from voice_cache import get_voice_conditionals
            model._conds = get_voice_conditionals(model, voice)
            proof.log("VOICE_LOADED", voice)
        except Exception as e:
            proof.log("VOICE_FALLBACK", str(e))

    # Initialize player
    sample_rate = model.sample_rate
    player = AudioPlayer(sample_rate=sample_rate)

    # Wrap AudioPlayer to capture playback start
    original_start_stream = AudioPlayer.start_stream
    def instrumented_start_stream(self):
        proof.log("PLAYBACK_STARTED", "AudioPlayer stream started")
        return original_start_stream(self)
    AudioPlayer.start_stream = instrumented_start_stream

    # Initialize audio processor
    audio_processor = None
    if create_processor and tts_config:
        compressor = tts_config.get_effective_compressor(voice)
        limiter = tts_config.get_effective_limiter(voice)
        audio_processor = create_processor(
            sample_rate=sample_rate,
            input_gain_db=compressor.get("input_gain_db"),
            threshold_db=compressor.get("threshold_db"),
            ratio=compressor.get("ratio"),
            attack_ms=compressor.get("attack_ms"),
            release_ms=compressor.get("release_ms"),
            gain_db=compressor.get("gain_db"),
            master_gain_db=compressor.get("master_gain_db"),
            compressor_enabled=compressor.get("enabled"),
            limiter_threshold_db=limiter.get("threshold_db"),
            limiter_release_ms=limiter.get("release_ms"),
            limiter_enabled=limiter.get("enabled"),
        )

    try:
        proof.log("GENERATION_START", "stream=True, streaming_interval=0.5")

        # Use stream=True for TRUE streaming!
        for result in model.generate(
            text,
            stream=True,
            streaming_interval=0.5,  # Yield every 0.5 seconds of audio
        ):
            audio = np.asarray(result.audio)
            proof.log("AUDIO_CHUNK_GENERATED", samples=len(audio))

            # Apply compression
            if audio_processor:
                audio = audio_processor(audio)

            # Queue to player immediately
            player.queue_audio(audio)

        proof.log("GENERATION_COMPLETE")

        # Wait for playback
        if player:
            if not player.playing and player.buffered_samples() > 0:
                proof.log("FORCE_START", "Starting stream for remaining audio")
                player.start_stream()

            if player.playing:
                proof.log("WAITING_FOR_DRAIN")
                player.wait_for_drain()
                proof.log("PLAYBACK_COMPLETE")

    finally:
        AudioPlayer.start_stream = original_start_stream

    return proof


def main():
    parser = argparse.ArgumentParser(
        description="Prove HTTP streaming is truly streaming with timing analysis"
    )
    parser.add_argument("text", nargs="?", default="This is a comprehensive test of streaming text to speech. The audio should start playing well before all chunks have arrived from the server, proving true streaming behavior with low latency.",
                       help="Text to synthesize")
    parser.add_argument("--voice", "-v", default=None, help="Voice name to use")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Also run non-streaming for A/B comparison")
    parser.add_argument("--direct", "-d", action="store_true",
                       help="Use direct MLX API with stream=True (bypasses HTTP server)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show summary, not individual events")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MLX TTS STREAMING PROOF")
    print("=" * 60)
    print(f"\nText: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"Voice: {args.voice or '(default)'}")
    print(f"Mode: {'Direct MLX API (stream=True)' if args.direct else 'HTTP Server'}")
    print("=" * 60 + "\n")

    if args.direct:
        # Run direct MLX API streaming test (TRUE streaming)
        print("--- Direct MLX API Streaming (stream=True) ---\n")
        proof = run_direct_mlx_streaming_proof(args.text, args.voice)
        proof.print_summary()
    else:
        # Run HTTP streaming test with our new /v1/audio/speech/stream endpoint
        print("--- HTTP Server Streaming (/v1/audio/speech/stream) ---\n")
        print("Using our custom endpoint that passes stream=True to model.generate()\n")
        proof = run_streaming_proof(args.text, args.voice)
        proof.print_summary()

    # Optional A/B comparison
    if args.compare:
        print("\n" + "-" * 60)
        baseline = run_non_streaming_baseline(args.text, args.voice)

        streaming_result = proof.analyze()

        print("\n" + "=" * 60)
        print("A/B COMPARISON")
        print("=" * 60)
        print(f"\n📈 Perceived Latency (time to first audio):")
        print(f"   Streaming:     {streaming_result['ttft_seconds']:.3f}s")
        print(f"   Non-streaming: {baseline['generation_time']:.3f}s")
        if streaming_result['ttft_seconds'] and baseline['generation_time']:
            improvement = baseline['generation_time'] / streaming_result['ttft_seconds']
            print(f"   Improvement:   {improvement:.1f}x faster start with streaming")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
