"""In-process audio transcoder using PyAV (bundled with faster-whisper).

We use PyAV rather than a `ffmpeg` subprocess because:
  - PyAV is already in the venv as a transitive faster-whisper dep, so
    there's no extra dependency and no need for `ffmpeg` on PATH (which
    Windows service deployments often lack).
  - It runs in-process, so there's no Popen lifecycle to manage and no
    quoting / argument-injection concerns.

The output format is fixed:
    16 kHz · mono · signed 16-bit little-endian PCM · RIFF/WAVE container.

That format is what Whisper expects as input internally and what every
browser plays without any system-codec help. Critically: Firefox on
Linux doesn't ship AAC decoders, so .m4a captures (which the dictation
client sends) won't play back on the /captures admin page until we
transcode to a codec-independent format. WAV is universal.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("whisper-api")

_OUT_RATE = 16000
_OUT_LAYOUT = "mono"
_OUT_FORMAT = "s16"          # signed 16-bit
_OUT_CODEC = "pcm_s16le"     # WAV's native uncompressed codec


def transcode_to_wav_16k_mono(src_path: str, dst_path: str) -> int:
    """Decode anything PyAV understands, resample to 16 kHz mono,
    write a RIFF/WAVE file at dst_path. Returns bytes written.

    Caller is responsible for ensuring dst_path's parent directory
    exists. On any failure the destination is best-effort unlinked so
    we never leave a partial WAV that would confuse the magic-byte
    mime sniff on serve.

    Raises:
      RuntimeError: PyAV not importable on this deployment.
      ValueError:   source has no audio stream.
      Exception:    any underlying av.AVError / OSError from decode
                    or mux. Caller logs and skips.
    """
    try:
        import av  # PyAV
    except ImportError as e:
        raise RuntimeError("PyAV (av) not installed; cannot transcode") from e

    in_container = None
    out_container = None
    try:
        in_container = av.open(src_path)
        in_stream = next(
            (s for s in in_container.streams if s.type == "audio"), None,
        )
        if in_stream is None:
            raise ValueError("source has no audio stream")

        out_container = av.open(dst_path, mode="w", format="wav")
        out_stream = out_container.add_stream(_OUT_CODEC, rate=_OUT_RATE)
        # In PyAV's audio-encode API, the codec context defaults to
        # the input frame's layout / format. We pin both so the encoder
        # rejects any frame the resampler somehow lets through with a
        # mismatched shape (defense-in-depth — should never fire).
        out_stream.layout = _OUT_LAYOUT
        out_stream.format = _OUT_FORMAT

        resampler = av.AudioResampler(
            format=_OUT_FORMAT, layout=_OUT_LAYOUT, rate=_OUT_RATE,
        )

        for frame in in_container.decode(in_stream):
            # Let PyAV recompute pts at the output rate — the input
            # frame's pts is on the input timebase and would otherwise
            # leak into the WAV header as garbage.
            frame.pts = None
            for resampled in resampler.resample(frame):
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)

        # Flush the resampler (drains buffered samples from rate-change
        # interpolation), then flush the encoder.
        for resampled in resampler.resample(None):
            for packet in out_stream.encode(resampled):
                out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)

    except Exception:
        # Tear down the output container first so the file handle is
        # released before we try to unlink it (Windows holds the lock
        # otherwise).
        try:
            if out_container is not None:
                out_container.close()
                out_container = None
        except Exception:
            pass
        try:
            if os.path.exists(dst_path):
                os.unlink(dst_path)
        except OSError:
            pass
        raise
    finally:
        # Close in reverse-open order. close() is idempotent in PyAV,
        # so the except-branch closes above are safe to repeat here.
        try:
            if out_container is not None:
                out_container.close()
        except Exception:
            pass
        try:
            if in_container is not None:
                in_container.close()
        except Exception:
            pass

    return os.path.getsize(dst_path)
