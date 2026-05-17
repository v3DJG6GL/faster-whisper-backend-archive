"""In-process audio transcoder using PyAV (already a faster-whisper dep,
so no extra requirement and no ffmpeg-on-PATH needed on Windows).

Output is fixed: 16 kHz · mono · signed 16-bit little-endian PCM in a
RIFF/WAVE container. That's Whisper's native input rate AND the only
format every browser plays without a system codec — Firefox on Linux
ships no AAC decoder, so storing the dictation client's raw .m4a would
mean a dead Play button on the /captures page.
"""
from __future__ import annotations

import os

_OUT_RATE = 16000
_OUT_LAYOUT = "mono"
_OUT_FORMAT = "s16"          # signed 16-bit
_OUT_CODEC = "pcm_s16le"     # WAV's native uncompressed codec


def transcode_to_wav_16k_mono(src_path: str, dst_path: str) -> int:
    """Decode anything PyAV understands, resample to 16 kHz mono, write
    a RIFF/WAVE file at dst_path. Returns bytes written. On any failure
    the destination is best-effort unlinked."""
    try:
        import av
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
        out_stream.layout = _OUT_LAYOUT
        out_stream.format = _OUT_FORMAT

        resampler = av.AudioResampler(
            format=_OUT_FORMAT, layout=_OUT_LAYOUT, rate=_OUT_RATE,
        )

        for frame in in_container.decode(in_stream):
            # PyAV recomputes pts when None; the input frame's pts is on
            # the input timebase and would corrupt the output otherwise.
            frame.pts = None
            for resampled in resampler.resample(frame):
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)

        # Flush resampler and encoder.
        for resampled in resampler.resample(None):
            for packet in out_stream.encode(resampled):
                out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)

    except Exception:
        # Release the output handle before unlink (Windows holds the
        # lock otherwise).
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
