"""Streaming endpointing (VAD) for the live transcription session.

One question, asked per fixed 512-sample (32 ms @ 16 kHz) float32 frame: *is this
speech?* The two-tier silence timing (inner partial-boundary gate, outer commit
gate) and the forced-commit cap live in :class:`streaming_session.StreamSession`,
not here — this module only classifies frames.

Two backends behind one interface:

  * :class:`SileroEndpointer` — Silero VAD via faster-whisper's BUNDLED ONNX model
    (``silero_vad_v6.onnx``). No torch, no extra install — it ships with
    faster-whisper on every platform. Construction probes the model API and raises
    on an incompatible faster-whisper version so the factory falls back.
  * :class:`EnergyEndpointer` — pure-numpy RMS gate with hysteresis. No
    dependencies at all; the fallback when the bundled VAD can't be loaded (e.g.
    faster-whisper not importable, as on a deps-free CI/dev box) and in tests.

``"auto"`` (the default) uses bundled Silero when faster-whisper is present and
silently falls back to the energy gate otherwise — so it works on every deployment
without an extra package.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero v5 requires exactly this at 16 kHz (= 32 ms)
FRAME_MS = 1000 * FRAME_SAMPLES // SAMPLE_RATE  # 32


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level of float32 [-1, 1] samples in dBFS (full-scale = 0 dB)."""
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    if rms <= 1e-9:
        return float("-inf")
    return 20.0 * np.log10(rms)


class EnergyEndpointer:
    """RMS gate with hysteresis. Speech turns on above ``threshold_dbfs`` and off
    only after dropping ``hysteresis_db`` below it — so a momentary dip mid-word
    doesn't flip the state."""

    def __init__(self, threshold_dbfs: float = -42.0, hysteresis_db: float = 6.0):
        self._on = threshold_dbfs
        self._off = threshold_dbfs - hysteresis_db
        self._speaking = False

    def is_speech(self, frame: np.ndarray) -> bool:
        level = rms_dbfs(frame)
        if self._speaking:
            if level < self._off:
                self._speaking = False
        elif level > self._on:
            self._speaking = True
        return self._speaking

    def reset(self) -> None:
        self._speaking = False


class SileroEndpointer:
    """Frame-by-frame Silero VAD using faster-whisper's BUNDLED ONNX model
    (``silero_vad_v6.onnx``) via ``faster_whisper.vad.get_vad_model``.

    No torch and no extra install — the model ships inside faster-whisper on every
    platform (Linux / Docker / Windows), and onnxruntime is already a faster-whisper
    dependency. The bundled model takes a 1-D float32 array whose length is a
    multiple of 512 and returns one speech probability per 512-sample window
    (LSTM/context state handled internally per call).

    Each 512-sample (32 ms) frame is scored together with a few frames of rolling
    lookback so the decision window has real left-context. Construction probes the
    call signature (it differs on older faster-whisper), so an incompatible version
    raises and :func:`make_endpointer` falls back to the energy gate instead of
    crashing mid-stream.
    """

    _CTX_FRAMES = 3  # frames of lookback prepended to the decision window

    def __init__(self, threshold: float = 0.5):
        from faster_whisper.vad import get_vad_model  # noqa: PLC0415

        self._model = get_vad_model()
        self.threshold = float(threshold)
        self._off = self.threshold - 0.15          # Silero's built-in hysteresis
        self._speaking = False
        self._win = FRAME_SAMPLES * (self._CTX_FRAMES + 1)  # multiple of 512
        self._buf = np.zeros(self._win, dtype=np.float32)
        self._degraded = False
        self._probe()

    def _probe(self) -> None:
        out = self._model(np.zeros(self._win, dtype=np.float32))
        float(np.asarray(out).reshape(-1)[-1])     # must yield a scalar probability

    def _prob(self, frame: np.ndarray) -> float:
        self._buf = np.concatenate([self._buf[FRAME_SAMPLES:], frame])
        out = self._model(self._buf)               # (_CTX_FRAMES + 1) window probs
        return float(np.asarray(out).reshape(-1)[-1])

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.shape[0] != FRAME_SAMPLES:        # strict 512-sample window
            buf = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            n = min(FRAME_SAMPLES, frame.shape[0])
            buf[:n] = frame[:n]
            frame = buf
        frame = frame.astype(np.float32, copy=False)
        try:
            prob = self._prob(frame)
        except Exception:  # noqa: BLE001 — never break the stream on a VAD hiccup
            if not self._degraded:
                self._degraded = True
                logger.warning("[streaming-vad] Silero call failed mid-stream; "
                               "degrading to the energy gate for this session.")
            prob = 1.0 if rms_dbfs(frame) > -42.0 else 0.0
        if self._speaking:
            if prob < self._off:
                self._speaking = False
        elif prob >= self.threshold:
            self._speaking = True
        return self._speaking

    def reset(self) -> None:
        self._speaking = False
        self._buf = np.zeros(self._win, dtype=np.float32)


def make_endpointer(backend: str = "auto", *, threshold: float = 0.5,
                    energy_dbfs: float = -42.0):
    """Build an endpointer. ``backend`` is ``"silero"``, ``"energy"`` or ``"auto"``
    (try Silero, fall back to Energy with a logged warning)."""
    if backend in ("silero", "auto"):
        try:
            return SileroEndpointer(threshold=threshold)
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            if backend == "silero":
                raise
            logger.warning(
                "[streaming-vad] bundled Silero VAD unavailable (%s); using the "
                "energy gate (threshold %.0f dBFS).", exc, energy_dbfs,
            )
    return EnergyEndpointer(threshold_dbfs=energy_dbfs)


def iter_frames(samples: np.ndarray, frame_samples: int = FRAME_SAMPLES):
    """Yield consecutive full ``frame_samples`` slices; drops a trailing partial
    frame (the caller carries it forward)."""
    n_full = samples.shape[0] // frame_samples
    for i in range(n_full):
        yield samples[i * frame_samples:(i + 1) * frame_samples]
