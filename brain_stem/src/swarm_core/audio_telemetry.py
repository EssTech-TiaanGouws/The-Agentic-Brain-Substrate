from __future__ import annotations

import hashlib
import io
import json
import math
import wave
from dataclasses import dataclass

import numpy as np

from substrate.contracts import Intent, ScopeVector

from .identity import ScopeAuthorizationVerifier, SignedScopeGrant, intent_authorization_digest


class AudioTelemetryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AudioTelemetryPolicy:
    maximum_source_bytes: int
    maximum_duration_seconds: float
    minimum_sample_rate_hz: int
    maximum_sample_rate_hz: int
    maximum_channels: int
    frame_duration_ms: int
    maximum_frames: int

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_source_bytes",
            "minimum_sample_rate_hz",
            "maximum_sample_rate_hz",
            "maximum_channels",
            "frame_duration_ms",
            "maximum_frames",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AudioTelemetryError(f"{field_name} must be a positive integer")
        if self.minimum_sample_rate_hz > self.maximum_sample_rate_hz:
            raise AudioTelemetryError("minimum sample rate exceeds maximum sample rate")
        if (
            isinstance(self.maximum_duration_seconds, bool)
            or not isinstance(self.maximum_duration_seconds, (int, float))
            or not math.isfinite(self.maximum_duration_seconds)
            or self.maximum_duration_seconds <= 0
        ):
            raise AudioTelemetryError("maximum_duration_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class AudioTelemetryFrame:
    frame_index: int
    start_seconds: float
    rms: float
    peak: float
    spectral_centroid_hz: float
    low_band_energy_ratio: float


@dataclass(frozen=True, slots=True)
class AudioTelemetryEnvelope:
    transaction_id: str
    correlation_id: str
    scope: ScopeVector
    consent_reference: str
    source_sha256: str
    sample_rate_hz: int
    channel_count: int
    duration_seconds: float
    frames: tuple[AudioTelemetryFrame, ...]


class ConsentedAudioTelemetryService:
    """Processes explicit PCM-WAV payloads; it never opens a microphone or stores audio."""

    def __init__(
        self,
        *,
        authorization_verifier: ScopeAuthorizationVerifier,
        policy: AudioTelemetryPolicy,
    ) -> None:
        if not callable(getattr(authorization_verifier, "authorize", None)):
            raise ValueError("authorization_verifier must implement authorize()")
        if not isinstance(policy, AudioTelemetryPolicy):
            raise TypeError("policy must be an AudioTelemetryPolicy")
        self._authorization_verifier = authorization_verifier
        self._policy = policy

    def process_wav(
        self,
        payload: bytes,
        *,
        transaction_id: str,
        correlation_id: str,
        scope: ScopeVector,
        consent_reference: str,
        authorization: SignedScopeGrant,
    ) -> AudioTelemetryEnvelope:
        if not isinstance(payload, bytes) or not payload:
            raise AudioTelemetryError("audio payload must be non-empty bytes")
        if len(payload) > self._policy.maximum_source_bytes:
            raise AudioTelemetryError("audio payload exceeds the configured byte limit")
        for field_name, value in (
            ("transaction_id", transaction_id),
            ("correlation_id", correlation_id),
            ("consent_reference", consent_reference),
        ):
            if not isinstance(value, str) or not value.strip():
                raise AudioTelemetryError(f"{field_name} must be a non-empty string")
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise AudioTelemetryError("complete four-field scope is required")
        source_sha256 = hashlib.sha256(payload).hexdigest()
        authorization_goal = json.dumps(
            {
                "consent_reference": consent_reference,
                "source_sha256": source_sha256,
                "source_size_bytes": len(payload),
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        intent = Intent(
            transaction_id=transaction_id,
            correlation_id=correlation_id,
            action="PROCESS_CONSENTED_AUDIO",
            goal=authorization_goal,
            scope=scope,
        )
        try:
            authorized_scope = self._authorization_verifier.authorize(intent, authorization)
        except Exception as error:
            raise AudioTelemetryError("audio payload lacks exact scope-bound consent authorization") from error
        if authorized_scope != scope:
            raise AudioTelemetryError("audio authorization returned a different scope")

        try:
            with wave.open(io.BytesIO(payload), "rb") as audio:
                if audio.getcomptype() != "NONE":
                    raise AudioTelemetryError("only uncompressed PCM WAV audio is accepted")
                channel_count = audio.getnchannels()
                sample_rate_hz = audio.getframerate()
                sample_width = audio.getsampwidth()
                frame_count = audio.getnframes()
                if not 1 <= channel_count <= self._policy.maximum_channels:
                    raise AudioTelemetryError("WAV channel count is outside the configured limit")
                if not self._policy.minimum_sample_rate_hz <= sample_rate_hz <= self._policy.maximum_sample_rate_hz:
                    raise AudioTelemetryError("WAV sample rate is outside the configured range")
                if sample_width not in {1, 2, 4}:
                    raise AudioTelemetryError("WAV PCM sample width must be 8, 16, or 32 bits")
                duration = frame_count / sample_rate_hz
                if duration <= 0 or duration > self._policy.maximum_duration_seconds:
                    raise AudioTelemetryError("WAV duration is outside the configured limit")
                expected_bytes = frame_count * channel_count * sample_width
                if expected_bytes > self._policy.maximum_source_bytes:
                    raise AudioTelemetryError("decoded WAV sample buffer exceeds the configured byte limit")
                raw_samples = audio.readframes(frame_count)
                if len(raw_samples) != expected_bytes:
                    raise AudioTelemetryError("WAV sample data is truncated")
        except AudioTelemetryError:
            raise
        except (wave.Error, EOFError, OSError) as error:
            raise AudioTelemetryError("audio payload is not a valid PCM WAV file") from error

        samples = self._decode_pcm(raw_samples, sample_width)
        samples = samples.reshape((-1, channel_count)).mean(axis=1, dtype=np.float64)
        frame_size = max(1, sample_rate_hz * self._policy.frame_duration_ms // 1000)
        frame_total = (samples.size + frame_size - 1) // frame_size
        if frame_total > self._policy.maximum_frames:
            raise AudioTelemetryError("WAV produces more analysis frames than the configured limit")
        features: list[AudioTelemetryFrame] = []
        window = np.hanning(frame_size)
        frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate_hz)
        for frame_index, start in enumerate(range(0, samples.size, frame_size)):
            frame = samples[start : start + frame_size]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            frame = np.asarray(frame, dtype=np.float64)
            rms = float(np.sqrt(np.mean(frame * frame)))
            peak = float(np.max(np.abs(frame)))
            spectrum = np.abs(np.fft.rfft(frame * window))
            power = spectrum * spectrum
            total_power = float(np.sum(power))
            if total_power > 0:
                centroid = float(np.sum(frequencies * power) / total_power)
                low_power = float(np.sum(power[frequencies <= min(1000.0, sample_rate_hz / 2)]))
                low_ratio = min(1.0, max(0.0, low_power / total_power))
            else:
                centroid = 0.0
                low_ratio = 0.0
            features.append(
                AudioTelemetryFrame(
                    frame_index,
                    start / sample_rate_hz,
                    rms,
                    peak,
                    centroid,
                    low_ratio,
                )
            )
        return AudioTelemetryEnvelope(
            transaction_id,
            correlation_id,
            scope,
            consent_reference,
            source_sha256,
            sample_rate_hz,
            channel_count,
            duration,
            tuple(features),
        )

    @staticmethod
    def _decode_pcm(raw_samples: bytes, sample_width: int) -> np.ndarray:
        if sample_width == 1:
            return (np.frombuffer(raw_samples, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        if sample_width == 2:
            return np.frombuffer(raw_samples, dtype="<i2").astype(np.float64) / 32768.0
        return np.frombuffer(raw_samples, dtype="<i4").astype(np.float64) / 2147483648.0