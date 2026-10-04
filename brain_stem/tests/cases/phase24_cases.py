from __future__ import annotations

import hashlib
import io
import json
import unittest
import wave
from dataclasses import replace
from time import time

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import Intent, ScopeVector
from src.swarm_core.audio_telemetry import (
    AudioTelemetryError,
    AudioTelemetryPolicy,
    ConsentedAudioTelemetryService,
)
from src.swarm_core.identity import (
    Ed25519ScopeAuthorizationVerifier,
    SignedScopeGrant,
    intent_authorization_digest,
    scope_grant_message,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
AUDIENCE = "audio-telemetry-test"
TRANSACTION_ID = "tx-audio"
CORRELATION_ID = "turn-audio"
CONSENT_REFERENCE = "consent:audio-capture:1"


def make_wav(*, duration_seconds: float = 0.25, sample_rate_hz: int = 8000, channels: int = 1) -> bytes:
    frame_count = int(duration_seconds * sample_rate_hz)
    times = np.arange(frame_count, dtype=np.float64) / sample_rate_hz
    mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * times) * 32767).astype("<i2")
    interleaved = np.repeat(mono[:, np.newaxis], channels, axis=1).reshape(-1)
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(interleaved.tobytes())
    return output.getvalue()


class ConsentedAudioTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.verifier = Ed25519ScopeAuthorizationVerifier(
            public_key,
            audience=AUDIENCE,
        )
        self.service = ConsentedAudioTelemetryService(
            authorization_verifier=self.verifier,
            policy=AudioTelemetryPolicy(
                maximum_source_bytes=30_000,
                maximum_duration_seconds=1.0,
                minimum_sample_rate_hz=8000,
                maximum_sample_rate_hz=16000,
                maximum_channels=2,
                frame_duration_ms=100,
                maximum_frames=16,
            ),
        )

    def sign(self, payload: bytes, *, consent_reference: str = CONSENT_REFERENCE) -> SignedScopeGrant:
        goal = json.dumps(
            {
                "consent_reference": consent_reference,
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "source_size_bytes": len(payload),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        intent = Intent(
            transaction_id=TRANSACTION_ID,
            correlation_id=CORRELATION_ID,
            action="PROCESS_CONSENTED_AUDIO",
            goal=goal,
            scope=SCOPE,
        )
        now = int(time())
        unsigned = SignedScopeGrant(
            scope=SCOPE,
            action=intent.action,
            transaction_id=intent.transaction_id,
            correlation_id=intent.correlation_id,
            audience=AUDIENCE,
            issued_at=now,
            expires_at=now + 60,
            grant_id="audio-grant:1",
            request_sha256=intent_authorization_digest(intent),
            signature=b"",
        )
        return replace(unsigned, signature=self.private_key.sign(scope_grant_message(unsigned)))

    def process(self, payload: bytes, authorization: SignedScopeGrant):
        return self.service.process_wav(
            payload,
            transaction_id=TRANSACTION_ID,
            correlation_id=CORRELATION_ID,
            scope=SCOPE,
            consent_reference=CONSENT_REFERENCE,
            authorization=authorization,
        )

    def test_authorized_wav_emits_bounded_fft_features_and_provenance(self) -> None:
        payload = make_wav()

        result = self.process(payload, self.sign(payload))

        self.assertEqual(result.source_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(result.sample_rate_hz, 8000)
        self.assertEqual(result.channel_count, 1)
        self.assertEqual(result.consent_reference, CONSENT_REFERENCE)
        self.assertLessEqual(len(result.frames), 16)
        self.assertAlmostEqual(result.frames[0].spectral_centroid_hz, 440.0, delta=30.0)
        self.assertGreater(result.frames[0].rms, 0.3)
        self.assertFalse(hasattr(result, "raw_samples"))

    def test_changed_audio_or_missing_consent_signature_is_rejected_before_decode(self) -> None:
        payload = make_wav()
        grant = self.sign(payload)
        tampered = payload[:-1] + bytes((payload[-1] ^ 1,))

        with self.assertRaisesRegex(AudioTelemetryError, "consent authorization"):
            self.process(tampered, grant)

        with self.assertRaises(AudioTelemetryError):
            self.service.process_wav(
                payload,
                transaction_id=TRANSACTION_ID,
                correlation_id=CORRELATION_ID,
                scope=SCOPE,
                consent_reference=CONSENT_REFERENCE,
                authorization=replace(grant, signature=b""),
            )

    def test_duration_sample_rate_and_malformed_pcm_are_rejected(self) -> None:
        too_long = make_wav(duration_seconds=1.1)
        with self.assertRaisesRegex(AudioTelemetryError, "duration"):
            self.process(too_long, self.sign(too_long))

        too_fast = make_wav(sample_rate_hz=32000)
        with self.assertRaisesRegex(AudioTelemetryError, "sample rate"):
            self.process(too_fast, self.sign(too_fast))

        malformed = b"not a wave file"
        with self.assertRaisesRegex(AudioTelemetryError, "valid PCM WAV"):
            self.process(malformed, self.sign(malformed))

    def test_channel_and_frame_caps_are_enforced(self) -> None:
        stereo = make_wav(channels=2)
        result = self.process(stereo, self.sign(stereo))
        self.assertEqual(result.channel_count, 2)

        too_many_channels = make_wav(channels=3)
        with self.assertRaisesRegex(AudioTelemetryError, "channel count"):
            self.process(too_many_channels, self.sign(too_many_channels))


if __name__ == "__main__":
    unittest.main()