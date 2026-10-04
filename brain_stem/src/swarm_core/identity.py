from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from substrate.contracts import Intent, ScopeVector


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class SignedScopeGrant:
    scope: ScopeVector
    action: str
    transaction_id: str
    correlation_id: str
    audience: str
    issued_at: int
    expires_at: int
    grant_id: str
    request_sha256: str
    signature: bytes


class ScopeAuthorizationVerifier(Protocol):
    def authorize(self, intent: Intent, grant: SignedScopeGrant) -> ScopeVector: ...


def scope_grant_message(grant: SignedScopeGrant) -> bytes:
    if not isinstance(grant, SignedScopeGrant):
        raise TypeError("grant must be a SignedScopeGrant")
    payload = {
        "action": grant.action,
        "audience": grant.audience,
        "correlation_id": grant.correlation_id,
        "expires_at": grant.expires_at,
        "grant_id": grant.grant_id,
        "issued_at": grant.issued_at,
        "request_sha256": grant.request_sha256,
        "scope": {
            "project_id": grant.scope.project_id,
            "tenant_id": grant.scope.tenant_id,
            "user_id": grant.scope.user_id,
            "workspace_id": grant.scope.workspace_id,
        },
        "transaction_id": grant.transaction_id,
        "version": 1,
    }
    return json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def intent_authorization_digest(intent: Intent) -> str:
    if not isinstance(intent, Intent):
        raise TypeError("intent must be an Intent")
    payload = {
        "action": intent.action,
        "correlation_id": intent.correlation_id,
        "goal": intent.goal,
        "scope": {
            "project_id": intent.scope.project_id,
            "tenant_id": intent.scope.tenant_id,
            "user_id": intent.scope.user_id,
            "workspace_id": intent.scope.workspace_id,
        },
        "transaction_id": intent.transaction_id,
        "version": 1,
    }
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class Ed25519ScopeAuthorizationVerifier:
    def __init__(
        self,
        public_key: bytes,
        *,
        audience: str,
        clock: Callable[[], float] = time.time,
        max_lifetime_seconds: int = 300,
    ) -> None:
        if not isinstance(public_key, bytes):
            raise ValueError("public_key must be raw Ed25519 public key bytes")
        if not isinstance(audience, str) or not audience.strip():
            raise ValueError("audience must be a non-empty string")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(max_lifetime_seconds, bool) or not isinstance(max_lifetime_seconds, int):
            raise ValueError("max_lifetime_seconds must be an integer")
        if max_lifetime_seconds <= 0:
            raise ValueError("max_lifetime_seconds must be positive")
        try:
            self._public_key = Ed25519PublicKey.from_public_bytes(public_key)
        except ValueError as error:
            raise ValueError("public_key must contain exactly 32 bytes") from error
        self._audience = audience
        self._clock = clock
        self._max_lifetime_seconds = max_lifetime_seconds

    def authorize(self, intent: Intent, grant: SignedScopeGrant) -> ScopeVector:
        if not isinstance(intent, Intent) or not isinstance(grant, SignedScopeGrant):
            raise AuthorizationError("A typed intent and signed scope grant are required")
        if not isinstance(grant.scope, ScopeVector) or not grant.scope.is_complete():
            raise AuthorizationError("Signed scope grant is incomplete")
        if (
            grant.action != intent.action
            or grant.transaction_id != intent.transaction_id
            or grant.correlation_id != intent.correlation_id
            or grant.audience != self._audience
            or grant.request_sha256 != intent_authorization_digest(intent)
        ):
            raise AuthorizationError("Signed scope grant does not match this request")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (grant.grant_id, grant.action, grant.transaction_id, grant.correlation_id)
        ):
            raise AuthorizationError("Signed scope grant has invalid identifiers")
        if (
            not isinstance(grant.request_sha256, str)
            or len(grant.request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in grant.request_sha256)
        ):
            raise AuthorizationError("Signed scope grant has an invalid request digest")
        if (
            isinstance(grant.issued_at, bool)
            or not isinstance(grant.issued_at, int)
            or isinstance(grant.expires_at, bool)
            or not isinstance(grant.expires_at, int)
        ):
            raise AuthorizationError("Signed scope grant has invalid timestamps")
        now = self._clock()
        if (
            grant.expires_at <= grant.issued_at
            or grant.expires_at - grant.issued_at > self._max_lifetime_seconds
            or grant.issued_at > now
            or grant.expires_at <= now
        ):
            raise AuthorizationError("Signed scope grant is outside its valid time window")
        if not isinstance(grant.signature, bytes) or len(grant.signature) != 64:
            raise AuthorizationError("Signed scope grant has an invalid signature")
        try:
            self._public_key.verify(grant.signature, scope_grant_message(grant))
        except InvalidSignature as error:
            raise AuthorizationError("Signed scope grant signature is invalid") from error
        if intent.scope != grant.scope:
            raise AuthorizationError("Intent scope does not match signed operator scope")
        return grant.scope