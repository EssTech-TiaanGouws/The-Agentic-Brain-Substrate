from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from time import time
from typing import Callable, Protocol

from substrate.contracts import Intent

from .identity import (
    SignedScopeGrant,
    intent_authorization_digest,
    scope_grant_message,
)


class OperatorAuthorityError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class OperatorConfirmation:
    approved: bool
    confirmation_id: str
    request_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a boolean")
        if not isinstance(self.confirmation_id, str) or not self.confirmation_id.strip():
            raise ValueError("confirmation_id must be a non-empty string")
        if not isinstance(self.request_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.request_sha256) is None:
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")


class OperatorConfirmationProvider(Protocol):
    def confirm(self, intent: Intent, request_sha256: str) -> OperatorConfirmation: ...


class ProtectedSigningProvider(Protocol):
    @property
    def signer_reference(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


class ConfirmedScopeGrantIssuer:
    """Issues exact-intent grants only after trusted confirmation and protected signing."""

    def __init__(
        self,
        *,
        confirmation_provider: OperatorConfirmationProvider,
        signing_provider: ProtectedSigningProvider,
        audience: str,
        permitted_actions: tuple[str, ...],
        clock: Callable[[], float] = time,
        lifetime_seconds: int = 120,
    ) -> None:
        if not callable(getattr(confirmation_provider, "confirm", None)):
            raise ValueError("confirmation_provider must implement confirm()")
        if not callable(getattr(signing_provider, "sign", None)):
            raise ValueError("signing_provider must implement sign()")
        if not isinstance(audience, str) or not audience.strip():
            raise ValueError("audience must be a non-empty string")
        if not isinstance(permitted_actions, tuple) or not permitted_actions or any(
            not isinstance(action, str) or not action.strip() for action in permitted_actions
        ):
            raise ValueError("permitted_actions must be a non-empty tuple")
        if len(set(permitted_actions)) != len(permitted_actions):
            raise ValueError("permitted_actions must not contain duplicates")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(lifetime_seconds, bool) or not isinstance(lifetime_seconds, int) or lifetime_seconds <= 0:
            raise ValueError("lifetime_seconds must be a positive integer")
        signer_reference = getattr(signing_provider, "signer_reference", None)
        if not isinstance(signer_reference, str) or not signer_reference.strip():
            raise ValueError("signing_provider must expose a non-empty signer_reference")
        self._confirmation_provider = confirmation_provider
        self._signing_provider = signing_provider
        self._audience = audience
        self._permitted_actions = frozenset(permitted_actions)
        self._clock = clock
        self._lifetime_seconds = lifetime_seconds

    def issue(self, intent: Intent) -> SignedScopeGrant:
        if not isinstance(intent, Intent) or not intent.scope.is_complete():
            raise OperatorAuthorityError("a typed intent with complete trusted scope is required")
        if intent.action not in self._permitted_actions:
            raise OperatorAuthorityError("intent action is not allowlisted for operator signing")
        request_sha256 = intent_authorization_digest(intent)
        try:
            confirmation = self._confirmation_provider.confirm(intent, request_sha256)
        except Exception as error:
            raise OperatorAuthorityError("operator confirmation provider failed closed") from error
        if not isinstance(confirmation, OperatorConfirmation):
            raise OperatorAuthorityError("operator confirmation provider returned an untyped result")
        if not confirmation.approved:
            raise OperatorAuthorityError("operator denied the exact requested action")
        if confirmation.request_sha256 != request_sha256:
            raise OperatorAuthorityError("operator confirmation does not bind the exact intent digest")
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or now < 0:
            raise OperatorAuthorityError("clock returned an invalid timestamp")
        issued_at = int(now)
        unsigned = SignedScopeGrant(
            scope=intent.scope,
            action=intent.action,
            transaction_id=intent.transaction_id,
            correlation_id=intent.correlation_id,
            audience=self._audience,
            issued_at=issued_at,
            expires_at=issued_at + self._lifetime_seconds,
            grant_id=f"{self._signing_provider.signer_reference}:{confirmation.confirmation_id}",
            request_sha256=request_sha256,
            signature=b"",
        )
        try:
            signature = self._signing_provider.sign(scope_grant_message(unsigned))
        except Exception as error:
            raise OperatorAuthorityError("protected signing provider failed closed") from error
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise OperatorAuthorityError("protected signing provider returned a malformed signature")
        return replace(unsigned, signature=signature)