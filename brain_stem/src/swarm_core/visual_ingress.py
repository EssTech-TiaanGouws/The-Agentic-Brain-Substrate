from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps


class VisualIngressError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VisualIngressPolicy:
    maximum_input_bytes: int
    maximum_normalized_bytes: int
    maximum_width: int
    maximum_height: int
    maximum_pixels: int
    maximum_dom_nodes: int
    maximum_node_text_characters: int
    maximum_dom_text_characters: int
    allowed_image_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
    allowed_dom_roles: tuple[str, ...] = (
        "button",
        "checkbox",
        "combobox",
        "heading",
        "img",
        "link",
        "listitem",
        "menuitem",
        "option",
        "row",
        "textbox",
    )
    allowed_dom_attributes: tuple[str, ...] = (
        "alt",
        "aria-label",
        "href",
        "placeholder",
        "type",
    )

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_input_bytes",
            "maximum_normalized_bytes",
            "maximum_width",
            "maximum_height",
            "maximum_pixels",
            "maximum_dom_nodes",
            "maximum_node_text_characters",
            "maximum_dom_text_characters",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VisualIngressError(f"{field_name} must be a positive integer")
        for field_name in ("allowed_image_formats", "allowed_dom_roles", "allowed_dom_attributes"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise VisualIngressError(f"{field_name} must be a non-empty tuple of strings")
            if len(set(values)) != len(values):
                raise VisualIngressError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class NormalizedRaster:
    source_ref: str
    source_sha256: str
    source_format: str
    normalized_sha256: str
    width: int
    height: int
    content_type: str
    png_bytes: bytes


@dataclass(frozen=True, slots=True)
class ViewportRect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class DomNodeObservation:
    node_id: str
    role: str
    text: str
    rect: ViewportRect
    attributes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DomSnapshot:
    source_ref: str
    source_sha256: str
    viewport_width: int
    viewport_height: int
    nodes: tuple[DomNodeObservation, ...]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VisualIngressError("DOM snapshot contains a duplicate JSON key")
        result[key] = value
    return result


class VisualIngressAdapter:
    def __init__(self, policy: VisualIngressPolicy) -> None:
        if not isinstance(policy, VisualIngressPolicy):
            raise TypeError("policy must be a VisualIngressPolicy")
        self._policy = policy

    def normalize_raster(self, content: bytes, *, source_ref: str) -> NormalizedRaster:
        if not isinstance(content, bytes) or not content:
            raise VisualIngressError("raster content must be non-empty bytes")
        source_ref = self._required_text(source_ref, "source_ref")
        if len(content) > self._policy.maximum_input_bytes:
            raise VisualIngressError("raster source exceeds the configured byte limit")
        source_sha256 = hashlib.sha256(content).hexdigest()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as source:
                    source_format = source.format
                    if source_format not in self._policy.allowed_image_formats:
                        raise VisualIngressError("raster format is not allowlisted")
                    if getattr(source, "n_frames", 1) != 1:
                        raise VisualIngressError("animated images are not accepted by the viewport adapter")
                    width, height = source.size
                    if (
                        width <= 0
                        or height <= 0
                        or width > self._policy.maximum_width
                        or height > self._policy.maximum_height
                        or width * height > self._policy.maximum_pixels
                    ):
                        raise VisualIngressError("raster dimensions exceed the configured pixel bounds")
                    source.load()
                    image = ImageOps.exif_transpose(source).convert("RGB")
        except VisualIngressError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise VisualIngressError("raster triggered the image decompression bomb guard") from error
        except Exception as error:
            raise VisualIngressError("raster content is malformed or unsupported") from error

        image.thumbnail((self._policy.maximum_width, self._policy.maximum_height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False)
        png_bytes = output.getvalue()
        if len(png_bytes) > self._policy.maximum_normalized_bytes:
            raise VisualIngressError("normalized raster exceeds the configured byte limit")
        return NormalizedRaster(
            source_ref,
            source_sha256,
            source_format or "UNKNOWN",
            hashlib.sha256(png_bytes).hexdigest(),
            image.width,
            image.height,
            "image/png",
            png_bytes,
        )

    def parse_dom_snapshot(self, content: bytes, *, source_ref: str) -> DomSnapshot:
        if not isinstance(content, bytes) or not content:
            raise VisualIngressError("DOM snapshot must be non-empty bytes")
        source_ref = self._required_text(source_ref, "source_ref")
        if len(content) > self._policy.maximum_input_bytes:
            raise VisualIngressError("DOM snapshot exceeds the configured byte limit")
        try:
            payload = json.loads(
                content.decode("utf-8", errors="strict"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    VisualIngressError(f"invalid JSON numeric constant: {token}")
                ),
            )
        except VisualIngressError:
            raise
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise VisualIngressError("DOM snapshot is not strict UTF-8 JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"viewport_width", "viewport_height", "nodes"}:
            raise VisualIngressError("DOM snapshot has missing or unknown top-level fields")
        viewport_width = self._positive_integer(payload["viewport_width"], "viewport_width")
        viewport_height = self._positive_integer(payload["viewport_height"], "viewport_height")
        nodes_data = payload["nodes"]
        if not isinstance(nodes_data, list) or len(nodes_data) > self._policy.maximum_dom_nodes:
            raise VisualIngressError("DOM node count exceeds the configured limit")
        nodes: list[DomNodeObservation] = []
        seen_ids: set[str] = set()
        total_text_characters = 0
        for node_data in nodes_data:
            if not isinstance(node_data, dict) or set(node_data) != {
                "node_id", "role", "text", "rect", "attributes"
            }:
                raise VisualIngressError("DOM node has missing or unknown fields")
            node_id = self._required_text(node_data["node_id"], "node_id")
            role = self._required_text(node_data["role"], "role")
            if node_id in seen_ids:
                raise VisualIngressError("DOM node IDs must be unique")
            if role not in self._policy.allowed_dom_roles:
                raise VisualIngressError("DOM role is not allowlisted")
            seen_ids.add(node_id)
            text = node_data["text"]
            if not isinstance(text, str) or len(text) > self._policy.maximum_node_text_characters:
                raise VisualIngressError("DOM node text exceeds the configured character limit")
            total_text_characters += len(text)
            if total_text_characters > self._policy.maximum_dom_text_characters:
                raise VisualIngressError("DOM text exceeds the configured total character limit")
            rect = self._parse_rect(node_data["rect"], viewport_width, viewport_height)
            attributes_data = node_data["attributes"]
            if not isinstance(attributes_data, dict):
                raise VisualIngressError("DOM attributes must be an object")
            attributes: list[tuple[str, str]] = []
            for key, value in sorted(attributes_data.items()):
                if key not in self._policy.allowed_dom_attributes:
                    raise VisualIngressError("DOM snapshot includes a non-allowlisted attribute")
                if not isinstance(value, str) or len(value) > self._policy.maximum_node_text_characters:
                    raise VisualIngressError("DOM attribute value exceeds the configured limit")
                attributes.append((key, value))
            nodes.append(DomNodeObservation(node_id, role, text, rect, tuple(attributes)))
        return DomSnapshot(
            source_ref,
            hashlib.sha256(content).hexdigest(),
            viewport_width,
            viewport_height,
            tuple(nodes),
        )

    @staticmethod
    def _required_text(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise VisualIngressError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise VisualIngressError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _parse_rect(payload: object, viewport_width: int, viewport_height: int) -> ViewportRect:
        if not isinstance(payload, dict) or set(payload) != {"x", "y", "width", "height"}:
            raise VisualIngressError("DOM rectangle has missing or unknown fields")
        values = tuple(payload[name] for name in ("x", "y", "width", "height"))
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ):
            raise VisualIngressError("DOM rectangle coordinates must be finite numbers")
        x, y, width, height = (float(value) for value in values)
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > viewport_width
            or y + height > viewport_height
        ):
            raise VisualIngressError("DOM rectangle must fit inside the declared viewport")
        return ViewportRect(x, y, width, height)