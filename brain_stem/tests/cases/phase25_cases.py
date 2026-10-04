from __future__ import annotations

import io
import json
import unittest

from PIL import Image
from src.swarm_core.visual_ingress import (
    VisualIngressAdapter,
    VisualIngressError,
    VisualIngressPolicy,
)


def make_policy() -> VisualIngressPolicy:
    return VisualIngressPolicy(
        maximum_input_bytes=1_000_000,
        maximum_normalized_bytes=1_000_000,
        maximum_width=64,
        maximum_height=64,
        maximum_pixels=4096,
        maximum_dom_nodes=10,
        maximum_node_text_characters=80,
        maximum_dom_text_characters=200,
    )


def image_bytes(width: int = 32, height: int = 24, format: str = "PNG") -> bytes:
    image = Image.new("RGB", (width, height), color=(40, 100, 200))
    output = io.BytesIO()
    image.save(output, format=format)
    image.close()
    return output.getvalue()


def dom_payload(*, attributes: dict[str, str] | None = None, rect=None) -> bytes:
    payload = {
        "viewport_width": 320,
        "viewport_height": 200,
        "nodes": [
            {
                "node_id": "button-1",
                "role": "button",
                "text": "Save",
                "rect": rect or {"x": 10, "y": 20, "width": 80, "height": 30},
                "attributes": attributes or {"aria-label": "Save document", "type": "button"},
            }
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class VisualIngressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = VisualIngressAdapter(make_policy())

    def test_raster_is_bounded_normalized_metadata_free_and_digest_bound(self) -> None:
        source = image_bytes(format="JPEG")

        result = self.adapter.normalize_raster(source, source_ref="browser:viewport")

        self.assertEqual(result.content_type, "image/png")
        self.assertEqual((result.width, result.height), (32, 24))
        self.assertEqual(result.source_sha256, __import__("hashlib").sha256(source).hexdigest())
        with Image.open(io.BytesIO(result.png_bytes)) as normalized:
            self.assertEqual(normalized.format, "PNG")
            self.assertEqual(normalized.info, {})

    def test_raster_pixel_bounds_and_unapproved_formats_are_rejected(self) -> None:
        with self.assertRaisesRegex(VisualIngressError, "dimensions"):
            self.adapter.normalize_raster(image_bytes(100, 100), source_ref="browser:oversized")

        with self.assertRaisesRegex(VisualIngressError, "not allowlisted"):
            self.adapter.normalize_raster(image_bytes(format="BMP"), source_ref="browser:bmp")

        with self.assertRaisesRegex(VisualIngressError, "byte limit"):
            VisualIngressAdapter(
                VisualIngressPolicy(8, 1000, 64, 64, 4096, 10, 80, 200)
            ).normalize_raster(image_bytes(), source_ref="browser:large")

    def test_dom_snapshot_preserves_source_offsets_and_allowlisted_attributes(self) -> None:
        snapshot = self.adapter.parse_dom_snapshot(
            dom_payload(),
            source_ref="browser:dom-snapshot",
        )

        self.assertEqual(snapshot.viewport_width, 320)
        self.assertEqual(snapshot.nodes[0].node_id, "button-1")
        self.assertEqual(snapshot.nodes[0].rect.width, 80.0)
        self.assertEqual(snapshot.nodes[0].attributes, (("aria-label", "Save document"), ("type", "button")))
        self.assertEqual(len(snapshot.source_sha256), 64)

    def test_dom_snapshot_rejects_duplicate_keys_unsafe_attributes_and_bad_geometry(self) -> None:
        duplicate = b'{"viewport_width":320,"viewport_width":321,"viewport_height":200,"nodes":[]}'
        with self.assertRaisesRegex(VisualIngressError, "duplicate JSON key"):
            self.adapter.parse_dom_snapshot(duplicate, source_ref="browser:duplicate")

        with self.assertRaisesRegex(VisualIngressError, "non-allowlisted attribute"):
            self.adapter.parse_dom_snapshot(
                dom_payload(attributes={"onclick": "alert(1)"}),
                source_ref="browser:hostile",
            )

        with self.assertRaisesRegex(VisualIngressError, "fit inside"):
            self.adapter.parse_dom_snapshot(
                dom_payload(rect={"x": 300, "y": 20, "width": 80, "height": 30}),
                source_ref="browser:bad-geometry",
            )

    def test_dom_snapshot_node_and_text_limits_are_enforced(self) -> None:
        adapter = VisualIngressAdapter(
            VisualIngressPolicy(10000, 10000, 64, 64, 4096, 1, 4, 4)
        )
        payload = json.loads(dom_payload(attributes={"type": "btn"}))
        payload["nodes"][0]["text"] = "Longer than four characters"
        with self.assertRaisesRegex(VisualIngressError, "text exceeds"):
            adapter.parse_dom_snapshot(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                source_ref="browser:long-text",
            )


if __name__ == "__main__":
    unittest.main()