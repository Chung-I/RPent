# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The destination registry for the pick_place skill."""

from rpent.skills.pick_place.destinations import (
    DESTINATIONS,
    edge_distance,
    has_surface,
    release_height,
)

PRIMITIVES = {"table_centre", "table_edge", "table_gap", "mouse_pad", "plate", "box", "bin"}
KEYS = {"id", "kind", "xy", "surface_z", "mu", "rim_height", "opening_half_width", "fragile", "object_rest_xy"}


def test_destinations_table_is_complete():
    assert PRIMITIVES <= set(DESTINATIONS)
    for name in PRIMITIVES:
        entry = DESTINATIONS[name]
        assert set(entry) >= KEYS, name
        assert entry["id"] == name
        assert entry["kind"] in {"surface", "rim", "cavity", "open_bin", "none"}


def test_kinds_and_friction():
    assert DESTINATIONS["table_gap"]["kind"] == "none"
    assert DESTINATIONS["plate"]["kind"] == "rim" and DESTINATIONS["plate"]["fragile"] is True
    assert DESTINATIONS["box"]["kind"] == "cavity" and DESTINATIONS["box"]["opening_half_width"] == 0.08
    assert DESTINATIONS["bin"]["kind"] == "open_bin"
    assert DESTINATIONS["mouse_pad"]["mu"] == 1.2 and DESTINATIONS["plate"]["mu"] == 0.4


def test_release_height_rule():
    assert release_height(DESTINATIONS["table_centre"]) == 0.02
    assert release_height(DESTINATIONS["plate"]) == 0.02
    assert release_height(DESTINATIONS["box"]) == 0.02 and release_height(DESTINATIONS["bin"]) == 0.02
    assert release_height(DESTINATIONS["table_gap"]) == 0.02
    narrow = dict(DESTINATIONS["box"], opening_half_width=0.05, rim_height=0.10)
    assert abs(release_height(narrow) - 0.15) < 1e-9


def test_has_surface_and_edge_distance():
    assert has_surface(DESTINATIONS["mouse_pad"]) and not has_surface(DESTINATIONS["table_gap"])
    assert abs(edge_distance(DESTINATIONS["table_edge"]) - 0.06) < 1e-9
    assert edge_distance(DESTINATIONS["table_gap"]) < 0
    assert edge_distance(DESTINATIONS["bin"]) < 0
    assert abs(edge_distance(DESTINATIONS["mouse_pad"]) - 0.08) < 1e-9


def test_fixtures_clear_the_object_rest_box_and_fit_the_table():
    from rpent.skills.pick_place.destinations import OBJECT_REST_XY, TABLE_HALF
    ox, oy = OBJECT_REST_XY
    box = (ox - 0.17 - 0.02, ox + 0.17 + 0.02, oy - 0.07 - 0.02, oy + 0.07 + 0.02)   # longest object plus 2 cm
    extents = {"mouse_pad": (0.10, 0.06), "plate": (0.12, 0.12), "box": (0.09, 0.09)}
    for name, (hx, hy) in extents.items():
        x, y = DESTINATIONS[name]["xy"]
        assert abs(x) + hx <= TABLE_HALF[0] and abs(y) + hy <= TABLE_HALF[1], name
        overlaps = (x - hx < box[1] and x + hx > box[0] and y - hy < box[3] and y + hy > box[2])
        assert not overlaps, name
