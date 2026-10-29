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


"""Cavity estimates from a mesh: a synthetic cup of known shape, and the frame."""

import numpy as np
import pytest


def _cup(r_bottom=0.035, r_top=0.045, floor=0.006, rim=0.10, wall=0.004, shift=(0.0, 0.0, 0.0)):
    """A watertight cup of revolution about z: outer wall, rim, the inner frustum and the floor, plus a handle box."""
    import trimesh

    prof = np.array([[0.0, 0.0], [r_top + wall, 0.0], [r_top + wall, rim], [r_top, rim], [r_bottom, floor], [0.0, floor]])
    cup = trimesh.creation.revolve(prof, sections=96)
    handle = trimesh.creation.box(extents=(0.02, 0.012, 0.06), transform=trimesh.transformations.translation_matrix(
        (r_top + wall + 0.012, 0.0, 0.05)))
    m = trimesh.util.concatenate([cup, handle])
    m.apply_translation(shift)
    return m


def test_cavity_estimate_on_a_synthetic_cup():
    from robots.mujoco_lab.cavities import estimate_mesh

    c = estimate_mesh(_cup())
    assert np.allclose(c["axis_obj"], [0.0, 0.0, 1.0], atol=1e-9)
    assert c["r_bottom_m"] == pytest.approx(0.035, rel=0.05) and c["r_top_m"] == pytest.approx(0.045, rel=0.05)
    assert c["height_m"] == pytest.approx(0.094, rel=0.05)
    assert np.allclose(c["bottom_obj"], [0.0, 0.0, 0.006], atol=0.003)


def test_cavity_estimate_follows_the_cup():
    """The handle moves the bounding box, not the cavity: the axis passes through the cup's centre."""
    from robots.mujoco_lab.cavities import estimate_mesh

    c = estimate_mesh(_cup(shift=(0.02, -0.01, 0.03)))
    assert np.allclose(c["bottom_obj"], [0.02, -0.01, 0.036], atol=0.003)


def test_a_deep_ring_under_the_bowl_does_not_take_the_axis():
    """A goblet: seen from below, the rays beside the stem rise 6 cm to the bowl's underside, deeper than the cavity
    (5.5 cm) seen from above. That ring has no floor under its rim, so the cavity stays the bowl's."""
    import trimesh

    from robots.mujoco_lab.cavities import estimate_mesh

    prof = np.array([[0.0, 0.0], [0.03, 0.0], [0.03, 0.005], [0.006, 0.005], [0.006, 0.06], [0.04, 0.06], [0.04, 0.12],
                     [0.036, 0.12], [0.036, 0.065], [0.0, 0.065]])
    c = estimate_mesh(trimesh.creation.revolve(prof, sections=96))
    assert np.allclose(c["axis_obj"], [0.0, 0.0, 1.0], atol=1e-9)
    assert c["r_bottom_m"] == pytest.approx(0.036, rel=0.05) and c["r_top_m"] == pytest.approx(0.036, rel=0.05)
    assert c["height_m"] == pytest.approx(0.055, rel=0.05)
    assert np.allclose(c["bottom_obj"], [0.0, 0.0, 0.065], atol=0.003)


def test_a_round_bowl_keeps_its_rim_and_its_volume():
    """A hemispherical bowl (inner radius 6 cm): a line through the radii overshoots the rim of a curved wall, so the
    frustum takes the rim's own radius and the volume measured at the 10 heights."""
    import trimesh

    from robots.mujoco_lab.cavities import estimate_mesh
    from robots.mujoco_lab.fragility import liquid

    t = np.linspace(0.0, np.pi / 2, 40)
    outer = np.column_stack([0.065 * np.sin(t), 0.065 - 0.065 * np.cos(t)])
    inner = np.column_stack([0.06 * np.sin(t), 0.065 - 0.06 * np.cos(t)])[::-1]
    c = estimate_mesh(trimesh.creation.revolve(np.vstack([outer, inner]), sections=96))
    assert c["r_top_m"] == pytest.approx(0.06, rel=0.03)
    assert c["height_m"] == pytest.approx(0.06, rel=0.03)
    assert liquid(c, 1.0)["volume_l"] == pytest.approx(2.0 / 3.0 * np.pi * 0.06 ** 3 * 1000.0, rel=0.03)


def test_a_solid_has_no_cavity():
    import trimesh

    from robots.mujoco_lab.cavities import estimate_mesh

    with pytest.raises(ValueError, match="no cavity"):
        estimate_mesh(trimesh.creation.box(extents=(0.08, 0.08, 0.1)))
