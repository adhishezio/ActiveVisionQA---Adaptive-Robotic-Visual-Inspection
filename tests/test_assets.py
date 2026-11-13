"""Workcell/collision-model agreement and generated marking provenance."""

from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from activevisionqa.assets import generate_workcell, scene_boxes
from activevisionqa.config import Config
from activevisionqa.models import Outcome
from activevisionqa.perception import MarkerInspector


@pytest.mark.parametrize("scenario,expected", [("clear", Outcome.ACCEPT), ("wrong_id", Outcome.REJECT)])
def test_generated_marking_texture_matches_identifier(tmp_path, scenario, expected):
    path = generate_workcell(tmp_path / "world", scenario, Config())
    image = np.asarray(Image.open(path.parent / "marking.png").convert("RGB"))[:, :, ::-1].copy()
    assert MarkerInspector(Config()).inspect(image).outcome == expected


@pytest.mark.parametrize("scenario,count", [("clear", 2), ("occluded", 3), ("all_occluded", 4),
                                           ("top_occluded", 5), ("covered", 3), ("three_sides_occluded", 5)])
def test_physics_and_planning_collision_boxes_match(tmp_path, scenario, count):
    config = replace(Config(), target=(.45, .03, .25))
    path = generate_workcell(tmp_path / "world", scenario, config)
    world = ET.parse(path).find("world")
    boxes = scene_boxes(scenario, config)
    assert len(boxes) == count
    for box in boxes:
        model = world.find(f"model[@name='{box['name']}']")
        assert model is not None
        pose = list(map(float, model.findtext("pose").split()))
        dimensions = list(map(float, model.findtext("link/collision/geometry/box/size").split()))
        np.testing.assert_allclose(pose[:3], box["position"])
        np.testing.assert_allclose(dimensions, box["size"])
    part = world.find("model[@name='inspection_part']/link")
    cells = [v for v in part.findall("visual") if v.get("name").startswith("marking_cell_")]
    assert len(cells) > 20


def test_world_generation_rejects_unknown_cases_and_existing_directories(tmp_path):
    with pytest.raises(ValueError):
        generate_workcell(tmp_path / "invalid", "not_a_case", Config())
    generate_workcell(tmp_path / "world", "clear", Config())
    with pytest.raises(FileExistsError):
        generate_workcell(tmp_path / "world", "clear", Config())
