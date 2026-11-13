"""Generate self-authored test textures and a Gazebo workcell from one config."""

from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

from .config import Config
from .fixtures import marker_image
from .markings import TEXT_HEIGHT_M, TEXT_WIDTH_M, TEXT_FONT, ink_rectangles, surface_text, text_bitmap

WORLD_SCENARIOS = ("clear", "occluded", "wrong_id", "all_occluded", "three_sides_occluded", "top_occluded", "covered")


def scene_boxes(scenario: str, config: Config | None = None) -> list[dict]:
    """Collision geometry in world meters, also mirrored into MoveIt's scene."""
    config = config or Config()
    x, y, z = config.target
    table_height = z - .042
    boxes = [
        {"name": "inspection_table", "size": [0.40, 0.42, table_height], "position": [x, y, table_height/2]},
        {"name": "inspection_part", "size": [0.18, 0.18, 0.04], "position": [x, y, z-.022]},
    ]
    if scenario in ("occluded", "all_occluded", "three_sides_occluded", "top_occluded"):
        boxes.append({"name": "occluder_front", "size": [0.19, 0.012, 0.15], "position": [x, y-.065, z+.098]})
    if scenario in ("all_occluded", "three_sides_occluded", "top_occluded"):
        boxes.append({"name": "occluder_back", "size": [0.19, 0.012, 0.15], "position": [x, y+.065, z+.098]})
    if scenario == "three_sides_occluded":
        boxes.append({"name": "occluder_left", "size": [0.012, 0.19, 0.15], "position": [x-.065, y, z+.098]})
    if scenario == "covered":
        boxes.append({"name": "inspection_cover", "size": [.21, .12, .035], "position": [x, y, z+.0275]})
    if scenario == "top_occluded":
        boxes.append({"name": "overhead_occluder", "size": [.10, .08, .012], "position": [x, y, z+.20]})
    return boxes


def _sub(parent, tag: str, text: str | None = None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = text
    return child


def generate_workcell(directory: str | Path, scenario: str, config: Config) -> Path:
    """Create a world and textured part. No remote Gazebo assets are required."""
    from PIL import Image, ImageDraw
    if scenario not in WORLD_SCENARIOS:
        raise ValueError("Unsupported Gazebo scenario")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    image = Image.new("L", (512, 512), 255)
    identifier = 23 if scenario == "wrong_id" else config.expected_id
    if config.marking_kind == "text":
        bitmap = text_bitmap(surface_text(config, scenario))
        image = Image.fromarray(bitmap)
    else:
        image.paste(Image.fromarray(marker_image(identifier, 384)), (64, 48))
        draw = ImageDraw.Draw(image)
        draw.text((120, 459), f"AVQA-{identifier:03d}", fill=0, stroke_width=1)
    texture = root / "marking.png"
    image.convert("RGB").save(texture)
    (root / "marking_metadata.json").write_text(json.dumps({
        "kind": config.marking_kind, "surface_text": surface_text(config, scenario) if config.marking_kind == "text" else None,
        "expected_text": config.expected_text if config.marking_kind == "text" else None,
        "font": TEXT_FONT if config.marking_kind == "text" else None,
        "synthetic": True, "physical_engraving_simulated": False,
    }, indent=2)+"\n")
    sdf = ET.Element("sdf", version="1.9")
    world = _sub(sdf, "world", name="activevisionqa")
    physics = _sub(world, "physics", name="deterministic", type="ignored")
    _sub(physics, "max_step_size", "0.002")
    _sub(physics, "real_time_factor", "1.0")
    for filename, name in (("physics", "Physics"), ("user-commands", "UserCommands"),
                           ("scene-broadcaster", "SceneBroadcaster"), ("sensors", "Sensors")):
        plugin = _sub(world, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}")
        if name == "Sensors":
            _sub(plugin, "render_engine", "ogre2")
    scene = _sub(world, "scene")
    _sub(scene, "ambient", " ".join([str(min(1., .65*config.light_scale))]*3)+" 1")
    _sub(scene, "background", "0.30 0.35 0.40 1")
    light = _sub(world, "light", name="sun", type="directional")
    _sub(light, "pose", "0 0 5 0 0 0")
    _sub(light, "diffuse", " ".join([str(min(1., .9*config.light_scale))]*3)+" 1")
    _sub(light, "specular", "0.1 0.1 0.1 1")
    _sub(light, "direction", "-0.3 0.2 -1")
    _sub(light, "cast_shadows", "true")
    ground = _sub(world, "model", name="ground")
    _sub(ground, "static", "true")
    ground_link = _sub(ground, "link", name="link")
    for kind in ("collision", "visual"):
        element = _sub(ground_link, kind, name="ground_"+kind)
        plane = _sub(_sub(element, "geometry"), "plane")
        _sub(plane, "normal", "0 0 1")
        _sub(plane, "size", "5 5")
    for box in scene_boxes(scenario, config):
        model = _sub(world, "model", name=box["name"])
        _sub(model, "static", "true")
        _sub(model, "pose", " ".join(map(str, box["position"]))+" 0 0 0")
        link = _sub(model, "link", name="link")
        for kind in ("collision", "visual"):
            element = _sub(link, kind, name=kind)
            geometry = _sub(_sub(element, "geometry"), "box")
            _sub(geometry, "size", " ".join(map(str, box["size"])))
            if kind == "visual":
                material = _sub(element, "material")
                color = "0.10 0.12 0.15 1" if "occluder" in box["name"] else "0.55 0.57 0.59 1"
                _sub(material, "ambient", color)
                _sub(material, "diffuse", color)
        if box["name"] == "inspection_part":
            yaw = config.marking_yaw_rad
            visual = _sub(link, "visual", name="marking")
            _sub(visual, "pose", f"0 0 0.022 0 0 {yaw}")
            board_height = TEXT_HEIGHT_M if config.marking_kind == "text" else .15
            _sub(_sub(_sub(visual, "geometry"), "box"), "size", f"{TEXT_WIDTH_M} {board_height} 0.0001")
            material = _sub(visual, "material")
            _sub(material, "ambient", "1 1 1 1")
            _sub(material, "diffuse", "1 1 1 1")
            _sub(visual, "cast_shadows", "false")
            if config.marking_kind == "text":
                height, width = bitmap.shape
                for index, (x0, y0, x1, y1) in enumerate(ink_rectangles(bitmap)):
                    x = ((x0+x1)/2-width/2)*TEXT_WIDTH_M/width
                    y = (height/2-(y0+y1)/2)*TEXT_HEIGHT_M/height
                    cx, cy = x*math.cos(yaw)-y*math.sin(yaw), x*math.sin(yaw)+y*math.cos(yaw)
                    cell = _sub(link, "visual", name=f"text_ink_{index}")
                    _sub(cell, "pose", f"{cx} {cy} 0.02211 0 0 {yaw}")
                    _sub(_sub(_sub(cell, "geometry"), "box"), "size",
                         f"{(x1-x0)*TEXT_WIDTH_M/width} {(y1-y0)*TEXT_HEIGHT_M/height} 0.0001")
                    material = _sub(cell, "material")
                    _sub(material, "ambient", "0 0 0 1")
                    _sub(material, "diffuse", "0 0 0 1")
                    _sub(cell, "cast_shadows", "false")
                continue
            # Native flat visual cells avoid engine-specific Collada texture handling.
            # These form the same 4x4 identifier plus its one-cell black border.
            bits = marker_image(identifier, 60)
            cell_size = .1125 / 6
            for row in range(6):
                for column in range(6):
                    if bits[row*10+5, column*10+5] != 0:
                        continue
                    cell = _sub(link, "visual", name=f"marking_cell_{row}_{column}")
                    x, y = (column-2.5)*cell_size, (2.5-row)*cell_size
                    cx, cy = x*math.cos(yaw)-y*math.sin(yaw), x*math.sin(yaw)+y*math.cos(yaw)
                    _sub(cell, "pose", f"{cx} {cy} 0.02211 0 0 {yaw}")
                    _sub(_sub(_sub(cell, "geometry"), "box"), "size", f"{cell_size} {cell_size} 0.0001")
                    material = _sub(cell, "material")
                    _sub(material, "ambient", "0 0 0 1")
                    _sub(material, "diffuse", "0 0 0 1")
                    _sub(cell, "cast_shadows", "false")
    ET.indent(sdf)
    path = root / "workcell.sdf"
    ET.ElementTree(sdf).write(path, encoding="utf-8", xml_declaration=True)
    return path
