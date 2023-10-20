import copy
import os
import random
import xml.etree.ElementTree as ET

from .import DMCR_VARY, SUITE_DIR
from .rng import dmcr_random


def load_asset_file(name):
    with open(name, mode="r") as f:
        return f.read()


_FILENAMES = [
    "./assets/materials.xml",
    "./assets/skybox.xml",
    "./assets/visual.xml",
]
DEFAULT_ASSETS = {
    filename: load_asset_file(os.path.join(SUITE_DIR, filename))
    for filename in _FILENAMES
}

_SKIES_PATH = os.path.join(SUITE_DIR, "assets/skies")
_ALL_SKIES = [
    os.path.abspath(os.path.join(_SKIES_PATH, pth)) for pth in os.listdir(_SKIES_PATH)
]

_FLOORS_PATH = os.path.join(SUITE_DIR, "assets/floors")
_ALL_FLOORS = [
    os.path.abspath(os.path.join(_FLOORS_PATH, pth)) for pth in os.listdir(_FLOORS_PATH)
]


def random_rgb_string():
    return f"{random.random()} {random.random()} {random.random()}"


def random_rgba_string():
    alpha = random.uniform(0.75, 1.0)
    return f"{random.random()} {random.random()} {random.random()} {alpha}"


def get_assets(visual_seed, vary=DMCR_VARY):
    choices_dict = {
        "background": "default",
        "floor": "default",
        "body": "default",
        "target": "default",
        "reflectance": "default",
        "ambience": "default",
        "diffuse": "default",
        "specular": "default",
        "shadowsize": "default",
    }

    if visual_seed == 0:
        return DEFAULT_ASSETS, choices_dict
    else:
        new_assets = copy.deepcopy(DEFAULT_ASSETS)

    with dmcr_random(visual_seed):
        skybox_xml = ET.fromstring(new_assets["./assets/skybox.xml"])
        materials_xml = ET.fromstring(new_assets["./assets/materials.xml"])
        visual_xml = ET.fromstring(new_assets["./assets/visual.xml"])

        # positive seeds use random photos as a background
        if visual_seed > 0:
            # change background texture
            sky = random.choice(_ALL_SKIES)
            if "bg" in vary:
                skybox_xml[0][0].attrib["builtin"] = "none"
                skybox_xml[0][0].attrib["file"] = sky

            # change floor texture
            floor = random.choice(_ALL_FLOORS)
            reflectance = str(random.uniform(0.0, 0.5))
            if "floor" in vary:
                materials_xml[0][0].attrib["builtin"] = "none"
                materials_xml[0][0].attrib["file"] = floor
                choices_dict["floor"] = floor
            if "reflectance" in vary:
                materials_xml[0][1].attrib["reflectance"] = reflectance
                choices_dict["background"] = sky

        # negative seeds use random RGB variations of the default
        else:
            # change background color
            skybox_xml[0][0].attrib["rgb1"] = random_rgb_string()
            # change star color
            skybox_xml[0][0].attrib["rgb2"] = random_rgb_string()

            # change the floor color
            materials_xml[0][0].attrib["rgb1"] = random_rgb_string()
            materials_xml[0][0].attrib["rgb2"] = random_rgb_string()

        # change agent body color
        body_color = random_rgba_string()
        effector_color = random_rgba_string()
        if "body" in vary:
            choices_dict["body"] = body_color
            materials_xml[0][5].attrib["rgba"] = effector_color
            materials_xml[0][2].attrib["rgba"] = body_color

        # change target color (used in Fish Swim, for example)
        target_color = random_rgba_string()
        if "target" in vary:
            materials_xml[0][10].attrib["rgba"] = target_color
            choices_dict["target"] = target_color

        # change the light
        amb_base = random.uniform(0.1, 0.8)
        amb_delr = random.uniform(-0.05, 0.05)
        amb_delg = random.uniform(-0.05, 0.05)
        amb_delb = random.uniform(-0.05, 0.05)
        ambience = f"{amb_base+amb_delr} {amb_base+amb_delg} {amb_base+amb_delb}"
        if "light" in vary:
            visual_xml[0][0].attrib["ambient"] = ambience
            choices_dict["ambience"] = ambience

        dif_base = random.uniform(0.4, 0.9)
        dif_delr = random.uniform(-0.1, 0.1)
        dif_delg = random.uniform(-0.1, 0.1)
        dif_delb = random.uniform(-0.1, 0.1)
        diffuse = f"{dif_base+dif_delr} {dif_base+dif_delg} {dif_base+dif_delb}"
        if "light" in vary:
            visual_xml[0][0].attrib["diffuse"] = diffuse
            choices_dict["diffuse"] = diffuse

        spec_base = random.uniform(0.05, 0.3)
        spec_delr = random.uniform(-0.02, 0.02)
        spec_delg = random.uniform(-0.02, 0.02)
        spec_delb = random.uniform(-0.02, 0.02)
        specular = f"{spec_base+spec_delr} {spec_base+spec_delg} {spec_base+spec_delb}"
        if "light" in vary:
            visual_xml[0][0].attrib["specular"] = specular
            choices_dict["specular"] = specular

        shadow_size = str(random.randint(2048 - 500, 2048 + 500))
        if "light" in vary:
            visual_xml[0][2].attrib["shadowsize"] = shadow_size
            choices_dict["shadowsize"] = shadow_size

        new_assets["./assets/skybox.xml"] = ET.tostring(
            skybox_xml, encoding="utf8", method="xml"
        )
        new_assets["./assets/materials.xml"] = ET.tostring(
            materials_xml, encoding="utf8", method="xml"
        )
        new_assets["./assets/visual.xml"] = ET.tostring(
            visual_xml, encoding="utf8", method="xml"
        )
    return new_assets, choices_dict
