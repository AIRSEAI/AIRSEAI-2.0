"""Build the glassware_tray_transfer environment."""

from sim.envs import glassware_tray_transfer as scene
from sim.manager_based.common.env_cfg import build_cfg


def make_cfg(**kwargs):
    cfg = build_cfg(scene, **kwargs)
    cfg.sim.render.enable_dl_denoiser = True
    cfg.sim.render.samples_per_pixel = 4
    cfg.sim.render.carb_settings = {"/rtx/raytracing/cached/enabled": False}
    return cfg
