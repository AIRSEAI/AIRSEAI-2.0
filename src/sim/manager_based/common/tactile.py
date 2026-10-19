# SPDX-License-Identifier: Apache-2.0
"""Batched TacSL contact views with independently selected targets per slot."""

import torch
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor


class CompatibleTactileSensor(VisuoTactileSensor):
    """Keep TacSL camera output compatible with its persistent tensor buffer."""

    def _update_camera_tactile(self, env_ids):
        if self._nominal_tactile is None:
            raise RuntimeError("Call get_initial_render before reading tactile RGB")
        self._camera_sensor.update(self._sim_physics_dt)
        output = self._camera_sensor.data.output
        depth_key = next((key for key in ("distance_to_image_plane", "depth") if key in output), None)
        if depth_key is None:
            return
        depth = self._data.tactile_depth_image
        depth[env_ids] = output[depth_key][env_ids].to(depth.dtype)
        difference = self._nominal_tactile[depth_key][env_ids] - depth[env_ids]
        rgb = self._data.tactile_rgb_image
        rgb[env_ids] = self._tactile_rgb_render.render(difference.squeeze(-1)).to(rgb.dtype)


class SelectableTactileSensor(CompatibleTactileSensor):
    target_names = ()

    def _initialize_impl(self):
        super()._initialize_impl()
        self._target_views = {}
        original = self.cfg.contact_object_prim_path_expr
        self.default_target = original.rsplit("/", 1)[-1]
        if self.default_target not in self.target_names:
            raise ValueError("The default tactile target must be in target_names")
        for name in self.target_names:
            self.cfg.contact_object_prim_path_expr = f"/World/envs/env_.*/{name}"
            self._create_physx_views()
            if self._contact_object_body_view.count != self._num_envs:
                raise RuntimeError(f"TacSL target view is not one body per environment: {name}")
            self._target_views[name] = (
                self._contact_object_sdf_view,
                self._contact_object_body_view,
                self._contact_object_com_b,
            )
        self.cfg.contact_object_prim_path_expr = original
        self.target_indices = torch.full(
            (self._num_envs,),
            self.target_names.index(self.default_target),
            dtype=torch.long,
            device=self._device,
        )

    def select_target(self, env_ids, name):
        if name not in self._target_views:
            raise ValueError(f"Unknown tactile target {name}")
        self.target_indices[env_ids] = list(self._target_views).index(name)
        self.reset(env_ids)

    def _update_force_field(self, env_ids):
        if not hasattr(self, "_target_views"):
            return super()._update_force_field(env_ids)
        requested = torch.arange(self._num_envs, device=self._device)[env_ids]
        # The force kernel requires full-size buffers; merge only selected slots.
        fields = ("penetration_depth", "tactile_normal_force", "tactile_shear_force")
        output = {name: getattr(self._data, name).clone() for name in fields}
        for index, views in enumerate(self._target_views.values()):
            selected = requested[self.target_indices[requested] == index]
            if selected.numel():
                (
                    self._contact_object_sdf_view,
                    self._contact_object_body_view,
                    self._contact_object_com_b,
                ) = views
                super()._update_force_field(slice(None))
                for name in fields:
                    output[name][selected] = getattr(self._data, name)[selected]
        for name in fields:
            getattr(self._data, name).copy_(output[name])
