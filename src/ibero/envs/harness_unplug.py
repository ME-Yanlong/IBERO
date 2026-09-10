"""同一自由卡扣加真实 Flex 尾线；材料相互作用在每个物理子步生效。"""

from pathlib import Path
import mujoco
import numpy as np
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.materials.mechanics import CableMechanics
from ibero.core.loads import AppliedLoads


class HarnessUnplugEnv(LatchReleaseEnv):
    scene_kind = "harness_unplug"

    def __init__(self, scene_path=None, render_mode=None):
        super().__init__(
            scene_path or Path(__file__).resolve().parents[3] / "scenes/harness_unplug",
            render_mode,
        )

    def _reset_materials(self, cell):
        self.mechanics = CableMechanics(self.model, cell.harness_parameters)
        self.loads = AppliedLoads(self.model)
        self.peak_tension_n = 0.0
        self.peak_cable_penetration_m = 0.0
        self.cable_tension_n = 0.0
        self.receiver = self.model.geom("harness_receiver").id

    def _before_substep(self):
        self.loads.begin()
        self.mechanics.apply(self.data, self.loads)
        self.loads.commit(self.data)

    def _after_substep(self):
        self.cable_tension_n = self.mechanics.tension(self.data)
        self.peak_tension_n = max(self.peak_tension_n, self.cable_tension_n)
        safety = self.scene.constraints["safety"]
        if self.cable_tension_n > safety["cable_tension_limit_n"]:
            self.invalid_reason = "cable_over_tension"
        flex_contacts = np.any(self.data.contact.flex >= 0, axis=1)
        if flex_contacts.any():
            depth = float(np.max(-self.data.contact.dist[flex_contacts]))
            self.peak_cable_penetration_m = max(self.peak_cable_penetration_m, depth)
            if depth > safety["cable_penetration_m"]:
                self.invalid_reason = "cable_penetration"

    def _material_task_state(self):
        # 放置成功必须有插头实体—托盘承载接触，不接受线尾接触或仅进入坐标框。
        supported = False
        f = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == self.receiver else a if b == self.receiver else -1
            if other >= 0 and self.model.geom_bodyid[other] == self.plug:
                mujoco.mj_contactForce(self.model, self.data, i, f)
                supported |= f[0] > 0.05
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.plug, velocity, 0
        )
        receiver = self.scene.config["workcell"]["receiver"]
        center = np.asarray(receiver["center_m"])
        half = np.asarray(receiver["half_size_m"])
        within = np.all(np.abs(self.data.xpos[self.plug, :2] - center[:2]) < half[:2])
        cable = self.mechanics.state(self.data)
        return {
            "cable_tension_n": self.cable_tension_n,
            "peak_cable_tension_n": self.peak_tension_n,
            "peak_cable_penetration_m": self.peak_cable_penetration_m,
            "cable_state": cable,
            "receiver_supported": bool(supported),
            "receiver_contains_plug": bool(within),
            "plug_speed_m_s": float(np.linalg.norm(velocity[3:])),
            "plug_angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
            "placement_speed_limit_m_s": self.scene.constraints["task"][
                "placement_speed_limit_m_s"
            ],
        }
