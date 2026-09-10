"""S4 机器人接触诊断：先检验真实自由插头与工具，不以此替代多种子任务验收。"""

import mujoco
import numpy as np

from ibero.robots.g1_industrial import robot_spec, robot_handles
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.materials.parameters import BeamParameters, LatchParameters
from ibero.mechanisms.snap_latch import add_latch, LatchObserver
from ibero.control.latch import LatchArmController


# 运动学筛选所得碰撞自由准备姿态；只在 reset 初始化，不在运行时写关节位姿。
RESET_ARMS = [
    -0.4696653478,
    0.1480567558,
    -0.4091856842,
    0.5957613957,
    -0.3352425581,
    -0.2216617468,
    0.3704244701,
    -0.5299340992,
    0.0325193249,
    0.3605278197,
    0.9340626702,
    0.1998618952,
    -0.4490535980,
    -0.1869254598,
]


class RobotLatchDiagnostic:
    def __init__(self):
        self.origin = np.array([0.48, 0, 0.87])
        spec = robot_spec()
        self.names = add_latch(
            spec,
            BeamParameters(),
            LatchParameters(),
            fixture=False,
            origin=self.origin,
            quaternion=[2**-0.5, 0, 0, 2**-0.5],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.handles = robot_handles(self.model)
        self.controller = LatchArmController(self.model, self.handles, 0.01)
        material_geoms = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith(("tongue_", "latch_hook"))
        ]
        self.observer = LatchObserver(
            self.model, self.names, audited_geom_ids=material_geoms
        )
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for name, value in zip(ARM_JOINTS, RESET_ARMS):
            self.data.joint(name).qpos[0] = value
        self.observer.reset()
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset(self.data)
        self.data.ctrl[self.handles.gripper_actuator_ids[0]] = 0
        self.left_target = self.data.site_xpos[self.handles.left_anchor_site_id].copy()
        self.right_target = self.data.site_xpos[
            self.handles.right_anchor_site_id
        ].copy()
        self.left_rotation = (
            self.data.site_xmat[self.handles.left_anchor_site_id].reshape(3, 3).copy()
        )
        self.right_rotation = (
            self.data.site_xmat[self.handles.right_anchor_site_id].reshape(3, 3).copy()
        )

    def step(self, grip=1):
        action = np.zeros(14)
        for offset, site, target, rotation in (
            (0, self.handles.left_anchor_site_id, self.left_target, self.left_rotation),
            (
                7,
                self.handles.right_anchor_site_id,
                self.right_target,
                self.right_rotation,
            ),
        ):
            action[offset : offset + 3] = np.clip(
                (target - self.data.site_xpos[site])
                * 0.3
                / self.controller.action_scale[:3],
                -0.05,
                0.05,
            )
            current = self.data.site_xmat[site].reshape(3, 3)
            dr = 0.5 * sum(np.cross(current[:, i], rotation[:, i]) for i in range(3))
            action[offset + 3 : offset + 6] = np.clip(
                dr * 0.2 / self.controller.action_scale[3:], -0.05, 0.05
            )
        action[13] = grip
        self.controller.apply(self.data, action)
        for _ in range(round(0.01 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # 此诊断读取所有接触，机器人抓取力与舌片有效载荷的正式分账仍需完成。
        state = self.observer.observe(self.data)
        state["plug_position"] = self.data.body("latch_plug").xpos.copy().tolist()
        return state
