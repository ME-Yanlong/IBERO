"""离散线束材料状态：实际边弹簧张力与能量一致的弯曲力。"""

from __future__ import annotations

import mujoco
import numpy as np


def bending_energy_force(points, rigidity, rest_step):
    """E=EI/(2h) Σ|t[i+1]-t[i]|²；其负梯度产生旋转不变弯曲力。

    采用无扭转的离散中心线模型，适合圆截面线束；不是三维实体应力模型。
    """
    edges = np.diff(points, axis=0)
    lengths = np.maximum(np.linalg.norm(edges, axis=1), 1e-9)
    tangent = edges / lengths[:, None]
    delta = np.diff(tangent, axis=0)
    scale = rigidity / rest_step
    gradient_t = np.zeros_like(tangent)
    gradient_t[:-1] -= scale * delta
    gradient_t[1:] += scale * delta
    gradient_e = (
        gradient_t - tangent * np.sum(tangent * gradient_t, axis=1)[:, None]
    ) / lengths[:, None]
    force = np.zeros_like(points)
    force[:-1] += gradient_e
    force[1:] -= gradient_e
    return float(0.5 * scale * np.sum(delta * delta)), force


class CableMechanics:
    def __init__(self, model, params):
        self.model, self.params = model, params
        f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cable_flex")
        self.edges = slice(
            model.flex_edgeadr[f], model.flex_edgeadr[f] + model.flex_edgenum[f]
        )
        self.vertices = slice(
            model.flex_vertadr[f], model.flex_vertadr[f] + model.flex_vertnum[f]
        )
        self.bodies = model.flex_vertbodyid[self.vertices]
        # flexcomp 为每个节点创建三平移自由度；坐标轴与世界系相同。
        self.dofs = np.array(
            [model.jnt_dofadr[model.body_jntadr[b]] for b in self.bodies]
        )
        self.indices = self.dofs[:, None] + np.arange(3)
        self.flex_id = f

    def apply(self, data, loads=None):
        # mj_step 返回的 xpos 是积分前位置；外加材料力必须由当前 qpos 重算，
        # 否则一子步相位滞后会向弯曲模态注入非物理能量。
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_flex(self.model, data)
        # 只覆盖材料节点的外加广义力，不碰机器人控制或实验外力。
        _, force = bending_energy_force(
            data.flexvert_xpos[self.vertices],
            self.params.bending_stiffness_nm2,
            self.params.length_m / self.params.segments,
        )
        if loads is None:
            # 保持原有线束入口语义，避免改变已经冻结的基线积分路径。
            data.qfrc_applied[self.indices] = force
        else:
            loads.add_generalized(self.indices, force)

    def state(self, data):
        m, p = self.model, self.params
        # 与 MuJoCo engine_passive.c 使用完全相同的力律和真实边长/速度。
        signed = (
            m.flex_edgestiffness[self.flex_id]
            * (data.flexedge_length[self.edges] - m.flexedge_length0[self.edges])
            + m.flex_edgedamping[self.flex_id] * data.flexedge_velocity[self.edges]
        )
        tensile = np.maximum(signed, 0)
        pts = data.flexvert_xpos[self.vertices]
        v = np.diff(pts, axis=0)
        lengths = np.linalg.norm(v, axis=1)
        tangent = v / np.maximum(lengths[:, None], 1e-9)
        curvature = np.linalg.norm(np.diff(tangent, axis=0), axis=1) / np.maximum(
            (lengths[:-1] + lengths[1:]) / 2, 1e-9
        )
        radius = float(1 / max(float(np.max(curvature)), 1e-9))
        return {
            "tension_n": float(np.max(tensile)),
            "endpoint_tension_n": [float(tensile[0]), float(tensile[-1])],
            "arc_length_m": float(lengths.sum()),
            "minimum_radius_m": radius,
            "bend_alarm": radius < p.minimum_bend_radius_m,
            "bending_energy_j": bending_energy_force(
                pts, p.bending_stiffness_nm2, p.length_m / p.segments
            )[0],
        }

    def tension(self, data):
        m = self.model
        return float(
            max(
                0,
                np.max(
                    m.flex_edgestiffness[self.flex_id]
                    * (
                        data.flexedge_length[self.edges]
                        - m.flexedge_length0[self.edges]
                    )
                    + m.flex_edgedamping[self.flex_id]
                    * data.flexedge_velocity[self.edges]
                ),
            )
        )
