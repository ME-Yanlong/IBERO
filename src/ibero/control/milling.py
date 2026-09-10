"""机器人加工姿态伺服；只写原执行器命令，不改关节状态或能力。"""

import mujoco
import numpy as np
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.materials.parameters import finite_number


class MillingArmServo:
    """笛卡尔阻抗映射到原位置执行器；重力补偿也计入同一力矩限值。

    原位置执行器 tau = kp*ctrl + bias(q,qvel)，反解 ctrl 下发期望力矩。
    没有绕过执行器的 qfrc_applied；原关节 actuatorfrcrange 仍由引擎限幅。
    下发饱和单独记录，不能把饱和后的轨迹完成当成形状合格。
    """

    def __init__(
        self,
        model,
        data,
        *,
        position_kp=50000.0,
        position_kd=500.0,
        rotation_kp=200.0,
        rotation_kd=12.0,
    ):
        self.model = model
        self.actuators = np.array([model.actuator(n).id for n in ARM_JOINTS])
        self.joints = model.actuator_trnid[self.actuators, 0]
        self.dofs = model.jnt_dofadr[self.joints]
        self.qadr = model.jnt_qposadr[self.joints]
        self.site = model.site("mill_tip").id
        self.kp, self.kd = (
            finite_number(position_kp, "position_kp"),
            finite_number(position_kd, "position_kd"),
        )
        self.kr, self.dr = (
            finite_number(rotation_kp, "rotation_kp"),
            finite_number(rotation_kd, "rotation_kd"),
        )
        self.rest = data.qpos[self.qadr].copy()
        self.limits = np.max(np.abs(model.jnt_actfrcrange[self.joints]), axis=1)
        self.hold = np.array(
            [
                i
                for i in range(model.nu)
                if i
                not in set(self.actuators)
                | {
                    model.actuator("mill_motor").id,
                    model.actuator("right_fingers_actuator").id,
                }
            ]
        )
        self.last_requested_fraction = 0.0
        self.hold_targets = {
            int(i): float(data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]])
            for i in self.hold
        }

    def apply(self, data, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Finite robot tool target required")
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jp, jr, self.site)
        rotation = data.site_xmat[self.site].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.T.ravel())
        error_r = np.empty(3)
        mujoco.mju_quat2Vel(error_r, quaternion, 1.0)
        force = self.kp * (target - data.site_xpos[self.site]) - self.kd * (
            jp @ data.qvel
        )
        torque = self.kr * error_r - self.dr * (jr @ data.qvel)
        required = data.qfrc_bias[self.dofs].copy()
        required[:7] += jp[:, self.dofs[:7]].T @ force + jr[:, self.dofs[:7]].T @ torque
        # 右臂关节阻抗保持安全位；左臂只作很弱的阻尼，不添加会扭曲 TCP 的姿态弹簧。
        required[:7] -= 0.5 * data.qvel[self.dofs[:7]]
        required[7:] += (
            100 * (self.rest[7:] - data.qpos[self.qadr[7:]])
            - 8 * data.qvel[self.dofs[7:]]
        )
        if not np.isfinite(required).all():
            raise ValueError("Robot requested torque overflow; commands unchanged")
        self.last_requested_fraction = float(np.max(np.abs(required) / self.limits))
        desired = np.clip(required, -self.limits, self.limits)
        bias = self.model.actuator_biasprm[self.actuators]
        gain = self.model.actuator_gainprm[self.actuators, 0]
        ctrl = (
            desired
            - bias[:, 0]
            - bias[:, 1] * data.actuator_length[self.actuators]
            - bias[:, 2] * data.actuator_velocity[self.actuators]
        ) / gain
        # 命令范围也是原资产能力的一部分，不能为重力补偿扩展 ctrlrange。
        ctrl = np.clip(
            ctrl,
            self.model.actuator_ctrlrange[self.actuators, 0],
            self.model.actuator_ctrlrange[self.actuators, 1],
        )
        data.ctrl[self.actuators] = ctrl
        for actuator in self.hold:
            joint = self.model.actuator_trnid[actuator, 0]
            dof = self.model.jnt_dofadr[joint]
            data.ctrl[actuator] = (
                self.hold_targets[int(actuator)]
                + data.qfrc_bias[dof] / self.model.actuator_gainprm[actuator, 0]
            )
        data.ctrl[self.model.actuator("right_fingers_actuator").id] = 0
        return {
            "requested_joint_limit_fraction": self.last_requested_fraction,
            "tip_error_m": float(np.linalg.norm(target - data.site_xpos[self.site])),
            "orientation_error_rad": float(np.linalg.norm(error_r)),
        }
