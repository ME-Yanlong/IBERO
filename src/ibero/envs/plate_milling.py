"""G1 真实执行器主轴加工工作站，复用已验证的材料—载荷—积分事务。"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import mujoco
import numpy as np
from ibero.core.milling_runtime import MillingRuntime
from ibero.core.scene_loader import SceneLoader
from ibero.core.task_loading import load_task_spec
from ibero.core.task_api import TaskState
from ibero.core.audit import AuditSnapshot
from ibero.control.milling import MillingArmServo
from ibero.robots.g1_industrial import solve_reset_pose
from ibero.robots.g1_upperbody import ARM_JOINTS
from ibero.robots.g1_upperbody_2f85 import vendored_robot_asset_hash

DEFAULT_SCENE = Path(__file__).resolve().parents[3] / "scenes/plate_milling"


class PlateMillingEnv(MillingRuntime):
    scene_kind = "plate_milling"

    def __init__(self, scene_path=DEFAULT_SCENE, *, shape="through_hole"):
        self.scene = SceneLoader().validate(scene_path)
        cfg = self.scene.config
        if cfg["kind"] != self.scene_kind:
            raise ValueError("Expected plate_milling recipe")
        self.task = load_task_spec(self.scene)
        self.target = self.task.make_target(shape)
        from ibero.core.scene_compiler import SceneCompiler

        self._initialize_runtime(SceneCompiler().compile(self.scene))
        self.stock_geom_ids = frozenset(int(i) for i in self.binding.geom_ids)
        r = cfg["robot"]
        self.servo_stride = round(1 / r["servo_hz"] / self.model.opt.timestep)
        self.inspect_target(self.target)  # 编译后的实体范围必须能容纳声明目标。
        self.reset(seed=0)

    @classmethod
    def from_scene(cls, root, *, shape="through_hole"):
        return cls(root, shape=shape)

    def reset(self, *, seed=0):
        super().reset(seed=seed)
        cfg = self.scene.config
        for name, value in zip(ARM_JOINTS, cfg["robot"]["arm_qpos"]):
            if (
                not self.model.joint(name).range[0]
                <= value
                <= self.model.joint(name).range[1]
            ):
                raise ValueError("Initial G1 joints outside original asset limits")
            self.data.joint(name).qpos[0] = value
        solve_reset_pose(
            self.model, self.data, "left", "mill_tip", self.start, np.eye(3)
        )
        jitter = (
            np.random.default_rng(seed).uniform(-1, 1, 14)
            * cfg["initialization"]["joint_jitter_rad"]
        )
        for name, value in zip(ARM_JOINTS, jitter):
            self.data.joint(name).qpos[0] += value
            if (
                not self.model.joint(name).range[0]
                <= self.data.joint(name).qpos[0]
                <= self.model.joint(name).range[1]
            ):
                raise ValueError("Seed jitter would exceed original joint range")
        mujoco.mj_forward(self.model, self.data)
        r = cfg["robot"]
        self.servo = MillingArmServo(
            self.model,
            self.data,
            position_kp=r["position_kp_n_m"],
            position_kd=r["position_kd_ns_m"],
            rotation_kp=r["rotation_kp_nm_rad"],
            rotation_kd=r["rotation_kd_nms_rad"],
        )
        self.motion_target = self.start.copy()
        self.physics_steps = 0
        self.robot_audit = dict(
            peak_requested_joint_limit_fraction=0.0,
            peak_actual_joint_limit_fraction=0.0,
            peak_orientation_error_rad=0.0,
            peak_forbidden_penetration_m=0.0,
        )
        self.servo.apply(self.data, self.motion_target)
        self.task.reset()
        self.last_info.update(
            tip_position_m=self.data.site("mill_tip").xpos.copy().tolist(),
            invalid_reason=None,
        )

    def manifest(self):
        result = super().manifest()
        result["asset_hash"] = vendored_robot_asset_hash()
        result["scene_kind"] = self.scene_kind
        result["target"] = asdict(self.target)
        result["scene_hash"] = hashlib.sha256(
            json.dumps(
                {
                    "physical_recipe": result["scene_hash"],
                    "selected_target": result["target"],
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        return result

    def _set_motion_command(self, target):
        self.motion_target = target.copy()

    def _before_physics_step(self):
        if self.physics_steps % self.servo_stride == 0:
            metrics = self.servo.apply(self.data, self.motion_target)
            self.robot_audit["peak_requested_joint_limit_fraction"] = max(
                self.robot_audit["peak_requested_joint_limit_fraction"],
                metrics["requested_joint_limit_fraction"],
            )
        self.physics_steps += 1

    def _after_physics_step(self):
        a = self.robot_audit
        actual = float(
            np.max(np.abs(self.data.qfrc_actuator[self.servo.dofs]) / self.servo.limits)
        )
        for joint, adr in zip(self.servo.joints, self.servo.qadr):
            # 位置约束软接触的数值容差单列为 0.1 mrad；不裁剪运行中的 qpos。
            if (
                not self.model.jnt_range[joint, 0] - 0.0001
                <= self.data.qpos[adr]
                <= self.model.jnt_range[joint, 1] + 0.0001
            ):
                return "original_joint_position_limit_violated"
        a["peak_actual_joint_limit_fraction"] = max(
            a["peak_actual_joint_limit_fraction"], actual
        )
        rotation = self.data.site("mill_tip").xmat.reshape(3, 3)
        angle = float(
            np.arccos(np.clip(rotation[:, 2] @ self.stock.rotation[:, 2], -1, 1))
        )
        a["peak_orientation_error_rad"] = max(a["peak_orientation_error_rad"], angle)
        for contact in self.data.contact:
            blade = self.process.blade_geom
            working_pair = (
                contact.geom1 == blade and contact.geom2 in self.stock_geom_ids
            ) or (contact.geom2 == blade and contact.geom1 in self.stock_geom_ids)
            # 只豁免工作刃—毛坯正常接触。刃区撞支架、机器人或地面仍必须审计。
            if not working_pair:
                a["peak_forbidden_penetration_m"] = max(
                    a["peak_forbidden_penetration_m"], -float(contact.dist)
                )
        if (
            a["peak_forbidden_penetration_m"]
            > self.scene.constraints["safety"]["max_shank_penetration_m"]
        ):
            return "robot_or_fixture_contact_limit"
        if actual > 1 + 1e-8:
            return "original_joint_torque_limit_violated"
        if self.servo.last_requested_fraction > 1:
            return "robot_servo_torque_saturation"
        if angle > self.scene.config["robot"]["axis_tolerance_rad"]:
            return "robot_axis_tracking_limit"
        if (
            np.linalg.norm(self.data.site("mill_tip").xpos - self.motion_target)
            > self.scene.constraints["safety"]["max_tracking_error_m"]
        ):
            return "tracking_error_limit"
        return None

    def step(self, target, rpm, *, substeps=100, fault_at=None):
        info = super().step(target, rpm, substeps=substeps, fault_at=fault_at)
        info.update(self.robot_audit)
        return info

    def evaluate_task(self, target):
        """控制器之外检查实际孔槽；只读快照传给 P4，不把模型交给菜谱修改。"""
        if target != self.target:
            raise ValueError("Task evaluation must use the declared P4 target")
        shape = self.inspect_target(target)
        tip = self.stock.world_to_local(self.data.site("mill_tip").xpos)
        rpm = abs(self.process.kinematics(self.data)[3])
        extra = dict(
            shape_passed=shape["passed"],
            tool_retracted=bool(
                tip[2] > self.stock.params.size_m[2] / 2 + self.stock.cell_size_m
            ),
            spindle_stopped=bool(rpm < 1),
            invalid_reason=self.process.invalid_reason,
        )
        state = TaskState(
            {},
            {},
            {},
            {},
            frozenset(),
            0.0,
            self.physics_steps,
            1,
            extra=MappingProxyType(extra),
        )
        return dict(
            result=asdict(
                self.task.evaluate(state, AuditSnapshot(False, False, 0.0, 0.0, ()))
            ),
            shape_check=shape,
            measurements=extra,
        )
