"""P2 实体主轴工具：安装、惯量、限矩转子和腕部 F/T 传感器。"""

import mujoco
from ibero.materials.parameters import finite_number
from ibero.processes.tools import quaternion_matrix


def add_spindle(
    spec,
    parent,
    tool,
    *,
    prefix="mill_",
    rotor_mass_kg=0.02,
    housing_mass_kg=0.08,
    torque_limit_nm=0.15,
    ft_quaternion=(1, 0, 0, 0),
):
    """可复用工具：端面/侧刃圆柱与非切削刀柄分开，转速来自实际转子关节。"""
    quaternion_matrix(ft_quaternion)
    for name, value in (
        ("rotor_mass_kg", rotor_mass_kg),
        ("housing_mass_kg", housing_mass_kg),
        ("torque_limit_nm", torque_limit_nm),
    ):
        finite_number(value, name)
    mount = parent.add_body(name=prefix + "mount")
    mount.add_geom(
        name=prefix + "housing",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[tool.shank_radius_m, tool.shank_length_m / 2],
        pos=[0, 0, tool.cutting_length_m + tool.shank_length_m / 2],
        mass=housing_mass_kg,
        rgba=[0.3, 0.35, 0.4, 1],
    )
    mount.add_site(
        name=prefix + "ft_site",
        pos=[0, 0, tool.cutting_length_m + tool.shank_length_m],
        size=[0.001],
        quat=ft_quaternion,
    )
    mount.add_site(name=prefix + "tip", pos=[0, 0, 0], size=[0.001])
    rotor = mount.add_body(name=prefix + "rotor")
    rotor.add_joint(
        name=prefix + "spindle",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        limited=False,
        # 工具转子不是 G1 手臂电机，不能继承父级 g1 的 0.3 N·m 摩擦和 0.01 附加惯量。
        # 真实惯量来自已声明的转子几何/质量；此原型未标定轴承损耗。
        frictionloss=0,
        armature=0,
        damping=0,
    )
    rotor.add_geom(
        name=prefix + "blade",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, tool.cutting_length_m / 2],
        size=[tool.radius_m, tool.cutting_length_m / 2],
        mass=rotor_mass_kg,
        contype=1,
        conaffinity=1,
        rgba=[0.8, 0.8, 0.85, 1],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    # 转子有真实惯量与限矩速度伺服；绘制转动标记不代替 qvel 转速。
    rotor.add_site(
        name=prefix + "rotation_mark",
        pos=[tool.radius_m, 0, tool.cutting_length_m],
        size=[0.0006],
        rgba=[1, 0.2, 0.1, 1],
    )
    actuator = spec.add_actuator(
        name=prefix + "motor",
        target=prefix + "spindle",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    actuator.set_to_velocity(kv=0.002)
    actuator.forcelimited = True
    actuator.forcerange = [-torque_limit_nm, torque_limit_nm]
    for name, sensor in (
        ("force", mujoco.mjtSensor.mjSENS_FORCE),
        ("torque", mujoco.mjtSensor.mjSENS_TORQUE),
    ):
        spec.add_sensor(
            name=prefix + name,
            type=sensor,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=prefix + "ft_site",
        )
