"""三轴加工台架：只负责台架构造和三轴命令，物理事务由 Core 运行时拥有。"""

from ibero.core.milling_builder import build_milling_fixture
from ibero.core.milling_runtime import MillingRuntime

# 保留已发布的 Python 导入路径；工具实现与机器人均不再依赖 benches。
from ibero.robots.spindle import add_spindle as add_spindle


class MillingFixture(MillingRuntime):
    """纯 Python 台架参数保持兼容；正式 P4 菜谱通过 SceneCompiler 构建。"""

    def __init__(
        self,
        stock_parameters,
        cell_size_m,
        tool,
        coefficients,
        limits,
        *,
        timestep=0.0002,
        start=(-0.025, 0, 0.025),
        max_cells=60000,
        angular_samples=128,
        ft_quaternion=(1, 0, 0, 0),
        edge_transition_chip_m=1e-8,
        axis_stiffness_n_m=5000.0,
        axis_damping_ns_m=80.0,
        axis_force_limit_n=50.0,
    ):
        compiled = build_milling_fixture(
            stock_parameters,
            cell_size_m,
            tool,
            coefficients,
            limits,
            timestep=timestep,
            start=start,
            max_cells=max_cells,
            angular_samples=angular_samples,
            ft_quaternion=ft_quaternion,
            edge_transition_chip_m=edge_transition_chip_m,
            axis_stiffness_n_m=axis_stiffness_n_m,
            axis_damping_ns_m=axis_damping_ns_m,
            axis_force_limit_n=axis_force_limit_n,
        )
        self._initialize_runtime(compiled)
        self.reset(seed=0)

    @classmethod
    def from_scene(cls, root):
        from ibero.core.scene_loader import SceneLoader
        from ibero.core.scene_compiler import SceneCompiler

        scene = SceneLoader().validate(root)
        if scene.config["kind"] != "milling_bench":
            raise ValueError("Expected milling_bench recipe")
        compiled = SceneCompiler().compile(scene)
        # 已有编译产物直接接管，避免构造函数重新编译并重复 reset。
        env = cls.__new__(cls)
        env._initialize_runtime(compiled)
        env.reset(seed=0)
        return env

    def _set_motion_command(self, target):
        """台架只写三轴执行器；机器人复用过程步进而提供自己的执行器映射。"""
        for i in range(3):
            self.data.ctrl[self.model.actuator(f"mill_axis_{i}").id] = (
                target[i] - self.start[i]
            )
