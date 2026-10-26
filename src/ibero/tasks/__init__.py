"""可复用的任务语义；不修改物理模型或控制器状态。"""

from ibero.tasks.cable_routing import CableRoutingTask, RoutingGoal

__all__ = ["CableRoutingTask", "RoutingGoal"]
