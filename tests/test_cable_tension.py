from __future__ import annotations

import gymnasium.utils.env_checker
import numpy as np

import ibero
from ibero.demo import scripted_action


def test_reset_is_seed_deterministic() -> None:
    env = ibero.make("ibero/CableTension-v0")
    try:
        first, _ = env.reset(seed=11)
        second, _ = env.reset(seed=11)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], atol=1e-6)
    finally:
        env.close()


def test_step_has_valid_spaces_and_audit() -> None:
    env = ibero.make("ibero/CableTension-v0")
    try:
        _, _ = env.reset(seed=2)
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert set(info["audit"]) == {
            "cable_tension_n",
            "damage_flags",
            "self_collision_flags",
            "in_band_steps",
        }
    finally:
        env.close()


def test_scripted_controller_reaches_safe_force_band() -> None:
    env = ibero.make("ibero/CableTension-v0")
    try:
        observation, _ = env.reset(seed=7)
        for _ in range(100):
            observation, _, terminated, truncated, info = env.step(
                scripted_action(observation)
            )
            if terminated or truncated:
                break
        assert info["is_success"]
        assert not info["audit"]["damage_flags"]["cable_over_tension"]
        assert not info["audit"]["self_collision_flags"]["left_right"]
    finally:
        env.close()


def test_gymnasium_checker() -> None:
    env = ibero.make("ibero/CableTension-v0")
    try:
        gymnasium.utils.env_checker.check_env(env, skip_render_check=True)
    finally:
        env.close()
