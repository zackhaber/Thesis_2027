# This file will
import unittest

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.env import Action
from env.game_modes import GameMode


class EnvironmentTests(unittest.TestCase):
    def test_seeded_collision_replay_is_identical(self):
        scenario = generate_level(DifficultySpec(level=2, seed=11), GameMode.MAZE)
        first = PursuitEvasionEnv(scenario, GameMode.MAZE, seed=23)
        second = PursuitEvasionEnv(scenario, GameMode.MAZE, seed=23)
        first.reset()
        second.reset()
        actions: tuple[list[Action], list[Action]] = (
            ["right"] * len(scenario.pursuers),
            ["left"] * len(scenario.evaders),
        )
        self.assertEqual(first.step(*actions), second.step(*actions))

    def test_horizon_is_truncation(self):
        scenario = generate_level(DifficultySpec(level=1, seed=4), GameMode.PURE_PURSUIT)
        environment = PursuitEvasionEnv(scenario, max_steps=1)
        environment.reset()
        pursuer_actions: list[Action] = ["stay"] * len(environment.state.pursuers)
        evader_actions: list[Action] = ["stay"] * len(environment.state.evaders)
        _, _, terminated, truncated, info = environment.step(pursuer_actions, evader_actions)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["truncated"], truncated)

    def test_observation_masks_and_global_state(self):
        environment = PursuitEvasionEnv(generate_level(DifficultySpec(level=1, seed=5), GameMode.MAZE), GameMode.MAZE)
        observation = environment.reset()
        self.assertEqual(len(observation["global_state"]), environment.scenario.rows * environment.scenario.cols * 8)
        self.assertEqual(len(observation["action_masks"]["pursuers"]), len(environment.state.pursuers))
        self.assertTrue(all(mask[0] for mask in observation["action_masks"]["pursuers"]))


if __name__ == "__main__":
    unittest.main()