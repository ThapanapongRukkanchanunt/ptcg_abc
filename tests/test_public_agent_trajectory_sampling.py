from __future__ import annotations

import unittest

from ptcg_abc.rl.public_opponents import _sample_trajectory_record_indices


class PublicAgentTrajectorySamplingTests(unittest.TestCase):
    def test_zero_samples_retains_every_record(self) -> None:
        self.assertEqual(
            _sample_trajectory_record_indices(
                5,
                samples_per_game=0,
                seed=17,
            ),
            [0, 1, 2, 3, 4],
        )

    def test_sampling_is_bounded_sorted_and_deterministic(self) -> None:
        first = _sample_trajectory_record_indices(
            20,
            samples_per_game=3,
            seed=42,
        )
        second = _sample_trajectory_record_indices(
            20,
            samples_per_game=3,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_sampling_more_than_available_retains_every_record(self) -> None:
        self.assertEqual(
            _sample_trajectory_record_indices(
                2,
                samples_per_game=5,
                seed=9,
            ),
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
