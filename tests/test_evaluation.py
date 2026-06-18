import unittest

from ptcg_abc.evaluation import (
    SampleDragapultBenchmarkRow,
    phase3_benchmark_deck_coverage,
    required_phase3_prepared_decks,
)


class EvaluationTests(unittest.TestCase):
    def test_phase3_benchmark_coverage_reports_missing_required_decks(self):
        rows = [
            SampleDragapultBenchmarkRow(
                deck_index=1,
                deck_label="Dragapult ex / None / Player 1st",
                archetype="Dragapult ex",
                games=10,
            ),
            SampleDragapultBenchmarkRow(
                deck_index=2,
                deck_label="Crustle / All / Player 1st",
                archetype="Crustle",
                games=10,
            ),
        ]

        coverage = phase3_benchmark_deck_coverage(rows)
        by_name = {row["required_deck"]: row for row in coverage}

        self.assertEqual(by_name["Crustle"]["status"], "covered")
        self.assertEqual(by_name["Crustle"]["deck_indices"], [2])
        self.assertEqual(by_name["Mega Lucario"]["status"], "missing")
        self.assertEqual(by_name["Mega Abomasnow"]["status"], "missing")
        self.assertEqual(by_name["Iono"]["status"], "missing")

    def test_required_phase3_prepared_decks_cover_named_targets(self):
        decks = required_phase3_prepared_decks(start_index=28)
        rows = [
            SampleDragapultBenchmarkRow(
                deck_index=deck.index,
                deck_label=deck.label,
                archetype=deck.archetype,
                games=10,
            )
            for deck in decks
        ]
        coverage = phase3_benchmark_deck_coverage(rows)

        self.assertEqual([len(deck.card_ids) for deck in decks], [60, 60, 60, 60])
        self.assertTrue(all(row["status"] == "covered" for row in coverage))
        self.assertEqual([deck.index for deck in decks], [28, 29, 30, 31])


if __name__ == "__main__":
    unittest.main()
