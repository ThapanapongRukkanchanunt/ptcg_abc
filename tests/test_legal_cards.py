import tempfile
import unittest
from pathlib import Path

from ptcg_abc.legal_cards import choose_legal_card_candidate


class LegalCardsTests(unittest.TestCase):
    def test_uses_card_name_and_legal_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legal_cards.csv"
            path.write_text(
                "Card Name,legal\nPikachu,true\nCharizard,false\nDreepy,true\n",
                encoding="utf-8",
            )

            candidate = choose_legal_card_candidate(Path(temp_dir))

        self.assertEqual(candidate.names, {"Pikachu", "Dreepy"})
        self.assertEqual(candidate.source_column, "Card Name")
        self.assertEqual(candidate.legal_column, "legal")


if __name__ == "__main__":
    unittest.main()
