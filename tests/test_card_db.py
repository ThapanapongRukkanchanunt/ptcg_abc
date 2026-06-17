import tempfile
import unittest
from pathlib import Path

from ptcg_abc.card_db import load_card_id_lookup


class CardDbTests(unittest.TestCase):
    def test_load_card_id_lookup_prefers_lowest_id_for_ambiguous_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cards.csv"
            path.write_text(
                "\n".join(
                    [
                        "Card ID,Card Name",
                        "42,Boss's Orders",
                        "7,Boss's Orders",
                        "5,Basic {R} Energy",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            lookup = load_card_id_lookup(path)

            self.assertEqual(lookup.preferred_id("Boss's Orders"), 7)
            self.assertEqual(lookup.preferred_id("Fire Energy"), 5)
            self.assertEqual(lookup.ambiguous_names(["Boss's Orders"]), {"Boss's Orders": (7, 42)})

    def test_missing_names_reports_unresolved_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cards.csv"
            path.write_text("Card ID,Card Name\n1,Dreepy\n", encoding="utf-8")

            lookup = load_card_id_lookup(path)

            self.assertEqual(lookup.missing_names(["Dreepy", "Unknown Card"]), ["Unknown Card"])


if __name__ == "__main__":
    unittest.main()
