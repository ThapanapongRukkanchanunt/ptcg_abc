import unittest

from ptcg_abc.models import CardLine
from ptcg_abc.normalize import deck_fingerprint, normalize_card_name


class NormalizeTests(unittest.TestCase):
    def test_card_name_normalization_collapses_space_and_case(self):
        self.assertEqual(normalize_card_name("  Boss's   Orders "), "boss's orders")

    def test_card_name_normalization_matches_curly_apostrophes(self):
        self.assertEqual(normalize_card_name("Boss's Orders"), normalize_card_name("Boss’s Orders"))

    def test_card_name_normalization_repairs_mojibake(self):
        self.assertEqual(
            normalize_card_name("Biancaâ€™s Devotion"),
            normalize_card_name("Bianca’s Devotion"),
        )

    def test_card_name_normalization_maps_basic_energy_names(self):
        self.assertEqual(normalize_card_name("Fire Energy"), normalize_card_name("Basic {R} Energy"))
        self.assertEqual(
            normalize_card_name("Darkness Energy"),
            normalize_card_name("Basic {D} Energy"),
        )

    def test_card_name_normalization_maps_known_kaggle_aliases(self):
        self.assertEqual(
            normalize_card_name("Growing Grass Energy"),
            normalize_card_name("Grow Grass Energy"),
        )
        self.assertEqual(
            normalize_card_name("Telepathic Psychic Energy"),
            normalize_card_name("Telepath Psychic Energy"),
        )
        self.assertEqual(
            normalize_card_name("Rocky Fighting Energy"),
            normalize_card_name("Rock Fighting Energy"),
        )

    def test_deck_fingerprint_is_order_independent(self):
        first = [CardLine(count=4, name="Dreepy"), CardLine(count=2, name="Dragapult ex")]
        second = [CardLine(count=2, name="Dragapult ex"), CardLine(count=4, name="Dreepy")]
        self.assertEqual(deck_fingerprint(first), deck_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
