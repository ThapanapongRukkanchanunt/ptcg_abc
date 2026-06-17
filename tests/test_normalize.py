import unittest

from ptcg_abc.models import CardLine
from ptcg_abc.normalize import deck_fingerprint, normalize_card_name


class NormalizeTests(unittest.TestCase):
    def test_card_name_normalization_collapses_space_and_case(self):
        self.assertEqual(normalize_card_name("  Boss's   Orders "), "boss's orders")

    def test_deck_fingerprint_is_order_independent(self):
        first = [CardLine(count=4, name="Dreepy"), CardLine(count=2, name="Dragapult ex")]
        second = [CardLine(count=2, name="Dragapult ex"), CardLine(count=4, name="Dreepy")]
        self.assertEqual(deck_fingerprint(first), deck_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
