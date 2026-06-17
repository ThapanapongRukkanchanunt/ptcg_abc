import json
import tempfile
import unittest
from pathlib import Path

from ptcg_abc.card_db import load_card_id_lookup
from ptcg_abc.corpus import deck_to_card_ids, get_deck_by_index, load_deck_corpus


def corpus_record() -> dict:
    return {
        "archetype": {
            "rank": 1,
            "name": "Dragapult ex",
            "deck_id": "284",
            "points": 100,
            "share": "10%",
            "source_url": "https://limitlesstcg.com/decks/284",
        },
        "variant": {
            "name": "None",
            "value": "0",
            "source_url": "https://limitlesstcg.com/decks/284/results",
        },
        "result": {
            "event_name": "Example",
            "event_date": "17th June 2026",
            "placement": "1st",
            "placement_rank": 1,
            "player": "Player",
            "decklist_url": "https://limitlesstcg.com/decks/list/1",
            "source_url": "https://limitlesstcg.com/decks/284/results",
            "page_order": 0,
        },
        "title": "Dragapult",
        "cards": [
            {"count": 4, "name": "Dreepy", "section": "Pokemon"},
            {"count": 56, "name": "Fire Energy", "section": "Energy"},
        ],
        "total_cards": 60,
        "fingerprint": "abc",
        "source_url": "https://limitlesstcg.com/decks/list/1",
    }


class CorpusTests(unittest.TestCase):
    def test_load_deck_corpus_and_convert_to_card_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / "deck_corpus.jsonl"
            corpus_path.write_text(json.dumps(corpus_record()) + "\n", encoding="utf-8")
            cards_path = temp_path / "cards.csv"
            cards_path.write_text(
                "Card ID,Card Name\n11,Dreepy\n2,Basic {R} Energy\n",
                encoding="utf-8",
            )

            decks = load_deck_corpus(corpus_path)
            deck = get_deck_by_index(decks, 1)
            card_ids = deck_to_card_ids(deck, load_card_id_lookup(cards_path))

            self.assertEqual(deck.title, "Dragapult")
            self.assertEqual(card_ids[:4], [11, 11, 11, 11])
            self.assertEqual(card_ids[4:], [2] * 56)

    def test_get_deck_by_index_is_one_based(self):
        with self.assertRaises(IndexError):
            get_deck_by_index([], 1)


if __name__ == "__main__":
    unittest.main()
