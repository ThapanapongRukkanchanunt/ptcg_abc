import csv
import json
import tempfile
import unittest
from pathlib import Path

from ptcg_abc.limitless import write_deck_collection
from ptcg_abc.models import Archetype, CardLine, CollectionResult, Decklist, TournamentResult, Variant
from ptcg_abc.normalize import deck_fingerprint


def sample_deck() -> Decklist:
    archetype = Archetype(
        rank=1,
        name="Dragapult ex",
        deck_id="284",
        points=123,
        share="10%",
        source_url="https://limitlesstcg.com/decks/284",
    )
    variant = Variant(
        name="None",
        value="0",
        source_url="https://limitlesstcg.com/decks/284/results?format=TEF-POR&variant=0",
    )
    result = TournamentResult(
        event_name="Example Event",
        event_date="17th June 2026",
        placement="1st",
        placement_rank=1,
        player="Test Player",
        decklist_url="https://limitlesstcg.com/decks/list/1",
        source_url="https://limitlesstcg.com/decks/284/results?format=TEF-POR&variant=0",
        page_order=0,
    )
    cards = [
        CardLine(count=4, name="Dreepy", section="Pokemon"),
        CardLine(count=4, name="Ultra Ball", section="Trainer"),
        CardLine(count=52, name="Fire Energy", section="Energy"),
    ]
    return Decklist(
        archetype=archetype,
        variant=variant,
        result=result,
        title="Dragapult ex",
        cards=cards,
        total_cards=60,
        fingerprint=deck_fingerprint(cards),
        source_url=result.decklist_url,
    )


class CorpusExportTests(unittest.TestCase):
    def test_write_deck_collection_exports_all_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collection = CollectionResult(decks=[sample_deck()])
            outputs = write_deck_collection(
                collection,
                snapshot_date="2026-06-17",
                output_root=Path(temp_dir),
            )

            self.assertTrue(outputs["jsonl"].exists())
            self.assertTrue(outputs["csv"].exists())
            self.assertTrue(outputs["manifest"].exists())
            deck_files = list(outputs["decks_dir"].glob("*.txt"))
            self.assertEqual(len(deck_files), 1)

            record = json.loads(outputs["jsonl"].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["total_cards"], 60)

            with outputs["csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["archetype"], "Dragapult ex")

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["deck_count"], 1)
            self.assertEqual(manifest["outputs"]["jsonl"], outputs["jsonl"].as_posix())


if __name__ == "__main__":
    unittest.main()
