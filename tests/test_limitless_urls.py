import unittest

from ptcg_abc.limitless import with_query_params


class LimitlessUrlTests(unittest.TestCase):
    def test_with_query_params_preserves_existing_variant(self):
        url = with_query_params(
            "https://limitlesstcg.com/decks/284/results?variant=3",
            format="TEF-POR",
        )

        self.assertEqual(
            url,
            "https://limitlesstcg.com/decks/284/results?variant=3&format=TEF-POR",
        )


if __name__ == "__main__":
    unittest.main()
