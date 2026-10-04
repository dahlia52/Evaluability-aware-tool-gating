import json
import unittest
from collections import Counter
from pathlib import Path

from precall_gating.conditions import CONDITIONS


ROOT = Path(__file__).resolve().parents[1]


class PaperArtifactTests(unittest.TestCase):
    def test_frozen_contract_inventory_matches_the_paper(self):
        records = json.loads(
            (ROOT / "artifacts" / "contracts" / "preconditions.json").read_text()
        )
        counts = Counter(record["kind"] for record in records)

        self.assertEqual(len(records), 112)
        self.assertEqual(counts, {"STATE": 33, "MIXED": 50, "ARG": 29})

    def test_frozen_effect_inventory_matches_the_paper(self):
        effects = json.loads(
            (ROOT / "artifacts" / "contracts" / "effects.json").read_text()
        )

        self.assertEqual(len(effects), 34)

    def test_all_eleven_paper_conditions_are_exposed(self):
        self.assertEqual(
            set(CONDITIONS),
            {"B0", "B1", "B2", "B3", "B3r", "M1a", "M1b", "M2", "M2b", "M3", "M3b"},
        )


if __name__ == "__main__":
    unittest.main()
