import sys
import types
import unittest


checker = types.ModuleType("multi_turn_checker")
checker.multi_turn_checker = lambda *args, **kwargs: {"valid": True}
sys.modules.setdefault("bfcl_eval", types.ModuleType("bfcl_eval"))
sys.modules.setdefault("bfcl_eval.eval_checker", types.ModuleType("eval_checker"))
sys.modules.setdefault(
    "bfcl_eval.eval_checker.multi_turn_eval", types.ModuleType("multi_turn_eval")
)
sys.modules.setdefault(
    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker", checker
)

from precall_gating.score import gating_metrics  # noqa: E402


class ScoreTests(unittest.TestCase):
    def test_execution_error_rate_excludes_rejected_calls(self):
        records = [
            {
                "id": "sample",
                "events": [
                    {
                        "function": "blocked",
                        "label": "CONDITIONAL",
                        "pre_violation": True,
                        "violation_kind": "MIXED",
                        "outcome": "gate_blocked",
                        "enabling_tools": [],
                    },
                    {
                        "function": "failed",
                        "label": "AVAILABLE",
                        "pre_violation": False,
                        "outcome": "executed_error",
                        "enabling_tools": [],
                    },
                    {
                        "function": "succeeded",
                        "label": "AVAILABLE",
                        "pre_violation": False,
                        "outcome": "executed_ok",
                        "enabling_tools": [],
                    },
                ],
                "turn_stats": [],
                "force_quit": False,
            }
        ]

        metrics = gating_metrics(records, {"sample": []})

        self.assertEqual(metrics["n_calls"], 3)
        self.assertEqual(metrics["n_executed_calls"], 2)
        self.assertEqual(metrics["execution_error_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
