import unittest
from types import SimpleNamespace

from precall_gating.contracts import ContractStore, Precondition


class PreconditionTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(level=4, ready=False)

    def test_state_condition_is_evaluable_before_call(self):
        condition = Precondition(
            api_class="VehicleControlAPI",
            function="start",
            kind="STATE",
            state_vars=["ready"],
            arg_vars=[],
            condition="not self.ready",
            error_message="not ready",
        )

        self.assertTrue(condition.decidable_precall)
        self.assertTrue(condition.violated_precall(self.env))
        self.assertTrue(condition.violated_with_args(self.env, {}))

    def test_mixed_condition_waits_for_bound_arguments(self):
        condition = Precondition(
            api_class="VehicleControlAPI",
            function="fill",
            kind="MIXED",
            state_vars=["level"],
            arg_vars=["amount"],
            condition="self.level + amount > 10",
            error_message="capacity exceeded",
        )

        self.assertFalse(condition.decidable_precall)
        self.assertFalse(condition.violated_precall(self.env))
        self.assertFalse(condition.violated_with_args(self.env, {"amount": 6}))
        self.assertTrue(condition.violated_with_args(self.env, {"amount": 7}))

    def test_arg_only_condition_is_not_part_of_the_state_gate(self):
        condition = Precondition(
            api_class="VehicleControlAPI",
            function="fill",
            kind="ARG",
            state_vars=[],
            arg_vars=["amount"],
            condition="amount < 0",
            error_message="negative amount",
        )

        self.assertFalse(condition.violated_with_args(self.env, {"amount": -1}))


class ContractStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ContractStore(
            records=[
                {
                    "api_class": "VehicleControlAPI",
                    "function": "start",
                    "kind": "STATE",
                    "state_vars": ["ready"],
                    "arg_vars": [],
                    "condition": "not self.ready",
                    "error_message": "not ready",
                },
                {
                    "api_class": "VehicleControlAPI",
                    "function": "fill",
                    "kind": "MIXED",
                    "state_vars": ["level"],
                    "arg_vars": ["amount"],
                    "condition": "self.level + amount > 10",
                    "error_message": "capacity exceeded",
                },
            ],
            effects={"VehicleControlAPI.prepare": ["ready"]},
        )
        self.env = SimpleNamespace(level=4, ready=False)

    def test_store_classifies_and_gates_calls(self):
        self.assertIsNotNone(
            self.store.blocked_precall("VehicleControlAPI", "start", self.env)
        )
        self.assertTrue(self.store.is_conditional("VehicleControlAPI", "fill"))
        self.assertIsNone(
            self.store.gate("VehicleControlAPI", "fill", self.env, {"amount": 6})
        )
        self.assertIsNotNone(
            self.store.gate("VehicleControlAPI", "fill", self.env, {"amount": 7})
        )

    def test_effect_index_proposes_enabling_tool(self):
        precondition = self.store.by_func[("VehicleControlAPI", "start")][0]
        self.assertEqual(self.store.repair(precondition), ["prepare"])


if __name__ == "__main__":
    unittest.main()
