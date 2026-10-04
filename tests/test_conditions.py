import json
import unittest
from types import SimpleNamespace

from precall_gating.conditions import CONDITIONS, ExposureController
from precall_gating.contracts import ContractStore


class ExposureControllerTests(unittest.TestCase):
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
            effects={"VehicleControlAPI.prepare": ["ready", "level"]},
        )
        self.envs = {
            "VehicleControlAPI": SimpleNamespace(ready=False, level=4)
        }
        self.docs = [
            {
                "_api_class": "VehicleControlAPI",
                "name": "start",
                "description": "Start.",
                "parameters": {"type": "dict", "properties": {}},
            },
            {
                "_api_class": "VehicleControlAPI",
                "name": "fill",
                "description": "Fill.",
                "parameters": {
                    "type": "dict",
                    "properties": {"amount": {"type": "float"}},
                },
            },
        ]

    def test_triage_labels_blocked_and_conditional_tools(self):
        tools, labels = ExposureController(self.store, CONDITIONS["M1a"]).build_tools(
            self.docs, self.envs
        )

        self.assertEqual(labels["BLOCKED"], ["start"])
        self.assertEqual(labels["CONDITIONAL"], ["fill"])
        self.assertEqual(len(tools), 2)
        self.assertIn("[UNAVAILABLE NOW]", tools[0]["function"]["description"])
        self.assertIn("[CONDITIONAL]", tools[1]["function"]["description"])
        self.assertEqual(tools[1]["function"]["parameters"]["type"], "object")
        self.assertEqual(
            tools[1]["function"]["parameters"]["properties"]["amount"]["type"],
            "number",
        )

    def test_remove_mode_omits_blocked_tools(self):
        tools, labels = ExposureController(self.store, CONDITIONS["M1b"]).build_tools(
            self.docs, self.envs
        )

        self.assertEqual(labels["BLOCKED"], ["start"])
        self.assertEqual([tool["function"]["name"] for tool in tools], ["fill"])

    def test_gate_returns_structured_repair_signal(self):
        controller = ExposureController(self.store, CONDITIONS["M3"])
        violation = controller.gate(
            "VehicleControlAPI", "fill", self.envs["VehicleControlAPI"], {"amount": 7}
        )
        payload = json.loads(controller.gate_message(violation))

        self.assertEqual(payload["status"], "blocked_before_execution")
        self.assertEqual(payload["resolved_by"], ["prepare"])


if __name__ == "__main__":
    unittest.main()
