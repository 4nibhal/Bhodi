from unittest import TestCase

from bhodi_platform.evaluation import load_fixture


class EvaluationFixtureTest(TestCase):
    def test_packaged_fixture_loads_with_expected_cases(self) -> None:
        fixture = load_fixture()

        self.assertEqual(fixture.name, "retrieval-grounding-baseline")
        self.assertEqual(
            tuple(case.query_id for case in fixture.retrieval_cases),
            ("corpus-policy", "conversation-recall"),
        )
        self.assertEqual(
            tuple(case.query_id for case in fixture.grounding_cases),
            ("corpus-policy", "conversation-recall"),
        )
