"""Guard against silently mixing predictions, references and reported scores."""

import copy
import json
import unittest
from pathlib import Path
from chaintranslate.report import validate_run


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.test = json.loads(Path("data/test.json").read_text())
        self.score = json.loads(Path("results/basic_metrics.json").read_text())
        self.predictions = json.loads(
            Path("results/basic_translations.json").read_text()
        )
        self.samples = json.loads(Path("results/basic_temperatures.json").read_text())

    def test_measured_run_validates(self):
        validate_run("basic", self.score, self.predictions, self.samples, self.test)

    def test_changed_reference_rejected(self):
        changed = copy.deepcopy(self.predictions)
        changed[0]["reference"] = "Une autre référence."
        with self.assertRaises(ValueError):
            validate_run("basic", self.score, changed, self.samples, self.test)

    def test_unmeasured_score_rejected(self):
        changed = dict(self.score, bleu=self.score["bleu"] + 1.0)
        with self.assertRaises(ValueError):
            validate_run("basic", changed, self.predictions, self.samples, self.test)

    def test_missing_temperature_rejected(self):
        with self.assertRaises(ValueError):
            validate_run(
                "basic", self.score, self.predictions, self.samples[:-1], self.test
            )
