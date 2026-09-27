"""Общий инструмент потоков: версии _wNN без перезаписи и журнал самопроверок."""
import json
import os
import tempfile
import unittest

from tools.flowkit import Checks, save_state, versioned


class TestVersioned(unittest.TestCase):
    def test_versions_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            a = versioned(d, "res", "json")
            open(a, "w").close()
            b = versioned(d, "res", "json")
            self.assertTrue(a.endswith("res_w01.json"))
            self.assertTrue(b.endswith("res_w02.json"))

    def test_other_stems_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "other_w07.json"), "w").close()
            self.assertTrue(versioned(d, "res", "json").endswith("res_w01.json"))


class TestChecks(unittest.TestCase):
    def test_failed_check_sets_exit_code(self):
        c = Checks("t")
        c.eq("сходится", 2 + 2, 4)
        self.assertEqual(c.exit_code(), 0)
        c.close("не сходится", 1.0, 1.1, tol=1e-3)
        self.assertEqual(c.exit_code(), 1)
        self.assertIn("не сходится", c.report())

    def test_nan_never_passes(self):
        self.assertFalse(Checks("t").close("nan", float("nan"), float("nan")))

    def test_state_has_params_checks_results(self):
        with tempfile.TemporaryDirectory() as d:
            c = Checks("t")
            c.ok("да", True)
            p = save_state(os.path.join(d, "x_w01.json"), params={"L": 2.25},
                           results={"r": 1}, checks=c)
            doc = json.load(open(p, encoding="utf-8"))
            self.assertEqual(doc["params"]["L"], 2.25)
            self.assertEqual(doc["checks"]["failed"], 0)
            self.assertEqual(doc["results"]["r"], 1)
            self.assertIn("git", doc)


if __name__ == "__main__":
    unittest.main()
