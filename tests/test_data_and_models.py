import json, unittest
from pathlib import Path
import torch
from chaintranslate.models import BasicTransformer


class IntegrityTests(unittest.TestCase):
    def test_disjoint_sources_and_files(self):
        data = Path("data")
        if not (data / "train.json").exists():
            self.skipTest("Prepare data")
        rows = {
            s: json.loads((data / f"{s}.json").read_text())
            for s in ["train", "validation", "test"]
        }
        for a, b in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            self.assertFalse(
                {r["source"].casefold() for r in rows[a]}
                & {r["source"].casefold() for r in rows[b]}
            )
            self.assertFalse(
                {r["file"] for r in rows[a]} & {r["file"] for r in rows[b]}
            )

    def test_causal_decoder(self):
        torch.manual_seed(42)
        m = BasicTransformer(50).eval()
        src = torch.tensor([[4, 5, 3]])
        a = torch.tensor([[2, 6, 7, 8]])
        b = a.clone()
        b[0, -1] = 9
        with torch.no_grad():
            x = m(src, a)
            y = m(src, b)
        self.assertTrue(torch.allclose(x[:, :-1], y[:, :-1], atol=1e-6))

    def test_same_test_ids(self):
        expected = [r["id"] for r in json.loads(Path("data/test.json").read_text())]
        for p in Path("results").glob("*_translations.json"):
            rows = json.loads(p.read_text())
            self.assertEqual([r["id"] for r in rows], expected)


if __name__ == "__main__":
    unittest.main()
