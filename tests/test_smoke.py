"""Smoke tests that need no dataset.

The corpora are not redistributed, so these check what can be checked without
them: the Appendix propositions verify numerically, the referenced STLAT model
builds and runs, seeding is reproducible, and the corpus paths can be
configured from the environment.

    python -m unittest discover -s tests -v
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pretrain_finetune as pf  # noqa: E402
import theory  # noqa: E402
from model import ReferencedSTLAT  # noqa: E402


class TestPropositions(unittest.TestCase):
    """Every quantitative claim in the Appendix, verified by brute force."""

    def test_proposition_1_closed_form(self):
        self.assertTrue(theory.verify_prop1(verbose=False))

    def test_proposition_1_maximum(self):
        self.assertTrue(theory.verify_prop1_maximum(verbose=False))

    def test_proposition_2_symmetry(self):
        self.assertTrue(theory.verify_prop2(verbose=False))

    def test_proposition_2_band_certificate(self):
        self.assertTrue(theory.verify_prop2_no_fixed_point(verbose=False))

    def test_proposition_3_coverage(self):
        self.assertTrue(theory.verify_prop3(verbose=False))


class TestReferencedSTLAT(unittest.TestCase):

    def _model(self, **kw):
        return ReferencedSTLAT(input_size=16, hidden_size=64, num_classes=5, **kw)

    def test_forward_shape(self):
        model = self._model().eval()
        logits, diag = model(torch.randn(2, 4, 16))
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertIsInstance(diag, dict)

    def test_asfg_forward_shape(self):
        model = self._model(use_asfg=True).eval()
        logits, _ = model(torch.randn(3, 6, 16))
        self.assertEqual(tuple(logits.shape), (3, 5))

    def test_output_is_finite(self):
        model = self._model(use_asfg=True).eval()
        logits, _ = model(torch.randn(2, 4, 16))
        self.assertTrue(torch.isfinite(logits).all())


class TestReproducibility(unittest.TestCase):

    def _run(self, seed):
        pf.seed_all(seed)
        model = ReferencedSTLAT(input_size=16, hidden_size=64, num_classes=5).eval()
        x = torch.randn(2, 4, 16)
        with torch.no_grad():
            return model(x)[0].numpy()

    def test_same_seed_same_output(self):
        np.testing.assert_array_equal(self._run(42), self._run(42))

    def test_different_seed_different_output(self):
        self.assertFalse(np.array_equal(self._run(42), self._run(123)))


class TestConfiguration(unittest.TestCase):

    def test_csl_paths_follow_environment(self):
        env = dict(os.environ, CSL_TRAIN="/data/csl/train", CSL_TEST="/data/csl/test")
        out = subprocess.run(
            [sys.executable, "-c",
             "import train; print(train.CSL_TRAIN); print(train.CSL_TEST)"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["/data/csl/train", "/data/csl/test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
