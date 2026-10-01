"""Invariant tests for the reduced model, not empirical fluid validation."""
import unittest
import numpy as np
from liquid import WaterTransfer

class WaterTests(unittest.TestCase):
    def test_upright_does_not_pour(self):
        w=WaterTransfer()
        for _ in range(100):w.step(.1,np.array([0,0,1]),np.eye(3),np.array([0,0,.8]),np.array([0,0,.6]))
        self.assertEqual(w.source_ml,200);self.assertEqual(w.spilled_ml,0)

    def test_inverted_caught_and_drained(self):
        w=WaterTransfer();R=np.diag([1.,-1.,-1.])
        for _ in range(600):w.step(.1,np.array([0,0,1]),R,np.array([0,0,.8]),np.array([0,0,.6]))
        self.assertLess(w.source_ml,.001);self.assertGreater(w.receiver_ml,199.99);self.assertLess(w.spilled_ml,1e-8)

    def test_miss_and_overflow_conserve(self):
        for xy in (0.,1.):
            w=WaterTransfer(filter_capacity_ml=5,receiver_capacity_ml=8)
            for _ in range(600):w.step(.1,np.array([xy,0,1]),np.diag([1.,-1.,-1.]),np.array([0,0,.8]),np.array([0,0,.6]))
            self.assertAlmostEqual(w.source_ml+w.filter_ml+w.receiver_ml+w.spilled_ml,200)
            self.assertLessEqual(w.receiver_ml,8)
            self.assertGreater(w.spilled_ml,190)

if __name__=='__main__':unittest.main()
