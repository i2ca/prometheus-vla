"""Numerical model checks, not validation against real pouring experiments."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from kettle_liquid import KettleWater

class LiquidChecks(unittest.TestCase):
    def poses(self, angle, miss=False):
        r=Rotation.from_euler('y',-angle,degrees=True).as_matrix();spout=np.array([.4,0,1.]);w=KettleWater();p=spout-r@w.spout_local
        return p,r,np.array([.4+(.2 if miss else 0),0,.75]),np.eye(3),np.array([.4,0,.76]),np.eye(3)
    def test_no_flow_upright(self):
        w=KettleWater()
        for _ in range(20):r=w.step(.05,*self.poses(0))
        self.assertEqual(r['discharged_ml'],0)
    def test_capture_drain_conservation(self):
        w=KettleWater(initial_ml=200,coffee_g=20,max_flow_ml_s=5)
        poses=self.poses(85)
        for _ in range(2200):r=w.step(.05,*poses)
        self.assertLess(r['source_ml'],.1)
        self.assertLess(r['spilled_ml'],.01)
        self.assertAlmostEqual(r['cloth_retained_ml'],5)
        self.assertAlmostEqual(r['coffee_retained_ml'],40)
        self.assertGreater(r['receiver_ml'],150)
        self.assertLess(abs(r['mass_balance_error_ml']),1e-7)
    def test_miss_is_spill(self):
        w=KettleWater(initial_ml=40)
        for _ in range(500):r=w.step(.02,*self.poses(85,True))
        self.assertGreater(r['spilled_ml'],39)
        self.assertEqual(r['captured_ml'],0)
    def test_spill_from_tipped_cup(self):
        w=KettleWater(initial_ml=40);w.source_ml=0;w.receiver_ml=40
        poses=list(self.poses(0));poses[-1]=Rotation.from_euler('x',110,degrees=True).as_matrix();r=w.step(.02,*poses)
        self.assertAlmostEqual(r['spilled_ml'],40)
        self.assertAlmostEqual(r['receiver_ml'],0)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):KettleWater(initial_ml=2000)
        with self.assertRaises(ValueError):KettleWater().step(0,*self.poses(0))
if __name__=='__main__':unittest.main()
