"""Conservation and geometric gate checks for reduced grounds model."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from coffee_grounds import CoffeeGrounds

class GroundTests(unittest.TestCase):
 def setUp(self):self.I=np.eye(3);self.pot=np.array([0.,0.,0.]);self.filt=np.array([1.,0.,0.]);self.g=CoffeeGrounds()
 def step(self,bowl,R=None,dt=.02):
  R=self.I if R is None else R
  return self.g.step(dt,np.array(bowl)-R@np.array([-.045,0,.003]),R,self.pot,self.I,self.filt,self.I)
 def test_capacity_and_stationary_no_creation(self):
  self.assertAlmostEqual(self.g.capacity_ml,1.264491043,places=8)
  for _ in range(100):self.step([0,0,.03])
  self.assertEqual(self.g.spoon_g,0)
 def test_pickup_requires_actual_sweep_inside_bed(self):
  for x in np.linspace(-.02,.02,100):self.step([x,0,.03])
  self.assertGreater(self.g.spoon_g,.1);self.assertLessEqual(self.g.spoon_g,self.g.capacity_ml*.35)
  self.assertAlmostEqual(self.g.pot_g+self.g.spoon_g,100)
 def test_angled_scoop_retains_some_powder_without_creating_mass(self):
  R=Rotation.from_euler('y',-32,degrees=True).as_matrix()
  for x in np.linspace(-.02,.02,100):self.step([x,0,.03],R)
  self.assertGreater(self.g.spoon_g,0);self.assertLess(self.g.spoon_g,self.g.capacity_ml*.35)
  self.assertAlmostEqual(self.g.pot_g+self.g.spoon_g+self.g.spilled_g,100)
 def test_above_or_outside_pot_no_pickup(self):
  for x in np.linspace(-.02,.02,100):self.step([x,0,.2])
  for x in np.linspace(.2,.3,100):self.step([x,0,.03])
  self.assertEqual(self.g.spoon_g,0)
 def test_tilt_releases_into_filter_or_spill(self):
  for x in np.linspace(-.02,.02,100):self.step([x,0,.03])
  dose=self.g.spoon_g
  # Move slowly above the filter then stop; mass still belongs to the spoon.
  for x in np.linspace(.02,1,500):self.step([x,0,.4])
  for _ in range(20):self.step([1,0,.4])
  self.assertEqual(self.g.filter_g,0)
  R=Rotation.from_euler('x',100,degrees=True).as_matrix()
  for _ in range(100):self.step([1,0,.4],R)
  self.assertAlmostEqual(self.g.filter_g,dose);self.assertAlmostEqual(self.g.spilled_g,0);self.assertAlmostEqual(self.g.pot_g+self.g.filter_g,100)
 def test_miss_does_not_create_brew_dose(self):
  for x in np.linspace(-.02,.02,100):self.step([x,0,.03])
  dose=self.g.spoon_g
  for x in np.linspace(.02,.5,200):self.step([x,0,.4])
  for _ in range(20):self.step([.5,0,.4])
  R=Rotation.from_euler('x',100,degrees=True).as_matrix()
  for _ in range(100):self.step([.5,0,.4],R)
  self.assertAlmostEqual(self.g.spilled_g,dose);self.assertEqual(self.g.filter_g,0)
if __name__=='__main__':unittest.main()
