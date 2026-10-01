import unittest
from kettle_heater import KettleHeater
class HeaterTests(unittest.TestCase):
 def test_timer_cannot_heat_unpressed_kettle(self):
  h=KettleHeater()
  for _ in range(1000):h.step(.1,800,True,True,True)
  self.assertEqual(h.temperature_C,25);self.assertEqual(h.input_energy_J,0)
 def test_contact_and_stroke_both_required(self):
  for angle,force in [(0,5),(.2,0)]:
   h=KettleHeater()
   for _ in range(10):h.step(.01,800,True,True,True,angle,force)
   self.assertFalse(h.on)
 def test_boil_energy_and_cutoff(self):
  h=KettleHeater()
  for _ in range(5):h.step(.01,800,True,True,True,.2,1.)
  self.assertTrue(h.on);seconds=0
  while h.on and seconds<1000:h.step(.1,800,True,True,True,.18,0);seconds+=.1
  self.assertEqual(h.last_event,'automatic_boil_cutoff');self.assertGreater(seconds,250);self.assertLess(seconds,400)
  stored=(h.temperature_C-25)*(390+.8*4180)
  self.assertAlmostEqual(h.delivered_energy_J-h.loss_energy_J,stored,places=6)
 def test_lift_removes_power(self):
  h=KettleHeater(on=True);h.step(.02,800,False,True,True);self.assertFalse(h.on);self.assertEqual(h.input_energy_J,0)
 def test_underfill_rejects_press(self):
  h=KettleHeater()
  for _ in range(10):h.step(.01,200,True,True,True,.2,1.)
  self.assertFalse(h.on);self.assertEqual(h.temperature_C,25)
if __name__=='__main__':unittest.main()
