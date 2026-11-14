import unittest
from lm.resource_guard import Guard,Policy
class SourcePolicyTests(unittest.TestCase):
 def test_half_gib_threshold(self):
  n=512*1024**2;g=Guard(Policy(available_reserve_bytes=n,immediate_available_bytes=n))
  self.assertIsNone(g.observe(n,[]))
  self.assertEqual(g.observe(n-1,[]),'available_system_ram_below_0.5_GiB_first_sample')
 def test_full_family_objective(self):
  from lm.source_learning import family_weights
  rows=[{'family':'a'}]*8+[{'family':'b'}]*2
  w=family_weights(rows)
  self.assertAlmostEqual(float(w.mean()),1.)
  losses=[2.]*8+[6.]*2
  self.assertAlmostEqual(sum(float(v)*l for v,l in zip(w,losses))/10,4.)
