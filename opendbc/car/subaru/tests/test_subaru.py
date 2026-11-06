import unittest

from opendbc.car.subaru.carcontroller import _ascent_adaptive_accel_request, _throttle_max_for_car
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CarControllerParams, SubaruSafetyFlags


class TestSubaruFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"


class TestSubaruAscentLongitudinal(unittest.TestCase):
  def test_adaptive_accel_feedback_is_speed_gated_and_bounded(self):
    accel, filtered_error = _ascent_adaptive_accel_request(0.4, 0.0, 80 / 3.6, 0.0, 0.01)
    self.assertGreater(filtered_error, 0.0)
    self.assertGreater(accel, 0.4)
    self.assertLess(accel, 0.41)

    for _ in range(1000):
      accel, filtered_error = _ascent_adaptive_accel_request(0.4, -0.2, 110 / 3.6, filtered_error, 0.01)
    self.assertAlmostEqual(accel, 1.05)

  def test_adaptive_accel_feedback_reaches_full_weight_by_80_kph(self):
    accel, _ = _ascent_adaptive_accel_request(0.4, 0.4, 75 / 3.6, 1.0, 0.01)
    self.assertAlmostEqual(accel, 0.725, delta=0.001)

  def test_adaptive_accel_feedback_does_not_boost_low_speed_or_braking(self):
    low_speed_accel, low_speed_error = _ascent_adaptive_accel_request(0.4, 0.0, 60 / 3.6, 0.0, 0.01)
    braking_accel, braking_error = _ascent_adaptive_accel_request(-0.5, -1.0, 110 / 3.6, 0.4, 0.01)

    self.assertAlmostEqual(low_speed_accel, 0.4)
    self.assertAlmostEqual(low_speed_error, 0.0)
    self.assertAlmostEqual(braking_accel, -0.5)
    self.assertAlmostEqual(braking_error, 0.0)

  def test_alpha_longitudinal_and_higher_throttle_limit_are_ascent_only(self):
    ascent_cp = CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT)
    other_cp = CarInterface.get_non_essential_params(CAR.SUBARU_IMPREZA_2020)

    self.assertTrue(ascent_cp.alphaLongitudinalAvailable)
    self.assertFalse(other_cp.alphaLongitudinalAvailable)
    self.assertEqual(CarControllerParams.THROTTLE_MAX, 3400)
    self.assertEqual(CarControllerParams.ASCENT_THROTTLE_MAX, 3800)
    self.assertEqual(_throttle_max_for_car(CAR.SUBARU_ASCENT), 3800)
    self.assertEqual(_throttle_max_for_car(CAR.SUBARU_IMPREZA_2020), 3400)
    self.assertTrue(ascent_cp.safetyConfigs[0].safetyParam & SubaruSafetyFlags.ASCENT_LONG_THROTTLE)
    self.assertFalse(other_cp.safetyConfigs[0].safetyParam & SubaruSafetyFlags.ASCENT_LONG_THROTTLE)
