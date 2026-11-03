import math
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg
from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CAR, CarControllerParams, SubaruFlags

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut


_ASCENT_ACCEL_FEEDBACK_MIN_SPEED = 70.0 / 3.6
_ASCENT_ACCEL_FEEDBACK_FULL_SPEED = 80.0 / 3.6
_ASCENT_ACCEL_FEEDBACK_TIME_CONSTANT = 0.75
_ASCENT_ACCEL_FEEDBACK_GAIN = 1.5
_ASCENT_ACCEL_FEEDBACK_MAX_CORRECTION = 0.65


def _throttle_max_for_car(car_fingerprint):
  if car_fingerprint == CAR.SUBARU_ASCENT:
    return CarControllerParams.ASCENT_THROTTLE_MAX
  return CarControllerParams.THROTTLE_MAX


def _ascent_adaptive_accel_request(requested_accel, measured_accel, v_ego, filtered_error, dt):
  """Apply a bounded, speed-gated correction for the Ascent's measured accel shortfall."""
  values = (requested_accel, measured_accel, v_ego, filtered_error, dt)
  if not all(math.isfinite(value) for value in values) or requested_accel <= 0.0 or dt <= 0.0:
    return requested_accel, 0.0

  if v_ego < _ASCENT_ACCEL_FEEDBACK_MIN_SPEED:
    return requested_accel, 0.0

  alpha = dt / (_ASCENT_ACCEL_FEEDBACK_TIME_CONSTANT + dt)
  filtered_error += alpha * (requested_accel - measured_accel - filtered_error)
  filtered_error = min(max(filtered_error, -1.0), 1.0)
  speed_weight = min(max(
    (v_ego - _ASCENT_ACCEL_FEEDBACK_MIN_SPEED) /
    (_ASCENT_ACCEL_FEEDBACK_FULL_SPEED - _ASCENT_ACCEL_FEEDBACK_MIN_SPEED),
    0.0), 1.0)
  correction = min(max(filtered_error, 0.0) * _ASCENT_ACCEL_FEEDBACK_GAIN,
                   _ASCENT_ACCEL_FEEDBACK_MAX_CORRECTION) * speed_weight
  return requested_accel + correction, filtered_error


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_torque_last = 0

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0
    self.ascent_accel_feedback_error = 0.0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:
      apply_torque = int(round(actuators.torque * self.p.STEER_MAX))

      # limits due to driver torque

      new_torque = int(round(apply_torque))
      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

      if not CC.latActive:
        apply_torque = 0

      if self.CP.flags & SubaruFlags.PREGLOBAL:
        can_sends.append(subarucan.create_preglobal_steering_control(self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive))
      else:
        apply_steer_req = CC.latActive

        if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
          # Steering rate fault prevention
          self.steer_rate_counter, apply_steer_req = \
            common_fault_avoidance(abs(CS.out.steeringRateDeg) > MAX_STEER_RATE, apply_steer_req,
                                   self.steer_rate_counter, MAX_STEER_RATE_FRAMES)

        can_sends.append(subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req))

      self.apply_torque_last = apply_torque

    # *** longitudinal ***

    if CC.longActive:
      if self.CP.carFingerprint == CAR.SUBARU_ASCENT and actuators.accel >= 0.0:
        accel_ff, self.ascent_accel_feedback_error = _ascent_adaptive_accel_request(
          actuators.accel, CS.out.aEgo, CS.out.vEgo, self.ascent_accel_feedback_error, DT_CTRL,
        )
        accel_ff = np.clip(accel_ff, 0.0, 2.0)
        throttle_base = np.interp(CS.out.vEgo, CarControllerParams.ASCENT_LONG_SPEED_BP,
                                  CarControllerParams.ASCENT_THROTTLE_BASE_V)
        rpm_base = np.interp(CS.out.vEgo, CarControllerParams.ASCENT_LONG_SPEED_BP,
                             CarControllerParams.ASCENT_RPM_BASE_V)
        apply_throttle = int(round(throttle_base + accel_ff * CarControllerParams.ASCENT_THROTTLE_ACCEL_GAIN))
        apply_rpm = int(round(rpm_base + accel_ff * CarControllerParams.ASCENT_RPM_ACCEL_GAIN))
      else:
        if self.CP.carFingerprint == CAR.SUBARU_ASCENT:
          self.ascent_accel_feedback_error = 0.0
        apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
        apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      throttle_max = _throttle_max_for_car(self.CP.carFingerprint)
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, throttle_max)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      self.ascent_accel_feedback_error = 0.0
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                        self.CP.openpilotLongitudinalControl, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, self.frame // 10, CS.es_lkas_state_msg, CC.enabled, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        if pcm_cancel_cmd:
          if not (self.CP.flags & SubaruFlags.HYBRID):
            bus = CanBus.alt if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else CanBus.main
            can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg["COUNTER"] + 1, CS.es_distance_msg, bus, pcm_cancel_cmd))

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    new_actuators = actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends
