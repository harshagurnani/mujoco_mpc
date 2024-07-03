// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/fruitfly/flytrackingfree/flytrackingfree.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include "mjpc/utilities.h"

namespace {
// compute interpolation between mocap frames
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

// Hardcoded constant matching keyframes from CMU mocap dataset.
constexpr double kFps = 200.0;

constexpr int kMotionLengths[] = {
    8,  // FlyStand
    8,  // FlyStand
    1700,   // FlytrackingFree
    1560,  // FlytrackingFree
           // 121,  // Jump - CMU-CMU-02-02_04
           // 154,  // Kick Spin - CMU-CMU-87-87_01
           // 115,  // Spin Kick - CMU-CMU-88-88_06
           // 78,   // Cartwheel (1) - CMU-CMU-88-88_07
           // 145,  // Crouch Flip - CMU-CMU-88-88_08
           // 188,  // Cartwheel (2) - CMU-CMU-88-88_09
           // 260,  // Monkey Flip - CMU-CMU-90-90_19
           // 279,  // Dance - CMU-CMU-103-103_08
           // 39,   // Run - CMU-CMU-108-108_13
           // 510,  // Walk - CMU-CMU-137-137_40
};

// return length of motion trajectory
int MotionLength(int id) { return kMotionLengths[id]; }

// return starting keyframe index for motion
int MotionStartIndex(int id) {
  int start = 0;
  for (int i = 0; i < id; i++) {
    start += MotionLength(i);
  }
  return start;
}

// names for fruitfly bodies
const std::array<std::string, 30> body_names = {
    "coxa_T1_left",    "femur_T1_left",  "tibia_T1_left",   "tarsus_T1_left",
    "claw_T1_left",    "coxa_T1_right",  "femur_T1_right",  "tibia_T1_right",
    "tarsus_T1_right", "claw_T1_right",  "coxa_T2_left",    "femur_T2_left",
    "tibia_T2_left",   "tarsus_T2_left", "claw_T2_left",    "coxa_T2_right",
    "femur_T2_right",  "tibia_T2_right", "tarsus_T2_right", "claw_T2_right",
    "coxa_T3_left",    "femur_T3_left",  "tibia_T3_left",   "tarsus_T3_left",
    "claw_T3_left",    "coxa_T3_right",  "femur_T3_right",  "tibia_T3_right",
    "tarsus_T3_right", "claw_T3_right"};

}  // namespace

namespace mjpc::fruitfly {

std::string FlyTrackingFree::XmlPath() const {
  return GetModelPath("fruitfly/flytrackingfree/task.xml");
}
std::string FlyTrackingFree::Name() const { return "Fruitfly TrackFree"; }

// ------------- Residuals for fruitfly tracking task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2): 
//     Residual (2-31): Tracking position: minimise tracking position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (31-66): Tracking velocity: minimise tracking velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------
void FlyTrackingFree::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
  // ----- get mocap frames ----- //
  // get motion start index
  int start = MotionStartIndex(current_mode_);
  // get motion trajectory length
  int length = MotionLength(current_mode_);
  double current_index = (data->time - reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- residual ----- //
  int counter = 0;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel - 12, model->nv - 12);
  counter += model->nv - 12;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;


  // ----- balance ----- //
  // ----- thorax height ----- //
  double* thorax_pos = data->xipos + 3*thorax_body_id_;
  // double thorax_height = SensorByName(model, data, "thorax_position")[2];
  double thorax_height = thorax_pos[2];
  residual[counter++] = thorax_height - parameters_[0];

  // ----- Thorax / feet ----- //
  double* foot_pos[kNumFoot];
  for (FlyFoot foot : kFootAll)
    foot_pos[foot] = data->site_xpos + 3 * foot_geom_id_[foot];

  double avg_foot_pos = 0.167*(foot_pos[kFootT1L][2] + foot_pos[kFootT1R][2] + foot_pos[kFootT2L][2] + foot_pos[kFootT2R][2] + foot_pos[kFootT3L][2] + foot_pos[kFootT3R][2]);
  // avg_foot_pos = 0.167*(foot_pos[kFootT1L][2] + foot_pos[kFootT1R][2] + foot_pos[kFootT2L][2] + foot_pos[kFootT2R][2] + foot_pos[kFootT3L][2] + foot_pos[kFootT3R][2]);
  double* coxa_right = SensorByName(model, data, "tracking_pos[coxa_T2_left]");
  double* coxa_left = SensorByName(model, data, "tracking_pos[coxa_T2_right]");
  // double* foot_right = foot_pos[kFootT2L];
  // double* foot_left = foot_pos[kFootT2R];
  residual[counter++] = avg_foot_pos - thorax_height - 0.2;


  // capture point
  double* subcom = SensorByName(model, data, "thorax_subcom");
  double* subcomvel = SensorByName(model, data, "thorax_subcomvel");
  

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, coxa_right, coxa_left);
  axis[2] = 1.0e-3;
  double ax_len = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, coxa_right, coxa_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center); // maybe create axis going length of body?

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-ax_len, mju_min(ax_len, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing = thorax_height / mju_sqrt(thorax_height * thorax_height + (0.1 * 0.1)) - 0.2;
  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ----- balance gryo ----- //
  double* thorax_gyro = SensorByName(model, data, "thorax_gyro");
  for (int i=0; i < 3; i++) {
    // current gyro
    residual[counter] = thorax_gyro[i];
    counter += 1;
  }

  // ----- walk ----- //
  double* thorax_forward = SensorByName(model, data, "thorax_forward");
  // double* foot_right_forward = SensorByName(model, data, "foot_right_forward");
  // double* foot_left_forward = SensorByName(model, data, "foot_left_forward");

  double forward[2];
  mju_copy(forward, thorax_forward, 2);
  // mju_addTo(forward, foot_right_forward, 2);
  // mju_addTo(forward, foot_left_forward, 2);
  mju_normalize(forward, 2);

  // com vel
  double* head_subcomvel = SensorByName(model, data, "head_subcomvel");
  double* thorax_velocity = SensorByName(model, data, "thorax_velocity");
  double com_vel[2];
  mju_add(com_vel, head_subcomvel, thorax_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // walk forward
  residual[counter++] = standing * (mju_dot(com_vel, forward, 2) - parameters_[1]);

  // ----- move feet ----- //
  // double* foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  // double* foot_left_vel = SensorByName(model, data, "foot_left_velocity");
  // double move_feet[2];
  // mju_copy(move_feet, com_vel, 2);
  // mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  // mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  // mju_copy(&residual[counter], move_feet, 2);
  // mju_scl(&residual[counter], &residual[counter], standing, 2);
  // counter += 2;

  // ----- position ----- //
  // Compute interpolated frame.
  auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // current frame
    mju_scl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
        weight_0);

    // next frame
    mju_addToScl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
        weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy3(result, sensor_pos);
  };

  // compute marker and sensor averages
  double avg_mpos[3] = {0};
  double avg_sensor_pos[3] = {0};
  int num_body = 0;
  for (const auto &body_name : body_names) {
    double body_mpos[3];
    double body_sensor_pos[3];
    get_body_mpos(body_name, body_mpos);
    mju_addTo3(avg_mpos, body_mpos);
    get_body_sensor_pos(body_name, body_sensor_pos);
    mju_addTo3(avg_sensor_pos, body_sensor_pos);
    num_body++;
  }
  mju_scl3(avg_mpos, avg_mpos, 1.0 / num_body);
  mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0 / num_body);

  // residual for averages
  // mju_sub3(&residual[counter], avg_mpos, avg_sensor_pos);
  // counter += 3;

  for (const auto &body_name : body_names) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    mju_subFrom3(body_mpos, avg_mpos);
    mju_subFrom3(body_sensor_pos, avg_sensor_pos);

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);

    counter += 3;
  }

  // ----- velocity ----- //
  for (const auto &body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // compute finite-difference velocity
    mju_copy3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
    mju_subFrom3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
    mju_scl3(&residual[counter], &residual[counter], kFps);

    // subtract current velocity
    double *sensor_linvel =
        SensorByName(model, data, linvel_sensor_name.c_str());
    mju_subFrom3(&residual[counter], sensor_linvel);

    counter += 3;
  }

  CheckSensorDim(model, counter);
}

// --------------------- Transition for fruitfly task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void FlyTrackingFree::TransitionLocked(mjModel *model, mjData *d) {
  // get motion start index
  int start = MotionStartIndex(mode);
  // get motion trajectory length
  int length = MotionLength(mode);

  // check for motion switch
  if (residual_.current_mode_ != mode || d->time == 0.0) {
    residual_.current_mode_ = mode;       // set motion id
    residual_.reference_time_ = d->time;  // set reference time

    // set initial state
    mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
    mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  }

  // indices
  double current_index = (d->time - residual_.reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mj_markStack(d);

  mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
  mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

  // Compute interpolated frame.
  mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);

  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

  mj_freeStack(d);
}

// TODO: Make ids calulcatable for fly
//  ============  task-state utilities  ============
// save task-related ids
void FlyTrackingFree::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  residual_.jointVel_id_ = CostTermByName(model, "JointVel");
  residual_.control_id_ = CostTermByName(model, "Control");
  residual_.height_id_ = CostTermByName(model, "Height");
  residual_.balance_id_ = CostTermByName(model, "Balance");
  residual_.upright_id_ = CostTermByName(model, "Upright");

  // ----------  model identifiers  ----------
  residual_.thorax_body_id_ = mj_name2id(model, mjOBJ_XBODY, "thorax");
  if (residual_.thorax_body_id_ < 0) mju_error("body 'thorax' not found");

  residual_.head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
  if (residual_.head_site_id_ < 0) mju_error("site 'head' not found");

  // foot geom ids
  int foot_index = 0;
  for (const char* footname : {"tracking[claw_T1_left]", "tracking[claw_T1_right]",
                               "tracking[claw_T2_left]", "tracking[claw_T2_right]",
                               "tracking[claw_T3_left]", "tracking[claw_T3_right]"}) {
    int foot_id = mj_name2id(model, mjOBJ_SITE, footname);
    if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
    residual_.foot_geom_id_[foot_index] = foot_id;
    foot_index++;
  }

  // shoulder body ids
  // int shoulder_index = 0;
  // for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
  //   int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
  //   if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
  //   residual_.shoulder_body_id_[shoulder_index] = foot_id;
  //   shoulder_index++;
  // }

}

}  // namespace mjpc::fruitfly
