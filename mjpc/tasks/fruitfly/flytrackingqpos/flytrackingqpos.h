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

#ifndef MJPC_TASKS_FRUITFLY_TRACKINGQPOS_TASK_H_
#define MJPC_TASKS_FRUITFLY_TRACKINGQPOS_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace fruitfly {

class FlyTrackingQpos : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FlyTrackingQpos* task, int current_mode = 0,
                        double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          current_mode_(current_mode),
          reference_time_(reference_time) {}

    // ------------- Residuals for fruitfly tracking task -------------
    //   Number of residuals:
    //     Residual (0): Joint vel: minimise joint velocity
    //     Residual (1): Control: minimise control
    //     Residual (2-11): Tracking position: minimise tracking position error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //     Residual (11-20): Tracking velocity: minimise tracking velocity error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //   Number of parameters: 0
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
   private:
    friend class FlyTrackingQpos;
    int current_mode_;
    double reference_time_;
    // feet
    enum FlyFoot {
      kFootT1L  = 0,
      kFootT1R,
      kFootT2L,
      kFootT2R,
      kFootT3L,
      kFootT3R,
      kNumFoot
    };

    enum FlyJoint {
      kJointT1_CA_L  = 0,
      kJointT1_CT_L, kJointT1_CE_L, kJointT1_FE_L, kJointT1_FT_L, kJointT1_TiE_L, kJointT1_TaT_L,
      kJointT1_CA_R, kJointT1_CT_R, kJointT1_CE_R, kJointT1_FE_R, kJointT1_FT_R, kJointT1_TiE_R, kJointT1_TaT_R,
      kJointT2_CA_L, kJointT2_CT_L, kJointT2_CE_L, kJointT2_FE_L, kJointT2_FT_L, kJointT2_TiE_L, kJointT2_TaT_L,
      kJointT2_CA_R, kJointT2_CT_R, kJointT2_CE_R, kJointT2_FE_R, kJointT2_FT_R, kJointT2_TiE_R, kJointT2_TaT_R,
      kJointT3_CA_L, kJointT3_CT_L, kJointT3_CE_L, kJointT3_FE_L, kJointT3_FT_L, kJointT3_TiE_L, kJointT3_TaT_L, 
      kJointT3_CA_R, kJointT3_CT_R, kJointT3_CE_R, kJointT3_FE_R, kJointT3_FT_R, kJointT3_TiE_R, kJointT3_TaT_R, 
      kNumJoint
    };


    constexpr static FlyFoot kFootAll[kNumFoot] = {kFootT1L, kFootT1R, 
                                                   kFootT2L, kFootT2R, 
                                                   kFootT3L, kFootT3R, };


    constexpr static FlyJoint kJointAll[kNumJoint] = {kJointT1_CA_L, kJointT1_CT_L, kJointT1_CE_L, kJointT1_FE_L, kJointT1_FT_L, kJointT1_TiE_L, kJointT1_TaT_L,
                                                      kJointT1_CA_R, kJointT1_CT_R, kJointT1_CE_R, kJointT1_FE_R, kJointT1_FT_R, kJointT1_TiE_R, kJointT1_TaT_R,
                                                      kJointT2_CA_L, kJointT2_CT_L, kJointT2_CE_L, kJointT2_FE_L, kJointT2_FT_L, kJointT2_TiE_L, kJointT2_TaT_L,
                                                      kJointT2_CA_R, kJointT2_CT_R, kJointT2_CE_R, kJointT2_FE_R, kJointT2_FT_R, kJointT2_TiE_R, kJointT2_TaT_R,
                                                      kJointT3_CA_L, kJointT3_CT_L, kJointT3_CE_L, kJointT3_FE_L, kJointT3_FT_L, kJointT3_TiE_L, kJointT3_TaT_L, 
                                                      kJointT3_CA_R, kJointT3_CT_R, kJointT3_CE_R, kJointT3_FE_R, kJointT3_FT_R, kJointT3_TiE_R, kJointT3_TaT_R,};

    //  ============  enums  ============
    int jointVel_id_           = -1;
    int thorax_body_id_        = -1;
    int head_site_id_          = -1;
    int control_id_            = -1;
    int height_id_             = -1;
    int balance_id_            = -1;
    int upright_id_            = -1;
    int foot_geom_id_[kNumFoot];
    int joint_geom_id_[kNumJoint];
  };

  FlyTrackingQpos() : residual_(this) {}

  // --------------------- Transition for fruitfly task ------------------------
  //   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
  //   Linearly interpolate between two consecutive key frames in order to
  //   smooth the transitions between keyframes.
  // ---------------------------------------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;

  void ResetLocked(const mjModel* model) override;

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.reference_time_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  // int current_mode_;
  // double reference_time_;
  ResidualFn residual_;
};

}  // namespace fruitfly
}  // namespace mjpc

#endif  // MJPC_TASKS_FRUITFLY_TRACKING_TASK_H_
