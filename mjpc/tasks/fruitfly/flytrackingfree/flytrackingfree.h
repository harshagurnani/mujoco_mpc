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

#ifndef MJPC_TASKS_FRUITFLY_TRACKINGFREE_TASK_H_
#define MJPC_TASKS_FRUITFLY_TRACKINGFREE_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace fruitfly {

class FlyTrackingFree : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FlyTrackingFree* task, int current_mode = 0,
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
    friend class FlyTrackingFree;
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
    constexpr static FlyFoot kFootAll[kNumFoot] = {kFootT1L, kFootT1R, 
                                                   kFootT2L, kFootT2R, 
                                                   kFootT3L, kFootT3R, };
    //  ============  enums  ============
    int jointVel_id_           = -1;
    int thorax_body_id_        = -1;
    int head_site_id_          = -1;
    int control_id_            = -1;
    int height_id_             = -1;
    int balance_id_            = -1;
    int upright_id_            = -1;
    int foot_geom_id_[kNumFoot];
  };

  FlyTrackingFree() : residual_(this) {}

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
