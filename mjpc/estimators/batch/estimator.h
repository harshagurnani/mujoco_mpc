// Copyright 2023 DeepMind Technologies Limited
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

#ifndef MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_
#define MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_

#include <mujoco/mujoco.h>

#include <vector>

#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {

const int MAX_HISTORY = 32;  // maximum length configuration trajectory

// convert sequence of configurations to velocities
void ConfigurationToVelocity(double* velocity, const double* configuration,
                             int configuration_length, const mjModel* model);

// convert sequence of configurations to accelerations
void VelocityToAcceleration(double* acceleration, const double* velocity,
                            int velocity_length, const mjModel* model);

class Estimator {
 public:
  // constructor
  Estimator() {}

  // destructor
  ~Estimator() {}

  // initialize
  void Initialize(mjModel* model);

  // prior cost
  double CostPrior(double* gradient, double* hessian);

  // prior residual
  void ResidualPrior();

  // prior Jacobian
  void JacobianPrior();

  // measurement cost
  double CostMeasurement(double* gradient, double* hessian);

  // measurement residual
  void ResidualMeasurement();

  // measurement Jacobian
  void JacobianMeasurement();

  // compute measurements
  void ComputeMeasurements();

  // inverse dynamics cost
  double CostInverseDynamics(double* gradient, double* hessian);

  // inverse dynamics residual
  void ResidualInverseDynamics();

  // inverse dynamics Jacobian
  void JacobianInverseDynamics();

  // compute inverse dynamics
  void ComputeInverseDynamics();

  // update configuration trajectory
  void UpdateConfiguration(double* configuration, const double* update);

  // update configuration, velocity, acceleration, measurement, and qfrc trajectories
  void UpdateTrajectory(double* configuration, const double* update);

  // model
  mjModel* model_;

  // data
  mjData* data_;

  // trajectories
  int configuration_length_;
  std::vector<double> configuration_;
  std::vector<double> configuration_prior_;
  std::vector<double> configuration_copy_;
  std::vector<double> velocity_;
  std::vector<double> acceleration_;

  // measurements
  int dim_measurement_;
  std::vector<double> measurement_sensor_;
  std::vector<double> measurement_model_;

  // qfrc
  std::vector<double> qfrc_actuator_;
  std::vector<double> qfrc_inverse_;

  // residual
  std::vector<double> residual_prior_;
  std::vector<double> residual_measurement_;
  std::vector<double> residual_inverse_dynamics_;

  // Jacobian
  std::vector<double> jacobian_prior_;
  std::vector<double> jacobian_measurement_;
  std::vector<double> jacobian_inverse_dynamics_;

  // cost gradient
  std::vector<double> cost_gradient_prior_;
  std::vector<double> cost_gradient_measurement_;
  std::vector<double> cost_gradient_inverse_dynamics_;
  std::vector<double> cost_gradient_total_;

  // cost Hessian
  std::vector<double> cost_hessian_prior_;
  std::vector<double> cost_hessian_measurement_;
  std::vector<double> cost_hessian_inverse_dynamics_;
  std::vector<double> cost_hessian_total_;

  // scratch 
  std::vector<double> scratch_prior_;
  std::vector<double> scratch_measurement_;
  std::vector<double> scratch_inverse_dynamics_;

  // weight TODO(taylor): matrices
  double weight_prior_;
  double weight_measurement_;
  double weight_inverse_dynamics_;

  // cost norms
  NormType norm_prior_;
  NormType norm_measurement_;
  NormType norm_inverse_dynamics_;

  // cost norm parameters
  std::vector<double> norm_parameters_prior_;
  std::vector<double> norm_parameters_measurement_;
  std::vector<double> norm_parameters_inverse_dynamics_;

  // norm gradient
  std::vector<double> norm_gradient_prior_;
  std::vector<double> norm_gradient_measurement_;
  std::vector<double> norm_gradient_inverse_dynamics_;

  // norm Hessian
  std::vector<double> norm_hessian_prior_;
  std::vector<double> norm_hessian_measurement_;
  std::vector<double> norm_hessian_inverse_dynamics_;

  // update
  std::vector<double> update_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_
