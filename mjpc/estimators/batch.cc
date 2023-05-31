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

#include "mjpc/estimators/batch.h"

#include <chrono>

#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// initialize estimator
void Estimator::Initialize(mjModel* model) {
  // model
  model_ = model;

  // data
  for (int i = 0; i < MAX_HISTORY; i++) {
    data_.push_back(MakeUniqueMjData(mj_makeData(model)));
  }

  // dimension
  int nq = model->nq, nv = model->nv;

  // trajectories
  configuration_length_ = GetNumberOrDefault(10, model, "batch_length");
  configuration_.resize(nq * MAX_HISTORY);
  velocity_.resize(nv * MAX_HISTORY);
  acceleration_.resize(nv * MAX_HISTORY);
  time_.resize(MAX_HISTORY);

  // prior
  configuration_prior_.resize(nq * MAX_HISTORY);

  // sensor
  dim_sensor_ = model->nsensordata;  // TODO(taylor): grab from model
  sensor_measurement_.resize(dim_sensor_ * MAX_HISTORY);
  sensor_prediction_.resize(dim_sensor_ * MAX_HISTORY);

  // force
  force_measurement_.resize(nv * MAX_HISTORY);
  force_prediction_.resize(nv * MAX_HISTORY);

  // residual
  residual_prior_.resize(nv * MAX_HISTORY);
  residual_sensor_.resize(dim_sensor_ * MAX_HISTORY);
  residual_force_.resize(nv * MAX_HISTORY);

  // Jacobian
  jacobian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_sensor_.resize((dim_sensor_ * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior Jacobian block
  block_prior_current_configuration_.resize((nv * nv) * MAX_HISTORY);

  // sensor Jacobian blocks
  block_sensor_configuration_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_velocity_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_acceleration_.resize((dim_sensor_ * nv) * MAX_HISTORY);

  block_sensor_previous_configuration_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_current_configuration_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_next_configuration_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_configurations_.resize((dim_sensor_ * 3 * nv) * MAX_HISTORY);

  block_sensor_scratch_.resize(mju_max(nv, dim_sensor_) *
                               mju_max(nv, dim_sensor_) * MAX_HISTORY);

  // force Jacobian blocks
  block_force_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_force_velocity_.resize((nv * nv) * MAX_HISTORY);
  block_force_acceleration_.resize((nv * nv) * MAX_HISTORY);

  block_force_previous_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_force_current_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_force_next_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_force_configurations_.resize((nv * 3 * nv) * MAX_HISTORY);

  block_force_scratch_.resize((nv * nv) * MAX_HISTORY);

  // velocity Jacobian blocks
  block_velocity_previous_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_velocity_current_configuration_.resize((nv * nv) * MAX_HISTORY);

  // acceleration Jacobian blocks
  block_acceleration_previous_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_acceleration_current_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_acceleration_next_configuration_.resize((nv * nv) * MAX_HISTORY);

  // cost gradient
  cost_gradient_prior_.resize(nv * MAX_HISTORY);
  cost_gradient_sensor_.resize(nv * MAX_HISTORY);
  cost_gradient_force_.resize(nv * MAX_HISTORY);
  cost_gradient_.resize(nv * MAX_HISTORY);

  // cost Hessian
  cost_hessian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_sensor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_band_.resize(BandMatrixNonZeros(nv * MAX_HISTORY, 3 * nv));
  cost_hessian_factor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior weights
  scale_prior_ = GetNumberOrDefault(1.0, model, "batch_scale_prior");
  weight_prior_dense_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  weight_prior_band_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  weight_prior_update_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // sensor weights
  // TODO(taylor): only grab measurement sensors
  weight_sensor_.resize(model->nsensor);

  // TODO(taylor): method for xml to initial weight
  for (int i = 0; i < model->nsensor; i++) {
    weight_sensor_[i] = GetNumberOrDefault(1.0, model, "batch_scale_sensor");
  }

  // force weights
  weight_force_[0] = GetNumberOrDefault(1.0, model, "batch_scale_force_free");
  weight_force_[1] = GetNumberOrDefault(1.0, model, "batch_scale_force_ball");
  weight_force_[2] = GetNumberOrDefault(1.0, model, "batch_scale_force_slide");
  weight_force_[3] = GetNumberOrDefault(1.0, model, "batch_scale_force_hinge");

  // cost norms
  // TODO(taylor): only grab measurement sensors
  norm_sensor_.resize(model->nsensor);

  // TODO(taylor): method for xml to initial weight
  for (int i = 0; i < model->nsensor; i++) {
    norm_sensor_[i] =
        (NormType)GetNumberOrDefault(0, model, "batch_norm_sensor");
  }

  norm_force_[0] =
      (NormType)GetNumberOrDefault(0, model, "batch_norm_force_free");
  norm_force_[1] =
      (NormType)GetNumberOrDefault(0, model, "batch_norm_force_ball");
  norm_force_[2] =
      (NormType)GetNumberOrDefault(0, model, "batch_norm_force_slide");
  norm_force_[3] =
      (NormType)GetNumberOrDefault(0, model, "batch_norm_force_hinge");

  // cost norm parameters
  norm_parameters_sensor_.resize(model->nsensor * 3);

  // norm gradient
  norm_gradient_sensor_.resize(dim_sensor_ * MAX_HISTORY);
  norm_gradient_force_.resize(nv * MAX_HISTORY);

  // norm Hessian
  norm_hessian_sensor_.resize((dim_sensor_ * MAX_HISTORY) *
                              (dim_sensor_ * MAX_HISTORY));
  norm_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  norm_blocks_sensor_.resize(dim_sensor_ * dim_sensor_ * MAX_HISTORY);
  norm_blocks_force_.resize(nv * nv * MAX_HISTORY);

  // scratch
  scratch0_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  scratch0_sensor_.resize((mju_max(nv, dim_sensor_) * MAX_HISTORY) *
                          (mju_max(nv, dim_sensor_) * MAX_HISTORY));
  scratch1_sensor_.resize((mju_max(nv, dim_sensor_) * MAX_HISTORY) *
                          (mju_max(nv, dim_sensor_) * MAX_HISTORY));

  scratch0_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // candidate
  configuration_copy_.resize(nq * MAX_HISTORY);

  // search direction
  search_direction_.resize(nv * MAX_HISTORY);

  // covariance
  covariance_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  covariance_update_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch0_covariance_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_covariance_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch2_covariance_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  covariance_initial_scaling_ =
      GetNumberOrDefault(1.0, model, "batch_covariance_initial_scaling");

  // timer
  timer_prior_step_.resize(MAX_HISTORY);
  timer_sensor_step_.resize(MAX_HISTORY);
  timer_force_step_.resize(MAX_HISTORY);

  // status
  prior_reset_ = true;

  // settings
  band_covariance_ =
      (bool)GetNumberOrDefault(0, model, "batch_band_covariance");

  // reset
  Reset();
}

// reset memory
void Estimator::Reset() {
  // trajectories
  std::fill(configuration_.begin(), configuration_.end(), 0.0);
  std::fill(velocity_.begin(), velocity_.end(), 0.0);
  std::fill(acceleration_.begin(), acceleration_.end(), 0.0);
  std::fill(time_.begin(), time_.end(), 0.0);

  // prior
  std::fill(configuration_prior_.begin(), configuration_prior_.end(), 0.0);

  // sensor
  std::fill(sensor_measurement_.begin(), sensor_measurement_.end(), 0.0);
  std::fill(sensor_prediction_.begin(), sensor_prediction_.end(), 0.0);

  // force
  std::fill(force_measurement_.begin(), force_measurement_.end(), 0.0);
  std::fill(force_prediction_.begin(), force_prediction_.end(), 0.0);

  // residual
  std::fill(residual_prior_.begin(), residual_prior_.end(), 0.0);
  std::fill(residual_sensor_.begin(), residual_sensor_.end(), 0.0);
  std::fill(residual_force_.begin(), residual_force_.end(), 0.0);

  // Jacobian
  std::fill(jacobian_prior_.begin(), jacobian_prior_.end(), 0.0);
  std::fill(jacobian_sensor_.begin(), jacobian_sensor_.end(), 0.0);
  std::fill(jacobian_force_.begin(), jacobian_force_.end(), 0.0);

  // prior Jacobian block
  std::fill(block_prior_current_configuration_.begin(),
            block_prior_current_configuration_.end(), 0.0);

  // sensor Jacobian blocks
  std::fill(block_sensor_configuration_.begin(),
            block_sensor_configuration_.end(), 0.0);
  std::fill(block_sensor_velocity_.begin(), block_sensor_velocity_.end(), 0.0);
  std::fill(block_sensor_acceleration_.begin(),
            block_sensor_acceleration_.end(), 0.0);

  std::fill(block_sensor_previous_configuration_.begin(),
            block_sensor_previous_configuration_.end(), 0.0);

  std::fill(block_sensor_current_configuration_.begin(),
            block_sensor_current_configuration_.end(), 0.0);

  std::fill(block_sensor_next_configuration_.begin(),
            block_sensor_next_configuration_.end(), 0.0);

  std::fill(block_sensor_configurations_.begin(),
            block_sensor_configurations_.end(), 0.0);

  std::fill(block_sensor_scratch_.begin(), block_sensor_scratch_.end(), 0.0);

  // force Jacobian blocks
  std::fill(block_force_configuration_.begin(),
            block_force_configuration_.end(), 0.0);
  std::fill(block_force_velocity_.begin(), block_force_velocity_.end(), 0.0);
  std::fill(block_force_acceleration_.begin(), block_force_acceleration_.end(),
            0.0);

  std::fill(block_force_previous_configuration_.begin(),
            block_force_previous_configuration_.end(), 0.0);

  std::fill(block_force_current_configuration_.begin(),
            block_force_current_configuration_.end(), 0.0);

  std::fill(block_force_next_configuration_.begin(),
            block_force_next_configuration_.end(), 0.0);

  std::fill(block_force_configurations_.begin(),
            block_force_configurations_.end(), 0.0);

  std::fill(block_force_scratch_.begin(), block_force_scratch_.end(), 0.0);

  // velocity Jacobian blocks
  std::fill(block_velocity_previous_configuration_.begin(),
            block_velocity_previous_configuration_.end(), 0.0);
  std::fill(block_velocity_current_configuration_.begin(),
            block_velocity_current_configuration_.end(), 0.0);

  // acceleration Jacobian blocks
  std::fill(block_acceleration_previous_configuration_.begin(),
            block_acceleration_previous_configuration_.end(), 0.0);
  std::fill(block_acceleration_current_configuration_.begin(),
            block_acceleration_current_configuration_.end(), 0.0);
  std::fill(block_acceleration_next_configuration_.begin(),
            block_acceleration_next_configuration_.end(), 0.0);

  // cost
  cost_prior_ = 0.0;
  cost_sensor_ = 0.0;
  cost_force_ = 0.0;
  cost_ = 0.0;

  // cost gradient
  std::fill(cost_gradient_prior_.begin(), cost_gradient_prior_.end(), 0.0);
  std::fill(cost_gradient_sensor_.begin(), cost_gradient_sensor_.end(), 0.0);
  std::fill(cost_gradient_force_.begin(), cost_gradient_force_.end(), 0.0);
  std::fill(cost_gradient_.begin(), cost_gradient_.end(), 0.0);

  // cost Hessian
  std::fill(cost_hessian_prior_.begin(), cost_hessian_prior_.end(), 0.0);
  std::fill(cost_hessian_sensor_.begin(), cost_hessian_sensor_.end(), 0.0);
  std::fill(cost_hessian_force_.begin(), cost_hessian_force_.end(), 0.0);
  std::fill(cost_hessian_.begin(), cost_hessian_.end(), 0.0);
  std::fill(cost_hessian_band_.begin(), cost_hessian_band_.end(), 0.0);
  std::fill(cost_hessian_factor_.begin(), cost_hessian_factor_.end(), 0.0);

  // weight
  std::fill(weight_prior_dense_.begin(), weight_prior_dense_.end(), 0.0);
  std::fill(weight_prior_band_.begin(), weight_prior_band_.end(), 0.0);
  std::fill(weight_prior_update_.begin(), weight_prior_update_.end(), 0.0);

  // norm gradient
  std::fill(norm_gradient_sensor_.begin(), norm_gradient_sensor_.end(), 0.0);
  std::fill(norm_gradient_force_.begin(), norm_gradient_force_.end(), 0.0);

  // norm Hessian
  std::fill(norm_hessian_sensor_.begin(), norm_hessian_sensor_.end(), 0.0);
  std::fill(norm_hessian_force_.begin(), norm_hessian_force_.end(), 0.0);

  std::fill(norm_blocks_sensor_.begin(), norm_blocks_sensor_.end(), 0.0);
  std::fill(norm_blocks_force_.begin(), norm_blocks_force_.end(), 0.0);

  // scratch
  std::fill(scratch0_prior_.begin(), scratch0_prior_.end(), 0.0);
  std::fill(scratch1_prior_.begin(), scratch1_prior_.end(), 0.0);

  std::fill(scratch0_sensor_.begin(), scratch0_sensor_.end(), 0.0);
  std::fill(scratch1_sensor_.begin(), scratch1_sensor_.end(), 0.0);

  std::fill(scratch0_force_.begin(), scratch0_force_.end(), 0.0);
  std::fill(scratch1_force_.begin(), scratch1_force_.end(), 0.0);

  // candidate
  std::fill(configuration_copy_.begin(), configuration_copy_.end(), 0.0);

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // covariance
  std::fill(covariance_.begin(), covariance_.end(), 0.0);
  std::fill(covariance_update_.begin(), covariance_update_.end(), 0.0);
  std::fill(scratch0_covariance_.begin(), scratch0_covariance_.end(), 0.0);
  std::fill(scratch1_covariance_.begin(), scratch1_covariance_.end(), 0.0);
  std::fill(scratch2_covariance_.begin(), scratch2_covariance_.end(), 0.0);

  // timer
  std::fill(timer_prior_step_.begin(), timer_prior_step_.end(), 0.0);
  std::fill(timer_sensor_step_.begin(), timer_sensor_step_.end(), 0.0);
  std::fill(timer_force_step_.begin(), timer_force_step_.end(), 0.0);

  // timing
  ResetTimers();

  // status
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;
}

// prior cost
double Estimator::CostPrior(double* gradient, double* hessian,
                            ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // residual dimension
  int nv = model_->nv;
  int dim = model_->nv * configuration_length_;

  // total scaling
  double scale = scale_prior_ / dim;

  // unpack
  double* r = residual_prior_.data();
  double* P = (band_covariance_ ? weight_prior_band_.data()
                                : weight_prior_dense_.data());
  double* tmp = scratch0_prior_.data();

  // compute cost
  if (band_covariance_) {  // approximate covariance
    // dimensions
    int ntotal = dim;
    int nband = 3 * model_->nv;
    int ndense = 0;

    // multiply: tmp = P * r
    mju_bandMulMatVec(tmp, P, r, ntotal, nband, ndense, 1, true);
  } else {  // exact covariance
    // multiply: tmp = P * r
    mju_mulMatVec(tmp, P, r, dim, dim);
  }

  // weighted quadratic: 0.5 * w * r' * tmp
  double cost = 0.5 * scale * mju_dot(r, tmp, dim);

  // derivatives 
  if (!gradient && !hessian) return cost;

  // loop over estimation horizon
  for (int t = 0; t < configuration_length_; t++) {
    // cost gradient wrt configuration
    if (gradient) {
      // unpack
      double* gt = gradient + t * nv;
      double* block = block_prior_current_configuration_.data() + t * nv * nv;

      // compute
      mju_mulMatTVec(gt, block, tmp + t * nv, nv, nv);

      // scale gradient: w * drdq' * scratch
      mju_scl(gt, gt, scale, nv);
    }

    // cost Hessian wrt configuration (sparse)
    if (hessian && band_covariance_) {
      // number of columns to loop over for row
      int num_cols = mju_min(3, configuration_length_ - t);

      for (int j = t; j < t + num_cols; j++) {
        // shift index
        int shift = 0;  // shift_index(i, j);

        // unpack
        double* bbij = scratch1_prior_.data() + 4 * nv * nv * shift + 0 * nv * nv;
        double* tmp0 = scratch1_prior_.data() + 4 * nv * nv * shift + 1 * nv * nv;
        double* tmp1 = scratch1_prior_.data() + 4 * nv * nv * shift + 2 * nv * nv;
        double* tmp2 = scratch1_prior_.data() + 4 * nv * nv * shift + 3 * nv * nv;

        // get matrices
        BlockFromMatrix(bbij, weight_prior_dense_.data(), nv, nv, dim, dim,
                        t * nv, j * nv);
        const double* bdi =
            block_prior_current_configuration_.data() + nv * nv * t;
        const double* bdj =
            block_prior_current_configuration_.data() + nv * nv * j;

        // -- bdi' * bbij * bdj -- //

        // tmp0 = bbij * bdj
        mju_mulMatMat(tmp0, bbij, bdj, nv, nv, nv);

        // tmp1 = bdi' * tmp0
        mju_mulMatTMat(tmp1, bdi, tmp0, nv, nv, nv);

        // set scaled block in matrix
        SetBlockInMatrix(hessian, tmp1, scale, dim, dim, nv, nv, t * nv,
                          j * nv);
        if (j > t) {
          mju_transpose(tmp2, tmp1, nv, nv);
          SetBlockInMatrix(hessian, tmp2, scale, dim, dim, nv, nv, j * nv,
                            t * nv);
        }
      }
    }
  }
  
  // serial method for dense computation
  if (hessian && !band_covariance_) {
    // unpack
    double* J = jacobian_prior_.data();

    // multiply: scratch = P * drdq
    mju_mulMatMat(tmp, P, J, dim, dim, dim);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, J, tmp, dim, dim, dim);

    // step 3: scale
    mju_scl(hessian, hessian, scale, dim * dim);
  }

  // stop timer
  timer_cost_prior_derivatives_ += GetDuration(start);

  return cost;
}

// prior residual
void Estimator::ResidualPrior(int t) {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // terms
  double* rt = residual_prior_.data() + t * nv;
  double* qt_prior = configuration_prior_.data() + t * nq;
  double* qt = configuration_.data() + t * nq;

  // configuration difference
  mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
}

// prior Jacobian
void Estimator::AssembleJacobianPrior(int t) {
  // dimension
  int nv = model_->nv, dim = model_->nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_prior_.data() + t * nv * dim, nv * dim);

  // unpack
  double* block = block_prior_current_configuration_.data() + t * nv * nv;

  // set block in matrix
  SetBlockInMatrix(jacobian_prior_.data(), block, 1.0, dim, dim, nv, nv, t * nv,
                   t * nv);
}

// prior Jacobian blocks
void Estimator::BlockPrior(int t) {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // unpack
  double* qt = configuration_.data() + t * nq;
  double* qt_prior = configuration_prior_.data() + t * nq;
  double* block = block_prior_current_configuration_.data() + t * nv * nv;

  // compute Jacobian
  DifferentiateDifferentiatePos(NULL, block, model_, 1.0, qt_prior, qt);
}

// prior Jacobian
// note: pool wait is called outside this function
void Estimator::JacobianPrior(ThreadPool& pool) {
  // loop over estimation horizon
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, t]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // block
      estimator.BlockPrior(t);

      // assemble
      if (!estimator.band_covariance_) estimator.AssembleJacobianPrior(t);

      // stop Jacobian timer
      estimator.timer_prior_step_[t] = GetDuration(jacobian_prior_start);
    });
  }
}

// sensor cost
double Estimator::CostSensor(double* gradient, double* hessian,
                             ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int dim_update = model_->nv * configuration_length_;
  int nv = model_->nv;
  int ns = dim_sensor_;

  // ----- cost ----- //

  // initialize
  double cost = 0.0;
  int shift = 0;
  int shift_mat = 0;
  if (gradient) mju_zero(gradient, dim_update);
  if (hessian) mju_zero(hessian, dim_update * dim_update);

  // loop over time steps
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // unpack block
    double* block = block_sensor_configurations_.data() + ns * (3 * nv) * t;

    // sensor shift
    int shift_sensor = 0;

    // loop over sensors
    for (int i = 0; i < model_->nsensor; i++) {
      // dimension
      int nsi = model_->sensor_dim[i];

      // weight
      double weight = weight_sensor_[i];

      // total scaling
      double scale = weight / nsi;

      // ----- cost ----- //
      cost +=
          scale * Norm(gradient ? norm_gradient_sensor_.data() + shift : NULL,
                       hessian ? norm_blocks_sensor_.data() + shift_mat : NULL,
                       residual_sensor_.data() + shift,
                       norm_parameters_sensor_.data() + 3 * i, nsi,
                       norm_sensor_[i]);

      // gradient wrt configuration: dridq012' * dndri
      if (gradient) {
        // sensor block
        double* blocki = block + (3 * nv) * shift_sensor;

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_sensor_.data(), blocki,
                       norm_gradient_sensor_.data() + shift, nsi, 3 * nv);

        // add
        mju_addToScl(gradient + t * nv, scratch0_sensor_.data(), scale, 3 * nv);
      }

      // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
      if (hessian) {
        // sensor block
        double* blocki = block + (3 * nv) * shift_sensor;

        // step 1: tmp0 = d2ndri2 * dridq
        double* tmp0 = scratch0_sensor_.data();
        mju_mulMatMat(tmp0, norm_blocks_sensor_.data() + shift_mat, blocki, nsi,
                      nsi, 3 * nv);

        // step 2: hessian = dridq' * tmp
        double* tmp1 = scratch1_sensor_.data();
        mju_mulMatTMat(tmp1, blocki, tmp0, nsi, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, scale, dim_update, dim_update, 3 * nv,
                         3 * nv, nv * t, nv * t);
      }

      // shift
      shift += nsi;
      shift_mat += nsi * nsi;
      shift_sensor += nsi;
    }
  }

  // stop timer
  timer_cost_sensor_derivatives_ += GetDuration(start);

  return cost;
}

// sensor residual
void Estimator::ResidualSensor(int t) {
  // terms
  double* rt = residual_sensor_.data() + t * dim_sensor_;
  double* yt_sensor = sensor_measurement_.data() + t * dim_sensor_;
  double* yt_model = sensor_prediction_.data() + t * dim_sensor_;

  // sensor difference
  mju_sub(rt, yt_model, yt_sensor, dim_sensor_);
}

// sensor Jacobian
void Estimator::AssembleJacobianSensor(int t) {
  // velocity dimension
  int nv = model_->nv, ns = dim_sensor_;

  // residual dimension
  int dim_residual = ns * (configuration_length_ - 2);

  // update dimension
  int dim_update = nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_sensor_.data() + t * ns * dim_update, ns * dim_update);

  // indices
  int row = t * ns;
  int col_previous = t * nv;
  int col_current = (t + 1) * nv;
  int col_next = (t + 2) * nv;

  // ----- configuration previous ----- //

  // unpack
  double* dsdq0 = block_sensor_previous_configuration_.data() + ns * nv * t;

  // set
  SetBlockInMatrix(jacobian_sensor_.data(), dsdq0, 1.0, dim_residual,
                   dim_update, dim_sensor_, nv, row, col_previous);

  // ----- configuration current ----- //

  // unpack
  double* dsdq1 = block_sensor_current_configuration_.data() + ns * nv * t;

  // set
  SetBlockInMatrix(jacobian_sensor_.data(), dsdq1, 1.0, dim_residual,
                   dim_update, dim_sensor_, nv, row, col_current);

  // ----- configuration next ----- //

  // unpack
  double* dsdq2 = block_sensor_next_configuration_.data() + ns * nv * t;

  // set
  SetBlockInMatrix(jacobian_sensor_.data(), dsdq2, 1.0, dim_residual,
                   dim_update, dim_sensor_, nv, row, col_next);
}

// sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
void Estimator::BlockSensor(int t) {
  // dimensions
  int nv = model_->nv, ns = dim_sensor_;

  // dqds
  double* dqds = block_sensor_configuration_.data() + t * ns * nv;

  // dvds
  double* dvds = block_sensor_velocity_.data() + t * ns * nv;

  // dads
  double* dads = block_sensor_acceleration_.data() + t * ns * nv;

  // -- configuration previous: dsdq0 = dsdv * dvdq0 + dsda * dadq0 -- //

  // unpack
  double* dsdq0 = block_sensor_previous_configuration_.data() + ns * nv * t;
  double* tmp = block_sensor_scratch_.data() + t * ns * nv;

  // dsdq0 <- dvds' * dvdq0
  double* dvdq0 = block_velocity_previous_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(dsdq0, dvds, dvdq0, nv, ns, nv);

  // dsdq0 += dads' * dadq0
  double* dadq0 =
      block_acceleration_previous_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(tmp, dads, dadq0, nv, ns, nv);
  mju_addTo(dsdq0, tmp, ns * nv);

  // -- configuration current: dsdq1 = dsdq + dsdv * dvdq1 + dsda * dadq1 --

  // unpack
  double* dsdq1 = block_sensor_current_configuration_.data() + ns * nv * t;

  // dsdq1 <- dqds'
  mju_transpose(dsdq1, dqds, nv, ns);

  // dsdq1 += dvds' * dvdq1
  double* dvdq1 = block_velocity_current_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(tmp, dvds, dvdq1, nv, ns, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // dsdq1 += dads' * dadq1
  double* dadq1 =
      block_acceleration_current_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(tmp, dads, dadq1, nv, ns, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // -- configuration next: dsdq2 = dsda * dadq2 -- //

  // unpack
  double* dsdq2 = block_sensor_next_configuration_.data() + ns * nv * t;

  // dsdq2 = dads' * dadq2
  double* dadq2 = block_acceleration_next_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(dsdq2, dads, dadq2, nv, ns, nv);

  // -- assemble dsdq012 block -- //

  // unpack
  double* dsdq012 = block_sensor_configurations_.data() + t * ns * (3 * nv);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq0, 1.0, ns, 3 * nv, ns, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dsdq012, dsdq1, 1.0, ns, 3 * nv, ns, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq2, 1.0, ns, 3 * nv, ns, nv, 0, 2 * nv);
}

// sensor Jacobian
// note: pool wait is called outside this function
void Estimator::JacobianSensor(ThreadPool& pool) {
  // loop over estimation horizon
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, t]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // block
      estimator.BlockSensor(t);

      // assemble
      if (!estimator.band_covariance_) estimator.AssembleJacobianSensor(t);

      // stop Jacobian timer
      estimator.timer_sensor_step_[t] = GetDuration(jacobian_sensor_start);
    });
  }
}

// force cost
double Estimator::CostForce(double* gradient, double* hessian,
                            ThreadPool& pool) {
  // start derivative timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int dim_update = model_->nv * configuration_length_;
  int nv = model_->nv;

  // initialize
  double cost = 0.0;
  int shift = 0;
  int shift_mat = 0;
  int dof;
  if (gradient) mju_zero(gradient, dim_update);
  if (hessian) mju_zero(hessian, dim_update * dim_update);

  // loop over time steps
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // unpack block
    double* block = block_force_configurations_.data() + nv * (3 * nv) * t;

    // shift by joint
    int shift_joint = 0;

    // loop over joints
    for (int i = 0; i < model_->njnt; i++) {
      // joint type
      int jnt_type = model_->jnt_type[i];

      // dof
      if (jnt_type == mjJNT_FREE) {
        dof = 6;
      } else if (jnt_type == mjJNT_BALL) {
        dof = 3;
      } else {  // jnt_type == mjJNT_SLIDE | mjJNT_HINGE
        dof = 1;
      }

      // weight
      double weight = weight_force_[jnt_type];

      // total scaling
      double scale = weight / dof * model_->opt.timestep;

      // norm
      NormType norm = norm_force_[jnt_type];

      // add weighted norm
      cost +=
          scale * Norm(gradient ? norm_gradient_force_.data() + shift : NULL,
                       hessian ? norm_blocks_force_.data() + shift_mat : NULL,
                       residual_force_.data() + shift,
                       norm_parameters_force_[jnt_type], dof, norm);

      // gradient wrt configuration: dridq012' * dndri
      if (gradient) {
        // joint block
        double* blocki = block + (3 * nv) * shift_joint;

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_force_.data(), blocki,
                       norm_gradient_force_.data() + shift, dof, 3 * nv);

        // add
        mju_addToScl(gradient + t * nv, scratch0_force_.data(), scale, 3 * nv);
      }

      // Hessian (Gauss-Newton) wrt configuration: drdq * d2ndr2 * drdq
      if (hessian) {
        // joint block
        double* blocki = block + (3 * nv) * shift_joint;

        // step 1: tmp0 = d2ndri2 * dridq012
        double* tmp0 = scratch0_force_.data();
        mju_mulMatMat(tmp0, norm_blocks_force_.data() + shift_mat, blocki, dof,
                      dof, 3 * nv);

        // step 2: tmp1 = dridq' * tmp0
        double* tmp1 = scratch1_force_.data();
        mju_mulMatTMat(tmp1, blocki, tmp0, dof, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, scale, dim_update, dim_update, 3 * nv,
                         3 * nv, nv * t, nv * t);
      }

      // shift
      shift += dof;
      shift_mat += dof * dof;
      shift_joint += dof;
    }
  }

  // stop timer
  timer_cost_force_derivatives_ += GetDuration(start);

  return cost;
}

// force residual
void Estimator::ResidualForce(int t) {
  // dimension
  int nv = model_->nv;

  // terms
  double* rt = residual_force_.data() + t * nv;
  double* ft_actuator = force_measurement_.data() + t * nv;
  double* ft_inverse_ = force_prediction_.data() + t * nv;

  // force difference
  mju_sub(rt, ft_inverse_, ft_actuator, nv);
}

// force Jacobian
void Estimator::AssembleJacobianForce(int t) {
  // velocity dimension
  int nv = model_->nv;

  // residual dimension
  int dim_residual = nv * (configuration_length_ - 2);

  // update dimension
  int dim_update = nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_force_.data() + t * nv * dim_update, nv * dim_update);

  // indices
  int row = t * nv;
  int col_previous = t * nv;
  int col_current = (t + 1) * nv;
  int col_next = (t + 2) * nv;

  // ----- configuration previous ----- //
  // unpack
  double* dfdq0 = block_force_previous_configuration_.data() + nv * nv * t;

  // set
  SetBlockInMatrix(jacobian_force_.data(), dfdq0, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_previous);

  // ----- configuration current ----- //

  // unpack
  double* dfdq1 = block_force_current_configuration_.data() + nv * nv * t;

  // set
  SetBlockInMatrix(jacobian_force_.data(), dfdq1, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_current);

  // ----- configuration next ----- //

  // unpack
  double* dfdq2 = block_force_next_configuration_.data() + nv * nv * t;

  // set
  AddBlockInMatrix(jacobian_force_.data(), dfdq2, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_next);
}

// force Jacobian (dfdq0, dfdq1, dfdq2)
void Estimator::BlockForce(int t) {
  // velocity dimension
  int nv = model_->nv;

  // dqdf
  double* dqdf = block_force_configuration_.data() + t * nv * nv;

  // dvdf
  double* dvdf = block_force_velocity_.data() + t * nv * nv;

  // dadf
  double* dadf = block_force_acceleration_.data() + t * nv * nv;

  // -- configuration previous: dfdq0 = dfdv * dvdq0 + dfda * dadq0 -- //

  // unpack
  double* dfdq0 = block_force_previous_configuration_.data() + t * nv * nv;
  double* tmp = block_force_scratch_.data() + t * nv * nv;

  // dfdq0 <- dvdf' * dvdq0
  double* dvdq0 = block_velocity_previous_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(dfdq0, dvdf, dvdq0, nv, nv, nv);

  // dfdq0 += dadf' * dadq0
  double* dadq0 =
      block_acceleration_previous_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(tmp, dadf, dadq0, nv, nv, nv);
  mju_addTo(dfdq0, tmp, nv * nv);

  // -- configuration current: dfdq1 = dfdq + dfdv * dvdq1 + dfda * dadq1 -- //

  // unpack
  double* dfdq1 = block_force_current_configuration_.data() + nv * nv * t;

  // dfdq1 <- dqdf'
  mju_transpose(dfdq1, dqdf, nv, nv);

  // dfdq1 += dvdf' * dvdq1
  double* dvdq1 = block_velocity_current_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(tmp, dvdf, dvdq1, nv, nv, nv);
  mju_addTo(dfdq1, tmp, nv * nv);

  // dfdq1 += dadf' * dadq1
  double* dadq1 =
      block_acceleration_current_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(tmp, dadf, dadq1, nv, nv, nv);
  mju_addTo(dfdq1, tmp, nv * nv);

  // -- configuration next: dfdq2 = dfda * dadq2 -- //

  // unpack
  double* dfdq2 = block_force_next_configuration_.data() + nv * nv * t;

  // dfdq2 <- dadf' * dadq2
  double* dadq2 = block_acceleration_next_configuration_.data() + t * nv * nv;
  mju_mulMatTMat(dfdq2, dadf, dadq2, nv, nv, nv);

  // -- assemble dfdq012 block -- //

  // unpack
  double* dfdq012 = block_force_configurations_.data() + t * nv * (3 * nv);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq0, 1.0, nv, 3 * nv, nv, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dfdq012, dfdq1, 1.0, nv, 3 * nv, nv, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq2, 1.0, nv, 3 * nv, nv, nv, 0, 2 * nv);
}

// force Jacobian
void Estimator::JacobianForce(ThreadPool& pool) {
  // loop over estimation horizon
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, t]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // block
      estimator.BlockForce(t);

      // assemble
      if (!estimator.band_covariance_) estimator.AssembleJacobianForce(t);

      // stop Jacobian timer
      estimator.timer_force_step_[t] = GetDuration(jacobian_force_start);
    });
  }
}

// compute force
void Estimator::InverseDynamicsPrediction(int t) {
  // dimension
  int nq = model_->nq, nv = model_->nv, ns = dim_sensor_;

  // terms
  double* qt = configuration_.data() + (t + 1) * nq;
  double* vt = velocity_.data() + t * nv;
  double* at = acceleration_.data() + t * nv;

  // data
  mjData* d = data_[0].get();

  // set qt, vt, at
  mju_copy(d->qpos, qt, nq);
  mju_copy(d->qvel, vt, nv);
  mju_copy(d->qacc, at, nv);

  // inverse dynamics
  mj_inverse(model_, d);

  // copy sensor
  double* st = sensor_prediction_.data() + t * ns;
  mju_copy(st, d->sensordata, ns);

  // copy force
  double* ft = force_prediction_.data() + t * nv;
  mju_copy(ft, d->qfrc_inverse, nv);
}

// compute inverse dynamics derivatives (via finite difference)
void Estimator::InverseDynamicsDerivatives(ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model_->nq, nv = model_->nv, ns = dim_sensor_;

  // pool count
  int count_before = pool.GetCount();

  // loop over estimation horizon
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // schedule
    pool.Schedule([&estimator = *this, nq, nv, ns, t]() {
      // unpack
      double* q = estimator.configuration_.data() + (t + 1) * nq;
      double* v = estimator.velocity_.data() + t * nv;
      double* a = estimator.acceleration_.data() + t * nv;
      double* dqds = estimator.block_sensor_configuration_.data() + t * ns * nv;
      double* dvds = estimator.block_sensor_velocity_.data() + t * ns * nv;
      double* dads = estimator.block_sensor_acceleration_.data() + t * ns * nv;
      double* dqdf = estimator.block_force_configuration_.data() + t * nv * nv;
      double* dvdf = estimator.block_force_velocity_.data() + t * nv * nv;
      double* dadf = estimator.block_force_acceleration_.data() + t * nv * nv;
      mjData* data = estimator.data_[t].get();  // TODO(taylor): WorkerID

      // set (state, acceleration)
      mju_copy(data->qpos, q, nq);
      mju_copy(data->qvel, v, nv);
      mju_copy(data->qacc, a, nv);

      // finite-difference derivatives
      mjd_inverseFD(estimator.model_, data,
                    estimator.finite_difference_.tolerance,
                    estimator.finite_difference_.flg_actuation, dqdf, dvdf,
                    dadf, dqds, dvds, dads, NULL);
    });
  }

  // wait
  pool.WaitCount(count_before + configuration_length_ - 2);

  // reset pool count
  pool.ResetCount();

  // stop timer
  timer_inverse_dynamics_derivatives_ += GetDuration(start);
}

// update configuration trajectory
void Estimator::UpdateConfiguration(double* candidate,
                                    const double* configuration,
                                    const double* search_direction,
                                    double step_size) {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // copy configuration to candidate
  mju_copy(candidate, configuration, nq * configuration_length_);

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // configuration
    double* q = candidate + t * nq;

    // search direction
    const double* dq = search_direction + t * nv;

    // integrate
    mj_integratePos(model_, q, dq, step_size);
  }
}

// convert sequence of configurations to velocities and accelerations
void Estimator::ConfigurationToVelocityAcceleration(int t) {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // previous and current configurations
  const double* q0 = configuration_.data() + t * nq;
  const double* q1 = configuration_.data() + (t + 1) * nq;

  // compute velocity
  double* v1 = velocity_.data() + t * nv;
  mj_differentiatePos(model_, v1, model_->opt.timestep, q0, q1);

  // compute acceleration
  if (t > 0) {
    // previous velocity
    const double* v0 = velocity_.data() + (t - 1) * nv;

    // compute acceleration
    double* a1 = acceleration_.data() + (t - 1) * nv;
    mju_sub(a1, v1, v0, nv);
    mju_scl(a1, a1, 1.0 / model_->opt.timestep, nv);
  }
}

// compute finite-difference velocity, acceleration derivatives
// TODO(taylor): benchmark pool v. no pool
void Estimator::VelocityAccelerationDerivatives(ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over estimation horizon
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // unpack
    double* q1 = configuration_.data() + t * nq;
    double* q2 = configuration_.data() + (t + 1) * nq;
    double* dv2dq1 =
        block_velocity_previous_configuration_.data() + t * nv * nv;
    double* dv2dq2 = block_velocity_current_configuration_.data() + t * nv * nv;

    // compute velocity Jacobians
    DifferentiateDifferentiatePos(dv2dq1, dv2dq2, model_, model_->opt.timestep,
                                  q1, q2);

    // compute acceleration Jacobians
    if (t > 0) {
      // unpack
      double* dadq0 =
          block_acceleration_previous_configuration_.data() + (t - 1) * nv * nv;
      double* dadq1 =
          block_acceleration_current_configuration_.data() + (t - 1) * nv * nv;
      double* dadq2 =
          block_acceleration_next_configuration_.data() + (t - 1) * nv * nv;

      // previous velocity Jacobians
      double* dv1dq0 =
          block_velocity_previous_configuration_.data() + (t - 1) * nv * nv;
      double* dv1dq1 =
          block_velocity_current_configuration_.data() + (t - 1) * nv * nv;

      // dadq0 = -dv1dq0 / h
      mju_copy(dadq0, dv1dq0, nv * nv);
      mju_scl(dadq0, dadq0, -1.0 / model_->opt.timestep, nv * nv);

      // dadq1 = dv2dq1 / h - dv1dq1 / h = (dv2dq1 - dv1dq1) / h
      mju_sub(dadq1, dv2dq1, dv1dq1, nv * nv);
      mju_scl(dadq1, dadq1, 1.0 / model_->opt.timestep, nv * nv);

      // dadq2 = dv2dq2 / h
      mju_copy(dadq2, dv2dq2, nv * nv);
      mju_scl(dadq2, dadq2, 1.0 / model_->opt.timestep, nv * nv);
    }
  }

  // stop timer
  timer_velacc_derivatives_ += GetDuration(start);
}

// compute total cost
double Estimator::Cost(ThreadPool& pool) {
  // ----- trajectories ----- //

  // finite-difference velocities, accelerations
  for (int t = 0; t < configuration_length_ - 1; t++) {
    ConfigurationToVelocityAcceleration(t);
  }

  // compute sensor and force predictions
  for (int t = 0; t < configuration_length_ - 2; t++) {
    InverseDynamicsPrediction(t);
  }

  // residuals
  for (int t = 0; t < configuration_length_; t++) {
    // prior
    if (prior_flag_) ResidualPrior(t);

    // skip
    if (t >= configuration_length_ - 2) continue;

    // sensor
    if (sensor_flag_) ResidualSensor(t);

    // force
    if (force_flag_) ResidualForce(t);
  }

  // prior
  cost_prior_ = (prior_flag_ ? CostPrior(NULL, NULL, pool) : 0.0);

  // sensor
  cost_sensor_ = (sensor_flag_ ? CostSensor(NULL, NULL, pool) : 0.0);

  // force
  cost_force_ = (force_flag_ ? CostForce(NULL, NULL, pool) : 0.0);

  // total cost
  return cost_prior_ + cost_sensor_ + cost_force_;
}

// prior update
void Estimator::PriorUpdate() {
  // start timer
  auto prior_update_start = std::chrono::steady_clock::now();

  // dimension
  int dim = model_->nv * configuration_length_;

  if (prior_reset_) {
    // set initial covariance, weight
    double* covariance = covariance_.data();
    double* weight = weight_prior_dense_.data();
    mju_eye(covariance, dim);
    mju_eye(weight, dim);
    for (int i = 0; i < dim; i++) {
      covariance[i * dim + i] *= covariance_initial_scaling_;
      weight[i * dim + i] *= covariance_initial_scaling_;
    }

    // approximate covariance
    if (band_covariance_) {
      int ntotal = dim;
      int nband = 3 * model_->nv;
      int ndense = 0;
      mju_dense2Band(weight_prior_band_.data(), weight_prior_dense_.data(),
                     ntotal, nband, ndense);
    }
  } else {  // TODO(taylor): shift + utilize inverse Hessian
  }

  // stop timer
  timer_prior_update_ = GetDuration(prior_update_start);
}

// compute total gradient
void Estimator::CostGradient() {
  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model_->nv;

  // unpack
  double* gradient = cost_gradient_.data();

  // individual gradients
  if (prior_flag_) {
    mju_copy(gradient, cost_gradient_prior_.data(), dim);
  } else {
    mju_zero(gradient, dim);
  }
  if (sensor_flag_) mju_addTo(gradient, cost_gradient_sensor_.data(), dim);
  if (force_flag_) mju_addTo(gradient, cost_gradient_force_.data(), dim);

  // stop gradient timer
  timer_cost_gradient_ += GetDuration(start);
}

// compute total Hessian
void Estimator::CostHessian() {
  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model_->nv;

  // unpack
  double* hessian = cost_hessian_.data();

  // individual Hessians
  if (prior_flag_) {
    mju_copy(hessian, cost_hessian_prior_.data(), dim * dim);
  } else {
    mju_zero(hessian, dim * dim);
  }
  if (sensor_flag_) mju_addTo(hessian, cost_hessian_sensor_.data(), dim * dim);
  if (force_flag_) mju_addTo(hessian, cost_hessian_force_.data(), dim * dim);

   // stop Hessian timer
  timer_cost_hessian_ += GetDuration(start);
}

// covariance update
void Estimator::CovarianceUpdate(ThreadPool& pool) {
  // update = covariance - covariance * hessian * covariance'

  // dimension
  int dim = model_->nv * configuration_length_;

  // unpack
  double* covariance = covariance_.data();
  double* update = covariance_update_.data();

  // -- tmp0 = covariance * hessian -- //

  // unpack
  double* tmp0 = scratch0_covariance_.data();
  double* tmp1 = scratch1_covariance_.data();
  double* tmp2 = scratch2_covariance_.data();

  // select solver
  if (band_covariance_) {
    int ntotal = dim;
    int nband = 3 * model_->nv;
    int ndense = 0;

    // -- tmp0 = mul(hessian, covariance) -- //

    // pool count
    int count_before = pool.GetCount();

    // loop over covariance
    for (int i = 0; i < ntotal; i++) {
      pool.Schedule(
          [&estimator = *this, &tmp0, &covariance, ntotal, nband, ndense, i]() {
            mju_bandMulMatVec(
                tmp0 + ntotal * i, estimator.cost_hessian_band_.data(),
                covariance + ntotal * i, ntotal, nband, ndense, 1, true);
          });
    }

    // wait
    pool.WaitCount(count_before + ntotal);

    // reset pool count
    pool.ResetCount();

    // band covariance
    mju_dense2Band(tmp2, covariance, ntotal, nband, ndense);

    // -- tmp1 = mul(covariance, mul(hessian, covariance)) -- //

    // pool count
    count_before = pool.GetCount();

    // loop over tmp0
    for (int i = 0; i < ntotal; i++) {
      pool.Schedule([&tmp0, &tmp1, &tmp2, ntotal, nband, ndense, i]() {
        mju_bandMulMatVec(tmp1 + ntotal * i, tmp2, tmp0 + ntotal * i, ntotal,
                          nband, ndense, 1, true);
      });
    }

    // wait
    pool.WaitCount(count_before + ntotal);

    // reset pool count
    pool.ResetCount();

  } else {  // dense
    // tmp0 = hessian * covariance'
    mju_mulMatMatT(tmp0, cost_hessian_.data(), covariance, dim, dim, dim);

    // tmp1 = covariance * tmp0
    mju_mulMatMat(tmp1, covariance, tmp0, dim, dim, dim);
  }

  // update = covariance - tmp1
  mju_sub(update, covariance, tmp1, dim * dim);

  // // -- update covariance -- //

  // // start timer
  // auto covariance_update_start = std::chrono::steady_clock::now();

  // // update covariance
  // CovarianceUpdate(pool);

  // // factorize covariance
  // double* factor = scratch0_covariance_.data();

  // if (band_covariance_) {
  //   // convert
  //   mju_dense2Band(factor, covariance_update_.data(), ntotal, nband, ndense);

  //   // factorize
  //   mju_cholFactorBand(factor, ntotal, nband, ndense, 0.0, 0.0);
  // } else {
  //   // copy
  //   mju_copy(factor, covariance_update_.data(), nvar);

  //   // factorize
  //   mju_cholFactor(factor, nvar, 0.0);
  // }

  // // update prior weight
  // double* PT = weight_prior_update_.data();
  // double* In = scratch1_covariance_.data();
  // mju_eye(In, nvar);

  // // -- P^T = L^-T L^-1 -- //

  // // pool count
  // int count_begin = pool.GetCount();

  // // loop
  // for (int i = 0; i < nvar; i++) {
  //   pool.Schedule([&PT, &factor, &In, nvar, ntotal, nband, ndense, i]() {
  //     mju_cholSolveBand(PT + i * nvar, factor, In, ntotal, nband, ndense);
  //   });
  // }

  // // wait
  // pool.WaitCount(count_begin + nvar);

  // // reset count
  // pool.ResetCount();

  // // update
  // mju_copy(covariance_.data(), covariance_update_.data(), nvar * nvar);
}

// optimize trajectory estimate
void Estimator::Optimize(ThreadPool& pool) {
  // TODO(taylor): if configuration_length_ changes

  // dimensions
  int nconfig = model_->nq * configuration_length_;
  int nvar = model_->nv * configuration_length_;

  // operations
  int nprior = prior_flag_ * configuration_length_;
  int nsensor = sensor_flag_ * (configuration_length_ - 2);
  int nforce = force_flag_ * (configuration_length_ - 2);

  // band dimensions
  int ntotal = model_->nv * configuration_length_;
  int nband = 3 * model_->nv;
  int ndense = 0;

  // reset timers
  ResetTimers();

  // prior update
  PriorUpdate();

  // initial cost
  cost_ = Cost(pool);

  // print initial cost
  PrintCost();

  // ----- smoother iterations ----- //

  // reset
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;

  // iterations
  for (; iterations_smoother_ < max_smoother_iterations_;
       iterations_smoother_++) {
    // ----- cost derivatives ----- //

    // start timer (total cost derivatives)
    auto cost_derivatives_start = std::chrono::steady_clock::now();

    // inverse dynamics derivatives
    InverseDynamicsDerivatives(pool);

    // velocity, acceleration derivatives
    VelocityAccelerationDerivatives(pool);

    // -- Jacobians -- //

    // pool count
    int count_begin = pool.GetCount();

    // individual derivatives
    if (prior_flag_) JacobianPrior(pool);
    if (sensor_flag_) JacobianSensor(pool);
    if (force_flag_) JacobianForce(pool);
    
    // wait
    pool.WaitCount(count_begin + nprior + nsensor + nforce);

    // reset count
    pool.ResetCount();

    // timers
    timer_jacobian_prior_ += mju_sum(timer_prior_step_.data(), nprior);
    timer_jacobian_sensor_ += mju_sum(timer_sensor_step_.data(), nsensor);
    timer_jacobian_force_ += mju_sum(timer_force_step_.data(), nforce);

    // -- cost derivatives -- //

    // pool count
    count_begin = pool.GetCount();

    // prior derivatives
    if (prior_flag_)
      CostPrior(cost_gradient_prior_.data(), cost_hessian_prior_.data(), pool);

    // sensor derivatives
    if (sensor_flag_)
      CostSensor(cost_gradient_sensor_.data(), cost_hessian_sensor_.data(),
                 pool);

    // force derivatives
    if (force_flag_)
      CostForce(cost_gradient_force_.data(), cost_hessian_force_.data(), pool);

    // wait
    // pool.WaitCount(count_begin + prior_flag_ + sensor_flag_ + force_flag_);

    // pool reset
    pool.ResetCount();

    // -- gradient -- // 

    // unpack
    double* gradient = cost_gradient_.data();
    
    // assemble total gradient
    CostGradient();

    // gradient tolerance check
    double gradient_norm = mju_norm(gradient, nvar) / nvar;
    if (gradient_norm < gradient_tolerance_) break;

    // -- Hessian -- // 

    // unpack
    double* hessian = cost_hessian_.data();
    
    // assemble total Hessian
    CostHessian();

    // stop timer
    timer_cost_derivatives_ += GetDuration(cost_derivatives_start);

    // ----- search direction ----- //

    // start timer
    auto search_direction_start = std::chrono::steady_clock::now();

    // unpack
    double* dq = search_direction_.data();

    // regularize TODO(taylor): LM reg.
    for (int j = 0; j < nvar; j++) {
      hessian[j * nvar + j] += 1.0e-3;
    }

    // -- band Hessian -- //

    // unpack
    double* hessian_band = cost_hessian_band_.data();

    // convert
    if (band_covariance_) {  // band solver
      // dense to banded
      mju_dense2Band(hessian_band, cost_hessian_.data(), ntotal, nband, ndense);
    }

    // -- linear system solver -- //

    // unpack factor
    double* factor = cost_hessian_factor_.data();

    // select solver
    if (band_covariance_) {  // band solver
      // factorize
      mju_copy(factor, hessian_band, ntotal * nband);
      mju_cholFactorBand(factor, ntotal, nband, ndense, 0.0, 0.0);

      // compute search direction
      mju_cholSolveBand(dq, factor, gradient, ntotal, nband, ndense);
    } else {  // dense solver
      // factorize
      mju_copy(factor, hessian, nvar * nvar);
      mju_cholFactor(factor, nvar, 0.0);

      // compute search direction
      mju_cholSolve(dq, factor, gradient, nvar);
    }

    // set prior reset flag
    if (prior_reset_) {
      prior_reset_ = false;
    }

    // end timer
    timer_search_direction_ += GetDuration(search_direction_start);

    // ----- line search ----- //
    // TODO(taylor): option for curve search

    // start timer
    auto line_search_start = std::chrono::steady_clock::now();

    // copy configuration
    mju_copy(configuration_copy_.data(), configuration_.data(), nconfig);

    // initialize
    double cost_candidate = cost_;
    int iteration_line_search = 0;
    double step_size = 2.0;

    // backtracking until cost decrease
    // TODO(taylor): Armijo, Wolfe conditions
    while (cost_candidate >= cost_) {
      // check for max iterations
      if (iteration_line_search > max_line_search_) {
        // reset configuration
        mju_copy(configuration_.data(), configuration_copy_.data(), nconfig);

        return;
        // mju_error("Batch Estimator: Line search failure\n");
      }

      // decrease cost
      step_size *= 0.5;  // TODO(taylor): log schedule

      // candidate
      UpdateConfiguration(configuration_.data(), configuration_copy_.data(), dq,
                          -1.0 * step_size);

      // cost
      cost_candidate = Cost(pool);

      // update iteration
      iteration_line_search++;
    }
    // increment
    iterations_line_search_ += iteration_line_search;

    // end timer
    timer_line_search_ += GetDuration(line_search_start);

    // update cost
    cost_ = cost_candidate;

    // print cost
    PrintCost();
  }

  // ----- GUI update ----- //
  // update costs
  // set timers
  // set status

  // status
  PrintStatus();
}

// print status
void Estimator::PrintStatus() {
  if (!verbose_status_) return;

  // title
  printf("Batch Estimator Status:\n\n");

  // timing
  printf("Timing:\n");
  printf("  covariance: %.5f (ms) \n", 1.0e-3 * timer_prior_update_);
  printf("  inverse dynamics derivatives: %.5f (ms) \n",
         1.0e-3 * timer_inverse_dynamics_derivatives_);
  printf("  velacc derivatives: %.5f (ms) \n",
         1.0e-3 * timer_velacc_derivatives_);
  printf("  jacobian prior: %.5f (ms) \n", 1.0e-3 * timer_jacobian_prior_);
  printf("  jacobian sensor: %.5f (ms) \n", 1.0e-3 * timer_jacobian_sensor_);
  printf("  jacobian force: %.5f (ms) \n", 1.0e-3 * timer_jacobian_force_);
  printf("  cost prior derivatives: %.5f (ms) \n",
         1.0e-3 * timer_cost_prior_derivatives_);
  printf("  cost sensor derivatives: %.5f (ms) \n",
         1.0e-3 * timer_cost_sensor_derivatives_);
  printf("  cost force derivatives: %.5f (ms) \n",
         1.0e-3 * timer_cost_force_derivatives_);
  printf("  cost gradient: %.5f (ms) \n", 1.0e-3 * timer_cost_gradient_);
  printf("  cost hessian: %.5f (ms) \n", 1.0e-3 * timer_cost_hessian_);
  printf("  search direction: %.5f (ms) \n", 1.0e-3 * timer_search_direction_);
  printf("  covariance update: %.5f (ms) \n",
         1.0e-3 * timer_covariance_update_);
  printf("  line search: %.5f (ms) \n", 1.0e-3 * timer_line_search_);
  printf("  TOTAL: %.5f (ms) \n",
         1.0e-3 * (timer_cost_derivatives_ + timer_search_direction_ +
                   timer_covariance_update_ + timer_line_search_));
  printf("\n");

  // status
  printf("Status:\n");
  printf("  iterations line search: %i\n", iterations_line_search_);
  printf("  iterations smoother: %i\n", iterations_smoother_);
}

// print cost
void Estimator::PrintCost() {
  if (verbose_cost_) {
    printf("cost (total): %.5f\n", cost_);
    printf("  prior: %.5f\n", cost_prior_);
    printf("  sensor: %.5f\n", cost_sensor_);
    printf("  force: %.5f\n", cost_force_);
  }
}

// reset timers
void Estimator::ResetTimers() {
  timer_prior_update_ = 0.0;
  timer_inverse_dynamics_derivatives_ = 0.0;
  timer_velacc_derivatives_ = 0.0;
  timer_jacobian_prior_ = 0.0;
  timer_jacobian_sensor_ = 0.0;
  timer_jacobian_force_ = 0.0;
  timer_cost_prior_derivatives_ = 0.0;
  timer_cost_sensor_derivatives_ = 0.0;
  timer_cost_force_derivatives_ = 0.0;
  timer_cost_gradient_ = 0.0;
  timer_cost_hessian_ = 0.0;
  timer_cost_derivatives_ = 0.0;
  timer_covariance_update_ = 0.0;
  timer_search_direction_ = 0.0;
  timer_line_search_ = 0.0;
}

}  // namespace mjpc
