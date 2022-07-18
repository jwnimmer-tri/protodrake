/*
 * TODO: This license is not consistent with license used in the project.
 *       Delete the inconsistent license and above line and rerun pre-commit to
 * insert a good license. Copyright © 2022 Dexai Robotics. All rights reserved.
 */

// drake headers

#pragma once
#include <drake/common/polynomial.h>
#include <drake/solvers/constraint.h>

#include "drake/common/autodiff.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/math/autodiff_gradient.h"

#include <drake/solvers/constraint.h>

#include <easy/profiler.h>



// dexai headers
// #include "constraint_solver.h"
#include "drac_types.h"
#include "constraint_solver.h"
#include "parameters.h"

using PPType_ad = drake::trajectories::PiecewisePolynomial<drake::AutoDiffXd>;
using WAYPTS_ad = std::vector<drake::MatrixX<drake::AutoDiffXd>>;
using params = parameters::Parameters;

// 
// template <typename T>
// class JointLimitChecker {
//  public:
//   // JointLimitCost();
//   JointLimitChecker(const drake::VectorX<double>& lower_limit,
//                     const drake::VectorX<double>& upper_limit);
//   void Update(T distance, size_t i);

//   inline bool AreLimitsSatisfied() {
//     return *min_distance > 0;
//   }
//   inline T* GetCost() {
//     return cost.get();
//   }
//   inline T GetMinDistance() {
//     return *min_distance;
//   }

//  private:
//   const drake::VectorX<double> lower_limit_;
//   const drake::VectorX<double> upper_limit_;
//   std::unique_ptr<T> cost = std::make_unique<T>(static_cast<T>(0));
//   // T* cost_ = static_cast<T*>(0);
//   // This is ony for verification and user information purposes
//   std::unique_ptr<T> min_distance =
//       std::make_unique<T>(static_cast<T>(std::numeric_limits<double>::max()));
//   // T* min_distance = static_cast<T*>(std::numeric_limits<double>::max());
//   double joint_limit_threshold {0.05};
// };

// This object is responsible for encoding constraints.
template <typename T>
class ConstraintChecker {
  /**
   * @brief This class is responsible for encoding constraints.
   *
   * @param min_distance_ is the minimum margin within all constraints.
   * @param cost_ is the cost function for this constraint.

   */
 private:
  T cost_;
  T min_distance_;
  std::vector<T> distances_ {};

  const std::string name_;
  double activation_threshold_;
  double repel_threshold_;

 public:
  ConstraintChecker(const std::string& name, double activation_threshold = 0.015,
                    double repel_threshold = 0.0);

  void Push(T distance) {
    if (distance < min_distance_) {
      min_distance_ = distance;
    }
    distances_.push_back(distance);
  }

  inline void Reset() {
    distances_.clear();
    cost_ = static_cast<T>(0);
    min_distance_ = std::numeric_limits<T>::max();
    std::cout<<"Reset:: repel_threshold_ = "<< repel_threshold_ << std::endl;
  };

  inline void ResetHard() {
    distances_.clear();
    cost_ = static_cast<T>(0);
    min_distance_ = std::numeric_limits<T>::max();
    repel_threshold_ = 0.0;
  };

  inline T GetCost() {
    return cost_;
  }

  inline T GetMinimumDistance() {
    return min_distance_;
  }

  inline void SetInfinityCost() {
    cost_ = std::numeric_limits<T>::max();
  }

  void Evaluate();

  // Public Parameters
  double activation_threshold() const {
    return activation_threshold_;
  }
  double repel_threshold() const {
    return repel_threshold_;
  }
  inline void set_repel_threshold(double repel_threshold) {
    repel_threshold_ = repel_threshold;
  }
};

// To come soon!
// template<typename T>
// T CalcSplineTime(drake::trajectories::PiecewisePolynomial<T>& path, const
// Eigen::VectorXd& vlim, const Eigen::VectorXd& alim);

/* SplineOptimizer
This class acts as an optimizer
 */
class SplineOptimizer {
 public:
  // Constructor
  SplineOptimizer();

  SplineOptimizer(std::vector<drake::MatrixX<double>>& waypts);

  SplineOptimizer(std::vector<drake::MatrixX<double>>& waypts,
                  std::shared_ptr<ConstraintSolver> cs,
                  const params& parameters);

  SplineOptimizer(const system_poly_t& syspoly, std::shared_ptr<ConstraintSolver> cs, const params& parameters, int n_waypts=10);

  /**
   @brief Computes ...
   The method is based on checking the extremum points of 3rd degree
   piecewise polynomial. Therefore, it is exact.
  */
  template <typename T>
  void CalcBoxLimitsCost(drake::trajectories::PiecewisePolynomial<T>& spline,
                         Eigen::VectorXd& upper_limit,
                         Eigen::VectorXd& lower_limit);

  template <typename T>
  void CalcJointLimitsCost(drake::trajectories::PiecewisePolynomial<T>& spline);

  // destructor
  ~SplineOptimizer() {
    std::cout << "SplineOptimizer Closed" << std::endl;
  }

  template <typename T>
  void CalcCollisionCost(drake::trajectories::PiecewisePolynomial<T>& spline,
                         int n_points = 100);

  
  // drake::solvers::LinearConstraint AddProximityConstraint(system_conf_t sysconf);

  // void ProjectPoint(FeasibleSet);

  template <typename T>
  void CalcCollisionCost(std::vector<drake::MatrixX<T>>& waypts_vec);

  void AddAnchorPoint();
  
  // Optimizes the spline. 
  void Optimize(int max_iterations = 30);

// Uses Backtracking line search to select the gradient step size
// Look at, e.g., https://en.wikipedia.org/wiki/Backtracking_line_search
double BackTracking(drake::AutoDiffXd current_cost, double step,
                      double alpha = 0.5, double c = 0.5);

// Implements one step of the optimization algorithm
void DescentOneStep();

public: 

template<typename T>
T CalcSplineTime(drake::trajectories::PiecewisePolynomial<T>& spline);

template <typename T>
T CalcApproximateSplineTime(drake::trajectories::PiecewisePolynomial<T>& spline);

template <typename T>
T CalcApproximateSplineTime(drake::trajectories::PiecewisePolynomial<T>& spline, const Eigen::VectorXd& vlim,
                                  const Eigen::VectorXd& alim);

private:
template<typename T>
T CalcApproximateMultiRobotSplineTime(drake::trajectories::PiecewisePolynomial<T>& spline);

public:
// Helper function for getting a pointer to the current spline
  inline PPType_ad* GetSpline() {
    return &spline_;
  }

 public:
  robot_state_vec_t GetPlan();
  std::pair<std::vector<double>, std::vector<double>> min_distance_trajectory_;

  // Hyper parameters
  double collision_cutoff {0.02};
  double jointlimit_cutoff {0.02};
  bool convergence_ {false};

 private:
  // Initialized with class constructor
  std::vector<drake::MatrixX<double>> waypts_;
  std::shared_ptr<ConstraintSolver> cs_;  // shared ptr
  size_t dim_;                      // dimension of the spline space
  params parameters_;
  std::unique_ptr<ConstraintChecker<drake::AutoDiffXd>> collision_checker_ptr_,
      jointlimit_checker_ptr_;  // owned ptrs for autodiff spline
  std::unique_ptr<ConstraintChecker<double>> collision_checker_temp_ptr_,
      jointlimit_checker_temp_ptr_;  // owned ptrs for double spline
  const double collision_check_threshold {0.02};  // 5cm
  double min_collision_distance {std::numeric_limits<double>::max()};
  std::vector<drake::AutoDiffXd> breaks_;
  std::vector<double> breaks_double_;
  WAYPTS_ad waypts_ad_, waypts_ad_temp_;
  PPType_ad spline_, spline_temp_;
};
