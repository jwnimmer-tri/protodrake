/*
 * Copyright © 2022 Dexai Robotics. All rights reserved.
 */

#include "drake/solvers/ipopt_solver.h"
#include "mock_dracula.h"
#include "path_optimization.h"

TEST(Optimization, TestAA) {
  parameters::Parameters params;
  auto pdrac {
      dut::make_dracula(params, "/src/config/franka_aa.yaml", false, 0, 2)};
  params.urdf =
      "/src/catkin_src/salad_bar_description/urdf/"
      ".go_hotel_pan_third_6in_000_disher_2oz-drake.urdf";
  params.UpdateUrdf(params.urdf);
  pdrac->ResetParams(params, true);
  pdrac->StartMeshcatVisualizer();
  robot_state_vec_t traj;
  std::string mpac_filepath = "test_data/example_traj.mpac";
  dru::load_msg_pack(traj, mpac_filepath);
  momap::log()->info("traj size: {}", traj.size());
  const auto syspoly {dru::ToSysPoly(traj, params)};
  momap::log()->info("syspoly size: {}", syspoly.size());
  double T {dru::Duration(syspoly)};
  momap::log()->info("syspoly time : {}", T);
  // Setup Robot Diagram
  auto pop = std::make_unique<PathOptimization>(params);
  const auto& plant = pop->GetPlant();
  auto& mutable_plant_context = pop->MutuablePlantContext();
  // Setup joint limits
  const auto& cobot_name {params.GetCobotName()};
  const auto& aa_name {params.GetAncillaryArmName()};
  system_conf_t sys_conf_joint_lim_lower, sys_conf_joint_lim_upper,
      sys_conf_vel_limit, sys_conf_acc_limit;
  sys_conf_joint_lim_lower[cobot_name] =
      params.planning_limits_map.at(cobot_name)->joint_limits.lower_limits;
  sys_conf_joint_lim_upper[cobot_name] =
      params.planning_limits_map.at(cobot_name)->joint_limits.upper_limits;
  sys_conf_joint_lim_lower[aa_name] =
      params.planning_limits_map.at(aa_name)->joint_limits.lower_limits;
  sys_conf_joint_lim_upper[aa_name] =
      params.planning_limits_map.at(aa_name)->joint_limits.upper_limits;
  sys_conf_vel_limit[cobot_name] =
      params.planning_limits_map.at(cobot_name)->velocity_limits;
  sys_conf_vel_limit[aa_name] =
      params.planning_limits_map.at(aa_name)->velocity_limits;
  sys_conf_acc_limit[cobot_name] =
      params.planning_limits_map.at(cobot_name)->acceleration_limits;
  sys_conf_acc_limit[aa_name] =
      params.planning_limits_map.at(aa_name)->acceleration_limits;
  const Eigen::VectorXd joint_limit_lower {
      pdrac->GetCS()->PackModelAccelerations(sys_conf_joint_lim_lower)};
  const Eigen::VectorXd joint_limit_upper {
      pdrac->GetCS()->PackModelAccelerations(sys_conf_joint_lim_upper)};
  const Eigen::VectorXd joint_vel_limit {
      pdrac->GetCS()->PackModelAccelerations(sys_conf_vel_limit)};
  const Eigen::VectorXd joint_acc_limit {
      pdrac->GetCS()->PackModelAccelerations(sys_conf_acc_limit)};
  const auto joint_limits {anzu::planning::JointLimits(
      joint_limit_lower, joint_limit_upper, -joint_vel_limit, joint_vel_limit,
      -joint_acc_limit, joint_acc_limit)};
  // Geometry port for collisions
  const auto& query_port {plant.get_geometry_query_input_port()};
  // **************************************************************************
  // *************************  Set up the robot ******************************
  // **************************************************************************
  drake::MatrixX<double> q_franka = Eigen::VectorXd::Zero(7);
  drake::MatrixX<double> q_aa = Eigen::VectorXd::Zero(2);
  auto franka_model {pop->model_map_[cobot_name]};
  auto aa_model {pop->model_map_[aa_name]};
  auto prog = std::make_unique<drake::solvers::MathematicalProgram>();
  // **************************************************************************
  // ********************  Set up the System Polynomial ***********************
  // **************************************************************************
  // Step 1: Setup the AA polynomial
  std::vector<drake::MatrixX<drake::symbolic::Expression>> aa_vars_vec;
  std::vector<drake::symbolic::Expression> breaks_vec;
  int n_aa_vars = 5;
  {
    q_aa = (syspoly.at(params.GetAncillaryArmName())).value(0);
    drake::MatrixX<drake::symbolic::Expression> aa_start {q_aa};
    aa_vars_vec.push_back(aa_start);
    auto t_start = drake::symbolic::Expression {0.0};
    breaks_vec.push_back(t_start);
  }
  for (int i = 1; i < n_aa_vars - 2; i++) {
    auto q_var = prog->NewContinuousVariables(2, "q_aa_" + std::to_string(i));
    aa_vars_vec.push_back(q_var);
    const double t {i * T / (n_aa_vars - 1)};
    breaks_vec.push_back(drake::symbolic::Expression(t));
  }
  {
    q_aa = (syspoly.at(params.GetAncillaryArmName())).value(T);
    drake::MatrixX<drake::symbolic::Expression> aa_end {q_aa};
    aa_vars_vec.push_back(aa_end);
    auto t_dispense =
        drake::symbolic::Expression {(n_aa_vars - 2) * T / (n_aa_vars - 1)};
    breaks_vec.push_back(t_dispense);
  }
  {
    q_aa = (syspoly.at(params.GetAncillaryArmName())).value(T);
    drake::MatrixX<drake::symbolic::Expression> aa_end {q_aa};
    aa_vars_vec.push_back(aa_end);
    auto t_end = drake::symbolic::Expression {T};
    breaks_vec.push_back(t_end);
  }
  // Step 3: Build the AA polynomial from the AA variables
  auto aa_symbolic_poly =
      drake::trajectories::PiecewisePolynomial<drake::symbolic::Expression>::
          CubicWithContinuousSecondDerivatives(breaks_vec, aa_vars_vec,
                                               Eigen::VectorXd::Zero(2),
                                               Eigen::VectorXd::Zero(2));
  momap::log()->debug("aa_symbolic_poly: {}",
                      aa_symbolic_poly.value(0.0).transpose());
  momap::log()->debug("aa_symbolic_poly: {}",
                      aa_symbolic_poly.value(0.1).transpose());
  // Step 4: Setup the Franka polynomial
  int n_franka_pts = 20;
  std::vector<drake::MatrixX<double>> franka_vec;
  std::vector<drake::MatrixX<double>> aa_double_vec;
  std::vector<double> franka_breaks_vec;
  for (int i = 0; i < n_franka_pts; i++) {
    const double t {i * T / (n_franka_pts - 1)};
    q_franka = (syspoly.at(params.GetCobotName())).value(t);
    q_aa = (syspoly.at(params.GetAncillaryArmName())).value(t);
    franka_vec.push_back(q_franka);
    aa_double_vec.push_back(q_aa);
    franka_breaks_vec.push_back(t);
  }
  auto franka_double_poly = drake::trajectories::PiecewisePolynomial<
      double>::CubicWithContinuousSecondDerivatives(franka_breaks_vec,
                                                    franka_vec);
  auto aa_double_poly = drake::trajectories::PiecewisePolynomial<
      double>::CubicWithContinuousSecondDerivatives(franka_breaks_vec,
                                                    aa_double_vec);
  momap::log()->info("franka_double_poly: {}",
                     franka_double_poly.value(0.7).transpose());
  momap::log()->info("aa_double_poly: {}",
                     aa_double_poly.value(0.7).transpose());
  // Step 5: Sample the franka_polynomial and the aa_polynomial to construct
  // collision avoidance constraint
  momap::log()->info("lower joints = {}",
                     joint_limits.position_lower().transpose());
  momap::log()->info("higher joints = {}",
                     joint_limits.position_upper().transpose());
  int n_sample = 50;
  for (int i = 0; i < n_sample; i++) {
    const double t {i * T / (n_sample - 1)};
    q_franka = franka_double_poly.value(t);
    q_aa = aa_double_poly.value(t);
    plant.SetPositions(&mutable_plant_context, franka_model, q_franka);
    plant.SetPositions(&mutable_plant_context, aa_model, q_aa);
    auto q_vars {
        prog->NewContinuousVariables(9, 1, "q_collision " + std::to_string(i))};
    auto con {std::shared_ptr<drake::multibody::MinimumDistanceConstraint>(
        new drake::multibody::MinimumDistanceConstraint(
            &plant, 0.01, &mutable_plant_context, {}, 0.15))};
    prog->AddBoundingBoxConstraint(joint_limits.position_lower(),
                                   joint_limits.position_upper(), q_vars);
    // Bond the q_vars to the aa_polynomial
    momap::log()->info("q_aa: {},  {}", i, q_aa.transpose());
    for (size_t j = 0; j < 2; j++) {
      prog->AddLinearEqualityConstraint(
          aa_symbolic_poly.value(t)(j, 0) - q_vars(7 + j, 0), 0.0);
    }
    momap::log()->info("q_franka: {},  {}", i, q_franka.transpose());
    for (size_t j = 0; j < 7; j++) {
      prog->AddLinearEqualityConstraint(q_vars(j, 0), q_franka(j, 0));
    }
    auto constraint {prog->AddConstraint(con, q_vars)};
    // Collision check print here
    auto q {plant.GetPositions(mutable_plant_context)};
    momap::log()->warn("Collision constraint satisfied at sample {}: {}", i,
                       con->CheckSatisfied(q, 0.0));
    const auto& query_object {
        query_port.Eval<drake::geometry::QueryObject<double>>(
            mutable_plant_context)};
    const auto& inspector {query_object.inspector()};
    auto distance_pairs {
        query_object.ComputeSignedDistancePairwiseClosestPoints(0.01)};
    for (const auto& dist_pair : distance_pairs) {
      const auto name_body_1 {
          plant.GetBodyFromFrameId(inspector.GetFrameId(dist_pair.id_A))
              ->name()};
      const auto name_body_2 {
          plant.GetBodyFromFrameId(inspector.GetFrameId(dist_pair.id_B))
              ->name()};
      momap::log()->info("{} / {} :{}", name_body_1, name_body_2,
                         dist_pair.distance);
    }
  }
  // Step 6: Add time cost!
  drake::symbolic::Expression time_cost {0.0};
  momap::log()->info("time_cost: {}", time_cost.to_string());
  // Just doing this for AA for now
  int n_time_samples {10};
  for (int i = 0; i < n_time_samples; i++) {
    const drake::symbolic::Expression s {i * T / (n_time_samples - 1)};
    // const double delta_s {T / (n_time_samples - 1)};
    auto q_der_s {aa_symbolic_poly.EvalDerivative(s, 1)};
    auto q_der2_s {aa_symbolic_poly.EvalDerivative(s, 2)};
    // drake::symbolic::Expression sdot {std::numeric_limits<double>::max()};
    // for (size_t j = 0; j < 2; j++){
    //   sdot = drake::symbolic::min(sdot, joint_limits.velocity_upper()(j) /
    //   q_der_s(j,0)); sdot = drake::symbolic::min(sdot,
    //   drake::symbolic::sqrt(joint_limits.acceleration_upper()(j) /
    //   q_der2_s(j,0)));
    // }
    // TODO(@sadraddini) is not a polynomial. ParseCost does not support
    // non-polynomial expression.
    drake::symbolic::Expression sdot;
    for (size_t j = 0; j < 2; j++) {
      sdot += 10 * q_der2_s(j, 0) * q_der2_s(j, 0);
      sdot += 1000 * q_der_s(j, 0) * q_der_s(j, 0);
    }
    time_cost += sdot;
  }
  prog->AddCost(time_cost);
  // Step 7: Set initial guess
  for (int i = 1; i < n_aa_vars - 2; i++) {
    const double t {i * T / (n_aa_vars - 1)};
    q_aa = aa_double_poly.value(t);
    for (size_t j = 0; j < 2; j++) {
      auto vars = aa_vars_vec[i](j, 0).GetVariables();
      momap::log()->info("vars: {}, {}, {}", i, j, q_aa(j, 0));
      auto var = *vars.begin();
      prog->SetInitialGuess(var, q_aa(j, 0));
    }
  }
  // Step 8: Solve!
  drake::solvers::SnoptSolver solver;
  // drake::solvers::IpoptSolver solver;
  const std::string print_file = "snopt.out";
  drake::solvers::SolverOptions solver_options;
  solver_options.SetOption(drake::solvers::SnoptSolver::id(), "Print file",
                           print_file);
  // solver_options.SetOption(drake::solvers::CommonSolverOption::kPrintFileName,
  // "snopting.out");
  const auto result = solver.Solve(*prog, {}, solver_options);
  // drake::solvers::MathematicalProgramResult result = solver.Solve(*prog, {},
  // {});
  momap::log()->info("Solver used: {}", result.get_solver_id().name());
  momap::log()->critical("Solver success: {}", result.is_success());
  momap::log()->critical("Solver success: {}", result.get_solution_result());

  if (result.is_success()) {
    momap::log()->info("Optimal cost is {}", result.get_optimal_cost());
    // Now: let's go back and spline the optimized decisions!
    // Step 1: Construct the optimal AA spline
    std::vector<drake::MatrixX<double>> aa_optimized_q_vec;
    std::vector<double> aa_optimized_t_vec;
    for (int i = 0; i < n_aa_vars; i++) {
      drake::MatrixX<double> q_optimized =
          drake::symbolic::Evaluate(result.GetSolution(aa_vars_vec[i]));
      aa_optimized_q_vec.push_back(q_optimized);
      aa_optimized_t_vec.push_back(i * T / (n_aa_vars - 1));
      momap::log()->info("q_optimized {}: {}", i, q_optimized.transpose());
    }
    auto aa_optimized_q_poly = drake::trajectories::PiecewisePolynomial<
        double>::CubicWithContinuousSecondDerivatives(aa_optimized_t_vec,
                                                      aa_optimized_q_vec,
                                                      Eigen::VectorXd::Zero(2),
                                                      Eigen::VectorXd::Zero(2));
    // Step 2: Sample from the optimized AA spline and franka_spline to
    // construct a plan
    int n_sample_plan {20};
    robot_state_vec_t plan;
    for (int i = 0; i < n_sample_plan; i++) {
      system_conf_t sys_conf;
      const double t {i * T / (n_sample_plan - 1)};
      sys_conf[cobot_name] = franka_double_poly.value(t);
      sys_conf[aa_name] = aa_optimized_q_poly.value(t);
      plan.push_back(pdrac->GetCS()->ToState(sys_conf));
    }
    const auto optimized_syspoly {
        pdrac->GetTS()->TimeOptimalSpline(plan, 1, 1, true, true, true)};
    double T_optimized {dru::Duration(optimized_syspoly)};
    momap::log()->info("syspoly time : old: {} vs new: {}", T, T_optimized);
    // display results in meshcat:
    pdrac->GetMeshcat()->DisplaySysPoly(syspoly, "original trajectory");
    pdrac->GetMeshcat()->DisplaySysPoly(optimized_syspoly,
                                        "optimized trajectory");
  } else {
    momap::log()->warn("Solver failed");
    auto infeasible_vec {result.GetInfeasibleConstraints(*prog)};
    momap::log()->info("Infeasible constraints size: {}",
                       infeasible_vec.size());
    for (const auto& infeasible : infeasible_vec) {
      momap::log()->info("Infeasible constraint: {}", infeasible.to_string());
    }
  }
}
