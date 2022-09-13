import logging
import os.path
import sys
import time

from bazel_tools.tools.python.runfiles.runfiles import Create as CreateRunfiles
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Expression,
    LoadModelDirectives,
    MathematicalProgram,
    MinimumDistanceConstraint,
    Parser,
    PiecewisePolynomial,
    PiecewisePolynomial_,
    Polynomial_,
    ProcessModelDirectives,
    SnoptSolver,
    configure_logging,
)


def _resource(respath):
    """Given a Bazel resource path, returns a filesystem path.
    """
    runfiles = CreateRunfiles()
    result = runfiles.Rlocation(respath)
    assert result is not None
    assert os.path.exists(result), result
    return os.path.abspath(result)


def _reference_traj():
    """Returns the reference trajectory to be optimized.
    """
    num_dofs = 7
    samples = np.array([
        [0.10, 0.11, 0.12, -0.13, 0.14, 0.15, 0.16],
        [0.20, 0.21, 0.22, -0.23, 0.24, 0.25, 0.26],
        [0.30, 0.31, 0.32, -0.33, 0.34, 0.35, 0.36],
        [0.40, 0.41, 0.42, -0.43, 0.44, 0.45, 0.46],
        [0.50, 0.51, 0.52, -0.53, 0.54, 0.55, 0.56],
    ])
    num_samples, num_dofs = samples.shape
    breaks = np.linspace(0, 1, num_samples)
    result = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        breaks=breaks, samples=samples.T,
        sample_dot_at_start=np.zeros(num_dofs),
        sample_dot_at_end=np.zeros(num_dofs))
    return result


def _decision_traj(*, traj_float, prog):
    """Returns a copy of the traj_float with the non-terminal samples
    replaced by newly-added decision variables.
    """
    breaks = [
        Expression(s)
        for s in traj_float.get_segment_times()
    ]
    num_samples = len(breaks)
    num_dofs = traj_float.value(traj_float.start_time()).size

    sample_vars = []
    samples = []
    samples.append(traj_float.value(traj_float.start_time()).squeeze())
    for i in range(1, num_samples - 1):
        q = prog.NewContinuousVariables(num_dofs, f"q{i}")
        sample_vars.append(q)
        samples.append(q)
    samples.append(traj_float.value(traj_float.end_time()).squeeze())
    samples = np.array(samples)

    PPE = PiecewisePolynomial_[Expression]
    result = PPE.CubicWithContinuousSecondDerivatives(
        breaks=breaks, samples=samples.T,
        sample_dot_at_start=np.zeros(num_dofs),
        sample_dot_at_end=np.zeros(num_dofs))
    return result, sample_vars


def _solution_traj(*, traj_float, prog_result, sample_vars):
    breaks = traj_float.get_segment_times()
    num_samples = len(breaks)
    num_dofs = traj_float.value(traj_float.start_time()).size

    samples = []
    samples.append(traj_float.value(traj_float.start_time()).squeeze())
    for i in range(1, num_samples - 1):
        samples.append(prog_result.GetSolution(sample_vars[i - 1]))
    samples.append(traj_float.value(traj_float.end_time()).squeeze())
    samples = np.array(samples)

    result = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        breaks=breaks, samples=samples.T,
        sample_dot_at_start=np.zeros(num_dofs),
        sample_dot_at_end=np.zeros(num_dofs))
    return result


def _make_robot():
    # Create the motion planning plant.
    # TODO(jwnimmer-tri) Use RobotDiagramBuilder here instead.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    directives = LoadModelDirectives(_resource(
        "protodrake/path_optimization/demo.dmd.yaml"))
    ProcessModelDirectives(directives, Parser(plant))
    plant.Finalize()
    return builder, plant, scene_graph


def _visualize(*, traj_sol):
    builder, plant, scene_graph = _make_robot()
    AddDefaultVisualization(builder)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    s_start = traj_sol.start_time()
    s_end = traj_sol.end_time()
    for i in range(2):
        if i > 0:
            time.sleep(0.5)
        for s in np.linspace(s_start, s_end, 100):
            plant.SetPositions(plant_context, traj_sol.value(s))
            diagram.Publish(diagram_context)
            time.sleep(0.01)


def run():
    # Create the motion planning plant.
    builder, plant, scene_graph = _make_robot()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # Load the nominal trajectory (to be optimized).
    traj_float = _reference_traj()
    s_start = traj_float.start_time()
    s_end = traj_float.end_time()
    num_dofs = traj_float.value(traj_float.start_time()).size

    # Define an optimal trajectory using decision variables for samples.
    prog = MathematicalProgram()
    traj_expr, sample_vars = _decision_traj(traj_float=traj_float, prog=prog)

    # Add constraints sampled along the trajectory.
    num_constraint_samples = 50
    for i, s in enumerate(np.linspace(s_start, s_end, num_constraint_samples)):
        # Express this position in terms of the sample decision variables.
        traj_q = traj_expr.value(s).squeeze()

        # Create decision variables for this intermediate configuration and
        # bond to traj_q.
        var_q = prog.NewContinuousVariables(num_dofs, "qc{i}")
        for j in range(num_dofs):
            c = prog.AddLinearEqualityConstraint(var_q[j] - traj_q[j], 0.0)
            c.evaluator().set_description(f"{i}th subsample at {j}")

        # Add joint limit constraints.
        c = prog.AddBoundingBoxConstraint(
            plant.GetPositionLowerLimits(),
            plant.GetPositionUpperLimits(),
            var_q)
        c.evaluator().set_description(f"{i}th joint limit")

        # Add collision avoidance constraints.
        c = MinimumDistanceConstraint(
            plant, 0.01, plant_context, None, 0.15)
        prog.AddConstraint(c, var_q)

    # Add objective cost.
    cost = Expression(0)
    num_cost_samples = 10
    for i, s in enumerate(np.linspace(s_start, s_end, num_cost_samples)):
        q_der2_s = traj_expr.EvalDerivative(s, 2).squeeze()
        cost += 10 * q_der2_s.dot(q_der2_s)
    prog.AddCost(cost)

    # Solve the program.
    solver = SnoptSolver()
    prog_result = solver.Solve(prog)
    logging.info(f"Solver is_success: {prog_result.is_success()}")
    logging.info(f"Solver solution_result: {prog_result.get_solution_result()}")

    # Help debug failures.
    if not prog_result.is_success():
        for x in prog_result.GetInfeasibleConstraints(prog):
            print(f"infeasible constraint: {c}")
        return 1

    # Substitue the optimial decision variables into the trajectory.
    traj_sol = _solution_traj(
        traj_float=traj_float,
        prog_result=prog_result,
        sample_vars=sample_vars)

    # Plot the two trajectories in cspace.
    fig, ax = plt.subplots()
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-4, 4])
    ax.set_xlabel("s")
    ax.set_ylabel("q")
    s = np.arange(traj_float.start_time(), traj_float.end_time(), 0.001)
    for traj in traj_float, traj_sol:
        x = np.hstack([traj.value(time) for time in s])
        for dof in range(x.shape[0]):
            ax.plot(s, x[dof, :], linewidth=2)
    filename = f"{os.environ['BUILD_WORKING_DIRECTORY']}/demo.png"
    plt.savefig(filename)

    # Visualize
    _visualize(traj_sol=traj_sol)


def main():
    configure_logging()
    return run()


assert __name__ == "__main__"
sys.exit(main())
