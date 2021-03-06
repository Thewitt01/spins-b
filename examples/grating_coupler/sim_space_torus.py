#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 05:33:19 2019

@author: tom

To run an optimization:
$ python3 sim_space_torus.py run save-folder-name

replace run with resume to continue
"""

import os
import pickle
import shutil

import gdspy
import numpy as np
from typing import List, NamedTuple, Tuple

# `spins.invdes.problem_graph` contains the high-level spins code.
from spins.invdes import problem_graph
# Import module for handling processing optimization logs.
from spins.invdes.problem_graph import log_tools
# `spins.invdes.problem_graph.optplan` contains the optimization plan schema.
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace

# If `True`, also minimize the back-reflection.
MINIMIZE_BACKREFLECTION = False


# edit grating.py simulation space


def run_opt(save_folder: str, sim_width: float, wg_width: float) -> None:
    """Main optimization script.

    This function setups the optimization and executes it.

    Args:
        save_folder: Location to save the optimization data.
    """
    os.makedirs(save_folder)

    sim_space = create_sim_space(
        "sim_fg.gds",
        "sim_bg.gds",
        sim_width=sim_width,
        wg_width=wg_width
    )
    obj, monitors = create_objective(
        sim_space, sim_width=sim_width, wg_width = wg_width)  # or a grating length
    trans_list = create_transformations(
        obj, monitors, 60, 200, sim_space, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    # Save the optimization plan so we have an exact record of all the
    # parameters.
    with open(os.path.join(save_folder, "optplan.json"), "w") as fp:
        fp.write(optplan.dumps(plan))
    # Copy over the GDS files.
    shutil.copyfile("sim_fg.gds", os.path.join(save_folder, "sim_fg.gds"))
    shutil.copyfile("sim_bg.gds", os.path.join(save_folder, "sim_bg.gds"))

    # Execute the optimization and indicate that the current folder (".") is
    # the project folder. The project folder is the root folder for any
    # auxiliary files (e.g. GDS files).
    problem_graph.run_plan(plan, ".", save_folder=save_folder)

    # Generate the GDS file.
    gen_gds(save_folder, sim_width, wg_width)


def create_sim_space(
        gds_fg_name: str,
        gds_bg_name: str,
        sim_width: float = 5000,  # size of sim_space
        box_width: float = 3000,  # size of our editing structure
        wg_width: float = 800,
        buffer_len: float = 1500,  # not sure we'll need
        dx: int = 40,
        num_pmls: int = 10,
        visualize: bool = False,
) -> optplan.SimulationSpace:
    """Creates the simulation space.

    The simulation space contains information about the boundary conditions,
    gridding, and design region of the simulation.

    Args:
        gds_fg_name: Location to save foreground GDS.
        gds_bg_name: Location to save background GDS.
        etch_frac: Etch fraction of the grating. 1.0 indicates a fully-etched
            grating.
        box_thickness: Thickness of BOX layer in nm.
        wg_width: Width of the waveguide.
        buffer_len: Buffer distance to put between grating and the end of the
            simulation region. This excludes PMLs.
        dx: Grid spacing to use.
        num_pmls: Number of PML layers to use on each side.
        visualize: If `True`, draws the polygons of the GDS file.

    Returns:
        A `SimulationSpace` description.
    """
    # Calculate the simulation size, including  PMLs
    # TODO change the first part of ech dimension to be equal
    sim_size = [
        sim_width + 2 * buffer_len + dx * num_pmls,
        sim_width + 2 * buffer_len + dx * num_pmls
    ]
    # First, we use `gdspy` to draw the waveguides and shapes that we would
    # like to use. Instead of programmatically generating a GDS file using
    # `gdspy`, we could also simply provide a GDS file (e.g. drawn using
    # KLayout).

    # Declare some constants to represent the different layers.
    # Not sure if we need layers
    LAYER = 100


    # Create rectangles corresponding to the waveguide, the BOX layer, and the
    # design region. We extend the rectangles outside the simulation region
    # by multiplying locations by a factor of 1.1.

    # We distinguish between the top part of the waveguide (which is etched)
    # and the bottom part of the waveguide (which is not etched).

    # TODO define our single waveguide and surface, I don't believe it will be etched.
    waveguide = gdspy.Rectangle((-1.1 * sim_size[0] / 2, -wg_width / 2),
                                (box_width / 2, wg_width / 2),
                                LAYER)

    design_region = gdspy.Rectangle((-box_width / 2, -box_width / 2),
                                    (box_width / 2, box_width / 2),
                                    LAYER)

    # Generate the foreground and background GDS files.
    gds_fg = gdspy.Cell("FOREGROUND", exclude_from_current=True)
    gds_fg.add(waveguide)
    gds_fg.add(design_region)

    # I guess we keep this the same and not include the design_region
    gds_bg = gdspy.Cell("BACKGROUND", exclude_from_current=True)
    gds_bg.add(waveguide)

    gdspy.write_gds(gds_fg_name, [gds_fg], unit=1e-9, precision=1e-9)
    gdspy.write_gds(gds_bg_name, [gds_bg], unit=1e-9, precision=1e-9)

    # The BOX layer/silicon device interface is set at `z = 0`.
    #
    # Describe materials in each layer.

    # 1) Silicon Nitride

    # Note that the layer numbering in the GDS file is arbitrary. Layer 300 is a dummy
    # layer; it is used for layers that only have one material (i.e. the
    # background and foreground indices are identical) so the actual structure
    # used does not matter.

    # Will need to define out material, just silicon nitride
    # Remove the etching stuff
    # Can define Si3N4 - the material we want to use
    # Fix: try to make multiple layers, but all the same?
    air = optplan.Material(index=optplan.ComplexNumber(real=1))
    stack = [
        optplan.GdsMaterialStackLayer(
            foreground=air,
            background=air,
            gds_layer=[100, 0],
            extents=[-10000, -110],  # will probably need to define a better thickness for our layer
        ),
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="Si3N4"),
            background=air,
            gds_layer=[100, 0],
            extents=[-110, 110],  # will probably need to define a better thickness for our layer
        ),
    ]

    mat_stack = optplan.GdsMaterialStack(
        # Any region of the simulation that is not specified is filled with
        # air.
        background=air,
        stack=stack,
    )

    # these define the entire region you wish to scan in the z -direction, not sure for us
    # as we don't require etching or layers
    # will probably change this as thickness may be wrong
    #sim_z_start = -3000
    #sim_z_end = 2000

    # Create a simulation space for both continuous and discrete optimization.
    # TODO there is something wrong here giving a zero size design grid error
    simspace = optplan.SimulationSpace(
        name="simspace",
        mesh=optplan.UniformMesh(dx=dx),
        eps_fg=optplan.GdsEps(gds=gds_fg_name, mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds=gds_bg_name, mat_stack=mat_stack),
        # Note that we explicitly set the simulation region. Anything
        # in the GDS file outside of the simulation extents will not be drawn.
        sim_region=optplan.Box3d(
            center=[0, 0, 0],
            extents=[5000, 5000, 2000], # this is what's messing things up
        ),
        selection_matrix_type="uniform",
        # PMLs are applied on x- and z-axes. No PMLs are applied along y-axis
        # because it is the axis of translational symmetry.
        pml_thickness=[num_pmls, num_pmls, 0, 0, num_pmls, num_pmls],
    )

    if visualize:
        # To visualize permittivity distribution, we actually have to
        # construct the simulation space object.
        import matplotlib.pyplot as plt
        from spins.invdes.problem_graph.simspace import get_fg_and_bg

        context = workspace.Workspace()
        eps_fg, eps_bg = get_fg_and_bg(context.get_object(simspace), wlen=1070)

        def plot(x):
            plt.imshow(np.abs(x)[:, 0, :].T.squeeze(), origin="lower")

        plt.figure()
        plt.subplot(3, 1, 1)
        plot(eps_fg[2])
        plt.title("eps_fg")

        plt.subplot(3, 1, 2)
        plot(eps_bg[2])
        plt.title("eps_bg")

        plt.subplot(3, 1, 3)
        plot(eps_fg[2] - eps_bg[2])
        plt.title("design region")
        plt.show()
    return simspace


# put the objective function here
# Yee cell grid spacing
GRID_SPACING = 40
# define these in a more appropriate place
dx = 40
wg_width = 800


def create_objective(
        sim_space: optplan.SimulationSpace,
        sim_width: float,
        wg_width: float
) -> Tuple[optplan.Function, List[optplan.Monitor]]:
    """"Creates the objective function.
    It will hopefully use the annulur overlap to optimise over
    our desired region"""

    # Create the waveguide source - align with our sim_space
    wg_source = optplan.WaveguideModeSource(
        center=[-1500, 0, 0],  # may need to edit these, not too sure
        extents=[GRID_SPACING, wg_width, dx],  # these too
        normal=[1, 0, 0],
        mode_num=0,
        power=1.0,
    )

    # Creates the annular overlap
    annulus = optplan.AnnulusOverlap(
        normal=[1, 0, 0],
        mode_num=0,
        power=1.0,
    )

    # May want to define a way out, not sure

    power_objs = []
    # Monitor the metrics and fields
    monitor_list = []
    for wlen, overlap in zip([1070], [annulus]):
        epsilon = optplan.Epsilon(
            simulation_space=sim_space,
            wavelength=wlen,
        )

        sim = optplan.FdfdSimulation(
            source=wg_source,
            # Use a direct matrix solver (e.g. LU-factorization) on CPU for
            # 2D simulations and the GPU Maxwell solver for 3D.
            solver="local_direct",
            wavelength=wlen,
            simulation_space=sim_space,
            epsilon=epsilon,
        )
        # Take a field slice through the z=0 plane to save each iteration.
        monitor_list.append(
            optplan.FieldMonitor(
                name="field{}".format(wlen),
                function=sim,
                normal=[0, 0, 1],  # may want to change these normals
                center=[0, 0, 0],
            ))
        if wlen == 1070:
            monitor_list.append(
                optplan.FieldMonitor(
                    name="epsilon",
                    function=epsilon))

        overlap = optplan.Overlap(simulation=sim, overlap=annulus)

    power = optplan.abs(overlap) ** 2
    power_objs.append(power)
    monitor_list.append(
        optplan.SimpleMonitor(name="power{}".format(wlen), function=power))

    # Spins minimizes the objective function, so to make `power` maximized,
    # we minimize `1 - power`.
    obj = 0
    for power in power_objs:
        obj += (1 - power) ** 2

    monitor_list.append(optplan.SimpleMonitor(name="objective", function=obj))

    return obj, monitor_list


def create_transformations(
        obj: optplan.Function,
        monitors: List[optplan.Monitor],
        cont_iters: int,
        disc_iters: int,
        sim_space: optplan.SimulationSpaceBase,
        min_feature: float = 100,
        cont_to_disc_factor: float = 1.1,
) -> List[optplan.Transformation]:
    """Creates a list of transformations for the optimization.

    The grating coupler optimization proceeds as follows:
    1) Continuous optimization whereby each pixel can vary between device and
       background permittivity.
    2) Discretization whereby the continuous pixel parametrization is
       transformed into a discrete grating (Note that L2D is implemented here).
    3) Further optimization of the discrete grating by moving the grating
       edges.

    Args:
        opt: The objective function to minimize.
        monitors: List of monitors to keep track of.
        cont_iters: Number of iterations to run in continuous optimization.
        disc_iters: Number of iterations to run in discrete optimization.
        sim_space: Simulation space ot use.
        min_feature: Minimum feature size in nanometers.
        cont_to_disc_factor: Discretize the continuous grating with feature size
            constraint of `min_feature * cont_to_disc_factor`.
            `cont_to_disc_factor > 1` gives discrete optimization more wiggle
            room.

    Returns:
        A list of transformations.
    """
    # Setup empty transformation list.
    trans_list = []

    # First do continuous relaxation optimization.
    cont_param = optplan.PixelParametrization(
        simulation_space=sim_space,
        init_method=optplan.UniformInitializer(min_val=0, max_val=1))
    trans_list.append(
        optplan.Transformation(
            name="opt_cont",
            parametrization=cont_param,
            transformation=optplan.ScipyOptimizerTransformation(
                optimizer="L-BFGS-B",
                objective=obj,
                monitor_lists=optplan.ScipyOptimizerMonitorList(
                    callback_monitors=monitors,
                    start_monitors=monitors,
                    end_monitors=monitors),
                optimization_options=optplan.ScipyOptimizerOptions(
                    maxiter=cont_iters),
            ),
        ))

    # Discretize. Note we add a little bit of wiggle room by discretizing with
    # a slightly larger feature size that what our target is (by factor of
    # `cont_to_disc_factor`). This is to give the optimization a bit more wiggle
    # room later on.
    disc_param = optplan.GratingParametrization(
        simulation_space=sim_space, inverted=True)
    trans_list.append(
        optplan.Transformation(
            name="cont_to_disc",
            parametrization=disc_param,
            transformation=optplan.GratingEdgeFitTransformation(
                parametrization=cont_param,
                min_feature=cont_to_disc_factor * min_feature)))

    # Discrete optimization.
    trans_list.append(
        optplan.Transformation(
            name="opt_disc",
            parametrization=disc_param,
            transformation=optplan.ScipyOptimizerTransformation(
                optimizer="SLSQP",
                objective=obj,
                constraints_ineq=[
                    optplan.GratingFeatureConstraint(
                        min_feature_size=min_feature,
                        simulation_space=sim_space,
                        boundary_constraint_scale=1.0,
                    )
                ],
                monitor_lists=optplan.ScipyOptimizerMonitorList(
                    callback_monitors=monitors,
                    start_monitors=monitors,
                    end_monitors=monitors),
                optimization_options=optplan.ScipyOptimizerOptions(
                    maxiter=disc_iters),
            ),
        ))
    return trans_list


def view_opt(save_folder: str) -> None:
    """Shows the result of the optimization.

    This runs the auto-plotter to plot all the relevant data.
    See `examples/wdm2` IPython notebook for more details on how to process
    the optimization logs.

    Args:
        save_folder: Location where the log files are saved.
    """
    log_df = log_tools.create_log_data_frame(
        log_tools.load_all_logs(save_folder))
    monitor_descriptions = log_tools.load_from_yml(
        os.path.join(os.path.dirname(__file__), "monitor_spec.yml"))
    log_tools.plot_monitor_data(log_df, monitor_descriptions)


def view_opt_quick(save_folder: str) -> None:
    """Prints the current result of the optimization.

    Unlike `view_opt`, which plots fields and optimization trajectories,
    `view_opt_quick` prints out scalar monitors in the latest log file. This
    is useful for having a quick look into the state of the optimization.

    Args:
        save_folder: Location where the log files are saved.
    """
    with open(workspace.get_latest_log_file(save_folder), "rb") as fp:
        log_data = pickle.load(fp)
        for key, data in log_data["monitor_data"].items():
            if np.isscalar(data):
                print("{}: {}".format(key, data.squeeze()))


def resume_opt(save_folder: str) -> None:
    """Resumes a stopped optimization.

    This restarts an optimization that was stopped prematurely. Note that
    resuming an optimization will not lead the exact same results as if the
    optimization were finished the first time around.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())

    # Run the plan with the `resume` flag to restart.
    problem_graph.run_plan(plan, ".", save_folder=save_folder, resume=True)


def gen_gds(save_folder: str, sim_width: float, wg_width: float) -> None:
    """Generates a GDS file of the grating.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
        sim width: width of the simulation
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())
    dx = plan.transformations[-1].parametrization.simulation_space.mesh.dx

    # Load the data from the latest log file.
    with open(workspace.get_latest_log_file(save_folder), "rb") as fp:
        log_data = pickle.load(fp)
        if log_data["transformation"] != plan.transformations[-1].name:
            raise ValueError("Optimization did not run until completion.")

        coords = log_data["parametrization"]["vector"] * dx

        if plan.transformations[-1].parametrization.inverted:
            coords = np.insert(coords, 0, 0, axis=0)
            coords = np.insert(coords, -1, sim_width, axis=0)

    # TODO Not sure about this part below creating rectangles
    # Change the variables and names here

    # `coords` now contains the location of the grating edges. Now draw a
    # series of rectangles to represent the grating.
    grating_poly = []
    for i in range(0, len(coords), 2):
        grating_poly.append(
            ((coords[i], -sim_width / 2), (coords[i], sim_width / 2),
             (coords[i + 1], sim_width / 2), (coords[i + 1], -sim_width / 2)))

    # Save the grating to `annulus.gds`.
    grating = gdspy.Cell("ANNULUS", exclude_from_current=True)
    grating.add(gdspy.PolygonSet(grating_poly, 100))
    gdspy.write_gds(
        os.path.join(save_folder, "annulus.gds"), [grating],
        unit=1.0e-9,
        precision=1.0e-9)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("run", "view", "view_quick", "resume", "gen_gds"),
        help="Must be either \"run\" to run an optimization, \"view\" to "
             "view the results, \"resume\" to resume an optimization, or "
             "\"gen_gds\" to generate the grating GDS file.")
    parser.add_argument(
        "save_folder", help="Folder containing optimization logs.")

    sim_width = 5000
    wg_width = 800

    args = parser.parse_args()
    if args.action == "run":
        run_opt(args.save_folder, sim_width=sim_width, wg_width = wg_width)
    elif args.action == "view":
        view_opt(args.save_folder)
    elif args.action == "view_quick":
        view_opt_quick(args.save_folder)
    elif args.action == "resume":
        resume_opt(args.save_folder)
    elif args.action == "gen_gds":
        gen_gds(args.save_folder, sim_width=sim_width, wg_width = wg_width)
