from typing import List, Tuple

import numpy as np

from spins import fdfd_tools
from spins import gridlock
from spins.invdes import problem
from spins.invdes.problem_graph import creator_em
from spins.invdes.problem_graph import grid_utils
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
from spins.invdes.problem_graph.simspace import SimulationSpace
from spins.invdes.problem_graph import schema_utils


#@optplan.register_node_type() # Delete me later
class StoredEnergy(optplan.Function):
    """Defines an integral for calculating the stored energy in a resonator.

    Attributes:
        type: Must be "function.stored_energy".
        simulation: Simulation from which electric fields are obtained.
        center: Centre of integration region
        extents: Width and height of integration region.
        epsilon: Permittivity.
    """
    type = schema_utils.polymorphic_model_type("function.stored_energy")
    simulation = optplan.ReferenceType(optplan.Function)
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    center = optplan.vec3d()
    extents = optplan.vec3d()
    epsilon = optplan.ReferenceType(optplan.Function)


class StoredEnergyFunction(problem.OptimizationFunction):
    """Represents an optimization function for stored energy calcuation"""

    def __init__(self, input_function: problem.OptimizationFunction,
                 simulation_space: SimulationSpace,
                 center: np.ndarray,
                 extents: np.ndarray,
                 epsilon: problem.OptimizationFunction):
        """Constructs the Stored Energy Fn

        Args:
            input_function: Input objectives (typically a simulation from which
                            we get the e-field).
            simulation_space: sim space we are simulating
            centre:  Centre of integration region
            extents: extends of integration region
            epsilon: Permittivity
        """
        # Call superclass initialisor. Nb: this defines the fn inputs for eval etc
        super().__init__([input_function, epsilon])

        region_slice = grid_utils.create_region_slices(simulation_space.edge_coords, center, extents)

        # Create a selection filter for vectorised fields and permittivities
        filter_grid = [np.zeros(simulation_space.dims) for i in range(3)]
        for i in range(3):
            filter_grid[i][tuple(region_slice)] = 1  # X,Y,Z components set to 1 for region slice
        self._filter_vec = fdfd_tools.vec(filter_grid)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """ Returns integral(eps*|E|^2) over region defined by center and extents

        Args:
            input_vals: List of the input values i.e. [e-field, epsilon]

        Returns:
            Stored energy in region.
        """
        e_field = input_vals[0]  # Vectorised electric field
        eps = np.real(input_vals[1])  # Vectorised real permittivity
        return np.sum(eps * e_field * np.conj(e_field) * self._filter_vec)

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            gradient.
        """
        # Use chain to calculate derivative terms wrt inputs
        dval_deps = grad_val * input_vals[0] * np.conj(input_vals[0]) * self._filter_vec  # Derivative wrt epsilon
        dval_dE = grad_val * input_vals[1] * (
                input_vals[0] + np.conj(input_vals[0])) * self._filter_vec  # Derivative wrt E-field
        return [dval_dE, dval_deps]

    def __str__(self):
        return "Stored Energy Optimisation Fn"


@optplan.register_node(StoredEnergy)
def create_stored_energy_fn(params: optplan.ProblemGraphNode,
                            work: workspace.Workspace):
    return StoredEnergyFunction(input_function=work.get_object(params.simulation),
                                simulation_space=work.get_object(params.simulation_space),
                                center=params.center,
                                extents=params.extents,
                                epsilon=work.get_object(params.epsilon))
