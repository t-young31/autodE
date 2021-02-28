from abc import ABC
import autode as ade


class Optimiser(ABC):

    def run(self):
        """
        Optimise to convergence

        Returns:
            (None):
        """

    def _get_gradient(self, coordinates):
        """
        Calculate the gradient for a set of coordinates

        Args:
            coordinates (np.ndarray): Cartesian coordinates with shape
                        (n_atoms, 3) or (3*n_atoms,)

        Returns:
            (np.ndarray): Gradient shape = (3*n_atoms,)  (Ha / Å)

        Raises:
            (autode.exceptions.CalculationException): If a calculation fails
        """
        self.species.coordinates = coordinates

        # Run the calculation and remove all the files, if it's successful
        calc = ade.Calculation(name='tmp_grad',
                               molecule=self.species,
                               method=self.method,
                               keywords=self.method.keywords.grad,
                               n_cores=ade.Config.n_cores)
        calc.run()

        grad = calc.get_gradients()
        calc.clean_up(force=True, everything=True)

        return grad.flatten()

    def __init__(self, species, method):
        """
        Optimiser initialised from a species and a method used to perform
        gradient and/or Hessian calculations

        Arguments:
            species (autode.species.Species):

            method (autode.wrappers.base.ElectronicStructureMethod):
        """
        self.species = species
        self.method = method
