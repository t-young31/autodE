"""
https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
"""
import pytest
import numpy as np
from autode.species import Molecule
from autode.wrappers.base import Method
from .optimiers import TestBFGSOptimiser


def test_opt():

    blank_mol = Molecule(name='blank')
    blank_method = Method()

    optimiser = TestBFGSOptimiser()
    optimiser.run(blank_mol, method=blank_method)
    assert optimiser.converged
