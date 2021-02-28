# ------------------------ Calculation exceptions -----------------------------
class CalculationException(Exception):
    """Base autodE calculation exception"""


class AtomsNotFound(CalculationException):
    pass


class NoCalculationOutput(CalculationException):
    pass


class MethodUnavailable(CalculationException):
    pass


class UnsuppportedCalculationInput(CalculationException):
    def __init__(self, message='Parameters not supported'):
        super().__init__(message)


class NoNormalModesFound(CalculationException):
    pass


class CouldNotGetProperty(CalculationException):
    def __init__(self, name):
        super().__init__(f'Could not get {name}')


class SolventUnavailable(CalculationException):
    pass


class NoInput(CalculationException):
    pass


# -------------------------- Graph exceptions ---------------------------------
class GraphException(Exception):
    """Base autodE graph exception"""


class BondsInSMILESAndGraphDontMatch(GraphException):
    pass


class NoMapping(GraphException):
    pass


class NoMolecularGraph(GraphException):
    pass


class CannotSplitAcrossBond(GraphException):
    """A molecule cannot be partitioned by deleting one bond"""


# -------------------------- Reaction exceptions ------------------------------
class ReactionException(Exception):
    """Base autodE reaction exception"""


class UnbalancedReaction(ReactionException):
    pass


class SolventsDontMatch(ReactionException):
    pass


class ReactionFormationFalied(ReactionException):
    pass


# --------------------------- Other exceptions --------------------------------
class NoAtomsInMolecule(Exception):
    pass


class NoConformers(Exception):
    pass


class OptimisationFailed(Exception):
    pass


class FitFailed(Exception):
    pass


class CouldNotPlotSmoothProfile(Exception):
    pass


class SolventNotFound(Exception):
    pass


class XYZfileDidNotExist(Exception):
    pass


class XYZfileWrongFormat(Exception):
    pass


class NoClosestSpecies(Exception):
    pass


class RDKitFailed(Exception):
    pass


class InvalidSmilesString(Exception):
    pass


class TemplateLoadingFailed(Exception):
    pass
