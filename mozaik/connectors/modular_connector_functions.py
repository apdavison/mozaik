# encoding: utf-8
from parameters import ParameterSet, ParameterDist
import numpy

from ..core import ParametrizedObject
from ..tools.distribution_parametrization import PyNNDistribution
from ..tools.misc import *


class ModularConnectorFunction(ParametrizedObject):
    """
    Abstract class defining the interface of nodular connector functions.
    Each instance has to implement the evaluate(u) function that returns the pre-synaptic weights
    of neuron i.
    """

    def __init__(self, source, target, parameters):
        ParametrizedObject.__init__(self, parameters)
        self.source = source
        self.target = target

    def evaluate(self, index):
        raise NotImplemented


class ConstantModularConnectorFunction(ModularConnectorFunction):
    """
    Triavial modular connection function assigning each connections the same weight
    """

    def evaluate(self, index):
        return numpy.zeros(len(self.source.pop)) + 1


class PyNNDistributionConnectorFunction(ModularConnectorFunction):
    """
    ConnectorFunction which draws the values from the PyNNDistribution
    """

    required_parameters = ParameterSet(
        {"pynn_distribution": PyNNDistribution}  # The distribution
    )

    def evaluate(self, index):
        return self.parameters.pynn_distribution.next(len(self.source.pop))


class DistanceDependentModularConnectorFunction(ModularConnectorFunction):
    """
    Helper abstract class to ease the definitions of purely distance dependent connector functions.
    The distance is defined as the *horizontal* distance between the retinotopical positions of the neurons (one in source and one in destination sheet).
    The distane is translated into the native coordinates of the target sheet (e.g. micrometers for CorticlaSheet)!
    For the special case where source = target, this coresponds to the intuitive lateral distance of the neurons.
    """

    def distance_dependent_function(self, distance):
        """
        The is the function, dependent only on distance that each DistanceDependentModularConnectorFunction has to implement.
        The distance can be matrix.
        """
        raise NotImplemented

    def evaluate(self, index):
        # print(index)
        # print("DistanceDependentModularConnectorFunction evaluate ")
        # print("self.source.pop.positions.shape ", self.source.pop.positions.shape)
        # print("self.source.pop.positions", self.source.pop.positions)
        # print("self.target.pop.positions.shape ", self.target.pop.positions.shape)
        # print("self.target.pop.positions ", self.target.pop)
        """
        xs = []
        ys = []
        for (i, neuron2) in enumerate(self.source.pop.all()):
            xs = numpy.append(xs, self.source.pop.positions[i][0])
            ys = numpy.append(ys, self.source.pop.positions[i][1])
        sp = numpy.vstack((xs, ys))
        xt = []
        yt = []
        for (j, neuron2) in enumerate(self.target.pop.all()):
            xt = numpy.append(xt, self.target.pop.positions[j][0])
            yt = numpy.append(yt, self.target.pop.positions[j][1])
        tp = numpy.vstack((xt, yt))
        """
        # print("modular connector functions self.source.pop.positions ", self.source.pop.positions)
        return self.distance_dependent_function(
            self.source.dvf_2_dcs(
                numpy.sqrt(
                    numpy.power(
                        # self.source.pop.positions[0, :]
                        self.source.pop.positions[:, 0]
                        # xs[0, :]
                        # sp[0, :]
                        # - self.target.pop.positions[0, index],
                        - self.target.pop.positions[index, 0],
                        # - xt[0, index],
                        # - tp[0, index],
                        2
                    )
                    + numpy.power(
                        # self.source.pop.positions[1, :]
                        self.source.pop.positions[:, 1]
                        # ys[1, :]
                        # sp[1, :]
                        # - self.target.pop.positions[1, index],
                        - self.target.pop.positions[index, 1],
                        # - yt[1, index],
                        # - tp[1, index],
                        2
                    )
                )
            )
        )


class GaussianDecayModularConnectorFunction(DistanceDependentModularConnectorFunction):
    """
    Distance dependent arborization with gaussian fall-off of the connections: k * exp(-0.5*(distance/a)*2) / (a*sqrt(2*pi))
    where a = arborization_constant, k = arborization_scaler
    """

    required_parameters = ParameterSet(
        {
            # μm distance constant of the gaussian decay of the connections with respect (in cortical distance)
            "arborization_constant": float,
            # to the distance from the innervation point.
            "arborization_scaler": float,  # the scaler of the gaussian decay
        }
    )

    def distance_dependent_function(self, distance):
        return (
            self.parameters.arborization_scaler
            * numpy.exp(-0.5 * (distance / self.parameters.arborization_constant) ** 2)
            / (self.parameters.arborization_constant * numpy.sqrt(2 * numpy.pi))
        )


class ExponentialDecayModularConnectorFunction(
    DistanceDependentModularConnectorFunction
):
    """
    Distance dependent arborization with exponential fall-off of the connections: k * exp(-distance/a)
    """

    required_parameters = ParameterSet(
        {
            # μm distance constant of the exponential decay of the connections with respect (in cortical distance)
            "arborization_constant": float,
            # to the distance from the innervation point.
            "arborization_scaler": float,  # the scaler of the exponential decay
        }
    )

    def distance_dependent_function(self, distance):
        return self.parameters.arborization_scaler * numpy.exp(
            -distance / self.parameters.arborization_constant
        )


class LinearModularConnectorFunction(DistanceDependentModularConnectorFunction):
    """
    Corresponds to: distance*linear_scaler + constant_scaler, where distance is in micrometers
    """

    required_parameters = ParameterSet(
        {
            "constant_scaler": float,  # the aditive constant of the decay
            "linear_scaler": float,  # the scaler of the linear decay
        }
    )

    def distance_dependent_function(self, distance):
        return (
            self.parameters.linear_scaler * distance + self.parameters.constant_scaler
        )
        # x = self.parameters.linear_scaler * distance + self.parameters.constant_scaler
        # print("X distance_dependent_function", x)
        # if x <= 14.4:
        #    return x
        # else:
        #    return 14.4
        # return x


class LinearModularConnectorFunction1(DistanceDependentModularConnectorFunction):
    """
    Corresponds to: distance*linear_scaler + constant_scaler, where distance is in micrometers
    """

    required_parameters = ParameterSet(
        {
            "constant_scaler": PyNNDistribution,  # the aditive constant of the decay
            "linear_scaler": PyNNDistribution,  # the scaler of the linear decay
        }
    )

    def distance_dependent_function(self, distance):
        return (
            self.parameters.linear_scaler.next() * distance
            + self.parameters.constant_scaler.next()
        )


class HyperbolicModularConnectorFunction(DistanceDependentModularConnectorFunction):
    """
    Corresponds to: exp(-alpha*sqrt(\theta^2 + distance^2)) , where distance is in micrometers
    And is the best fit I could so far find to the data from:
    Stepanyants, A., Hirsch, J. a, Martinez, L. M., Kisvárday, Z. F., Ferecskó, A. S., & Chklovskii, D. B. (2008).
    Local potential connectivity in cat primary visual cortex. Cerebral cortex, 18(1), 13–28. doi:10.1093/cercor/bhm027
    """

    required_parameters = ParameterSet(
        {"alpha": float, "theta": float}  # see description  # see description
    )

    def distance_dependent_function(self, distance):
        return numpy.exp(
            -numpy.multiply(
                self.parameters.alpha,
                numpy.sqrt(
                    numpy.power(self.parameters.theta, 2) + numpy.power(distance, 2)
                )
            )
        )
