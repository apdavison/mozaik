# encoding: utf-8
"""
Mozaik connector interface.
"""
import math
import numpy
import pylab
import mozaik
import time
from pylab import griddata
from mozaik.framework.interfaces import Connector
from mozaik.framework.sheets import SheetWithMagnificationFactor
from parameters import ParameterSet, ParameterDist
from mozaik.tools.misc import sample_from_bin_distribution, normal_function
from collections import Counter
from pyNN import random, space

logger = mozaik.getMozaikLogger("Mozaik")


class MozaikConnector(Connector):
    """
    An abstract interface class for Connectors in mozaik. Each mozaik connector should derive from this class and implement 
    the _connect function. The usage is: create the instance of MozaikConnector and call connect() to realize the connections.
    """
    required_parameters = ParameterSet({
            'target_synapses' : str,
            'short_term_plasticity': ParameterSet({
                    'u': float, 
                    'tau_rec': float, 
                    'tau_fac': float,
                    'tau_psc': float
            }),
    
    })
    
    def __init__(self, network, name,source, target, parameters):
      Connector.__init__(self, network, name, source,target,parameters)
    
    
    def init_short_term_plasticity(self,weights=None,delays=None):
      if not self.parameters.short_term_plasticity != None:
        short_term_plasticity = None
      else:
        #self.short_term_plasticity = self.sim.SynapseDynamics(fast=self.sim.TsodyksMarkramMechanism(**self.parameters.short_term_plasticity_params)) 
        #short_term_plasticity = self.sim.NativeSynapseType(nest_name="tsodyks_synapse", default_parameters = self.parameters.short_term_plasticity,weights=weights,delay=delays)
        short_term_plasticity = self.sim.native_synapse_type("tsodyks_synapse")(weight=weights,delay=delays,**self.parameters.short_term_plasticity)                   
      return short_term_plasticity
        
    def connect(self):
          t0 = time.time()
          self._connect()
          connect_time = time.time() - t0
          logger.info('Connector %s took %.0fs to compute' % (self.__class__.__name__,connect_time))
            
        
    def _connect(self):
      raise NotImplementedError

    def connection_field_plot_continuous(self, index, afferent=True, density=30):
        weights = self.proj.getWeights(format='list')
        #print numpy.shape(weights)
        if afferent:
            idx = numpy.flatnonzero(weights[:,1]==index)
            x = self.proj.pre.positions[0][weights[idx,0]]
            y = self.proj.pre.positions[1][weights[idx,0]]
            w = weights[idx,2]
        else:
            idx = numpy.flatnonzero(weights[:,0]==index)
            x = self.proj.post.positions[0][weights[idx,1]]
            y = self.proj.post.positions[1][weights[idx,1]]
            w = weights[idx,2]

        xi = numpy.linspace(min(x), max(x), 100)
        yi = numpy.linspace(min(y), max(y), 100)
        zi = griddata(x, y, w, xi, yi)
        pylab.figure()
        #pylab.imshow(zi)
        pylab.scatter(x,y,marker='o',c=w,s=50)
        pylab.xlim(-self.source.parameters.sx/2,self.source.parameters.sx/2)
        pylab.ylim(-self.source.parameters.sy/2,self.source.parameters.sy/2)
        pylab.colorbar()
        pylab.title('Connection field from %s to %s of neuron %d' % (self.source.name,
                                                                     self.target.name,
                                                                     index))
        #pylab.colorbar()

    def store_connections(self, datastore):
        from mozaik.analysis.analysis_data_structures import Connections
        
        weights = numpy.array(self.proj.getWeights(format='list', gather=True))
        delays = numpy.array(self.proj.getDelays(format='list', gather=True))
        datastore.add_analysis_result(
            Connections(weights,delays,
                        proj_name=self.name,
                        source_name=self.source.name,
                        target_name=self.target.name,
                        analysis_algorithm='connection storage'))


class SpecificArborization(MozaikConnector):
    """
    Generic connector which gets directly list of connections as the list of
    quadruplets as accepted by the pyNN FromListConnector.

    This connector cannot be parametrized directly via the parameter file
    because that does not support list of tuples.
    """

    required_parameters = ParameterSet({
        'weight_factor': float,  # the overall (sum) weight that a single target neuron should receive
    })

    def __init__(self, network, source, target, connection_matrix,delay_matrix, parameters, name):
        MozaikConnector.__init__(self, network, name, source,
                                             target, parameters)
        self.connection_matrix = connection_matrix
        self.delay_matrix = delay_matrix

    def _connect(self):
        X = numpy.zeros(self.connection_matrix.shape)
        Y = numpy.zeros(self.connection_matrix.shape)
        
        for x in xrange(0,X.shape[0]):
            for y in xrange(0,X.shape[1]):
                X[x][y] = x
                Y[x][y] = y
        
        for i in xrange(0,self.target.pop.size):
            self.connection_matrix[:,i] = self.connection_matrix[:,i] / numpy.sum(self.connection_matrix[:,i])*self.parameters.weight_factor

        self.connection_list = zip(numpy.array(X).flatten(),numpy.array(Y).flatten(),self.connection_matrix.flatten(),self.delay_matrix.flatten())
        # get rid of very weak synapses
        z = numpy.max(self.connection_matrix.flatten())
        self.connection_list = [(int(a),int(b),c,d) for (a,b,c,d) in self.connection_list if c>(z/100.0)]
        method = self.sim.FromListConnector(self.connection_list)
        self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_type=self.init_short_term_plasticity(),
                                label=self.name,
                                rng=None,
                                receptor_type=self.parameters.target_synapses)


class SpecificProbabilisticArborization(MozaikConnector):
    """
    Generic connector which gets directly list of connections as the list
    of quadruplets as accepted by the pyNN FromListConnector.

    It interprets the weights as proportional probabilities of connectivity,
    and for each neuron out connections it samples num_samples of
    connections that actually get realized according to these weights.
    Each such sample connections will have weight equal to
    weight_factor/num_samples but note that there can be multiple
    connections between a pair of neurons in this sample (in which case the
    weights are set to the multiple of the base weights times the number of
    occurrences in the sample).

    This connector cannot be parameterized directly via the parameter file
    because that does not support list of tuples.
    """

    required_parameters = ParameterSet({
        'weight_factor': float,  # the overall strength of synapses in this connection per neuron (in µS) (i.e. the sum of the strength of synapses in this connection per target neuron)
        'num_samples': int
    })

    def __init__(self, network, source, target, connection_matrix,delay_matrix, parameters, name):
        MozaikConnector.__init__(self, network, name, source,target, parameters)
        self.connection_matrix = connection_matrix
        self.delay_matrix = delay_matrix

    def _connect(self):
        weights = self.connection_matrix
        delays = self.delay_matrix
        cl = []
        
        for i in xrange(0,self.target.pop.size):
            co = Counter(sample_from_bin_distribution(weights[:,i].flatten(), int(self.parameters.num_samples)))
            cl.extend([(int(k),int(i),self.parameters.weight_factor*co[k]/self.parameters.num_samples,delays[k][i]) for k in co.keys()])
        
        method = self.sim.FromListConnector(cl)
        
        print "Source length: ", len(self.source.pop)
        print "Target length: ", len(self.target.pop)
        print "Source max index: ",max(numpy.array(cl)[:,0].flatten())
        print "Target max index: ",max(numpy.array(cl)[:,1].flatten())
        print "Source min index: ",min(numpy.array(cl)[:,0].flatten())
        print "Target min index: ",min(numpy.array(cl)[:,1].flatten())
        self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_type=self.init_short_term_plasticity(),
                                label=self.name,
                                receptor_type=self.parameters.target_synapses)
                  


