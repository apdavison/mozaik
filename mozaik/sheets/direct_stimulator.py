# encoding: utf-8
"""
This file contains the API for direct stimulation of neurons. 
By direct stimulation here we mean a artificial stimulation that 
would happen during electrophisiological experiment - such a injection
of spikes/currents etc into cells. In mozaik this happens at population level - i.e.
each direct stimulator specifies how the given population is stimulated. In general each population can have several
stimultors.
"""
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
import numpy
import numpy.random
import mozaik
from mozaik.tools.stgen import StGen
from mozaik import load_component
from pyNN.parameters import Sequence
from mozaik import load_component
import math
from mozaik.tools.circ_stat import circular_dist ,circ_mean
import pylab

logger = mozaik.getMozaikLogger()

class DirectStimulator(ParametrizedObject):
      """
      The API for direct stimulation.
      The DirectStimulator specifies how are cells in the assigned population directly stimulated. 
        
      Parameters
      ----------
      parameters : ParameterSet
                   The dictionary of required parameters.
                    
      sheet : Sheet
              The sheet in which to stimulate neurons.
              
      Notes
      -----
      
      By defalut the direct stimulation should ensure that it is mpi-safe - this is especially crucial for 
      stimulators that involve source of randomnes. However, the DirectSimulators also can inspect the mpi_safe
      value of the population to which they are assigned, and if it is False they can switch to potentially 
      more efficient implementation that will however not be reproducible across multi-process simulations.
      
      Important: the functiona inactivate should only temporarily inactivate the stimulator, a subsequent call to prepare_stimulation
      should activate the stimulator back!
      """

      def __init__(self, sheet, parameters):
          ParametrizedObject.__init__(self, parameters)
          self.sheet = sheet
     
      def prepare_stimulation(self,duration,offset):
          """
          Prepares the stimulation during the next period of model simulation lasting `duration` seconds.
          
          Parameters
          ----------
          duration : double (seconds)
                     The period for which to prepare the stimulation
          
          offset : double (seconds)
                   The current simulator time.
                     
          """
          raise NotImplemented 
          
      def inactivate(self,offset):
          """
          Ensures any influences of the stimulation are inactivated for subsequent simulation of the model.

          Parameters
          ----------
          offset : double (seconds)
                   The current simulator time.
          
          Note that a subsequent call to prepare_stimulation should 'activate' the stimulator again.
          """
          raise NotImplemented 



class BackgroundActivityBombardment(DirectStimulator):
    """
    The BackgroundActivityBombardment simulates the poisson distrubated background bombardment of spikes onto a 
    neuron due to the other 'unsimulated' neurons in its pre-synaptic population.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    exc_firing_rate : float
                     The firing rate of external neurons sending excitatory inputs to each neuron of this sheet.
 
    inh_firing_rate : float
                     The firing rate of external neurons sending inhibitory inputs to each neuron of this sheet.
    
    exc_weight : float
                     The weight of the synapses for the excitatory external Poisson input.    

    inh_weight : float
                     The weight of the synapses for the inh external Poisson input.    
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!
    """
    
    
    required_parameters = ParameterSet({
            'exc_firing_rate': float,
            'exc_weight': float,
            'inh_firing_rate': float,
            'inh_weight': float,
    })
        
        
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=self.sheet.model.parameters.min_delay)
        inh_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.inh_weight,delay=self.sheet.model.parameters.min_delay)
        
        if not self.sheet.parameters.mpi_safe:
            from pyNN.nest import native_cell_type        
            if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                self.np_exc = self.sheet.sim.Population(1, native_cell_type("poisson_generator"),{'rate': 0})
                self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                self.np_inh = self.sheet.sim.Population(1, native_cell_type("poisson_generator"),{'rate': 0})
                self.sheet.sim.Projection(self.np_inh, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=inh_syn,receptor_type='inhibitory')
        
        else:
            if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                        self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_seeds((self.sheet.pop.size,))
                        self.stgene = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                        self.ssai = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_seeds((self.sheet.pop.size,))
                        self.stgeni = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssai, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=inh_syn,receptor_type='inhibitory')

    def prepare_stimulation(self,duration,offset):
        if not self.sheet.parameters.mpi_safe:
           self.np_exc[0].set_parameters(rate=self.parameters.exc_firing_rate)
           self.np_inh[0].set_parameters(rate=self.parameters.inh_firing_rate)
        else:
           if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                for j,i in enumerate(numpy.nonzero(self.sheet.pop._mask_local)[0]):
                    pp = self.stgene[j].poisson_generator(rate=self.parameters.exc_firing_rate,t_start=0,t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssae[i].set_parameters(spike_times=Sequence(a.astype(float)))
               
           if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                for j,i in enumerate(numpy.nonzero(self.sheet.pop._mask_local)[0]):
                    pp = self.stgene[j].poisson_generator(rate=self.parameters.inh_firing_rate,t_start=0,t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssai[i].set_parameters(spike_times=Sequence(a.astype(float)))
        

        
    def inactivate(self,offset):        
        if not self.sheet.parameters.mpi_safe:
           self.np_exc[0].set_parameters(rate=0)
           self.np_inh[0].set_parameters(rate=0)
            

class Kick(DirectStimulator):
    """
    This stimulator sends a kick of excitatory spikes into a specified subpopulation of neurons.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    exc_firing_rate : float
                     The firing rate of external neurons sending excitatory inputs to each neuron of this sheet.
 
    exc_weight : float
                     The weight of the synapses for the excitatory external Poisson input.    
    
    drive_period : float
                     Period over which the Kick will deposit the full drive defined by the exc_firing_rate, after this time the 
                     firing rates will be linearly reduced to reach zero at the end of stimulation.

    population_selector : ParemeterSet
                        Defines the population selector and its parameters to specify to which neurons in the population the 
                        background activity should be applied. 
                     
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!
    """
    
    
    required_parameters = ParameterSet({
            'exc_firing_rate': float,
            'exc_weight': float,
            'drive_period' : float,
            'population_selector' : ParameterSet({
                    'component' : str,
                    'params' : ParameterSet
                    
            })
            
    })

    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        population_selector = load_component(self.parameters.population_selector.component)
        self.ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        d = dict((j,i) for i,j in enumerate(self.sheet.pop.all_cells))
        self.to_stimulate_indexes = [d[i] for i in self.ids]
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=self.sheet.model.parameters.min_delay)
        if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
            self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
            seeds=mozaik.get_seeds((self.sheet.pop.size,))
            self.stgene = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in self.to_stimulate_indexes]
            self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory') 

    def prepare_stimulation(self,duration,offset):
        if (self.parameters.exc_firing_rate != 0 and self.parameters.exc_weight != 0):
           for j,i in enumerate(self.to_stimulate_indexes):
               if self.parameters.drive_period < duration:
                   z = numpy.arange(self.parameters.drive_period+0.001,duration-100,10)
                   times = [0] + z.tolist() 
                   rate = [self.parameters.exc_firing_rate] + ((1.0-numpy.linspace(0,1.0,len(z)))*self.parameters.exc_firing_rate).tolist()
               else:
                   times = [0]  
                   rate = [self.parameters.exc_firing_rate] 
               pp = self.stgene[j].inh_poisson_generator(numpy.array(rate),numpy.array(times),t_stop=duration).spike_times
               a = offset + numpy.array(pp)
               self.ssae[i].set_parameters(spike_times=Sequence(a.astype(float)))

    def inactivate(self,offset):        
        pass


class Depolarization(DirectStimulator):
    """
    This stimulator injects a constant current into neurons in the population.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    current : float (mA)
                     The current to inject into neurons.

    population_selector : ParemeterSet
                        Defines the population selector and its parameters to specify to which neurons in the population the 
                        background activity should be applied. 
                     
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!
    """
    
    
    required_parameters = ParameterSet({
            'current': float,
            'population_selector' : ParameterSet({
                    'component' : str,
                    'params' : ParameterSet
                    
            })
            
    })
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        
        population_selector = load_component(self.parameters.population_selector.component)
        self.ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        self.scs = self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
        for cell in self.sheet.pop.all_cells:
            cell.inject(self.scs)

    def prepare_stimulation(self,duration,offset):
        self.scs.set_parameters(times=[offset+self.sheet.sim.state.dt*2], amplitudes=[self.parameters.current])
        
    def inactivate(self,offset):
        self.scs.set_parameters(times=[offset+self.sheet.sim.state.dt*2], amplitudes=[0.0])


class LocalStimulatorArray(DirectStimulator):
    """
    This class assumes there is a regular grid of stimulators (parameters `size` and `spacing` control
    the geometry of the grid), with each stimulator stimulating indiscriminately the local population 
    of neurons in the given sheet. The intensity of stimulation falls of as Gaussian (parameter `itensity_fallof`), 
    and the stimulations from different stimulators add up linearly. 

    The temporal profile of the stimulator is given by function specified in the parameter `stimulating_signal`.
    This function receives the population to be stimulated, the list of coordinates of the stimulators, and any extra user parameters 
    specified in the parameter `stimulating_signal_parameters`. It should return the list of currents that 
    flow out of the stimulators. The function specified in `stimulating_signal` should thus look like this:

    def stimulating_signal_function(population,list_of_coordinates, parameters)

    The rate current changes that the stimulating_signal_function returns is specified by the `current_update_interval`
    parameter.

    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    size : float (μm) 
                     The size of the stimulator grid

    spacing : float (μm)
                     The distance between stimulators (the number of stimulators will thus be (size/distance)^2)

    itensity_fallof : float (μm)
                     The sigma of the Gaussian of the stimulation itensity falloff.

    stimulating_signal : str
                     The python path to a function that defines the stimulation.

    stimulating_signal_parameters : ParameterSet
                     The parameters passed to the function specified in  `stimulating_signal`

    current_update_interval : float
                     The interval at which the current is updated. Thus the length of the stimulation is current_update_interval times
                     the number of current values returned by the function specified in the `stimulating_signal` parameter.

    Notes
    -----

    For now this is not mpi optimized.
    """
    
    
    required_parameters = ParameterSet({
            'size': float,
            'spacing' : float,
            'itensity_fallof' : float,
            'stimulating_signal' : str,
            'stimulating_signal_parameters' : ParameterSet,
            'current_update_interval' : float,
    })
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)

        assert math.fmod(self.parameters.size,self.parameters.spacing) < 0.000000001 , "Error the size has to be multiple of spacing!"
        
        axis_coors = numpy.arange(0,self.parameters.size,self.parameters.spacing) - self.parameters.size/2.0 + self.parameters.spacing/2.0
        stimulator_coordinates = numpy.meshgrid(axis_coors,axis_coors)

        pylab.figure(figsize=(24,6))
      
        # now let's calculate mixing weights, this will be a matrix nxm where n is 
        # the number of neurons in the population and m is the number of stimulators
        mixing_weights = []
        x =  stimulator_coordinates[0].flatten()
        y =  stimulator_coordinates[1].flatten()
        for i in xrange(0,self.sheet.pop.size):
            xx,yy = self.sheet.pop.positions[0][i],self.sheet.pop.positions[1][i]
            xx,yy = self.sheet.vf_2_cs(xx,yy)
            mixing_weights.append(numpy.exp(-0.5  * (numpy.power(x - xx,2)  + numpy.power(y-yy,2)) / numpy.power(self.parameters.itensity_fallof,2)))
        assert numpy.shape(mixing_weights) == (self.sheet.pop.size,int(self.parameters.size/self.parameters.spacing) * int(self.parameters.size/self.parameters.spacing))

        signal_function = load_component(self.parameters.stimulating_signal)
        stimulator_signals = signal_function(sheet,zip(x,y),self.parameters.current_update_interval,self.parameters.stimulating_signal_parameters)
        assert numpy.shape(stimulator_signals)[0] == numpy.shape(mixing_weights)[1] , "ERROR: stimulator_signals and mixing_weights do not have matching sizes:" + str(numpy.shape(stimulator_signals)) + " " +str(numpy.shape(mixing_weights))

        self.mixed_signals = numpy.dot(mixing_weights,stimulator_signals)
        pylab.subplot(144)
        pylab.scatter(self.sheet.pop.positions[0],self.sheet.pop.positions[1],c=numpy.squeeze(numpy.mean(self.mixed_signals,axis=1)),cmap='gray',vmin=0)
        pylab.colorbar()
        pylab.savefig('LocalStimulatorArrayTest.png')
        assert numpy.shape(self.mixed_signals) == (self.sheet.pop.size,numpy.shape(stimulator_signals)[1]), "ERROR: mixed_signals doesn't have the desired size:" + str(numpy.shape(self.mixed_signals)) + " vs " +str((self.sheet.pop.size,numpy.shape(stimulator_signals)[1]))
        
        self.stimulation_duration = numpy.shape(self.mixed_signals)[1] * self.parameters.current_update_interval

        self.scs = [self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0]) for cell in self.sheet.pop.all_cells] 
        for cell,scs in zip(self.sheet.pop.all_cells,self.scs):
            cell.inject(scs)

    def prepare_stimulation(self,duration,offset):
        assert self.stimulation_duration == duration, "stimulation_duration != duration :"  + str(self.stimulation_duration) + " " + str(duration)
        times = numpy.arange(0,self.stimulation_duration,self.parameters.current_update_interval) + offset
        times[0] = times[0] + 3*self.sheet.sim.state.dt
        for i in xrange(0,len(self.scs)):
            self.scs[i].set_parameters(times=Sequence(times), amplitudes=Sequence(self.mixed_signals[i,:].flatten()))
            #HAAAAAAAAAAAAACK
            #self.scs[i].set_parameters(times=Sequence([times[0]]), amplitudes=Sequence([self.mixed_signals[i,:].flatten()[0]]))
        
    def inactivate(self,offset):
        for scs in self.scs:
            scs.set_parameters(times=[offset+3*self.sheet.sim.state.dt], amplitudes=[0.0])


def test_stimulating_function(sheet,coordinates,current_update_interval,parameters):
    z = sheet.pop.all_cells.astype(int)
    vals = numpy.array([sheet.get_neuron_annotation(i,'LGNAfferentOrientation') for i in xrange(0,len(z))])
    two_sigma_squared = 2*parameters.sigma * parameters.sigma 

    mean_orientations = []

    px,py = sheet.vf_2_cs(sheet.pop.positions[0],sheet.pop.positions[1])

    pylab.subplot(141)
    pylab.scatter(px,py,c=vals/numpy.pi,cmap='hsv')
    for sx,sy in coordinates:

             lhi_current_c=numpy.sum(numpy.exp(-((sx-px)*(sx-px)+(sy-py)*(sy-py))/(two_sigma_squared))*numpy.cos(2*vals))
             lhi_current_s=numpy.sum(numpy.exp(-((sx-px)*(sx-px)+(sy-py)*(sy-py))/(two_sigma_squared))*numpy.sin(2*vals))
             mean_orientations.append(circ_mean(vals,weights=numpy.exp(-((sx-px)*(sx-px)+(sy-py)*(sy-py))/(two_sigma_squared)),high=numpy.pi)[0])

    pylab.subplot(142)
    pylab.scatter([a[0] for a in coordinates],[a[1] for a in coordinates],c=numpy.array(mean_orientations),cmap='hsv')

    signals = []

    for i in xrange(0,len(coordinates)):
        signals.append(parameters.scale*numpy.array([numpy.exp(-numpy.power(circular_dist(parameters.orientation,mean_orientations[i],numpy.pi),2)/parameters.sharpness) for tmp in xrange(parameters.duration/current_update_interval)]))

    pylab.subplot(143)
    pylab.scatter([a[0] for a in coordinates],[a[1] for a in coordinates],c=numpy.squeeze(numpy.mean(signals,axis=1)),cmap='gray')
    pylab.colorbar()
    return  signals