# encoding: utf-8
"""
Sheet is an abstraction of a 2D continuous sheet of neurons, roughly
corresponding to the PyNN Population class with the added spatial structure.
"""
import numpy
import mozaik
from mozaik.framework.interfaces import MozaikComponent
from mozaik.framework import load_component
from mozaik.tools.distribution_parametrization import PyNNDistribution
from parameters import ParameterSet, UniformDist
from pyNN import space
from pyNN.errors import NothingToWriteError
from pyNN.parameters import Sequence
from string import Template
from neo.core.spiketrain import SpikeTrain
import quantities as pq

logger = mozaik.getMozaikLogger("Mozaik")

class Sheet(MozaikComponent):
    """
    Sheet is an abstraction of a 2D continuouse sheet of neurons, roughly
    corresponding to the PyNN Population class with the added spatial structure.

    The spatial position of all cells is kept within the PyNN Population object.
    Each sheet is assumed to be centered around (0,0) origin, corresponding to
    whatever excentricity the model is looking at. The internal representation
    of space is degrees of visual field. Thus x,y coordinates of a cell in all
    sheets correspond to the degrees of visual field this cell is away from the
    origin. However, the sheet and derived classes/methods are supposed to
    accept parameters in units that are most natural for the given parameter and
    recalculate these into the internal degrees of visual field representation.

    The units in visual space should be in degrees.
    The units for cortical space should be in μm.
    The units for time are in ms.
    """

    required_parameters = ParameterSet({
        'cell': ParameterSet({
            'model': str,  # the cell type of the sheet
            'params': ParameterSet,
            'initial_values': ParameterSet,
        }),

        'background_noise': ParameterSet({
            # the background noise to the population. This will be generated as Poisson
            # note that this is optimized for NEST !!!
            # it used native_cell_type("poisson_generator") to generate the noise

            'exc_firing_rate': float,
            'exc_weight': float,
            'inh_firing_rate': float,
            'inh_weight': float,
        }),
        'mpi_safe': bool,
        'artificial_stimulation' : bool, # Has to be set to True, if one wants to use 
                                         # stimulation beyond the non-specific 
                                         # one defined in background_noise parameters during 
                                         # the experiments. This is an efficiency flag 
        'name': str,
        'recorders' : ParameterSet
    })

    def __init__(self, model, parameters):
        MozaikComponent.__init__(self, model, parameters)
        self.sim = self.model.sim
        self.name = parameters.name  # the name of the population
        self.model.register_sheet(self)
        self._pop = None
        
        # We want to be able to define in cell.params the cell parameters as also PyNNDistributions so we can get variably parametrized populations
        # The problem is that the pyNN.Population can accept only scalar parameters. There fore we will remove from cell.params all parameters
        # that are PyNNDistributions, and will initialize them later just after the population is initialized (in property pop())
        self.dist_params = {}
        for k in self.parameters.cell.params.keys():
            if isinstance(self.parameters.cell.params[k],PyNNDistribution):
               self.dist_params[k]=self.parameters.cell.params[k]
               del self.parameters.cell.params[k]
        

    def setup_to_record_list(self):
        self.to_record = {}
        for k in  self.parameters.recorders.keys():
            recording_configuration = load_component(self.parameters.recorders[k].component)
            l = recording_configuration(self,self.parameters.recorders[k].params).generate_idd_list_of_neurons()
            if isinstance(self.parameters.recorders[k].variables,str):
               self.parameters.recorders[k].variables = [self.parameters.recorders[k].variables]
               
            for var in self.parameters.recorders[k].variables:
                self.to_record[var] = list(set(self.to_record.get(var,[])) | set(l))
        #convert ids to indexes
        for k in self.to_record.keys():
            idds = self.pop.all_cells.astype(int)
            self.to_record[k] = [numpy.flatnonzero(idds == idd)[0] for idd in self.to_record[k]]
            
    def size_in_degrees(self):
        """Returns the x, y size in degrees of visual field of the given area"""
        raise NotImplementedError
        pass

    def pop():
        doc = "PyNN population"

        def fget(self):
            if not self._pop:
                logger.error('Population have not been yet set in sheet: ' +  self.name + '!')
            return self._pop

        def fset(self, value):
            if self._pop:
                raise Exception("Error population has already been set. It is not allowed to do this twice!")
            self._pop = value
            self._neuron_annotations = [{} for i in xrange(0, len(value))]
            self.setup_background_noise()
            self.setup_to_record_list()
            self.setup_initial_values()
        return locals()

    pop = property(**pop())  # this will be populated by PyNN population, in the derived classes

    def add_neuron_annotation(self, neuron_number, key, value, protected=True):
        if not self._pop:
            logger.error('Population has not been yet set in sheet: ' + self.name + '!')
        if (key in self._neuron_annotations[neuron_number] and self._neuron_annotations[neuron_number][key][0]):
            print self._neuron_annotations[neuron_number]
            logger.warning('The annotation<' + '> for neuron ' + str(neuron_number) + ' is protected. Annotation not updated')
        else:
            self._neuron_annotations[neuron_number][key] = (protected, value)

    def get_neuron_annotation(self, neuron_number, key):
        if not self._pop:
            logger.error('Population has not been yet set in sheet: ' + self.name + '!')
        return self._neuron_annotations[neuron_number][key][1]

    def get_neuron_annotations(self):
        if not self._pop:
            logger.error('Population has not been yet set in sheet: ' +  self.name + '!')

        anns = []
        for i in xrange(0, len(self.pop)):
            d = {}
            for (k, v) in self._neuron_annotations[i].items():
                d[k] = v[1]
            anns.append(d)
        return anns

    def describe(self, template='default', render=lambda t, c: Template(t).safe_substitute(c)):
        context = {
            'name': self.__class__.__name__,
        }
        if template:
            render(template, context)
        else:
            return context

    def record(self):
        if self.to_record != None:
            for variable in self.to_record.keys():
                cells = self.to_record[variable]
                if cells != 'all':
                    self.pop[cells].record(variable)
                else:
                    self.pop.record(variable)

    def write_neo_object(self, stimulus_duration=None):
        """
        Retrieve recorded data from pyNN.

        In case offset is set it means we want to keep only data after time
        offset.
        """
        try:
            block = self.pop.get_data(['spikes', 'v', 'gsyn_exc', 'gsyn_inh'],
                                      clear=True)
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        
        s = block.segments[-1]
        s.annotations["sheet_name"] = self.name

        # lets sort spike train so that it is ordered by IDs and thus hopefully
        # population indexes
        def compare(a, b):
            return cmp(a.annotations['source_id'], b.annotations['source_id'])

        s.spiketrains = sorted(s.spiketrains, compare)
        if stimulus_duration != None:        
           for i in xrange(0, len(s.spiketrains)):
               s.spiketrains[i] -= s.spiketrains[i].t_start
               s.spiketrains[i].t_stop -= s.spiketrains[i].t_start
               s.spiketrains[i].t_start = 0 * pq.ms
               
           for i in xrange(0, len(s.analogsignalarrays)):
               s.analogsignalarrays[i].t_start = 0 * pq.ms
       
        return s

    def prepare_input(self, duration, offset,exc_spiking_stimulation,inh_spiking_stimulation):
        from NeuroTools import stgen
        if self.parameters.mpi_safe or self.parameters.artificial_stimulation:
            if (self.parameters.background_noise.exc_firing_rate != 0 and self.parameters.background_noise.exc_weight != 0) or self.parameters.artificial_stimulation:
                idds = self.pop.all_cells.astype(int)
                for i in numpy.nonzero(self.pop._mask_local)[0]:
                    pp = []
                    if (self.parameters.background_noise.exc_firing_rate != 0 and self.parameters.background_noise.exc_weight != 0):
                        pp = stgen.StGen(seed=i).poisson_generator(
                                    rate=self.parameters.background_noise.exc_firing_rate,
                                    t_start=0,
                                    t_stop=duration).spike_times
                    if self.parameters.artificial_stimulation and exc_spiking_stimulation!=None and (exc_spiking_stimulation[0] == "all" or (idds[i] in exc_spiking_stimulation[0])):
                       pp.extend(exc_spiking_stimulation[1](duration)) 
                    self.ssae[i].set_parameters(spike_times=Sequence(offset + numpy.array(pp)))

            if (self.parameters.background_noise.inh_firing_rate != 0 and self.parameters.background_noise.inh_weight) != 0 or self.parameters.artificial_stimulation:
                idds = self.pop.all_cells.astype(int)
                for i in numpy.nonzero(self.pop._mask_local)[0]:
                    pp = []
                    if (self.parameters.background_noise.inh_firing_rate != 0 and self.parameters.background_noise.inh_weight != 0):
                        pp = stgen.StGen(seed=2*i).poisson_generator(
                                    rate=self.parameters.background_noise.inh_firing_rate,
                                    t_start=self.model.simulator_time,
                                    t_stop=self.model.simulator_time + duration).spike_times
                    if self.parameters.artificial_stimulation and inh_spiking_stimulation!=None and (inh_spiking_stimulation[0] == "all" or (idds[i] in inh_spiking_stimulation[0])):
                       pp.extend(self.model.simulator_time+ inh_spiking_stimulation[1](duration)) 
                    self.ssai[i].set_parameters(spike_times=Sequence(offset + numpy.array(pp)))

    def setup_background_noise(self):
        from pyNN.nest import native_cell_type
        exc_syn = self.sim.StaticSynapse(weight=self.parameters.background_noise.exc_weight)
        inh_syn = self.sim.StaticSynapse(weight=self.parameters.background_noise.inh_weight)
        if not self.parameters.mpi_safe:
            if (self.parameters.background_noise.exc_firing_rate != 0
                  or self.parameters.background_noise.exc_weight != 0):
                np_exc = self.sim.Population(
                                1, native_cell_type("poisson_generator"),
                                {'rate': self.parameters.background_noise.exc_firing_rate})
                self.sim.Projection(
                                np_exc, self.pop,
                                self.sim.AllToAllConnector(),
                                synapse_type=exc_syn,
                                receptor_type='excitatory')

            if (self.parameters.background_noise.inh_firing_rate != 0
                  or self.parameters.background_noise.inh_weight != 0):
                np_inh = self.sim.Population(
                                1, native_cell_type("poisson_generator"),
                                {'rate': self.parameters.background_noise.inh_firing_rate})
                self.sim.Projection(
                                np_inh, self.pop,
                                self.sim.AllToAllConnector(),
                                synapse_type=inh_syn,
                                receptor_type='inhibitory')
        
        if self.parameters.mpi_safe or self.parameters.artificial_stimulation:
            if (self.parameters.background_noise.exc_firing_rate != 0
                  or self.parameters.background_noise.exc_weight != 0 or self.parameters.artificial_stimulation):
                        self.ssae = self.sim.Population(self.pop.size,
                                                        self.model.sim.SpikeSourceArray())
                        self.sim.Projection(self.ssae, self.pop,
                                            self.sim.OneToOneConnector(),
                                            synapse_type=exc_syn,
                                            receptor_type='excitatory')

            if (self.parameters.background_noise.inh_firing_rate != 0
                  or self.parameters.background_noise.inh_weight != 0 or self.parameters.artificial_stimulation):
                        self.ssai = self.sim.Population(self.pop.size,
                                                        self.model.sim.SpikeSourceArray())
                        self.sim.Projection(self.ssai, self.pop,
                                            self.sim.OneToOneConnector(),
                                            synapse_type=inh_syn,
                                            receptor_type='inhibitory')

    def setup_initial_values(self):
        # Initial state variables
        self.pop.initialize(**self.parameters.cell.initial_values)
        # Variable cell parameters
        self.pop.set(**self.dist_params)
        #for k,v in self.dist_params.iteritems():
        #    self.pop.rset(k,v)


class RetinalUniformSheet(Sheet):
    """
    Retinal sheet is organized on a grid
    """
    required_parameters = ParameterSet({
        'sx': float,  # degrees, x size of the region
        'sy': float,  # degrees, y size of the region
        'density': int,  # neurons along each axis
    })

    def __init__(self, model, parameters):
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__, int(parameters.density * parameters.density)))
        Sheet.__init__(self, model, parameters)
        #rs = space.RandomStructure(boundary=space.Cuboid(parameters.sx, parameters.sy, 0),
        #                           origin=(0.0, 0.0, 0.0),
        #                           rng=mozaik.rng)
        
        rs = space.Grid2D(aspect_ratio=1, dx=parameters.sx/parameters.density, dy=parameters.sy/parameters.density, x0=-parameters.sx/2,y0=-parameters.sy/2,z=0.0)
        
        self.pop = self.sim.Population(int(parameters.density * parameters.density),
                                       getattr(self.model.sim, self.parameters.cell.model),
                                       self.parameters.cell.params,
                                       structure=rs,
                                       initial_values=self.parameters.cell.initial_values,
                                       label=self.name)

    def size_in_degrees(self):
        return (self.parameters.sx, self.parameters.sy)


class SheetWithMagnificationFactor(Sheet):

    required_parameters = ParameterSet({
        'magnification_factor': float,  # μm / degree
        'sx': float,      # μm, x size of the region
        'sy': float,      # μm, y size of the region
    })

    def __init__(self, model, parameters):
        """
        """
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__, int(parameters.sx*parameters.sy/1000000*parameters.density)))
        Sheet.__init__(self, model, parameters)
        self.magnification_factor = parameters.magnification_factor

    def vf_2_cs(self, degree_x, degree_y):
        """
        vf_2_cs converts the position (degree_x, degree_y) in visual field to
        position in cortical space (in μm) given the magnification_factor.
        """
        return (degree_x * self.magnification_factor,
                degree_y * self.magnification_factor)

    def cs_2_vf(self, micro_meters_x, micro_meters_y):
        """
        cs_2_vf converts the position (micro_meters_x, micro_meters_y) in
        cortical space to the position in the visual field (in degrees) given
        the magnification_factor
        """
        return (micro_meters_x / self.magnification_factor,
                micro_meters_y / self.magnification_factor)

    def dvf_2_dcs(self, distance_vf):
        """
        dvf_2_dcs converts the distance in visual space to the distance in
        cortical space given the magnification_factor
        """
        return distance_vf * self.magnification_factor

    def size_in_degrees(self):
        return self.cs_2_vf(self.parameters.sx, self.parameters.sy)


class CorticalUniformSheet(SheetWithMagnificationFactor):

    required_parameters = ParameterSet({
        'density': float,  # neurons/(mm^2)
    })

    def __init__(self, model, parameters):
        SheetWithMagnificationFactor.__init__(self, model, parameters)
        dx, dy = self.cs_2_vf(parameters.sx, parameters.sy)

        rs = space.RandomStructure(boundary=space.Cuboid(dx, dy, 0),
                                   origin=(0.0, 0.0, 0.0),
                                   rng=mozaik.rng)

        self.pop = self.sim.Population(int(parameters.sx*parameters.sy/1000000*parameters.density),
                                       getattr(self.model.sim, self.parameters.cell.model),
                                       self.parameters.cell.params,
                                       structure=rs,
                                       initial_values=self.parameters.cell.initial_values,
                                       label=self.name)
