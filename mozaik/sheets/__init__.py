# encoding: utf-8
"""
Module containing the implementation of sheets - one of the basic building blocks of *mozaik* models.
"""

from string import Template
import logging
logging.basicConfig(filename='mozaik.log', level=logging.DEBUG)
from collections import OrderedDict
from neo.core.spiketrain import SpikeTrain
from parameters import ParameterSet, UniformDist
from pyNN import space
from pyNN.errors import NothingToWriteError
import numpy
import quantities as pq

import mozaik
from .. import load_component
from ..core import BaseComponent
from ..tools.distribution_parametrization import PyNNDistribution

logger = logging.getLogger(__name__)


class Sheet(BaseComponent):
    """
    Sheet is an abstraction of a volume of neurons positioned in a physical space.

    It roughly corresponding to the PyNN Sheet class with the added spatial structure 
    and various helper functions specific to the mozaik integration. The spatial position 
    of all cells is kept within the PyNN Sheet object and are assumed to be in μm.
       
    Other parameters
    ----------------
    
    cell : ParameterSet
         The parametrization of the cell model that all neurons in this sheet will have.
         
    cell.model : str
               The name of the cell model.
    
    cell.params : ParameterSet
               The set of parameters that the given model requires.
               
    cell.initial_values : ParameterSet
                   It can contain a ParameterSet containing the initial values for some of the parameters in cell.params
                   
    mpi_safe : bool
             Whether to set the sheet up to be reproducible in MPI environment. 
             This is computationally less efficient that if it is set to false, but it will
             guaruntee the same results irrespective of the number of MPI process used.
             
    artificial_stimulators : ParameterSet
             Contains a list of ParameterSet objects, one per each :class:`.direct_stimulator.DirectStimulator` object to be created.
             Each contains a parameter 'component' that specifies which :class:`.direct_stimulator.DirectStimulator` to use, and  a 
             parameter 'params' which is a ParameterSet to be passed to that `DirectStimulator`.
    
    name : str
        Name of the sheet.
    
    recorders : ParameterSet
                Parametrization of recorders in this sheet. The recorders ParameterSet will contain as keys the names
                of the different recording configuration user want to have in this sheet. For the format of each recording configuration see notes.

    recording_interval : float (ms)
                The interval at which analog signals in this sheet will be recorded. 

    Notes
    -----
    
    Each recording configuration requires the following parameters:
    
    *variables* 
        tuple of strings specifying the variables to measure (allowd values are: 'spikes' , 'v','gsyn_exc' , 'gsyn_inh' )
    *componnent* 
        the path to the :class:`mozaik.sheets.population_selector.PopulationSelector` class
    *params*
        a ParameterSet containing the parameters for the given :class:`mozaik.sheets.population_selector.PopulationSelector` class 
    """

    required_parameters = ParameterSet(
        {
            "cell": ParameterSet(
                {
                    "model": str,  # the cell type of the sheet
                    "params": ParameterSet,
                    "initial_values": ParameterSet
                }
            ),
            "mpi_safe": bool,
            "artificial_stimulators": ParameterSet,
            "name": str,
            "recorders": ParameterSet,
            "recording_interval": float
        }
    )

    def __init__(self, model, size_x, size_y, parameters):
        BaseComponent.__init__(self, model, parameters)
        self.sim = self.model.sim
        # self.dt = self.sim.state.dt
        self.name = parameters.name  # the name of the population
        self.model.register_sheet(self)
        self._pop = None
        self.size_x = size_x
        self.size_y = size_y
        self.msc = 0
        # We want to be able to define in cell.params the cell parameters as also PyNNDistributions so we can get variably parametrized populations
        # The problem is that the pyNN.Population can accept only scalar parameters. There fore we will remove from cell.params all parameters
        # that are PyNNDistributions, and will initialize them later just after the population is initialized (in property pop())
        self.dist_params = {}
        for k in self.parameters.cell.params:
            if isinstance(self.parameters.cell.params[k], PyNNDistribution):
                self.dist_params[k] = self.parameters.cell.params[k]
                del self.parameters.cell.params[k]

    def setup_to_record_list(self):
        """
        Set up the recording configuration.
        """
        # source of wrong neuron indexes?
        self.to_record = OrderedDict()
        for k in self.parameters.recorders.keys():
            # print("recorder param ", k)
            recording_configuration = load_component(self.parameters.recorders[k].component)
            # print("self.parameters.recorders[k].params ", self.parameters.recorders[k].params)
            l = recording_configuration(self, self.parameters.recorders[k].params).generate_idd_list_of_neurons()

            if isinstance(self.parameters.recorders[k].variables, str):
                self.parameters.recorders[k].variables = [self.parameters.recorders[k].variables]
            # print("l setup_to_record_list ", l)
            # print("k ", k)
            # print("l[0] ", l[0])
            # print(l[20])
            # print(l[0].id)
            # print(l[1])
            # print(type(l[0]))  # IDMixin object
            # print(type(l))
            # print(l.shape)
            # print(self.to_record.get(self.parameters.recorders[k].variables[0], []))
            # print(type(self.to_record.get(self.parameters.recorders[k].variables[0], [])))
            # print(self.to_record.get(self.parameters.recorders[k].variables[0]))
            # print(type(self.to_record.get(self.parameters.recorders[k].variables[0])))
            # print(self.parameters.recorders[k].variables[0])
            # print(type(self.parameters.recorders[k].variables[0]))
            # print("TEST")
            # print(self.to_record)
            # print(type(self.to_record))
            # print(self.parameters.recorders[k].variables)
            # print("ENDTEST")

            # get ids from IDMixin objects
            # l = numpy.array([i.id for i in l])
            if hasattr(l[0], 'id'):
                print("XXX i.id in sheets init setup_to_record_list")
                l = numpy.array([i.id for i in l])

            # print("NEW ARRAY")
            # print("l ", l)
            # print(l[0])
            # print(type(l[0]))
            # print(type(l))
            # print(l.shape)
            for var in self.parameters.recorders[k].variables:
                # print(var)

                # print(type(var))
                # print("self.to_record[var] ", self.to_record[var])
                # print("debug")
                # print(set(l))
                # print(set(self.to_record.get(var, [])))
                # print(set(self.to_record.get(var)))
                # self.to_record[var] = list(set(self.to_record.get(var, [])) | set(l))
                if var == "spikes" or var == "v":  # spinnaker records all spikes in the population
                    self.to_record[var] = [i.id for i in self.pop.all_cells]
                    # print("list ot record for all spikes ", self.to_record[var])
                else:
                    print("record other than spikes or v", var)
                    # print("list(set(self.to_record.get(var, [])) | set(l)) ", list(set(self.to_record.get(var, [])) | set(l)))
                    self.to_record[var] = list(set(self.to_record.get(var, [])) | set(l))  # unhashabse type: 'IDMixin'
        # print(self.to_record)

        # for k in self.to_record.keys():
        for k in self.to_record.keys():
            # print(k)
            # print("self.to_record[k] ", self.to_record[k])
            # idds = self.pop.all_cells.astype(int)
            # idds = numpy.asarray(self.pop.all_cells)
            idds = numpy.array([i.id for i in self.pop.all_cells])
            # print("idds setup_to_record_list ", idds)
            self.to_record[k] = [numpy.flatnonzero(idds == idd)[0] for idd in self.to_record[k]]
            # print("numpy.flatnonzero(idds == idd)[0] for idd in self.to_record[k] ",
            #      [numpy.flatnonzero(idds == idd)[0] for idd in self.to_record[k]])
            # print("self.to_record[k] ", self.to_record[k])

        """
        self.to_record = {}
        for k in self.parameters.recorders.keys():
            recording_configuration = load_component(self.parameters.recorders[k].component)
            l = recording_configuration(self, self.parameters.recorders[k].params).generate_idd_list_of_neurons()
            if isinstance(self.parameters.recorders[k].variables, str):
                self.parameters.recorders[k].variables = [self.parameters.recorders[k].variables]

            for var in self.parameters.recorders[k].variables:
                self.to_record[var] = list(set(self.to_record.get(var, [])) | set(l))

        for k in self.to_record.keys():
            # idds = self.pop.all_cells.astype(int)
            idds = numpy.asarray(self.pop.all_cells)
            self.to_record[k] = [numpy.flatnonzero(idds == idd)[0] for idd in self.to_record[k]]
         """
    def size_in_degrees(self):
        """Returns the x, y size in degrees of visual field of the given area."""
        raise NotImplementedError
        pass

    def pop():
        doc = "The PyNN population holding the neurons in this sheet."

        def fget(self):
            if not self._pop:
                logger.error(
                    "Population have not been yet set in sheet: " + self.name + "!"
                )
            return self._pop

        def fset(self, value):
            if self._pop:
                raise Exception(
                    "Error population has already been set. It is not allowed to do"
                    " this twice!"
                )
            self._pop = value
            # l = value.all_cells.astype(int)

            self._neuron_annotations = [{} for i in range(0, len(value))]
            self.setup_artificial_stimulation()
            self.setup_initial_values()

        return locals()

    pop = property(
        **pop()
    )  # this will be populated by PyNN population, in the derived classes

    def add_neuron_annotation(self, neuron_number, key, value, protected=True):
        """
        Adds annotation to neuron at index neuron_number.
        
        Parameters
        ----------
        neuron_number : int
                      The index of the neuron in the population to which the annotation will be added.  
        
        key : str
            The name of the annotation
        
        value : object
              The value of the annotation
        
        protected : bool (default=True)
                  If True, the annotation cannot be changed.
        """
        if not self._pop:
            logger.error("Population has not been yet set in sheet: " + self.name + "!")
        if (
            key in self._neuron_annotations[neuron_number]
            and self._neuron_annotations[neuron_number][key][0]
        ):
            pass
            # logger.warning(
            #    "The annotation<"
            #    + str(key)
            #    + "> for neuron "
            #    + str(neuron_number)
            #    + " is protected. Annotation not updated"
            # )
        else:
            # print("key ", key)
            # print("neuron_number ", neuron_number)
            # print("value ", value)
            # print("protected ", protected)
            self._neuron_annotations[neuron_number][key] = (protected, value)

    def get_neuron_annotation(self, neuron_number, key):
        """
        Retrieve annotation for a given neuron.
        
        Parameters
        ----------
        neuron_number : int
                      The index of the neuron in the population to which the annotation will be added.  
        
        key : str
            The name of the annotation
        
        Returns
        -------
            value : object
                  The value of the annotation
        """

        if not self._pop:
            logger.error("Population has not been yet set in sheet: " + self.name + "!")
        if key not in self._neuron_annotations[neuron_number]:
            logger.error(
                "ERROR, annotation does not exist:",
                self.name,
                neuron_number,
                key,
                list(self._neuron_annotations[neuron_number].keys())
            )
        return self._neuron_annotations[neuron_number][key][1]

    def get_neuron_annotations(self):
        if not self._pop:
            logger.error("Population has not been yet set in sheet: " + self.name + "!")

        anns = []
        for i in range(0, len(self.pop)):
            d = {}
            for (k, v) in list(self._neuron_annotations[i].items()):
                d[k] = v[1]
            anns.append(d)
        return anns

    def describe(
        self, template="default", render=lambda t, c: Template(t).safe_substitute(c)
    ):
        context = {
            "name": self.__class__.__name__
        }
        if template:
            render(template, context)
        else:
            return context

    def record(self):
        # this should be only called once.
        self.setup_to_record_list()
        # if self.to_record is None:
        #    print("self.to_record is None !!")
        #    return

        # spikes has no sampling interval, and so must be recorded last
        # otherwise PyNN throws an error about inconsistent sampling intervals
        # spike_var = "spikes"
        # variables = [k for k in self.to_record if k != spike_var]
        # if spike_var in self.to_record:
        #     variables.append(spike_var)
        if self.to_record != None:
            for variable in self.to_record.keys():
                # print("variable ", variable)
                cells = self.to_record[variable]
                # print("cells unsorted ", cells)
                # cells.sort()
                cells = "all"  # record all cells
                if cells != "all":
                    # print("self.parameters.recording_interval not all ", self.parameters.recording_interval)
                    # print("recording ", variable)
                    # print("recording cells ", cells)
                    # print("recording population ", self.pop.label)
                    self.pop[cells].record(variable, sampling_interval=self.parameters.recording_interval)
                    # self.pop[cells].record(
                    #    variable, sampling_interval=self.parameters.recording_interval
                    # )
                else:
                    # print("self.parameters.recording_interval *all* ", self.parameters.recording_interval)
                    # print("recording ", variable)
                    # print("recording all cells")
                    # print("recording population ", self.pop.label)
                    self.pop.record(variable, sampling_interval=self.parameters.recording_interval)
                    # self.pop.record(
                    #    variable, sampling_interval=self.parameters.recording_interval
                    # )

    def get_data(self, stimulus_duration=None):
        """
        Retrieve data recorded in this sheet from pyNN in response to the last presented stimulus.
        
        Parameters
        ----------
        stimulus_duration : float(ms)
                          The length of the last stimulus presentation.
        
        Returns
        -------
        segment : Segment
                The segment holding all the recorded data. See NEO documentation for detail on the format.  
        """

        try:
            # block = self.pop.get_data(
            #    ["spikes", "v", "gsyn_exc", "gsyn_inh"], clear=True
            # )
            # block = self.pop.get_data(variables=["spikes", "v", "gsyn_exc", "gsyn_inh"], clear=True)
            block = self.pop.get_data(variables=["spikes"], clear=True)
            # block = self.pop.get_data(variables=["spikes", "v"])
            # print("XXX Sheet gsyn_exc ", block.segments[0].filter(name='gsyn_exc')[0])
            # x = self.pop.get_data("gsyn_exc")
            # print("x ", x)
            # x = self.pop.get_data("spikes")
        except NothingToWriteError as e:
            logger.debug(e.message)

        if (mozaik.mpi_comm) and (mozaik.mpi_comm.rank != mozaik.MPI_ROOT):
            print("XXX return non in get_data XXX")
            return None
        s = block.segments[-1]
        # take second segment because of reset
        # s = block.segments[1]
        s.annotations["sheet_name"] = self.name
        # print("sheet name ", self.name)
        # print("source_ids for gsyn_exc ", [a.annotations["source_ids"] for a in s.analogsignals if a.name == "gsyn_exc"])
        # print("source_ids for gsyn_inh ", [a.annotations["source_ids"] for a in s.analogsignals if a.name == "gsyn_inh"])
        # print("source_ids for v ", [a.annotations["source_ids"] for a in s.analogsignals if a.name == "v"])
        # print("neurons recorded for gsyn_exc ", self.to_record["gsyn_exc"])
        # print("neurons recorded for gsyn_exc ", self.to_record["gsyn_inh"])
        # print("signal names ", [a.name for a in s.analogsignals])
        # print("signal annotations ", [a.annotations for a in s.analogsignals])
        # workaround for wrond source ids
        # print("workaround start")
        """
        for a in s.analogsignals:
            if a.name == "v":
                # print("signal source ids ", a.annotations["source_ids"])
                # print("to record v ", self.to_record["v"])
                if set(a.annotations["source_ids"]) != set(self.to_record["v"]):
                    a.annotations["source_ids"] = self.to_record["v"]
        """
        """
        for a in s.analogsignals:
            # print(a.name)
            if a.name == "gsyn_exc":
                # print(a.annotations["source_ids"])
                # print(type(a.annotations["source_ids"]))
                # print(self.to_record["gsyn_exc"])
                # print(type(self.to_record["gsyn_exc"]))
                # print(set(a.annotations["source_ids"]) != set(self.to_record["gsyn_exc"]))
                if set(a.annotations["source_ids"]) != set(self.to_record["gsyn_exc"]):
                    a.annotations["source_ids"] = self.to_record["gsyn_exc"]
                    # print("a.annotations['source_ids'] ", a.annotations["source_ids"])
            elif a.name == "gsyn_inh":
                if set(a.annotations["source_ids"]) != set(self.to_record["gsyn_inh"]):
                    a.annotations["source_ids"] = self.to_record["gsyn_inh"]
                    # print("a.annotations['source_ids'] ", a.annotations["source_ids"])
            elif a.name == "v":
                if set(a.annotations["source_ids"]) != set(self.to_record["v"]):
                    a.annotations["source_ids"] = self.to_record["v"]
                    # print("a.annotations['source_ids'] ", a.annotations["source_ids"])
        """
        # print("workaround end")
        # print("signal annotations 2 ", [a.annotations for a in s.analogsignals])
        # print("dir(s.analogsignals[0]) ", dir(s.analogsignals[0]))
        # block2 = self.pop.get_data(variables=["gsyn_exc"])
        # print("XXX2 Sheet gsyn_exc ", block2.segments[0].filter(name='gsyn_exc')[0])
        # print("source ids ", [a.annotations["source_ids"] for a in block2.segments[-1].analogsignals if a.name == "gsyn_exc"])
        # print("end")
        # lets sort spike train so that it is ordered by IDs and thus hopefully
        # population indexes

        def key(x):
            return x.annotations['source_id']
        self.msc = numpy.mean([numpy.sum(st)/(st.t_stop-st.t_start)/1000 for st in s.spiketrains])
        s.spiketrains = sorted(s.spiketrains, key=key)
        # self.msc = numpy.mean([numpy.sum(st) for st in s.spiketrains])
        # s.spiketrains = sorted(s.spiketrains, key=lambda a: a.annotations["source_id"])
        if stimulus_duration != None:
            for st in s.spiketrains:
                tstart = st.t_start
                st -= tstart
                st.t_stop -= tstart
                st.t_start = 0 * pq.ms
            # for i in range(0, len(s.analogsignals)):
            #    s.analogsignals[i].t_start = 0 * pq.ms
        # print("workaround spikes")
        # print("self.to_record ", self.to_record)
        # print("self.to_record[spikes] ", self.to_record["spikes"])
        n = sorted(self.to_record["spikes"])
        # print("n sorted ", n)
        print("spiketrains length ", len(s.spiketrains))
        print("spikes length ", len(self.to_record["spikes"]))
        print("times of first ", s.spiketrains[0].times)
        # print([h.annotations["source_id"] for h in s.spiketrains])
        for k, i in zip(s.spiketrains, n):
            # print("spike source id ", k.annotations["source_id"])
            # print("spike source id ", k.annotations["source_id"])
            # print(type(k.annotations["source_id"]))
            # print("i ", i)
            # print(type(i))
            # if len(k.times) > 0:
            #    print("spikes more than zero ", k.times)
            if k.annotations["source_id"] != i:
                k.annotations["source_id"] = i
        # print([j.annotations["source_id"] for j in s.spiketrains])
        # maybe the spikes array needs to be cut to be the same as selected neurons or it will cause plotting problems
        # print("workaround spikes end")
        # print("analog signal length ", len(s.analogsignals))
        # print("analog signal[0] length ", len(s.analogsignals[0]))
        # print("XX spiketrains ", s.spiketrains)
        # print("XX spiketrains times ", [i.times for i in s.spiketrains])
        return s

    def mean_spike_count(self):
        logger.info(self.msc)
        return self.msc

    def prepare_artificial_stimulation(self, duration, offset, additional_stimulators):
        """
        Prepares the background noise and artificial stimulation for the population for the stimulus that is 
        about to be presented. 
        
        Parameters
        ----------
        
        duration : float (ms)
                 The duration of the stimulus that will be presented.
        
        additional_stimulators : list
                               List of additional stimulators, defined by the experiment that should be applied during this stimulus. 
                
        offset : float (ms)
               The current time of the simulation.
        """
        for ds in self.artificial_stimulators + additional_stimulators:
            ds.prepare_stimulation(duration, offset)

    def setup_artificial_stimulation(self):
        """
        Called once population is created. Sets up the background noise.
        """
        self.artificial_stimulators = []
        for k in list(self.parameters.artificial_stimulators.keys()):
            direct_stimulator = load_component(
                self.parameters.artificial_stimulators[k].component
            )
            self.artificial_stimulators.append(
                direct_stimulator(
                    self, self.parameters.artificial_stimulators[k].params
                )
            )

    def setup_initial_values(self):
        """
        Called once population is set. Set's up the initial values of the neural model variables.
        """
        # Initial state variables
        self.pop.initialize(**self.parameters.cell.initial_values)
        # Variable cell parameters
        self.pop.set(**self.dist_params)

