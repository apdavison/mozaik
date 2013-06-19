"""
docstring goes here

"""
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet, load_parameters
import sys
import os
import mozaik
import time
from datetime import datetime
import logging
from NeuroTools import init_logging
from NeuroTools import visual_logging

import pyNN.nest as sim

logger = mozaik.getMozaikLogger("Mozaik")

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD


class Global:
    """global variable container"""
    root_directory = './'


def setup_logging():
    # Set-up logging
    if MPI:
        init_logging(Global.root_directory + "log", file_level=logging.DEBUG,
                     console_level=logging.DEBUG, mpi_rank=mpi_comm.rank)  # NeuroTools version
    else:
        init_logging(Global.root_directory + "log", file_level=logging.DEBUG,
                     console_level=logging.DEBUG)  # NeuroTools version
    visual_logging.basicConfig(Global.root_directory + "visual_log.zip",
                               level=logging.INFO)


def run_workflow(simulation_name, model_class, create_experiments):
    """
    This is the main function that executes a workflow. 
    
    It expects it gets the simulation, class of the model, and a function that will create_experiments.
    The create experiments function get a instance of a model as the only parameter and it is expected to return 
    a list of Experiment instances that should be executed over the model.
    
    The run workflow will automatically parse the command line to determine the simulator to be used and the path to the root parameter file. 
    It will also accept . (point) delimited path to parameteres in the configuration tree, and corresponding values. It will replace each such provided
    parameter's value with the provided one on the command line. The intended sintax of the commandline is as follows:
    
    python userscript simulator_name parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n
    """
    # Read parameters
    #exec("import pyNN.nest as sim" )
    
    if len(sys.argv) > 2 and len(sys.argv)%2 == 1:
        simulator_name = sys.argv[1]
        parameters_url = sys.argv[2]
        modified_parameters = { sys.argv[i*2+3] : eval(sys.argv[i*2+4])  for i in xrange(0,(len(sys.argv)-3)/2)}
    else:
        raise ValueError("Usage: runscript simulator_name parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n")
    
    parameters = load_parameters(parameters_url,modified_parameters)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # We exclude the results_dir parameter from the directory naming
    modified_params_str = '_'.join([str(k) + ":" + str(modified_parameters[k]) for k in modified_parameters.keys() if k!='results_dir'])
    if MPI:
        Global.root_directory = parameters.results_dir + simulation_name + '_' + \
                                  timestamp + 'rank' + str(mpi_comm.rank) + '_____' + modified_params_str + '/'
    else:
        Global.root_directory = parameters.results_dir + simulation_name + '_' + \
                                  timestamp + 'rank' + '______' + modified_params_str + '/'
    os.mkdir(Global.root_directory)
    parameters.save(Global.root_directory + "parameters", expand_urls=True)
    
    #let's store the modified parameters
    import pickle
    f = open(Global.root_directory+"modified_parameters","w")
    pickle.dump(modified_parameters,f)
    f.close()
    setup_logging()
    
    model = model_class(sim,parameters)
    data_store = run_experiments(model,create_experiments(model))
    data_store.save()
    import resource
    print "Final memory usage: %iMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024))
    return data_store

def run_experiments(model,experiment_list,load_from=None):
    # first lets run all the measurements required by the experiments
    logger.info('Starting Experiemnts')
    if load_from == None:
        data_store = PickledDataStore(load=False,
                                      parameters=MozaikExtendedParameterSet({'root_directory': Global.root_directory}))
    else:
        data_store = PickledDataStore(load=True,
                                      parameters=MozaikExtendedParameterSet({'root_directory': load_from}))
    
    data_store.set_neuron_ids(model.neuron_ids())
    data_store.set_neuron_positions(model.neuron_positions())
    data_store.set_neuron_annotations(model.neuron_annotations())
    
    t0 = time.time()
    simulation_run_time=0
    for i,experiment in enumerate(experiment_list):
        logger.info('Starting experiment: ' + experiment.__class__.__name__)
        stimuli = experiment.return_stimuli()
        unpresented_stimuli = data_store.identify_unpresented_stimuli(stimuli)
        logger.info('Running model')
        simulation_run_time += experiment.run(data_store,unpresented_stimuli)
        logger.info('Experiment %d/%d finished' % (i+1,len(experiment_list)))
    
    total_run_time = time.time() - t0
    mozaik_run_time = total_run_time - simulation_run_time
    
    logger.info('Total simulation run time: %.0fs' % total_run_time)
    logger.info('Simulator run time: %.0fs (%d%%)' % (simulation_run_time, int(simulation_run_time /total_run_time * 100)))
    logger.info('Mozaik run time: %.0fs (%d%%)' % (mozaik_run_time, int(mozaik_run_time /total_run_time * 100)))
    
    return data_store
