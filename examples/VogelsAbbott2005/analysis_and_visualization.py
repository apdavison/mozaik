from mozaik.analysis.analysis import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.storage.datastore import PickledDataStore
from mozaik.storage.queries import *
from mozaik.visualization.plotting import *
import numpy
# from elephant import statistics
# from viziphant.statistics import plot_time_histogram
import quantities as pq
from pyNN.utility.plotting import Figure, Panel
from matplotlib import pyplot as plt


def perform_analysis_and_visualization(data_store):
    # print("param_filter_query")
    # print(param_filter_query(data_store, sheet_name="Exc_Layer"))
    # print(param_filter_query(data_store, sheet_name="Exc_Layer").get_segments()[0].get_stored_esyn_ids())
    # print("param_filter_query stop")
    analog_ids = sorted(
        param_filter_query(data_store, sheet_name="Exc_Layer")
        .get_segments()[0]
        .get_stored_esyn_ids()
    )
    analog_ids_inh = sorted(
        param_filter_query(data_store, sheet_name="Inh_Layer")
        .get_segments()[0]
        .get_stored_esyn_ids()
    )
    spike_ids = sorted(
        param_filter_query(data_store, sheet_name="Exc_Layer")
        .get_segments()[0]
        .get_stored_spike_train_ids()
    )
    spike_ids_inh = sorted(
        param_filter_query(data_store, sheet_name="Inh_Layer")
        .get_segments()[0]
        .get_stored_spike_train_ids()
    )

    if True:  # PLOTTING
        activity_plot_param = {
            "frame_rate": 5,
            "bin_width": 5.0,
            "scatter": True,
            "resolution": 0,
        }

        PSTH(
            param_filter_query(data_store, st_direct_stimulation_name="None"),
            ParameterSet({"bin_length": 5.0}),
        ).analyse()
        TrialAveragedFiringRate(
            param_filter_query(data_store, st_direct_stimulation_name="None"),
            ParameterSet({}),
        ).analyse()
        Irregularity(
            param_filter_query(data_store, st_direct_stimulation_name="None"),
            ParameterSet({}),
        ).analyse()
        NeuronToNeuronAnalogSignalCorrelations(
            param_filter_query(data_store, analysis_algorithm="PSTH"),
            ParameterSet({"convert_nan_to_zero": True}),
        ).analyse()
        PopulationMeanAndVar(data_store, ParameterSet({})).analyse()

        data_store.print_content(full_ADS=True)

        dsv = param_filter_query(data_store, st_direct_stimulation_name=None)

        OverviewPlot(
            dsv,
            ParameterSet(
                {
                    "sheet_name": "Exc_Layer",
                    "neuron": analog_ids[0],
                    "sheet_activity": {},
                    "spontaneous": False,
                }
            ),
            fig_param={"dpi": 100, "figsize": (19, 12)},
            plot_file_name="ExcAnalog1.png",
        ).plot({"Vm_plot.y_lim": (-80, -50), "Conductance_plot.y_lim": (0, 500.0)})
        OverviewPlot(
            dsv,
            ParameterSet(
                {
                    "sheet_name": "Exc_Layer",
                    "neuron": analog_ids[1],
                    "sheet_activity": {},
                    "spontaneous": False,
                }
            ),
            fig_param={"dpi": 100, "figsize": (19, 12)},
            plot_file_name="ExcAnalog2.png",
        ).plot({"Vm_plot.y_lim": (-80, -50), "Conductance_plot.y_lim": (0, 500.0)})
        OverviewPlot(
            dsv,
            ParameterSet(
                {
                    "sheet_name": "Exc_Layer",
                    "neuron": analog_ids[2],
                    "sheet_activity": {},
                    "spontaneous": False,
                }
            ),
            fig_param={"dpi": 100, "figsize": (19, 12)},
            plot_file_name="ExcAnalog3.png",
        ).plot({"Vm_plot.y_lim": (-80, -50), "Conductance_plot.y_lim": (0, 500.0)})

        RasterPlot(
            dsv,
            ParameterSet(
                {
                    "sheet_name": "Exc_Layer",
                    "neurons": spike_ids,
                    "trial_averaged_histogram": False,
                    "spontaneous": False,
                }
            ),
            fig_param={"dpi": 100, "figsize": (17, 5)},
            plot_file_name="ExcRaster.png",
        ).plot({"SpikeRasterPlot.group_trials": True})
        RasterPlot(
            dsv,
            ParameterSet(
                {
                    "sheet_name": "Inh_Layer",
                    "neurons": spike_ids_inh,
                    "trial_averaged_histogram": False,
                    "spontaneous": False,
                }
            ),
            fig_param={"dpi": 100, "figsize": (17, 5)},
            plot_file_name="InhRaster.png",
        ).plot({"SpikeRasterPlot.group_trials": True})

    # plot population firing rate histogram
    sc = []
    print(len(param_filter_query(data_store, sheet_name="Exc_Layer").get_segments()[0].spiketrains))
    # print(len(param_filter_query(data_store, sheet_name="Inh_Layer").get_segments()[0].spiketrains))
    for s in param_filter_query(data_store, sheet_name="Exc_Layer").get_segments()[0].spiketrains:
        sc.append(len(s))
    # for k in param_filter_query(data_store, sheet_name="Inh_Layer").get_segments()[0].spiketrains:
    #    sc.append(len(k))
    # print(sc)
    print(len(sc))
    # h = histogram(sc, bins=50)
    # print(h)
    plt.figure()
    plt.hist(sc, bins=50)
    # plt.xlabel("Number of Spikes")
    # plt.ylabel("Count")
    # plt.title("Histogram of Spike Counts")
    plt.show()
    plt.savefig("histogramalle.png")
    sci = []
    print(len(param_filter_query(data_store, sheet_name="Inh_Layer").get_segments()[0].spiketrains))
    for k in param_filter_query(data_store, sheet_name="Inh_Layer").get_segments()[0].spiketrains:
        sci.append(len(k))
    # print(sc)
    print(len(sc))
    # h = histogram(sc, bins=50)
    # print(h)
    # sci = []
    plt.figure()
    plt.hist(sci, bins=50)
    # plt.xlabel("Number of Spikes")
    # plt.ylabel("Count")
    # plt.title("histogram of spike counts")
    plt.show()
    plt.savefig("histogramalli.png")
