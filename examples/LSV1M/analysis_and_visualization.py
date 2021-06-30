import os
import psutil
import sys
import mozaik
import logging
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import Global
from visualization_functions import *
from parameters import ParameterSet
# from elephant.spike_train_dissimilarity import victor_purpura_distance, van_rossum_distance
import quantities as pq
from matplotlib import pyplot as plt
from elephant import statistics
from viziphant.statistics import plot_time_histogram

# logger = mozaik.getMozaikLogger()
logger = logging.getLogger(__name__)

process = psutil.Process(os.getpid())


low_contrast = 30


def memory_usage_psutil():
    # return the memory usage in MB
    return process.memory_percent()


def analysis(data_store, analog_ids, analog_ids_inh, analog_ids23=None, analog_ids_inh23=None):
    sheets = list(set(data_store.sheets()) & set(
        ['V1_Exc_L4', 'V1_Inh_L4', 'V1_Exc_L2/3', 'V1_Inh_L2/3']))
    exc_sheets = list(set(data_store.sheets()) &
                      set(['V1_Exc_L4', 'V1_Exc_L2/3']))
    l23_flag = ('V1_Exc_L2/3' in set(sheets))

    logger.info('0: ' + str(memory_usage_psutil()))
    try:
        TrialAveragedFiringRate(param_filter_query(data_store, sheet_name=sheets,
                                                   st_name='FullfieldDriftingSinusoidalGrating'), ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    TrialAveragedFiringRate(param_filter_query(
        data_store, st_direct_stimulation_name=None, st_name='InternalStimulus'), ParameterSet({})).analyse()
    logger.info('1: ' + str(memory_usage_psutil()))
    Irregularity(param_filter_query(data_store, st_direct_stimulation_name=None,
                                    st_name='InternalStimulus'), ParameterSet({})).analyse()

    PSTH(param_filter_query(data_store),
         ParameterSet({'bin_length': 10.0})).analyse()

    logger.info('2: ' + str(memory_usage_psutil()))
    NeuronToNeuronAnalogSignalCorrelations(param_filter_query(
        data_store, analysis_algorithm='PSTH'), ParameterSet({'convert_nan_to_zero': True})).analyse()

    logger.info('3: ' + str(memory_usage_psutil()))
    # 1&3 ZeroDivisionError
    try:
        PopulationMeanAndVar(param_filter_query(data_store, st_direct_stimulation_name=None,
                                            st_name='InternalStimulus'), ParameterSet({'ignore_nan_and_inf': False})).analyse()
    except Exception as e:
        print(e)

    dsv = queries.param_filter_query(
        data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm='PSTH')
    try:
        TrialMean(dsv, ParameterSet(
            {'cond_exc': False, 'vm': True, 'cond_inh': False})).analyse()
    except Exception as e:
        print(e)

    dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                             analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', sheet_name=sheets)
    try:
        GaussianTuningCurveFit(dsv, ParameterSet(
            {'parameter_name': 'orientation'})).analyse()
    except Exception as e:
        print(e)
    dsv = param_filter_query(
        data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=sheets)
    try:
        Analog_F0andF1(dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                             analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', sheet_name=sheets)
    try:
        PeriodicTuningCurvePreferenceAndSelectivity_VectorAverage(
            dsv, ParameterSet({'parameter_name': 'orientation'})).analyse()
    except Exception as e:
        print(e)

    logger.info('4: ' + str(memory_usage_psutil()))

    dsv = param_filter_query(data_store, sheet_name=exc_sheets)
    ActionPotentialRemoval(dsv, ParameterSet({'window_length': 5.0})).analyse()

    logger.info('5: ' + str(memory_usage_psutil()))


    dsv = param_filter_query(
        data_store, st_name='InternalStimulus', st_direct_stimulation_name=None)
    try:
        Analog_MeanSTDAndFanoFactor(dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
        SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
        SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(VM)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Vm')
        OperationPNVfromPNVS(pnv, lambda x, y: -(x+y), '-(x+y)',
                             dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Inh_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 sheet_name='V1_Inh_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
        SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Inh_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 sheet_name='V1_Inh_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
        SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Inh_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(VM)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 sheet_name='V1_Inh_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Vm')
        OperationPNVfromPNVS(pnv, lambda x, y: -(x+y), '-(x+y)',
                             dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
        OperationPNVfromPNVS(pnv, lambda x, y: x-y, 'x-y',
                             dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)
    try:
        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
        OperationPNVfromPNVS(pnv, lambda x, y: x-y, 'x-y',
                             dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    logger.info('8: ' + str(memory_usage_psutil()))
    if True:
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
            SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
            SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(VM)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Vm')
            OperationPNVfromPNVS(pnv, lambda x, y: -(x+y),
                                 '-(x+y)', dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Inh_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                     sheet_name='V1_Inh_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
            SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Inh_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                     sheet_name='V1_Inh_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
            SubtractPNVfromPNVS(pnv, dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)

        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Inh_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(VM)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                     sheet_name='V1_Inh_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Vm')
            OperationPNVfromPNVS(pnv, lambda x, y: -(x+y),
                                 '-(x+y)', dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
            OperationPNVfromPNVS(pnv, lambda x, y: x-y, 'x-y',
                                 dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)
        try:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
            OperationPNVfromPNVS(pnv, lambda x, y: x-y, 'x-y',
                                 dsv, ParameterSet({})).analyse()
        except Exception as e:
            print(e)

    logger.info('9: ' + str(memory_usage_psutil()))
    try:
        dsv = queries.param_filter_query(
            data_store, y_axis_name='spike count (bin=13.0)')
        mozaik.analysis.analysis.TrialToTrialFanoFactorOfAnalogSignal(
            dsv, ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    if True:
        logger.info('10: ' + str(memory_usage_psutil()))
        try:
            TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L4', st_name="NaturalImageWithEyeMovement",
                                                                              analysis_algorithm='ActionPotentialRemoval'), ParameterSet({'neurons': list(analog_ids), 'window_min': 0, 'window_max': -1})).analyse()
            TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L4', st_name="NaturalImageWithEyeMovement", analysis_algorithm='PSTH'), ParameterSet({
                                                           'neurons': list(analog_ids), 'window_min': 0, 'window_max': -1})).analyse()
            TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L4', st_name='FullfieldDriftingSinusoidalGrating',
                                                                              analysis_algorithm='ActionPotentialRemoval', st_contrast=100), ParameterSet({'neurons': list(analog_ids), 'window_min': 0, 'window_max': -1})).analyse()
            TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L4', st_name='FullfieldDriftingSinusoidalGrating',
                                                                              analysis_algorithm='PSTH', st_contrast=100), ParameterSet({'neurons': list(analog_ids), 'window_min': 0, 'window_max': -1})).analyse()
        except Exception as e:
            print(e)
        logger.info('11: ' + str(memory_usage_psutil()))
        if l23_flag:
            try:
                TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L2/3', st_name="NaturalImageWithEyeMovement",
                                                                                  analysis_algorithm='ActionPotentialRemoval'), ParameterSet({'neurons': list(analog_ids23), 'window_min': 0, 'window_max': -1})).analyse()
                TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L2/3', st_name="NaturalImageWithEyeMovement",
                                                                                  analysis_algorithm='PSTH'), ParameterSet({'neurons': list(analog_ids23), 'window_min': 0, 'window_max': -1})).analyse()
                TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L2/3', st_name='FullfieldDriftingSinusoidalGrating',
                                                                                  analysis_algorithm='ActionPotentialRemoval', st_contrast=100), ParameterSet({'neurons': list(analog_ids23), 'window_min': 0, 'window_max': -1})).analyse()
                TrialToTrialCrossCorrelationOfAnalogSignalList(param_filter_query(data_store, sheet_name='V1_Exc_L2/3', st_name='FullfieldDriftingSinusoidalGrating',
                                                                                  analysis_algorithm='PSTH', st_contrast=100), ParameterSet({'neurons': list(analog_ids23), 'window_min': 0, 'window_max': -1})).analyse()
            except Exception as e:
                print(e)
        logger.info('12: ' + str(memory_usage_psutil()))
    try:
        dsv = param_filter_query(
            data_store, analysis_algorithm='ActionPotentialRemoval')
        dsv.print_content(full_ADS=True)
    except Exception as e:
        print(e)
    # 1&3 ValueError: operands could not be broadcast together with shapes (53382,1) (61990,1)
    try:
        TrialVariability(dsv, ParameterSet(
            {'vm': False,  'cond_exc': False, 'cond_inh': False})).analyse()
    except Exception as e:
        print(e)
    try:
        param_filter_query(
            data_store, analysis_algorithm='TrialVariability').print_content(full_ADS=True)
    except Exception as e:
        print(e)

    logger.info('13: ' + str(memory_usage_psutil()))
    try:
        ModulationRatio(param_filter_query(
            data_store, sheet_name=exc_sheets, st_contrast=[100]), ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    logger.info('14: ' + str(memory_usage_psutil()))
    try:
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating',
                                 analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate')
        CircularVarianceOfTuningCurve(dsv, ParameterSet(
            {'parameter_name': 'orientation'})).analyse()
    except Exception as e:
        print(e)
    try:
        dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating', value_name=[
                                 'F1(psth (bin=10.0) trial-to-trial mean)'], analysis_algorithm='Analog_F0andF1', sheet_name=sheets)
        GaussianTuningCurveFit(dsv, ParameterSet(
            {'parameter_name': 'orientation'})).analyse()
        CircularVarianceOfTuningCurve(dsv, ParameterSet(
            {'parameter_name': 'orientation'})).analyse()
    except Exception as e:
        print(e)

    dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating', value_name=[
                             'F0(psth (bin=10.0) trial-to-trial mean)'], analysis_algorithm='Analog_F0andF1', sheet_name=sheets)
    try:
        GaussianTuningCurveFit(dsv, ParameterSet(
            {'parameter_name': 'orientation'})).analyse()
        CircularVarianceOfTuningCurve(dsv, ParameterSet(
            {'parameter_name': 'orientation'})).analyse()
    except Exception as e:
        print(e)

    logger.info('15: ' + str(memory_usage_psutil()))

    data_store.save()


def perform_analysis_and_visualization_stc(data_store):
    l23 = 'V1_Exc_L2/3' in set(data_store.sheets())
    analog_ids = param_filter_query(data_store, sheet_name="V1_Exc_L4").get_segments()[
        0].get_stored_esyn_ids()
    analog_ids_inh = param_filter_query(
        data_store, sheet_name="V1_Inh_L4").get_segments()[0].get_stored_esyn_ids()
    spike_ids = param_filter_query(data_store, sheet_name="V1_Exc_L4").get_segments()[
        0].get_stored_spike_train_ids()
    spike_ids_inh = param_filter_query(data_store, sheet_name="V1_Inh_L4").get_segments()[
        0].get_stored_spike_train_ids()

    NeuronAnnotationsToPerNeuronValues(data_store, ParameterSet({})).analyse()

    if l23:
        analog_ids23 = param_filter_query(
            data_store, sheet_name="V1_Exc_L2/3").get_segments()[0].get_stored_esyn_ids()
        analog_ids_inh23 = param_filter_query(
            data_store, sheet_name="V1_Inh_L2/3").get_segments()[0].get_stored_esyn_ids()
        spike_ids23 = param_filter_query(
            data_store, sheet_name="V1_Exc_L2/3").get_segments()[0].get_stored_spike_train_ids()
        spike_ids_inh23 = param_filter_query(
            data_store, sheet_name="V1_Inh_L2/3").get_segments()[0].get_stored_spike_train_ids()
        l23_exc_or = data_store.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L2/3')[0]
        l23_exc_or_many = numpy.array(spike_ids23)[numpy.nonzero(numpy.array([circular_dist(
            l23_exc_or.get_value_by_id(i), 0, numpy.pi) for i in spike_ids23]) < 0.25)[0]]
        idx23 = data_store.get_sheet_indexes(
            sheet_name='V1_Exc_L2/3', neuron_ids=l23_exc_or_many)

    l4_exc_or = data_store.get_analysis_result(
        identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L4')[0]
    l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(
        l4_exc_or.get_value_by_id(i), 0, numpy.pi) for i in spike_ids]) < 0.25)[0]]
    idx4 = data_store.get_sheet_indexes(
        sheet_name='V1_Exc_L4', neuron_ids=l4_exc_or_many)
    # fix positions
    x = data_store.get_neuron_postions()['V1_Exc_L4'][0][idx4]
    y = data_store.get_neuron_postions()['V1_Exc_L4'][1][idx4]
    center4 = l4_exc_or_many[numpy.nonzero(numpy.sqrt(
        numpy.multiply(x, x)+numpy.multiply(y, y)) < 0.4)[0]]
    analog_center4 = set(center4).intersection(analog_ids)
    logger.info(str(analog_center4))

    if l23:
        x = data_store.get_neuron_postions()['V1_Exc_L2/3'][0][idx23]
        y = data_store.get_neuron_postions()['V1_Exc_L2/3'][1][idx23]
        center23 = l23_exc_or_many[numpy.nonzero(numpy.sqrt(
            numpy.multiply(x, x)+numpy.multiply(y, y)) < 0.4)[0]]
        analog_center23 = set(center23).intersection(analog_ids23)
        logger.info(str(analog_center23))

    if True:
        TrialAveragedFiringRate(param_filter_query(data_store, sheet_name=[
                                'V1_Exc_L4', 'V1_Exc_L2/3'], st_name='DriftingSinusoidalGratingDisk'), ParameterSet({})).analyse()

        dsv = param_filter_query(data_store, sheet_name=[
                                 'V1_Exc_L4', 'V1_Exc_L2/3'], st_name='DriftingSinusoidalGratingDisk')
        Analog_F0andF1(dsv, ParameterSet({})).analyse()

        dsv = param_filter_query(
            data_store, st_name='InternalStimulus', st_direct_stimulation_name=None)
        Analog_MeanSTDAndFanoFactor(dsv, ParameterSet({})).analyse()

        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(VM)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Vm')
        OperationPNVfromPNVS(pnv, lambda x, y: -(x+y),
                             '-(x+y)', dsv, ParameterSet({})).analyse()

        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
        OperationPNVfromPNVS(pnv, lambda x, y: x-y, 'x-y',
                             dsv, ParameterSet({})).analyse()

        pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L4', analysis_algorithm=[
                                 'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
        dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                 sheet_name='V1_Exc_L4', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
        OperationPNVfromPNVS(pnv, lambda x, y: x-y, 'x-y',
                             dsv, ParameterSet({})).analyse()

        if l23:
            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(VM)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Vm')
            OperationPNVfromPNVS(pnv, lambda x, y: -(x+y),
                                 '-(x+y)', dsv, ParameterSet({})).analyse()

            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ECond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Exc_Cond')
            OperationPNVfromPNVS(pnv, lambda x, y: x-y,
                                 'x-y', dsv, ParameterSet({})).analyse()

            pnv = param_filter_query(data_store, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3', analysis_algorithm=[
                                     'Analog_MeanSTDAndFanoFactor'], value_name='Mean(ICond)', st_direct_stimulation_name=None).get_analysis_result()[0]
            dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                     sheet_name='V1_Exc_L2/3', analysis_algorithm=['Analog_F0andF1'], value_name='F0_Inh_Cond')
            OperationPNVfromPNVS(pnv, lambda x, y: x-y,
                                 'x-y', dsv, ParameterSet({})).analyse()

        dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk',
                                 analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate')

        SizeTuningAnalysis(dsv, ParameterSet(
            {'neurons': center4.tolist(), 'sheet_name': 'V1_Exc_L4'})).analyse()
        if l23:
            SizeTuningAnalysis(dsv, ParameterSet(
                {'neurons': center23.tolist(), 'sheet_name': 'V1_Exc_L2/3'})).analyse()
        data_store.save()

    dsv = param_filter_query(data_store, st_name='DriftingSinusoidalGratingDisk', analysis_algorithm=[
                             'TrialAveragedFiringRate'], value_name="Firing rate")
    PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': list(center4), 'sheet_name': 'V1_Exc_L4', 'centered': False,
                                       'mean': False, 'polar': False, 'pool': False}), plot_file_name='SizeTuningExcL4.png', fig_param={'dpi': 100, 'figsize': (32, 7)}).plot()
    PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': list(center4), 'sheet_name': 'V1_Exc_L4', 'centered': False,
                                       'mean': True, 'polar': False, 'pool': False}), plot_file_name='SizeTuningExcL4M.png', fig_param={'dpi': 100, 'figsize': (32, 7)}).plot()

    if l23:
        PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': list(center23), 'sheet_name': 'V1_Exc_L2/3', 'centered': False,
                                           'mean': False, 'polar': False, 'pool': False}), plot_file_name='SizeTuningExcL23.png', fig_param={'dpi': 100, 'figsize': (32, 7)}).plot()
        PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': list(center23), 'sheet_name': 'V1_Exc_L2/3', 'centered': False,
                                           'mean': True, 'polar': False, 'pool': False}), plot_file_name='SizeTuningExcL23M.png', fig_param={'dpi': 100, 'figsize': (32, 7)}).plot()

    if True:
        dsv = param_filter_query(data_store, st_name=[
                                 'DriftingSinusoidalGratingDisk'])
        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': list(analog_center4)[0], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL4_1.png').plot()

        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': list(analog_center23)[0], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL23_1.png').plot()

    if l23:
        SizeTuningOverview(data_store, ParameterSet({'l4_neurons': list(center4), 'l23_neurons': list(center23), 'l4_neurons_analog': list(
            analog_center4), 'l23_neurons_analog': list(analog_center23)}), plot_file_name='SizeTuningOverview.png', fig_param={'dpi': 300, 'figsize': (18, 8)}).plot()
        SizeTuningOverviewNew(data_store, ParameterSet({'l4_neurons': list(center4), 'l23_neurons': list(center23), 'l4_neurons_analog': list(
            analog_center4), 'l23_neurons_analog': list(analog_center23)}), plot_file_name='SizeTuningOverviewNew.png', fig_param={'dpi': 300, 'figsize': (18, 8)}).plot()
    else:
        SizeTuningOverview(data_store, ParameterSet({'l4_neurons': list(center4), 'l23_neurons': [], 'l4_neurons_analog': list(
            analog_center4), 'l23_neurons_analog': []}), plot_file_name='SizeTuningOverview.png', fig_param={'dpi': 300, 'figsize': (18, 8)}).plot()
        SizeTuningOverviewNew(data_store, ParameterSet({'l4_neurons': list(center4), 'l23_neurons': [], 'l4_neurons_analog': list(
            analog_center4), 'l23_neurons_analog': []}), plot_file_name='SizeTuningOverviewNew.png', fig_param={'dpi': 300, 'figsize': (18, 8)}).plot()

    if True:
        dsv = param_filter_query(data_store, st_name=[
                                 'DriftingSinusoidalGratingDisk'], st_size=[5.0])
        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': list(analog_center4)[0], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL4_Small1.png').plot()
        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': list(analog_center4)[1], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL4_Small2.png').plot()
        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': list(analog_center4)[2], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL4_Small3.png').plot()

        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': list(analog_center23)[0], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL23_Small1.png').plot()
        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': list(analog_center23)[1], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL23_Small2.png').plot()
        OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': list(analog_center23)[2], 'sheet_activity': {
        }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='Overview_ExcL23_Small3.png').plot()

        RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neurons': spike_ids, 'trial_averaged_histogram': False, 'spontaneous': False}), fig_param={
                   'dpi': 100, 'figsize': (28, 12)}, plot_file_name='EvokedExcRasterL4.png').plot({'SpikeRasterPlot.group_trials': True})
        RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neurons': spike_ids, 'trial_averaged_histogram': False, 'spontaneous': False}),
                   fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='EvokedExcRasterL2/3.png').plot({'SpikeRasterPlot.group_trials': True})

        dsv = param_filter_query(data_store, st_name=['InternalStimulus'])
        RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neurons': spike_ids, 'trial_averaged_histogram': False, 'spontaneous': False}), fig_param={
                   'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSExcRasterL4.png').plot({'SpikeRasterPlot.group_trials': True})
        RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neurons': spike_ids, 'trial_averaged_histogram': False, 'spontaneous': False}),
                   fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSExcRasterL2/3.png').plot({'SpikeRasterPlot.group_trials': True})



def perform_analysis_and_visualization(data_store):

    sheets = list(set(data_store.sheets()) & set(
        ['V1_Exc_L4', 'V1_Inh_L4', 'V1_Exc_L2/3', 'V1_Inh_L2/3']))
    exc_sheets = list(set(data_store.sheets()) &
                      set(['V1_Exc_L4', 'V1_Exc_L2/3']))
    l23_flag = 'V1_Exc_L2/3' in set(sheets)
    try:
        NeuronAnnotationsToPerNeuronValues(data_store, ParameterSet({})).analyse()
    except Exception as e:
        print(e)

    # analog_ids = param_filter_query(data_store, sheet_name="V1_Exc_L4").get_segments()[
    #    0].get_stored_esyn_ids()
    analog_ids = None
    # analog_ids_inh = param_filter_query(
    #    data_store, sheet_name="V1_Inh_L4").get_segments()[0].get_stored_esyn_ids()
    analog_ids_inh = None
    spike_ids = param_filter_query(data_store, sheet_name="V1_Exc_L4").get_segments()[
        0].get_stored_spike_train_ids()
    spike_ids_inh = param_filter_query(data_store, sheet_name="V1_Inh_L4").get_segments()[
        0].get_stored_spike_train_ids()

    if l23_flag:
        # analog_ids23 = param_filter_query(
        #    data_store, sheet_name="V1_Exc_L2/3").get_segments()[0].get_stored_esyn_ids()
        analog_ids23 = None
        # analog_ids_inh23 = param_filter_query(
        #    data_store, sheet_name="V1_Inh_L2/3").get_segments()[0].get_stored_esyn_ids()
        analog_ids_inh23 = None
        spike_ids23 = param_filter_query(
            data_store, sheet_name="V1_Exc_L2/3").get_segments()[0].get_stored_spike_train_ids()
        spike_ids_inh23 = param_filter_query(
            data_store, sheet_name="V1_Inh_L2/3").get_segments()[0].get_stored_spike_train_ids()
    else:
        analog_ids23 = None
        analog_ids_inh23 = None

    if l23_flag:
        try:
            l23_exc_or = data_store.get_analysis_result(
                identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L2/3')[0]
            l23_inh_or = data_store.get_analysis_result(
                identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Inh_L2/3')[0]
        except Exception as e:
            print(e)
    try:
        l4_exc_or = data_store.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L4')
        l4_exc_phase = data_store.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentPhase', sheet_name='V1_Exc_L4')
    except Exception as e:
        print(e)
    # l4_exc = analog_ids[numpy.argmin([circular_dist(o, 0, numpy.pi) for (o, p) in zip(
    #    l4_exc_or[0].get_value_by_id(analog_ids), l4_exc_phase[0].get_value_by_id(analog_ids))])]
    try:
        l4_inh_or = data_store.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Inh_L4')
        l4_inh_phase = data_store.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentPhase', sheet_name='V1_Inh_L4')
    except Exception as e:
        print(e)
    # l4_inh = analog_ids_inh[numpy.argmin([circular_dist(o, 0, numpy.pi) for (o, p) in zip(
    #    l4_inh_or[0].get_value_by_id(analog_ids_inh), l4_inh_phase[0].get_value_by_id(analog_ids_inh))])]
    try:
        l4_exc_or_many = numpy.array(l4_exc_or[0].ids)[numpy.nonzero(numpy.array([circular_dist(
            o, 0, numpy.pi) for (o, p) in zip(l4_exc_or[0].values, l4_exc_phase[0].values)]) < 0.1)[0]]
    except Exception as e:
        print(e)
    try:
        l4_exc_or_many = list(set(l4_exc_or_many) & set(spike_ids))
    except Exception as e:
        print(e)

    if l23_flag:
        try:
            l23_exc_or_many = numpy.array(l23_exc_or.ids)[numpy.nonzero(numpy.array(
                [circular_dist(o, 0, numpy.pi) for o in l23_exc_or.values]) < 0.1)[0]]
            l23_exc_or_many = list(set(l23_exc_or_many) & set(spike_ids23))
        except Exception as e:
            print(e)
    try:
        orr = list(set([MozaikParametrized.idd(s).orientation for s in queries.param_filter_query(
            data_store, st_name='FullfieldDriftingSinusoidalGrating', st_contrast=100).get_stimuli()]))
    except Exception as e:
        print(e)
    # l4_exc_or_many_analog = numpy.array(analog_ids)[numpy.nonzero(numpy.array(
    #    [circular_dist(l4_exc_or[0].get_value_by_id(i), 0, numpy.pi) for i in analog_ids]) < 0.1)[0]]
    # l4_inh_or_many_analog = numpy.array(analog_ids_inh)[numpy.nonzero(numpy.array(
    #    [circular_dist(l4_inh_or[0].get_value_by_id(i), 0, numpy.pi) for i in analog_ids_inh]) < 0.15)[0]]

    if l23_flag:
        pass
        # wrong ids
        # try:
        #    l23_inh_or_many_analog = numpy.array(analog_ids_inh23)[numpy.nonzero(numpy.array(
        #        [circular_dist(l23_inh_or.get_value_by_id(i), 0, numpy.pi) for i in analog_ids_inh23]) < 0.15)[0]]
        #    l23_exc_or_many_analog = numpy.array(analog_ids23)[numpy.nonzero(numpy.array(
        #        [circular_dist(l23_exc_or.get_value_by_id(i), 0, numpy.pi) for i in analog_ids23]) < 0.1)[0]]
        # except Exception as e:
        #    print(e)

    if True:
        if l23_flag:
            # pass
            # ZeroDivisionError
            # print("analog_ids ", analog_ids)
            # print("analog_ids_inh ", analog_ids_inh)
            # print("analog_ids23 ", analog_ids23)
            # print("analog_ids_inh23 ", analog_ids_inh23)
            try:
                analysis(data_store, analog_ids, analog_ids_inh,
                         analog_ids23, analog_ids_inh23)
            except Exception as e:
                print(e)
        else:
            try:
                analysis(data_store, analog_ids, analog_ids_inh)
            except Exception as e:
                print(e)

    if True:  # PLOTTING
        activity_plot_param = {
            'frame_rate': 5,
            'bin_width': 5.0,
            'scatter':  True,
            'resolution': 0
        }
        # spike count histogram
        try:
            sc = []
            for s in param_filter_query(data_store, sheet_name="V1_Exc_L4").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist4exc.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(data_store, sheet_name="V1_Exc_L4").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist4excel.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(data_store, sheet_name="V1_Inh_L4").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist4inh.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(data_store, sheet_name="V1_Inh_L4").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist4inhel.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(data_store, sheet_name="V1_Exc_L2/3").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist23exc.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(data_store, sheet_name="V1_Exc_L2/3").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist23excel.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(data_store, sheet_name="V1_Inh_L2/3").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist23inh.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(data_store, sheet_name="V1_Inh_L2/3").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist23inhel.png")
        except Exception as e:
            print(e)

        # self sustained plotting


        try:
            dsv = param_filter_query(data_store, st_name=[
                'InternalStimulus'], st_direct_stimulation_name=None)
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': analog_ids[0], 'sheet_activity': {
            }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSExcAnalog.png').plot()
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': analog_ids_inh[0], 'sheet_activity': {
            }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSInhAnalog.png').plot()
        except Exception as e:
            print(e)
        try:
            RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neurons': spike_ids, 'trial_averaged_histogram': False, 'spontaneous': False}), fig_param={
                   'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSExcRasterL4.png').plot({'SpikeRasterPlot.group_trials': True})
            RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neurons': spike_ids_inh, 'trial_averaged_histogram': False, 'spontaneous': False}), fig_param={
                   'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSInhRasterL4.png').plot({'SpikeRasterPlot.group_trials': True})
        except Exception as e:
            print(e)

        if l23_flag:
            try:
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': analog_ids23[0], 'sheet_activity': {
                }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSExcAnalog23.png').plot()
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': analog_ids_inh23[0], 'sheet_activity': {
                }, 'spontaneous': True}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSInhAnalog23.png').plot()
            except Exception as e:
                print(e)
            try:
                RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neurons': spike_ids23, 'trial_averaged_histogram': False, 'spontaneous': False}),
                           fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSExcRasterL23.png').plot({'SpikeRasterPlot.group_trials': True})
                RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neurons': spike_ids_inh23, 'trial_averaged_histogram': False, 'spontaneous': False}),
                           fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSInhRasterL23.png').plot({'SpikeRasterPlot.group_trials': True})
            except Exception as e:
                print(e)

        try:
            dsv = param_filter_query(
                data_store, st_name='FullfieldDriftingSinusoidalGrating')
            # print("is dsv empty? ", dsv)
            # AssertionError: Error, empty datastore
            RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neurons': spike_ids, 'trial_averaged_histogram': False, 'spontaneous': False}), fig_param={
                       'dpi': 100, 'figsize': (28, 12)}, plot_file_name='EvokedExcRaster.png').plot({'SpikeRasterPlot.group_trials': True})
            RasterPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neurons': spike_ids_inh, 'trial_averaged_histogram': False, 'spontaneous': False}), fig_param={
                       'dpi': 100, 'figsize': (28, 12)}, plot_file_name='EvokedInhRaster.png').plot({'SpikeRasterPlot.group_trials': True})
        except Exception as e:
            print(e)

        # spike count histogram
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Exc_L4").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist4exc2.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Exc_L4").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist4excel2.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Inh_L4").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist4inh2.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Inh_L4").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist4inhel2.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Exc_L2/3").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist23exc2.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Exc_L2/3").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist23excel2.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Inh_L2/3").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist23inh2.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Inh_L2/3").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist23inhel2.png")
        except Exception as e:
            print(e)

        try:
            dsv = param_filter_query(
                data_store, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[0, numpy.pi / 2])
            # NameError: name 'l4_exc' is not defined
            # 1&2 ValueError: x and y must have same first dimension, but have shapes (83242,) and (87546, 1)
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': l4_exc, 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Exc.png").plot({'Vm_plot.y_lim': (-80, -50)})
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': l4_inh, 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Inh.png").plot({'Vm_plot.y_lim': (-80, -50)})
        except Exception as e:
            print(e)

        # AssertionError: Error, empty datastore
        try:
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': analog_ids[0], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Exc1.png").plot({'Vm_plot.y_lim': (-80, -50)})
        except Exception as e:
            print(e)
        # IndexError: list index out of range
        try:
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': analog_ids[1], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Exc2.png").plot({'Vm_plot.y_lim': (-80, -50)})
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': analog_ids[2], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Exc3.png").plot({'Vm_plot.y_lim': (-80, -50)})
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': analog_ids[3], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Exc4.png").plot({'Vm_plot.y_lim': (-80, -50)})
        except Exception as e:
            print(e)
        try:
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': analog_ids_inh[0], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Inh1.png").plot({'Vm_plot.y_lim': (-80, -50)})
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': analog_ids_inh[1], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Inh2.png").plot({'Vm_plot.y_lim': (-80, -50)})
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': analog_ids_inh[2], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Inh3.png").plot({'Vm_plot.y_lim': (-80, -50)})
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': analog_ids_inh[3], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                         'dpi': 100, 'figsize': (25, 12)}, plot_file_name="Inh4.png").plot({'Vm_plot.y_lim': (-80, -50)})
        except Exception as e:
            print(e)
        # if False:
        if l23_flag:
            try:
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': analog_ids23[0], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                             'dpi': 100, 'figsize': (25, 12)}, plot_file_name="ExcL231.png").plot({'Vm_plot.y_lim': (-80, -50)})
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': analog_ids23[1], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                             'dpi': 100, 'figsize': (25, 12)}, plot_file_name="ExcL232.png").plot({'Vm_plot.y_lim': (-80, -50)})
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': analog_ids23[2], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                             'dpi': 100, 'figsize': (25, 12)}, plot_file_name="ExcL233.png").plot({'Vm_plot.y_lim': (-80, -50)})
            except Exception as e:
                print(e)
            try:
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': analog_ids_inh23[0], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                             'dpi': 100, 'figsize': (25, 12)}, plot_file_name="InhL231.png").plot({'Vm_plot.y_lim': (-80, -50)})
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': analog_ids_inh23[1], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                             'dpi': 100, 'figsize': (25, 12)}, plot_file_name="InhL232.png").plot({'Vm_plot.y_lim': (-80, -50)})
                OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': analog_ids_inh23[2], 'sheet_activity': {}, 'spontaneous': True}), fig_param={
                             'dpi': 100, 'figsize': (25, 12)}, plot_file_name="InhL233.png").plot({'Vm_plot.y_lim': (-80, -50)})
            except Exception as e:
                print(e)

        # spike count histogram
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Exc_L4").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist4exc3.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Exc_L4").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist4excel3.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Inh_L4").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist4inh3.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Inh_L4").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist4inhel3.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Exc_L2/3").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist23exc3.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Exc_L2/3").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist23excel3.png")
        except Exception as e:
            print(e)
        try:
            sc = []
            for s in param_filter_query(dsv, sheet_name="V1_Inh_L2/3").get_segments()[0].spiketrains:
                sc.append(len(s))
            print(len(sc))
            plt.figure()
            plt.hist(sc, bins=50)
            plt.savefig("hist23inh3.png")
        except Exception as e:
            print(e)
        try:
            histogram = statistics.time_histogram(param_filter_query(dsv, sheet_name="V1_Inh_L2/3").get_segments()[0].spiketrains, bin_size=5*pq.ms)
            plot_time_histogram(histogram, units='s')
            plt.savefig("hist23inhel3.png")
        except Exception as e:
            print(e)

        if l23_flag:
            try:
                SpontActOverview(data_store, ParameterSet({'l4_exc_neuron': analog_ids[0], 'l4_inh_neuron': analog_ids_inh[0], 'l23_exc_neuron': analog_ids23[
                                 0], 'l23_inh_neuron': analog_ids_inh23[0]}), plot_file_name='SpontActOverview.png', fig_param={'dpi': 200, 'figsize': (18, 14.5)}).plot()
            except Exception as e:
                print(e)
            try:
                OrientationTuningSummaryAnalogSignals(data_store, ParameterSet({'exc_sheet_name1': 'V1_Exc_L4', 'inh_sheet_name1': 'V1_Inh_L4', 'exc_sheet_name2': 'V1_Exc_L2/3', 'inh_sheet_name2': 'V1_Inh_L2/3'}), fig_param={
                                                      'dpi': 200, 'figsize': (18, 12)}, plot_file_name='OrientationTuningSummaryAnalogSignals.png').plot({'*.fontsize': 19, '*.y_lim': (0, None)})
                OrientationTuningSummaryFiringRates(data_store, ParameterSet({'exc_sheet_name1': 'V1_Exc_L4', 'inh_sheet_name1': 'V1_Inh_L4', 'exc_sheet_name2': 'V1_Exc_L2/3', 'inh_sheet_name2': 'V1_Inh_L2/3'}), fig_param={
                                                    'dpi': 200, 'figsize': (18, 12)}, plot_file_name='OrientationTuningSummary.png').plot({'*.fontsize': 19})
            except Exception as e:
                print(e)
        else:
            try:
                SpontActOverview(data_store, ParameterSet({'l4_exc_neuron': analog_ids[0], 'l4_inh_neuron': analog_ids_inh[0], 'l23_exc_neuron': -1,
                                                           'l23_inh_neuron': -1}), plot_file_name='SpontActOverview.png', fig_param={'dpi': 200, 'figsize': (18, 14.5)}).plot()
            except Exception as e:
                print(e)
            try:
                OrientationTuningSummaryAnalogSignals(data_store, ParameterSet({'exc_sheet_name1': 'V1_Exc_L4', 'inh_sheet_name1': 'V1_Inh_L4', 'exc_sheet_name2': 'None', 'inh_sheet_name2': 'None'}), fig_param={
                                                      'dpi': 200, 'figsize': (18, 12)}, plot_file_name='OrientationTuningSummaryAnalogSignals.png').plot({'*.fontsize': 19, '*.y_lim': (0, None)})
            except Exception as e:
                print(e)
        # 1&3 *Result was expected to have only single ADS, it contains 0*
        try:
            SpontStatisticsOverview(data_store, ParameterSet({}), fig_param={
                                    'dpi': 200, 'figsize': (18, 12)}, plot_file_name='SpontStatisticsOverview.png').plot()
        except Exception as e:
            print(e)
        try:
            TrialToTrialVariabilityComparisonNew(data_store, ParameterSet({'sheet_name1': 'V1_Exc_L4', 'sheet_name2': 'V1_Exc_L2/3', 'data_dg': 0.93, 'data_ni': 1.19}), fig_param={
                                             'dpi': 200, 'figsize': (15, 7.5)}, plot_file_name='TrialToTrialVariabilityComparisonNew.png').plot()
        except Exception as e:
            print(e)
        if l23_flag:
            try:
                MRfigReal(param_filter_query(data_store, sheet_name=['V1_Exc_L2/3', 'V1_Exc_L4', 'V1_Inh_L2/3', 'V1_Inh_L4'], st_contrast=[100], st_name='FullfieldDriftingSinusoidalGrating'), ParameterSet(
                    {'SimpleSheetName': 'V1_Exc_L4', 'ComplexSheetName': 'V1_Exc_L2/3'}), plot_file_name='MRReal.png', fig_param={'dpi': 100, 'figsize': (19, 12)}).plot()
            except Exception as e:
                print(e)
        else:
            try:
                MRfigReal(param_filter_query(data_store, sheet_name=['V1_Exc_L2/3', 'V1_Exc_L4', 'V1_Inh_L2/3', 'V1_Inh_L4'], st_contrast=[100], st_name='FullfieldDriftingSinusoidalGrating'), ParameterSet(
                    {'SimpleSheetName': 'V1_Exc_L4', 'ComplexSheetName': 'V1_Exc_L2/3'}), plot_file_name='MRReal.png', fig_param={'dpi': 100, 'figsize': (19, 12)}).plot()
            except Exception as e:
                print(e)

        try:
            dsv = param_filter_query(
                data_store, st_name='NaturalImageWithEyeMovement')
            # 1&3 ValueError: x and y must have same first dimension, but have shapes (83242,) and (87546, 1)
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': l4_exc, 'sheet_activity': {}, 'spontaneous': True}), plot_file_name='NMExc.png', fig_param={
                         'dpi': 100, 'figsize': (28, 12)}).plot({'Vm_plot.y_lim': (-70, -50), 'Conductance_plot.y_lim': (0, 50.0)})
        except Exception as e:
            print(e)
        try:
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': l4_inh, 'sheet_activity': {}, 'spontaneous': True}), plot_file_name='NMInh.png', fig_param={
                     'dpi': 100, 'figsize': (28, 12)}).plot({'Vm_plot.y_lim': (-70, -50), 'Conductance_plot.y_lim': (0, 50.0)})
        except Exception as e:
            print(e)
        # 1&3 ValueError: attempt to get argmin of an empty sequence
        try:
            TrialCrossCorrelationAnalysis(data_store, ParameterSet({'neurons1': list(analog_ids), 'sheet_name1': 'V1_Exc_L4', 'neurons2': list(
                analog_ids23), 'sheet_name2': 'V1_Exc_L2/3', 'window_length': 250}), fig_param={"dpi": 100, "figsize": (15, 6.5)}, plot_file_name="trial-to-trial-cross-correlation.png").plot({'*.Vm.title': None, '*.fontsize': 19})
        except Exception as e:
            print(e)

        try:
            dsv = queries.param_filter_query(data_store, value_name=[
                'orientation HWHH of Firing rate', 'orientation CV(Firing rate)'], sheet_name=["V1_Exc_L2/3"],
                                             st_contrast=100)
            # 1&3 AssertionError: Error, not pairs of PerNeuronValue ADS in datastore seem to have the same value_units
            PerNeuronValueScatterPlot(dsv, ParameterSet({'only_matching_units': False, 'ignore_nan': True, 'lexicographic_order': False}), plot_file_name='CVvsHWHH.png').plot(
                {'*.x_lim': (0, 90), '*.y_lim': (0, 1.0)})
        except Exception as e:
            print(e)

        try:
            dsv = param_filter_query(data_store, st_name=['InternalStimulus'])
            OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': analog_ids_inh[0], 'sheet_activity': {
            }, 'spontaneous': False}), fig_param={'dpi': 100, 'figsize': (28, 12)}, plot_file_name='SSInhAnalog.png').plot()
        except Exception as e:
            print(e)

        # orientation tuning plotting
        try:
            dsv = param_filter_query(data_store,sheet_name=['V1_Exc_L4','V1_Inh_L4'],value_name='LGNAfferentOrientation')
            PerNeuronValuePlot(dsv,ParameterSet({"cortical_view" : True}),plot_file_name='ORSet.png').plot()
        except Exception as e:
            print(e)