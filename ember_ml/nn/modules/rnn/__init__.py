"""
Recurrent Neural Network (RNN) module.

This module provides implementations of various RNN layers,
including LSTM, GRU, RNN, CfC (Closed-form Continuous-time), LTC (Liquid Time-Constant),
ELTC (Enhanced Liquid Time-Constant), CTGRU (Continuous-Time GRU), CTRNN (Continuous-Time RNN),
GUCE (Grand Unified Cognitive Equation), and StrideAware cells for multi-timescale processing.
"""

from ember_ml.nn.modules.rnn.cfc_cell import CfCCell
from ember_ml.nn.modules.rnn.wired_cfc_cell import WiredCfCCell
from ember_ml.nn.modules.rnn.cfc import CfC
from ember_ml.nn.modules.rnn.ltc_cell import LTCCell
from ember_ml.nn.modules.rnn.ltc import LTC
from ember_ml.nn.modules.rnn.eltc import ELTC, ODESolver
from ember_ml.nn.modules.rnn.ctgru import CTGRU
from ember_ml.nn.modules.rnn.ctrnn import CTRNN
from ember_ml.nn.modules.rnn.guce import GUCE, HolographicCorrector, OscillatoryGating
from ember_ml.nn.modules.rnn.lstm_cell import LSTMCell
from ember_ml.nn.modules.rnn.lstm import LSTM
from ember_ml.nn.modules.rnn.gru_cell import GRUCell
from ember_ml.nn.modules.rnn.gru import GRU
from ember_ml.nn.modules.rnn.rnn_cell import RNNCell
from ember_ml.nn.modules.rnn.rnn import RNN
from ember_ml.nn.modules.rnn.stride_aware_cell import StrideAwareCell
from ember_ml.nn.modules.rnn.stride_aware import StrideAware
from ember_ml.nn.modules.rnn.stride_aware_cfc import StrideAwareWiredCfCCell # Keep cell import
from ember_ml.nn.modules.rnn.stride_aware_cfc_layer import StrideAwareCfC # Import layer from new file

__all__ = [
    'CfCCell',
    'WiredCfCCell',
    'CfC',
    'LTCCell',
    'LTC',
    'ELTC',
    'ODESolver',
    'CTGRU',
    'CTRNN',
    'GUCE',
    'HolographicCorrector',
    'OscillatoryGating',
    'LSTMCell',
    'LSTM',
    'GRUCell',
    'GRU',
    'RNNCell',
    'RNN',
    'StrideAwareCell',
    'StrideAware',
    'StrideAwareWiredCfCCell',
    'StrideAwareCfC',
]