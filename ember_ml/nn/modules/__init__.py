"""This module provides backend-agnostic implementations of neural network modules
using the ops abstraction layer.

Available Modules:
    Base Classes:
        RNNCell: Base recurrent cell
        LSTMCell: Long Short-Term Memory cell
        GRUCell: Gated Recurrent Unit cell
        
    Specialized Cells:
        StrideAware: Stride-aware cell base class
        StrideAwareCfC: Stride-aware Closed-form Continuous-time cell
        StrideAwareCell: Stride-aware general cell
        
    Advanced Modules:
        CfC: Closed-form Continuous-time cell
        WiredCfCCell: CfC with wiring capabilities
        LTC: Liquid Time-Constant cell
        LTCCell: LTC cell implementation
        StrideAwareWiredCfCCell: Stride-aware wired cell implementation

"""

from ember_ml.nn.modules.module import Module, Parameter
from ember_ml.nn.modules.base_module import BaseModule
from ember_ml.nn.modules.module_cell import ModuleCell
from ember_ml.nn.modules.module_wired_cell import ModuleWiredCell
from ember_ml.nn.modules.ncp import NCP
from ember_ml.nn.modules.auto_ncp import AutoNCP
from ember_ml.nn.modules.rnn import RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell, StrideAware, StrideAwareCfC, StrideAwareCell
from ember_ml.nn.modules.rnn import CfC, CfCCell, WiredCfCCell, StrideAwareWiredCfCCell, LTC, LTCCell, RNNCell
__all__ = [
    'Module',
    'Parameter',
    'BaseModule',
    'ModuleCell',
    'ModuleWiredCell',
    'NCP',
    'AutoNCP',
    'RNN',
    'LSTM',
    'GRU',
    'RNNCell',
    'LSTMCell', 
    'GRUCell',
    
    # Stride-aware
    'StrideAware',
    'StrideAwareCfC',
    'StrideAwareCell',
    
    'StrideAwareWiredCfCCell',
    # Advanced modules
    'CfC',
    'CfCCell',
    'WiredCfCCell',
    'LTC',
    'LTCCell',
]