# Current Ember ML RNN Structure (Refactored)

This diagram illustrates the class relationships and initialization flow in the refactored Ember ML library, focusing on LTC and CfC components and their base classes.

Key characteristics & Conflicts:
- **Goal:** Refactoring aims for **deferred initialization**, where dimensions are resolved and weights/maps built during the first `forward` call via a `build(input_shape)` method.
- **Base Classes:** Introduces `ModuleCell` and `ModuleWiredCell` for better hierarchy.
- **NeuronMap:** Replaces `Wiring`, intended as the structural blueprint.
- **Conflict:** The current `ModuleWiredCell` base class still contains logic that attempts to build the `NeuronMap` during `__init__` (build-at-init), conflicting with the deferred build goal. This forces subclasses like `LTCCell` into inconsistent states.
- **Terminology:** Uses `NeuronMap`, `input_size`, `output_size`.

```mermaid
graph TD
    subgraph EmberML_NN_Modules [ember_ml.nn.modules]
        direction TB
        BaseModule[Module]
        BaseCell[ModuleCell]
        BaseWiredCell[ModuleWiredCell] -- Contains --> ConflictLogic{Build-at-Init Logic !!}

        subgraph EmberML_RNN [rnn]
            direction TB
            LTCCell[LTCCell] -- inherits --> BaseWiredCell
            CfCCell[CfCCell] -- inherits --> BaseCell
            LSTMCell[LSTMCell] -- inherits --> BaseCell
            GRUCell[GRUCell] -- inherits --> BaseCell
            RNNCell[RNNCell] -- inherits --> BaseCell
        end

        subgraph EmberML_Wiring [wiring]
             NeuronMap[NeuronMap] -- Base --> BuiltFlag{_built: bool}
             NCPMap[NCPMap] -- inherits --> NeuronMap
             FullyConnectedMap[FullyConnectedMap] -- inherits --> NeuronMap
             RandomMap[RandomMap] -- inherits --> NeuronMap
        end
    end

    BaseCell --> BaseModule
    BaseWiredCell --> BaseCell
    NCPMap --> NeuronMap
    FullyConnectedMap --> NeuronMap
    RandomMap --> NeuronMap

    subgraph DeferredBuild [Intended Deferred Build Flow]
        direction TB
        Layer_Forward[Layer/Cell forward input] -- First Call --> Build{build input_shape}
        Build -- calls --> ParentBuild{super.build input_shape}
        ParentBuild -- e.g., ModuleWiredCell.build --> GetDim{Extract input_dim from shape}
        GetDim --> MapBuild{neuron_map.build input_dim}
        MapBuild --> SetMapDim{map.input_dim = input_dim}
        MapBuild --> SetMapBuilt{map._built = True}
        ParentBuild --> SetCellDims{cell.input_size = map.input_dim<br/>cell.hidden_size = map.units}
        Build -- after super.build --> AllocateParams{_allocate_parameters}
        Build --> SetCellBuilt{cell.built = True}
        Layer_Forward -- Subsequent Calls --> UseParams[Uses Initialized Params]
    end

    subgraph LTCCellInit [Current LTCCell Init Conflict]
        LTCCell_Init[LTCCell.__init__ neuron_map, **kwargs]
        LTCCell_Init -- gets --> KwargsInFeatures{in_features from kwargs Optional}
        LTCCell_Init -- checks --> MapInputDim{neuron_map.input_dim Maybe None}
        LTCCell_Init -- determines --> InitInputSize{input_size for super init}
        LTCCell_Init -- calls --> SuperInit{super.__init__ input_size, neuron_map}
        SuperInit -- calls --> BaseWiredCell_Init[ModuleWiredCell.__init__]
        BaseWiredCell_Init -- attempts --> BuildMapOnInit{neuron_map.build input_size !!}
        LTCCell_Init -- calls --> AllocateOnInit{_allocate_parameters !! Too early}
    end

    classDef base fill:#D8BFD8,stroke:#333,stroke-width:2px;
    classDef cellbase fill:#ADD8E6,stroke:#333,stroke-width:1px;
    classDef wiredbase fill:#87CEEB,stroke:#333,stroke-width:1px;
    classDef cell fill:#B0E0E6,stroke:#333,stroke-width:1px;
    classDef wiring fill:#lightgrey,stroke:#333,stroke-width:1px;
    classDef flow fill:#white,stroke:#ccc,stroke-dasharray: 5 5;
    classDef conflict fill:#FFCCCB,stroke:#f00;


    class BaseModule base;
    class BaseCell cellbase;
    class BaseWiredCell wiredbase;
    class LTCCell,CfCCell,LSTMCell,GRUCell,RNNCell cell;
    class NeuronMap,NCPMap,FullyConnectedMap,RandomMap wiring;
    class DeferredBuild,LTCCellInit flow;
    class ConflictLogic conflict;