# Original ncps.torch RNN Structure

This diagram illustrates the class relationships and initialization flow in the original `ncps` library (specifically `ncps.torch`), focusing on LTC and CfC components.

Key characteristics:
- **Build-at-Init:** Dimensions are resolved and components (cells, wirings) are fully constructed during the `__init__` phase.
- **Direct Inheritance:** Cells inherit directly from `torch.nn.Module`.
- **Wiring Class:** Uses the `Wiring` base class and specific implementations (`NCP`, `FullyConnected`, etc.).
- **Layer Responsibility:** Layers (`LTC`, `CfC`) handle wiring creation/lookup and pass necessary dimensions (`input_size` / `in_features`) to the cell during initialization.

```mermaid
graph TD
    subgraph ncps_torch [ncps.torch]
        direction LR
        LTC_Layer[LTC Layer nn.Module]
        LTCCell[LTCCell nn.Module]
        CfC_Layer[CfC Layer nn.Module]
        CfCCell[CfCCell nn.Module]
        WiredCfCCell[WiredCfCCell nn.Module]
        LSTMCell_Orig[LSTMCell nn.Module]
    end

    subgraph ncps_wirings [ncps.wirings]
        direction LR
        WiringBase[Wiring Base]
        FullyConnected[FullyConnected]
        NCPWiring[NCP]
        RandomWiring[Random]
        AutoNCPWiring[AutoNCP]
    end

    LTC_Layer -- instantiates --> LTCCell
    LTC_Layer -- creates_uses --> WiringBase
    LTCCell -- requires --> WiringBase

    CfC_Layer -- instantiates --> CfCCell
    CfC_Layer -- instantiates --> WiredCfCCell
    WiredCfCCell -- instantiates_multiple --> CfCCell
    WiredCfCCell -- requires --> WiringBase

    %% Inheritance relationships
    FullyConnected -- inherits --> WiringBase
    NCPWiring -- inherits --> WiringBase
    RandomWiring -- inherits --> WiringBase
    AutoNCPWiring -- inherits --> NCPWiring

    subgraph Initialization [Initialization Build-at-Init]
        User --> LTC_Layer_Init[LTC_Layer init: input_size, units_or_wiring]
        LTC_Layer_Init --> WiringBase_Create[WiringBase determine/create]
        LTC_Layer_Init --> LTCCell_Init[LTCCell init: wiring, in_features=input_size]
        LTCCell_Init --> WiringBase_Build[WiringBase wiring.build in_features]
        LTCCell_Init --> LTCCell_Alloc[LTCCell _allocate_parameters]

        User --> CfC_Layer_Init[CfC_Layer init: input_size, units_or_wiring]
        CfC_Layer_Init --> CfCCell_Init[CfCCell init: input_size, units]
        CfC_Layer_Init --> WiredCfCCell_Init[WiredCfCCell init: input_size, wiring]
        WiredCfCCell_Init --> CfCCell_Init2[CfCCell init: calc_input_size, calc_hidden_size]
    end

    classDef module fill:#lightgreen,stroke:#333,stroke-width:1px;
    classDef cell fill:#lightblue,stroke:#333,stroke-width:1px;
    classDef wiring fill:#lightgrey,stroke:#333,stroke-width:1px;
    classDef flow fill:#white,stroke:#fff;


    class LTC_Layer,CfC_Layer module;
    class LTCCell,CfCCell,WiredCfCCell,LSTMCell_Orig cell;
    class WiringBase,FullyConnected,NCPWiring,RandomWiring,AutoNCPWiring wiring;
    class Initialization flow;