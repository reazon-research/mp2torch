"""mediapipe/mediapipe/calculators/core/gate_calculator.proto
mediapipe/mediapipe/calculators/core/gate_calculator.cc"""
from dataclasses import dataclass
from enum import Enum


class GateState(Enum):
    """GateState
    mediapipe/mediapipe/calculators/core/gate_calculator.proto"""

    UNSPECIFIED = 0
    GATE_UNINITIALIZED = 1
    GATE_ALLOW = 2
    GATE_DISALLOW = 3


@dataclass
class GateCalculatorOptions:
    """GateCalculatorOptions
    mediapipe/mediapipe/calculators/core/gate_calculator.proto"""

    empty_packets_as_allow: bool
    allow: bool = False
    initial_gate_state: int = GateState.GATE_UNINITIALIZED


class GateCalculator:
    """GateCalculator
    mediapipe/mediapipe/calculators/core/gate_calculator.cc"""

    def __init__(self, options: GateCalculatorOptions) -> None:
        self.__options = options
        self.__use_option_for_allow_disallow = False
        self.__use_side_packet_for_allow_disallow = False

    def process(self):
        raise NotImplementedError
