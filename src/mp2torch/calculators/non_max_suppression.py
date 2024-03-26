"""mediapipe/mediapipe/calculators/util/non_max_suppression_calculator.cc
Breadcrumbsmediapipe/mediapipe/calculators/util/non_max_suppression_calculator.proto"""
from dataclasses import dataclass
from enum import Enum


class OverlapType(Enum):
    """OverlapType
    During the overlap computation, which is used to determine whether a
    rectangle suppresses another rectangle, one can use the Jaccard similarity,
    defined as the ration of the intersection over union of the two rectangles.
    Alternatively a modified version of Jaccard can be used, where the
    normalization is done by the area of the rectangle being checked for
    suppression."""

    UNSPECIFIED_OVERLAP_TYPE = 0
    JACCARD = 1
    MODIFIED_JACCARD = 2
    INTERSECTION_OVER_UNION = 3


class NmsAlgorithm(Enum):
    """NmsAlgorithm
    Algorithms that can be used to apply non-maximum suppression."""

    DEFAULT = 0
    # Only supports relative bounding box for weighted NMS.
    WEIGHTED = 1


@dataclass
class NonMaxSuppressionCalculatorOptions:
    # Number of input streams. Each input stream should contain a vector of detections.
    num_detection_streams: int = 1

    # Maximum number of detections to be returned. If -1, then all detections are returned.
    max_num_detections: int = -1

    # Minimum score of detections to be returned.
    min_score_threshold: float - 1.0

    # Jaccard similarity threshold for suppression -- a detection would suppress
    # all other detections whose scores are lower and overlap by at least the specified threshold.
    min_suppression_threshold: float = 1.0

    overlap_type: OverlapType = OverlapType.JACCARD

    # Whether to put empty detection vector in output stream.
    # unspecified in non_max_suppression_calculator.proto
    return_empty_detections: bool = True

    algorithm: NmsAlgorithm = NmsAlgorithm.DEFAULT


class NonMaxSuppressionCalculator:
    def __init__(self, options: NonMaxSuppressionCalculatorOptions) -> None:
        self.__options = options

    def process(self):
        pass
