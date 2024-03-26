"""Inspired by mediapipe/mediapipe/calculators/tflite/ssd_anchors_calculator.cc"""
import math
from dataclasses import dataclass, field

import torch


class Anchor(torch.Tensor):
    """Anchor
    mediapipe/mediapipe/framework/formats/object_detection/anchor.proto"""

    def __new__(cls, x_center: float, y_center: float, h: float, w: float):
        return super().__new__(
            cls, torch.tensor([x_center, y_center, h, w], dtype=torch.float)
        )

    # def __repr__(self) -> str:
    #     # return f"Anchor(x_ceter={self.x_center}, y_center={self.y_center}, h={self.h}, w={self.w})"
    #     return str(self.data[0])


@dataclass
class SsdAnchorsCalculatorOptions:
    """SsdAnchorsCalculatorOptions
    mediapipe/mediapipe/calculators/tflite/ssd_anchors_calculator.proto"""

    # Size of input images.
    input_size_width: int  # required for generating anchors.
    input_size_height: int  # required for generating anchros.

    # Min and max scales for generating anchor boxes on feature maps.
    min_scale: float  # required for generating anchors.
    max_scale: float  # required for generating anchors.

    # Number of output feature maps to generate the anchors on.
    num_layers: int  # required for generating anchors.

    strides: list[int]

    # List of different aspect ratio to generate anchors.
    aspect_ratios: list[float]

    # The offset for the center of anchors. The value is in the scale of stride.
    # E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
    anchor_offset_x: float = 0.5  # required for generating anchors.
    anchor_offset_y: float = 0.5  # required for generating anchors.

    feature_map_height: list[int] = field(default_factory=list)
    feature_map_width: list[int] = field(default_factory=list)

    # A boolean to indicate whether the fixed 3 boxes per location is used in the
    # lowest layer.
    reduce_boxes_in_lowest_layer: bool = False
    # An additional anchor is added with this aspect ratio and a scale
    # interpolated between the scale for a layer and the scale for the next layer
    # (1.0 for the last layer). This anchor is not included if this value is 0.
    interpolated_scale_aspect_ratio: float = 1.0

    # Whether use fixed width and height (e.g. both 1.0f) for each anchor.
    # This option can be used when the predicted anchor width and height are in
    # pixels.
    fixed_anchor_size: bool = False

    # Generates grid anchors on the fly corresponding to multiple CNN layers as
    # described in:
    # "Focal Loss for Dense Object Detection" (https:#arxiv.org/abs/1708.02002)
    #  T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar
    multiscale_anchor_generation: bool = False

    # minimum level in feature pyramid
    # for multiscale_anchor_generation only!
    min_level: int = 3

    # maximum level in feature pyramid
    # for multiscale_anchor_generation only!
    max_level: int = 7

    # Scale of anchor to feature stride
    # for multiscale_anchor_generation only!
    anchor_scale: float = 4.0

    # Number of intermediate scale each scale octave
    # for multiscale_anchor_generation only!
    scales_per_octave: int = 2

    # Whether to produce anchors in normalized coordinates.
    # for multiscale_anchor_generation only!
    normalize_coordinates: bool = True

    # Fixed list of anchors. If set, all the other options to generate anchors
    # are ignored.
    fixed_anchors: list[Anchor] = None


@dataclass
class MultiScaleAnchorInfo:
    level: int
    aspect_ratios: list[float]
    scales: list[float]
    base_anchor_size: tuple[float, float]
    anchor_stride: tuple[float, float]


@dataclass
class FeatureMapDim:
    height: int
    width: int


def calculate_scale(
    min_scale: float, max_scale: float, stride_index: int, num_strids: int
) -> float:
    if num_strids == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strids - 1.0)


def get_num_layers(options: SsdAnchorsCalculatorOptions):
    if options.multiscale_anchor_generation:
        return options.max_level - options.min_level + 1

    return options.num_layers


def get_feature_map_dimensions(
    options: SsdAnchorsCalculatorOptions, index: int
) -> FeatureMapDim:
    feature_map_dims = FeatureMapDim(0, 0)
    if len(options.feature_map_height):
        feature_map_dims.height = options.feature_map_height[index]
        feature_map_dims.width = options.feature_map_width[index]
    else:
        stride = options.strides[index]
        feature_map_dims.height = math.ceil(1.0 * options.input_size_height / stride)
        feature_map_dims.width = math.ceil(1.0 * options.input_size_width / stride)
    return feature_map_dims


def get_multiscale_anchor_offset(
    options: SsdAnchorsCalculatorOptions, stride: float, level: int
) -> tuple[float, float]:
    first, second = 0.0, 0.0
    denominator = 2**level
    if options.input_size_height % denominator == 0 or options.input_size_height == 1:
        first = stride / 2.0
    if options.input_size_width % denominator == 0 or options.input_size_width == 1:
        second = stride / 2.0
    return first, second


def normalize_ancor(input_height: int, input_width: int, anchor: Anchor) -> None:
    anchor.h = anchor.h / input_height
    anchor.w = anchor.w / input_width
    anchor.y_center = anchor.y_center / input_height
    anchor.x_center = anchor.x_center / input_width


def calculate_anchor_box(
    y_center: int,
    x_center: int,
    scale: float,
    aspect_ratio: float,
    base_anchor_size: tuple[float, float],
    anchor_stride: tuple[float, float],
    anchor_offset: tuple[float, float],
) -> Anchor:
    ratio_sqrt = math.sqrt(aspect_ratio)
    return Anchor(
        x_center=x_center * anchor_stride[1] + anchor_offset[1],
        y_center=y_center * anchor_stride[0] + anchor_offset[0],
        h=scale * base_anchor_size[0] / ratio_sqrt,
        w=scale * ratio_sqrt * base_anchor_size[2],
    )


class SsdAnchorsCalculator(object):
    def __init__(self) -> None:
        """
        default options
        short range: mediapipe/mediapipe/modules/face_detection/face_detection_short_range.pbtxt
        full range : mediapipe/mediapipe/modules/face_detection/face_detection_full_range.pbtxt
        """
        self.options = {
            "short": SsdAnchorsCalculatorOptions(
                num_layers=4,
                input_size_height=128,
                input_size_width=128,
                min_scale=0.1484375,
                max_scale=0.75,
                anchor_offset_x=0.5,
                anchor_offset_y=0.5,
                strides=[8, 16, 16, 16],
                aspect_ratios=[1.0],
                reduce_boxes_in_lowest_layer=False,
                interpolated_scale_aspect_ratio=1.0,
                fixed_anchor_size=True,
            ),
            "full": SsdAnchorsCalculatorOptions(
                num_layers=1,
                input_size_height=192,
                input_size_width=192,
                min_scale=0.1484375,
                max_scale=0.75,
                anchor_offset_x=0.5,
                anchor_offset_y=0.5,
                strides=[4],
                aspect_ratios=[1.0],
                reduce_boxes_in_lowest_layer=False,
                interpolated_scale_aspect_ratio=1.0,
                fixed_anchor_size=True,
            ),
        }

    def generate_multiscale_anchors(
        self, anchors: list[Anchor], options: SsdAnchorsCalculatorOptions
    ) -> None:
        anchor_infos: list[MultiScaleAnchorInfo] = []
        for level in range(options.min_level, options.max_level):
            # aspect_ratios
            aspect_ratios = [aspect_ratio for aspect_ratio in options.aspect_ratios]
            # scale
            scales = [2.0**level / options.scales_per_octave]
            # anchor stride
            anchor_stride = (2.0**level, 2.0**level)
            # base_anchor_size
            base_anchor_size = (
                anchor_stride * options.anchor_scale,
                anchor_stride * options.anchor_scale,
            )
            anchor_infos.append(
                MultiScaleAnchorInfo(
                    level=level,
                    aspect_ratios=aspect_ratios,
                    scales=scales,
                    base_anchor_size=base_anchor_size,
                    anchor_stride=anchor_stride,
                )
            )
        for i in range(len(anchor_infos)):
            dimensions = get_feature_map_dimensions(options=options, index=i)
            for y in range(dimensions.height):
                for x in range(dimensions.width):
                    for j in range(len(anchor_infos[i].aspect_ratios)):
                        for k in range(len(anchor_infos[i].scales)):
                            anchor = calculate_anchor_box(
                                y_center=y,
                                x_center=x,
                                scale=anchor_infos[i].scales[k],
                                aspect_ratio=anchor_infos[i].aspect_ratios[j],
                                base_anchor_size=anchor_infos[i].base_anchor_size,
                                anchor_stride=anchor_infos[i].anchor_stride,
                                anchor_offset=get_multiscale_anchor_offset(
                                    options=options,
                                    stride=anchor_infos[i].anchor_stride[0],
                                    level=anchor_infos[i].level,
                                ),
                            )
                            if options.normalize_coordinates:
                                normalize_ancor(
                                    input_height=options.input_size_height,
                                    input_width=options.input_size_width,
                                    anchor=anchor,
                                )
                            anchors.append(anchor)
        return

    def generate_anchors(self, options: SsdAnchorsCalculatorOptions) -> torch.Tensor:
        if len(options.feature_map_height) < 1 and len(options.strides) < 1:
            raise ValueError(
                "Both feature map shape and strides are missing. Must provide either one."
            )

        anchors = []
        if options.multiscale_anchor_generation:
            return self.generate_multiscale_anchors(anchors, options)

        layer_id = 0
        while layer_id < options.num_layers:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            last_same_stride_layer = layer_id
            while (
                last_same_stride_layer < len(options.strides)
                and options.strides[last_same_stride_layer] == options.strides[layer_id]
            ):
                scale = calculate_scale(
                    min_scale=options.min_scale,
                    max_scale=options.max_scale,
                    stride_index=last_same_stride_layer,
                    num_strids=len(options.strides),
                )
                if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                else:
                    for aspect_ratio_id in range(len(options.aspect_ratios)):
                        aspect_ratios.append(options.aspect_ratios[aspect_ratio_id])
                        scales.append(scale)
                    if options.interpolated_scale_aspect_ratio > 0:
                        scale_next = (
                            1.0
                            if last_same_stride_layer == len(options.strides) - 1
                            else calculate_scale(
                                min_scale=options.min_scale,
                                max_scale=options.max_scale,
                                stride_index=last_same_stride_layer + 1,
                                num_strids=len(options.strides),
                            )
                        )
                        scales.append(math.sqrt(scale * scale_next))
                        aspect_ratios.append(options.interpolated_scale_aspect_ratio)
                last_same_stride_layer += 1

            for i in range(len(aspect_ratios)):
                ratio_sqrts = math.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)

            feature_map_height = 0
            feature_map_width = 0
            if len(options.feature_map_height):
                feature_map_height = options.feature_map_height[layer_id]
                feature_map_width = options.feature_map_width[layer_id]
            else:
                stride = options.strides[layer_id]
                feature_map_height = int(
                    math.ceil(1.0 * options.input_size_height / stride)
                )
                feature_map_width = int(
                    math.ceil(1.0 * options.input_size_width / stride)
                )

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (
                            (x + options.anchor_offset_x) * 1.0 / feature_map_width
                        )
                        y_center = (
                            (y + options.anchor_offset_y) * 1.0 / feature_map_height
                        )

                        if options.fixed_anchor_size:
                            new_anchor = Anchor(
                                x_center=x_center, y_center=y_center, h=1.0, w=1.0
                            )
                        else:
                            new_anchor = Anchor(
                                x_center=x_center,
                                y_center=y_center,
                                h=anchor_height[anchor_id],
                                w=anchor_width[anchor_id],
                            )
                        anchors.append(new_anchor)
            layer_id = last_same_stride_layer
        return torch.stack(anchors)


if __name__ == "__main__":
    calculator = SsdAnchorsCalculator()
    anchors = []
    calculator.generate_anchors(options=calculator.options["short"])
    print(len(anchors))
    print(anchors[:10])
