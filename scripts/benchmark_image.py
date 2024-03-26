import numpy as np
import torch

from mediapipe.python import solution_base
from mp2torch.calculators.image_to_tensor import (
    ImageToTensorCalculator,
    ImageToTensorCalculatorOptions,
)
from mp2torch.utils import VideoFramesBatchLoader

graph_config = """
input_stream: "IMAGE:image"
output_stream: "FLOATS:floats"
# Converts the input CPU image (ImageFrame) to the multi-backend image type
# (Image).
node: {
  calculator: "ToImageCalculator"
  input_stream: "IMAGE_CPU:image"
  output_stream: "IMAGE:multi_backend_image"
}
# Transforms the input image into a 192x192 tensor while keeping the aspect
# ratio (what is expected by the corresponding face detection model), resulting
# in potential letterboxing in the transformed image.
node: {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:multi_backend_image"
  output_stream: "TENSORS:input_tensors"
  output_stream: "MATRIX:transform_matrix"
  options: {
	[mediapipe.ImageToTensorCalculatorOptions.ext] {
      output_tensor_width: 128
	  output_tensor_height: 128
	  keep_aspect_ratio: true
      output_tensor_float_range {
		min: -1.0
		max: 1.0
	  }
	  border_mode: BORDER_ZERO
	}
  }
}
node: {
  calculator: "TensorsToFloatsCalculator"
  input_stream: "TENSORS:input_tensors"
  output_stream: "FLOATS:floats"
}
"""

if __name__ == "__main__":
    import argparse

    import cv2
    import ffmpegcv
    from tqdm import trange, tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    mp_input_graph = solution_base.SolutionBase(graph_config=graph_config)

    cap_ffmpegcv = ffmpegcv.VideoCapture(args.video, pix_fmt="rgb24")
    cap = cv2.VideoCapture(args.video)
    cap2 = cv2.VideoCapture(args.video)
    print("=== Image Loading Accuracy===")
    means, mins, maxs = [], [], []
    for _ in trange(len(cap_ffmpegcv)):
        _, img_ffmpegcv = cap_ffmpegcv.read()
        _, img_cv2 = cap.read()
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        diff = img_ffmpegcv - img_cv2
        means.append(diff.mean())
        mins.append(diff.min())
        maxs.append(diff.max())
    print(f"MEAN: {sum(means) / len(means)} | MIN: {min(mins)} | MAX: {max(maxs)}")

    input_graph_mp2torch = ImageToTensorCalculator(
        ImageToTensorCalculatorOptions(
            output_tensor_width=128,
            output_tensor_height=128,
            output_tensor_float_range={"min": -1, "max": 1},
            keep_aspect_ratio=True,
        )
    )

    cap_ffmpegcv = ffmpegcv.VideoCapture(args.video)
    means, mins, maxs = [], [], []
    for _ in trange(len(cap_ffmpegcv)):
        _, frame = cap_ffmpegcv.read()
        frame = frame[..., [2, 1, 0]]
        res = mp_input_graph.process(frame)
        image_mp = np.array(res.floats).reshape(128, 128, -1)
        image_mp2torch, _ = input_graph_mp2torch(torch.from_numpy(frame))
        diff = image_mp2torch - image_mp
        diff = diff.to(torch.float32)
        means.append(diff.mean())
        mins.append(diff.min())
        maxs.append(diff.max())
    print("=== ImageToTensor Accuracy ===")
    print(f"MEAN: {sum(means) / len(means)} | MIN: {min(mins)} | MAX: {max(maxs)}")

    print("=== VideoFramesBatchLoader w/ Single Accuracy ===")
    means, mins, maxs = [], [], []
    loader = VideoFramesBatchLoader()
    cap = ffmpegcv.VideoCapture(args.video)
    for batch, _ in tqdm(loader.make_loader(video_path=args.video)):
        _, frame = cap.read()
        frame = frame[..., [2, 1, 0]].transpose((2, 0, 1))
        diff = (batch[0] - frame).float()
        means.append(diff.mean())
        mins.append(diff.min())
        maxs.append(diff.max())
    print(f"MEAN: {sum(means) / len(means)} | MIN: {min(mins)} | MAX: {max(maxs)}")

    print("=== VideoFramesBatchLoader w/ batching Accuracy ===")
    means, mins, maxs = [], [], []
    loader = VideoFramesBatchLoader(32)
    cap = ffmpegcv.VideoCapture(args.video)
    for batch, _ in tqdm(loader.make_loader(video_path=args.video)):
        for sample in batch:
            _, frame = cap.read()
            frame = frame[..., [2, 1, 0]].transpose((2, 0, 1))
            diff = (sample - frame).float()
            means.append(diff.mean())
            mins.append(diff.min())
            maxs.append(diff.max())
    print(f"MEAN: {sum(means) / len(means)} | MIN: {min(mins)} | MAX: {max(maxs)}")
