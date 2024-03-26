from pathlib import Path

from mp2torch.benchmarks.blazeface import BlazeFaceBenchmark


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cuda_index", type=int, required=False, default=0)
    args = parser.parse_args()

    devices = []
    if args.cpu:
        devices.append("cpu")
    if args.cuda:
        devices.append(f"cuda:{args.cuda_index}")
    if args.mps:
        devices.append("mps")

    benchmarker = BlazeFaceBenchmark(devices=devices)
    print("start benchmark")
    benchmarker.benchmark(video_path=args.video, metrics=["accuracy", "speed"])
    print("finish benchmark")

