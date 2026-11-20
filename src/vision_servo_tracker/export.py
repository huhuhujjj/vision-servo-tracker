"""在目标平台导出固定输入尺寸的 NCNN 检测模型。"""
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/yolo26n.pt"))
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("imgsz 必须是32的正整数倍")
    if not args.model.is_file():
        parser.error("模型文件不存在，请先运行 vision-download-model")
    from ultralytics import YOLO

    # 导出尺寸与网页推理尺寸必须一致；这里显式使用 CPU，避免 CUDA 依赖。
    result = YOLO(str(args.model.resolve())).export(
        format="ncnn", imgsz=args.imgsz, device="cpu", half=False)
    print(f"NCNN 模型：{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
