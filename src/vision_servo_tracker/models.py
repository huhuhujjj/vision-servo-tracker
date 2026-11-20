"""从 Ultralytics 官方发布页下载本项目使用的预训练权重。"""
import argparse
from pathlib import Path
import shutil
import urllib.request


MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/yolo26n.pt"))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"文件已存在，保留原文件：{output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "vision-servo-tracker"})
    # 只有完整下载后才改为最终文件名，断线时不会留下看似可用的 .pt。
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        expected = response.headers.get("Content-Length")
        shutil.copyfileobj(response, target)
    if expected is not None and temporary.stat().st_size != int(expected):
        raise OSError("下载长度不完整，请重新运行下载命令")
    if temporary.stat().st_size == 0:
        raise OSError("服务器返回空文件")
    temporary.replace(output)
    print(f"模型已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
