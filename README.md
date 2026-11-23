# vision-servo-tracker

[![Checks](https://github.com/huhuhujjj/vision-servo-tracker/actions/workflows/checks.yml/badge.svg)](https://github.com/huhuhujjj/vision-servo-tracker/actions/workflows/checks.yml)

基于 Logitech C922、YOLO26n 和 ByteTrack 的目标检测与点选跟踪原型。主机端输出选定目标相对画面中心的偏差，供后续双轴舵机控制使用。

## 当前进度

| 部分 | 状态 |
| --- | --- |
| Windows 摄像头采集、GPU 检测、目标编号 | 已用真实 C922 画面验证 |
| 树莓派 C922 → Wi-Fi / SSH视频转发 → 主机 GPU 检测 | 已通过30帧实时取流验证，并打开实时识别窗口 |
| 点选、解除锁定、慢速移动跟踪、LOST 后不自动换目标 | 已进行实机操作验证 |
| 偏差计算、类别变化处理、过期点击拒绝 | 有逻辑测试 |
| 树莓派 CPU / NCNN 部署与浏览器预览 | 代码已准备，尚未完成树莓派实机验收 |
| PCA9685 与 MG996R 驱动、角度校准、闭环跟随 | 尚未实现 |

模型采用 Ultralytics 官方 COCO 预训练权重，没有进行模型训练、微调或无人机视角验证。快速移动、遮挡、手机转动后的类别变化和目标重新入画仍需测试。

## Windows 运行

需要 Python 3.11 或更新版本。主机已验证的组合为 Python 3.14、PyTorch 2.11.0、CUDA 12.8、Ultralytics 8.4.213；其他组合需自行验证。

在仓库根目录的 PowerShell 中创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)先安装适合显卡的 PyTorch。下面是已验证主机使用的版本：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\.venv\Scripts\python.exe -m vision_servo_tracker.models
.\.venv\Scripts\python.exe -m vision_servo_tracker.desktop
```

也可以双击 `scripts/run_windows.cmd`。不需要修改 PowerShell 执行策略。模型下载工具会将权重放到 `models/yolo26n.pt`；已有文件会保留。

仅有 CPU 时，先从 PyTorch 的 CPU 仓库安装对应版本，再使用 `--device cpu`。不保证 CPU 下的帧率。

```powershell
.\.venv\Scripts\python.exe -m vision_servo_tracker.desktop --device cpu --imgsz 640
```

### 操作

- 鼠标点击带编号的检测框：选择目标。多个框重叠时取较小的框，便于选择人物手里的手机。
- 空格：在尚未锁定时选择最靠近画面中心的目标。
- `R`：解除锁定。
- `Q`：退出并释放摄像头。

默认参数为摄像头索引0、输入尺寸960、检测入口阈值0.1、镜像开启。需要时可调整：

```powershell
.\.venv\Scripts\python.exe -m vision_servo_tracker.desktop --camera 1 --target-class person --imgsz 640 --no-mirror
```

目标丢失时保留所选 ID 和类别、清空位置，不自动切换到其他编号。相同 ID 和类别重新出现可恢复输出，但 ByteTrack ID 是临时编号，不保证遮挡或出画后的物理身份一致。

### 偏差约定

设检测框中心为 `(cx, cy)`，画面尺寸为 `(width, height)`：

```text
dx = cx - width / 2        # 向右为正
dy = height / 2 - cy       # 向上为正
x  = dx / (width / 2)
y  = dy / (height / 2)
```

镜像在检测前完成，点击、检测框和偏差始终使用同一张显示图像。舵机方向必须另行标定，不能直接把像素偏差当成角度。

运行摘要保存在 `runs/`，包含帧数、临时 ID、类别计数和目标状态，不保存视频。跨帧检测次数不是独立目标数量，处理帧数也不能代表检测准确率。

## 树莓派采集、电脑识别

摄像头接树莓派，由 ustreamer 输出 HTTP MJPEG；电脑读取视频并使用本机GPU检测。树莓派无需为此安装YOLO。

1. 树莓派以1280×720、MJPEG、15FPS启动取流服务，只监听本机：

   ```bash
   ustreamer --device=/dev/video0 --format=MJPEG --resolution=1280x720 --desired-fps=15 --host=127.0.0.1 --port=8080
   ```

2. 电脑建立SSH转发，替换用户名和地址，保持终端打开：

   ```powershell
   ssh -N -L 127.0.0.1:8080:127.0.0.1:8080 YOUR_USER@PI_ADDRESS
   ```

3. 在仓库根目录运行识别，或双击 `scripts/run_network.cmd`：

   ```powershell
   .\.venv\Scripts\python.exe -m vision_servo_tracker.desktop --source http://127.0.0.1:8080/stream
   ```

`--source`与`--camera`互斥。网络读取线程持续解包MJPEG，仅保留最新帧；推理变慢时不积压已接收的旧帧。三秒未收到新图像或连接失败时，窗口显示`STREAM OFFLINE`并清空偏差；自动重连后重置跟踪器并解除选择，需要重新点选目标。

首次30帧实时取流检查产生了跟踪编号，后续窗口已显示真实场景与检测框。该结果验证采集、传输和推理链路，不代表检测准确率、目标锁定稳定率或端到端延迟已经验收。画面质量、镜头朝向与Wi-Fi状况仍影响实际表现。

按`Q`关闭识别窗口，只结束电脑识别；树莓派取流程序与SSH转发需分别停止。

## 树莓派部署（待实机验收）

如果树莓派只负责采集、由电脑运行模型，可先使用下面的网络取流方案；本节的树莓派独立推理是另一种部署方式。

目标平台为 Raspberry Pi 5、64位 Raspberry Pi OS、C922 USB 摄像头。先确认系统和存储卡能够稳定读写，再安装依赖。

把仓库源文件放到树莓派后，在仓库根目录执行：

```bash
bash scripts/setup_pi.sh
.venv/bin/python -m vision_servo_tracker.web
```

脚本会安装系统依赖、CPU 版 PyTorch 和 NCNN，然后将模型导出为416输入尺寸。首次安装需要联网，NCNN 导出可能较慢。CPU 约束用于避免 ARM64 默认依赖下载不需要的 CUDA 组件。导出失败时脚本停止；成功导出不等于摄像头和网页链路已经验收。

服务只监听树莓派的 `127.0.0.1:8081`。在电脑另开终端，替换用户名与地址后建立转发：

```powershell
ssh -N -L 127.0.0.1:8081:127.0.0.1:8081 YOUR_USER@PI_ADDRESS
```

保持该窗口打开，浏览器访问 <http://127.0.0.1:8081/>。在运行网页服务的树莓派终端按 `Ctrl+C` 停止识别；只关闭网页不会结束推理。

网页点击附带实际显示帧编号，超过3秒或已从缓存移除的帧不能用于选择。摄像头故障或画面超过3秒未更新时，状态清空偏差并显示离线。相机读取线程只保留最新帧，避免 CPU 推理积压。

其他摄像头节点或模型尺寸可通过参数设置：

```bash
.venv/bin/python -m vision_servo_tracker.web --camera /dev/video0 --model models/yolo26n_ncnn_model --imgsz 416
```

NCNN 的推理尺寸必须与导出尺寸一致。将416改为640前，先用 `vision-export-ncnn --imgsz 640` 重新导出。更大的输入可能有利于小目标检测，但需要实测速度和漏检情况。

## 测试

核心逻辑与网页协议测试只用 Python 标准库，不打开摄像头、不下载模型：

只查看代码或运行这些测试时，直接安装基础包即可，不需要安装 PyTorch、OpenCV 或下载权重：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

开发时也可不安装包，直接从源码测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

它们覆盖重叠框点选、目标丢失、类别变化、偏差方向、离线清空、帧过期及 HTTP 控制请求。测试通过不能代替树莓派部署和舵机联调。

每次推送和提交 Pull Request 后，GitHub Actions 会在 Windows、Linux 的 Python 3.13 环境检查安装和核心测试；Linux 任务还检查安装脚本的 Bash 语法及网页脚本的 JavaScript 语法。流程不打开摄像头、不安装推理依赖、不发布模型。

后续验收项目见 [开发计划](docs/roadmap.md)。提交问题时可使用仓库中的故障报告模板，附上运行平台、复现步骤和报错即可，不需要提供摄像头画面或密码。

## 目录

```text
src/vision_servo_tracker/
  desktop.py           主机摄像头窗口
  target_lock.py       目标选择、状态和中心偏差
  web.py               树莓派推理线程与 HTTP 服务
  models.py            官方预训练权重下载
  export.py            NCNN 导出入口
  static/              网页预览和远程点选
scripts/               Windows 启动和树莓派安装
tests/                 目标逻辑与网页协议测试
docs/architecture.md   数据流、状态约定与硬件联调计划
```

第三方来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。权重、虚拟环境、日志、摄像头画面和个人设备配置不随仓库分发。
