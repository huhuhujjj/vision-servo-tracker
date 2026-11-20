# 第三方来源

本项目集成以下组件；这些组件和模型不是本项目自行训练或实现的。

| 组件 | 本项目中的用途 | 来源 |
| --- | --- | --- |
| Ultralytics / YOLO26n | 预训练 COCO 检测、模型导出、跟踪接口 | https://github.com/ultralytics/ultralytics |
| ByteTrack | 通过 Ultralytics 接口关联跨帧检测 | https://docs.ultralytics.com/modes/track/ |
| PyTorch / torchvision | GPU 或 CPU 推理与模型导出 | https://pytorch.org/ |
| OpenCV | 摄像头读取、画面处理和窗口交互 | https://opencv.org/ |
| NCNN | 树莓派 ARM 推理后端 | https://github.com/Tencent/ncnn |

Ultralytics 的开源版本使用 AGPL-3.0，另有商业许可选项；参见其[许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)和[许可说明](https://www.ultralytics.com/license)。本仓库未重新许可或打包第三方库及模型权重。分发或部署时应保留并遵守所用组件的许可证。

模型下载入口固定为 Ultralytics 官方 assets 发布地址，不提供其他来源的权重。运行环境、数据集、个人视频及依赖二进制不提交到仓库。
