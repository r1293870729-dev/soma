from __future__ import annotations  # 使用延迟类型注解，避免运行时提前解析注解对象。

import csv  # 用于把训练日志和 probe 日志逐行写入 CSV。
import json  # 用于保存 resolved_args.json 和 summary.json。
import random  # 用于固定 Python 标准库随机性。
from pathlib import Path  # 用于跨平台路径拼接。
from typing import Any  # 用于 JSON/CSV 工具函数的宽松类型标注。

import numpy as np  # 用于固定 NumPy 随机性。
import torch  # 用于固定 PyTorch 随机性并创建随机数发生器。


def set_seed(seed: int) -> None:  # 定义统一随机种子入口，所有训练脚本都先调用它。
    random.seed(int(seed))  # 固定 Python 标准库随机数。
    np.random.seed(int(seed))  # 固定 NumPy 随机数。
    torch.manual_seed(int(seed))  # 固定 PyTorch CPU 随机数。
    torch.cuda.manual_seed_all(int(seed))  # 固定所有 CUDA 设备随机数。
    torch.backends.cudnn.benchmark = False  # 关闭 CUDNN 自动搜索，减少非确定性。
    torch.backends.cudnn.deterministic = True  # 要求 CUDNN 尽量使用确定性算子。
    # 不调用 torch.use_deterministic_algorithms(True)，因为 CPU ConvTranspose 在该模式下会极慢；seed 仍然统一固定。


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:  # 创建独立随机流。
    return torch.Generator(device=str(device)).manual_seed(int(seed))  # 用固定 seed 初始化 generator。


def resolve_device(name: str) -> torch.device:  # 把命令行设备字符串解析成 torch.device。
    if name == "auto":  # auto 表示优先使用 GPU。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 CUDA 就用 cuda，否则用 cpu。
    return torch.device(name)  # 其它情况直接按用户传入的设备名解析。


def append_csv(path: Path, row: dict[str, Any]) -> None:  # 以追加方式写入 CSV 行。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保 CSV 所在目录存在。
    exists = path.exists()  # 判断文件是否已存在，决定是否写 header。
    with path.open("a", encoding="utf-8", newline="") as handle:  # 追加打开 CSV 文件。
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))  # 使用当前 row 的 key 作为列名。
        if not exists:  # 新文件需要先写表头。
            writer.writeheader()  # 写入 CSV header。
        writer.writerow(row)  # 写入当前日志行。


def write_json(path: Path, payload: dict[str, Any]) -> None:  # 保存 JSON 文件。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")  # 写入格式化 JSON，Path 会转为字符串。


def count_parameters(module: torch.nn.Module) -> int:  # 统计模型可训练参数量。
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)  # 只统计 requires_grad 参数。
