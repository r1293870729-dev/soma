from __future__ import annotations  # 使用延迟类型注解。

from dataclasses import dataclass  # 用于定义阶段配置结构。
from pathlib import Path  # 用于路径处理。
from typing import Any  # 用于缓存对象的宽松标注。

import numpy as np  # 用于读取 CIFAR-C npy 和筛选 dSprites 索引。
import torch  # 用于张量转换和随机采样。
from torch import Tensor  # Tensor 类型标注。
from torch.utils.data import DataLoader, Dataset  # Dataset/DataLoader 基类。
from torchvision import datasets as tv_datasets  # 用于读取 CIFAR-10。
from torchvision import transforms  # 用于 CIFAR-10 转 tensor。

from utils import make_generator  # 使用统一的独立随机数发生器。


class DspritesStore:  # 兼容 dSprites 的 npz 和 hdf5 两种存储格式。
    def __init__(self, path: Path) -> None:  # 初始化存储读取器。
        self.original_path = Path(path)  # 保存用户传入路径。
        self.path = self._resolve_path(self.original_path)  # 解析实际可读路径。
        self.kind = self.path.suffix.lower()  # 保存文件类型后缀。
        self._npz: Any | None = None  # npz 句柄延迟打开。
        self._h5: Any | None = None  # hdf5 句柄延迟打开。

    def _resolve_path(self, path: Path) -> Path:  # 根据可用依赖和文件存在性选择实际路径。
        if path.suffix.lower() == ".hdf5":  # 如果用户传入 hdf5。
            try:  # 尝试导入 h5py。
                import h5py  # type: ignore  # h5py 仅在需要 hdf5 时导入。
                _ = h5py  # 保留变量，避免静态检查误报未使用。
                return path  # h5py 可用时使用 hdf5。
            except Exception:  # 如果 h5py 不可用或 ABI 不兼容。
                npz_path = path.with_suffix(".npz")  # 尝试使用同名 npz。
                if npz_path.exists():  # 如果同名 npz 存在。
                    return npz_path  # fallback 到 npz。
                raise  # 没有 npz 时重新抛出 h5py 错误。
        return path  # 其它后缀直接使用原路径。

    def classes(self) -> np.ndarray:  # 读取 dSprites latents_classes。
        if self.path.suffix.lower() == ".npz":  # npz 格式。
            data = np.load(self.path, allow_pickle=True, encoding="latin1")  # 用 latin1 兼容旧 pickle 编码。
            return np.asarray(data["latents_classes"], dtype=np.int64)  # 返回 int64 classes。
        import h5py  # type: ignore  # hdf5 格式需要 h5py。
        with h5py.File(self.path, "r") as handle:  # 打开 hdf5。
            return np.asarray(handle["latents/classes"], dtype=np.int64)  # 返回 classes。

    def image(self, index: int) -> np.ndarray:  # 读取一张 dSprites 图像。
        if self.path.suffix.lower() == ".npz":  # npz 格式。
            if self._npz is None:  # 如果 npz 尚未打开。
                self._npz = np.load(self.path, allow_pickle=True, encoding="latin1")  # 延迟打开 npz。
            return np.asarray(self._npz["imgs"][int(index)], dtype=np.float32)  # 返回 64x64 图。
        if self._h5 is None:  # hdf5 格式且句柄未打开。
            import h5py  # type: ignore  # 延迟导入 h5py。
            self._h5 = h5py.File(self.path, "r")  # 打开 hdf5 句柄。
        return np.asarray(self._h5["imgs"][int(index)], dtype=np.float32)  # 返回 64x64 图。
    def images(self, indices: np.ndarray) -> np.ndarray:  # 批量读取一组 dSprites 图像，避免训练时逐样本随机读盘。
        indices = np.asarray(indices, dtype=np.int64)  # 把输入索引统一成 int64，方便 npz/hdf5 读取。
        if self.path.suffix.lower() == ".npz":  # npz 格式通常已经整体压缩在一个文件里。
            if self._npz is None:  # 如果 npz 尚未打开。
                self._npz = np.load(self.path, allow_pickle=True, encoding="latin1")  # 延迟打开 npz，并兼容旧版 dSprites pickle 编码。
            return np.asarray(self._npz["imgs"][indices], dtype=np.uint8)  # 直接批量取出抽样后的二值图像，并保持 uint8 节省内存。
        if self._h5 is None:  # hdf5 格式且句柄未打开。
            import h5py  # type: ignore  # 延迟导入 h5py，避免没有 h5py 时影响其它数据集。
            self._h5 = h5py.File(self.path, "r")  # 打开 hdf5 句柄。
        order = np.argsort(indices)  # hdf5 对大量随机高级索引很慢，所以先排序，把连续索引合并成切片读取。
        sorted_indices = indices[order]  # 得到递增索引，便于识别连续块。
        pieces: list[np.ndarray] = []  # 保存每个连续块读出的图像数组。
        block_start = 0  # 当前连续块在 sorted_indices 中的起点。
        for pos in range(1, len(sorted_indices) + 1):  # 扫描所有索引，并在连续块结束时读取一次 hdf5 切片。
            is_end = pos == len(sorted_indices)  # 判断是否已经到最后一个索引。
            is_break = (not is_end) and sorted_indices[pos] != sorted_indices[pos - 1] + 1  # 判断相邻索引是否不再连续。
            if is_end or is_break:  # 如果一个连续块结束。
                start = int(sorted_indices[block_start])  # 当前连续块的 hdf5 起始索引。
                stop = int(sorted_indices[pos - 1]) + 1  # 当前连续块的 hdf5 结束索引，Python 切片右开。
                pieces.append(np.asarray(self._h5["imgs"][start:stop], dtype=np.uint8))  # 用切片读取连续块，避免慢速随机高级索引。
                block_start = pos  # 下一个连续块从当前位置开始。
        sorted_images = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 64, 64), dtype=np.uint8)  # 拼回递增索引对应图像。
        inverse_order = np.empty_like(order)  # 准备把递增顺序还原回调用方传入顺序。
        inverse_order[order] = np.arange(len(order))  # inverse_order[i] 表示原第 i 个索引在 sorted_images 中的位置。
        return sorted_images[inverse_order]  # 按调用方原始顺序返回图像。


@dataclass(frozen=True)  # 阶段配置是只读结构，避免训练中被误改。
class StageSpec:  # 描述一个训练阶段。
    name: str  # 阶段名，例如 s1_position。
    dataset: Dataset[Any]  # 当前阶段的数据集对象。
    active_tasks: tuple[str, ...]  # 当前阶段真实变化的 probe 任务。
    num_classes: dict[str, int]  # 每个 probe 任务的类别数。


class DspritesFactorDataset(Dataset[dict[str, Tensor]]):  # dSprites 因子筛选数据集。
    factor_sizes = {"shape": 3, "scale": 6, "orientation": 40, "x": 32, "y": 32}  # dSprites 各因子类别数。

    def __init__(self, path: Path, phase: str, samples: int, seed: int) -> None:  # 初始化某个阶段的数据集。
        self.path = Path(path)  # 保存 hdf5 路径。
        self.phase = str(phase)  # 保存阶段类型。
        self.samples = int(samples)  # 保存虚拟样本数。
        self.seed = int(seed)  # 保存阶段采样 seed。
        self.store = DspritesStore(self.path)  # 创建兼容 npz/hdf5 的 dSprites 读取器。
        classes = self.store.classes()  # 读取 latents_classes，列顺序为 color/shape/scale/orientation/x/y。
        mask = np.ones(classes.shape[0], dtype=bool)  # 初始化全 True mask。
        mask &= classes[:, 2] == 3  # 固定 scale 为中间尺度，减少无关变量。
        if self.phase == "position":  # S1 只变化位置。
            mask &= classes[:, 1] == 0  # 固定 shape。
            mask &= classes[:, 3] == 0  # 固定 orientation。
        elif self.phase == "position_shape":  # S2 变化位置和 shape。
            mask &= classes[:, 3] == 0  # 固定 orientation。
        elif self.phase == "position_shape_orientation":  # S3 变化位置、shape 和 orientation。
            pass  # S3 不额外限制 shape/orientation。
        else:  # 未知阶段直接报错。
            raise ValueError(f"Unknown dSprites phase: {self.phase}")  # 给出明确错误。
        valid_indices = np.flatnonzero(mask).astype(np.int64)  # 得到所有符合当前阶段的真实样本索引，天然按 hdf5 文件顺序递增。
        rng = np.random.default_rng(self.seed)  # 使用固定 seed 的 NumPy 随机数发生器。
        replace = self.samples > len(valid_indices)  # 如果虚拟样本数超过可用索引，则允许重复采样。
        self.sample_positions = rng.choice(np.arange(len(valid_indices)), size=self.samples, replace=replace).astype(np.int64)  # 在合法样本表内抽样位置。
        self.indices = valid_indices[self.sample_positions]  # 把合法样本位置映射回 dSprites 原始索引。
        self.classes = classes[self.indices]  # 缓存抽样后的标签，避免每次再索引大数组。
        self.image_lookup = torch.from_numpy(self.sample_positions)  # 虚拟样本直接指向合法样本缓存中的位置。
        self.images = torch.from_numpy(self.store.images(valid_indices)).unsqueeze(1)  # 缓存当前阶段所有合法图像，训练时只做内存索引。
    def __len__(self) -> int:  # 返回虚拟数据集大小。
        return int(self.samples)  # 使用固定 samples。

    def __getitem__(self, item: int) -> dict[str, Tensor]:  # 读取一个样本。
        image_tensor = self.images[self.image_lookup[int(item)]].to(torch.float32)  # 通过映射读取缓存图像，并转成模型需要的 float。
        cls = self.classes[int(item)]  # 读取对应因子标签。
        labels = torch.tensor([cls[4], cls[5], cls[1], cls[3]], dtype=torch.long)  # 保存 x,y,shape,orientation。
        return {"image": image_tensor, "labels": labels}  # 返回图像和 probe 标签。


class CifarCleanDataset(Dataset[dict[str, Tensor]]):  # CIFAR-10 clean 阶段数据集。
    def __init__(self, root: Path, train: bool, download: bool) -> None:  # 初始化 CIFAR-10。
        transform = transforms.ToTensor()  # 将 PIL 图像转为 0-1 tensor。
        self.dataset = tv_datasets.CIFAR10(root=str(root), train=train, download=download, transform=transform)  # 读取 CIFAR-10。

    def __len__(self) -> int:  # 返回数据集大小。
        return len(self.dataset)  # 直接使用 torchvision 数据集长度。

    def __getitem__(self, item: int) -> dict[str, Tensor]:  # 读取一个样本。
        image, label = self.dataset[int(item)]  # 读取图像和类别。
        return {"image": image, "labels": torch.tensor([int(label), 0], dtype=torch.long)}  # labels 为 class, domain。


class CifarCDataset(Dataset[dict[str, Tensor]]):  # 单一 CIFAR-C corruption/severity 数据集。
    def __init__(self, root: Path, corruption: str, severity: int, domain_id: int) -> None:  # 初始化 CIFAR-C 阶段。
        self.root = Path(root)  # 保存 CIFAR-C 根目录。
        self.corruption = str(corruption)  # 保存 corruption 名。
        self.severity = int(severity)  # 保存 severity。
        self.domain_id = int(domain_id)  # 保存 domain 标签。
        if not 1 <= self.severity <= 5:  # CIFAR-C severity 必须在 1 到 5。
            raise ValueError("CIFAR-C severity must be in [1, 5].")  # 给出明确错误。
        image_path = self.root / f"{self.corruption}.npy"  # 当前 corruption 的图像文件。
        label_path = self.root / "labels.npy"  # CIFAR-C 标签文件。
        if not image_path.exists():  # 检查图像文件是否存在。
            raise FileNotFoundError(f"Missing CIFAR-C file: {image_path}")  # 缺失时报错。
        if not label_path.exists():  # 检查标签文件是否存在。
            raise FileNotFoundError(f"Missing CIFAR-C labels file: {label_path}")  # 缺失时报错。
        all_images = np.load(image_path, mmap_mode="r")  # 以 mmap 方式读取图像，避免一次性占内存。
        all_labels = np.load(label_path, mmap_mode="r")  # 以 mmap 方式读取标签。
        start = (self.severity - 1) * 10_000  # CIFAR-C 每个 severity 对应 10000 张图。
        end = self.severity * 10_000  # 当前 severity 结束位置。
        self.images = all_images[start:end]  # 当前 severity 的图像视图。
        self.labels = all_labels[start:end]  # 当前 severity 的标签视图。

    def __len__(self) -> int:  # 返回当前 severity 的样本数。
        return int(self.images.shape[0])  # 通常为 10000。

    def __getitem__(self, item: int) -> dict[str, Tensor]:  # 读取一个样本。
        image = np.asarray(self.images[int(item)], dtype=np.float32) / 255.0  # 转为 0-1 float。
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # HWC 转为 CHW。
        class_label = int(self.labels[int(item)])  # 读取 CIFAR-10 类别。
        labels = torch.tensor([class_label, self.domain_id], dtype=torch.long)  # labels 为 class, domain。
        return {"image": image_tensor, "labels": labels}  # 返回图像和标签。


class Shapes3DStore:  # 3D Shapes HDF5 读取器，只在需要时打开文件句柄。
    def __init__(self, path: Path) -> None:  # 初始化 3D Shapes 存储读取器。
        self.path = Path(path)  # 保存 3D Shapes 的 h5 文件路径。
        if not self.path.exists():  # 训练开始前先检查文件是否存在。
            raise FileNotFoundError(f"Missing 3D Shapes file: {self.path}")  # 给出明确的数据缺失错误。
        self._h5: Any | None = None  # HDF5 文件句柄延迟打开，避免只读 labels 时长期占用句柄。

    def _handle(self) -> Any:  # 返回可复用的 HDF5 文件句柄。
        if self._h5 is None:  # 如果当前进程还没有打开 HDF5。
            import h5py  # type: ignore  # h5py 只在使用 3D Shapes 时导入，避免影响其它数据集。
            self._h5 = h5py.File(self.path, "r")  # 以只读方式打开 3D Shapes 文件。
        return self._h5  # 返回已打开的句柄。

    def labels(self) -> np.ndarray:  # 读取 3D Shapes 的连续因子标签。
        import h5py  # type: ignore  # 局部导入 h5py，保证没有 3D Shapes 时其它流程仍可运行。
        with h5py.File(self.path, "r") as handle:  # 临时打开文件读取 labels。
            return np.asarray(handle["labels"], dtype=np.float64)  # labels 形状为 N x 6，列为六个生成因子。

    def images(self, indices: np.ndarray, chunk_size: int = 2048) -> np.ndarray:  # 批量读取抽样后的 RGB 图像。
        indices = np.asarray(indices, dtype=np.int64)  # 统一索引类型，方便排序和 HDF5 读取。
        if indices.size == 0:  # 空索引直接返回空图像数组。
            return np.empty((0, 64, 64, 3), dtype=np.uint8)  # 保持 3D Shapes 图像形状。
        order = np.argsort(indices)  # HDF5 fancy indexing 要求索引递增，因此先排序。
        sorted_indices = indices[order]  # 得到递增顺序的索引。
        pieces: list[np.ndarray] = []  # 保存分块读取出的图像。
        handle = self._handle()  # 取得复用文件句柄。
        block_start = 0  # 当前连续块在 sorted_indices 中的起点。
        for pos in range(1, len(sorted_indices) + 1):  # 扫描排序索引，把连续位置合并成 HDF5 切片。
            is_end = pos == len(sorted_indices)  # 判断是否到达最后一个索引。
            is_break = (not is_end) and sorted_indices[pos] != sorted_indices[pos - 1] + 1  # 判断是否出现非连续位置。
            if is_end or is_break:  # 连续块结束时读取一次 HDF5。
                start_index = int(sorted_indices[block_start])  # 连续块的 HDF5 起始索引。
                stop_index = int(sorted_indices[pos - 1]) + 1  # 连续块的 HDF5 结束索引，右开。
                if stop_index - start_index <= int(chunk_size):  # 小连续块直接一次切片读取。
                    pieces.append(np.asarray(handle["images"][start_index:stop_index], dtype=np.uint8))  # 用切片读取连续图像。
                else:  # 超长连续块拆成多个小切片，降低单次内存峰值。
                    for chunk_start in range(start_index, stop_index, int(chunk_size)):  # 对长块做分块切片。
                        chunk_stop = min(stop_index, chunk_start + int(chunk_size))  # 当前子块右边界。
                        pieces.append(np.asarray(handle["images"][chunk_start:chunk_stop], dtype=np.uint8))  # 读取当前子块。
                block_start = pos  # 下一个连续块从当前位置开始。
        sorted_images = np.concatenate(pieces, axis=0)  # 拼回递增顺序的图像缓存。
        inverse_order = np.empty_like(order)  # 准备把递增顺序还原成调用方原始顺序。
        inverse_order[order] = np.arange(len(order))  # inverse_order[i] 指向第 i 个原始索引在 sorted_images 中的位置。
        return sorted_images[inverse_order]  # 按调用方传入顺序返回图像。

    def images_by_scan(self, indices: np.ndarray, scan_chunk: int = 4096) -> np.ndarray:  # 顺序扫描 HDF5 来读取稀疏索引。
        indices = np.asarray(indices, dtype=np.int64)  # 统一索引类型。
        if indices.size == 0:  # 空索引直接返回空数组。
            return np.empty((0, 64, 64, 3), dtype=np.uint8)  # 保持 3D Shapes 图像形状。
        order = np.argsort(indices)  # 先排序，便于最终还原顺序。
        sorted_indices = indices[order]  # 得到递增目标索引。
        handle = self._handle()  # 取得 HDF5 文件句柄。
        total = int(handle["images"].shape[0])  # 读取数据集中图像总数。
        keep = np.zeros(total, dtype=bool)  # 创建目标索引布尔表，便于顺序扫描时快速判断。
        keep[sorted_indices] = True  # 标记需要保留的图像位置。
        pieces: list[np.ndarray] = []  # 保存扫描过程中命中的图像块。
        start_scan = int(sorted_indices[0])  # 从最小目标索引所在位置开始扫。
        end_scan = int(sorted_indices[-1]) + 1  # 扫到最大目标索引之后。
        for start in range(start_scan, end_scan, int(scan_chunk)):  # 按连续大块顺序扫描 HDF5。
            stop = min(end_scan, start + int(scan_chunk))  # 当前扫描块右边界。
            relative = np.flatnonzero(keep[start:stop])  # 找出当前扫描块内真正需要的相对位置。
            if relative.size == 0:  # 如果当前块没有目标样本。
                continue  # 不读取该块，节省解压和内存。
            block = np.asarray(handle["images"][start:stop], dtype=np.uint8)  # 连续读取当前块。
            pieces.append(block[relative])  # 只保留目标样本。
        sorted_images = np.concatenate(pieces, axis=0)  # 拼接为递增索引顺序的图像。
        inverse_order = np.empty_like(order)  # 准备还原调用方原始顺序。
        inverse_order[order] = np.arange(len(order))  # 构造逆排列。
        return sorted_images[inverse_order]  # 返回原始顺序图像。


class Shapes3DFactorDataset(Dataset[dict[str, Tensor]]):  # 3D Shapes 三阶段因子流数据集。
    factor_names = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")  # labels 六列对应的因子名。
    factor_sizes = {"floor_hue": 10, "wall_hue": 10, "object_hue": 10, "scale": 8, "shape": 4, "orientation": 15}  # 每个因子的离散类别数。

    @staticmethod
    def _sample_contiguous_blocks(valid_indices: np.ndarray, samples: int, rng: np.random.Generator, block_size: int = 512) -> np.ndarray:  # 从合法索引中抽连续块，避免 HDF5 随机读取过慢。
        if samples <= 0:  # 防御性处理空样本数。
            return np.empty((0,), dtype=np.int64)  # 返回空索引。
        if samples >= len(valid_indices):  # 如果需要的样本数覆盖合法集合。
            return valid_indices.copy()  # 直接使用全部合法索引。
        pieces: list[np.ndarray] = []  # 保存每个连续块。
        remaining = int(samples)  # 记录还需要多少样本。
        max_block = max(1, int(block_size))  # 保证块大小至少为 1。
        while remaining > 0:  # 直到凑够目标样本数。
            current = min(max_block, remaining)  # 当前块长度不超过剩余样本数。
            start = int(rng.integers(0, len(valid_indices) - current + 1))  # 随机选择合法索引表中的块起点。
            pieces.append(valid_indices[start : start + current])  # 取连续合法索引块，HDF5 后续可用切片/近连续读取。
            remaining -= current  # 更新剩余样本数。
        return np.concatenate(pieces, axis=0).astype(np.int64)  # 拼接成阶段样本索引。

    def __init__(self, path: Path, phase: str, samples: int, seed: int) -> None:  # 初始化某个阶段的 3D Shapes 数据流。
        self.path = Path(path)  # 保存 h5 路径。
        self.phase = str(phase)  # 保存阶段类型。
        self.samples = int(samples)  # 保存该阶段虚拟样本数。
        self.seed = int(seed)  # 保存阶段抽样 seed。
        self.store = Shapes3DStore(self.path)  # 创建 HDF5 读取器。
        raw_labels = self.store.labels()  # 读取所有连续因子值，用于筛选和离散化。
        class_columns: list[np.ndarray] = []  # 保存每个因子的离散类别编号。
        for column in range(raw_labels.shape[1]):  # 遍历六个因子列。
            values = np.unique(raw_labels[:, column])  # 取该因子的所有合法取值。
            class_columns.append(np.searchsorted(values, raw_labels[:, column]).astype(np.int64))  # 将连续值映射为 0-based 类别编号。
        classes = np.stack(class_columns, axis=1)  # 得到 N x 6 的整数标签矩阵。
        mask = np.ones(classes.shape[0], dtype=bool)  # 初始候选集合为所有 3D Shapes 样本。
        if self.phase == "background":  # S1 只让 floor_hue 和 wall_hue 变化。
            mask &= classes[:, 2] == 0  # 固定 object_hue，避免物体颜色提前成为变化因子。
            mask &= classes[:, 3] == 4  # 固定 scale 为中间附近取值，减少尺寸影响。
            mask &= classes[:, 4] == 0  # 固定 shape，避免类别因素提前变化。
            mask &= classes[:, 5] == 0  # 固定 orientation，避免姿态因素提前变化。
        elif self.phase == "background_object":  # S2 加入 object_hue 和 scale。
            mask &= classes[:, 4] == 0  # 仍固定 shape，只观察背景/颜色/尺度变化。
            mask &= classes[:, 5] == 0  # 仍固定 orientation，姿态变化留给 S3。
        elif self.phase == "background_object_shape_pose":  # S3 六个因子全部变化。
            pass  # 不再额外筛选，保留完整 3D Shapes 因子组合。
        else:  # 未知阶段名直接报错，防止静默跑错数据流。
            raise ValueError(f"Unknown 3D Shapes phase: {self.phase}")  # 明确指出非法 phase。
        valid_indices = np.flatnonzero(mask).astype(np.int64)  # 得到当前阶段所有合法样本索引。
        rng = np.random.default_rng(self.seed)  # 使用固定 seed 的 NumPy 随机数发生器。
        if self.phase == "background_object_shape_pose":  # S3 合法集合很大，随机 HDF5 读取会很慢。
            self.indices = self._sample_contiguous_blocks(valid_indices, self.samples, rng)  # 改为多段连续块抽样，保留全因子变化同时提升读取速度。
        else:  # S1/S2 合法集合较小，可以直接从合法集合中随机抽样。
            replace = self.samples > len(valid_indices)  # 如果目标虚拟样本数大于合法集合，则允许重复抽样。
            self.indices = rng.choice(valid_indices, size=self.samples, replace=replace).astype(np.int64)  # 从合法集合抽出固定阶段流。
        self.classes = classes[self.indices]  # 缓存抽样后样本的六因子类别标签。
        unique_indices, inverse = np.unique(self.indices, return_inverse=True)  # 只缓存抽样中真正出现过的图像，降低内存占用。
        self.image_lookup = torch.from_numpy(inverse.astype(np.int64))  # 每个虚拟样本指向唯一图像缓存中的位置。
        cache_dir = self.path.parent / "cache_dataset_generalization_xi"  # 3D Shapes 阶段图像缓存目录。
        cache_dir.mkdir(parents=True, exist_ok=True)  # 确保缓存目录存在。
        cache_path = cache_dir / f"{self.phase}_samples{self.samples}_seed{self.seed}.pt"  # 缓存键由阶段、样本数和 seed 决定。
        if cache_path.exists():  # 如果之前已经构建过同样阶段。
            payload = torch.load(cache_path, map_location="cpu")  # 直接加载缓存图像，避免重复扫描 HDF5。
            self.images = payload["images"]  # 复用唯一图像缓存。
        else:  # 如果缓存不存在，需要从 HDF5 构建一次。
            if self.phase == "background_object" and len(unique_indices) > 1000:  # S2 大样本索引很稀疏，直接 fancy/random 读取很慢。
                images = self.store.images_by_scan(unique_indices)  # 用顺序扫描方式读取稀疏目标图像。
            else:  # S1 很小、S3 为连续块抽样，普通连续切片读取即可。
                images = self.store.images(unique_indices)  # 读取唯一 RGB 图像。
            self.images = torch.from_numpy(images)  # 缓存唯一 RGB 图像，训练时只做内存索引。
            torch.save({"images": self.images}, cache_path)  # 保存磁盘缓存，后续 latent 维度复用。

    def __len__(self) -> int:  # 返回虚拟数据集长度。
        return int(self.samples)  # 使用固定 samples 保证每个阶段训练步数可控。

    def __getitem__(self, item: int) -> dict[str, Tensor]:  # 读取一个样本。
        image = self.images[self.image_lookup[int(item)]].to(torch.float32) / 255.0  # 取 RGB uint8 图像并归一化到 0-1。
        image_tensor = image.permute(2, 0, 1).contiguous()  # 3D Shapes 原始为 HWC，模型需要 CHW。
        labels = torch.tensor(self.classes[int(item)], dtype=torch.long)  # labels 为六个 0-based 因子类别。
        return {"image": image_tensor, "labels": labels}  # 返回图像与 probe 标签。


def make_loader(dataset: Dataset[Any], batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:  # 创建可复现 DataLoader。
    return DataLoader(  # 返回 PyTorch DataLoader。
        dataset,  # 当前阶段数据集。
        batch_size=int(batch_size),  # batch size。
        shuffle=bool(shuffle),  # 是否打乱。
        drop_last=True,  # 保证每个 batch 大小一致，方便 Xi 窗口。
        num_workers=0,  # 使用单进程加载，减少随机性和 Windows/服务器差异。
        pin_memory=torch.cuda.is_available(),  # 有 GPU 时启用 pinned memory。
        generator=make_generator(int(seed)),  # 固定 shuffle 随机性。
    )


def build_dsprites_stages(path: Path, samples_per_stage: int, batch_size: int, seed: int) -> list[StageSpec]:  # 构造 dSprites 三阶段。
    phases = ("position", "position_shape", "position_shape_orientation")  # 三个阶段内部名称。
    names = ("S1_position", "S2_position_shape", "S3_position_shape_orientation")  # 三个阶段日志名称。
    active = (("x", "y"), ("x", "y", "shape"), ("x", "y", "shape", "orientation"))  # 每阶段有效 probe。
    num_classes = {"x": 32, "y": 32, "shape": 3, "orientation": 40}  # dSprites probe 类别数。
    stages: list[StageSpec] = []  # 初始化阶段列表。
    for offset, phase in enumerate(phases):  # 逐阶段构造。
        dataset = DspritesFactorDataset(path, phase, samples_per_stage, seed + offset * 101)  # 构造阶段数据集。
        stages.append(StageSpec(names[offset], dataset, active[offset], num_classes))  # 加入阶段配置。
    return stages  # 返回三阶段。


def build_shapes3d_stages(path: Path, samples_per_stage: int, batch_size: int, seed: int) -> list[StageSpec]:  # 构造 3D Shapes 三阶段。
    phases = ("background", "background_object", "background_object_shape_pose")  # 三个阶段的筛选逻辑名称。
    names = ("S1_background", "S2_background_object", "S3_background_object_shape_pose")  # 三个阶段的日志名称。
    active = (("floor_hue", "wall_hue"), ("floor_hue", "wall_hue", "object_hue", "scale"), Shapes3DFactorDataset.factor_names)  # 每阶段真实变化的 probe 因子。
    num_classes = dict(Shapes3DFactorDataset.factor_sizes)  # 六个 3D Shapes probe 任务的类别数。
    stages: list[StageSpec] = []  # 初始化阶段列表。
    for offset, phase in enumerate(phases):  # 逐阶段构造数据集。
        dataset = Shapes3DFactorDataset(path, phase, samples_per_stage, seed + offset * 101)  # 每阶段使用独立但可复现的抽样 seed。
        stages.append(StageSpec(names[offset], dataset, tuple(active[offset]), num_classes))  # 加入阶段配置。
    return stages  # 返回三阶段。


def build_cifar_stages(cifar10_root: Path, cifarc_root: Path, download_cifar10: bool) -> list[StageSpec]:  # 构造 CIFAR 三阶段。
    clean = CifarCleanDataset(cifar10_root, train=True, download=download_cifar10)  # S1 clean CIFAR-10。
    motion = CifarCDataset(cifarc_root, corruption="motion_blur", severity=3, domain_id=1)  # S2 motion_blur s3。
    frost = CifarCDataset(cifarc_root, corruption="frost", severity=3, domain_id=2)  # S3 frost s3。
    num_classes = {"class": 10, "domain": 3}  # CIFAR probe 类别数。
    return [  # 返回阶段配置列表。
        StageSpec("S1_cifar10_clean", clean, ("class",), num_classes),  # S1 只把 class 作为主 probe。
        StageSpec("S2_motion_blur_s3", motion, ("class", "domain"), num_classes),  # S2 class/domain 都可记录。
        StageSpec("S3_frost_s3", frost, ("class", "domain"), num_classes),  # S3 class/domain 都可记录。
    ]
