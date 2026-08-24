# 导入 pandas，用于推断时间索引的采样频率。
import hashlib
import json
import os

import numpy as np
import pandas as pd
from pathlib import Path

# 导入 TFB 深度预测模型基类，复用通用训练与预测流程。
from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.data.data_pool import DataPool

# 导入 VisionTS 的核心模型实现。
from .models.model import VisionTS as VisionTSModel
# 导入“采样频率到季节周期”的转换工具。
from .utils.util import freq_to_seasonality_list


# 定义 VisionTS 在 TFB 中使用的默认超参数。
MODEL_HYPER_PARAMS = {
    "arch": "mae_base",  # 指定所使用的 MAE backbone 规格。
    "ckpt_path": None,  # 默认不显式指定 pretrained checkpoint 路径。
    "load_ckpt": True,  # 默认加载 pretrained checkpoint。
    "periodicity": "auto",  # 默认根据数据频率自动推断周期。
    "norm_const": 0.4,  # 设置输入 normalization 的缩放常数。
    "align_const": 0.4,  # 设置图像尺寸对齐时的控制常数。
    "interpolation": "bilinear",  # 使用 bilinear interpolation 调整输入尺寸。
    "num_latents": 1,  # 使用单个 latent query 聚合全局修正表示。
    "latent_dim": 192,  # 设置 latent embedding 的维度。
    "adapter_num_heads": 4,  # 设置 adapter 的 attention head 数量。
    "fp64": False,  # 默认不使用 float64 计算。
    "batch_size": 32,  # 设置默认训练 batch size。
    "num_epochs": 1,  # 设置默认训练 epoch 数。
}


# 定义接入 TFB 统一接口的 VisionTS 预测器。
class VisionTS(DeepForecastingModelBase):
    # 说明该类是支持全局 patch 修正机制的 VisionTS TFB adapter。
    """TFB adapter for globally corrected VisionTS patch tokens."""

    # 接收外部覆盖参数，并交由基类合并默认配置。
    def __init__(self, **kwargs):
        super(VisionTS, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        parent_pid = os.getppid()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        parent_stat = Path(f"/proc/{parent_pid}/stat").read_text()
        parent_start = parent_stat[parent_stat.rfind(")") + 2:].split()[19]
        excluded = {"horizon", "pred_len", "output_chunk_length"}
        group_config = {
            key: value for key, value in vars(self.config).items()
            if key not in excluded
        }
        fingerprint = json.dumps(
            [boot_id, parent_pid, parent_start, group_config],
            sort_keys=True, default=str
        )
        self._run_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:32]
        self._output_dir = None
        self._artifact_horizon = None
        self._routing_epoch = 0
        self._routing_learning_rows = []

    # 将模型名称暴露给 TFB 的注册与记录流程。
    @property
    def model_name(self):
        # 返回统一使用的模型标识。
        return "VisionTS"

    # 根据当前配置创建并初始化底层 VisionTS 模型。
    def _init_model(self):
        checkpoint_path = getattr(self.config, "checkpoint_path", None)
        if checkpoint_path is None:
            checkpoint_path = self.config.ckpt_path
        # 构造核心模型并传入 backbone 与全局修正 adapter 配置。
        model = VisionTSModel(
            arch=self.config.arch,  # 指定模型 backbone architecture。
            ckpt_path=checkpoint_path,  # 指定 pretrained checkpoint 路径。
            load_ckpt=self.config.load_ckpt,  # 控制是否加载 pretrained checkpoint。
            num_latents=self.config.num_latents,  # 传入全局修正 latent query 数量。
            latent_dim=self.config.latent_dim,  # 传入 latent embedding 维度。
            adapter_num_heads=self.config.adapter_num_heads,  # 传入 attention head 数量。
        )
        # 补充与当前预测任务相关、需要在运行时确定的配置。
        model.update_config(
            context_len=self.config.seq_len,  # 设置历史上下文长度。
            pred_len=self.config.horizon,  # 设置预测步数。
            periodicity=self.config.periodicity,  # 设置时间序列周期。
            norm_const=self.config.norm_const,  # 设置 normalization 常数。
            align_const=self.config.align_const,  # 设置尺寸对齐常数。
            interpolation=self.config.interpolation,  # 设置 interpolation 方式。
        )
        # 返回完成配置的底层模型。
        return model

    # 从训练集时间索引推断频率，并确定模型使用的周期。
    def _set_data_frequency(self, train_data):
        # 尝试从训练数据的时间索引推断完整频率字符串。
        frequency = pd.infer_freq(train_data.index)
        # 无法推断时说明时间间隔不规则，终止后续处理。
        if frequency is None:
            raise ValueError("Irregular time intervals")

        # 保留完整频率，供 TFB 补齐时间戳并生成时间特征。
        self.config.freq = frequency

        # 读取用户配置的周期值。
        periodicity = self.config.periodicity
        # 将空周期统一转换为自动推断标记 0。
        if periodicity is None:
            periodicity = 0
        # 若周期为字符串，则解析自动标记或整数字符串。
        if isinstance(periodicity, str):
            # 去除首尾空白并统一为小写，便于比较。
            value = periodicity.strip().lower()
            # 将支持的自动模式名称转换为标记 0。
            if value in {"auto", "freq"}:
                periodicity = 0
            else:
                # 尝试把显式字符串周期转换为整数。
                try:
                    periodicity = int(value)
                # 转换失败时抛出含明确配置要求的异常。
                except ValueError as error:
                    raise ValueError(
                        "periodicity must be a positive integer or 'auto'."
                    ) from error

        # 将数值周期统一转换为整数类型。
        periodicity = int(periodicity)
        # 标记 0 时，根据数据频率选取首个默认季节周期。
        if periodicity == 0:
            periodicity = freq_to_seasonality_list(frequency)[0]
        # 拒绝负周期，避免向模型传入无效配置。
        elif periodicity < 0:
            raise ValueError("periodicity must be positive or 'auto'.")
        # 保存最终确定的正周期供模型初始化使用。
        self.config.periodicity = periodicity

    # 完成多变量预测的基类调参后，设置数据频率与周期。
    def multi_forecasting_hyper_param_tune(self, train_data):
        # 先执行基类提供的多变量预测参数准备逻辑。
        super().multi_forecasting_hyper_param_tune(train_data)
        # 再从当前训练集确定频率相关配置。
        self._set_data_frequency(train_data)

    # 完成单变量预测的基类调参后，设置数据频率与周期。
    def single_forecasting_hyper_param_tune(self, train_data):
        # 先执行基类提供的单变量预测参数准备逻辑。
        super().single_forecasting_hyper_param_tune(train_data)
        # 再从当前训练集确定频率相关配置。
        self._set_data_frequency(train_data)

    @staticmethod
    def _dataset_name(train_valid_data, covariates):
        identity_data = train_valid_data
        exog_data = (covariates or {}).get("exog")
        if exog_data is not None:
            identity_data = pd.concat([identity_data, exog_data], axis=1)
        data_pool = DataPool().get_pool()
        dataset = getattr(data_pool, "_global_dataset", None)
        data_dict = dataset.get_state()["data_dict"] if dataset is not None else {}
        if len(data_dict) == 1:
            return Path(next(iter(data_dict))).stem
        matches = []
        for name, data in data_dict.items():
            if len(data) < len(identity_data):
                continue
            if not set(identity_data.columns).issubset(data.columns):
                continue
            candidate = data.iloc[:len(identity_data)].loc[
                :, identity_data.columns
            ]
            if candidate.equals(identity_data):
                matches.append(name)
        if len(matches) == 1:
            return Path(matches[0]).stem
        return "unknown_dataset"

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv=1.0,
        **kwargs,
    ):
        dataset_name = self._dataset_name(train_valid_data, covariates)
        self._routing_epoch = 0
        self._routing_learning_rows = []
        self._output_dir = (
            Path(__file__).resolve().parent
            / dataset_name
            / (
                f"seqlen{self.config.seq_len}_lr{format(self.config.lr, '.12g')}_"
                f"runid{self._run_id}"
            )
        )
        result = super().forecast_fit(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            **kwargs,
        )
        return result

    def _core_model(self):
        return getattr(self.model, "module", self.model)

    def _set_routing_statistics_mode(self, mode):
        if self.model is not None:
            self._core_model().set_routing_statistics_mode(mode)

    def validate(self, valid_data_loader, series_dim, criterion):
        self._set_routing_statistics_mode("validation")
        try:
            val_loss = super().validate(
                valid_data_loader, series_dim, criterion
            )
        finally:
            self._set_routing_statistics_mode("train")
        records = self._core_model().pop_routing_learning_records()
        if records:
            self._routing_epoch += 1
            for record in records:
                self._routing_learning_rows.append({
                    "epoch": self._routing_epoch,
                    **record,
                    "val_loss": float(val_loss),
                })
        return val_loss

    def _routing_deviation_frame(self):
        model = self._core_model()
        metadata = model.routing_metadata()
        source_names = list(metadata["source_names"])
        target_names = list(metadata["target_names"])
        matrix = model.routing_deviation_matrix()
        if matrix is None:
            return pd.DataFrame(), metadata
        frame = pd.DataFrame(
            matrix.numpy(), index=target_names, columns=source_names
        )
        frame.index.name = "target_branch"
        return frame, metadata

    def _save_routing_visualizations(self, frame, metadata):
        if frame.empty:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        source_names = list(metadata["source_names"])
        target_names = list(metadata["target_names"])
        mean_matrix = frame.to_numpy()
        prefix = f"horizon{self._artifact_horizon}"
        finite_values = np.abs(mean_matrix[np.isfinite(mean_matrix)])
        if finite_values.size == 0:
            return
        limit = finite_values.max()
        if limit == 0:
            limit = np.finfo(np.float64).eps

        figure, axis = plt.subplots(figsize=(14, 8))
        color_map = plt.get_cmap("coolwarm").copy()
        color_map.set_bad(color="#d9d9d9")
        image = axis.imshow(
            np.ma.masked_invalid(mean_matrix), aspect="auto",
            vmin=-limit, vmax=limit, cmap=color_map
        )
        axis.set_xticks(np.arange(len(source_names)), source_names, rotation=45,
                        ha="right")
        axis.set_yticks(np.arange(len(target_names)), target_names)
        axis.set_xlabel("Source branch")
        axis.set_ylabel("Target branch")
        axis.set_title("Mean Deviation-from-Uniform Routing Heatmap")
        figure.colorbar(
            image, ax=axis, label="Mean routing weight minus uniform weight"
        )
        figure.tight_layout()
        figure.savefig(
            self._output_dir / f"{prefix}_routing_deviation_heatmap.png",
            dpi=200
        )
        plt.close(figure)

    def _save_routing_artifacts(self, include_visualizations=False):
        if self._output_dir is None or self.model is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"horizon{self._artifact_horizon}"
        learning_columns = [
            "epoch", "target_branch", "query_norm", "routing_deviation",
            "normalized_entropy", "token_routing_variation",
            "correction_ratio", "val_loss"
        ]
        pd.DataFrame(
            self._routing_learning_rows, columns=learning_columns
        ).to_csv(
            self._output_dir / f"{prefix}_routing_learning.csv", index=False
        )
        frame, metadata = self._routing_deviation_frame()
        if not frame.empty:
            frame.to_csv(
                self._output_dir / f"{prefix}_routing_deviation.csv"
            )
        if include_visualizations and not frame.empty:
            self._save_routing_visualizations(frame, metadata)

    def forecast(self, horizon, series, *, covariates=None):
        self._artifact_horizon = horizon
        self._set_routing_statistics_mode("test")
        result = super().forecast(
            horizon, series, covariates=covariates
        )
        self._save_routing_artifacts(include_visualizations=True)
        return result

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        self._artifact_horizon = horizon
        self._set_routing_statistics_mode("test")
        result = super().batch_forecast(horizon, batch_maker, **kwargs)
        has_more_batches = getattr(batch_maker, "has_more_batches", None)
        if has_more_batches is None or not has_more_batches():
            self._save_routing_artifacts(include_visualizations=True)
        return result

    # 执行一次 forward，并按 TFB 约定封装预测结果。
    # shape 约定：B 为 batch size，S 为 seq_len，H 为 horizon，V 为变量数，M 为时间特征数。
    # 输入 shape：input [B, S, V]，target [B, label_len+H, V]。
    # mark shape：input_mark [B, S, M]，target_mark [B, label_len+H, M]。
    def _process(self, input, target, input_mark, target_mark):
        # 仅使用 input tensor；返回 output tensor 的 shape 为 [B, H, V]。
        return {"output": self.model(input, fp64=self.config.fp64)}
