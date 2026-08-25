# 导入 pandas，用于推断时间索引的采样频率。
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# 导入 TFB 深度预测模型基类，复用通用训练与预测流程。
from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
)
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.utils.get_device import get_device

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
    "use_variable_chunk": False,  # 是否启用固定 16-variable 的低显存分块流程。
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
        self._variable_names = []
        self._global_test_batch_active = False

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
        variable_names = list(train_valid_data.columns)
        exog_data = (covariates or {}).get("exog")
        if exog_data is not None:
            variable_names.extend(exog_data.columns)
        self._variable_names = [str(name) for name in variable_names]
        self._routing_epoch = 0
        self._routing_learning_rows = []
        self._global_test_batch_active = False
        self._output_dir = (
            Path(__file__).resolve().parent
            / dataset_name
            / (
                f"seqlen{self.config.seq_len}_lr{format(self.config.lr, '.12g')}_"
                f"runid{self._run_id}"
            )
        )
        if self.config.use_variable_chunk:
            result = self._forecast_fit_variable_chunks(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
            )
        else:
            result = super().forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                **kwargs,
            )
        return result

    def _backward_variable_chunks(
        self, input, target, series_dim, criterion, scaler=None
    ):
        model = self._core_model()
        context = model.prepare_variable_chunk_context(
            input, fp64=self.config.fp64
        )
        correction = context["correction"]
        correction_leaf = correction.detach().requires_grad_(True)
        router = model.residual_router
        router.begin_variable_chunk_statistics()
        try:
            for start in range(
                0, context["num_variables"], model.variable_chunk_size
            ):
                end = min(
                    start + model.variable_chunk_size,
                    context["num_variables"],
                )
                supervised_end = min(end, series_dim)
                supervised_width = max(0, supervised_end - start)

                if supervised_width == 0:
                    # exogenous-only 块没有 forecasting loss，仅保留原 routing 统计。
                    with torch.no_grad():
                        model.forward_variable_chunk(
                            context, start, end, correction=correction_leaf
                        )
                    continue

                output = model.forward_variable_chunk(
                    context, start, end, correction=correction_leaf
                )
                output = output[
                    :, -self.config.horizon:, :supervised_width
                ]
                chunk_target = target[
                    :, -self.config.horizon:, start:supervised_end
                ]
                output, chunk_target = self._post_process(
                    output, chunk_target
                )
                weighted_loss = criterion(output, chunk_target) * (
                    supervised_width / series_dim
                )
                if scaler is None:
                    weighted_loss.backward()
                else:
                    scaler.scale(weighted_loss).backward()
        finally:
            router.end_variable_chunk_statistics()

        # 各块已把梯度累积到小 correction leaf；这里只回传一次完整 Global 图。
        if correction_leaf.grad is not None and correction.requires_grad:
            correction.backward(correction_leaf.grad)

    def _forecast_fit_variable_chunks(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv=1.0,
    ):
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog")
        if exog_data is not None:
            train_valid_data = pd.concat(
                [train_valid_data, exog_data], axis=1
            )

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.model = self._init_model()
        # 分块训练需要在同一模型实例上累积 correction leaf 与各块梯度；
        # 保持顺序单模型执行，避免 DataParallel 令训练和验证使用不同副本语义。
        print(
            "----------------------------------------------------------",
            self.model_name,
        )

        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )
        self.scaler.fit(train_data.values)
        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            _, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        _, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        criterion, optimizer = self._init_criterion_and_optimizer()
        grad_scaler = (
            torch.cuda.amp.GradScaler() if config.use_amp == 1 else None
        )
        device = get_device()
        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            for input, target, input_mark, target_mark in self.train_data_loader:
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                self._backward_variable_chunks(
                    input,
                    target,
                    series_dim,
                    criterion,
                    scaler=grad_scaler,
                )
                if grad_scaler is None:
                    optimizer.step()
                else:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                if config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(
                    valid_data_loader, series_dim, criterion
                )
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint(self.model)
                if self.early_stopping.early_stop:
                    break

            if config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

    def _core_model(self):
        return getattr(self.model, "module", self.model)

    def _set_routing_statistics_mode(self, mode):
        if self.model is not None:
            self._core_model().set_routing_statistics_mode(mode)

    def _reset_global_test_collection(self):
        self._core_model().reset_global_token_statistics()

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

    _global_token_columns = [
        "record_type", "sample_index", "global_token_index",
        "variable_index", "variable_name", "other_variable_index",
        "other_variable_name", "patch_index", "statistic", "value"
    ]

    @classmethod
    def _global_token_frame(cls, record_type, values, **coordinates):
        values = np.asarray(values).reshape(-1)
        row_count = values.size
        data = {
            "record_type": np.full(row_count, record_type, dtype=object),
            "sample_index": np.full(row_count, np.nan),
            "global_token_index": np.full(row_count, np.nan),
            "variable_index": np.full(row_count, np.nan),
            "variable_name": np.full(row_count, "", dtype=object),
            "other_variable_index": np.full(row_count, np.nan),
            "other_variable_name": np.full(row_count, "", dtype=object),
            "patch_index": np.full(row_count, np.nan),
            "statistic": np.full(row_count, "", dtype=object),
            "value": values,
        }
        for name, coordinate in coordinates.items():
            coordinate = np.asarray(coordinate)
            if coordinate.ndim == 0:
                coordinate = np.full(row_count, coordinate.item())
            else:
                coordinate = coordinate.reshape(-1)
            if coordinate.size != row_count:
                raise ValueError(
                    f"Coordinate {name!r} does not match statistic rows."
                )
            data[name] = coordinate
        return pd.DataFrame(data, columns=cls._global_token_columns)

    @staticmethod
    def _entropy_summaries(values):
        statistic_names = (
            "mean", "std", "min", "p25", "median", "p75", "max"
        )
        if np.isnan(values).all():
            empty = np.full(values.shape[1], np.nan)
            return [(name, empty.copy()) for name in statistic_names]
        return [
            ("mean", np.nanmean(values, axis=0)),
            ("std", np.nanstd(values, axis=0, ddof=0)),
            ("min", np.nanmin(values, axis=0)),
            ("p25", np.nanpercentile(values, 25, axis=0)),
            ("median", np.nanpercentile(values, 50, axis=0)),
            ("p75", np.nanpercentile(values, 75, axis=0)),
            ("max", np.nanmax(values, axis=0)),
        ]

    def _global_token_statistic_frames(self):
        statistics = self._core_model().global_token_statistics()
        if statistics is None:
            return

        variable_routing = statistics["variable_routing"].numpy()
        patch_routing = statistics["patch_routing"].numpy()
        variable_entropy = statistics["variable_entropy"].numpy()
        patch_entropy = statistics["patch_entropy"].numpy()
        temporal_profile = statistics["temporal_profile"].numpy()
        temporal_similarity = statistics[
            "temporal_cosine_similarity"
        ].numpy()

        sample_count, num_global_tokens, num_variables = (
            variable_routing.shape
        )
        num_patches = patch_routing.shape[-1]
        if len(self._variable_names) == num_variables:
            variable_names = np.asarray(self._variable_names, dtype=object)
        else:
            variable_names = np.asarray([
                f"variable_{index}" for index in range(num_variables)
            ], dtype=object)

        # Sample-level distributions are chunked only for CSV memory usage;
        # no sample or head is averaged at the attention capture point.
        max_frame_rows = 200_000
        variable_rows_per_sample = num_global_tokens * num_variables
        samples_per_frame = max(
            1, max_frame_rows // variable_rows_per_sample
        )
        for start in range(0, sample_count, samples_per_frame):
            end = min(start + samples_per_frame, sample_count)
            chunk_sample_count = end - start
            variable_index = np.tile(
                np.arange(num_variables), chunk_sample_count * num_global_tokens
            )
            yield self._global_token_frame(
                "sample_variable_routing",
                variable_routing[start:end].reshape(-1),
                sample_index=np.repeat(
                    np.arange(start, end), variable_rows_per_sample
                ),
                global_token_index=np.tile(
                    np.repeat(np.arange(num_global_tokens), num_variables),
                    chunk_sample_count
                ),
                variable_index=variable_index,
                variable_name=variable_names[variable_index],
            )

        summary_global_index = np.repeat(
            np.arange(num_global_tokens), num_variables
        )
        summary_variable_index = np.tile(
            np.arange(num_variables), num_global_tokens
        )
        for statistic, values in (
            ("mean", variable_routing.mean(axis=0)),
            ("std", variable_routing.std(axis=0, ddof=0)),
        ):
            yield self._global_token_frame(
                "variable_routing_summary", values.reshape(-1),
                global_token_index=summary_global_index,
                variable_index=summary_variable_index,
                variable_name=variable_names[summary_variable_index],
                statistic=statistic,
            )

        entropy_sample_index = np.repeat(
            np.arange(sample_count), num_global_tokens
        )
        entropy_global_index = np.tile(
            np.arange(num_global_tokens), sample_count
        )
        yield self._global_token_frame(
            "sample_variable_normalized_entropy",
            variable_entropy.reshape(-1),
            sample_index=entropy_sample_index,
            global_token_index=entropy_global_index,
        )
        for statistic, values in self._entropy_summaries(variable_entropy):
            yield self._global_token_frame(
                "variable_normalized_entropy_summary", values,
                global_token_index=np.arange(num_global_tokens),
                statistic=statistic,
            )

        patch_rows_per_sample = num_global_tokens * num_patches
        samples_per_frame = max(1, max_frame_rows // patch_rows_per_sample)
        for start in range(0, sample_count, samples_per_frame):
            end = min(start + samples_per_frame, sample_count)
            chunk_sample_count = end - start
            yield self._global_token_frame(
                "sample_patch_routing",
                patch_routing[start:end].reshape(-1),
                sample_index=np.repeat(
                    np.arange(start, end), patch_rows_per_sample
                ),
                global_token_index=np.tile(
                    np.repeat(np.arange(num_global_tokens), num_patches),
                    chunk_sample_count
                ),
                patch_index=np.tile(
                    np.arange(num_patches),
                    chunk_sample_count * num_global_tokens
                ),
            )

        summary_global_index = np.repeat(
            np.arange(num_global_tokens), num_patches
        )
        summary_patch_index = np.tile(
            np.arange(num_patches), num_global_tokens
        )
        for statistic, values in (
            ("mean", patch_routing.mean(axis=0)),
            ("std", patch_routing.std(axis=0, ddof=0)),
        ):
            yield self._global_token_frame(
                "patch_routing_summary", values.reshape(-1),
                global_token_index=summary_global_index,
                patch_index=summary_patch_index,
                statistic=statistic,
            )

        yield self._global_token_frame(
            "sample_patch_normalized_entropy", patch_entropy.reshape(-1),
            sample_index=entropy_sample_index,
            global_token_index=entropy_global_index,
        )
        for statistic, values in self._entropy_summaries(patch_entropy):
            yield self._global_token_frame(
                "patch_normalized_entropy_summary", values,
                global_token_index=np.arange(num_global_tokens),
                statistic=statistic,
            )

        profile_global_index = np.repeat(
            np.arange(num_global_tokens), num_variables * num_patches
        )
        profile_variable_index = np.tile(
            np.repeat(np.arange(num_variables), num_patches),
            num_global_tokens
        )
        yield self._global_token_frame(
            "variable_patch_temporal_profile", temporal_profile.reshape(-1),
            global_token_index=profile_global_index,
            variable_index=profile_variable_index,
            variable_name=variable_names[profile_variable_index],
            patch_index=np.tile(
                np.arange(num_patches), num_global_tokens * num_variables
            ),
            statistic="mean_over_samples_and_heads",
        )

        variables_per_frame = max(1, max_frame_rows // num_variables)
        for global_token_index in range(num_global_tokens):
            for start in range(0, num_variables, variables_per_frame):
                end = min(start + variables_per_frame, num_variables)
                first_variable_index = np.repeat(
                    np.arange(start, end), num_variables
                )
                second_variable_index = np.tile(
                    np.arange(num_variables), end - start
                )
                yield self._global_token_frame(
                    "variable_temporal_cosine_similarity",
                    temporal_similarity[
                        global_token_index, start:end
                    ].reshape(-1),
                    global_token_index=global_token_index,
                    variable_index=first_variable_index,
                    variable_name=variable_names[first_variable_index],
                    other_variable_index=second_variable_index,
                    other_variable_name=variable_names[second_variable_index],
                    statistic="mean_over_samples_and_heads",
                )

            if num_variables > 1:
                off_diagonal = temporal_similarity[global_token_index][
                    ~np.eye(num_variables, dtype=bool)
                ]
                off_diagonal_values = np.asarray([
                    off_diagonal.mean(), off_diagonal.std(ddof=0)
                ])
            else:
                off_diagonal_values = np.asarray([np.nan, np.nan])
            yield self._global_token_frame(
                "variable_temporal_cosine_off_diagonal_summary",
                off_diagonal_values,
                global_token_index=global_token_index,
                statistic=np.asarray(["mean", "std"], dtype=object),
            )

    def _save_global_token_statistics(self):
        path = self._output_dir / f"horizon{self._artifact_horizon}.csv"
        wrote_header = False
        for frame in self._global_token_statistic_frames():
            if frame.empty:
                continue
            frame.to_csv(
                path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True

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
        self._save_global_token_statistics()

    def forecast(self, horizon, series, *, covariates=None):
        self._reset_global_test_collection()
        self._global_test_batch_active = False
        self._artifact_horizon = horizon
        self._set_routing_statistics_mode("test")
        result = super().forecast(
            horizon, series, covariates=covariates
        )
        self._save_routing_artifacts(include_visualizations=True)
        return result

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        new_collection = (
            not self._global_test_batch_active
            or self._artifact_horizon != horizon
        )
        self._artifact_horizon = horizon
        if new_collection:
            self._reset_global_test_collection()
            self._global_test_batch_active = True
        self._set_routing_statistics_mode("test")
        result = super().batch_forecast(horizon, batch_maker, **kwargs)
        has_more_batches = getattr(batch_maker, "has_more_batches", None)
        if has_more_batches is None or not has_more_batches():
            self._save_routing_artifacts(include_visualizations=True)
            self._global_test_batch_active = False
        return result

    # 执行一次 forward，并按 TFB 约定封装预测结果。
    # shape 约定：B 为 batch size，S 为 seq_len，H 为 horizon，V 为变量数，M 为时间特征数。
    # 输入 shape：input [B, S, V]，target [B, label_len+H, V]。
    # mark shape：input_mark [B, S, M]，target_mark [B, label_len+H, M]。
    def _process(self, input, target, input_mark, target_mark):
        # 仅使用 input tensor；返回 output tensor 的 shape 为 [B, H, V]。
        return {
            "output": self.model(
                input,
                fp64=self.config.fp64,
                use_variable_chunk=self.config.use_variable_chunk,
            )
        }
