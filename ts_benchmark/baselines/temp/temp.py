# 导入 pandas，用于推断时间索引的采样频率。
# import hashlib  # Disabled summary diagnostics.
# import json  # Disabled summary diagnostics.
# import os  # Disabled summary diagnostics.

# import numpy as np  # Disabled summary diagnostics.
import pandas as pd
import torch
# from pathlib import Path  # Disabled summary diagnostics.
from torch.utils.data import DataLoader

# 导入 TFB 深度预测模型基类，复用通用训练与预测流程。
from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
)
# from ts_benchmark.data.data_pool import DataPool  # Disabled summary diagnostics.
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
    "num_latents": 1,  # 每个变量使用一个 channel token。
    "latent_dim": 192,  # 设置 latent embedding 的维度。
    "adapter_num_heads": 4,  # 设置 adapter 的 attention head 数量。
    "tp_bottleneck_dim": 64,  # 设置 TP Adapter 的 bottleneck 维度。
    "channel_depth": 1,
    "ablation_mode": "full",
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
        # Existing summary/diagnostic state is intentionally disabled.
        # parent_pid = os.getppid()
        # boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        # parent_stat = Path(f"/proc/{parent_pid}/stat").read_text()
        # parent_start = parent_stat[parent_stat.rfind(")") + 2:].split()[19]
        # excluded = {"horizon", "pred_len", "output_chunk_length"}
        # group_config = {
        #     key: value for key, value in vars(self.config).items()
        #     if key not in excluded
        # }
        # fingerprint = json.dumps(
        #     [boot_id, parent_pid, parent_start, group_config],
        #     sort_keys=True, default=str
        # )
        # self._run_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:32]
        # self._summary_dir = None
        # self._collect_channel_statistics = False
        # self._channel_statistics_horizon = None
        # self._channel_statistics_sum = None
        # self._channel_statistics_count = None
        # self._channel_statistics_batch_active = False
        # self._validation_epoch = 0
        # self._best_validation_mse = float("inf")
        # self._best_relative_biases = None
        self._series_dim = None

    # 将模型名称暴露给 TFB 的注册与记录流程。
    @property
    def model_name(self):
        # 返回统一使用的模型标识。
        return "VisionTS"

    def _init_criterion_and_optimizer(self):
        gate = self._core_model().fusion_logit
        adapter_parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad and parameter is not gate
        ]
        if gate is None:
            optimizer = torch.optim.Adam([{
                "params": adapter_parameters,
                "lr": self.config.lr,
                "parameter_group": "adapters",
            }])
        else:
            optimizer = torch.optim.Adam([
                {
                    "params": adapter_parameters,
                    "lr": self.config.lr,
                    "parameter_group": "adapters",
                },
                {
                    "params": [gate],
                    "lr": 10 * self.config.lr,
                    "weight_decay": 0.0,
                    "parameter_group": "gate",
                },
            ])
        return torch.nn.MSELoss(), optimizer

    def _adjust_lr(self, optimizer, epoch, config):
        super()._adjust_lr(optimizer, epoch, config)
        adapter_lr = next(
            group["lr"] for group in optimizer.param_groups
            if group["parameter_group"] == "adapters"
        )
        for group in optimizer.param_groups:
            if group["parameter_group"] == "gate":
                group["lr"] = 10 * adapter_lr

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
            num_latents=self.config.num_latents,  # 传入每个变量的 channel token 数量。
            latent_dim=self.config.latent_dim,  # 传入 latent embedding 维度。
            adapter_num_heads=self.config.adapter_num_heads,  # 传入 attention head 数量。
            tp_bottleneck_dim=self.config.tp_bottleneck_dim,
            channel_depth=self.config.channel_depth,
            ablation_mode=self.config.ablation_mode,
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

    # Existing summary dataset identification is intentionally disabled.
    # @staticmethod
    # def _dataset_name(train_valid_data, covariates):
    #     identity_data = train_valid_data
    #     exog_data = (covariates or {}).get("exog")
    #     if exog_data is not None:
    #         identity_data = pd.concat([identity_data, exog_data], axis=1)
    #     data_pool = DataPool().get_pool()
    #     dataset = getattr(data_pool, "_global_dataset", None)
    #     data_dict = (
    #         dataset.get_state()["data_dict"] if dataset is not None else {}
    #     )
    #     if len(data_dict) == 1:
    #         return Path(next(iter(data_dict))).stem
    #     matches = []
    #     for name, data in data_dict.items():
    #         if len(data) < len(identity_data):
    #             continue
    #         if not set(identity_data.columns).issubset(data.columns):
    #             continue
    #         candidate = data.iloc[:len(identity_data)].loc[
    #             :, identity_data.columns
    #         ]
    #         if candidate.equals(identity_data):
    #             matches.append(name)
    #     if len(matches) == 1:
    #         return Path(matches[0]).stem
    #     return "unknown_dataset"

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv=1.0,
        **kwargs,
    ):
        self._series_dim = train_valid_data.shape[-1]
        # Existing summary/diagnostic output is intentionally disabled.
        # self._validation_epoch = 0
        # self._best_validation_mse = float("inf")
        # self._best_relative_biases = None
        # dataset_name = self._dataset_name(train_valid_data, covariates)
        # self._summary_dir = (
        #     Path("result")
        #     / "Temp"
        #     / "summary"
        #     / dataset_name
        #     / (
        #         f"seqlen{self.config.seq_len}_lr{format(self.config.lr, '.12g')}_"
        #         f"runid{self._run_id}"
        #     )
        # )
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
        # self._save_best_relative_biases()
        return result

    # Existing validation diagnostics and summary CSV writers are
    # intentionally disabled and retained as comments.
    # @staticmethod
    # def _scalar(value):
    #     if torch.is_tensor(value):
    #         return value.detach().float().mean().cpu().item()
    #     return float(value)
    #
    # def _save_validation_statistics(self, row):
    #     if self._summary_dir is None:
    #         return
    #     self._summary_dir.mkdir(parents=True, exist_ok=True)
    #     path = self._summary_dir / (
    #         f"horizon{self.config.horizon}_stage3_internal_statistics.csv"
    #     )
    #     pd.DataFrame([row]).to_csv(
    #         path, mode="a", header=not path.exists(), index=False,
    #         float_format="%.8g"
    #     )
    #
    # def _capture_best_relative_biases(self, validation_mse):
    #     if validation_mse >= self._best_validation_mse:
    #         return
    #     self._best_validation_mse = validation_mse
    #     self._best_relative_biases = tuple(
    #         (
    #             adapter.temporal_attention.relative_bias.detach().cpu().clone(),
    #             adapter.periodic_attention.relative_bias.detach().cpu().clone(),
    #         )
    #         for adapter in self._core_model().temporal_periodic_adapters
    #     )
    #
    # def _save_best_relative_biases(self):
    #     if self._summary_dir is None or self._best_relative_biases is None:
    #         return
    #     self._summary_dir.mkdir(parents=True, exist_ok=True)
    #     temporal_offsets = range(
    #         -self._core_model().num_patch_input + 1,
    #         self._core_model().num_patch_input,
    #     )
    #     for stage, (temporal, periodic) in zip(
    #         ("early", "middle", "deep"), self._best_relative_biases
    #     ):
    #         pd.DataFrame(
    #             temporal.numpy(),
    #             columns=[str(offset) for offset in temporal_offsets],
    #         ).to_csv(
    #             self._summary_dir / f"temporal_relative_bias_{stage}.csv",
    #             index=False,
    #             float_format="%.8g",
    #         )
    #         pd.DataFrame(
    #             periodic.numpy(), columns=[str(offset) for offset in range(14)]
    #         ).to_csv(
    #             self._summary_dir / f"periodic_relative_bias_{stage}.csv",
    #             index=False,
    #             float_format="%.8g",
    #         )

    def validate(self, valid_data_loader, series_dim, criterion):
        config = self.config
        valid_data_loader = DataLoader(
            valid_data_loader.dataset,
            batch_size=valid_data_loader.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        self.model.eval()
        device = get_device()
        final_square_sum = 0.0
        value_count = 0
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target = input.to(device), target.to(device)
                output = self.model(
                    input,
                    fp64=config.fp64,
                    use_variable_chunk=config.use_variable_chunk,
                )
                target = target[:, -config.horizon:, :series_dim]
                output = output[:, -config.horizon:, :series_dim]
                output, target = self._post_process(output, target)
                final_error = (output - target).double()
                final_square_sum += final_error.square().sum().item()
                value_count += final_error.numel()
        val_mse_final = final_square_sum / value_count
        self.model.train()
        return val_mse_final

        # Existing validation diagnostics are intentionally disabled and kept
        # below for reference. Early stopping above still monitors the exact
        # full-validation MSE of the mode's final prediction.
        # self._validation_epoch += 1
        # square_sums = {
        #     "channel": 0.0,
        #     "tp": 0.0,
        #     "fused": 0.0,
        #     "disagreement": 0.0,
        # }
        # error_dot = 0.0
        # channel_error_square = 0.0
        # tp_error_square = 0.0
        # diagnostics = None
        # for batch_index, (
        #     input, target, input_mark, target_mark
        # ) in enumerate(valid_data_loader):
        #     input, target = input.to(device), target.to(device)
        #     result = self.model(
        #         input,
        #         fp64=config.fp64,
        #         use_variable_chunk=config.use_variable_chunk,
        #         return_branches=True,
        #         return_diagnostics=batch_index == 0,
        #     )
        #     if batch_index == 0:
        #         diagnostics = {
        #             key: self._scalar(value)
        #             for key, value in result["diagnostics"].items()
        #         }
        #     target = target[:, -config.horizon:, :series_dim]
        #     predictions = {
        #         name: result[name][:, -config.horizon:, :series_dim]
        #         for name in ("channel", "tp", "output")
        #     }
        #     channel, target_processed = self._post_process(
        #         predictions["channel"], target
        #     )
        #     tp, _ = self._post_process(predictions["tp"], target)
        #     fused, _ = self._post_process(predictions["output"], target)
        #     channel_error = (channel - target_processed).double()
        #     tp_error = (tp - target_processed).double()
        #     fused_error = (fused - target_processed).double()
        #     square_sums["channel"] += channel_error.square().sum().item()
        #     square_sums["tp"] += tp_error.square().sum().item()
        #     square_sums["fused"] += fused_error.square().sum().item()
        #     disagreement = (
        #         result["channel_normalized"][:, :, :series_dim]
        #         - result["tp_normalized"][:, :, :series_dim]
        #     ).double()
        #     square_sums["disagreement"] += (
        #         disagreement.square().sum().item()
        #     )
        #     value_count += fused_error.numel()
        #     error_dot += (channel_error * tp_error).sum().item()
        #     channel_error_square += channel_error.square().sum().item()
        #     tp_error_square += tp_error.square().sum().item()
        # val_mse_channel = square_sums["channel"] / value_count
        # val_mse_tp = square_sums["tp"] / value_count
        # val_mse_fused = square_sums["fused"] / value_count
        # branch_error_cosine = error_dot / (
        #     np.sqrt(channel_error_square * tp_error_square) + 1e-12
        # )
        # row = {
        #     "epoch": self._validation_epoch,
        #     "val_mse_channel": val_mse_channel,
        #     "val_mse_tp": val_mse_tp,
        #     "val_mse_fused": val_mse_fused,
        #     "fusion_gate_g": self._scalar(
        #         torch.sigmoid(self._core_model().fusion_logit)
        #     ),
        #     "prediction_disagreement_rms": np.sqrt(
        #         square_sums["disagreement"] / value_count
        #     ),
        #     "branch_error_cosine": branch_error_cosine,
        #     "fusion_gain_over_best_branch": (
        #         min(val_mse_channel, val_mse_tp) - val_mse_fused
        #     ),
        #     **(diagnostics or {}),
        # }
        # self._save_validation_statistics(row)
        # self._capture_best_relative_biases(val_mse_fused)

    def _backward_variable_chunks(
        self, input, target, series_dim, criterion, scaler=None
    ):
        model = self._core_model()
        context = model.prepare_variable_chunk_context(
            input, fp64=self.config.fp64
        )
        correction = context["correction"]
        correction_leaf = (
            correction.detach().requires_grad_(True)
            if correction is not None else None
        )
        dual_branch = model.use_channel and model.use_tp
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
                continue

            result = model.forward_variable_chunk(
                context,
                start,
                end,
                correction=correction_leaf,
                return_branches=dual_branch,
            )
            chunk_target = target[
                :, -self.config.horizon:, start:supervised_end
            ]
            if dual_branch:
                channel = result["channel"][
                    :, -self.config.horizon:, :supervised_width
                ]
                tp = result["tp"][
                    :, -self.config.horizon:, :supervised_width
                ]
                channel, processed_target = self._post_process(
                    channel, chunk_target
                )
                tp, _ = self._post_process(
                    tp, chunk_target
                )
                gate = torch.sigmoid(model.fusion_logit)
                gate_prediction = (
                    gate * channel.detach() + (1 - gate) * tp.detach()
                )
                weighted_loss = (
                    criterion(channel, processed_target)
                    + criterion(tp, processed_target)
                    + criterion(gate_prediction, processed_target)
                ) * (
                    supervised_width / series_dim
                )
            else:
                prediction = result[
                    :, -self.config.horizon:, :supervised_width
                ]
                prediction, processed_target = self._post_process(
                    prediction, chunk_target
                )
                weighted_loss = criterion(
                    prediction, processed_target
                ) * (supervised_width / series_dim)
            if scaler is None:
                weighted_loss.backward()
            else:
                scaler.scale(weighted_loss).backward()

        # 各块已把梯度累积到小 correction leaf；这里只回传一次完整 Global 图。
        if (
            correction_leaf is not None
            and correction_leaf.grad is not None
            and correction.requires_grad
        ):
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
                shuffle=False,
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

    # Existing channel-statistics collection and summary CSV output are
    # intentionally disabled. With these overrides commented, forecast and
    # batch_forecast use the unchanged TFB base implementations.
    # def _reset_channel_statistics(self, horizon):
    #     shape = (self.config.channel_depth, 11)
    #     self._channel_statistics_horizon = horizon
    #     self._channel_statistics_sum = torch.zeros(shape, dtype=torch.float64)
    #     self._channel_statistics_count = torch.zeros(
    #         shape, dtype=torch.float64
    #     )
    #
    # def _accumulate_channel_statistics(self, statistics):
    #     statistics = statistics.detach().cpu().to(torch.float64).reshape(
    #         -1, self.config.channel_depth, 11
    #     )
    #     valid = torch.isfinite(statistics)
    #     self._channel_statistics_sum.add_(
    #         torch.where(valid, statistics, 0).sum(dim=0)
    #     )
    #     self._channel_statistics_count.add_(valid.sum(dim=0))
    #
    # def _channel_statistics_frame(self):
    #     count = self._channel_statistics_count
    #     values = self._channel_statistics_sum / count.clamp_min(1)
    #     values[count == 0] = float("nan")
    #     return pd.DataFrame({
    #         "horizon": self._channel_statistics_horizon,
    #         "round": range(1, self.config.channel_depth + 1),
    #         "intra_channel_update_ratio": values[:, 0].numpy(),
    #         "intra_patch_update_ratio": values[:, 1].numpy(),
    #         "inter_channel_update_ratio": values[:, 2].numpy(),
    #         "channel_pair_cosine_before_inter": values[:, 3].numpy(),
    #         "channel_pair_cosine": values[:, 4].numpy(),
    #         "within_variable_patch_cosine_before_intra": values[:, 5].numpy(),
    #         "within_variable_patch_cosine": values[:, 6].numpy(),
    #         "between_variable_patch_cosine_before_intra": values[:, 7].numpy(),
    #         "between_variable_patch_cosine": values[:, 8].numpy(),
    #         "correction_rms": values[:, 9].numpy(),
    #         "correction_to_patch_rms_ratio": values[:, 10].numpy(),
    #     })
    #
    # def _save_channel_statistics(self):
    #     if self._summary_dir is None or self._channel_statistics_sum is None:
    #         return
    #     self._summary_dir.mkdir(parents=True, exist_ok=True)
    #     frame = self._channel_statistics_frame()
    #     prefix = f"horizon{self._channel_statistics_horizon}_channel_module"
    #     frame.to_csv(
    #         self._summary_dir / f"{prefix}.csv", index=False,
    #         float_format="%.8g"
    #     )
    #
    # def forecast(self, horizon, series, *, covariates=None):
    #     self._reset_channel_statistics(horizon)
    #     self._collect_channel_statistics = True
    #     try:
    #         result = super().forecast(
    #             horizon, series, covariates=covariates
    #         )
    #     finally:
    #         self._collect_channel_statistics = False
    #     self._save_channel_statistics()
    #     return result
    #
    # def batch_forecast(self, horizon, batch_maker, **kwargs):
    #     new_collection = (
    #         not self._channel_statistics_batch_active
    #         or self._channel_statistics_horizon != horizon
    #     )
    #     if new_collection:
    #         self._reset_channel_statistics(horizon)
    #         self._channel_statistics_batch_active = True
    #     self._collect_channel_statistics = True
    #     try:
    #         result = super().batch_forecast(horizon, batch_maker, **kwargs)
    #     finally:
    #         self._collect_channel_statistics = False
    #     has_more_batches = getattr(batch_maker, "has_more_batches", None)
    #     if has_more_batches is None or not has_more_batches():
    #         self._save_channel_statistics()
    #         self._channel_statistics_batch_active = False
    #     return result

    # 执行一次 forward，并按 TFB 约定封装预测结果。
    # shape 约定：B 为 batch size，S 为 seq_len，H 为 horizon，V 为变量数，M 为时间特征数。
    # 输入 shape：input [B, S, V]，target [B, label_len+H, V]。
    # mark shape：input_mark [B, S, M]，target_mark [B, label_len+H, M]。
    def _process(self, input, target, input_mark, target_mark):
        core_model = self._core_model()
        separate_losses = (
            core_model.use_channel
            and core_model.use_tp
            and self.model.training
            and torch.is_grad_enabled()
        )
        result = self.model(
            input,
            fp64=self.config.fp64,
            use_variable_chunk=self.config.use_variable_chunk,
            # return_statistics=self._collect_channel_statistics,
            return_branches=separate_losses,
        )
        # Existing channel statistics are intentionally disabled.
        # if self._collect_channel_statistics:
        #     output, statistics = result
        #     self._accumulate_channel_statistics(statistics)
        #     return {"output": output}
        if not separate_losses:
            return {"output": result}
        target = target[
            :, -self.config.horizon:, :self._series_dim
        ]
        channel, processed_target = self._post_process(
            result["channel"][:, :, :self._series_dim], target
        )
        tp, _ = self._post_process(
            result["tp"][:, :, :self._series_dim], target
        )
        gate = torch.sigmoid(core_model.fusion_logit)
        gate_prediction = gate * channel.detach() + (1 - gate) * tp.detach()
        tp_loss = torch.mean((tp - processed_target) ** 2)
        gate_loss = torch.mean((gate_prediction - processed_target) ** 2)
        return {
            "output": result["channel"],
            "additional_loss": tp_loss + gate_loss,
        }
