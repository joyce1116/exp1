# 导入反射工具以检查 Resize 的函数签名。
import inspect

# 导入 pandas 以解析时间频率。
import pandas as pd
# 导入 torchvision 的图像缩放变换。
from torchvision.transforms import Resize


# 根据当前 torchvision 版本的接口安全创建 Resize。
# size 为目标 (H, W)，H/W 分别是高/宽；返回的变换将 [..., H_in, W_in] 缩放为 [..., H, W]。
def safe_resize(size, interpolation):
    # 获取 Resize 构造函数的完整签名。
    signature = inspect.signature(Resize)
    # 取出签名中的参数映射。
    params = signature.parameters
    # 检查当前版本是否支持 antialias 参数。
    if 'antialias' in params:
        # 在新版本中显式关闭抗锯齿以保持旧行为。
        return Resize(size, interpolation, antialias=False)
    else:
        # 在旧版本中仅传入尺寸和 interpolation 方式。
        return Resize(size, interpolation)


# 定义各 pandas 基础频率可能对应的季节周期长度。
POSSIBLE_SEASONALITIES = {
    # 秒频率对应一小时周期。
    "S": [3600],
    # 分钟频率对应一天或一周周期。
    "T": [1440, 10080],
    # 小时频率对应一天或一周周期。
    "H": [24, 168],
    # 天频率对应一周、一月或一年周期。
    "D": [7, 30, 365],
    # 周频率对应一年或一月周期。
    "W": [52, 4],
    # 月频率对应一年、半年或一季度周期。
    "M": [12, 6, 3],
    # 工作日频率对应一个工作周。
    "B": [5],
    # 季度频率对应一年或半年周期。
    "Q": [4, 2],
}


# 将 pandas 频率名规范化为季节性映射使用的键。
def norm_freq_str(freq_str: str) -> str:
    # 移除频率名中连字符后的锚定信息。
    base_freq = freq_str.split("-")[0]
    # 识别并移除复合频率名末尾的起始标记 S。
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        # 返回去掉末尾 S 的基础频率名。
        return base_freq[:-1]
    # 无需处理时直接返回基础频率名。
    return base_freq


# 将给定时间频率转换为兼容的季节周期列表。
# 实际返回 list；现有 ``-> int`` 是原代码保留的类型标注。
def freq_to_seasonality_list(freq: str, mapping_dict=None) -> int:
    # 未提供自定义映射时使用内置季节性配置。
    if mapping_dict is None:
        # 将映射表指向默认频率配置。
        mapping_dict = POSSIBLE_SEASONALITIES
    # 将频率字符串解析为 pandas offset 对象。
    offset = pd.tseries.frequencies.to_offset(freq)
    # 按规范化频率名查找候选的基础季节周期。
    base_seasonality_list = mapping_dict.get(norm_freq_str(offset.name), [])
    # 初始化用于收集换算后周期的列表。
    seasonality_list = []
    # 逐一换算每个候选基础周期。
    for base_seasonality in base_seasonality_list:
        # 用频率倍数除基础周期，同时获取余数。
        seasonality, remainder = divmod(base_seasonality, offset.n)
        # 仅保留能被当前频率倍数整除的周期。
        if not remainder:
            # 将有效的换算周期加入结果列表。
            seasonality_list.append(seasonality)
    # 始终加入 1，作为无季节性的基准周期。
    seasonality_list.append(1)
    # 返回所有适用于当前频率的季节周期。
    return seasonality_list
