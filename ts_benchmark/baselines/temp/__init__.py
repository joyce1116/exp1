# 声明从该包公开导出的模型名称。
__all__ = ["VisionTS"]

# 导入 TFB 适配器，使调用方可以直接从包级路径访问 VisionTS。
from ts_benchmark.baselines.temp.temp import VisionTS
