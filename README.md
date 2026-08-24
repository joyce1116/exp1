# 环境安装与运行

## 1. 创建 Conda 环境

实验使用 Python 3.11.15：

```bash
conda create -n xxx(actual name) python=3.11.15 -y
conda activate xxx
```

## 2. 安装依赖

原始实验环境存在部分依赖版本冲突。请关闭依赖自动解析，并严格按照 `requirements.txt` 中记录的版本安装：

```bash
python -m pip install --no-deps -r requirements.txt
```

也可以使用清华 PyPI 镜像：

```bash
python -m pip install --no-deps -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

以上命令需在项目根目录执行。

## 3. 配置模型权重路径

运行前，需要修改 `scripts/multivariate_forecast/temp/final/all.sh` 中每条任务的 `checkpoint_path`。将路径开头的项目根目录 `/home/vision_cxy` 替换为实际的项目根目录；后面的相对路径 `pretrained_weights/mae/mae_visualize_vit_base.pth` 保持不变。

## 4. 运行实验

进入项目根目录后执行：

```bash
cd /实际路径/exp
bash ./scripts/multivariate_forecast/temp/final/all.sh
```
