# Ga-EPR

## 安装指南

1. 克隆仓库
```bash
git clone https://github.com/ubv3t2b4iktvbi/GA-EPR.git
cd Ga-Epr
pip install -r requirements.txt
```

## 运行方式

### 1. 使用默认配置运行
直接运行 main.py，将使用默认配置在 `results\sl` 目录下进行训练：
```bash
python epr/main.py
```

### 2. 指定配置目录运行
可以指定包含 config.yaml 和 problem.yaml 的目录进行运行：
```bash
python main.py /path/to/config/directory
```

### 3. 高维问题训练
对于高维问题，可以使用以下命令进行训练：
直接运行 high.py，将使用默认配置在 `results\transcription_factor` 目录下进行训练：
```bash
python epr/high.py
```