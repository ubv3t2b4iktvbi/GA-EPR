# Ga-EPR: 

## 安装指南
1. 克隆仓库
   ```bash
   git clone https://github.com/ubv3t2b4iktvbi/GA-EPR.git
   cd Ga-Epr
   pip install -r requirements.txt


## 运行方式
1. 使用默认配置运行
直接运行main.py，将使用默认配置在results\sl目录下进行训练：

bash
python epr/main.py
2. 指定配置目录运行
可以指定包含config.yaml和problem.yaml的目录进行运行：

bash
python main.py /path/to/config/directory

3. 对于高维问题，可以使用以下命令进行训练：
直接运行high.py，将使用默认配置在results\sl目录下进行训练：

bash
python epr/high.py