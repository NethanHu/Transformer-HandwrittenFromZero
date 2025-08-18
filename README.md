## What is this project?
为了更好理解 Transformer 内部结构和优化策略，本项目从头使用 PyTorch 开始构建 Transformer，以便对于位置编码、多头注意力、掩码等操作有更深的理解。

## How to start?
### Construct the virtual environment
环境使用 `uv` 进行搭建，并且已经完全配置好了所有需要的依赖。具体构建方式如下：

```bash
# 同步所有的依赖项并且构建环境
uv sync
# 激活构建好的虚拟环境
source .venv/bin/activate
```