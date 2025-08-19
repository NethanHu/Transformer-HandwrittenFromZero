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

## Optimization
### Flash Attention
对于 Flash Attention 背后的数学原理在 [FlashAttentionDesc](./flash_attention/FlashAttentionDesc.md) 这篇文章里面可以看到。之后有一个使用 CUDA 实现的简单代码，
没有在 SGEMM 矩阵乘法方面做优化，因此效率会比预期低很多。

项目已经在 `pyproject.toml` 中配置好了包路径要求，但是编译环境依赖于 PyTorch Extension，因此必须使用 uv 同步环境，安装好 torch 依赖项。
这里对实现好的 CUDA 代码提供了一种快速的 `JIT` 编译方法，运行方法如下：

```bash
# 运行脚本的时候会自动根据 __init__.py 中的要求编译源代码
python3 flash_attention/simple_test.py
```

第一次运行的时候会花很多时间进行编译 Flash Attention，之后就会写进缓存中不再需要编译。