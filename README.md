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

## Naive Implementation of Transformer
### Download the dataset file and process

```bash
# 下载并处理成需要的binary文件
python3 naive_attention/data/get_dataset.py
```

### Run the training script

```bash
python3 naive_attention/train.py
```

## Optimization
### Flash Attention
对于 Flash Attention 背后的数学原理在 [FlashAttentionDesc](./flash_attention/FlashAttentionDesc.md) 这篇文章里面可以看到。之后有一个使用 CUDA 实现的简单代码，
没有在 SGEMM 矩阵乘法方面做优化，因此效率会比预期低很多。

项目已经在 `pyproject.toml` 中配置好了包路径要求，但是编译环境依赖于 PyTorch Extension，因此必须使用 uv 同步环境，安装好 torch 依赖项。
这里对实现好的 CUDA 代码提供了一种快速的 `JIT` 编译方法，运行方法如下：

```bash
# 运行脚本的时候会自动根据 __init__.py 中的要求编译源代码
python -m test.test_flash_attention
# 使用结束之后可以手动清除 JIT 缓存
rm -rf ~/.cache/torch_extensions/py3XX_cu1XX/flash_attention_cuda/
```

第一次运行的时候会花很多时间进行编译 Flash Attention，之后就会写进缓存中不再需要编译。

### KV Cache
对于推理阶段，由于上下文过长带来的计算难度指数级上升的问题，KV Cache通过「空间换时间」的做法，通过保存住计算过程中的 K、V 中间量，
能够用线性增长的空间换取指数级下降。一般的 KV Cache 技术可以在 PyTorch 层面进行编写，因此根据 KV Cache 的思想实现了一个简易版本。运行方法如下：

```bash
# 运行脚本的时候会自动根据 __init__.py 中的要求编译源代码
python -m test.test_kv_cache
```

### Grouped-Query Attention
是一种对于 Multi-head Attention 的优化形式，本质上是通过牺牲一些参数换取更小的内存占用和训练速度。
GQA 让所有的 Head 之间共享同样的一份 K 和 V 矩阵（意味K和V的计算唯一），只让 Q 保留了原始多头的性质，从而大大减少 K 和 V 矩阵的参数量以及 
KV Cache的显存占用，以此来达到提升推理速度，但是会带来精度上的损失。

```bash
python -m test.test_gqa
```