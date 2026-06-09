# MNIST-C 研究性专题

本目录已经完成指导书中的必做部分和选做部分，核心交付文件为：

- `MNIST-C_研究性专题报告.ipynb`：Jupyter Notebook 研究报告，可用于提交和现场展示。
- `main.py`：完整实验脚本，包含数据读取、预处理对比、3 个神经网络训练、损失函数对比、参数搜索、选做组合训练和报告生成。
- `outputs/tables/`：实验结果 CSV。
- `outputs/figures/`：报告图表。
- `outputs/summary.json`：关键结果摘要。

## 运行方式

默认使用分层抽样训练，测试集使用完整 10000 张：

```powershell
python .\main.py
```

如需复现实验报告中的正式结果：

```powershell
python .\main.py --train-limit 15000 --optional-train-limit 8000 --test-limit 0 --study-limit 8000 --max-iter 12 --study-max-iter 8 --loss-epochs 8
```

如需使用完整训练集，可把训练样本限制设为 `0`：

```powershell
python .\main.py --train-limit 0 --optional-train-limit 0 --test-limit 0 --max-iter 20
```

## 已选择的数据集

必做训练数据集：

- `identity`
- `shot_noise`
- `rotate`

选做部分使用以上 3 个训练集合并训练一个神经网络，并在本地全部 16 个 MNIST-C 测试集上评估。
