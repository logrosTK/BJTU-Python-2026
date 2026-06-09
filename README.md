# MNIST-C 噪声图像识别研究专题

本项目围绕 MNIST-C 数据集完成了“面向噪声数据的图像识别系统”研究性专题。我们不仅完成了指导书中的必做和选做任务，还补充了无监督学习、交叉验证和误分类可视化，用来更完整地分析模型在不同噪声干扰下的表现。

## 我们完成的工作

- 读取并整理本地 `mnist_c/` 数据集，覆盖 `identity` 和 15 种噪声/破坏类型。
- 选择 `identity`、`shot_noise`、`rotate` 作为必做训练数据集。
- 对 3 个训练数据集分别训练 3 个神经网络：
  - 单隐层 MLP
  - 双隐层 MLP
  - PCA + MLP
- 统计 `identity` 训练模型在 `shot_noise` 和 `rotate` 测试集上的跨噪声表现。
- 将 `identity + shot_noise + rotate` 合并训练一个模型，并在全部 16 个 MNIST-C 测试集上评估。
- 对数据预处理、损失函数、超参数、验证方法进行对比实验。
- 增加 `PCA + KMeans` 无监督基线，补充监督学习之外的对照结果。
- 增加 3 折交叉验证，降低单次训练/验证划分带来的偶然性。
- 绘制全部噪声类型样例图和误分类样例图，方便解释模型为什么在某些噪声下表现较差。

## 我们的优势和创新点

1. **实验覆盖完整**  
   必做部分、选做部分、跨噪声测试、全部数据集评估都已完成，结果集中保存在 `outputs/tables/` 中。

2. **对比维度更丰富**  
   除了 3 个神经网络外，还比较了预处理方式、损失函数、超参数、KMeans 无监督基线和交叉验证结果，报告内容比只训练模型更充分。

3. **重视鲁棒性分析**  
   我们没有只看 `identity` 上的准确率，而是重点比较模型在 `shot_noise`、`rotate` 以及全部 MNIST-C 噪声集上的表现，能更清楚地说明分布偏移对分类器的影响。

4. **结果解释更直观**  
   报告中加入了数据样例图和误分类图。例如选做模型在 `brightness` 上表现较差，误分类图可以看到模型出现明显预测偏置，这比单独列准确率更容易讲清楚。

5. **复现流程清晰**  
   `main.py` 可以一键重新运行实验，自动生成表格、图像和 Notebook 报告，便于展示前检查或重新调整参数。

## 主要文件

- `MNIST-C_研究性专题报告.ipynb`：提交和展示用 Notebook。
- `MNIST-C_研究性专题报告.html`：Notebook 导出的 HTML 版本，方便直接预览。
- `main.py`：完整实验脚本。
- `outputs/tables/`：实验结果表格。
- `outputs/figures/`：实验图表和可视化结果。
- `outputs/summary.json`：关键指标摘要。
- `requirements.txt`：运行依赖。

## 运行方式

默认运行：

```powershell
python .\main.py
```

复现实验报告中的正式参数：

```powershell
python .\main.py --train-limit 15000 --optional-train-limit 8000 --test-limit 0 --study-limit 8000 --kmeans-limit 8000 --cv-limit 6000 --max-iter 12 --study-max-iter 8 --loss-epochs 8
```

使用完整训练集运行：

```powershell
python .\main.py --train-limit 0 --optional-train-limit 0 --test-limit 0 --max-iter 20
```

## 当前关键结果

- 必做同分布最佳模型：`identity + mlp_2hidden`，测试准确率约 `0.9486`。
- 选做组合模型在全部 16 个测试集上的平均准确率约 `0.4468`。
- KMeans 无监督基线最佳测试准确率约 `0.3717`。
- 3 折交叉验证最佳模型为 `mlp_2hidden`，平均验证准确率约 `0.9112`。

