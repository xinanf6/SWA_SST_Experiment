# C0训练闭环：第一阶段实现

本阶段提供Torch数据适配、原DSPG主干移植、共享热力编码和C0残差预测。CF、CH、CFH、梯度结构损失及系统算力benchmark属于下一阶段；本包当前只有C0训练入口。

## 数据及模型

默认配置为 `configs/training_c0.json`，读取已通过完整验收的 `output/processed_window_qc/`。训练3580、验证704、测试714，全部沿用同一缺测筛选索引。

数据适配直接包装 `PreparedSSTDataset`，不移动日期、不重新归一化、不展开保存滑动窗口。基础输入 `[B,14,7,H,W]`、热力输入 `[B,14,3,H,W]`、目标 `[B,3,1,H,W]`。Windows worker重新打开只读memmap；默认workers=0，可通过配置测试增加。

C0按需读取基础历史、热力历史、标签和起报SST。完整样本接口仍可读取锋面和显式热力增量。RTX 3060的两次短程对比各测16个完整网格批次，按需读取平均约0.91秒/批，旧读取约1.47秒/批。两个worker平均约1.35秒/批，计入启动时间。短程结果会随文件缓存变化，默认仍使用0个worker。

主干保留多尺度卷积与CBAM、学习位移和扩散模块、全局BiLSTM、像素ConvBiLSTM、全局局部门控融合与三提前期解码。原文件哈希及锁定参数保存在 `sst_training/models/dspg_provenance.json`，运行时不依赖原项目文件。可选的原代码一致性测试在本机原文件存在时执行；其他机器可用 `DSPG_REFERENCE_PATH` 指定相同版本，否则明确跳过该项。

热力编码为3→16→32两层卷积，加1×1投影后以初始0.1的可学习标量加入32通道空间特征。额外模块使用独立、随种子变化的初始化随机流，主干初始化不受额外模块消费随机数影响。

解码器输出无量纲残差r，物理残差R=σ_SST·r，C0输出T_t+R。残差不加SST均值，不加入显式H。内部动力模块仍使用标准化输入，输出学习的像素位移，不作为m/s物理流速。

## 运行环境与命令

已在现有PyTorch 1.12.0、CUDA与RTX 3060环境验证，使用torch.cuda.amp接口。无需为运行本阶段升级环境；独立新环境的额外训练依赖见 `requirements-training.txt`。预处理/分析不强制导入Torch。

先运行短程检查：

```powershell
python -m sst_training smoke --config configs/training_c0.json
```

默认只使用4个训练样本、2个验证样本、2轮；输出独立的 `output/training/C0_seed2026_smoke/`，状态为 `smoke_complete`，这些损失仅检查实现。可用 `--output-root` 指定独立目录、`--epochs`、`--train-samples`、`--validation-samples` 改变短程范围。

正式训练需要显式运行：

```powershell
python -m sst_training train --config configs/training_c0.json
python -m sst_training train --config configs/training_c0.json --resume
```

训练使用AdamW、lr=1e−3、weight_decay=1e−5、batch=2、梯度累积2、最多50轮、早停8轮、平台3轮后学习率减半、梯度裁剪1。最后不足整批的累积组按真实样本数加权。AMP溢出会降低缩放并记录跳过次数；没有任何有效更新的训练轮次会报错。

训练和验证默认各显示批次进度条，标明`Epoch 当前轮/配置上限 Train`或`Valid`，含已完成/总批次数、累计SST MSE（℃²）、累计辅助损失、当前学习率、已用时间、当前阶段预计剩余时间和批次速度。默认完整训练每轮1790个训练批次、352个验证批次。进度条约每秒刷新，避免频繁终端输出；每轮原有MSE汇总继续打印。

进度条写入stderr；ETA只针对当前训练或验证阶段，不包含后续阶段、检查点保存或全部剩余轮次。重定向日志时可关闭进度条：

```powershell
python -m sst_training train --config configs/training_c0.json --no-progress
```

已运行的Python进程要在保存`last.pt`后通过`--resume`重新启动才能加载进度显示。尚未保存检查点的首轮中断后需要重新开始。`sst_training/checkpoint_compatibility.json`记录已验证等价的进度显示和C0按需读取版本，供恢复时核对代码指纹。

温度主损失在℃空间按核心海洋面积权重计算，3个提前期等权；辅助损失比较学习位移与对应历史标准化uo/vo，权重0.1，保留原有启发式含义。验证以物理单位SST MSE选择best。

训练只读取训练/验证分区。每轮结束保存：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 数据身份、训练配置、模型结构、代码哈希、原DSPG来源、运行设备及软件版本 |
| `last.pt` | 当前模型、优化器、调度器、AMP、RNG、历史、最佳模型快照和更新计数；恢复仅使用此文件 |
| `best.pt` | 验证SST MSE最佳的模型权重及元数据，供推理；不是恢复优化器的入口 |
| `history.json` | 每轮训练/验证MSE、辅助损失、样本数、耗时及学习率 |
| `training_summary.json` | 完成状态、参数量、显存峰值、累计有效更新/AMP溢出跳过和历史 |

配置、数据身份、代码哈希或短程范围不匹配会拒绝恢复。`last.pt`内包含独立的最佳模型快照；若进程在两个文件发布之间中断，恢复时可据此修复`best.pt`。只恢复到已完成轮次边界，未完成轮次重新执行。CPU离线测试验证恢复与不中断训练逐位一致；CUDA的grid_sample反传不保证逐位确定性，仍须按固定种子控制并最终报告多种子结果。

## 预测与统一评价

使用正式训练的best检查点导出完整验证集：

```powershell
python -m sst_training predict --config configs/training_c0.json --checkpoint output/training/C0_seed2026/best.pt --split validation --output output/predictions/C0_seed2026_validation.nc
python -m sst_analysis evaluate --config configs/analysis.json --predictions output/predictions/C0_seed2026_validation.nc --split validation --output-root output/analysis/C0_seed2026_validation --model-name C0_seed2026
```

默认不覆盖已有预测文件；需要时显式传 `--overwrite`。预测文件包含全部质量筛选后起报日，维度和顺序符合既有评价器；保存predictions、residual、last_sst（℃），日期、经纬度、run_identity及训练类型。核心海洋域之外的陆地写为NaN，整个固定海洋域必须有限。临时文件完成后原子替换，失败时关闭并移除临时文件。

短程检查点默认拒绝用于预测；仅做通路调试时可显式添加 `--allow-smoke-checkpoint`。此类文件带有 `training_kind=smoke`，应存入带smoke标记的独立目录，不作为正式模型结果。完整验证分区导出可以检验文件契约，但不能把小样本训练后的分数用于论文结论。

## 8 通道并行时序 C0

紧凑版使用独立配置 `configs/training_c0_parallel8.json`，模型标识为 `C0_parallel8`。三组膨胀卷积将逐日基础输入编码为 8 通道，热力编码为 3→8→8，逐日位移头的中间宽度为 4。平流与扩散将历史日期合并到批次维并行计算。局部时序分支把 14 天的有序特征融合为 16 通道，再用两层分组空间卷积传播。所有局部空间特征和预测均保留 270×250 网格。

沿用 C0 残差输出 `T_t + R`、核心区面积加权损失和海流辅助监督。默认采用 batch=2 和梯度累积 2，训练与验证使用最新的 3580 和 704 个样本索引。

```powershell
python -m sst_training smoke --config configs/training_c0_parallel8.json
python -m sst_training train --config configs/training_c0_parallel8.json
```

短程运行写入 `output/training/C0_parallel8_seed2026_smoke/`。正式训练写入 `output/training/C0_parallel8_seed2026/`。恢复时使用同一配置加 `--resume`。配置和模型标识用于检查恢复和导出，原版 DSPG 的检查点须继续搭配 `configs/training_c0.json`。

完成正式训练后，使用对应检查点导出完整验证分区并沿用统一评价器。

```powershell
python -m sst_training predict --config configs/training_c0_parallel8.json --checkpoint output/training/C0_parallel8_seed2026/best.pt --split validation --output output/predictions/C0_parallel8_seed2026_validation.nc
python -m sst_analysis evaluate --config configs/analysis.json --predictions output/predictions/C0_parallel8_seed2026_validation.nc --split validation --output-root output/analysis/C0_parallel8_seed2026_validation
```

## 七变量原版 DSPG 对照

`configs/training_dspg7.json` 启用 `DSPG7`，保留原版 DSPG 的空间编码、动力启发模块、双流时序和解码器。只使用 SST、10 m 风、表层海流、2 m 气温和海平面气压共 7 个通道，直接预测三提前期 SST。计算损失和导出前，模型把预测反标准化为 ℃。NetCDF 中的 `residual` 为预测 SST 减起报 SST。

这是在当前西南大西洋数据和统一训练协议下重训原版结构。使用种子 2026、3580 个训练窗、704 个验证窗、270×250 网格、14 天历史和三个提前期。保留统一的核心区面积加权 SST MSE、海流辅助项及验证 SST MSE 选优规则。DSPG7 的 AMP 初始缩放为 512，以减少训练开始时的梯度溢出跳步。

```powershell
python -m sst_training train --config configs/training_dspg7.json
python -m sst_training predict --config configs/training_dspg7.json --checkpoint output/training/DSPG7_seed2026/best.pt --split validation --output output/predictions/DSPG7_seed2026_validation.nc
python -m sst_analysis evaluate --config configs/analysis.json --predictions output/predictions/DSPG7_seed2026_validation.nc --split validation --output-root output/analysis/DSPG7_seed2026_validation --model-name DSPG7_seed2026 --front-lines
```

训练结束后运行导出与评价。检查点写在 `output/training/DSPG7_seed2026/`，中断后可用原训练命令加 `--resume`，从最近保存的完整轮次继续。用于调试可先将 `train` 改成 `smoke`，短程检查点写入独立目录。

```powershell
python -m sst_analysis compare --config configs/analysis.json --evaluation-a output/analysis/C0_parallel8_seed2026_validation --evaluation-b output/analysis/DSPG7_seed2026_validation --output-root output/analysis/C0_parallel8_vs_DSPG7_validation
```

比较按相同起报日配对，差值方向是 `C0_parallel8 − DSPG7`。逐提前期查看温度 RMSE、强锋区 RMSE、梯度指标和置信区间，锋线 F1 和位置误差可在两边的 `metrics_summary.csv` 中对照。两个模型的主干、输入和输出形式均有差异，因此这一组比较用于说明整体性能和开销差异。

## 离线验收

```powershell
python -B -m pytest tests -q
```

新增测试覆盖：℃残差单位、完整分支梯度、核心掩膜与辅助项手算、不足整批的累积一致性、CPU断点恢复一致性、最佳检查点恢复、原DSPG逐值一致性、Windows worker重开映射、已训练预测与统一评价的完整闭环，以及配置和数据身份隔离。

没有安装Torch时，模型相关测试明确跳过；数据配置等不依赖Torch的测试照常执行。Windows沙箱可能禁止创建DataLoader命名管道，这种情况下workers=0仍可运行；worker测试需要在允许命名管道的本机终端执行。
