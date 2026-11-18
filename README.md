# SWA_SST_Experiment

西南大西洋海表温度预测实验的独立代码仓库。采用最初方案：14 天历史、未来 3 天标签、环境场滞后 1 天、有效混合层厚度下限 3 m、净表面热量直接驱动。本仓库从独立历史开始，仅包含本研究新实现的代码；预处理入口为 `python -m sst_preprocessing`，配置为 `configs/preprocess.json`。

预处理读取本地 OSTIA、CMEMS 和 ERA5 NetCDF，不运行下载脚本，不修改原始数据。源码同步目标为私有仓库 [xinanf6/SWA_SST_Experiment](https://github.com/xinanf6/SWA_SST_Experiment)；原始数据、预处理产物、图片、日志、缓存及凭据不进入提交。

**运行环境**

已验证环境为 Python 3.8.20、NumPy 1.24.4、pandas 1.5.3、xarray 0.20.1、netCDF4 1.5.8、SciPy 1.10.1、pytest 7.4.4。现有环境已具备依赖，无需为运行本流程升级。运行依赖见 `requirements.txt`，离线测试依赖入口为 `requirements-dev.txt`；仅在新环境缺少依赖时执行：

```powershell
python -m pip install -r requirements-dev.txt
```

本机约 15.73 GiB RAM，流程按月读取、按 `chunk_days` 分块插值，再写逐月物理场和每日内存映射数组，不把 14 年全量数据或展开后的滑动窗口装入内存。

**配置与命令**

在仓库根目录运行。配置中的 `data_root`、`output_root` 相对于配置文件目录解析，默认对应仓库的 `data/` 和 `output/processed_window_qc/`；`--output-root` 可覆盖输出位置。

```powershell
python -m sst_preprocessing inspect --config configs/preprocess.json
python -m sst_preprocessing smoke --config configs/preprocess.json
python -m sst_preprocessing run --config configs/preprocess.json
python -m sst_preprocessing verify --config configs/preprocess.json
```

- `inspect`：核对文件、变量、单位、日期、坐标及完整覆盖，计算输入文件哈希，打印所选文件数量、排除项及清单指纹。重复下载测试文件 `current_202301.nc` 不进入正式清单；意外重复日期会报错。
- `smoke`：默认写到独立的 `output/smoke/`。检查完整输入并拟合训练期 MLD 气候态后，仅处理配置覆盖的首个月物理场，输出 `smoke_report.json`，不生成 `READY.json`。它用于检查真实数据的物理处理阶段，不代表完整训练数据已就绪。
- `run`：处理所有月份，再拟合固定掩膜和训练统计、记录逐日质量、归一化、排除缺测窗口并深度验证，通过后发布 `READY.json` 和质量报告。
- `verify`：重新核对已有完整产物，包括标准化往返、热力公式、掩膜、日期和样本索引。可附加 `--quick` 跳过物理场逐月往返等深度检查；完整验收使用默认深度验证。

中断后使用同一配置和输出目录续跑：

```powershell
python -m sst_preprocessing run --config configs/preprocess.json --resume
```

续跑会重新核对配置、输入清单和产物模式版本，校验已完成月文件；归一化按已刷盘月份续写。月文件、统计文件和最终数组完成后原子替换。若输入或配置改变，应换用新的输出目录；不匹配的旧产物会被拒绝。已有输出目录再次运行时须明确使用 `--resume`。

**物理处理与数据边界**

实际 OSTIA 经纬度为目标网格，不重建坐标。默认输入尺寸 `270×250`，核心区为 `35°—48°S、48°—60°W`，对应 `260×240`；额外网格作为缓冲。CMEMS 和 ERA5 使用双线性插值，分别保留有限源数据的有效权重。主海洋掩膜要求完整有效插值模板，默认有效权重阈值 `0.999999`，不跨陆地缺测补成海洋值。

SST 和气温由 K 转为 ℃；SST 梯度用球面距离计算，输出东西向、南北向及模长，单位为 `℃/km`。中心差分要求中心与东、西、南、北邻点均有效，最外层和无完整模板的格点不作为梯度有效点。

MLD 月气候态仅由训练期的源海洋格点拟合。非正值及非有限值按局地月均值、该月海洋中位数依次回填并保留标记；陆地不填。仅当整个月气候态缺失时允许年气候态回退，供短期离线合成数据使用。所有有效正 MLD 保留原值，包括大于 500 m 的深度。

ERA5 四项热量 `ssr、str、slhf、sshf` 已是逐日累积能量，单位 `J/m²`，采用向海洋为正的原始符号直接相加：

\[
E_{\mathrm{net}}=ssr+str+slhf+sshf,\qquad
Q_{\mathrm{net}}=E_{\mathrm{net}}/86400
\]

\[
h_{\mathrm{eff}}=\max(h,3\,\mathrm m),\qquad
\Delta T_{\mathrm{heat}}=\frac{E_{\mathrm{net}}}{1025\times3990\times h_{\mathrm{eff}}}
\]

`Q_net` 的单位是 `W/m²`，日增温的单位是 `℃/day`；能量不重复聚合，也不重复乘除 86400。原协议不加入短波穿透或提前期衰减，热力先验为 `H_k = k × ΔT_heat,t−1`。例如 `100 W/m²、20 m` 对应约 `0.10563 ℃/day`。

训练期为 2011—2020 年，验证期为 2021—2022 年，测试期为 2023—2024 年。各通道均值、总体标准差、MLD 气候态、固定海洋掩膜和锋面第 90 百分位阈值均仅由训练数据确定。通道统计使用固定有效海洋格点；锋面阈值使用核心区内的有效梯度格点。验证与测试期不参与拟合。

物理日数组保持原始 UTC 日期含义，环境滞后只通过样本索引应用。对预测起点 `t`，SST 和锋面历史为 `t−13…t`，其他基础环境通道及热力历史为 `t−14…t−1`，目标为 `t+1…t+3`。全部日期支持必须留在同一分区，不跨训练、验证和测试边界借用历史。

完整 2011—2024 年覆盖预期为 **5114 个日日期**。无缺日且质控通过时，训练、验证、测试窗口数预期为 **3636／713／714**；这些是协议预期值，实际全量结果以生成的质量报告为准。

源资料的偶发缺测保持原样，不用未来观测或空间插补替代。逐日检查固定海洋／梯度掩膜内的数据有效性，再分别核对样本的 SST／锋面历史、滞后环境历史及 SST 标签；任何必需输入或标签不完整的窗口都会从所有模型共用的索引排除，并记录日期与原因。归一化数组里的缺测位置只是零占位，不会作为完整样本读取。环境变量仅在未来标签日期缺测时，不额外排除该窗口，因为标签只使用 SST。

**输出约定**

以下路径相对于配置中的 `output_root`：

| 路径 | 内容 |
| --- | --- |
| `manifest.json` | 实际输入清单、哈希、覆盖情况和排除记录 |
| `run.json` | 配置快照、模式版本和运行身份 |
| `mld_climatology.npz`、`mld_climatology.json` | 训练期源海洋 MLD 气候态和说明 |
| `physical/YYYY-MM.nc` | 逐月物理单位场，保留四项热量、海洋与大气最小插值权重、`mld_was_filled` 质量标记 |
| `grid_masks.npz` | 经纬度、余弦纬度面积权重、`core_mask`、`ocean_mask`、`gradient_mask` |
| `stats.json` | 各通道训练期均值／标准差、单位、锋面 P90 阈值 |
| `normalized/base.npy` | `float32 [日期, 7, 纬度, 经度]` 日数组 |
| `normalized/front.npy` | `float32 [日期, 3, 纬度, 经度]` 日数组 |
| `normalized/thermal.npy` | `float32 [日期, 3, 纬度, 经度]` 日数组 |
| `normalization.json` | 已完成归一化月份及断点身份 |
| `dates.npy`、`sample_index.csv` | 日日期及带分区、历史和目标位置的样本索引 |
| `daily_quality.csv`、`daily_quality.json` | 逐日 SST、锋面及环境完整性、缺测数量和日期 |
| `excluded_samples.csv` | 因必需数据缺测被排除的窗口及明确原因 |
| `quality_report.json`、`quality_report.md` | 完整流程通过深度验证后的质量报告 |
| `READY.json` | 完整流程已通过验证、可供数据集读取的标志 |

13 个输入通道顺序固定：`base=(sst,u10,v10,uo,vo,t2m,msl)`，`front=(gradient_x,gradient_y,gradient_magnitude)`，`thermal=(net_heat_flux,mixed_layer_depth,heat_increment)`。物理文件保留 NaN；归一化数组中的无效格点填 0，必须配合固定掩膜使用。核心区损失和评价可使用 `core_mask & ocean_mask`，梯度项再与 `gradient_mask` 相交。

**供后续训练读取**

`PreparedSSTDataset` 仅依赖 NumPy/pandas，后续可包装成 PyTorch Dataset。它要求存在 `READY.json`，读取每日 memmap 并动态提取窗口，不预先展开存储 14 天样本。

```python
from pathlib import Path
from sst_preprocessing.dataset import PreparedSSTDataset

with PreparedSSTDataset(Path("output/processed_window_qc"), split="train") as dataset:
    sample = dataset[0]
    print(len(dataset), sample["origin_date"])
    print(sample["base"].shape)        # (14, 7, 270, 250)
    print(sample["front"].shape)       # (14, 3, 270, 250)
    print(sample["thermal"].shape)     # (14, 3, 270, 250)
    print(sample["target"].shape)      # (3, 1, 270, 250)
    print(sample["heat_increment"].shape)  # (3, 1, 270, 250)
    loss_mask = dataset.metadata["core_mask"] & dataset.metadata["ocean_mask"]
```

`base/front/thermal/target` 是训练统计标准化后的值；`last_sst` 为物理单位 ℃，形状 `[1,H,W]`；`heat_increment` 为物理单位 ℃ 的三提前期先验 `[ΔT,2ΔT,3ΔT]`，所用强迫日期为 `t−1`。无效格点在返回的物理量中也置 0。样本另含 `target_dates`。

使用 `with` 或显式调用 `close()` 释放映射；Windows 数据加载 worker 序列化时不携带完整数组，worker 启动后仅重新打开只读 memmap。后续构建 DataLoader 的入口仍应放在 `if __name__ == "__main__":` 内。

**离线验证**

```powershell
python -B -m pytest tests -q
```

`pytest.ini` 将默认发现目录限定为 `tests`，避免导入根目录会联网或下载数据的旧测试脚本。离线测试覆盖跨年／闰日及分区隔离、真实尺寸核心区、海岸插值、常量与线性场梯度、热量符号、MLD 回填、3 m 下限、标准化往返和训练统计隔离。

`tests/test_pipeline.py` 的 8 项合成数据端到端用例验证完整流水线与样本读取、断点续跑一致性、不同分块一致性、记录缺测并排除窗口，以及配置或输入改变时拒绝续跑。`tests/test_quality.py` 另验证不同来源时钟下的缺测排除边界。该验证与真实数据 `smoke` 的首月物理检查互为补充；全量处理完成情况由实际 `READY.json` 和质量报告确认。

**缺测窗口与区域诊断**

预处理模式版本2将固定海洋域与逐日缺测分开。固定域采用训练期至少一天同时具有完整SST、海流和MLD插值模板的格点；永久陆地仍被屏蔽。ERA5偶发缺测进入逐日质量记录，消费该日环境历史的窗口从所有模型共用索引排除，原始缺测值不填补。训练期统计只使用相应变量的有限值。

新产物目录为 `output/processed_window_qc/`，配置为 `configs/preprocess_window_qc.json`。旧的 `output/processed/` 保留用于追溯。掩膜、统计量、P90、归一化数组和样本索引必须使用同一批次，不能把新掩膜叠到旧数组上。

从已有完整物理场重建，先核对原始输入清单与全部数值配置，再复制物理文件并更新运行身份，物理场数值保持原样：

```powershell
python -m sst_preprocessing rebuild --config configs/preprocess_window_qc.json --source-root output/processed
# 中断时对同一命令添加 --resume。
```

`sst_analysis` 提供统一温度／梯度／锋线评价、分季节／锋区／MLD分组、配对时间块Bootstrap以及训练期观测诊断。只读取真实模型预测，不生成无训练基线。命令、预测格式及指标定义见 [完整使用说明](docs/analysis.md)。

诊断完成后的模型开发顺序、训练对照、算力验收和论文证据安排见 [下一步实现计划](docs/next_implementation_plan.md)。

第一阶段C0训练—验证—预测闭环已实现，入口为 `python -m sst_training`，配置为 `configs/training_c0.json`。先运行短程检查；命令、恢复规则及正式训练说明见 [C0训练使用说明](docs/training.md)。
