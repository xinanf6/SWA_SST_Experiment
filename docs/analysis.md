# 统一评价器与训练期区域诊断

`sst_analysis` 承接已经完成的预处理结果，提供两条独立流程：训练期观测场区域诊断，以及已提供模型预测文件的统一评价。本模块不生成预测，不训练模型，也未实现持续性、气候态或纯热力等无训练基线。没有真实模型预测文件时，不会产生实际预测分数。

## 1. 前置条件与运行命令

从仓库根目录运行，使用已安装预处理依赖的 Python 环境。配置默认为 `configs/analysis.json`：

```json
{
  "prepared_root": "../output/processed_window_qc",
  "diagnostics_output": "../output/analysis/regional_diagnostics_window_qc",
  "figure_dpi": 300,
  "batch_size": 1
}
```

配置中的两个路径相对于 JSON 所在目录解析。`--prepared-root` 可覆盖输入目录；命令行路径相对于当前工作目录解析。分析输出必须独立于预处理目录，不能是其内部目录或祖先目录。

输入目录至少需要匹配同一 `run_identity` 的 `READY.json`、`run.json`、`stats.json`、`grid_masks.npz`。`READY.json` 必须为 `complete`。诊断另需训练期的 `physical/YYYY-MM.nc`；评价另需 `sample_index.csv`、`dates.npy` 和 `normalized/base.npy`。网格、固定掩膜、面积权重和训练期锋面 P90 阈值直接复用预处理结果。

生成训练期诊断统计及图表：

```powershell
python -m sst_analysis diagnose --config configs/analysis.json
```

仅计算诊断统计，暂缓绘图：

```powershell
python -m sst_analysis diagnose --config configs/analysis.json --no-plots
```

已有诊断统计时重新绘图，不重新读取十年逐日物理场：

```powershell
python -m sst_analysis plot-diagnostics --config configs/analysis.json
```

评价完整验证集的真实模型预测：

```powershell
python -m sst_analysis evaluate --config configs/analysis.json --predictions output/predictions/model_validation.nc --split validation --output-root output/analysis/model_validation --model-name model_name
```

需要补充原始骨架锋线指标时，在评价命令后加 `--front-lines`。`--split` 还可为 `train` 或 `test`，默认 `validation`。`batch_size` 只控制评价批大小，不改变指标定义。

`diagnose` 和 `evaluate` 遇到非空输出目录会拒绝执行。需要重算已有结果时显式加 `--overwrite`，或指定新的 `--output-root`。该选项允许替换本流程同名产物，不清空目录中的其他文件。`plot-diagnostics` 不接受 `--overwrite`：它检查现有诊断的身份和训练期范围后，替换图表、报告和分析清单，保留统计表。

## 2. 区域诊断的时间与空间口径

当前实验训练期为 **2011-01-01 至 2020-12-31**，共 **3653 天、120 个月**。诊断只打开这些月份的物理单位文件，按最多 7 天的块读取；不使用验证期、测试期数据。这里的日数是物理场诊断日数，不是扣除历史窗口和标签后的训练样本数。

研究核心区为 35°—48°S、48°—60°W，对应原预处理实际网格的 `260×240` 核心区。输入保留 `270×250` 网格。区域温度、海流和热力统计使用 `core_mask & ocean_mask`，梯度和锋面统计使用 `core_mask & gradient_mask`。缓冲区不进入统计归约；图中的灰色区域为固定掩膜遮蔽区域。

各格点地图先对有效逐日值作时间统计，均值为有效日等权平均，SST 标准差使用总体定义 `ddof=0`。区域逐日统计采用 `cos(latitude)` 面积权重，每个变量独立计算有效权重分母。月循环再对对应月份的有效区域日统计等权平均，同时保留每项有效日数。没有有效观测的项保持未定义。

季节按日历月分类：DJF 为南半球夏季，MAM 为秋季，JJA 为冬季，SON 为春季。统计保留训练期内所有对应月份，包括训练边界的 1、2、12 月；不是仅保留完整跨年 DJF 的合成。

诊断量包括 SST 均值及时间标准差、表层流速和流矢量、梯度强度、强锋频率、净热通量、混合层厚度、热力日增量及其绝对值、观测 SST 日变化绝对值。平均流速为逐日 `sqrt(uo²+vo²)` 的平均，流矢量分别平均 `uo`、`vo`。SST 日变化为相邻训练日之差的绝对值，记在后一日；跨月、跨分块连续，首个训练日不计算。

强锋区采用冻结的训练期第 90 百分位阈值，比较符号为 `>=`。格点锋面频率的分母是该点有效梯度日数；区域强锋面积比例的分母是当天有效梯度区域的面积权重。原始数据文件保存比例 `0—1`，图表显示为百分数。

热力约定保持最初方案：

\[
h_{\mathrm{eff}}=\max(h,3\,\mathrm m),\quad
Q_{\mathrm{net}}=\frac{ssr+str+slhf+sshf}{86400},\quad
\Delta T_{\mathrm{heat}}=\frac{ssr+str+slhf+sshf}{1025\times3990\times h_{\mathrm{eff}}}.
\]

四项 ERA5 热量已是日累积能量，单位 `J m-2`；正号表示海洋得热，不再重复累积。未加入短波穿透或衰减系数。厚度地图展示物理文件中的 MLD，3 m 下限只用于热力增量计算。诊断描述同日物理场；模型输入的环境场滞后一天规则仍由样本索引负责。

这些诊断用于描述区域分布与季节变化，不能单独解释为过程贡献比例或因果证据。

## 3. 模型预测文件契约

评价要求所选分区的**全部质量筛选后样本**，顺序必须与 `sample_index.csv` 中该分区一致；不接受缺行、重排或临时抽样。预测起报日必须逐项匹配 `origin_date`，目标日期必须匹配 `target_1_date`、`target_2_date`、`target_3_date`。当前预处理有效样本数为训练 3580、验证 704、测试 714；后续更换预处理批次时以新索引为准，不硬编码这些数目。

验证集 704 与无缺日理论窗口数不同，是原始数据质控筛掉了 9 个窗口；不能为凑足理论数目补造预测日期。14 天历史、环境场滞后一天以及未来 3 天标签的边界均由原预处理索引决定。

经纬度必须与预处理实际网格同形、同序；比较容差为 `1e-6` 度，不在评价阶段插值。`run_identity` 必须与输入预处理批次一致。温度单位必须显式为以下三者之一：

| `units` | 读取后的处理 |
| --- | --- |
| `degree_Celsius` | 直接使用物理℃ |
| `K` | 减去 273.15 |
| `sst_zscore` | 使用该批次训练期 `stats.json` 的 SST 均值、标准差反标准化 |

所有指标最终在物理℃空间计算，不根据数值大小猜测单位。预测必须在整个固定海洋域（含梯度所需缓冲区）有限；任一海点 NaN/Inf 会报错。陆地值可为 NaN 或占位数，评价时统一屏蔽。

### NPZ

只使用 `.npz` 后缀，禁止 pickle。必需键如下：

| 键 | 形状或类型 |
| --- | --- |
| `predictions` | 实数数组 `[N,3,H,W]` 或 `[N,3,1,H,W]`；当前 `H=270,W=250` |
| `origin_dates` | `[N]`，推荐 `datetime64` 或 ISO 日期字符串 |
| `target_dates` | `[N,3]`，依次对应 Day 1、Day 2、Day 3 |
| `latitude` / `longitude` | 实际网格一维坐标 `[H]` / `[W]` |
| `units` | 单个 Unicode/ASCII 字符串 |
| `run_identity` | 单个 Unicode/ASCII 字符串 |

NPZ 预测数组读取时会整体解压到内存；大规模预测可优先使用下面的 NetCDF 格式按起报日分批读取。

### NetCDF

只使用 `.nc` 后缀，变量和维度约定为：

```text
predictions(origin, lead, latitude, longitude)
target_dates(origin, lead)
origin: 与完整分区索引一致的起报日期坐标
lead: [1, 2, 3]
latitude, longitude: 预处理实际坐标
predictions.attrs["units"]: 上述三种单位之一
dataset.attrs["run_identity"]: 匹配的预处理批次标识
```

NetCDF 中 `predictions` 的维度名称和顺序必须精确匹配，不接受额外 channel 维。`target_dates` 必须显式提供，不能只依赖文件名推断日期。

评价真值始终从预处理 `normalized/base.npy` 的 SST 通道按目标索引读取，再使用冻结训练统计反标准化。外部文件中的真值不能替换这套参考；这里评价的是相对于准备好的 OSTIA 分析目标的误差。

## 4. 指标的精确定义

每个提前期分别统计。记该提前期所有样本核心海洋格点的误差为 \(e=\hat T-T\)，权重为 \(w=\cos\varphi\)，\(W=\sum w\)。

\[
\mathrm{RMSE}=\sqrt{\frac{\sum we^2}{W}},\quad
\mathrm{MAE}=\frac{\sum w|e|}{W},\quad
\mathrm{Bias}=\frac{\sum we}{W},\quad
R^2=1-\frac{\sum we^2}{\sum w(T-\bar T_w)^2}.
\]

RMSE、MAE、Bias 单位为℃；正 Bias 表示预报偏暖。\(\bar T_w\) 是该提前期全部参考样本及格点的合并加权均值，R² 不是逐图 R² 的平均。累计使用加权误差平方和、绝对误差和、误差和及参考一、二阶矩，不平均各批次 RMSE。

梯度使用完整输入场的球面米制中心差分，先屏蔽陆地，再在 `core_mask & gradient_mask` 上归约；单位为℃/km。梯度矢量误差与强度误差分别为：

\[
\mathrm{gradient\_vector\_mae}
=\frac{\sum w\bigl(|\hat g_x-g_x|+|\hat g_y-g_y|\bigr)/2}{\sum w},
\quad
\mathrm{gradient\_magnitude\_mae}
=\frac{\sum w\bigl||\hat{\boldsymbol g}|-|\boldsymbol g|\bigr|}{\sum w}.
\]

`strong_front_rmse/mae` 只在真值梯度强度大于等于训练 P90 的核心有效梯度点上计算温度误差。强锋真值只用于评分掩膜，不改变模型输入。

`front_region_precision/recall/f1` 是原始超阈值二值锋区的面积加权指标。记预测锋区为 \(P\)，真值锋区为 \(G\)，则 Precision 为 \(A(P\cap G)/A(P)\)，Recall 为 \(A(P\cap G)/A(G)\)，F1 为 \(2A(P\cap G)/(A(P)+A(G))\)。按提前期合并面积统计后计算，属于 micro 聚合，不是逐图 F1 等权平均。

### 可选的骨架锋线补充指标

开启 `--front-lines` 后，对完整有效梯度输入域的二值锋区进行原始 Zhang–Suen 两子迭代细化，然后裁到核心区。实现为纯 NumPy，不依赖 scikit-image；未增加平滑、连通域面积/长度过滤等最初方案之外的处理。它遵循 [Zhang 与 Suen 原论文第 3 节](https://teaching.vda.univie.ac.at/ipa/18w/Lab3/Zhan-Suen_algorithm.pdf) 的同步删除规则，算法 DOI 为 [10.1145/357994.358023](https://doi.org/10.1145/357994.358023)。[scikit-image 官方说明](https://scikit-image.org/docs/0.21.x/api/skimage.morphology.html#skimage.morphology.skeletonize)介绍了二维 Zhang 骨架方法；本实现不声称与其查找表变体逐像元一致。

锋线点按球面大圆距离求最近邻。单幅双向平均距离为“预测点到真值线的平均最近距离”与“真值点到预测线的平均最近距离”的算术平均，单位 km；汇总距离是所有有定义单幅距离的等权平均，并报告有效幅数。

10 km 容差下的锋线 Precision 是预测骨架点中距任一真值骨架点不超过 10 km 的比例，Recall 反向计算。汇总先合并两方向的匹配点数和总点数，再计算 Precision、Recall、F1。锋线按点计数，不作面积加权。保留零碎骨架，因此这些指标作为补充，与主温度和梯度指标共同解读。

### 未定义与空锋面

分母为零时不填造分数：R² 遇常量或数值近常量真值为未定义；没有真值强锋点时，强锋 RMSE/MAE 为未定义，计数为 0。

对锋区和锋线，若预测与真值都为空，Precision、Recall、F1 都未定义；只有一侧为空时 F1 为 0，空侧对应的比率分母为零，记为未定义，另一方向为 0。任一侧骨架为空时距离未定义，不作为 0 km 纳入均值；输出同时保留空线幅数和有效距离幅数。

JSON 使用 `null`，CSV 使用空单元格，NetCDF 浮点场使用 NaN；评价 Markdown 使用 `undefined` 表示未定义。

## 5. 输出目录与复现记录

诊断默认输出到 `output/analysis/regional_diagnostics_window_qc/`：

| 文件 | 内容 |
| --- | --- |
| `seasonal_maps.nc` | `annual/DJF/MAM/JJA/SON` 地图及每变量有效日数；完整坐标上核心域外为 NaN，计数为 0 |
| `daily_regional.csv` | 每个训练日的区域统计及各变量有效格点数 |
| `monthly_cycle.csv` | 1—12 月区域日统计的均值及各指标有效日数 |
| `diagnostics_summary.json` | 数据范围、训练月文件列表、阈值、单位、覆盖率、方法 |
| `diagnostics_report.md` | 区域日统计汇总、解释边界与图表链接 |
| `figures/` | 五组图的 PNG/PDF；PNG 默认 300 dpi |
| `analysis_manifest.json` | 预处理身份、分析代码 SHA256、设置、软件版本与 UTC 完成时间 |

五组图名为 `regional_overview`、`front_structure`、`thermal_environment`、`seasonal_structure`、`monthly_cycles`。命令运行完成后按报告链接查看；本文说明产物契约，不预先认定某次真实数据绘图已完成。

物理量单位为：SST 和标准差℃，流速 m/s，梯度℃/km，净热通量 W/m²，MLD m，热力日增量与绝对 SST 日变化℃/day，频率和面积比例为无量纲。地图中的计数是该格点有效观测日数；日统计中的计数是有效格点数，二者不能混用。

模型评价写入 `--output-root`：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 模型名、分区、预测文件 SHA256、单位、规则和三个提前期汇总 |
| `metrics_summary.csv` | Day 1、Day 2、Day 3 各一行汇总指标及充分统计量 |
| `daily_errors.csv` | 每个起报日、提前期、目标日的指标和充分统计量 |
| `evaluation_report.md` | 主要物理单位指标表 |
| `analysis_manifest.json` | 本次分析身份与环境记录 |

充分统计量包括 `weight_sum`、`weighted_sse`、`weighted_absolute_error`、`weighted_error_sum`、参考温度一、二阶矩，强锋和梯度的对应加权和，以及锋区/可选锋线的匹配计数。`compare` 命令使用逐日起报充分统计量进行配对时间块Bootstrap。

后续训练程序只需按上述文件契约导出真实预测，即可共享同一真值、日期、掩膜和指标定义。区域诊断本身不会写出模型预测，也不会替代后续模型实验结果。

## 6. 固定分组与配对比较

`diagnose` 另生成 `training_mld_median.json`：训练期核心海洋域所有有效正MLD的精确中位数，按格点日等权，不作面积加权。7天分块写临时float32内存映射，原地分区求中位数，完成后删除临时文件。缓存校验训练范围、运行身份、网格和源文件记录。

CLI `evaluate` 默认输出8个分组：DJF、MAM、JJA、SON、strong_front、weak_front、shallow_mld、deep_mld。四季按各提前期目标日期分类；强弱锋按参考梯度是否达到冻结P90划分，只包括核心有效梯度点；浅深混合层使用起报前一天的MLD，`<=`训练中位数为浅，`>`为深。梯度仍先在完整场求导，再按中心格点所属组归约；分组不计算骨架锋线。

`group_metrics.csv` 含8组×3提前期汇总；`group_errors.csv` 保留所有起报日、提前期和组，包括不属于该季节时权重为0的行。全域 `daily_errors.csv`、`metrics_summary.csv` 保持独立。零权重行用于保留连续起报时间轴。

```powershell
python -m sst_analysis compare --config configs/analysis.json --evaluation-a output/analysis/model_a --evaluation-b output/analysis/model_b --output-root output/analysis/a_vs_b
```

比较要求两份真实模型评价具有一致的运行身份、分区、日期、分组和参考支持。默认以起报日期做30天移动块、2000次配对重采样，种子20261117，输出95%百分位区间。缺日起报序列拆为连续段，各段保持原长度，块不跨缺日；不足30天的段保留为整段短块，并记录此限制。季节零权重行保留在完整日历轴上。

输出为 `comparison.csv` 与 `comparison.json`。点估计是原始样本重新汇总后的模型A减模型B，不是重采样差值的均值。RMSE、MAE等误差指标差值为负表示A较小；Bias差值只表示偏差移动方向。零分母或无有效重复的区间为未定义，不自动作因果或创新结论。

## 7. 缺测窗口重建（模式版本2）

当前预处理目录为 `output/processed_window_qc/`。所有模型共用同一索引：训练3580、验证704、测试714，分别排除56、9、0个窗口。固定海洋域不再因单日大气缺测永久挖洞。原始值保持不变，剔除起报日期及原因保存于 `excluded_samples.csv`。旧目录 `output/processed/` 保留追溯，其掩膜、统计量和归一化数组不得与新批次混用。
