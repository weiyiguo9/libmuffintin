# 轨道配置（orbital definition）工作笔记 — doc/17 V2 方向

- Status: closed 2026-08-24 (V2 implemented; `doc/17_minimal_dft_scf.md` on `main` is the normative form)
- Date: 2026-08-24
- Imported: 2026-09-08 from `scratch/design_desision_01_orbit_config.md`, body unchanged

状态：**V2 架构、实现与内部验收已 closure（2026-08-24）**。两个问题在同一结构下统一：
channels token 表（身份）+ treatment（角色）+ 全局 energy-generator +
`@` 后缀例外（能量，每迭代从当前势重生成）+ recipe 工件（IR 文档，
默认值与 provenance）。细节决定记录见 §5。
实现：doc/17 V2 契约、`input.rs`/runner 硬替换迁移及 C1–C7 gate 已完成；
外部材料/cross-code acceptance 仍按 doc/17 §11 单独开放。
来源：本次会话对 doc/17 §2 轨道定义的讨论 +
四份代码调查（SPEX / Questaal / Elk / FLEUR，锚点见附录）。

本笔记刻意把两个正交的问题分开：

1. **输入文件里轨道用什么格式描述**（identity / syntax）——**已定并实现**，见 §3.3；
2. **能量参数要不要每迭代从当前势重新生成**（generator）——**已定：必须要**。
   四家参考代码的默认行为全部如此；原 doc/17 冻结能量缺口已由 V2 修正（§2）。

---

## 1. 现状诊断

"球内额外径向函数"目前有三套互不相通的定义词汇：

| 路径 | 身份键 | 能量来源 | 代码位置 |
|---|---|---|---|
| snapshot `LinearizationV1` | 每-$l$ | 从 FLEUR 导入后**全程冻结** | `crates/mt-io/src/snapshot.rs:369` |
| `[[task.scf.basis.local-orbitals]]` | site + 带符号 κ | 用户手写裸 Hartree 数 | `crates/mt-runtime/src/input.rs:278` |
| `[[task.scf.state-overrides]]` + 自动 $5p_{1/2}/6p_{1/2}$ 策略 | site + $(n,\kappa)$ | **每迭代在当前势上重解束缚 Dirac** | `input.rs:440`, `crates/mt-dft/src/atomic_configuration.rs` |

具体缺陷：

- `snapshot_dft.rs` 每迭代重建 `ScalarSiteInput` 时，势取当前迭代，但
  `linearization_energies` 始终取 snapshot 模板值——$E_l$ 和每-$l$ LO 能量
  冻结在"某次 FLEUR 收敛运行的输出"上。唯一活的能量是相对论 LO 的
  每迭代 Dirac 解（doc/17 §8）。
- `hdlo` 的 `energy` 字段是死的（HDLO 用 $E_l$，输入却强制要一个数）。
- 输入用带符号 κ，scalar 路线内部降回 $l$
  （`ScalarLocalOrbitalRequest`，`crates/mt-dft/src/scalar.rs:30`；
  `Kappa::large_l` 归约在 `snapshot_dft.rs`），于是需要
  "继承的 $l$-partner 移除一次"这类散文规则。
- 裸能量 LO 没有节点数/主量子数，说不清代表哪个壳层；裸能量本身是
  势依赖的，换元素/体积即失效——直接堵死固定 recipe。
- 按 site 字符串键控而不是 species；recipe 天然是 per-element 的。

公平性说明：全冻结能量并非绝对错误（Elk 发行的 `Si.in` 就是全冻结
0.15 Ha 的朴素 LAPW+lo），但那是"recipe 自带的势无关种子"。现状是把
**另一个代码收敛后的势相关输出**冻结复用，SCF 的正确性隐式依赖外部
代码的收敛状态；脱离该 snapshot 的势（换混合路径、换体系、从头起算）
就不成立。

---

## 2. 问题 2（已定）：能量每迭代重生成

### 2.1 四码对照

| 机制 | Elk | FLEUR | SPEX | Questaal |
|---|---|---|---|---|
| 原子/束缚态解（按 $n,l$ 节点数） | 离线种子（genspecies） | **默认**：节点数括界 + `differ` 原子解，每迭代 | `pbas=n` → `atom_ene` | lmfa 自由原子（离线） |
| 带中心/对数导数搜索 | **默认**：`findband` | 仅 HELO：二分到 $D=-(l+1)$ | — | $P$ 的 $\arctan$(对数导数) 映射 |
| 占据带重心浮动 | — | （enpara mix） | `pbas<0`（未实现） | **默认**：IDMOD=0 浮到带重心 |
| $E_F$ + 偏移 | `autolinengy`: $E_F+\delta$，$\delta=-0.1$ Ha | — | — | — |
| 显式冻结数 | `ve=F` | `qn=0` | `pbas=0` | IDMOD=1 |

共性：**默认都是每迭代从当前球势生成，手写冻结数只是逃生舱口。**

### 2.2 各家机制细节

**Elk**（`linengy.f90`，每迭代由 `gndstate.f90:157` 调用，用当前 $V_{00}$）：

- `findband.f90`：带顶 = $u(R_{\rm MT})=0$（向上搜索），带底 = $u'(R_{\rm MT})=0$
  （从带顶向下搜索），$E=(E_t+E_b)/2$。自适应步长括界
  （初始 `de0=0.001`，收敛 `epsband=1e-12` Ha，`demaxbnd=2.5` Ha 仅对
  负种子生效，每方向 250 步上限，双 pass 补救）。
- **每迭代从 species 文件种子重启搜索**，不从上一迭代继续——结果只依赖
  (recipe, 当前势)，无迭代历史依赖，可复现性好。
- 每函数一个开关 `apwve/lorbve`；失败时保留种子值，仅每迭代一条聚合警告
  （非致命）。
- `autolinengy`：冻结函数改用 $E_F + \texttt{dlefe}$，每迭代随 $E_F$ 更新。

**FLEUR**（`global/find_enpara.f90`，`t_enpara%update` 每迭代在当前
`v%mt(:,0,n,jsp)` 上运行）：

- `qn>0`（valence 通道与 SCLO）：目标节点数 $=n-l-1$，0.01 Ha 步长括界出
  能带分支，取中点后用 `differ`（原子束缚态解）精化——**最终能量本质是
  当前球势上该 $(n,l)$ 原子样本征值**。$l>0$ 的 LO 额外做 j-平均
  $(2e_{j=l+1/2}+e_{j=l-1/2})/3$。
- `qn<0`（HELO，负号即分派开关）：粗括界（5 Ha 步）后对
  对数导数二分至 $D=-(l+1)$（$R_{\rm MT}$ 处的"自然"边界条件）。
- `qn=0` = 冻结（保持 enpara 文件值）。失败 = 硬错误 `juDFT_error`，
  无静默回退。
- inpgen 默认表 `default.econfig`（静态人工整理：econfig 的 `core|valence`
  分割 + `lo=` 列表），profile 可自动加 HELO/HDLO。

**SPEX**（`getinput.f:2342-2346`，`iterate.f:2420` `get_ebas` → `2490` `get_bas`）：

- 每个径向函数带 `ebas`（能量）+ `pbas`（来源标签）：`0` 显式 /
  `n>0` 原子解（`atom_ene`）/ `n<0` PDOS 重心（预留未实现）；
  另有 `fbas` 填充模式（`l=...:+` 自动加 LO 到 PW 截断能）。
- 双模式：自有 DSL 自解，或从 FLEUR `gwa/radfun/ecore` 逐字继承。

**Questaal**：连续主量子数 $P_l = 0.5 - \arctan(D_l)/\pi + n$；
IDMOD=0（默认）每迭代浮到占据带重心，IDMOD=1 冻结；
P/PZ 中主量子数较深者浮动、较高者自动冻结。

### 2.3 收敛后的生成器枚举（provenance）

```text
explicit          冻结种子（现状唯一模式，降级为逃生舱口）
atomic            (n, l|κ) 束缚本征值，当前 V00      [FLEUR method1 / SPEX atom_ene / 已实现的 rLO Dirac 解]
band-center       findband：顶/底括界取中点          [Elk 默认]
log-derivative    D = −(l+1)（或指定目标）           [FLEUR HELO；Questaal P 映射的一般化]
band-cog          占据 PDOS 重心                     [Questaal IDMOD=0 / SPEX pbas<0]
fermi-offset      E_F + δ                            [Elk autolinengy]
frozen-snapshot   保留：对 FLEUR/SPEX 交叉验证的回归锚
```

### 2.4 契约级决定（建议随问题 2 一起定下）

- **每迭代从 recipe 种子重启**（Elk 语义）：生成结果 = f(recipe, 当前势)，
  不依赖迭代历史。不做 FLEUR 式 enmix 能量混合（Elk 证明不需要）。
- **失败显式**：符合仓库既有原则（core 解的 `NotFound/Ambiguous`
  不许启发式选根）。Elk 的"静默回退种子+聚合警告"最多作为显式
  `fallback = "seed"` 的 opt-in，不做默认。
- `frozen-snapshot` 模式保留为回归锚：生成器是**增量**，不破坏
  现有 FLEUR 快照交叉验证。

### 2.5 实现落点（现有组件盘点）

- `band-center`：在 `RadialSolver::solve(l,E)` 的 boundary 输出上包一个
  括界循环即可（findband 本体不足百行）。
- `atomic`：节点计数括界已存在（`core_dirac::isolate_core_dirac_bracket`）；
  相对论 LO 的每迭代 `solve_valence_dirac` 就是**已在运行的第一个生成器**，
  只是尚未抽象成公共接口。
- j-平均方向可以反转：FLEUR 从 scalar 出发做
  $(2e_{j=l+1/2}+e_{j=l-1/2})/3$；libmuffintin 的 4c 机器是一等公民，
  可**按 κ 生成、按简并度平均给 scalar**，比 FLEUR 干净。

---

## 3. 问题 1（未定）：输入格式候选

### 3.1 与语法无关的设计不变量

1. 身份键 = (species|site, n, κ|l, derivative-order)；**裸能量从身份键
   降级为生成器的一种**（正是 SPEX `pbas` 的结构）。
2. 一套词汇覆盖所有 treatment：core / valence / lo / hdlo；
   `relativistic-local-orbital` 不再特殊 = lo + κ 标签 + `atomic` 生成器；
   `state-overrides` 整个消失。
3. **输入语法与规范化 IR 分层**：紧凑语法进，逐通道 typed 表只作为
   IR/provenance 回显，永不手写。单向 parse→normalize，不违反
   doc/17 "only one syntax" 原则。
4. 不把语义编码进数字位：Questaal `PZ=15.936`（十位=extended 标志、
   个位=n、小数=arctan 对数导数）和 FLEUR 负 n = HELO 都是反例。
   紧凑该学"数组"，不该学"位数编码"。

### 3.2 候选 A：逐通道 typed 记录（第一版提案）

```toml
[[task.scf.basis.channels]]
species = "Si"
n = 2
kappa = 1
treatment = "local-orbital"
energy = { reference = "atomic" }
derivative-order = 0
boundary = "confined"
```

判定：作为**输入**太重（一个通道 7 行），已否决；作为**规范化 IR**
保留（provenance 回显、snapshot 记录、`BasisSpec` 编译输入）。

### 3.3 候选 B：光谱学 token 数组（第二版提案，当前倾向）

```toml
[task.scf.basis]
l-max = 8
energy-generator = "atomic"        # 全局默认生成器：所有通道一个，默认 atomic（已定，2026-08-23）
# recipe = "si-standard"           # 可选：整体引用 recipe 工件（IR 文档），见 §3.6

[task.scf.basis.envelope]          # 包络轴，与通道表正交；v0.3 加 site-centered 块
kind = "plane-wave"
cutoff = 4.0

[task.scf.basis.channels.Ti]
lo   = ["3s", "3p"]                # 能量默认走全局/recipe 生成器
hdlo = ["3d"]                      # 语法层面就没有能量字段

[task.scf.basis.channels.Pb]
valence = ["5p-"]                  # 撤销自动策略给的 5p1/2 rLO
lo      = ["5d@frozen", "4f@-0.15"]  # @ 后缀 = 单通道例外（低频功能）

[task.scf.basis.channels."Pb-1"]
lo = ["+5f", "-4f"]                # site 级 = token 编辑：+ 增 / − 删（细合并，已定）
```

token 文法（唯一需要解析的字符串，单 token 词法，非自由句子）：

```text
token  := ["+"|"-"] n l [j-tag] ["@" suffix]   # 前缀 +/− 仅在 site 级编辑合法
n l    := "5p"                     # 主量子数 + 轨道字母
j-tag  := "-" | "+" | "1/2"式全写   # 5p- ≡ 5p1/2 (κ=+1)；5p+ ≡ 5p3/2 (κ=−2)；两种写法同义，均接受
suffix := 生成器名 | "frozen" | 显式能量   # "@atomic" | "@band-center" | "@frozen" | "@-0.15" | "@-4.08ev"
能量单位：裸数 = Hartree；带 ev/eV/EV 后缀 = 电子伏（已定，2026-08-23）
删除 token 按通道身份 (n, l, j-tag) 匹配；未命中、同级重复均报错
```

生成器优先级链（单向，不做 Questaal `LOC=1/2, MTO=1..4` 式可配置覆盖开关）：

```text
内置默认  <  recipe 工件  <  任务级 energy-generator  <  单通道 @ 后缀
```

### 3.3b 行内不做双语法（IR 写法按文档分层，2026-08-23 讨论结论）

问题：单通道冻结/改生成器要不要同时支持 IR（typed 记录）和 token 两种
行内写法？用户预估该功能低频。

结论建议：**行内只保留 token 一种**。低频恰是反对双语法的论据——低频
路径易腐烂，双语法带来规范化歧义（两种写法各写一条时谁赢）、双倍
`input.rs` 校验面，且违反 doc/17 "only one syntax" 原则。冻结/改生成器
用 `@frozen`/`@atomic`/`@-0.15` 后缀即覆盖。

IR 写法的真实动机（精确性、程序化生成、Python 接口）改由**文档分层**
满足：IR 是独立的 recipe 工件文件格式（basp 式"生成→人审→冻结引用"），
workflow TOML 用 `recipe = "..."` 整体引用，provenance 回显亦为该格式。
每种文档内部单语法，parse→normalize 保持单向。

- scalar 语境：裸 `"5p"` 即 $l$ 通道；spinor 语境：裸 token 展开成两个
  κ partner，带 j-tag 则单选。
- 同 $l$ 多 LO：`["4f", "5f"]`；双能量：`["3d@-0.1", "3d@0.5"]`。
- SPEX `:+` 填充模式（fill 类生成器 token）**推迟进 v0.3 计划**：改变
  通道数量属于 recipe 生成器职责，token 文法不动结构（v0.3 plan §0.16）。
- 全部默认行来自 econfig recipe，用户只写"编辑"。

### 3.4 各家人类实际书写形态（对照）

```text
SPEX      LO (1s,2p:5.0,3d:+)                    紧凑 DSL，nl 标签 + 显式能量 + 填充
FLEUR     <energyParameters s="3" p="3" d="3" f="4"/>
          <lo type="SCLO" l="1" n="4" eDeriv="0"/>   量子数驱动，逐条 XML
Elk       apwe0, apwdm, apwve 三元组逐阶排列        位置式，冗长但显式
Questaal  P= 6.896 6.817 ...  PZ= 0 0 15.936        每-l 数组，紧凑但位数编码难读
```

### 3.5 映射闭环（recipe 对应性验证）

- FLEUR econfig 默认 = channels 表的隐式行；override = 编辑一行。
- Questaal：`P=` ↔ valence 行 + band-cog 生成器；`PZ=` ↔ lo 行（$n\pm1$）；
  `RSMH/EH`（至多 2 kappa）↔ 通道的包络头参数（正交的 envelope 块，v0.3）；
  `basp0→basp` ↔ recipe 工件的"生成→人审→冻结"流程。
- SPEX：`2p:5.0`→explicit、`1s`→atomic、`:+`→填充生成器；`pbas` ↔ §2.3 枚举。
- v0.3 `BasisSpec` 五轴：channels 表 = SiteChannel 选择 +
  `Linearized/IndependentHeads` 能量表示；envelope 正交；
  extended LO（Questaal `PZ+10`，带 smooth-Hankel 尾）= 未来
  `boundary = "matched-envelope"`，v0.2 只有 `confined`。

---

## 4. 广义 MTO 适用性结论（背景）

- **球内层逐字存活**：Questaal lmf 球内仍是每-$l$ 线性化对
  $(\phi,\dot\phi)$（能量由 $P$ 参数化）+ `PZ` 第三径向函数，与
  $(u,\dot u,\mathrm{LO})$ 同构。SPEX 式定义在这一层对 MTO 系列完全适用。
- **不存活的部分**：把基组压平成"全局 PW cutoff + confined LO 列表"；
  MTO 需要每-species-每-$l$-每-κ 包络参数表；LO 不必然禁闭
  （extended LO 带自己的间隙尾巴）。
- PMT：APW 只是同一久期方程里的额外包络，球内共用同一套 $(\phi,\dot\phi)$
  机器——支持"envelope 正交轴"的分解。
- JPO：实验特性（`--v8 --tbeh`），包络推广（`e0parm.f`：smooth Hankel +
  Gaussian 修正，按动能匹配在硬核半径拟合）仍只是每通道多几个包络参数，
  channels 抽象不受影响。

---

## 5. 细节决定记录（2026-08-23 全部拍板）

1. **token 文法**：j-tag 两种写法都收（`5p-`/`5p+` 与 `5p1/2`/`5p3/2`
   同义，规范化到 IR 后无别）；`@` 后缀 = 生成器名 | `frozen` | 显式能量；
   显式能量裸数 = Hartree，带 `ev/eV/EV` 后缀 = eV（如 `"4f@-4.08ev"`）；
   fill 类生成器 token（SPEX `fbas`/`l:+`、Elk `lorbcnd`）推迟进
   **v0.3 计划**（recipe 生成器职责，v0.3 plan §0.16）。
2. **species/site 冲突**：token 级细合并——species 级写完整列表，
   site 级写编辑 token（`+` 增 / `-` 删，前缀位置，与 j-tag 后缀无歧义）；
   删除按通道身份 (n,l,j-tag) 匹配，未命中、同级重复均报错。
3. **生成器结构**：全局唯一 `energy-generator`，默认 **`atomic`**；
   单通道 `@` 后缀写例外；band-center 等降为 recipe/后缀选项；
   行内不做 IR/token 双语法，IR 走 recipe 工件文档（§3.3b）。
4. **失败策略**：V2 一律硬错误（带通道标识与括界诊断）；
   `fallback = "seed"` 显式 opt-in **推迟到 v0.3**。
5. **生成方向**：V2 唯一实现 = **κ-first**（按 κ 用 Dirac 机器生成，
   scalar 取 (2j+1) 简并平均，$l>0$ 即 $(2e_{j+}+e_{j-})/3$）；
   l-first 保留为将来选项但**推迟到 v0.3**。文档注明各家归属：
   Elk/SPEX/Questaal 为 l-first，FLEUR 的 LO 生成为按 j 解再平均
   （κ-first 同类）。
6. **schema 迁移**：硬替换——`format version 2`，V1 输入直接拒绝并报
   指向新写法的清晰错误；snapshot V1→V2 数据兼容不受影响（数据工件，
   另一回事）。
7. **recipe 工件**：内置 FLEUR econfig 表为零配置默认；
   `recipe = "path.toml"` 引用外部工件（生成→人审→冻结）；
   provenance 回显即该格式，可直接另存为新 recipe。
8. **诊断**：生成能量进 `ScfState`/provenance 回显（生成器名 + 种子 +
   实际值）。
9. **导数阶是开放整数轴，不硬定在 2**（2026-08-24 补定）：Elk 的
   `apwdm`/`lorbdm` 本来就是逐函数任意整数（`modmain.f90:737,773`；
   上限 `maxapword=3`/`maxlorbord=4` 只是编译期数组帽，`nxoapwlo` 加阶
   即得 superLAPW 级匹配）。IR 的 `derivative-order` 字段 = 任意非负
   整数；V2 实现范围仍是 0/1（u/u̇ 隐含）+ 2（`hdlo` 键），$\ge 3$
   schema 合法但报显式 NotImplemented；token 文法暂不加阶数后缀，
   $\ge 3$ 的书写语法与实现一起推迟到 v0.3（Elk 的实现路径极廉价：
   在 $E + m\,\Delta E$ 移位解 + 对低阶 Gram–Schmidt 正交化，
   `genapwfr.f90:57-76`，无需高阶能量导数 ODE）。V2 契约与迁移已经实现；
推迟项（fill、l-first、fallback、`derivative-order >= 3` 的 token/执行）
已挂入 v0.3 plan §0.16。

---

## 附录：调查锚点

**SPEX**（本地 `spex06.00pre36/src`，06.00pre38 树）：

- 身份/数组：`global.f:120-127`（`bas1/bas2(grid,n,l,itype,ispin)`,
  `ebas`, `pbas`, `fbas`, `nindx`）；`nindx` 起始 2（`getinput.f:930`），
  n=1:u, 2:u̇, ≥3:LO。
- `pbas` 语义与 LO DSL：`getinput.f:2342-2346`（`parse_lo`），
  `parse_epar` `getinput.f:2271`。
- 每迭代生成：`iterate.f:2420`（`get_ebas`：`atom_ene`/`cog_ene`）、
  `iterate.f:2490`（`get_bas`：n≠2 齐次解 @ `ebas(n)`；n=2 非齐次 + 正交化）。
- FLEUR 继承模式：`readwrite_fleur.f:53,101-112,231-297,359+`
  （`gwa`/`radfun`/`ecore` 逐字读入）。
- 组合指标序：$l$ 外层、$m$ 中层、$n$ 最内（`iterate.f:2549-2573`,
  `mixedbasis.f:184-198`）。

**Questaal**（github `70akaline/questaal`）：

- 包络存储：`src/subs/structures77.h` `orbp(10,2,nkap0)`，
  解包 `src/subs/uspecb.f`（槽 3 = extended LO）。
- PZ 类型判定：`src/fp/makusp.f`（lpzi=1 禁闭 / 2 extended / 3 未实现）。
- $P$ 定义：doc/lmto.html（$P=0.5-\arctan D/\pi + n$）；球内线性化确认：
  doc/fp.html。
- AUTOBAS/basp：doc/tokens.html、doc/Building_FP_input_file.html
  （ELOC=−2 Ry、QLOC=0.005 判据）；`--optbas` 坐标下降。
- PMT：HAM `PWMODE/PWEMIN/PWEMAX/OVEPS`、SPEC `KMXA`。
- JPO：`src/lmv7.f`（`--v8 --tbeh`）、`src/fp/modfsm.f90`、
  `src/subs/e0parm.f`。

**Elk**（本地 `elk-10.4.9/src`）：

- 驱动：`linengy.f90`（`gndstate.f90:157` 每迭代调用；种子重启在
  `linengy.f90:68`；冻结分支 `autolinengy` 在 `:74,102`）。
- 搜索：`findband.f90`（顶 `:70` $p_0(nr)$、底 `:94` $p_1(nr)$、
  中点 `:114`；失败不写回 `e`）。参数默认 `readinput.f90:173-178`。
- 开关数组：`modmain.f90:733-739`（apw）、`:769-775`（lorb）。
- recipe 生成器：`genspecies.f90`、`writespecies.f90`
  （`ecvcut=-3.5` 芯价分割 `:25`；半芯 LO 判据 `eval<esccut=-0.4` 或
  `l>=2` `:44`；第二阶种子=原子本征值且 `ve=T` `:76-78`）。
- 扩展生成器：`nxoapwlo`（HDLO 型加阶，`readspecies.f90:198-212`：复制
  末阶能量、`dm+1` 递增；`nxoapwlo=1` 即 APW→LAPW，`elk.f90:1266-1272`）、
  `nxlo`（加 $l$）、`lorbcnd`/`addlorbcnd.f90`（传导带 LO，全搜索使能）。
- 任意导数阶：`apwdm`/`lorbdm` 逐函数整数（`modmain.f90:737,773`），
  帽 `maxapword=3`（`:721`，恰到 superLAPW 级）/`maxlorbord=4`（`:747`）；
  高阶实现 = 移位能量 $E + m\,\Delta E$ 解 + Gram–Schmidt
  （`genapwfr.f90:57-76`），非能量导数 ODE。

**FLEUR**（本地 checkout，XML 路径）：

- 生成器：`global/find_enpara.f90`（`priv_method1` `:35-130` 节点括界 +
  `differ` 原子精化 + j-平均 `:120-126`；`priv_method2` `:132-259` HELO
  对数导数二分 `:243-257`）。
- 每迭代调用：`types/types_enpara.F90:95-304`（`t_enpara%update`，
  `qn=0` 即冻结 `:145-164`）。
- XML schema：`io/xml/FleurInputSchema.xsd:617-641`
  （`LOType`：SCLO/HELO + `eDeriv`；HELO = 负 n 编码，
  `fleurinput/types_enparaXML.f90:83-85`）。
- inpgen 默认：`inpgen2/default.econfig`（静态表）、
  `inpgen2/atompar.F90:101-182`、`inpgen2/make_atomic_defaults.f90:84-183`
  （`addHELOs_noSC`/`addHDLOs_noSC`；自动 HELO 规则 `:175-181`）。
- 失败：`find_enpara.f90:83,92-109,181` 硬错误 `juDFT_error`。
- 冻结/浮动：`floating`（lepr）机制在 XML 路径休眠
  （`types_enpara.F90:71` 初始 false，仅 `inpgen2/old_inp/inped.F90` 可达）。

**libmuffintin 现状锚点**：

- 冻结能量路径：`crates/mt-runtime/src/snapshot_dft.rs`
  （`ScalarSiteInput` 组装：势=当前、能量=snapshot 模板）。
- 三套输入词汇：`crates/mt-runtime/src/input.rs:278`（`LocalOrbital`）、
  `:440`（`ElectronicStateOverride`）、`crates/mt-io/src/snapshot.rs:369`
  （`LinearizationV1`）。
- scalar κ→l 归约：`crates/mt-dft/src/scalar.rs:30`
  （`ScalarLocalOrbitalRequest`）。
- 已存在的生成器实例：相对论 LO 每迭代 `solve_valence_dirac`
  （doc/17 §8；`crates/mt-radial/src/core_dirac.rs`）。
- 架构目标：`scratch/libmuffintin_v0.2_plan.md` §2b、
  `scratch/libmuffintin_v0.3_mto_family_plan.md` §0-§2
  （`BasisSpec` 五轴、`EnergyRepresentation`、`recipes::questaal_*`）。
