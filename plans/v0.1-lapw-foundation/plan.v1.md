# libmuffintin — 架构与 v0.1 实现计划

- Workstream ID: `v0.1-lapw-foundation`
- Plan version: 1
- Approval: accepted (historical; workstream closed)
- Supersedes: none
- Imported: 2026-09-08 from `scratch/libmuffintin_v0.1_plan.md` (last modified 2026-07-16), body unchanged
- Status source: false; state lives in `STATUS.md` and `ledger.md`

**版本**:draft-2(2026-07-16)——按 LAPW/SPEX 优先序重排;LMTO/KKR 后移
**状态**:讨论稿。开放决策点以 [D#] 标注,风险以 [R#] 标注。
**语言/实现**:Rust(workspace,多 crate)

---

## 0. 一句话定位

把 MT 方法家族(LAPW、LMTO、KKR、EMTO)的公共代数拆成带 trait 接口的库:**球内散射数据 + 包络展开 + 组装策略**。名字命名的是数据结构(MT 分割)而不是任何一家的方法——四个社区唯一无门派色彩共用的词就是 muffin-tin。这是 libxc 式命名:命名公共对象,不命名算法。

服务对象是全世界 ~20–30 个在 MT 层面工作的研究者。它**不是第 N 个 DFT 代码**:DFT 能带算对是健全性测试,散射数据与张量的可组合导出才是交付物。

与 SIRIUS 的区别一句话说清:SIRIUS 序列化的是**方法**(一个跑着的 LAPW 运行时,通过 API 替你算);libmuffintin 序列化的是**散射数据**,方法是这份数据上的投影。

**排序原则(本 draft 与 draft-1 的差异)**:路线图跟随 LAPW→SPEX 的生产管线次序——LAPW 频段引擎 → MPB/Coulomb → THC/CoQui 导出与 Wannier 投影子;LMTO/KKR(及其携带的双出口/方法统一论题)后移。理由:v0.1 的每一行代码都直接服务现役 GW(+EDMFT) 工作流,而方法统一论题的验证可以延迟、不能替代。代价见 §3。

---

## 1. 设计公理

**A1 势是输入,基永远是导出。** 原语 = 球内势 v(r) + 几何(+ 间隙区势的 PW 系数)。基函数表、匹配系数 A/B、LO 系数、t 矩阵、kink 矩阵全部是缓存,由 trait 按需生成。"收敛快照" = 每球径向标量函数 + 线性化能量 + 间隙 PW 系数,比存基函数表小两个量级,且方法中立——同一份 v(r),将来 KKR/LMTO 后端长出自己的基。

**A2 声明式数据 + 封闭代数。** 角向/几何层(Gaunt、step function 解析 Fourier 变换)解析闭合;径向叶子只在**求积**下闭合,因此网格参数与求积规则是 spec 的一等公民——否则两个实现在 1e-10 处对不上,cross-code diff 变成扯皮。

**A3 方法 = 能量离散化策略。** KKR 保留完整 E 依赖;LAPW/LMTO 是 E_ν 处的一阶 Taylor(LAPW 把匹配条件转嫁给间隙平面波);NMTO 是多点插值;EMTO 用 kink cancellation。组装层因此是 trait,不是 if-else。v0.1 只落地 LAPW 这一个策略,但 trait 边界按全家设计。

**A4 双出口,同一数据。** 波函数出口 (H, S)(LAPW/LMTO 血统:接 Wannier、接多体)与 Green 函数出口 M(E)(KKR 血统:杂化函数、CPA、输运)共享同一份径向 + 几何数据。v0.1 只实现波函数出口;M(E) 出口随 KKR 落地(§9)。

**A5 kernel 文化(PySCF 模板)。** 每个中间对象暴露且可劫持:劫持某个 l 通道的径向解、替换匹配系数、在同一份快照上对比不同基组参数。one-liner 门面是演示品,不是产品。

**A6 导出层消化基组残留。** metric、Poisson、q→0 head/wings 这些"基组残留"必须在导出层内部消化干净,对外交付 basis-clean 的张量。脏活留在球里,干净的代数留给世界。v0.1 不实现导出层,但这条公理约束所有内部接口的形状。

**A7 残渣隔离。** 真正抽象不动的私货很少:LAPW/APW+lo 的面上不连续与动能算符对称化约定、EMTO 重叠球与形状函数、FP-KKR 截断 folklore。全部放进各 strategy 的私有实现,不污染公共层。

**A8 非球项是装饰不是新类型。** full-potential 的 v_LM(r) 是同一结构的附加 LM 通道(schema decoration)。v0.1 的势模型为"球内 spherical(+可选 v_LM 通道)+ 间隙 warped(PW 系数)",schema 从第一天按多通道设计。注意:给定冻结势后,含 v_LM 的哈密顿矩阵元只是 Gaunt 加权径向积分——FP 的脓包在势的**构造**(Weinert/形状函数),不在矩阵元;v0.1 不做任何势构造。

---

## 2. 分层架构

```
L0  conventions   球谐约定 / Gaunt / log mesh / 求积 / 单位 / G 矢量 / step function  [解析闭合]
L1  radial        径向引擎: (v_00, Z, R, 方程, E) → u_l, u̇_l, u_l^lo, D_l, (t_l)      [求积闭合]
L2  envelope      包络 trait + 球面匹配 → 增广线性映射 (A/B/LO 系数)                   [纯几何+边界数据]
L3  assembly      组装策略 trait: (L1 × L2 × 势) → {(H,S) | M(E)}
L4  spectra       广义本征(Cholesky 约化) / k-path / (det-zero 追踪, 随 KKR)
L5  exits (预留)  orbitals_on_grid / projectors / mpb / thc_export / green / downfolding
```

### L0 conventions

实球谐 vs 复球谐、Condon–Shortley 相位、(l,m) 排序、Gaunt、对数网格、求积规则、单位制、G 矢量集生成、MT 球 step function 的解析 Fourier 变换 Θ̃(G) 及其截断约定、动能算符对称化(∇·∇ 形式)约定。全部收进 `CONVENTIONS.md` + `mt-core`,单一真源,100% property-tested。这一层没有数值妥协余地——它是 cross-code diff 与"对 FLEUR 到 meV"这两个目标的共同地基。

[D1] 单位制:内部 **Hartree**(FLEUR/SPEX 内部即 Hartree,最小化对表摩擦;draft-1 的 Ry 选择随 MJW/LMTO 参照系一同后移),newtype 包裹,I/O 显式标注。

### L1 径向引擎

输入:球内势 v_00(r)@log mesh、Z、R_MT、径向方程、能量表(E_ν 与 LO 能量)。
输出:u_l(r,E_ν)、能量导数 u̇_l、LO 径向函数、对数导数 D_l(E)、径向积分件(⟨u|u⟩、⟨u|u̇⟩=0、⟨u̇|u̇⟩、含 v_LM 的 Gaunt 加权积分)。KKR 侧的 t_l(E) 属于同一引擎,随 v0.4 激活。

相对论全部在这一层:

```rust
enum RadialEquation {
    Schroedinger,          // v0.1 (解析测试用)
    ScalarKoellingHarmon,  // v0.1 (FLEUR 价态同款)
    DiracKappa,            // schema 预留; 标签 l → κ, 输出 (g_κ, f_κ)
}
```

二次变分 SOC(ξ_l = ⟨u|(1/r)dv/dr|u⟩ 径向积分 + L·S)是 L1+L4 的小模块,排 v0.2(现役 GW+SOC 工作流需要,但不进 v0.1 验收)。

### L2 包络与增广

```rust
trait Envelope {
    // 单中心展开 + 球面匹配 → 增广线性映射
    // v0.1 实现: PlaneWave  —— A_lm(k+G), B_lm(k+G), LO 系数;
    //            匹配条件即 "把 Rayleigh 展开缝到 (u, u̇) 上" 的线性映射, 作为数据缓存
    // 后续实现: SphericalHankel —— 结构常数 (canonical S / KKR G⁰), 随 v0.4
}
```

增广是**存下来的线性映射**,不是代码逻辑(A1)。LO 进入同一套匹配代数(额外行),v0.1 即支持 semicore LO;高能 HDLO 在 schema 上无新概念(只是更多能量条目),留给 MPB/GW 收敛需求(v0.2+)激活。

### L3 组装(v0.1: Lapw)

MT 球内:A/B/LO 系数 × 径向积分件 × Gaunt(球对称通道 + 可选 v_LM 通道);面上动能不连续按 A7 记入 strategy 私有约定。间隙区:动能 ½(k+G)·(k+G′)Θ̃(G−G′),势能 = v_I(G) 与 Θ̃ 的卷积(直接卷积,FFT 是后续优化)。输出 `Linearized{H,S}`。

### L4 谱

S 的 Cholesky 约化 + Hermitian 本征;高 RK_max 下 S 的近线性相关用带阈值的谱过滤兜底 [R4]。k-path 工具、与参考本征值的对表工具(误差直方图按 (band, k) 分辨——归因用,见 §4.3)。

### L5 出口(v0.1 全部不做,接口形状受 A6 约束)

`projectors(site, l, window)`(Wannier/关联子空间)、`mpb()`(SPEX 序的下一站)、`thc_export()`、Green 出口、NMTO downfolding。

---

## 3. v0.1 的论题与被推迟的论题

draft-1 的两个架构赌注(B1:Envelope trait 同时承载 Hankel 与 PW;B2:双出口在同一数据上 meV 重合)依赖双 backend,在 LAPW-first 排序下**无法于 v0.1 内被证伪**。这是本次重排的真实代价,明说不藏。

v0.1 改为检验更基础的论题:

- **B0(数据论题)**:"快照 + 公共代数"足以从一个生产代码(FLEUR)序列化出的势,重建其能带到 meV。conventions 层的完备性、求积规则一等公民、增广=数据 这三件事同时受审。B0 是 SPEX 序全部后续(MPB、THC、GW)的地基;B0 不成立,B1/B2 无从谈起。
- **B1/B2** 顺延至 v0.4(LMTO/KKR 落地时),届时的验收沿用 draft-1:同一 Cu、同一结构常数、同一径向解,KKR 谱与 LMTO 带在线性化窗口内 meV 重合。Envelope 与 SecularForm 的 trait 边界在 v0.1 就按双实现设计,但承认:单实现期的 trait 是假设,不是验证过的抽象——v0.4 落地时允许其破裂重写,代码组织上不把公共层过度提前固化。

---

## 4. v0.1 范围

### 4.1 交付物

冻结势下的 LAPW 能带引擎,零自洽,零势构造:

1. `mt-core`(L0 全量)+ `mt-radial`(Schrödinger + KH,球对称势;含 LO 径向函数)。
2. `mt-lapw`:PlaneWave 包络、匹配系数、(H,S) 组装(球势 + warped 间隙必做;v_LM 通道为 stretch)、LO 支持、collinear 双自旋通道(两套势独立对角化,零新概念)。
3. `mt-io`:快照 schema 读写 + **`fleur2mtsnap` 转换器**(读 FLEUR 收敛势 → 快照)。
4. 验证一:**空晶格**(v=0,u_l ∝ j_l;LAPW 复现自由电子带,并验证线性化误差的已知标度律随 |E−E_ν| 收敛——这是 LAPW 独有的解析压力测试)。
5. 验证二:**fcc Cu 冻结势**:同一份 FLEUR 势、同一组基组参数(RK_max、l_max、E_ν、LO 集),libmuffintin 本征值 vs FLEUR 本征值逐 (band, k) 对表。
6. README 只放一张图:**Cu 能带,libmuffintin 与 FLEUR 的线叠在一起**。对目标受众,这张 cross-code 图就是全部信任状。`examples/cu/` 一条命令复现。
7. stretch(不进验收):自旋极化示例(bcc Fe 或 CrSb)——对本组现役磁性/altermagnet 工作流的直接演示;v_LM 通道打开后的 FP 级对表。

### 4.2 显式非目标

不做 SCF/Poisson/mixing/势构造(FLEUR 是 SCF 提供者——LAPW-first 排序下自研 SCF 的优先级大幅下沉,不再是 v0.2);不做 SOC(v0.2);不做 MPB/Coulomb(v0.2)与任何导出层(THC/CoQui 是 v0.3);不做 LMTO/KKR/EMTO/NMTO 与结构常数(v0.4);不做 Dirac 4c(schema 预留);不做 MPI/GPU;不做 CIF [D2];不做 FLEUR 之外的第二转换器(Elk 文本格式作三角测量,stretch)。

### 4.3 验收标准(写进 CI 的数值容差,校准后只紧不松)

| 项 | 测试 | 容差 |
|---|---|---|
| Gaunt | 对称性/选择定则 property tests;对符号计算参考表 | ≤1e-13 |
| Θ̃(G) | 解析式 vs 数值积分;截断约定自洽性 | ≤1e-12 |
| 径向(非相对论) | 类氢解析能级;方势阱对数导数解析式 | ≤1e-10 Ha |
| 径向(KH) | 对 FLEUR 打印的 u(R)、u̇(R)、能量参数逐项 | 对表 |
| 匹配 | 增广波函数在 R_MT 的值/斜率连续性(数值残差) | ≤1e-10 |
| 空晶格 | 自由电子带复现;线性化误差随 \|E−E_ν\| 的标度律拟合 | ≤1e-8 Ha(E_ν 处) |
| 收敛 | RK_max、l_max、G_max 扫描单调收敛(魔数禁令) | 曲线入库 |
| Cu | 本征值 vs FLEUR,价带窗口,逐 (band, k) | ≤1 meV,残差**归因** |

"归因"指:残差若超标,必须能指认到约定项(动能对称化、Θ̃ 截断、u 归一化、E_ν 读取)而非笼统"数值误差"——这本身就是 cross-code diff 杀手应用的第一次实战。

---

## 5. 数据模型与快照 schema

快照 = "收敛单粒子问题的序列化"(A1):

```
SnapshotV1
├── meta          schema 版本 / 单位 / 生成来源 (代码, 版本, 参数指纹)
├── geometry      晶格向量, sites(位置, Z, R_MT)
├── spheres[]     per-site, per-spin:
│   ├── mesh         log mesh (r0, Δ, N) + 求积规则标识        ← 一等公民 (A2)
│   ├── v_L[]        径向势通道; L=(0,0) 必有, v_LM 为可选追加 (A8)
│   ├── radial_eq    RadialEquation 标签
│   └── energies     E_ν(l) 表 + LO 能量/类型表
├── interstitial  per-spin: v_I(G) 系数 + G_max + step function 约定标识
└── basis_hint    可选: RK_max, l_max —— 复现参考计算用, 非快照本体
```

[D3] 载体:快照本体用**带版本头的纯文本**(TOML 头 + 列数据)——cross-code diff(WIEN2k/FLEUR/exciting 导出到同一格式后在基组层面 diff、all-electron GW 差异在数据层归因)是这个 spec 的杀手应用,人类可读可 diff 优先。`fleur2mtsnap` 读 FLEUR 的 HDF5(hdf5 依赖隔离在转换器 crate,核心不沾)。HDF5 快照载体随 v0.3 导出层进入,schema 同源。先存在,后标准;纪律是别长成委员会格式——定义就是"libmuffintin 的代数所消费的数据",一个参考实现一份 de facto spec。

---

## 6. Rust 工程

### 6.1 workspace 布局

```
muffintin/
  crates/
    mt-core         # L0
    mt-radial       # L1
    mt-lapw         # L2(PlaneWave) + L3(Lapw) + L4
    mt-io           # 快照读写
    mt-fleur        # fleur2mtsnap 转换器 (hdf5 依赖隔离于此)
    mt-cli          # 薄二进制: 快照 + k-path → 能带数据 / 对表报告
  examples/cu/      # FLEUR 势 (provenance 注明) + 脚本 + 参考本征值
  CONVENTIONS.md    # 单一真源
```

预留名(不建目录,防蠕变):`mt-envelope-hankel`、`mt-structconst`、`mt-mpb`、`mt-thc`、`muffintin-coqui`、`mt-green`、`mt-py`。

### 6.2 依赖决策

[D4] 线性代数:**faer**(纯 Rust,免 LAPACK 链接痛,cargo add 即用——playground 定位下安装摩擦是头等 UX);广义本征自制 Cholesky 约化 + 谱过滤。fallback:ndarray-linalg。特殊函数(球 Bessel 族:上/下行递推 + Wronskian property test)与 Θ̃ 自实现。错误处理 `thiserror`;核心 crate 禁 `unsafe`(FFI 例外仅限未来 `mt-py`/C ABI);f64 + `num-complex`。v0.1 单线程正确性优先;rayon(k 点平凡并行)零风险后补。

### 6.3 API 草图(kernel 文化)

```rust
let snap = Snapshot::read("examples/cu/cu.mtsnap")?;   // fleur2mtsnap 的产物
let rad  = RadialEngine::new(&snap, RadialEquation::ScalarKoellingHarmon);
let pws  = PwSet::new(&snap.geometry, RKmax(9.0));
let lapw = Lapw::assemble(&rad, &pws, &snap)?;          // → Linearized{H,S} per k
let bands = lapw.bands(&kpath)?;
let report = compare::against(&bands, FleurEig::read("cu_fleur.eig")?);  // 逐(band,k)归因
```

中间对象全部可劫持:换掉某个 l 通道的 u_l、改 E_ν、开关 v_LM 通道观察带的移动——受控实验是这个库存在的理由。

### 6.4 测试文化

property tests(proptest)覆盖 L0;解析测试覆盖 L1 与匹配连续性;收敛扫描测试取代一切魔数;golden files 覆盖 Cu 端到端;CI 从 M-A 起全绿。

---

## 7. 实现顺序(依赖序,PR 粒度,无日历)

| 步 | 内容 | 验收 |
|---|---|---|
| M-A | mt-core:conventions / harmonics / Gaunt / mesh / G 矢量 / Θ̃ | property suite 全绿 |
| M-B | mt-radial:Schrödinger + KH,u/u̇/LO/积分件 | §4.3 解析行 |
| M-C | mt-io + mt-fleur:schema 定稿 + FLEUR 转换器(格式风险,尽早并行) | 往返读写 + 对 FLEUR 打印量 |
| M-D | 匹配系数 + S 组装 + **空晶格** | 连续性残差 + 标度律 |
| M-E | H 组装(球势 + warped)+ **Cu vs FLEUR** | ≤1 meV + 归因报告 |
| M-F | LO 激活验证 + 双自旋 + stretch(v_LM / Fe 或 CrSb)+ 文档 + tag v0.1 | README 叠图 |

---

## 8. 风险登记

**[R1] FLEUR 势文件格式耦合**(内部格式,随版本漂移,文档薄)。缓解:转换器隔离为独立 crate、钉死一个 MaX release 并写进 meta、golden files;Elk 文本格式作为第二来源三角测量(stretch)。这是 v0.1 排期上最不可控的外部依赖,放 M-C 抢跑。

**[R2] LAPW 约定 folklore**:动能算符对称化、Θ̃ 双截断(G_max vs 2G_max)约定、u 的归一化约定、E_ν 语义(相对 MT zero 还是绝对)。缓解:CONVENTIONS.md 逐条钉死并注明"与 FLEUR 对齐";空晶格标度律测试对动能/匹配约定尤其敏感,是免费的约定探测器;Cu 归因流程(§4.3)兜底。

**[R3] 高 RK_max 下 S 的病态**(LAPW 基近线性相关)。缓解:Cholesky 失败时回退谱过滤 + 阈值入约定;把过滤计数作为诊断输出。

**[R4] B1/B2 延迟证伪的架构风险**(§3):单实现 trait 可能在 v0.4 撞墙。缓解:trait 面积最小化(增广=线性映射、SecularForm 双枚举先立),接受届时重写的可能;不提前抽象未被第二实现检验的公共层。

**[R5] 范围蠕变**(SOC/MPB/自研 SCF 提前渗入)。缓解:§4.2 清单进 README,PR 模板检查项。

**[R6] 单人 bus factor。** 缓解:CONVENTIONS.md + 组件独立可验证使外部贡献在任何一层可能;playground 定位本身是缓解。

---

## 9. v0.1 之后(SPEX 序,概要)

**v0.2 MPB + SOC + 投影子**:MT 内 per-center 解析 product basis(radial SVD + Gaunt,坐在 on-site rank 的信息论下界上)、IPW、multipole-matched Coulomb metric、q→0 head/wings——SPEX 的核心对象,验收对 SPEX 的 v 矩阵。二次变分 SOC。`projectors()`/wannier90 A_mn 导出(现役 Wannier 插值工作流)。

**v0.3 THC/CoQui 桥**:MPB 之上二次 ISDF 选点,block 结构只存在于构造期,对外摊平全局 THC(μ = MT log-grid 点 ∪ IR 点,单一 product rule),交付 basis-clean 四件套 (X_i^{kμ}, V_μν(q), h_ij, head/wings)(A6)。工程形态:schema-first、双 transport——HDF5 文件先行(golden-file 可测、天然 checkpoint、GRAFT/EDMFT 侧读同一文件),C ABI(libmuffintin 拥有并 semver)+ 贡献到 CoQui 树内的薄 C++ adapter 后补;thin Python wrapper 纪律:只做 schema 校验/路径拼装/作业提交,碰数组内容即越界;金丝雀 CI 钉 CoQui main。契约三硬点:外部 (X,V) 注入点、q→0 约定对齐、frozen core 进 h;首版只支持 one-shot/evGW/QP 自洽。

**v0.4 LMTO/KKR + 双出口**:SphericalHankel 进 Envelope(B1 摊牌)、结构常数(canonical S + 能量依赖 Ewald,原 draft-1 的 R1 数值风险随之回归)、M(E) 出口与 det-zero/围道、B2 的 meV 交叉验收(fcc Cu,同一快照)。Green 出口随后:杂化函数直出 DMFT、CPA(CrSb/MnTe 掺杂合金化不绕 Wannier)、NMTO downfolding 一等操作。

**未排期**:自研 SCF(FLEUR 在,不急)、EMTO/NMTO、FP 势构造(Weinert)、Dirac 4c。

---

## 10. 参照系

Questaal:所有正确的数学都在里面,焊死在 Fortran 单体——本库的定义就是"把它的公共代数拆成 trait"。SIRIUS:服务,不是代数。SPEX/FLEUR:v0.1–v0.3 的真值来源与对表对象,同时是本库要在数据层上超越的接口形态。libxc:命名公共对象的先例。PySCF:kernel 文化。UPF/PSML/PAW-XML:格式先存在、后标准的先例;PAW 的 partial waves 与 (L)MTO/(L)APW+lo 在类型层面是同一代数的方言,抽象做对了地基比 LAPW 一家大得多——不承诺,记录在案。

## 附:开放决策点汇总

| # | 决策 | 倾向 |
|---|---|---|
| D1 | 单位制 | 内部 Hartree(FLEUR/SPEX 对表),newtype 包裹,I/O 标注 |
| D2 | 结构输入 | 几何随快照来(FLEUR),独立 CIF 输入后补 |
| D3 | 快照载体 | 纯文本(可 diff);HDF5 随 v0.3;hdf5 依赖隔离在 mt-fleur |
| D4 | 线代后端 | faer + 自制 Cholesky 约化/谱过滤;fallback ndarray-linalg |
| D5 | 参考势来源 | FLEUR 转储(provenance 入 meta);Elk 三角测量为 stretch |
