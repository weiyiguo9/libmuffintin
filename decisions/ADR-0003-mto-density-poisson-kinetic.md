# design decision 02 — MTO 密度/Poisson/动能三轴与 v0.3 两线优先(已归档)

- Status: superseded 2026-08-27 by `plans/v0.3-mto-family/plan.v1.md` and `plans/v0.4-emto-nmto/plan.v1.md`; archived rationale
- Date: 2026-08-27
- Imported: 2026-09-08 from `scratch/design_decision_02_mto_poisson.md`, body unchanged

> **SUPERSEDED 2026-08-27**:本文件全部规范性内容已并入
> `libmuffintin_v0.3_mto_family_plan.md`(LMTO 路线)与
> `libmuffintin_v0.4_emto_nmto_plan.md`(EMTO/NMTO 路线),以两份 plan 为准;
> doc/18 落地后以 doc/18 为准。本文件仅作 rationale 归档保留,不再更新。
>
> | 原内容 | 现归宿 |
> | --- | --- |
> | 三轴与归宿、合法组合矩阵 | v0.3 plan §2.8 与 doc/18(M-M 前写成) |
> | `RadialJet` / `BoundaryData` adapter([D1]) | v0.3 plan §2.1 |
> | `SphereRadii`([D4];2026-08-27 复原:`density` 保留在初版,虽暂无 production 消费者) | v0.3 plan §2.1 |
> | v&d/USW 降级 test-only oracle | v0.3 plan M-N(可选、不阻塞) |
> | 两线优先、[D2] 双一中心展开、[D3] 谱微分动能 | v0.3 plan M-O/M-P |
> | 诊断进 versioned artifact | v0.3 plan M-N |
> | binary 冻结、basisopt opt-in bin、协议纪律([D6]/[D7]) | v0.3 plan 头部、§3、§6 |
> | `CoulombKernel`(BC × backend、共享 BC 硬规则) | v0.3 plan §2.8;backend 扩展 v0.4 plan §3.2 |
> | DMK 第一类路线、FFI PeriodicDMK、选型提示 | v0.4 plan §3 |
> | 三层正交栈、ISDF rank 预期 | v0.4 plan §3;v0.3 plan §2.8、M-P |
> | auxiliary/empty center 自动摆放 | v0.4 plan §4 |
> | EMTO/NMTO/M-T | v0.4 plan §2、§5 |

**状态**:决策记录,2026-08-24 收口([D1]–[D7] 全部已定);2026-08-27 归档。
**来源**:三则讨论(Nohara–Andersen v&d 与 USW 失效模式;DMK 与三种 recipe 的层级;kink 矩阵动能与 EMTO 复用),加 2026-08-21 拍板的 `ChargeRepresentation` / `CoulombRecipe`。
**关联**:`scratch/libmuffintin_v0.3_mto_family_plan.md`(下称 v0.3 plan)、`scratch/orbit_config.md`。
**性质**:过渡性固化。聊天记录不是可靠介质,规范性采纳最终进 doc/18;在 doc/18 写成之前,本文件是这些决定的唯一落地文本。

---

## 0. 一句话结论

三条轴(representation / poisson / differential)不新建任何顶层结构,分别落入既有归宿;类型级预留现在做,实现严格挂在消费者上;v0.3 的实现优先级定为 three-component(M-O)与 smooth-Hankel(M-P)两条线先行,v&d/AnalyticUsw、NMTO 全量、EMTO、DMK 全部不插队。

**追加决定(2026-08-24 晚)**:v0.3 **不默认分发任何 binary**;现有 v0.2 `muffintin` binary 与 `mt-runtime` TOML input 冻结在 v0.2 范围。v0.3 正式发布 Rust spec/algorithm crates、`mt-recipes` preset 与 `libmuffintin-basisopt` crate;后者内带 opt-in `[[bin]]` target——不进默认发行物,用户 `cargo install`/自行构建取用,是 v0.3 唯一的可执行物。

**追加决定(2026-08-24,三)**:`SphericalWaveVd` + `AnalyticUsw` 不做 production 本体,降级为固定参数的 test-only oracle(§6.2);从公开 spec 词汇表与 doc/18 矩阵中移除。

**追加决定(2026-08-24,四)**:representation/poisson 层保留 **both**——three-component + FFT 与 adaptive-grid + DMK 均为第一类路线;DMK 不再是触发器守门的 fallback(§6.1 再修订)。调度不变:v0.3 仍只做两线,DMK 路线在 post-v0.3。用户在纯 Rust或实验性的薄 Rust/Python 绑定层自行组装。文件级输入随 [D7] 的 binary interface 定型再采纳。

---

## 1. 物理输入摘要

推导细节见讨论记录,此处只留结论,作为后续 doc/18 的依据:

- **v&d(Nohara–Andersen 2016)的失效模式**:USW 衰减尺度由硬球 interstitial 最低齐次本征值 $\varepsilon_{\mathrm{hom}}$ 控制,开放结构衰减慢($10^{-3}$ 精度下 bcc 需 $N_R \approx 59$,diamond 需 $\approx 159$);大 void 中球面值加前三阶径向导数不唯一决定内部,属不适定,需第五能量、电子数、void 中心值等附加约束;$\varepsilon \to \varepsilon_{\mathrm{hom}}$ 时 screened structure matrix 病态、局域化长度发散;窄通道需更高 $l_{\max}$;物理 $1/r$ 尾巴仍需 multipole/Ewald/FMM 单独处理。对致密晶体可靠,对开放/低对称/大 void 结构不是无参数黑盒。
- **动能不经密度插值**:MTO 家族动能走波动方程与 kink 矩阵——interstitial 度量 $O_I = a\dot S$,参考 Hamiltonian $H_{\mathrm{MT}} = \varepsilon\dot K - K$;EMTO 已在用($\dot K^a$ 即完整 orbital 度量)。"v&d 不适用于微分性质"仅指:不要对球面数据插值出的三维密度求导。
- **DMK 是 Poisson/kernel backend,不是通用微分 backend**:输入是全网格体密度,不是球面 Dirichlet/Neumann 数据;仅 Nohara 边界数据交给 DMK 不闭合,开放 void 仍需 interior samples 或全局约束或显式 PDE-extension。
- **EMTO 复用**:$(K,\dot K)$ 动能 trick 是 EMTO 既有核心;v&d 可另行升级其 interstitial density/Poisson(SCA/FCD 的替换项),二者互补不重复。density 插值需要共同几何边界,不能用随 $l$ 不同的 $a_{Rl}$,须另设非重叠 density 硬球 $a_R^\rho$。

---

## 2. 三轴与归宿

| 讨论中的轴 | 归宿 | 新增内容 |
| --- | --- | --- |
| representation:`analytic_vd` / `boundary_seeded_grid` / `three_component` | `ChargeRepresentation`(2026-08-21 拍板:MtInterstitial / ThreeComponent / Grid) | ~~新变体 `SphericalWaveVd`~~(2026-08-24 降级 test-only,不进词汇表,§6.2);`Grid` 加 seeding 策略字段 |
| poisson:`analytic_usw` / `fft` / `dmk_free` / `dmk_periodic` | `CoulombRecipe { sphere, kernel }`(2026-08-27 重构为 BC × backend,§5.5) | ~~smooth 第三变体 `AnalyticUsw`~~(随 v&d 降级移除,§6.2);Dmk 为 backend 变体之一,不锁死 |
| differential:`wave_equation_identity` / `weak_grid` / `radial_augmented` | operators 层既有路线(kink/$\partial_E K$、M-O double augmentation、`EnvelopeKernel::source_term`) | 仅预留 `WeakGrid` 变体名 |

三轴**不合并**。非法组合(如 `AnalyticUsw` 要求 `SphericalWaveVd`,后者要求非重叠硬球)由 v0.3 plan §4 的 capability validation 在 compile 期拒绝,类型系统保持轴分离。

另一处依赖纪律:v&d 闭包消费 screened slope matrix,但只通过带 `StructureConvention` 标记的数据接口取 $S(\epsilon_n)$,不 import screening 内部;依赖方向保持单向。

---

## 3. 对既定 staging 方案的三处修正

### 3.1 修正一:固化介质,doc/18 是第一项

已核对仓库:`ChargeRepresentation`、`CoulombRecipe`、`RadialJet`、`SphereRadii`、`EnvelopeKernel` 均只存在于 v0.3 plan 与聊天记录,代码停在 M-A..M-Kb。因此"合法组合矩阵写进 doc/18"不是清单中的一项,而是**第一项**:M-M 开工前,先把 §2、§4、§5 的表固化成 numbered doc,类型预留才有可引用的规范。本文件是其底稿。

### 3.2 修正二:`BoundaryJet` 改为独立任意阶 `RadialJet`,在 M-M 落地前改 plan

v0.3 plan §2.1 原定义径向只有一阶,并把它与轨道边界数据混名:

```rust
struct BoundaryJet<T> {
    value: T,
    radial_derivative: T,
    energy_derivative: Option<Box<BoundaryJet<T>>>,
}
```

两点观察:

1. v&d 需要的是**密度/场投影**在硬球上的值加前三阶径向导数;而 v0.2 的 `BoundaryData {value, derivative, log_derivative}`(`mt-radial/src/valence.rs`)是**轨道匹配数据**。两者是不同对象、不同消费者,不应通过原地改写前者来服务后者。
2. M-M 尚未实现,这类任意阶 jet 还不存在于代码。现在把方法无关值类型放在 `mt-core`,由 `mt-radial` 等生产者构造,是零成本、非 breaking 的 plan 修订:

```rust
pub struct RadialJet<T> {
    /// radial[k] = d^k f / dr^k at the sphere radius; radial[0] is the value.
    radial: Vec<T>, // len >= 1
    energy_derivative: Option<Box<RadialJet<T>>>,
}
```

`BoundaryData` 保留原结构,只作为从至少含零阶与一阶项的 `RadialJet<f64>` 加球半径生成的一阶 adapter;它不包住 jet,APW matching 路径一行不改。对数导数、Wronskian、slope 一律由 jet 派生(plan §2.1 原则不变)。这把原判断中"唯一真正紧急的 breaking 决定"降级为"M-M 落地前的 plan 文本修订"。Rust 内存表示与持久化 DTO 分开决定(见 [D1])。

### 3.3 修正三:诊断进 versioned artifact,不只是运行时监控

$\varepsilon_{\mathrm{hom}}$ 估计、screened 矩阵条件数、自适应 $N_R$ 与 $l_{\max}$ 的最终取值,必须写进 manifest/快照,退化显式可追溯——与 "make DFT mixing degradation explicit"(`710c946`)同一纪律。挂接点:M-N 的 real-space localization/decay report 扩成诊断组;失败模式接 v0.3 plan §9。

---

## 4. 合法组合矩阵(doc/18 底稿)

> 2026-08-24 修订:spherical-wave-vd 行与 analytic-usw 列随 §6.2 降级决定**不进 doc/18**;下表保留原状仅供 test oracle 场景参考。

representation × smooth Poisson:

| representation \ smooth | analytic-usw | fft | dmk(预留) |
| --- | --- | --- | --- |
| mt-interstitial | ✗ | ✓ | ✗ |
| spherical-wave-vd | ✓ | ✓(投影后) | ✓ |
| three-component | ✗ | ✓(Gaussian 补偿) | ✓ |
| grid | ✗ | ✓ | ✓ |

representation × kinetic route:

| representation \ kinetic | wave-equation-identity | radial-augmented | weak-grid |
| --- | --- | --- | --- |
| mt-interstitial | ✓(kink) | ✓ | ✗ |
| spherical-wave-vd | ✓(基函数层面) | ✓ | ✗(不得对插值密度求导) |
| three-component | ✓(source 恒等式) | ✓(球内) | ✓(smooth 部分谱形式) |
| grid | ✗ | ✗ | ✓ |

几何前置条件:

- `SphericalWaveVd` ⇒ 非重叠 density 硬球 $a_R^\rho$(仅 test-only 场景相关;`density` 角色不进初版 `SphereRadii`,见 §6.2);
- EMTO 的重叠 potential spheres 与非重叠 auxiliary augmentation/density 几何**分离**,否则 onsite corrections 在重叠区重复计数;
- Nohara 与现有 double augmentation 均假设非重叠 hard/augmentation spheres。

---

## 5. 最终 API/config 草图

三档入口,与 doc/12 的 facade/explicit-spec 双路同构:**preset 一行 → preset + 覆盖 → 完整显式 spec**。

**采纳状态(2026-08-24 追加决定)**:v0.3 只交付 §5.2(Rust spec 层)与 §5.4(recipe 层);前两档入口经纯 Rust API或实验性的薄 Rust/Python 绑定提供。§5.1 的 TOML 面是**将来** binary interface 的形状草图,不随 v0.3 交付,`mt-runtime/src/input.rs` 维持 v0.2 范围不动。采纳时点见 [D7]。

### 5.1 TOML 面(推迟;binary interface 定型后采纳,风格延续 `input.rs` 的 kebab-case + `kind` tag)

```toml
[task.scf]
kind = "dft-scf"            # 或未来 mto/emto task;现有字段不动
# ...

[task.scf.density]
representation = "three-component"     # mt-interstitial | grid

[task.scf.coulomb]
sphere = "weinert"

[task.scf.coulomb.kernel]
periodicity = "ppp"                    # ppp | ppo | poo | ooo
backend = "spectral"                   # dmk | spectral-ewald | fmm(预留)
# open 方向另挂 environment(vacuum | metallic-gate | dielectric-half-space …)与 zero-mode 策略

[task.scf.kinetic]
route = "radial-augmented"             # wave-equation-identity | weak-grid(预留)

[task.scf.diagnostics]
condition-limit = 1e12
on-degrade = "error"                   # 或 record-and-continue;结果一律进 manifest
```

### 5.2 Rust 面

enum 本体放消费者 crate 的 spec 层(`mt-dft` / `mt-coulomb`)。v0.3 不新增 runtime DTO;未来 [D7] 采纳文件级接口时,`input.rs` 只做这些 spec 的 DTO 镜像——与现在 `Basis` 引 `ChannelEnergyGenerator` 的关系一致:

```rust
#[non_exhaustive]
pub enum ChargeRepresentation {
    MtInterstitial,
    ThreeComponent,
    Grid(GridSeeding),          // boundary-seeded 是它的一个字段,不是新概念
}

// 2026-08-27 重构(§5.5):smooth 单枚举 → Coulomb kernel 层,
// BC 与 solver 正交,支持关系是 capability matrix,不是类型层级。
pub struct CoulombRecipe {
    pub sphere: SpherePoisson,     // Weinert(现有,MT 两区路线)
    pub kernel: CoulombKernel,
}

/// V(r) = ∫ v(r,r'; BC) ρ(r') dr'。数学上是 -Δ/4π 的 Green function,
/// 命名取 kernel,避免与电子/KKR Green function 混淆(§5.5)。
pub struct CoulombKernel {
    pub bc: CoulombBoundaryCondition,
    pub backend: CoulombBackend, // #[non_exhaustive] Spectral | Dmk | SpectralEwald | Fmm
}

/// 逐方向周期性 + open 方向环境 + 零模策略;BC 不是一个 dim 整数。
pub struct CoulombBoundaryCondition {
    pub periodicity: [Axis; 3],            // P | O → PPP / PPO / POO / OOO
    pub environment: OpenEnvironment,      // #[non_exhaustive] Vacuum | Dirichlet | MetallicGate
                                           //   | DielectricHalfSpace | TwoDielectricHalfSpaces
    pub zero_mode: ZeroModePolicy,         // G∥=0:Neutral | Dipolar | Charged | GateCompensated
}

#[non_exhaustive]
pub enum KineticRoute { WaveEquationIdentity, RadialAugmented, WeakGrid }
```

### 5.3 两条关键机制

1. **v0.3 只承诺 Rust 层"可构造、不可执行"**:`DmkPeriodic`、`WeakGrid` 等 `#[non_exhaustive]` enum 变体可以进入 spec,但 `validate()` / compile 阶段返回带指引的 typed error(风格同 `BasisError::UnknownSite`)。serde 的"可解析、不可执行"及具体 token 拼写随 §5.1 一起推迟到 [D7]。
2. **合法性走 compile 期 capability 校验**(v0.3 plan §4),违例报 typed error,例如 `UswRequiresVd`、`VdRequiresHardSpheres { site, overlap }`、`KineticRouteUnsupported { representation, route }`。

### 5.4 recipe 层

`mt-recipes` 加 preset 函数,产出上述 spec 组合,与 `recipes::lapw()` 同构:`fp_mto()`(ThreeComponent + Fft,production)。(原设想的 `nohara_andersen()` preset 随 §6.2 降级取消。)

### 5.5 Coulomb kernel 层与三层正交栈(2026-08-27)

**形式化**:Coulomb 层的对象是 BC 参数化的相互作用核,而非 "periodic Poisson solver":

```math
V(\mathbf r) = \int v(\mathbf r,\mathbf r';\mathrm{BC})\,\rho(\mathbf r')\,d^3 r' .
```

**命名(2026-08-27 定)**:类型名取 `CoulombKernel`。数学上 $v(\mathbf r,\mathbf r';\mathrm{BC})$ 就是 $-\Delta/4\pi$ 在该 BC 下的 Green function,但本库的 "Green function" 一词已被电子传播子占据(M-S 轮廓 GF、半无限表面 GF、NMTO 的 $g(z)=K^{-1}(z)$),`CoulombGreenFunction` 必然混淆;"kernel" 恰是截断库仑文献的标准用词(truncated Coulomb kernel),且与既有 `InjectedCoulombGram` 同族(kernel = $v$,gram = $\zeta^\dagger v\,\zeta$)。它把 kernel 与"3D 周期边界"拆开;BC 与 solver 正交,支持关系是 capability matrix。

**BC 不是一个 dim 整数**。逐方向周期性 `PPP / PPO / POO / OOO`,open 方向再挂环境(Vacuum / Dirichlet / MetallicGate / DielectricHalfSpace / TwoDielectricHalfSpaces——monolayer on substrate、gated 2D、电化学 slab 都是"2D periodic",但 $v(\mathbf r,\mathbf r')$ 完全不同,ESM 即此类),外加 $G_\parallel=0$ 零模策略(neutral / dipolar / charged / gate-compensated slab 各对应不同渐近条件)。monolayer 的三个等级:

1. 3D periodic + vacuum supercell(传统:15–30 Å 真空,image 相互作用收敛慢);
2. truncated Coulomb(Ismail-Beigi;Rozzi–Varsano–Marini–Gross–Rubio:仍有 vacuum cell,切掉 image);
3. **native PPO kernel(本库指定支持的形态)**:面内 Fourier + open 方向一维核

```math
v_{G_\parallel}(z,z') = \frac{2\pi}{G_\parallel}\, e^{-G_\parallel |z-z'|} ,
```

   Coulomb 层根本没有 vacuum thickness 概念,无 $L_z$ 收敛测试、无镜像层。

**一致性硬规则**:每个计算持有**一个** Coulomb BC 对象,Hartree、exact exchange、RPA/GW/BSE 的全部 kernel 由它派生。禁止 DFT Hartree 已无 image、到 $W = v + vPW$ 时又偷偷换回 $4\pi/G^2$ 的 3D 周期核。

**MTO 协同**:localized 基不需要用基函数表示真空——surface/monolayer 成为真正的 reduced-boundary 计算(atoms + interstitial around layer + $v^{\mathrm{PPO}}$),而非 30 Å 假晶体。KKR/LMTO 的半无限表面 Green function(80–90 年代)是同一架构思想的先例(注意那是电子 GF,与本节的 Coulomb kernel 是不同对象——这正是命名避开 "Green function" 的原因)。

**三层正交栈(post-HF 目标)**:

```text
OneParticleBasis   = NMTO + HELO + HDLO(通道层,已有)
PairRepresentation = ISDF / THC / RI / Cholesky(mt-auxiliary-ir / mt-thc,已有)
CoulombKernel      = BC × backend(DMK / FMM / SpectralEwald / Spectral)
```

低秩 ERI 链条:$\phi_p\phi_q \xrightarrow{\mathrm{ISDF}} \sum_\mu C^\mu_{pq}\zeta_\mu(\mathbf r)$,$(pq|rs) \approx \sum_{\mu\nu} C^\mu_{pq} V_{\mu\nu} C^\nu_{rs}$,其中 $V_{\mu\nu} = \iint \zeta_\mu(\mathbf r)\, v(\mathbf r,\mathbf r')\, \zeta_\nu(\mathbf r')\,d^3r\,d^3r'$。**这个 $V_{\mu\nu}$ 正是 `mt-thc` 的 `InjectedCoulombGram` seam**——三层正交在代码里已结构性存在,新增内容只是 gram 的构造统一经由 BC 对象。

**可检验预期(非断言)**:MTO 轨道 localized 且带 MT/interstitial 分块,pair density 比 PW orbital products 更有结构,ISDF rank 有望不劣于甚至低于 PW;在 M-O/M-P 的 THC 验收中直接测量。

**capability matrix 初稿(进 doc/18)**:Spectral(FFT)→ PPP;quasi-2D spectral(上式解析核)→ PPO;Dmk → PPP(PeriodicDMK,Bloch)与 OOO(free DMK),**PPO-native DMK 目前不存在**,PPO 首先经解析核落地;SpectralEwald / Fmm 预留。v0.3 两线即 PPP × Spectral,已在新形状内,无迁移成本。channels token DSL 不动——它管径向通道生成;三轴管密度/静电/动能,正交挂在同一 normalized IR(`ChannelRecipeArtifact` 同款模式)里,带 provenance。

---

## 6. v0.3 优先级决定:three-component 与 smooth-Hankel 两线先行

**决定**:v0.3 实现关键路径 = M-M → M-N → **M-O(three-component / FP-TB-LMTO preset)** → **M-P(smooth-Hankel JPO 与 PMT preset)**。以下全部不插队:

- v&d + `AnalyticUsw`:**不做 production 本体**(2026-08-24 降级为 test-only oracle,见 §6.2);EMTO FCD 走传统 SCA/FCD 路线;
- NMTO divided differences 全量机器:随 M-R;M-M 只保留结构矩阵 $\partial_E S$ 所需的最小集;
- EMTO:M-S 原位;
- DMK:**升格为保留的第一类路线**(§6.1 再修订),但调度不变——不进 v0.3;post-v0.3 与 adaptive-grid 表示层一起落地,放独立 feature-gated crate,实现走 FFI PeriodicDMK,不自研(§6.1)。

**理由**:两线是 production 路线,风险集中在已知代数(augmentation、source 恒等式)而非不适定问题;两线验收都含 MPB/THC 产物(v0.3 plan M-O、M-P 条目),直接服务 CoQuí THC 目标;v&d 的病态区(开放结构、$\varepsilon_{\mathrm{hom}}$ 邻域)完全避开。

**两线实际需要的部件清单**:

- envelope:`HelmholtzKernel` + `SmoothHankelKernel`(v0.3 plan §2.2),smooth-Hankel Gaussian source 因子对 Helmholtz 恒等式直接单测;
- Poisson:sphere = Weinert(现有),smooth = **Fft + Gaussian 补偿电荷——两线唯一需要的 smooth backend**;`Dmk` 仅保留 Rust spec 变体,不在 v0.3 固化 serde token;`AnalyticUsw` 随 v&d 降级移出词汇表(§6.2);
- kinetic:M-O 的 smooth 部分谱微分/倒空间、球内 radial-augmented;M-P 用 smooth-Hankel source 恒等式;kink 路线随 M-R/M-S 到位;`WeakGrid` 一般形式(任意数值 envelope 的 weak form)仍在 v0.3 之后;
- 双用途前移维持:M-M 加 $\langle\psi(\epsilon_1)|\psi(\epsilon_2)\rangle = a S_{12}$ Green 恒等式验收(结构常数 + screening 的免费交叉检验);M-N 的 decay/条件数诊断组(对 M-O/M-P 的 screened 表象同样有用)。

**反向约束(维持)**:v&d 的 Schur complement 组装、约束系统、Poisson 装配只有 v&d 自己用,在 EMTO workflow 落地之前不写。

### 6.1 DMK 冻结复核(2026-08-24,已对代码核实)

- **唯一的 backend 形状泄漏点**:`CoulombRequest`(`mt-coulomb/src/spec.rs`)携带 Weinert 形状参数——`lexp` 加 `Option<InterpolationProjection { pw_cutoff, l_max }>`(比先前判断略宽:`lexp` 本身也是 Weinert/SPEX 特有,不只 `InterpolationProjection` 两参数)。DMK 的对应参数会是精度目标/树深。**不构成现在动手的理由**:已有多入口先例(`assemble_coulomb` / `assemble_sampled_coulomb` / `assemble_point_charge_oracle`),将来加新入口或把请求参数收进 non_exhaustive 的 per-backend 变体是局部改动,不触及消费者。
- **THC seam 连预留都不需要**:`mt-thc` 只消费 `InjectedCoulombGram`(带 Hermitian/PSD 容差校验),对 gram 的来源不可见。此句进 doc/18 的 DMK 条目,作为"槽位在 `CoulombRecipe.smooth`、不在 THC"的依据。
- **冻结理由在周期晶体范围内成立**:MT 两区几何下 $\zeta$ 的球内投影解析、间隙 FFT 为 $N\log N$,DMK 无优势可提供。~~periodic k-point DMK 在文献中尚不存在(Zhu et al. 2510.20826 为自由边界)~~ **2026-08-24 更正**:[PeriodicDMK](https://github.com/xuanzhaogao/PeriodicDMK)(Jiang–Gao,Flatiron,2026-06,MIT)已实现任意三斜胞的周期 Coulomb 格和,含可选 Bloch 相位的 quasi-periodic `evaluate_complex`,带 C ABI(`pdmk_capi.h`)与 Julia 绑定。"算法不存在"这道门槛已消失——但这只压低解冻后的实现成本,不改变解冻条件本身:可得性不产生需求。

**解冻条件**(全部在 v0.3 范围外,写成可检验触发器,进 doc/18):

1. 非周期/低维边界条件(分子、真 2D slab,FFT 周期镜像失效)。注意它触发的是"需要非周期 Poisson backend",DMK 只是候选之一——截断核/加厚真空 FFT 是更廉价的备选,触发后仍需选型;
2. 无两区结构的基(GTO/NAO)——plan §0 明文 out of scope;
3. open 结构下间隙 PW 截断爆炸——此条弱于直觉:ISDF 的 $\zeta$ 是插值函数,光滑度由格点密度控制,与基函数尖锐度解耦;
4. 放弃 three-component 的 FP 路线(如纯 MTO 表示的 FP-NMTO):真实 envelope 在球面附近不光滑,均匀网格 FFT 不收敛,必须 adaptive grid,而 adaptive grid 上的 Poisson 正是 DMK 的本行。FP 的三选一(three-component / v&d / adaptive-grid+DMK)是完备的;v0.3 已把 three-component 定为 production FP 表示、v&d 降为 test-only(§6.2),故此触发器只在**有意绕开 three-component** 时点亮。NMTO 本身不触发:M-R 的 NMTO 是 transform 层,骑在 three-component 之上,与 representation 轴正交。

**解冻后的实现路径(2026-08-24 已定)**:不自研,直接 FFI PeriodicDMK——bindgen 包 `pdmk_capi.h`,放独立 feature-gated crate(命名按仓库惯例 `crates/mt-pdmk` / `libmuffintin-pdmk` / `muffintin_pdmk`),core 不依赖。届时剩余工作只有三件:(a) 连续密度 → 求积权重点电荷的 adapter(上游接口是点电荷格和,参照 Zhu et al. 2510.20826 的 adaptive-grid 构造);(b) finite-$q$ quasi-periodic 路径对 Ewald 的交叉验证(上游测试自带 DUCC Ewald 参考,`mt-coulomb` 另有自己的 `ewald.rs`,两头都能对);(c) 构建隔离——上游烘死 `-march=native`、configure 期需联网 FetchContent,只能源码构建,永不进默认 workspace 依赖。

**再修订(2026-08-24,地位变更):DMK 升格为保留的第一类路线,与 three-component 并存**

- **决定**:representation/poisson 层保留 both——three-component + FFT 与 adaptive-grid + DMK。上列"解冻条件"1–4 不再是守门条件,降为**选型提示**(标注何种问题上 DMK 相对 FFT 有净优势);doc/18 相应改写。
- **理由**:(a) Poisson 终究是数值问题——three-component 的 smooth/onsite 分解不是物理,是让均匀网格 FFT 可用的记账;当数值方法能在自适应网格上直接解真实密度时,不必费力构造解析合适的结构(smooth 对应物、$P_{kL}$、补偿电荷)。(b) 历史判断:Nohara/Andersen 时代的解析巧思(USW screening、v&d、double augmentation)在相当程度上是在补偿当年不存在的快速自适应求解器;DMK/PeriodicDMK 诞生后,"真实函数 + 自适应网格 + 快速求解"是对这个问题最直接的回答。(c) 与项目终点同向:Zhu et al. 2510.20826 即 adaptive-grid ISDF + DMK,与 CoQuí THC 目标同路;DMK 路线需要的自适应网格表示层与 adaptive-grid ISDF 共用,一份投入两处消费。
- **认识论分工(保留 both 的理由,非冗余)**:three-component 锚定已发表工作流——M-O/M-P 的回归验收都靠它;DMK 路线没有文献参考数字,其正确性由 three-component 在相同体系上交叉验证。前者是对外的锚,后者是向前的路。
- **配套**:adaptive-grid 表示的动能走 differential 轴的 `weak_grid`(weak form + 高阶局部基,不做谱微分),随 DMK 路线一起到位。DMK 路线的真实成本不在 Poisson 求解(已 FFI)而在自适应网格的表示/求积层——该层与 adaptive-grid ISDF 共享,见 (c)。
- **调度**:v0.3 关键路径不变(两线,three-component + FFT);DMK 路线是 post-v0.3 的**指定方向**,不是条件分支。

### 6.2 v&d/USW 降级为 test-only oracle(2026-08-24)

**决定**:`SphericalWaveVd` 与 `AnalyticUsw` 不做 production 本体,从公开 spec 词汇表移除——`ChargeRepresentation` 与 `CoulombRecipe.smooth` 均无此变体,无 `nohara_andersen()` preset,doc/18 矩阵不含对应行列。v&d 仅以 **test oracle** 形式存在:固定参数($N_R$、$l_{\max}$、能量网格取 Nohara 2016 发表值)、只跑致密良态 fixture(bcc/diamond 间隙常数密度,可选 Si/ZnSe/CuBr)、无自适应、无 Voronoi/void 约束、无诊断策略;代码住在 tests/dev-only harness,不进公开 API。

**用途**:对 M-M/M-N 的多能量 screened 结构常数、硬球边界 screening、`RadialJet` 边界数据做一条独立路径的正确性交叉检验——密度重构对上发表数字,即证明这几台机器没写错。定位为**可选检验,不设为验收门槛**;M-M 既有的 $aS_{12}$ 恒等式与 free-electron/empty-lattice kink 测试仍是必过项。附带价值:Nohara 的原始代码只存在于个人站点的冻结归档,这个 test harness 会是该算法少数可跑、带 fixture 的现代公开实现之一——test 内重型 oracle 兼作事实参考实现,是既有先例的延续(如 Graft.jl test 中的 Jordan–Wigner ED)。

**连带修订**:

- [D4]:`density: Option<f64>` 失去 production 消费者,不进初版 `SphereRadii`;test harness 自带半径;builder + `#[non_exhaustive]` 保证将来加回非 breaking;
- §4 矩阵中 spherical-wave-vd 行与 analytic-usw 列不进 doc/18,原表保留仅供 test oracle 场景参考;
- EMTO(M-S)的 interstitial density/Poisson 走传统 SCA/FCD,不做 v&d 升级;对公开 EMTO 代码的 elemental benchmark 交叉验证(plan M-S 原文)照旧。

**成本说明**:即使 test-only,多能量边界拟合的线性代数核心也要写一次,只是免掉了全部 production 化(自适应、约束生成、诊断策略、配置面、void 处理)。若连这个有界求解器也嫌多,$aS_{12}$ + kink 测试已覆盖结构常数正确性的大部分,oracle 可以无限期搁置——因此标为可选、不阻塞验收。

### 6.3 全局 Poisson 弱化 empty sphere 必要性;auxiliary center 自动化(2026-08-27)

DMK 类 backend 全局求解 $V = 4\pi(-\Delta)^{-1}\rho$ 之后,不再需要靠空球把空间硬切成 space-filling cells——空球只在 density/potential 的 **local representation 不够好**时才值得加。因此本库的 FP-NMTO 几何定为:

```text
real atomic centers + optional auxiliary/empty centers
```

而非默认要求 space-filling empty spheres。摆放可以自动化:在 octree 上看 OMT reconstruction residual

```math
\epsilon(\mathbf r) = \left|\rho_{\mathrm{tree}}(\mathbf r) - \rho_{\mathrm{OMT}}(\mathbf r)\right| ,
```

在最大 void/residual 区域插入 auxiliary center——比传统 KKR 靠经验摆 empty spheres 现代,且天然是 M-Q `basisopt` 式的搜索问题(离散摆放 + 半径/通道参数),可挂其 feasibility/多保真框架。

界限说明:此处改变的是 **density/Poisson 侧**的几何要求;kink/screening 的硬球几何是另一层对象,不因此改变。auxiliary center 由此从"几何必需品"降为"局部表示的收敛旋钮",与 §6.1 再修订(DMK 第一类路线)和 §5.5(Coulomb kernel 层)构成同一组设计。

---

## 7. 现在就做的类型级动作(零数值代码)

1. 按 §3.2 把 plan §2.1 的 `BoundaryJet` 改成 `mt-core::RadialJet`(M-M 落地前);
2. `SphereRadii` 首次公开即采用 `#[non_exhaustive]` + builder;`density` 角色随 v&d 降级不进初版(见 §6.2 与 [D4] 修订);
3. 上述所有 enum 打 `#[non_exhaustive]`,预留变体在 Rust 层走"可构造、不可执行";
4. doc/18 立项:§4 的两张合法性矩阵 + 几何前置条件为骨架;
5. doc/18 只固化 density/poisson/kinetic 的概念、Rust 变体语义与 capability matrix;serde token 拼写、orbit_config/runtime 接线均随 [D7] 的 binary interface 采纳,不在 v0.3。

---

## 8. 待决事项

**收口(2026-08-24)**:[D1]–[D7] 全部已定,v0.3 范围内无未决项。其余开放工作(adaptive-grid + DMK 路线落地及其 `WeakGrid` 动能、binary interface 采纳)全部在 post-v0.3,已在 §6.1/[D7] 记录;v&d + `AnalyticUsw` 已于同日降级为 test-only oracle(§6.2),不再列为待实现项。下一动作:doc/18。

- **[D1] `RadialJet` 内存表示与持久化边界——已定**:M-M 引入 `mt-core::RadialJet<T> { radial: Vec<T>, energy_derivative: Option<Box<RadialJet<T>>> }`,`radial` 非空且第 $k$ 项为 $d^k f/dr^k$;`mt-radial::BoundaryData` 仅在至少有前两项时结合球半径生成,不改变 APW matching。mt-io 不直接 serde runtime 类型,也不扩写 Snapshot V1/V2;首次确有 artifact 消费者时定义独立 versioned DTO,wire 字段固定为按阶数索引的 `radial_derivatives: Vec<f64>`及可选的同形 energy-derivative jet,校验非空、有限值与阶数语义。量纲由包含该 DTO 的 field/channel schema 声明:第 $k$ 项的单位是 quantity unit 乘 length unit 的 $-k$ 次方。
- **[D2] M-O 中 smooth envelope 的球内一中心展开——已定(2026-08-24)**:双实现同时做——数值径向投影(v0.2 grids 与 sphere algebra)与 Questaal 式 $P_{kL}$ 多项式解析并行落地,在 v0.3 plan §8 validation matrix 互检。代价是 $P_{kL}$ 解析代数进入 M-O 工作量;收益是 M-O 自带一组内部交叉检验,不依赖外部参考数据。
- **[D3] M-O smooth 部分动能——已定(2026-08-24)**:网格谱微分(与 grid 表象一致、可推广到任意 smooth envelope);M-P 走 source 恒等式,两线互为交叉检验。
- **[D4] `SphereRadii` 扩展纪律——已定,2026-08-24 修订**:首次公开定义即采用 `#[non_exhaustive]` + builder。`density: Option<f64>` 随 v&d 降级(§6.2)失去 production 消费者,**不进初版**;test-only harness 自带半径;builder + `#[non_exhaustive]` 保证将来需要时加回非 breaking。
- **[D5] doc/18 采纳时点——已定(2026-08-24)**:M-M 开工前写成;本文件届时降级为历史记录。
- **[D6] runtime input 版本策略**:已被 2026-08-24 追加决定覆盖——v0.3 期间无任何新块进 runtime input,`INPUT_VERSION` 不动;版本与迁移策略并入 [D7]。
- **[D7] binary interface 的采纳时点与形状**:v0.3 不默认分发任何 binary,正式发布 Rust 组件/spec/recipe crates,包括 `libmuffintin-basisopt`(内带 opt-in `[[bin]]`,不进默认发行物);主 DFT 文件级输入冻结,直到想清楚 binary interface 为止。**basisopt 协议纪律**:其输入/cache/provenance 是 v0.3 唯一的 file-level 契约,且序列化的恰是被挡在主 TOML 之外的 MTO 参数($E_1$、$E_2$、HCR、$r_{\mathrm{smh}}$ 等);必须独立版本化、标 experimental,并明确对本条**非规范**——将来主 TOML 的 token 重新设计,不继承 basisopt 的 wire 格式。建议触发条件:M-P 之后、两条 production 线的 spec 在 Rust 层用稳、且出现第一个文件级输入的真实需求;届时启用 §5.1 草图 + "可解析、不可执行"机制,一并定 token 拼写、`INPUT_VERSION` 递增与迁移路径。实验性薄 Rust/Python 绑定可先行,但在模块 docstring/README 明示不承诺 API 稳定;它不构成 v0.3 默认发行界面。
