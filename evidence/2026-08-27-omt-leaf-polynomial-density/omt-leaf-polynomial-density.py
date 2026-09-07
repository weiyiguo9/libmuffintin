#!/usr/bin/env python3
"""Generate scratch/omt-leaf-polynomial-density.svg.

OMT continuous density -> adaptive-octree leaf polynomials -> volume DMK.
All text (incl. STIX mathtext) is emitted as SVG paths, so the figure
renders identically in any viewer without font dependencies.

Run:  /Users/zerozaki07/tmp/xca/.triqs/venv/bin/python omt-leaf-polynomial-density.py
"""

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "svg.fonttype": "path",
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Helvetica Neue", "Arial"],
        "mathtext.fontset": "stix",
    }
)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse

# ---------------------------------------------------------------- palette
INK = "#172033"
MUTED = "#526174"
SLATE = "#64748b"
BORDER = "#cbd5e1"
BLUE = "#2563eb"
PURPLE = "#7c3aed"
CYAN = "#0891b2"
ORANGE = "#ea580c"
GREEN = "#16a34a"
INDIGO = "#6366f1"
FILL_SUBTLE = "#f8fafc"
FILL_ACCENT = "#eff6ff"
FILL_WARN = "#fff7ed"
FILL_GOOD = "#f0fdf4"
FILL_TAKE = "#eef2ff"

HEAD, LABEL, BODY, SMALL, MATH = 15.5, 12.0, 10.5, 9.3, 12.5

W, H = 1200, 690
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # y grows downward, like SVG coordinates
ax.set_aspect("equal")
ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.4, r=12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=1,
        )
    )


def text(x, y, s, size=BODY, color=INK, ha="left", va="baseline", weight="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight)


def arrow(p0, p1, color=SLATE, lw=2.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=14, color=color,
            linewidth=lw, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        )
    )


# ================================================================ panels
P1 = (28, 48, 320, 564)
P2 = (380, 48, 388, 564)
P3 = (800, 48, 372, 564)
for (x, y, w, h) in (P1, P2, P3):
    rbox(x, y, w, h, "white", BORDER, r=16)

arrow((348, 240), (380, 240))
arrow((768, 240), (800, 240))

# ================================================================ 1: OMT density
x0 = P1[0] + 24
text(x0, 88, "1  OMT 轨道乘积→连续密度", HEAD, weight="bold")
text(x0, 124, r"$\rho_{\mu\nu}(\mathbf{r}) \;=\; \psi_\mu(\mathbf{r})\,\psi_\nu(\mathbf{r})$", MATH)

# --- 1D density profile through two atoms
bx, by, bw, bh = 52, 148, 272, 226
rbox(bx, by, bw, bh, FILL_SUBTLE, BORDER, r=10)
pad, base = 18, by + bh - 46
u = np.linspace(0, 1, 400)
prof = 0.16 + 0.05 * np.sin(2 * np.pi * (u - 0.06)) ** 2
prof += 1.00 * np.exp(-((u - 0.28) / 0.050) ** 2)
prof += 0.78 * np.exp(-((u - 0.72) / 0.045) ** 2)
px = bx + pad + u * (bw - 2 * pad)
py = base - prof * 130
ax.fill_between(px, py, base, color=BLUE, alpha=0.16, lw=0)
ax.plot(px, py, color=CYAN, lw=2.2)
for c, r_mt, col in ((0.28, 0.105, BLUE), (0.72, 0.095, PURPLE)):
    for s in (-1, 1):
        xb = bx + pad + (c + s * r_mt) * (bw - 2 * pad)
        ax.plot([xb, xb], [by + 26, base], color=col, lw=1.1, ls=(0, (4, 3)), alpha=0.75)
text(bx + pad + 0.28 * (bw - 2 * pad), by + 20, "MT 球 A", SMALL, MUTED, ha="center")
text(bx + pad + 0.72 * (bw - 2 * pad), by + 20, "MT 球 B", SMALL, MUTED, ha="center")
text(bx + pad + 0.28 * (bw - 2 * pad), base + 18, "球内 augmentation", SMALL, MUTED, ha="center")
text(bx + pad + 0.50 * (bw - 2 * pad), base + 34, "间隙区 tail", SMALL, MUTED, ha="center")

text(x0, 415, "它是占满体积的连续函数，而不是", BODY)
text(x0, 450, r"$\sum_j q_j\,\delta(\mathbf{r}-\mathbf{r}_j)$", MATH)
rbox(52, 476, 272, 62, FILL_WARN, ORANGE, r=8)
text(66, 502, "若直接粗采样为点电荷：", SMALL, ORANGE, weight="bold")
text(66, 522, "丢失盒内形状与高阶多极矩", SMALL, MUTED)

# ================================================================ 2: leaf polynomials
x0 = P2[0] + 24
text(x0, 88, "2  叶盒多项式保存盒内形状", HEAD, weight="bold")
text(x0, 112, "自适应树只在球边界与快变区加密", SMALL, MUTED)

# --- adaptive quadtree refined around two MT-sphere boundaries
tx, ty, ts = 408, 140, 172
circles = (((0.30, 0.62), 0.38), ((0.80, 0.24), 0.26))


def crosses(cx0, cy0, s):
    for (cc, r) in circles:
        ddx = max(cc[0] - (cx0 + s), 0, cx0 - cc[0])
        ddy = max(cc[1] - (cy0 + s), 0, cy0 - cc[1])
        dmin = np.hypot(ddx, ddy)
        dmax = max(
            np.hypot(cc[0] - xx, cc[1] - yy)
            for xx in (cx0, cx0 + s) for yy in (cy0, cy0 + s)
        )
        if dmin < r < dmax:
            return True
    return False


leaves = []


def subdivide(cx0, cy0, s, depth):
    if depth == 0 or not crosses(cx0, cy0, s):
        leaves.append((cx0, cy0, s))
        return
    h2 = s / 2
    for ddx in (0, h2):
        for ddy in (0, h2):
            subdivide(cx0 + ddx, cy0 + ddy, h2, depth - 1)


subdivide(0, 0, 1, 3)
smin = min(s for (_, _, s) in leaves)
target = (circles[0][0][0] + circles[0][1] * 0.866, circles[0][0][1] + circles[0][1] * 0.5)
hi = next(
    (c for c in leaves
     if c[2] == smin and c[0] <= target[0] <= c[0] + c[2] and c[1] <= target[1] <= c[1] + c[2]),
    min((c for c in leaves if c[2] == smin),
        key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2),
)
for (cx0, cy0, s) in leaves:
    ax.add_patch(Rectangle((tx + cx0 * ts, ty + cy0 * ts), s * ts, s * ts,
                           facecolor="none", edgecolor="#94a3b8", lw=0.9))
frame = Rectangle((tx, ty), ts, ts, facecolor="none", edgecolor=SLATE, lw=1.4)
ax.add_patch(frame)
for (cc, r) in circles:
    circ = Circle((tx + cc[0] * ts, ty + cc[1] * ts), r * ts, facecolor="none",
                  edgecolor=PURPLE, lw=1.6, ls=(0, (5, 3)), alpha=0.85)
    ax.add_patch(circ)
    circ.set_clip_path(frame)  # keep the sphere boundaries inside the tree box
hx, hy = tx + hi[0] * ts, ty + hi[1] * ts
ax.add_patch(Rectangle((hx, hy), smin * ts, smin * ts,
                       facecolor="#dbeafe", edgecolor=BLUE, lw=2.0, zorder=4))
text(tx + ts / 2, ty + ts + 20, "选中的叶盒 B（虚线：MT 球面）", SMALL, MUTED, ha="center")

# --- zoom into the highlighted leaf
zx, zy, zs = 596, 150, 140
rbox(zx, zy, zs, zs, FILL_ACCENT, BLUE, lw=1.7, r=8)
arrow((hx + smin * ts, hy + smin * ts / 2), (zx - 4, zy + zs / 2), color=BLUE, lw=1.6, rad=-0.18)
n = 7
k = np.arange(n)
cheb = 0.5 * (1 - np.cos((2 * k + 1) * np.pi / (2 * n)))  # Chebyshev nodes on [0,1]
f = lambda t: 0.52 - 0.30 * np.cos(2.6 * np.pi * t + 0.4) * np.exp(-1.2 * t)
tt = np.linspace(0, 1, 200)
zpad = 12
zpx = zx + zpad + tt * (zs - 2 * zpad)
zpy = zy + zs - zpad - f(tt) * (zs - 2 * zpad)
for c in cheb:
    xb = zx + zpad + c * (zs - 2 * zpad)
    ax.plot([xb, xb], [zy + zpad, zy + zs - zpad], color=BLUE, lw=0.7, alpha=0.30)
ax.plot(zpx, zpy, color=BLUE, lw=2.4)
ax.scatter(zx + zpad + cheb * (zs - 2 * zpad),
           zy + zs - zpad - f(cheb) * (zs - 2 * zpad),
           s=22, color=BLUE, zorder=5)
text(zx + zs / 2, zy + zs + 20, "Chebyshev 节点（向盒边聚集）", SMALL, MUTED, ha="center")
text(zx + zs / 2, zy + zs + 36, "→ 张量积多项式系数", SMALL, MUTED, ha="center")

rbox(404, 366, 340, 74, FILL_ACCENT, BLUE, lw=1.7, r=10)
text(422, 398, r"$\rho\,|_B \;\approx\; \sum_{abc} c_{abc}\, T_a(x)\,T_b(y)\,T_c(z)$", MATH)
text(422, 426, "少量系数同时保留电荷、偶极与更高多极矩", SMALL, MUTED)

text(404, 470, "为什么这一层有用？", LABEL, weight="bold")
for i, line in enumerate((
    "自适应：只在需要处增加叶盒",
    "高阶：同一盒内无需铺大量点源",
    "近场：连续源与 1/r 奇点可解析积分",
    "远场：直接转成 DMK 的 moments / proxies",
)):
    text(422, 500 + 28 * i, "•  " + line, BODY)

# ================================================================ 3: DMK
x0 = P3[0] + 24
text(x0, 88, "3  DMK 消费连续体源", HEAD, weight="bold")
text(x0, 112, "每个尺度只处理它负责的空间范围", SMALL, MUTED)

bands = (
    (136, FILL_WARN, ORANGE, "Near field", "叶盒内解析 / 高精度连续积分"),
    (224, FILL_ACCENT, BLUE, "Compact field", "同层邻盒之间的紧支撑传播"),
    (312, FILL_SUBTLE, BORDER, "Root far field", "周期镜像求和与边界条件"),
)
bx, bw2, bh2 = 824, 324, 72
for (byy, fc, ec, name, note) in bands:
    rbox(bx, byy, bw2, bh2, fc, ec, lw=1.5, r=10)
    text(bx + 96, byy + 30, name, LABEL, weight="bold")
    text(bx + 96, byy + 52, note, SMALL, MUTED)

# icons: near = gaussian blob in leaf; compact = 3x3 neighbours; far = orbits
g = np.exp(-((np.linspace(-1, 1, 60)[None, :]) ** 2 + (np.linspace(-1, 1, 60)[:, None]) ** 2) / 0.35)
ax.imshow(g, extent=(bx + 22, bx + 70, 136 + 60, 136 + 12), cmap="Blues", alpha=0.9, zorder=3)
ax.add_patch(Rectangle((bx + 22, 136 + 12), 48, 48, facecolor="none", edgecolor=ORANGE, lw=1.6, zorder=4))
for i in range(3):
    for j in range(3):
        cell_fc = "#93c5fd" if (i, j) != (1, 1) else BLUE
        ax.add_patch(Rectangle((bx + 22 + 16 * i, 224 + 13 + 16 * j), 14, 14,
                               facecolor=cell_fc, edgecolor=BLUE, lw=0.9))
fx, fy = bx + 46, 312 + 36
ax.add_patch(Ellipse((fx, fy), 58, 24, facecolor="none", edgecolor=SLATE, lw=1.3))
ax.add_patch(Ellipse((fx, fy), 36, 52, facecolor="none", edgecolor=SLATE, lw=1.3))
ax.add_patch(Circle((fx, fy), 4.5, facecolor=SLATE, edgecolor="none"))

arrow((986, 384 + bh2 / 2 + 12), (986, 442), color=SLATE)
rbox(bx, 446, bw2, 118, FILL_GOOD, GREEN, lw=1.5, r=10)
text(986, 474, "输出", LABEL, ha="center", weight="bold")
text(986, 512, r"$V_\mathrm{H}(\mathbf{r}) \;=\; \int \frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}^3 r'$", MATH, ha="center")
text(986, 546, "Hartree 势 · Coulomb 矩阵元 · THC / RI", SMALL, MUTED, ha="center")

# ================================================================ takeaway
rbox(100, 636, 1000, 42, FILL_TAKE, INDIGO, lw=1.3, r=10)
text(600, 662, "关键：OMT 本身不强制 leaf polynomial；它是把 OMT 连续密度无损接入 volume-DMK 的紧凑接口。",
     BODY, ha="center")

# --- overflow check: no text may cross its panel border (x direction)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
for t in ax.texts:
    bb = t.get_window_extent(rend).transformed(inv)
    ax0, ay0 = t.get_position()
    for (px, py, pw, ph) in (P1, P2, P3):
        if px <= ax0 <= px + pw and py <= ay0 <= py + ph:
            if bb.x1 > px + pw - 6 or bb.x0 < px + 6:
                print(f"OVERFLOW: {t.get_text()[:34]!r} spans x [{bb.x0:.0f}, {bb.x1:.0f}] "
                      f"in panel [{px}, {px + pw}]")

out = __file__.replace(".py", ".svg")
fig.savefig(out, format="svg")
print("wrote", out)

import os
if os.environ.get("PREVIEW"):
    fig.savefig("/tmp/omt_preview.png", dpi=100)
    print("wrote /tmp/omt_preview.png")
