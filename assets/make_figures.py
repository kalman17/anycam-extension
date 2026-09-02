import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from pathlib import Path
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5})
INK, GREY, MUTED = "#222222", "#555555", "#8a8a8a"
BLUE, ORANGE, RED, GREEN, PURPLE = "#2b6a99", "#d9702e", "#b04a4a", "#2f7d4f", "#5b4b9a"
FILL_BB, FILL_DEC, FILL_HEAD, FILL_TRAIN = "#eef0f7", "#eaf5ee", "#fdf1e6", "#fafafa"
TEXTW = 6.5  # inches, = \linewidth of the PDF

K = Path.home()/"git/masters/anycam-extension/data/eval/kitti_odom"
poses = np.loadtxt(K/"poses/07.txt").reshape(-1,3,4); t = poses[:,:,3]
frames = [Image.open(K/f"sequences/07/image_2/{i:06d}.jpg") for i in (0, 10, 20)]

# ============================================================ overview (grid layout)
fig = plt.figure(figsize=(TEXTW, 2.7), dpi=300)
cv = fig.add_axes([0,0,1,1]); cv.set_axis_off(); cv.set_xlim(0,1); cv.set_ylim(0,1)
FS, SS = 7.0, 6.2
def box(x, y, w, h, title, sub="", fc="#f4f4f4", ec=INK, lw=0.7, ls="-", fs=FS, subfs=SS, tcol=None, z=2):
    cv.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if sub:
        cv.text(x+w/2, y+h*0.70, title, ha="center", va="center", fontsize=fs, color=tcol or INK, zorder=z+1)
        cv.text(x+w/2, y+h*0.32, sub, ha="center", va="center", fontsize=subfs, color=GREY, zorder=z+1, linespacing=1.15)
    else:
        cv.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=fs, color=tcol or INK, zorder=z+1)
def seg(pts, color=INK, lw=0.8, ls="-"):
    xs, ys = zip(*pts); cv.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=4)
def head(x0, y0, x1, y1, color=INK, lw=0.8, ls="-"):
    cv.annotate("", xy=(x1,y1), xytext=(x0,y0), arrowprops=dict(arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0), zorder=4)

# ---- geometry (all in axes fraction) ----
YC, YP, YK = 0.70, 0.84, 0.56           # chain centreline, pose row, calibration row
X_IN0, X_IN1 = 0.165, 0.735              # inference dotted box
# dotted box around the inference algorithm
cv.add_patch(FancyBboxPatch((X_IN0, 0.40), X_IN1-X_IN0, 0.545, boxstyle="round,pad=0,rounding_size=0.008", fc="none", ec=MUTED, lw=0.7, ls=(0,(1.5,2)), zorder=1))
cv.text(X_IN0, 0.975, "MCVO, inference  ·  154 M params  ·  75 ms and 0.7 GB per 4-frame window (A40)", ha="left", va="center", fontsize=6.6, color=GREY)

# input frames
axf = fig.add_axes([0.012, 0.50, 0.135, 0.42]); axf.set_axis_off(); axf.set_xlim(0,1); axf.set_ylim(0,1)
W, H = 0.72, 0.44
for k, im in enumerate(frames):
    x0, y0 = 0.0 + k*0.14, 0.54 - k*0.25
    axf.imshow(im, extent=[x0, x0+W, y0, y0+H], aspect="auto", zorder=k)
    axf.add_patch(plt.Rectangle((x0, y0), W, H, fill=False, ec="white", lw=0.6, zorder=k+.5))
cv.text(0.082, 0.43, "uncalibrated\nmonocular video\n(no IMU, no GPS)", ha="center", va="center", fontsize=SS, color=GREY, linespacing=1.15)

# chain
head(0.150, YC, 0.19, YC)
box(0.19, YC-0.11, 0.12, 0.22, "DINOv2-base", "frozen · 86 M\npatch tokens / frame", fc=FILL_BB, ec=PURPLE, tcol=PURPLE)
head(0.31, YC, 0.345, YC)
box(0.345, YC-0.155, 0.185, 0.31, "transformer decoder", "10 blocks · 67 M trained\ncross-frame, then\nwithin-frame attention\none camera token / frame", fc=FILL_DEC, ec=GREEN, tcol=GREEN)
# elbow connectors decoder -> heads
XE = 0.555
seg([(0.530, YC), (XE, YC), (XE, YP)]); head(XE, YP, 0.585, YP)
seg([(XE, YC), (XE, YK)]);            head(XE, YK, 0.585, YK)
box(0.585, YP-0.075, 0.125, 0.15, "pose head", "camera tokens (i, i+1)", fc=FILL_HEAD, ec=ORANGE, tcol=ORANGE, fs=6.8, subfs=5.8)
box(0.585, YK-0.075, 0.125, 0.15, "calibration head", "camera token", fc=FILL_HEAD, ec=ORANGE, tcol=ORANGE, fs=6.8, subfs=5.8)
# heads -> outputs (cross the dotted box)
head(0.710, YP, 0.765, YP, ORANGE, 0.9)
head(0.710, YK, 0.765, YK, ORANGE, 0.9)

# output 1: trajectory, centred on YP
ax2 = fig.add_axes([0.765, YP-0.11, 0.225, 0.24], projection="3d")
N = 320; x, y, z = t[:N,0], t[:N,1], t[:N,2]
ax2.plot(x, z, -y, color=BLUE, lw=1.2)
def frustum(R, c, s=3.0):
    pts = np.array([[0,0,0],[-1,-0.6,1.6],[1,-0.6,1.6],[1,0.6,1.6],[-1,0.6,1.6]])*s
    return (R @ pts.T).T + c
for i in range(0, N, 26):
    P = frustum(poses[i,:,:3], poses[i,:,3]); P = np.stack([P[:,0], P[:,2], -P[:,1]], 1)
    for a_, b_ in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]:
        ax2.plot([P[a_,0],P[b_,0]],[P[a_,1],P[b_,1]],[P[a_,2],P[b_,2]], color=ORANGE, lw=0.55)
ax2.set_box_aspect((np.ptp(x)+1, np.ptp(z)+1, 8), zoom=1.9); ax2.view_init(elev=32, azim=-60)
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([]); ax2.grid(False); ax2.patch.set_alpha(0)
for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
    axis.pane.fill=False; axis.line.set_color((1,1,1,0)); axis.pane.set_edgecolor((1,1,1,0))
cv.text(0.877, YP-0.135, "camera pose (R, t) of every frame", ha="center", va="center", fontsize=FS, color=INK)

# output 2: K, centred on YK
cv.text(0.877, YK-0.115, "camera intrinsics of the sequence", ha="center", va="center", fontsize=FS, color=INK)
cv.text(0.80, YK, "K =", ha="center", va="center", fontsize=7.5, color=INK)
for r, row in enumerate([("f","0","cₓ"),("0","f","cᵧ"),("0","0","1")]):
    for j, v in enumerate(row):
        cv.text(0.855 + j*0.035, YK+0.055 - r*0.055, v, ha="center", va="center", fontsize=6.8, color=INK)
cv.text(0.833, YK, "[", ha="center", va="center", fontsize=17, color=INK); cv.text(0.948, YK, "]", ha="center", va="center", fontsize=17, color=INK)

# ---- training row, aligned with the inference box ----
YT = 0.17
cv.add_patch(FancyBboxPatch((X_IN0, 0.02), X_IN1-X_IN0, 0.275, boxstyle="round,pad=0,rounding_size=0.008", fc=FILL_TRAIN, ec=MUTED, lw=0.7, ls=(0,(1.5,2)), zorder=1))
cv.text(X_IN0, 0.318, "training only, no labels", ha="left", va="center", fontsize=6.6, color=GREY, style="italic")
YT_ROWS = [0.235, 0.155, 0.075]
for yy, (name, what) in zip(YT_ROWS, [("UniDepth", "depth"), ("UniMatch", "flow"), ("AnyCalib", "K")]):
    box(0.19, yy-0.038, 0.115, 0.076, name, what, fc="#ffffff", ec=MUTED, fs=6.2, subfs=5.4, tcol=INK)
# merge lines from the three teachers into the loss box
XM = 0.38
for yy in YT_ROWS:
    seg([(0.305, yy), (XM, yy)], MUTED, 0.7)
seg([(XM, YT_ROWS[0]), (XM, YT_ROWS[-1])], MUTED, 0.7)
head(XM, 0.155, 0.44, 0.155, MUTED, 0.7)
box(0.44, 0.045, 0.27, 0.22, "flow re-projection loss", "unproject with depth and K,\nmove by the predicted pose, re-project,\ncompare to the teacher flow per pixel", fc="#ffffff", ec=MUTED, fs=6.6, subfs=5.5, tcol=INK)
# gradients: vertical dashed line from the loss box up into the inference box
seg([(0.575, 0.265), (0.575, 0.40)], MUTED, 0.7, (0,(2,2))); head(0.575, 0.40, 0.575, 0.43, MUTED, 0.7)
cv.text(0.585, 0.36, "gradients", ha="left", va="center", fontsize=5.6, color=MUTED, style="italic")
fig.savefig("overview_mcvo.png", facecolor="white", dpi=300); print("overview ok")

# ============================================================ results: same size, same style
def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.tick_params(length=2.5, width=0.5)

FW, FH = 3.15, 2.25
rows = [("MCVO (ours)", 75, 0.69, np.mean([0.46,0.89,0.19]), BLUE, True),
        ("AnyCam", 413, 3.7, np.mean([0.50,0.74,0.20]), MUTED, False),
        ("Monodepth2", 13, 0.09, np.mean([0.80,0.77,0.30]), MUTED, False),
        ("π³", 171, 5.5, np.mean([0.22,0.26,0.11]), RED, False),
        ("VGGT-1B", 203, 7.0, np.mean([0.28,0.32,0.12]), RED, False),
        ("Depth Anything 3", 600, 9.7, np.mean([0.19,0.27,0.09]), RED, False)]
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
for name, lat, mem, rot, col, ours in rows:
    ax.scatter(lat, rot, s=25+mem*28, color=col, alpha=0.95 if ours else 0.6, edgecolor="white", lw=0.6, zorder=3)
    dx = 0.78 if name == "π³" else 1.14
    dy = -0.05 if name in ("Monodepth2", "π³") else 0.018
    ax.annotate(name, (lat, rot), xytext=(lat*dx, rot+dy), fontsize=6.5, color=col if ours else INK, fontweight="bold" if ours else "normal", zorder=4)
ax.set_xscale("log"); ax.set_xlabel("latency per 4-frame window, A40 (ms)"); ax.set_ylabel("median rotation error (°)")
ax.set_xlim(9, 1500); ax.set_ylim(0.1, 0.75); style(ax)
ax.text(0.02, 0.97, "bubble area = peak GPU memory\nred = trained with GT poses · grey/blue = no labels", transform=ax.transAxes, fontsize=5.8, va="top", color=GREY, linespacing=1.15)
r_m, r_a = np.mean([0.46,0.89,0.19]), np.mean([0.50,0.74,0.20])
ax.annotate("", xy=(75*1.2, r_m), xytext=(413/1.15, r_a), arrowprops=dict(arrowstyle="-|>", mutation_scale=6, color=BLUE, lw=0.8), zorder=2)
ax.text(165, 0.58, "5.5× faster, 5× less memory,\nsame rotation accuracy", fontsize=5.8, color=BLUE, ha="center", linespacing=1.15)
fig.savefig("benchmark_mcvo.png", bbox_inches="tight", facecolor="white", dpi=300); print("benchmark ok")

d = json.load(open(Path.home()/"git/masters/anycam-extension/thesis_results/figures/figure_data.json"))
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
nw = d["n_windows"]; xw = np.arange(nw)
xa = np.linspace(0, nw-1, d["n_anycalib_points"])
ax.axhline(d["gt_fx"], color=INK, lw=0.8, ls="--", label=f"ground truth (f = {d['gt_fx']:.0f} px)", zorder=1)
ax.scatter(xa, d["anycalib_fx"], s=5, color=MUTED, alpha=0.6, label="AnyCalib, per frame", zorder=2)
ax.step(xw, d["anycam_fx"], where="mid", color=RED, lw=0.7, alpha=0.8, label="AnyCam, 32-candidate search", zorder=3)
ax.plot(xw, d["fat_fx"], color=BLUE, lw=1.3, label="ours, multi-frame calibration", zorder=4)
ax.set_xlabel(f"window index, Sintel {d['sequence']}"); ax.set_ylabel("focal length (px)")
lo, hi = np.percentile(np.r_[d["anycalib_fx"], d["fat_fx"]], [1, 99]); ax.set_ylim(min(lo, d["gt_fx"])*0.8, max(hi, d["gt_fx"])*1.25)
style(ax); ax.set_ylim(0, 1250); ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", ncol=1)
fig.savefig("focal_mcvo.png", bbox_inches="tight", facecolor="white", dpi=300); print("focal ok")
