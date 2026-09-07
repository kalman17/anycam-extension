"""Diagnostic: are the AnyCam pseudo-pose teachers geometrically consistent with the
student's flow-reprojection loss?

Round 1 (2026-08-11) showed the cached teachers give a flow error WORSE than
zero-motion, and that swapping the focal barely changes it -> the problem is not the
focal alone. Prime suspect now: the teachers were generated letting AnyCam recompute
its own depth internally (pred_depths = 0.1/metric) while our loss uses the cached
depth (a different scale), so teacher translations live on a different scale.

This script settles it three ways, on the same sequences:
  A) sweep a multiplicative scale on the cached teachers' translations
     -> a clear minimum away from 1.0 proves a scale-convention mismatch
  B) rotation-only teachers (translation zeroed)
     -> isolates whether the rotations are usable even if translations aren't
  C) regenerate poses with EVERYTHING provided from the cache
     (depth + flow + intrinsics) and measure those
     -> this is the proposed fix; it should beat zero-motion clearly

Usage: PYTHONPATH=$REPO python mcvo/diagnose_teacher_geometry.py \
           --data_dir /storage/user/maka/preprocessed \
           --pose_dir /storage/user/maka/pseudo_poses --n 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.datasets.preprocessed_dataset import PreprocessedMultiFrameDataset  # noqa
from anycam.trainer import induce_flow_dist, make_proj_from_focal_length  # noqa
from omegaconf import OmegaConf  # noqa
from anycam.scripts.common import load_model  # noqa


def flow_err(poses, focal_norm, depths, flow_occs, H, W):
    proj = make_proj_from_focal_length(focal_norm.unsqueeze(1), aspect_ratio=H / W)
    induced, _ = induce_flow_dist(depths.unsqueeze(2) * 0.1, proj, poses, flow_occs[:, :, :2])
    target = flow_occs[:, :-1, :2]
    pred = induced[:, :-1, 0].clamp(-1, 1)
    valid = flow_occs[:, :-1, 2:3] >= 0.5
    err = (pred - target).abs().mean(dim=2, keepdim=True)
    err = torch.where(valid, err, torch.zeros_like(err))
    return float(err.sum() / valid.float().sum().clamp(min=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pose_dir", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    ds = PreprocessedMultiFrameDataset(data_dir=args.data_dir, max_ahead=7,
                                       image_size=336, phase="A")
    pose_dir = Path(args.pose_dir)

    # teacher model with EVERYTHING taken from the cache (the proposed fix)
    cfg = OmegaConf.load(REPO / "pretrained_models/anycam_seq8/training_config.yaml")
    cfg["model"]["use_provided_flow"] = True
    cfg["model"]["use_provided_depth"] = True
    # NOTE: use_provided_proj is incompatible with this checkpoint
    # (separate_pose_candidates=true -> 32 pose candidates vs 1 provided proj).
    # Instead we pass target_focal so AnyCam SELECTS the candidate closest to our
    # calibration, which puts its poses in (approximately) our geometry.
    cfg["model"]["train_directions"] = "forward"
    model = load_model(cfg, REPO / "pretrained_models/anycam_seq8/training_checkpoint_247500.pt")
    model = model.to(dev).eval()

    scales = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
    acc = {"zero": [], "cached": [], "rotonly": [], "fixed": []}
    acc_scale = {s: [] for s in scales}
    checked = 0

    for idx in range(len(ds)):
        if checked >= args.n:
            break
        ds_name, video, start = ds.samples[idx]
        pf = pose_dir / ds_name / video / f"POSES_{start:06d}.npz"
        if not pf.exists():
            continue
        try:
            s = ds[idx]
        except Exception:
            continue

        depths = s["depths"].unsqueeze(0).to(dev)
        flows = s["flows_fwd"].unsqueeze(0).to(dev)
        flows_b = s["flows_bwd"].unsqueeze(0).to(dev)
        occs = s["occs_fwd"].unsqueeze(0).to(dev)
        calibs = s["calibs"].unsqueeze(0).to(dev)
        imgs = s["images"].unsqueeze(0).to(dev)
        B, N, _, H, W = depths.shape
        flow_occs = torch.cat([flows[:, :, 0:1] * 2.0 / W,
                               flows[:, :, 1:2] * 2.0 / H, occs], dim=2)
        flow_occs = torch.cat([flow_occs, torch.zeros(B, 1, 3, H, W, device=dev)], dim=1)
        fnorm = 2.0 * calibs.mean(dim=1)[:, 0] / W

        T = torch.from_numpy(np.load(pf)["rel_poses"]).unsqueeze(0).unsqueeze(2).to(dev)
        eye = torch.eye(4, device=dev).view(1, 1, 1, 4, 4).expand(B, N, 1, 4, 4).contiguous()

        acc["zero"].append(flow_err(eye, fnorm, depths, flow_occs, H, W))
        acc["cached"].append(flow_err(T, fnorm, depths, flow_occs, H, W))
        Trot = T.clone(); Trot[..., :3, 3] = 0
        acc["rotonly"].append(flow_err(Trot, fnorm, depths, flow_occs, H, W))
        for sc in scales:
            Ts = T.clone(); Ts[..., :3, 3] *= sc
            acc_scale[sc].append(flow_err(Ts, fnorm, depths, flow_occs, H, W))

        # C) regenerate with everything provided from the cache
        K = torch.eye(3, device=dev).view(1, 3, 3).repeat(N, 1, 1)
        c = calibs[0]
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        with torch.no_grad():
            out = model({"imgs": imgs, "projs": K.unsqueeze(0), "depths": depths,
                         "flows_fwd": flows, "flows_bwd": flows_b},
                        target_focal=fnorm.view(-1, 1))
        P = out["proc_poses"][0].unsqueeze(0).unsqueeze(2).to(dev).float()
        if P.shape[1] == N - 1:  # pad identity for the last frame
            P = torch.cat([P, torch.eye(4, device=dev).view(1, 1, 1, 4, 4)], dim=1)
        acc["fixed"].append(flow_err(P, fnorm, depths, flow_occs, H, W))
        checked += 1

    m = {k: float(np.median(v)) for k, v in acc.items() if v}
    print(f"\nn = {checked} sequences   (flow error, lower is better)\n")
    print(f"  zero-motion reference          {m['zero']:.5f}")
    print(f"  cached teachers (as used E4a)  {m['cached']:.5f}")
    print(f"  cached teachers, rotation only {m['rotonly']:.5f}")
    print(f"  REGENERATED, all-from-cache    {m['fixed']:.5f}   <- proposed fix\n")
    print("  translation-scale sweep on cached teachers:")
    for sc in scales:
        print(f"    x{sc:<6} {float(np.median(acc_scale[sc])):.5f}")
    best = min(scales, key=lambda s: float(np.median(acc_scale[s])))
    print(f"\n  best scale = x{best}")
    if m["fixed"] < m["zero"]:
        print("\nVERDICT: regenerating with cached depth+flow+intrinsics produces usable"
              "\n         teachers -> regenerate and relaunch E4.")
    else:
        print("\nVERDICT: even fully-provided teachers do not beat zero-motion on this data"
              "\n         -> AnyCam FF is not a usable teacher here; rethink E4.")


if __name__ == "__main__":
    main()
