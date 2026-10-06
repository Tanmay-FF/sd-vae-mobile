"""Parity test and sample-image accuracy check for the converted VAE.

Backends compared (all fp32 unless noted):
  pt_gpu   PyTorch on CUDA, TF32 disabled (the reference)
  pt_cpu   PyTorch on CPU (shows the fp32 noise floor between devices)
  ort      ONNX Runtime, CUDA EP with TF32 disabled
  tfl32    TFLite fp32 (LiteRT interpreter, CPU/XNNPACK)
  tfl16    TFLite fp16 weights (dequantized to fp32 at load on CPU)
  tfl8w    TFLite int8 weight-only (per-channel int8 weights, fp32 compute)
  tfl8d    TFLite int8 dynamic-range (int8 weights, activations quantized at runtime)

1. Parity: the encoder and decoder graphs are each fed identical inputs (one
   random tensor and one real tensor) and every backend is compared against
   pt_gpu.
2. Accuracy: each backend runs the full image -> encoder -> decoder round trip
   end to end (no mixing backends). The result is scored against the original
   image (absolute VAE quality, which catches bad weights) and against the
   pt_gpu reconstruction (conversion fidelity), then flagged MATCH/MISMATCH.

The thresholds below are fixed in advance. Do not tune them to the results.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from convert import Decoder, Encoder, load_vae

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# Parity is measured per tensor. nmax = max|a-b| / max|ref|.
PARITY_TOL = {
    "fp32": {"nmax": 1e-3, "cos": 0.99999},
    "fp16": {"nmax": 2e-2, "cos": 0.9995},
    "int8": {"nmax": 5e-2, "cos": 0.999},
}
# End-to-end reconstruction, uint8 image space.
ACC_TOL = {
    "fp32": {"psnr_vs_ref": 45.0, "ssim_vs_ref": 0.995, "dpsnr_vs_orig": 0.1},
    "fp16": {"psnr_vs_ref": 35.0, "ssim_vs_ref": 0.98, "dpsnr_vs_orig": 0.5},
    "int8": {"psnr_vs_ref": 30.0, "ssim_vs_ref": 0.95, "dpsnr_vs_orig": 1.0},
}
SANITY_PSNR_VS_ORIG = 22.0  # below this the VAE itself is broken
PRECISION = {"pt_cpu": "fp32", "ort": "fp32", "tfl32": "fp32", "tfl16": "fp16",
             "tfl8w": "int8", "tfl8d": "int8"}


# ---------------------------------------------------------------- backends
class TorchBackend:
    def __init__(self, vae, device):
        self.device = device
        vae = copy.deepcopy(vae).to(device)  # backends must not share modules
        self.enc = Encoder(vae).eval()
        self.dec = Decoder(vae).eval()

    @torch.no_grad()
    def _run(self, m, x):
        return m(torch.from_numpy(x).to(self.device)).float().cpu().numpy()

    def encode(self, x):
        return self._run(self.enc, x)

    def decode(self, z):
        return self._run(self.dec, z)


class OrtBackend:
    def __init__(self):
        providers = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
        so = ort.SessionOptions()
        self.enc = ort.InferenceSession(str(ART / "onnx/vae_encoder.onnx"), so, providers=providers)
        self.dec = ort.InferenceSession(str(ART / "onnx/vae_decoder.onnx"), so, providers=providers)
        self.provider = self.enc.get_providers()[0]

    def encode(self, x):
        return self.enc.run(None, {"image": x})[0]

    def decode(self, z):
        return self.dec.run(None, {"latent": z})[0]


class TfliteGraph:
    """Wraps one .tflite graph behind an NCHW interface (onnx2tf emits NHWC)."""

    def __init__(self, path, threads):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def __call__(self, x_nchw):
        want = tuple(self.inp["shape"])
        x = x_nchw
        if want != x.shape:
            x = np.transpose(x, (0, 2, 3, 1))
            assert tuple(x.shape) == want, (x.shape, want)
        self.it.set_tensor(self.inp["index"], np.ascontiguousarray(x, dtype=np.float32))
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])
        return np.transpose(y, (0, 3, 1, 2)) if y.shape[-1] in (3, 4) and y.shape[1] not in (3, 4) else y


class TfliteBackend:
    def __init__(self, precision, threads):
        self.enc = TfliteGraph(ART / f"tflite/vae_encoder/vae_encoder_{precision}.tflite", threads)
        self.dec = TfliteGraph(ART / f"tflite/vae_decoder/vae_decoder_{precision}.tflite", threads)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)


# ---------------------------------------------------------------- metrics
def tensor_metrics(a, ref):
    d = np.abs(a.astype(np.float64) - ref.astype(np.float64))
    af, rf = a.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "mse": float((d ** 2).mean()),
        "nmax": float(d.max() / (np.abs(ref).max() + 1e-12)),
        "cos": float(af @ rf / (np.linalg.norm(af) * np.linalg.norm(rf) + 1e-12)),
    }


def to_uint8(x_nchw):
    img = (np.clip(x_nchw[0].transpose(1, 2, 0), -1, 1) + 1) * 127.5
    return np.round(img).astype(np.uint8)


def image_metrics(img, ref):
    return {
        "psnr": float(peak_signal_noise_ratio(ref, img, data_range=255)),
        "mse": float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2)),
        "mse_01": float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2) / 255.0 ** 2),
        "ssim": float(structural_similarity(ref, img, channel_axis=2, data_range=255)),
    }


def load_sample(size):
    img = Image.fromarray(skdata.astronaut()).resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1)[None].copy(), np.asarray(img)


def timed(fn, x, reps=3):
    fn(x)  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        y = fn(x)
    return y, (time.perf_counter() - t) / reps * 1000


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(0)

    vae = load_vae()
    backends = {
        "pt_gpu": TorchBackend(vae, "cuda" if torch.cuda.is_available() else "cpu"),
        "pt_cpu": TorchBackend(vae, "cpu"),
        "ort": OrtBackend(),
        "tfl32": TfliteBackend("float32", args.threads),
        "tfl16": TfliteBackend("float16", args.threads),
        "tfl8w": TfliteBackend("int8_weight_only", args.threads),
        "tfl8d": TfliteBackend("int8_dynamic", args.threads),
    }
    print(f"ORT provider: {backends['ort'].provider}, PT ref device: {backends['pt_gpu'].device}")

    s, ls = args.size, args.size // 8
    x_real, orig_u8 = load_sample(s)
    ref = backends["pt_gpu"]
    inputs = {
        "encoder": {
            "random": np.random.uniform(-1, 1, (1, 3, s, s)).astype(np.float32),
            "real": x_real,
        },
        "decoder": {
            # Raw (unscaled) latents have std of about 1/0.18215.
            "random": (np.random.randn(1, 4, ls, ls) / 0.18215).astype(np.float32),
            "real": ref.encode(x_real),
        },
    }

    report = {"thresholds": {"parity": PARITY_TOL, "accuracy": ACC_TOL,
                             "sanity_psnr_vs_orig": SANITY_PSNR_VS_ORIG},
              "parity": {}, "accuracy": {}, "latency_ms": {}}

    # ---- 1. parity
    print("\n== Parity vs pt_gpu (per graph, identical inputs) ==")
    print(f"{'graph':8} {'input':7} {'backend':7} {'max_abs':>10} {'mean_abs':>10} {'mse':>10} {'nmax':>10} {'cos':>12}  verdict")
    parity_ok = {}
    for graph, ins in inputs.items():
        for in_name, x in ins.items():
            y_ref = ref.encode(x) if graph == "encoder" else ref.decode(x)
            for name, b in backends.items():
                if name == "pt_gpu":
                    continue
                y = b.encode(x) if graph == "encoder" else b.decode(x)
                m = tensor_metrics(y, y_ref)
                tol = PARITY_TOL[PRECISION[name]]
                ok = m["nmax"] <= tol["nmax"] and m["cos"] >= tol["cos"]
                m["pass"] = ok
                parity_ok[name] = parity_ok.get(name, True) and ok
                report["parity"].setdefault(graph, {}).setdefault(in_name, {})[name] = m
                print(f"{graph:8} {in_name:7} {name:7} {m['max_abs']:10.3e} {m['mean_abs']:10.3e} "
                      f"{m['mse']:10.3e} {m['nmax']:10.3e} {m['cos']:12.9f}  {'PASS' if ok else 'FAIL'}")

    # ---- 2. accuracy (end-to-end round trip per backend)
    print(f"\n== Accuracy: astronaut {s}x{s}, encode->decode on each backend ==")
    recon = {}
    for name, b in backends.items():
        z, t_enc = timed(b.encode, x_real)
        y, t_dec = timed(b.decode, z)
        recon[name] = to_uint8(y)
        report["latency_ms"][name] = {"encoder": round(t_enc, 1), "decoder": round(t_dec, 1)}

    ref_vs_orig = image_metrics(recon["pt_gpu"], orig_u8)
    sanity_ok = ref_vs_orig["psnr"] >= SANITY_PSNR_VS_ORIG
    report["accuracy"]["pt_gpu"] = {"vs_orig": ref_vs_orig, "sanity_pass": sanity_ok}
    print(f"pt_gpu   vs original: PSNR {ref_vs_orig['psnr']:.2f} dB  MSE {ref_vs_orig['mse']:.2f} ({ref_vs_orig['mse_01']:.3e} on 0-1)  SSIM {ref_vs_orig['ssim']:.4f}  "
          f"(sanity >= {SANITY_PSNR_VS_ORIG} dB: {'PASS' if sanity_ok else 'FAIL'})")

    verdicts = {}
    for name in PRECISION:
        tol = ACC_TOL[PRECISION[name]]
        vo = image_metrics(recon[name], orig_u8)
        vr = image_metrics(recon[name], recon["pt_gpu"])
        dpsnr = abs(vo["psnr"] - ref_vs_orig["psnr"])
        acc_ok = (vr["psnr"] >= tol["psnr_vs_ref"] and vr["ssim"] >= tol["ssim_vs_ref"]
                  and dpsnr <= tol["dpsnr_vs_orig"])
        match = acc_ok and parity_ok[name] and sanity_ok
        verdicts[name] = "MATCH" if match else "MISMATCH"
        report["accuracy"][name] = {"vs_orig": vo, "vs_pt_gpu": vr, "dpsnr_vs_orig": dpsnr,
                                    "accuracy_pass": acc_ok, "parity_pass": parity_ok[name],
                                    "verdict": verdicts[name]}
        print(f"{name:8} vs original: PSNR {vo['psnr']:.2f} dB  MSE {vo['mse']:.2f} ({vo['mse_01']:.3e})  SSIM {vo['ssim']:.4f} | "
              f"vs pt_gpu: PSNR {vr['psnr']:.2f} dB  MSE {vr['mse']:.3e} ({vr['mse_01']:.3e})  SSIM {vr['ssim']:.5f}  dPSNR {dpsnr:.3f} | "
              f"{verdicts[name]}")

    # ---- outputs
    out = ART / "report"
    out.mkdir(parents=True, exist_ok=True)
    tiles = [orig_u8] + [recon[k] for k in backends]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out / "reconstructions.png")
    # One file per method, for side-by-side viewing in the README.
    (out / "images").mkdir(exist_ok=True)
    Image.fromarray(orig_u8).save(out / "images" / "original.png")
    for k in backends:
        Image.fromarray(recon[k]).save(out / "images" / f"{k}.png")
        if k != "pt_gpu":
            diff = np.abs(recon[k].astype(int) - recon["pt_gpu"].astype(int)) * 10
            Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(out / "images" / f"{k}_diff_x10.png")
    diffs = [np.clip(np.abs(recon[k].astype(int) - recon["pt_gpu"].astype(int)) * 10, 0, 255).astype(np.uint8)
             for k in PRECISION]
    Image.fromarray(np.concatenate(diffs, axis=1)).save(out / "abs_diff_x10_vs_pt_gpu.png")
    report["tile_order"] = {"reconstructions.png": ["original"] + list(backends),
                            "abs_diff_x10_vs_pt_gpu.png": list(PRECISION)}
    report["verdicts"] = verdicts
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\nLatency (ms, encoder/decoder):",
          {k: f"{v['encoder']}/{v['decoder']}" for k, v in report["latency_ms"].items()})
    print("\nFINAL:", verdicts)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
