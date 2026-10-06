"""Build fully-int8 (calibrated) VAE models from a folder of your own images.

    python calibrate.py --images path/to/your/images

What it does:
  1. Loads every image in --images (jpg/png/bmp/webp), resizes and center-crops it
     to 256x256. With --crops N it also adds N random crops per image.
  2. Runs those images through the fp32 encoder, and the resulting latents through
     the fp32 decoder, recording the value range of every layer. This step is the
     "calibration".
  3. Writes int8 models that use those ranges:
       --activations 8   int8 weights + int8 activations   (*_int8_static.tflite)
       --activations 16  int8 weights + int16 activations  (*_int8x16_static.tflite)
  4. Checks the new models on a test image that was NOT used for calibration,
     against the fp32 TFLite model, and prints MATCH or MISMATCH.

The fp32 TFLite model is the reference here. It was verified against the original
PyTorch model at 88 dB PSNR (see report/metrics.json), so PyTorch is not needed.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer, recipe
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIGNATURE = "serving_default"
GRAPHS = {"encoder": ("vae_encoder", "image"), "decoder": ("vae_decoder", "latent")}

# Same int8 limits as evaluate.py. Fixed in advance; do not tune them to results.
PARITY_TOL = {"nmax": 5e-2, "cos": 0.999}
ACC_TOL = {"psnr_vs_ref": 30.0, "ssim_vs_ref": 0.95, "dpsnr_vs_orig": 1.0}


# ---------------------------------------------------------------- images
def load_rgb(path, size, rng=None):
    """Resize shorter side to `size` and center-crop; with `rng`, take a random crop
    covering 60-100% of the shorter side instead."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    if rng is not None:
        side = int(side * rng.uniform(0.6, 1.0))
        left, top = rng.randint(0, w - side), rng.randint(0, h - side)
    else:
        left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
    return np.asarray(img)


def to_input(u8):
    return (u8.astype(np.float32) / 127.5 - 1.0)[None]  # NHWC, [-1, 1]


def to_uint8(x):
    return np.round((np.clip(x[0], -1, 1) + 1) * 127.5).astype(np.uint8)


def collect_images(folder, size, crops, seed):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images found in {folder} (looked for {sorted(IMAGE_EXTS)})")
    rng = random.Random(seed)
    batch = []
    for f in files:
        batch.append(load_rgb(f, size))
        batch += [load_rgb(f, size, rng) for _ in range(crops)]
    return files, batch


# ---------------------------------------------------------------- models
class Runner:
    def __init__(self, path, threads=8):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def __call__(self, x):
        self.it.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.it.invoke()
        return self.it.get_tensor(self.out["index"]).astype(np.float32)


def tensor_metrics(a, ref):
    d = np.abs(a.astype(np.float64) - ref.astype(np.float64))
    af, rf = a.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    return {"max_abs": float(d.max()), "mse": float((d ** 2).mean()),
            "nmax": float(d.max() / (np.abs(ref).max() + 1e-12)),
            "cos": float(af @ rf / (np.linalg.norm(af) * np.linalg.norm(rf) + 1e-12))}


def image_metrics(img, ref):
    return {"psnr": float(peak_signal_noise_ratio(ref, img, data_range=255)),
            "mse": float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2)),
            "mse_01": float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2) / 255.0 ** 2),
            "ssim": float(structural_similarity(ref, img, channel_axis=2, data_range=255))}


def find_models_dir():
    for cand in (HERE / "models" / "tflite", HERE / "artifacts" / "tflite"):
        if (cand / "vae_encoder" / "vae_encoder_float32.tflite").exists():
            return cand
    raise SystemExit("Could not find the fp32 .tflite models; pass --models")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with calibration images (searched recursively)")
    ap.add_argument("--test-image", default=str(HERE / "sample_images" / "test" / "astronaut.png"),
                    help="image used only for the final check (keep it out of --images)")
    ap.add_argument("--models", type=Path, default=None, help="folder holding vae_encoder/ and vae_decoder/")
    ap.add_argument("--out", type=Path, default=HERE / "calibrated_models")
    ap.add_argument("--activations", type=int, choices=(8, 16), nargs="+", default=[8, 16],
                    help="activation bit width(s) to build (default: both)")
    ap.add_argument("--crops", type=int, default=2, help="extra random crops per image (default 2)")
    ap.add_argument("--size", type=int, default=256, help="must match the models (256)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = args.models or find_models_dir()
    fp32 = {g: models / stem / f"{stem}_float32.tflite" for g, (stem, _) in GRAPHS.items()}
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. calibration data
    files, batch = collect_images(args.images, args.size, args.crops, args.seed)
    test_path = Path(args.test_image)
    if test_path.resolve() in {f.resolve() for f in files}:
        print(f"WARNING: test image {test_path.name} is also in the calibration folder; "
              "the check will look better than reality.")
    print(f"Calibration set: {len(files)} images x {1 + args.crops} crops = {len(batch)} samples")
    if len(files) < 10:
        print("NOTE: fewer than 10 images. For real use, 50-200 images that look like your "
              "real inputs give much more reliable ranges.")

    enc32, dec32 = Runner(fp32["encoder"]), Runner(fp32["decoder"])
    enc_samples = [{"image": to_input(u8)} for u8 in batch]
    dec_samples = [{"latent": enc32(s["image"])} for s in enc_samples]
    calib = {"encoder": enc_samples, "decoder": dec_samples}

    # 2-3. calibrate + quantize
    recipes = {8: ("int8_static", recipe.static_wi8_ai8), 16: ("int8x16_static", recipe.static_wi8_ai16)}
    built = {}
    for bits in args.activations:
        suffix, make_recipe = recipes[bits]
        for g, (stem, _) in GRAPHS.items():
            t = time.perf_counter()
            q = quantizer.Quantizer(str(fp32[g]), make_recipe())
            result = q.calibrate({SIGNATURE: calib[g]})
            path = args.out / f"{stem}_{suffix}.tflite"
            q.quantize(result).export_model(str(path), overwrite=True)
            built[(bits, g)] = path
            print(f"[{suffix}] {path.name}: {path.stat().st_size / 2**20:.1f} MiB "
                  f"({time.perf_counter() - t:.0f} s)")

    # 4. check on the held-out test image
    orig = load_rgb(test_path, args.size)
    x = to_input(orig)
    z_ref = enc32(x)
    y_ref = dec32(z_ref)
    ref_u8 = to_uint8(y_ref)
    ref_vs_orig = image_metrics(ref_u8, orig)
    report = {"calibration_images": [str(f) for f in files], "samples": len(batch),
              "test_image": str(test_path), "thresholds": {"parity": PARITY_TOL, "accuracy": ACC_TOL},
              "fp32_vs_original": ref_vs_orig, "variants": {}}
    print(f"\nTest image {test_path.name}: fp32 vs original PSNR {ref_vs_orig['psnr']:.2f} dB, "
          f"MSE {ref_vs_orig['mse']:.2f} ({ref_vs_orig['mse_01']:.3e} on 0-1), SSIM {ref_vs_orig['ssim']:.4f}")

    tiles = [orig, ref_u8]
    for bits in args.activations:
        suffix = recipes[bits][0]
        enc, dec = Runner(built[(bits, "encoder")]), Runner(built[(bits, "decoder")])
        par = {"encoder": tensor_metrics(enc(x), z_ref), "decoder": tensor_metrics(dec(z_ref), y_ref)}
        y = to_uint8(dec(enc(x)))  # int8 encoder -> int8 decoder, end to end
        vr, vo = image_metrics(y, ref_u8), image_metrics(y, orig)
        dpsnr = abs(vo["psnr"] - ref_vs_orig["psnr"])
        par_ok = all(m["nmax"] <= PARITY_TOL["nmax"] and m["cos"] >= PARITY_TOL["cos"] for m in par.values())
        acc_ok = (vr["psnr"] >= ACC_TOL["psnr_vs_ref"] and vr["ssim"] >= ACC_TOL["ssim_vs_ref"]
                  and dpsnr <= ACC_TOL["dpsnr_vs_orig"])
        verdict = "MATCH" if par_ok and acc_ok else "MISMATCH"
        report["variants"][suffix] = {"parity": par, "vs_fp32": vr, "vs_original": vo,
                                      "dpsnr_vs_orig": dpsnr, "parity_pass": par_ok,
                                      "accuracy_pass": acc_ok, "verdict": verdict}
        tiles.append(y)
        print(f"\n[{suffix}]  verdict: {verdict}")
        for g, m in par.items():
            print(f"  {g:8} vs fp32: nmax {m['nmax']:.3e} (limit {PARITY_TOL['nmax']})  "
                  f"cos {m['cos']:.6f} (limit {PARITY_TOL['cos']})  MSE {m['mse']:.3e}")
        print(f"  image vs fp32:     PSNR {vr['psnr']:.2f} dB (limit {ACC_TOL['psnr_vs_ref']})  "
              f"SSIM {vr['ssim']:.4f} (limit {ACC_TOL['ssim_vs_ref']})  MSE {vr['mse']:.2f} ({vr['mse_01']:.3e} on 0-1)")
        print(f"  image vs original: PSNR {vo['psnr']:.2f} dB (fp32 {ref_vs_orig['psnr']:.2f}, "
              f"drop {dpsnr:.2f}, limit {ACC_TOL['dpsnr_vs_orig']})  MSE {vo['mse']:.2f} ({vo['mse_01']:.3e} on 0-1)")

    Image.fromarray(np.concatenate(tiles, axis=1)).save(args.out / "calibration_check.png")
    report["tile_order"] = ["original", "fp32"] + [recipes[b][0] for b in args.activations]
    (args.out / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nModels, report and picture written to {args.out}")


if __name__ == "__main__":
    main()
