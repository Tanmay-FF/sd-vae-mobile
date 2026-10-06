"""Convert the AutoencoderKL weights in ./models to ONNX, then to TFLite.

The VAE is split into two static-shape graphs, the usual layout for on-device
Stable Diffusion:
  encoder: image  [1,3,H,W] in [-1,1]  -> latent mean [1,4,H/8,W/8]
  decoder: latent [1,4,H/8,W/8]        -> image [1,3,H,W] in ~[-1,1]
The 0.18215 scaling factor is left out of both graphs, and the encoder returns
the posterior mean, so no sampling happens inside the graph.

Outputs go to ./artifacts/{onnx,tflite}.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import onnx
import torch
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
from onnxsim import simplify

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "artifacts"


class Encoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        moments = self.quant_conv(self.encoder(x))
        mean, _logvar = torch.chunk(moments, 2, dim=1)
        return mean


class Decoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        return self.decoder(self.post_quant_conv(z))


def load_vae():
    vae, info = AutoencoderKL.from_pretrained(
        MODEL_DIR, use_safetensors=True, output_loading_info=True
    )
    bad = {k: v for k, v in info.items() if v}
    if bad:
        raise RuntimeError(f"weights did not load cleanly: {bad}")
    # Plain matmul/softmax attention converts more predictably than SDPA.
    vae.set_attn_processor(AttnProcessor())
    return vae.eval()


def export_onnx(module, example, path, in_name, out_name, opset):
    torch.onnx.export(
        module,
        (example,),
        str(path),
        input_names=[in_name],
        output_names=[out_name],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    model = onnx.load(str(path))
    model, ok = simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim failed to validate {path}")
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"[onnx] {path.name}: {len(model.graph.node)} nodes, ops={ops}")


def to_tflite(onnx_path, out_dir):
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(out_dir),
        # flatbuffer_direct (onnx2tf's default). The tf_converter backend gets
        # the mid-block attention Reshape/Transpose layout wrong (cos ~0.97 on
        # that block alone), which corrupts the whole VAE output.
        "-tb", "flatbuffer_direct",
        "-nuo",  # graph is already simplified in export_onnx
    ]
    print("[tflite]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # onnx2tf's flatbuffer_direct *_float16.tflite is a fully fp16 graph (fp16 I/O
    # and activations) that only fp16-capable delegates/NPUs can run. Keep it under
    # its own name; *_float16.tflite is rebuilt as fp16-weights in quantize_variants.
    name = Path(onnx_path).stem
    (out_dir / f"{name}_float16.tflite").replace(out_dir / f"{name}_float16_full.tflite")


def quantize_variants(out_dir, name):
    """Derive the quantized models from the verified fp32 .tflite.

      *_float16.tflite           fp16 CONV/FC weights + DEQUANTIZE, fp32 activations/I/O
                                 (the standard mobile fp16 model: CPU/XNNPACK and GPU delegate)
      *_int8_weight_only.tflite  int8 per-channel weights + DEQUANTIZE, fp32 activations/I/O
      *_int8_dynamic.tflite      int8 per-channel weights, activations quantized to int8
                                 on the fly for int8 compute (CPU), fp32 I/O
    """
    from ai_edge_quantizer import qtyping, quantizer, recipe
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    fp32 = str(out_dir / f"{name}_float32.tflite")

    fp16 = quantizer.Quantizer(fp32)
    fp16.add_weight_only_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=16,
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    variants = {
        "float16": fp16,
        "int8_weight_only": quantizer.Quantizer(fp32, recipe.weight_only_wi8_afp32()),
        "int8_dynamic": quantizer.Quantizer(fp32, recipe.dynamic_wi8_afp32()),
    }
    for suffix, q in variants.items():
        path = out_dir / f"{name}_{suffix}.tflite"
        q.quantize().export_model(str(path), overwrite=True)
        print(f"[quant] {path.name}: {path.stat().st_size / 2**20:.1f} MiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256, help="image side (multiple of 8)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quant-only", action="store_true",
                    help="only rebuild the fp16/int8 models from the existing fp32 .tflite")
    args = ap.parse_args()

    onnx_dir = OUT_DIR / "onnx"
    tfl_dir = OUT_DIR / "tflite"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    names = ("vae_encoder", "vae_decoder")
    if args.quant_only:
        for name in names:
            quantize_variants(tfl_dir / name, name)
        return

    vae = load_vae()
    s, ls = args.size, args.size // 8
    with torch.no_grad():
        export_onnx(Encoder(vae), torch.randn(1, 3, s, s), onnx_dir / "vae_encoder.onnx",
                    "image", "latent", args.opset)
        export_onnx(Decoder(vae), torch.randn(1, 4, ls, ls), onnx_dir / "vae_decoder.onnx",
                    "latent", "image", args.opset)

    for name in names:
        to_tflite(onnx_dir / f"{name}.onnx", tfl_dir / name)
        quantize_variants(tfl_dir / name, name)


if __name__ == "__main__":
    main()
