#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch.py — Batch driver untuk FlashVSR+.

Perbedaan vs memanggil `python run.py` per chunk:
  1. Model (DiT + TCDecoder + LQ_proj) dimuat SEKALI, dipakai untuk semua chunk.
     -> hemat ~1-2 menit per chunk (import torch + load weights + JIT triton).
  2. Resume otomatis: chunk yang outputnya sudah ada akan di-skip.
  3. Fail-fast: kalau satu chunk gagal, script berhenti dengan exit code != 0
     (supaya cell berikutnya di Colab tidak concat video yang bolong).
  4. Patch feather-mask: menghilangkan border hitam 1px pada mode --tiled-dit.
  5. FPS float: memakai avg_frame_rate asli dari chunk (bukan dibulatkan ke int),
     supaya audio tidak drift untuk video 23.976/29.97 fps.

Taruh file ini di root repo FlashVSR_plus (sebelah run.py).

Contoh:
  python run_batch.py --chunk-dir /content/chunks --out-dir /content/results \
      -m tiny -s 2 --tiled-dit --tile-size 384
"""

import os
import sys
import glob
import time
import argparse

parser = argparse.ArgumentParser(description="FlashVSR+ batch driver (model dimuat sekali).")
parser.add_argument("--chunk-dir", required=True, help="Folder berisi potongan video *.mp4")
parser.add_argument("--out-dir", required=True, help="Folder output hasil per chunk")
parser.add_argument("-m", "--mode", default="tiny", choices=["tiny", "tiny-long", "full"])
parser.add_argument("-s", "--scale", type=int, default=4)
parser.add_argument("-v", "--version", default="11", choices=["10", "11"])
parser.add_argument("--tiled-vae", action="store_true")
parser.add_argument("--tiled-dit", action="store_true")
parser.add_argument("--tile-size", type=int, default=256)
parser.add_argument("--overlap", type=int, default=24)
parser.add_argument("--unload-dit", action="store_true")
parser.add_argument("--color-fix", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-t", "--dtype", default="bf16", choices=["fp16", "bf16"])
parser.add_argument("-d", "--device", default="auto")
parser.add_argument("-q", "--quality", type=int, default=10)
parser.add_argument("--fp32-rope", action="store_true",
                    help="Hitung RoPE di float32 alih-alih float64 (jauh lebih cepat di GPU "
                         "consumer/L4; hasil hampir identik — uji A/B dulu di klip pendek).")
bargs = parser.parse_args()

# ---------------------------------------------------------------------------
# run.py mem-parse sys.argv saat di-import, jadi kita beri argv dummy yang valid
# ---------------------------------------------------------------------------
sys.argv = ["run.py", "-i", "placeholder.mp4", bargs.out_dir]

import torch  # noqa: E402
import run    # noqa: E402  (import berat: torch, model utils, dll — hanya sekali)

# ---------------------------------------------------------------------------
# PATCH 1: cache pipeline — init_pipeline hanya benar-benar jalan sekali
# ---------------------------------------------------------------------------
_pipe_cache = {}
_orig_init_pipeline = run.init_pipeline

def _cached_init_pipeline(version, mode, device, dtype):
    key = (version, mode, device, str(dtype))
    if key not in _pipe_cache:
        run.log(f"[Batch] Loading pipeline (mode={mode}, ver={version}) ...", message_type="info")
        _pipe_cache[key] = _orig_init_pipeline(version, mode, device, dtype)
    return _pipe_cache[key]

run.init_pipeline = _cached_init_pipeline

# ---------------------------------------------------------------------------
# PATCH 2: feather mask — hilangkan weight=0 di baris/kolom terluar tile.
# Tanpa patch ini, output --tiled-dit punya frame border hitam setebal 1px
# di keempat sisi video (weight 0 -> canvas 0 -> pixel hitam).
# clamp(min=eps) membuat normalisasi (tile*eps)/eps mengembalikan nilai asli.
# ---------------------------------------------------------------------------
_orig_feather_mask = run.create_feather_mask

def _feather_mask_no_black_border(size, overlap):
    return _orig_feather_mask(size, overlap).clamp(min=1e-3)

run.create_feather_mask = _feather_mask_no_black_border

# ---------------------------------------------------------------------------
# PATCH 3: fps presisi float dari container (run.py membulatkan ke int,
# yang membuat 29.97 -> 30 dan audio drift ~3.6 detik per jam).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PATCH 4 (opt-in, --fp32-rope): rope_apply asli meng-cast q/k ke float64 dan
# melakukan perkalian complex128 di SETIAP block (30x) di SETIAP step.
# Throughput FP64 di L4/GPU consumer ~1/64 dari FP32, jadi ini hot-spot nyata.
# Versi fp32 memberi hasil yang secara praktis identik (beda numerik ~1e-6),
# tapi tetap uji A/B pada satu chunk sebelum dipakai penuh.
# ---------------------------------------------------------------------------
if bargs.fp32_rope:
    from einops import rearrange as _rearrange
    from src.models import wan_video_dit as _dit

    def _rope_apply_fp32(x, freqs, num_heads):
        # freqs dibangun ulang tiap step, jadi konversi complex64 dilakukan per call
        # (tensornya kecil; biayanya nol dibanding perkalian complex128 yang dihindari)
        f32 = freqs.to(torch.complex64)
        x = _rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        x_out = torch.view_as_complex(
            x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
        )
        x_out = torch.view_as_real(x_out * f32).flatten(2)
        return x_out.to(x.dtype)

    _dit.rope_apply = _rope_apply_fp32
    run.log("[Batch] fp32 RoPE aktif.", message_type="info")


def probe_fps(path, fallback=30.0):
    try:
        import ffmpeg
        info = ffmpeg.probe(path)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        num, den = v.get("avg_frame_rate", "0/0").split("/")
        num, den = float(num), float(den)
        if den > 0 and num > 0:
            return num / den
    except Exception:
        pass
    return fallback


def main():
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[bargs.dtype]

    chunks = sorted(glob.glob(os.path.join(bargs.chunk_dir, "*.mp4")))
    if not chunks:
        run.log(f"[Batch] Tidak ada *.mp4 di {bargs.chunk_dir}", message_type="error")
        sys.exit(1)

    os.makedirs(bargs.out_dir, exist_ok=True)
    t0 = time.time()
    done, skipped = 0, 0

    for i, f in enumerate(chunks):
        name = os.path.basename(f).rsplit(".", 1)[0]
        final = os.path.join(bargs.out_dir, f"FlashVSR_{bargs.mode}_{name}_{bargs.seed}.mp4")

        # Resume: skip chunk yang sudah selesai
        if os.path.exists(final) and os.path.getsize(final) > 0:
            run.log(f"[Batch] ({i+1}/{len(chunks)}) skip (sudah ada): {final}")
            skipped += 1
            continue

        run.log(f"[Batch] ({i+1}/{len(chunks)}) Processing: {f}", message_type="finish")
        t_chunk = time.time()
        try:
            result, _ = run.main(
                f, bargs.version, bargs.mode, bargs.scale, bargs.color_fix,
                bargs.tiled_vae, bargs.tiled_dit, bargs.tile_size, bargs.overlap,
                bargs.unload_dit, dtype, seed=bargs.seed, device=bargs.device,
                quality=bargs.quality, output=final,
            )
            if bargs.mode != "tiny-long":
                fps = probe_fps(f)
                run.save_video(result, final, fps=fps, quality=bargs.quality)
            del result
            run.clean_vram()
        except Exception as e:
            import traceback
            traceback.print_exc()
            run.log(f"[Batch] GAGAL di chunk {f}: {type(e).__name__}: {e}", message_type="error")
            run.log("[Batch] Berhenti. Jalankan ulang cell ini untuk resume dari chunk gagal.",
                    message_type="warning")
            sys.exit(2)

        run.log(f"[Batch] Chunk selesai dalam {time.time() - t_chunk:.1f}s")
        done += 1

    run.log(f"[Batch] Selesai: {done} diproses, {skipped} di-skip, "
            f"total {(time.time() - t0)/60:.1f} menit.", message_type="finish")


if __name__ == "__main__":
    main()
