#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_patches.py — Menerapkan perbaikan core FlashVSR+ ke repo yang sudah di-clone.

Jalankan dari ROOT repo (sebelah run.py), SETELAH `git clone`, SEBELUM menjalankan
run.py / run_batch.py:

    python apply_patches.py

Idempotent: aman dijalankan berkali-kali (melewati bagian yang sudah ditambal).
Memperbaiki:
  1. OOM GPU saat VAE decode (mode tiny)  -> decode ber-window bit-identik.
  2. OOM GPU saat VAE decode (mode full)  -> akumulasi output di-offload ke CPU.
  3. color-fix tiny: bounded + tidak menelan error.
  4. tiny-long stitcher: dekode O(N^2) -> O(N).
  5. Border hitam 1px pada output tiled.
  6. fps float (audio 29.97 tak drift) + count_frames tidak eager.
"""

import io, os, sys

def _read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()

def _write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)

def patch(path, old, new, label, sentinel):
    if not os.path.exists(path):
        print(f"  [LEWAT] {path} tidak ada — {label}")
        return
    s = _read(path)
    if sentinel in s:
        print(f"  [SUDAH] {label}")
        return
    if old not in s:
        print(f"  [!!!!!] pola tidak ditemukan untuk: {label}\n"
              f"         (versi repo mungkin berbeda; laporkan ini)")
        return
    _write(path, s.replace(old, new, 1))
    print(f"  [OK]    {label}")


# ---------------------------------------------------------------------------
# 1) TCDecoder.py — tambah decode_video_chunked (bounded, bit-identik)
# ---------------------------------------------------------------------------
TCD = "src/models/TCDecoder.py"
tcd_old = '''    def forward(self, *args, **kwargs):
        raise NotImplementedError("Decoder-only model: call decode_video(...) instead.")

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)'''
tcd_new = '''    def decode_video_chunked(self, x, cond=None, window=8, show_progress_bar=False, to_cpu=True):
        """Bounded-memory, bit-identical equivalent of decode_video.
        Decodes latents in temporal windows while carrying the causal MemBlock
        state (self.mem) across windows, so the full-res output never lives on
        the GPU all at once. Prevents CUDA OOM at torch.stack for long clips."""
        self.clean_mem()
        T = x.shape[1]
        outs = []
        rng = range(0, T, window)
        if show_progress_bar:
            rng = tqdm(rng, desc="[FlashVSR] VAE decoding")
        for s in rng:
            e = min(s + window, T)
            xw = x[:, s:e]
            cw = cond[:, :, s * 4:e * 4] if cond is not None else None
            y = self.decode_video(xw, parallel=False, show_progress_bar=False, cond=cw)
            outs.append(y.to("cpu") if to_cpu else y)
        return torch.cat(outs, dim=1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Decoder-only model: call decode_video(...) instead.")

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)'''

# ---------------------------------------------------------------------------
# 2) flashvsr_tiny.py — pakai decode ber-window + param decode_window + color-fix bounded
# ---------------------------------------------------------------------------
TINY = "src/pipelines/flashvsr_tiny.py"
tiny_sig_old = '''        color_fix = True,
        unload_dit = False,
        force_offload = False,
        **kwargs,
    ):
        # 只接受 cfg=1.0（与原代码一致）
        assert cfg_scale == 1.0, "cfg_scale must be 1.0"'''
tiny_sig_new = '''        color_fix = True,
        unload_dit = False,
        force_offload = False,
        decode_window = 8,
        **kwargs,
    ):
        # 只接受 cfg=1.0（与原代码一致）
        assert cfg_scale == 1.0, "cfg_scale must be 1.0"'''

tiny_dec_old = '''            latents = torch.cat(latents_total, dim=2)
            
            # Decode
            print("[FlashVSR] Starting VAE decoding...")
            frames = self.TCDecoder.decode_video(latents.transpose(1, 2),parallel=False, show_progress_bar=False, cond=LQ_video[:,:,:LQ_cur_idx,:,:]).transpose(1, 2).mul_(2).sub_(1)
            
            self.TCDecoder.clean_mem()
            if force_offload:
                self.offload_model()
                
            # 颜色校正（wavelet）
            try:
                if color_fix:
                    frames = self.ColorCorrector(
                        frames.to(device=LQ_video.device),
                        LQ_video[:, :, :frames.shape[2], :, :],
                        clip_range=(-1, 1),
                        chunk_size=16,
                        method='adain'
                    )
            except:
                pass
                
        return frames[0]'''
tiny_dec_new = '''            latents = torch.cat(latents_total, dim=2)
            
            # Decode — windowed (bit-identical) so full-res output never sits on
            # the GPU all at once. Prevents CUDA OOM on long chunks / large tiles.
            print("[FlashVSR] Starting VAE decoding...")
            frames = self.TCDecoder.decode_video_chunked(
                latents.transpose(1, 2),
                cond=LQ_video[:, :, :LQ_cur_idx, :, :],
                window=decode_window, show_progress_bar=False, to_cpu=True
            ).transpose(1, 2).mul_(2).sub_(1)
            
            self.TCDecoder.clean_mem()
            if force_offload:
                self.offload_model()
                
            # Color correction (adain) — chunked so we never move the whole
            # full-res clip to the GPU at once. Errors reported, not swallowed.
            if color_fix:
                try:
                    dev = LQ_video.device
                    T = frames.shape[2]
                    cf_out = []
                    for s in range(0, T, 16):
                        e = min(s + 16, T)
                        fc = frames[:, :, s:e].to(dev)
                        lc = LQ_video[:, :, s:e]
                        oc = self.ColorCorrector(fc, lc, clip_range=(-1, 1),
                                                 chunk_size=None, method='adain')
                        cf_out.append(oc.to(frames.device))
                        del fc, lc, oc
                    frames = torch.cat(cf_out, dim=2)
                except Exception as _e:
                    print(f"[FlashVSR] color-fix skipped: {_e}")
                
        return frames[0]'''

# ---------------------------------------------------------------------------
# 3) wan_video_vae.py — bound akumulasi decode full ke CPU
# ---------------------------------------------------------------------------
VAE = "src/models/wan_video_vae.py"
vae_old = '''        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :],
                                   feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2) # may add tensor offload
        return out'''
vae_new = '''        iter_ = z.shape[2]
        x = self.conv2(z)
        # Accumulate decoded frames on CPU (numerically identical; per-frame decode
        # and causal feat_cache unchanged) so full-res clip never fills the GPU.
        outs = []
        for i in range(iter_):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx)
            outs.append(out_.to("cpu"))  # bounded-memory offload
            del out_
        return torch.cat(outs, 2)'''

# ---------------------------------------------------------------------------
# 4) run.py — feather border (torch), stitcher O(N^2) + border (numpy), fps/count
# ---------------------------------------------------------------------------
RUN = "run.py"

run_feather_old = '''    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask'''
run_feather_new = '''    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    # clamp min so image-boundary pixels don't get zero weight -> no black border
    return mask.clamp(min=1e-3)'''

run_iter_old = '''        # 打开最终的写入器
        with imageio.get_writer(output_path, fps=fps, quality=quality) as writer:
            
            # 2. 按 chunk_size 遍历所有帧
            # tqdm 现在描述的是处理了多少个“块”
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):'''
run_iter_new = '''        # Persistent sequential iterators per tile -> O(N) instead of O(N^2)
        # (old code re-decoded every tile from frame 0 for each chunk).
        iters = [r.iter_data() for r in readers]

        # 打开最终的写入器
        with imageio.get_writer(output_path, fps=fps, quality=quality) as writer:
            
            # 2. 按 chunk_size 遍历所有帧
            # tqdm 现在描述的是处理了多少个“块”
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):'''

run_read_old = '''                    try:
                        # get_reader().iter_data() 是高效读取连续帧的方式
                        tile_chunk_frames = [
                            frame.astype(np.float32) / 255.0 
                            for idx, frame in enumerate(reader.iter_data()) 
                            if start_frame <= idx < end_frame
                        ]
                        # 将帧列表转换为一个 NumPy 数组
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        log(f"Warning: Could not read chunk from tile {i}. Error: {e}", message_type='warning')
                        continue'''
run_read_new = '''                    try:
                        tile_chunk_frames = []
                        for _ in range(current_chunk_size):
                            frame = next(iters[i])
                            tile_chunk_frames.append(frame.astype(np.float32) / 255.0)
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except StopIteration:
                        if tile_chunk_frames:
                            tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                        else:
                            log(f"Warning: tile {i} ran out of frames early.", message_type='warning')
                            continue
                    except Exception as e:
                        log(f"Warning: Could not read chunk from tile {i}. Error: {e}", message_type='warning')
                        continue'''

run_nmask_old = '''                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    # 扩展蒙版以匹配 chunk 的帧数维度
                    mask_4d = mask[np.newaxis, :, :, :] # 形状: (1, H, W, C)'''
run_nmask_new = '''                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    mask = np.clip(mask, 1e-3, None)  # no black border at image edge
                    # 扩展蒙版以匹配 chunk 的帧数维度
                    mask_4d = mask[np.newaxis, :, :, :] # 形状: (1, H, W, C)'''

run_fps_old = '''        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
        
        total = meta.get('nframes', rdr.count_frames())
        if total is None or total <= 0 :'''
run_fps_new = '''        fps_val = meta.get('fps', 30)
        # keep fps float: rounding 29.97 -> 30 drifts audio ~3.6s/hour
        fps = float(fps_val) if isinstance(fps_val, (int, float)) and fps_val > 0 else 30.0
        
        # lazy: only count_frames() (full decode) when metadata lacks nframes
        total = meta.get('nframes')
        if not total or total <= 0:
            total = rdr.count_frames()
        if total is None or total <= 0 :'''


# ---------------------------------------------------------------------------
# 7) run.py — canvas tiled hemat RAM (mengatasi SYSTEM-RAM OOM di chunk panjang).
#    (a) weight_sum_canvas cukup 1 frame (bobot identik tiap frame) -> broadcast.
#    (b) pembagian in-place -> tak ada alokasi klip-penuh tambahan.
#    Puncak RAM canvas: ~3 klip-penuh -> ~1. Hasil bit-identik (diverifikasi).
# ---------------------------------------------------------------------------
run_wcalloc_old = '''            weight_sum_canvas = torch.zeros_like(final_output_canvas)'''
run_wcalloc_new = '''            # weight identik untuk tiap frame -> simpan 1 frame saja (broadcast).
            weight_sum_canvas = torch.zeros((1, H * scale, W * scale, C), dtype=dtype, device="cpu")'''

run_wcdiv_old = '''            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output = final_output_canvas / weight_sum_canvas'''
run_wcdiv_new = '''            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output_canvas.div_(weight_sum_canvas)  # in-place, broadcast (1,H,W,C)
            final_output = final_output_canvas'''


def main():
    print("Menerapkan patch core FlashVSR+ ...")
    patch(RUN, run_wcalloc_old, run_wcalloc_new, "run.py: weight canvas 1-frame (RAM)", "weight identik untuk tiap frame")
    patch(RUN, run_wcdiv_old, run_wcdiv_new, "run.py: pembagian canvas in-place (RAM)", "in-place, broadcast")
    patch(TCD, tcd_old, tcd_new, "TCDecoder: decode_video_chunked", "def decode_video_chunked")
    patch(TINY, tiny_sig_old, tiny_sig_new, "tiny: param decode_window", "decode_window = 8,")
    patch(TINY, tiny_dec_old, tiny_dec_new, "tiny: windowed decode + color-fix", "decode_video_chunked(")
    patch(VAE, vae_old, vae_new, "full VAE: bounded decode", "bounded-memory offload")
    patch(RUN, run_feather_old, run_feather_new, "run.py: feather border (torch)", "no black border")
    patch(RUN, run_iter_old, run_iter_new, "run.py: stitcher persistent iters", "iters = [r.iter_data() for r in readers]")
    patch(RUN, run_read_old, run_read_new, "run.py: stitcher sequential read", "next(iters[i])")
    patch(RUN, run_nmask_old, run_nmask_new, "run.py: stitcher border (numpy)", "np.clip(mask, 1e-3, None)")
    patch(RUN, run_fps_old, run_fps_new, "run.py: fps float + lazy count_frames", "and fps_val > 0 else 30.0")

    # verifikasi sintaks
    import ast
    ok = True
    for p in [TCD, TINY, VAE, RUN]:
        if os.path.exists(p):
            try:
                ast.parse(_read(p))
            except SyntaxError as e:
                ok = False
                print(f"  [SYNTAX ERROR] {p}: {e}")
    print("Selesai." if ok else "SELESAI DENGAN ERROR SINTAKS — jangan lanjut, laporkan.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
