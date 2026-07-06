# FlashVSR+ core patches

Basis: commit terbaru dari github.com/antorio/FlashVSR_plus (identik dengan zip).
Semua perubahan diverifikasi tidak mengubah hasil (bit-identik) atau bounded-memory.
Cukup timpa 4 file ini di repo, tidak perlu ubah lain.

## 1. OOM VAE decoding di mode `tiny` — PENYEBAB CRASH-MU (FIXED)

**Gejala:** `CUDA out of memory` di `TCDecoder.decode_video` -> `torch.stack(out, 1)`,
tepat setelah "Starting VAE decoding", untuk chunk panjang / tile besar.

**Sebab:** pipeline `tiny` menumpuk seluruh latent chunk lalu men-decode-nya dalam
SATU panggilan. Di akhir decoder, `torch.stack(out, 1)` memaksa seluruh frame
full-res chunk ada di GPU serentak.

**Fix:**
- `src/models/TCDecoder.py`: tambah `decode_video_chunked()` — men-decode latent
  dalam window kecil (default 8 latent-frame) sambil mempertahankan memory-block
  kausal (`self.mem`) antar-window, dan meng-offload output tiap window ke CPU.
  Karena decoder murni kausal di sumbu waktu, hasilnya **bit-identik** dengan
  decode sekaligus (diverifikasi numerik: max abs diff = 0 untuk window 1/4/8/16).
- `src/pipelines/flashvsr_tiny.py`: pakai `decode_video_chunked(window=decode_window)`.
  `decode_window` jadi parameter `__call__` (default 8).

Ini menghilangkan OOM decode terlepas dari panjang chunk / ukuran tile.

## 2. OOM VAE decoding di mode `full` (FIXED)

`src/models/wan_video_vae.py`, `decode(z, scale)`: loop per-frame menumpuk output
di GPU (`out = torch.cat([out, out_], 2)`; komentar aslinya sudah menandai
"may add tensor offload"). Diubah agar tiap frame di-offload ke CPU lalu di-cat.
Nilai identik (decode per-frame & feat_cache kausal tak berubah); GPU tak lagi
menampung seluruh klip.

## 3. Color-fix (`--color-fix`) tak lagi OOM & tak lagi menelan error (FIXED)

`flashvsr_tiny.py`: dulu `frames.to(device=LQ_video.device)` memindah SELURUH klip
full-res ke GPU, lalu `except: pass` menelan kegagalan diam-diam. Sekarang diproses
per-16-frame (bounded), dan error dilaporkan, bukan disembunyikan.

## 4. Mode `tiny-long` stitcher — dekode O(N^2) (FIXED)

`run.py`, `stitch_video_tiles()`: untuk tiap chunk frame, kode lama memanggil
`reader.iter_data()` baru dan memfilter dari frame 0 — mendekode ulang SETIAP tile
dari awal untuk setiap chunk (kuadratik untuk video panjang). Diganti iterator
persisten per tile yang maju berurutan -> O(N).

## 5. Border hitam 1px pada output tiled (`--tiled-dit`) (FIXED)

Feather mask memakai `linspace(0,1,...)` yang bernilai 0 di tepi terluar gambar,
sehingga pixel bertutup satu tile mendapat weight 0 -> hitam setelah normalisasi.
Di-clamp ke min 1e-3 di dua tempat: `create_feather_mask` (torch, jalur tiny/full
tiled) dan mask numpy di `stitch_video_tiles` (jalur tiny-long tiled). Pixel
single-coverage kini mengembalikan nilai tile-nya; overlap antar-tile tak berubah.

## 6. Perbaikan kecil di `run.py > prepare_tensors` (FIXED)

- `fps` dijaga float, bukan `int(round(...))` — pembulatan 29.97 -> 30 membuat audio
  drift ~3.6 detik per jam untuk sumber NTSC.
- `count_frames()` (dekode penuh) dulu selalu dieksekusi karena ditaruh sebagai
  argumen-default; kini hanya dipanggil bila metadata tak punya `nframes`.

## Catatan pemakaian

- Tak ada perubahan CLI yang wajib. `decode_window=8` sudah menyelesaikan OOM.
  Kalau VRAM sangat lega dan ingin decode sedikit lebih cepat, window bisa
  dinaikkan; kalau masih mepet, turunkan.
- Fix-fix ini bekerja pada resolusi & durasi chunk berapa pun (batasnya kini
  RAM sistem untuk canvas tiled di run.py, bukan lagi GPU decode).
