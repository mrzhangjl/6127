# CA6127 — VAR vs. DiT comparative study

## Files
| File | Purpose |
|---|---|
| `VAR_vs_DiT_CA6127.ipynb` | The deliverable. Upload to Colab, Runtime → T4 GPU, Run all. |
| `var_vs_dit_pipeline.py` | Same code as a flat `# %%`-delimited script (easier to diff/read). |
| `report.tex` | IEEEtran report. Compiles clean with pdfLaTeX; no BibTeX pass needed. |

## Run order
1. **Cells 0.1–0.3** — GPU check, pip installs, clone `FoundationVision/VAR`. Restart is *not* required.
2. **Cell 2.1–2.2** — put your 50–100 collected images in `./custom_dataset/raw/<class_name>/`
   (folder names must match the 10 keys in `CLASSES`), then run the normaliser.
   **This is the one manual step.** Everything else is automatic.
3. **Cells 3–10** — run straight through. Expect ~60–90 min end-to-end on a free T4
   (the guidance sweep in 8.4 is the long pole; drop `SWEEP_N` to 10 to halve it).
4. `outputs/results/tables.tex` contains ready-to-paste table bodies, and
   `outputs/figures/*.png` the four report figures. Copy `outputs/figures/` next to
   `report.tex` before compiling.

## Filling in the report
Every number in `report.tex` is wrapped in `\R{...}` and renders **red**. Replace each with
your measured value from `results.json` and delete the `\newcommand{\R}` line — if anything
is still red, you missed it.

## Things worth knowing before you run
- **T4 = Turing = no bf16.** All mixed precision is fp16. `torch.cuda.is_bf16_supported()`
  returns `False` on this GPU; using bf16 silently falls back and is slower than fp32.
- **VAR is not in `diffusers`** — it needs the GitHub repo plus `.pth` weights from
  `FoundationVision/var` on the HF hub. Cell 0.3 handles it.
- **Do not call `.half()` on VAR.** The residual quantiser accumulates `f_hat` over 10
  scales; in fp16 that accumulator drifts and you get colour banding. The notebook keeps
  fp32 weights under `torch.autocast(float16)` instead — same speed, no artefact.
- **The SD-VAE decoder is kept in fp32** (`decode_vae_fp32=True`). In fp16 it overflows and
  emits black tiles. It costs ~340 MB, which you have.
- **FID on 86 real images is biased, badly.** The notebook reports KID alongside and the
  report says so explicitly in §III-A. Do not compare these FIDs to published ImageNet
  numbers — a marker will notice. If you can get to 500+ reference images, do.
- **Both models are class-conditional, not text-conditional.** CLIP-Score here measures
  class-prompt alignment via the template `"a photo of a {class}"`, which is a semantic
  fidelity proxy, not text-following. Stated as such in the report.
- `enable_model_cpu_offload()` is available and benchmarked in Cell 8.6, but is **off by
  default** — at 256² on a T4 it costs ~28% latency for VRAM you aren't short of. Keeping it
  as a measured ablation is worth more marks than using it blindly.

## If you hit OOM
Lower `CFG.batch_dit` (8 → 4) and `CFG.batch_var` (16 → 8). If VAR still OOMs, you are
probably on `var_depth=30` (2B params) — use 16 or 20.
