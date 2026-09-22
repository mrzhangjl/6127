# %% [markdown]
# # CA6127 Project — VAR (Next-Scale Autoregression) vs. DiT (Latent Diffusion Transformer)
#
# **Comparative analysis of two generative paradigms on a self-built dataset.**
#
# | | VAR-d16 | DiT-XL/2-256 |
# |---|---|---|
# | Paradigm | Discrete next-**scale** autoregression | Continuous latent denoising |
# | Backbone | GPT-2 style causal Transformer + multi-scale VQVAE | Transformer on patchified latents (adaLN-Zero) |
# | Conditioning | ImageNet-1k class embedding | ImageNet-1k class embedding |
# | Tokens / step | 1,4,9,...,256 (680 total, 10 forward passes) | 256 latent patches x NFE forward passes |
#
# **Hardware target:** Google Colab Free Tier, NVIDIA T4 (16 GB, compute capability 7.5).
#
# > **Important T4 note:** the T4 is a *Turing* GPU. It has **no bfloat16 support** and no
# > Flash-Attention kernels. All mixed precision below is **FP16**. Using `bfloat16` will either
# > throw or silently fall back to FP32 emulation and be slow.
#
# Run the cells in order. Sections 0-3 are setup, 4-6 are inference/benchmarking,
# 7-9 are metrics, 10-12 are the sweeps, plots and LaTeX export.

# %% [markdown]
# ## 0. Environment setup

# %%
# --- Cell 0.1: sanity check the GPU we were allocated -------------------------
import subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"Device        : {torch.cuda.get_device_name(0)}")
    print(f"Capability    : sm_{cap[0]}{cap[1]}")
    print(f"Total VRAM    : {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
else:
    raise RuntimeError("No GPU. Runtime > Change runtime type > T4 GPU.")

# %%
# --- Cell 0.2: dependencies ---------------------------------------------------
# diffusers  : DiT pipeline + schedulers
# clean-fid  : FID / KID with the standardised (anti-aliased) resizing pipeline
# torchmetrics[image] : CLIPScore + a second FID implementation for cross-checking
# timm, huggingface_hub : required by the official VAR repository
get_ipython().system('pip -q install "diffusers==0.31.0" "transformers>=4.44" accelerate safetensors')
get_ipython().system('pip -q install clean-fid "torchmetrics[image]>=1.4" timm huggingface_hub')
get_ipython().system('pip -q install matplotlib pandas scipy ftfy regex')
print("deps installed")

# %%
# --- Cell 0.3: clone the official VAR implementation --------------------------
# VAR is NOT distributed through `diffusers`; the reference implementation lives on GitHub
# and the weights on the HuggingFace hub (FoundationVision/var).
import os, sys
if not os.path.isdir("/content/VAR"):
    get_ipython().system('git clone -q https://github.com/FoundationVision/VAR.git /content/VAR')
if "/content/VAR" not in sys.path:
    sys.path.insert(0, "/content/VAR")
print(os.listdir("/content/VAR")[:12])

# %% [markdown]
# ## 1. Global configuration
#
# Everything that the report's ablations touch is centralised here so that a single
# `CFG` object fully determines a run (reproducibility requirement of the rubric).

# %%
# --- Cell 1.1: configuration --------------------------------------------------
import os, json, time, math, random, gc
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch

@dataclass
class CFG:
    # --- paths -------------------------------------------------------------
    root: str = "/content"
    real_dir: str = "./custom_dataset/real_images"   # self-built reference set
    var_out: str = "./outputs/var"
    dit_out: str = "./outputs/dit"
    fig_dir: str = "./outputs/figures"
    res_dir: str = "./outputs/results"
    # --- models ------------------------------------------------------------
    var_depth: int = 16            # 16 / 20 / 24 / 30. d16=310M params, safest on a T4.
    dit_id: str = "facebook/DiT-XL-2-256"
    clip_id: str = "openai/clip-vit-base-patch16"
    # --- generation --------------------------------------------------------
    image_size: int = 256
    n_per_class: int = 50          # generated images per class -> 10*50 = 500 per model
    batch_var: int = 16
    batch_dit: int = 8
    # --- default sampling hyper-parameters ---------------------------------
    var_cfg: float = 1.5           # classifier-free guidance strength for VAR
    var_top_k: int = 900
    var_top_p: float = 0.96
    dit_nfe: int = 50              # denoising steps (NFE)
    dit_cfg: float = 4.0
    dit_scheduler: str = "ddim"    # "ddim" for the NFE sweep, "ddpm" = paper default
    # --- precision / memory ------------------------------------------------
    use_cpu_offload: bool = False  # measured as an ablation in Section 6.3
    decode_vae_fp32: bool = True   # the SD-VAE decoder overflows in FP16 -> black tiles
    seed: int = 0

CFG = CFG()
os.chdir(CFG.root)
for d in [CFG.real_dir, CFG.var_out, CFG.dit_out, CFG.fig_dir, CFG.res_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)

# The 10 categories of the self-built dataset, mapped to their ImageNet-1k indices.
# Both models are class-conditional on ImageNet-1k, so the label space is shared and the
# comparison is strictly controlled (identical conditioning signal, identical label ids).
CLASSES = {
    "golden_retriever": (207, "a photo of a golden retriever"),
    "tabby_cat":        (281, "a photo of a tabby cat"),
    "macaw":            (88,  "a photo of a macaw"),
    "coral_reef":       (973, "a photo of a coral reef"),
    "volcano":          (980, "a photo of a volcano"),
    "lakeside":         (975, "a photo of a lakeside"),
    "daisy":            (985, "a photo of a daisy"),
    "sports_car":       (817, "a photo of a sports car"),
    "espresso":         (967, "a photo of an espresso"),
    "balloon":          (417, "a photo of a balloon"),
}
CLASS_NAMES = list(CLASSES.keys())
CLASS_IDS   = [CLASSES[c][0] for c in CLASS_NAMES]
CLASS_TEXTS = [CLASSES[c][1] for c in CLASS_NAMES]

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def free():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()

set_seed(CFG.seed)
print(json.dumps(asdict(CFG), indent=2))

# %% [markdown]
# ## 2. Self-built dataset
#
# Collect **50-100 high-quality reference images** (5-10 per category), upload them to
# `./custom_dataset/raw/<class_name>/`, then run the normaliser below. It center-crops to a
# square and resizes to 256x256 with a **Lanczos** filter, writing PNGs to
# `./custom_dataset/real_images/`.
#
# **Statistical caveat you must report.** FID is a *biased* estimator: the covariance of a
# 2048-d Inception feature cannot be estimated from 50-100 samples, so the absolute FID will be
# inflated by hundreds of points and dominated by sample-size bias. We therefore
# (i) report **KID** as the primary metric (unbiased MMD estimator, reliable for n<1000), and
# (ii) report FID only as a *relative* ranking between the two models measured against the
# *same* reference set. Never compare these numbers to published ImageNet FIDs.

# %%
# --- Cell 2.1: (optional) upload helper --------------------------------------
# from google.colab import files
# uploaded = files.upload()   # then move the files into ./custom_dataset/raw/<class>/

# For a smoke test you can synthesise a placeholder reference set from the DiT samples of a
# different seed -- but the submitted report MUST use genuinely collected images.
RAW_DIR = "./custom_dataset/raw"
for c in CLASS_NAMES:
    Path(f"{RAW_DIR}/{c}").mkdir(parents=True, exist_ok=True)
print("Drop your collected images into:", os.path.abspath(RAW_DIR))

# %%
# --- Cell 2.2: normalise the reference set -----------------------------------
from PIL import Image
import glob

def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS)

def build_real_set(raw_dir=RAW_DIR, out_dir=CFG.real_dir, size=CFG.image_size):
    n, per_class = 0, {}
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    for c in CLASS_NAMES:
        files_ = []
        for e in exts:
            files_ += glob.glob(os.path.join(raw_dir, c, e))
        per_class[c] = len(files_)
        for i, f in enumerate(sorted(files_)):
            try:
                im = center_crop_resize(Image.open(f), size)
            except Exception as ex:
                print("skip", f, ex); continue
            im.save(os.path.join(out_dir, f"{c}_{i:04d}.png"))
            n += 1
    return n, per_class

n_real, per_class = build_real_set()
print(f"Reference set: {n_real} images at {CFG.image_size}x{CFG.image_size}")
print(per_class)
if n_real < 50:
    print("WARNING: fewer than 50 reference images -- metrics will be meaningless.")

# %% [markdown]
# ## 3. Model initialisation
#
# ### 3.1 DiT-XL/2-256 (latent diffusion transformer)

# %%
# --- Cell 3.1: DiT ------------------------------------------------------------
from diffusers import DiTPipeline, DDIMScheduler, DDPMScheduler

free()
torch.cuda.reset_peak_memory_stats()

dit = DiTPipeline.from_pretrained(CFG.dit_id, torch_dtype=torch.float16)

# The transformer is the memory/compute hog -> keep it in FP16.
# The SD-VAE decoder is only 84M params but its mid-block activations overflow the FP16
# dynamic range and produce black/NaN tiles, so we keep it in FP32 (cheap, ~340 MB).
if CFG.decode_vae_fp32:
    dit.vae = dit.vae.to(torch.float32)

if CFG.use_cpu_offload:
    dit.enable_model_cpu_offload()     # accelerate hooks: weights stream in per sub-module
else:
    dit = dit.to("cuda")

dit.set_progress_bar_config(disable=True)
DIT_LOAD_VRAM = torch.cuda.max_memory_allocated() / 1024**2
print(f"DiT loaded. Weights resident: {DIT_LOAD_VRAM:.0f} MiB")
print("transformer params: %.1fM" % (sum(p.numel() for p in dit.transformer.parameters())/1e6))
LATENT_CH = dit.transformer.config.in_channels          # 4
VAE_SCALE = dit.vae.config.scaling_factor               # 0.18215
NULL_CLASS = dit.transformer.config.num_embeds_ada_norm - 1   # 1000 = "no class"
print("latent channels:", LATENT_CH, "| vae scaling:", VAE_SCALE, "| null class:", NULL_CLASS)

# %% [markdown]
# ### 3.2 VAR-d16 (next-scale autoregression)
#
# VAR is a two-stage model: a **multi-scale residual VQVAE** (shared codebook, V=4096, C=32)
# and a causal Transformer that predicts the token map of scale `k+1` given all of scales
# `1..k`. For 256x256 the ten scales are `(1,2,3,4,5,6,8,10,13,16)` -> 680 tokens in 10
# forward passes, versus 256 tokens x NFE forward passes for DiT.

# %%
# --- Cell 3.2: VAR ------------------------------------------------------------
from huggingface_hub import hf_hub_download
from models import build_vae_var

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)   # 256x256 schedule
assert sum(p * p for p in PATCH_NUMS) == 680

vae_ckpt = hf_hub_download("FoundationVision/var", "vae_ch160v4096z32.pth")
var_ckpt = hf_hub_download("FoundationVision/var", f"var_d{CFG.var_depth}.pth")

free()
torch.cuda.reset_peak_memory_stats()

vqvae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device="cuda", patch_nums=PATCH_NUMS,
    num_classes=1000, depth=CFG.var_depth, shared_aln=False,
)
vqvae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
var.load_state_dict(torch.load(var_ckpt,  map_location="cpu"), strict=True)
vqvae.eval(); var.eval()
for p in list(vqvae.parameters()) + list(var.parameters()):
    p.requires_grad_(False)

# NOTE ON PRECISION: we keep VAR's weights in FP32 and run under `torch.autocast(float16)`
# rather than calling `.half()`. The residual quantiser accumulates f_hat across ten scales
# and its interpolation + phi convolutions overflow when the *accumulator* is FP16. Autocast
# gives us FP16 matmuls (the speed win) while keeping the accumulator in FP32.
VAR_LOAD_VRAM = torch.cuda.max_memory_allocated() / 1024**2
print(f"VAR-d{CFG.var_depth} loaded. Weights resident: {VAR_LOAD_VRAM:.0f} MiB")
print("var params: %.1fM" % (sum(p.numel() for p in var.parameters())/1e6))

# %% [markdown]
# ## 4. Samplers with intermediate-state capture
#
# The stock `DiTPipeline.__call__` exposes no step callback and VAR's
# `autoregressive_infer_cfg` returns only the final image. Both are re-implemented below so
# that we can (a) dump the intermediate states required by the progression figure and
# (b) control the NFE / scale-depth used by the ablations.

# %%
# --- Cell 4.1: VAR sampler with per-scale decoding ---------------------------
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng

@torch.no_grad()
def var_sample(label_ids, cfg_scale=CFG.var_cfg, top_k=CFG.var_top_k, top_p=CFG.var_top_p,
               seed=None, more_smooth=False, max_scale=None, return_intermediates=False):
    """Mirror of VAR.autoregressive_infer_cfg that also decodes f_hat after every scale.

    max_scale (1..10): stop after that many scales. The residual pyramid is *additive*, so
    truncating simply leaves the high-frequency residuals at zero -- the decoder still emits a
    full 256x256 image, just a blurry one. This is the AR analogue of reducing DiT's NFE.
    Returns uint8 tensor (B,3,H,W) and, optionally, the list of per-scale decodes.
    """
    B = len(label_ids)
    dev = next(var.parameters()).device
    if seed is not None:
        var.rng.manual_seed(seed)
    rng = var.rng
    max_scale = len(var.patch_nums) if max_scale is None else int(max_scale)

    label_B = torch.tensor(label_ids, device=dev, dtype=torch.long)
    # CFG: batch the conditional and the null-class (1000) branch together.
    sos = cond_BD = var.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0))
    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_token_map = (sos.unsqueeze(1).expand(2 * B, var.first_l, -1)
                      + var.pos_start.expand(2 * B, var.first_l, -1)
                      + lvl_pos[:, :var.first_l])

    f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])
    inter, cur_L = [], 0

    for b in var.blocks:
        b.attn.kv_caching(True)
    try:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            for si, pn in enumerate(var.patch_nums):
                if si >= max_scale:
                    break
                ratio = si / var.num_stages_minus_1
                cur_L += pn * pn
                cond_BD_or_gss = var.shared_ada_lin(cond_BD)
                x = next_token_map
                for b in var.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = var.get_logits(x, cond_BD)

                # Linearly ramped CFG: weak guidance at coarse scales (global layout),
                # strong guidance at fine scales (class-discriminative texture).
                t = cfg_scale * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                if not more_smooth:
                    h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    h_BChw = (gumbel_softmax_with_rng(
                        logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng)
                        @ var.vae_quant_proxy[0].embedding.weight.unsqueeze(0))
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, var.Cvae, pn, pn)

                f_hat, next_token_map = var.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(var.patch_nums), f_hat, h_BChw)

                if return_intermediates:
                    inter.append(var.vae_proxy[0].fhat_to_img(
                        f_hat.detach().clone().float()).add(1).mul(0.5).clamp(0, 1).cpu())

                if si != var.num_stages_minus_1 and si + 1 < max_scale:
                    next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
                    next_token_map = (var.word_embed(next_token_map)
                                      + lvl_pos[:, cur_L:cur_L + var.patch_nums[si + 1] ** 2])
                    next_token_map = next_token_map.repeat(2, 1, 1)
    finally:
        for b in var.blocks:
            b.attn.kv_caching(False)

    img = var.vae_proxy[0].fhat_to_img(f_hat.float()).add(1).mul(0.5).clamp(0, 1)
    img = (img * 255).round().to(torch.uint8).cpu()
    return (img, inter) if return_intermediates else img

# %%
# --- Cell 4.2: DiT sampler with x0-prediction capture ------------------------
@torch.no_grad()
def dit_sample(label_ids, nfe=CFG.dit_nfe, cfg_scale=CFG.dit_cfg, seed=None,
               scheduler=CFG.dit_scheduler, capture=0):
    """Manual DiT ancestral loop with classifier-free guidance.

    `capture` > 0 additionally decodes `pred_original_sample` (the running x0 estimate) at
    `capture` evenly spaced steps -- this is what makes the denoising trajectory visualisable.
    DiT-XL/2 predicts 8 channels (learned sigma); guidance is applied to the eps half only,
    matching the reference implementation.
    """
    B = len(label_ids)
    dev = torch.device("cuda")
    sched = (DDIMScheduler.from_config(dit.scheduler.config) if scheduler == "ddim"
             else DDPMScheduler.from_config(dit.scheduler.config))
    sched.set_timesteps(nfe, device=dev)

    g = torch.Generator(device="cpu")
    g.manual_seed(CFG.seed if seed is None else seed)
    lat = torch.randn(B, LATENT_CH, CFG.image_size // 8, CFG.image_size // 8,
                      generator=g).to(dev, dtype=torch.float16)
    lat = lat * sched.init_noise_sigma

    y = torch.tensor(label_ids, device=dev, dtype=torch.long)
    y_null = torch.full_like(y, NULL_CLASS)
    y_in = torch.cat([y, y_null], dim=0)

    cap_idx = set(np.linspace(0, nfe - 1, min(capture, nfe)).round().astype(int).tolist()) \
        if capture else set()
    inter = []

    for i, t in enumerate(sched.timesteps):
        x_in = torch.cat([lat, lat], dim=0)
        x_in = sched.scale_model_input(x_in, t)
        t_in = t.expand(x_in.shape[0]) if torch.is_tensor(t) else \
            torch.tensor([t] * x_in.shape[0], device=dev)

        out = dit.transformer(x_in, timestep=t_in, class_labels=y_in).sample
        eps, _rest = out[:, :LATENT_CH], out[:, LATENT_CH:]
        cond_eps, uncond_eps = eps.chunk(2, dim=0)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        step = sched.step(eps, t, lat)
        lat = step.prev_sample
        if i in cap_idx:
            x0 = getattr(step, "pred_original_sample", None)
            inter.append(_decode(x0 if x0 is not None else lat))

    img = _decode(lat)
    return (img, inter) if capture else img

@torch.no_grad()
def _decode(lat):
    """Decode a latent to a uint8 CPU tensor (B,3,H,W)."""
    z = lat / VAE_SCALE
    z = z.float() if CFG.decode_vae_fp32 else z.half()
    img = dit.vae.decode(z).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return (img * 255).round().to(torch.uint8).cpu()

# %%
# --- Cell 4.3: smoke test -----------------------------------------------------
import matplotlib.pyplot as plt
free(); set_seed(CFG.seed)

test_ids = CLASS_IDS[:4]
v = var_sample(test_ids, seed=CFG.seed)
d = dit_sample(test_ids, nfe=20, seed=CFG.seed)

fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
for j in range(4):
    ax[0, j].imshow(v[j].permute(1, 2, 0).numpy()); ax[0, j].axis("off")
    ax[1, j].imshow(d[j].permute(1, 2, 0).numpy()); ax[1, j].axis("off")
    ax[0, j].set_title(CLASS_NAMES[j].replace("_", " "), fontsize=9)
ax[0, 0].set_ylabel("VAR"); ax[1, 0].set_ylabel("DiT")
fig.suptitle("Smoke test: VAR-d16 (top) vs DiT-XL/2 NFE=20 (bottom)")
plt.tight_layout(); plt.show()
free()

# %% [markdown]
# ## 5. Benchmark harness — latency and peak VRAM
#
# Protocol: 2 warm-up batches (CUDA context + cuDNN autotune + kv-cache allocation), then 5
# timed batches with an explicit `torch.cuda.synchronize()` on both sides of the timer.
# Peak VRAM is read from `torch.cuda.max_memory_allocated()` after
# `reset_peak_memory_stats()`, i.e. it excludes the caching allocator's reserved-but-free
# blocks and is therefore the *true* tensor high-water mark.

# %%
# --- Cell 5.1: timing utility -------------------------------------------------
import pandas as pd

def benchmark(fn, batch, n_warmup=2, n_iter=5):
    for _ in range(n_warmup):
        fn(); torch.cuda.synchronize()
    free(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    ts = np.array(ts)
    return {
        "ms_per_batch": float(ts.mean()),
        "ms_per_batch_std": float(ts.std()),
        "ms_per_image": float(ts.mean() / batch),
        "imgs_per_sec": float(batch / (ts.mean() / 1000.0)),
        "peak_vram_mib": float(peak),
    }

# %%
# --- Cell 5.2: headline benchmark --------------------------------------------
free()
bench = {}

ids_v = (CLASS_IDS * 10)[:CFG.batch_var]
bench["VAR-d%d" % CFG.var_depth] = benchmark(
    lambda: var_sample(ids_v, seed=None), CFG.batch_var)
free()

ids_d = (CLASS_IDS * 10)[:CFG.batch_dit]
for nfe in [10, 20, 50]:
    bench[f"DiT-XL/2 (NFE={nfe})"] = benchmark(
        lambda n=nfe: dit_sample(ids_d, nfe=n, seed=None), CFG.batch_dit)
    free()

bench_df = pd.DataFrame(bench).T.round(2)
bench_df.to_csv(f"{CFG.res_dir}/benchmark.csv")
display(bench_df)

# %% [markdown]
# ## 6. Bulk generation for the distributional metrics

# %%
# --- Cell 6.1: generation loop ------------------------------------------------
from torchvision.utils import save_image

def dump(batch_uint8, labels, out_dir, start_idx):
    for k in range(batch_uint8.shape[0]):
        Image.fromarray(batch_uint8[k].permute(1, 2, 0).numpy()).save(
            os.path.join(out_dir, f"{labels[k]}_{start_idx + k:05d}.png"))

def generate_all(model, out_dir, batch, **kw):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(f)
    idx, t0 = 0, time.time()
    for ci, cname in enumerate(CLASS_NAMES):
        cid = CLASSES[cname][0]
        remaining = CFG.n_per_class
        while remaining > 0:
            b = min(batch, remaining)
            ids = [cid] * b
            seed = CFG.seed + idx
            imgs = var_sample(ids, seed=seed, **kw) if model == "var" \
                else dit_sample(ids, seed=seed, **kw)
            dump(imgs, [cname] * b, out_dir, idx)
            idx += b; remaining -= b
        print(f"  [{model}] {cname}: done ({idx} total, {time.time()-t0:.0f}s)")
        free()
    return idx

n_var = generate_all("var", CFG.var_out, CFG.batch_var)
free()
n_dit = generate_all("dit", CFG.dit_out, CFG.batch_dit, nfe=CFG.dit_nfe)
free()
print(f"VAR: {n_var} images -> {CFG.var_out}")
print(f"DiT: {n_dit} images -> {CFG.dit_out}")

# %% [markdown]
# ## 7. Quantitative metrics
#
# * **FID / KID** via `clean-fid` (`mode="clean"`, PIL-Lanczos anti-aliased resizing — the
#   TF-Inception resize bug in legacy implementations shifts FID by 5-15 points).
# * **CLIP-Score** via `torchmetrics`, using the template `"a photo of a {class}"`. Neither
#   model is text-conditional, so this measures **class-prompt alignment**, i.e. whether the
#   generated image lands in the right CLIP semantic neighbourhood — a semantic-fidelity
#   proxy, not a text-following score. State this explicitly in the report.

# %%
# --- Cell 7.1: FID / KID ------------------------------------------------------
from cleanfid import fid as cfid

def dist_metrics(gen_dir, real_dir=CFG.real_dir):
    f = cfid.compute_fid(real_dir, gen_dir, mode="clean", num_workers=0, batch_size=16)
    k = cfid.compute_kid(real_dir, gen_dir, mode="clean", num_workers=0, batch_size=16)
    return {"FID": float(f), "KID_x1000": float(k) * 1000.0}

free()
m_var = dist_metrics(CFG.var_out)
m_dit = dist_metrics(CFG.dit_out)
print("VAR", m_var)
print("DiT", m_dit)

# %%
# --- Cell 7.2: CLIP-Score -----------------------------------------------------
from torchmetrics.multimodal.clip_score import CLIPScore

def clip_score(gen_dir, model_name=CFG.clip_id, batch=16):
    metric = CLIPScore(model_name_or_path=model_name).to("cuda")
    files_ = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
    name2text = {c: CLASSES[c][1] for c in CLASS_NAMES}
    buf_i, buf_t = [], []
    for f in files_:
        cname = "_".join(os.path.basename(f).split("_")[:-1])
        im = torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1)
        buf_i.append(im); buf_t.append(name2text[cname])
        if len(buf_i) == batch:
            metric.update(torch.stack(buf_i).to("cuda"), buf_t); buf_i, buf_t = [], []
    if buf_i:
        metric.update(torch.stack(buf_i).to("cuda"), buf_t)
    v = float(metric.compute()); del metric; free()
    return v

m_var["CLIP"] = clip_score(CFG.var_out)
m_dit["CLIP"] = clip_score(CFG.dit_out)

main_tbl = pd.DataFrame({
    f"VAR-d{CFG.var_depth}": {**m_var,
        "Latency_ms_img": bench[f"VAR-d{CFG.var_depth}"]["ms_per_image"],
        "PeakVRAM_MiB":   bench[f"VAR-d{CFG.var_depth}"]["peak_vram_mib"],
        "NFE": len(PATCH_NUMS)},
    "DiT-XL/2": {**m_dit,
        "Latency_ms_img": bench[f"DiT-XL/2 (NFE={CFG.dit_nfe})"]["ms_per_image"],
        "PeakVRAM_MiB":   bench[f"DiT-XL/2 (NFE={CFG.dit_nfe})"]["peak_vram_mib"],
        "NFE": CFG.dit_nfe},
}).T.round(3)
main_tbl.to_csv(f"{CFG.res_dir}/main_table.csv")
display(main_tbl)

# %% [markdown]
# ## 8. Hyper-parameter sweeps
#
# Two sweeps, each isolating the *compute knob* of its paradigm:
#
# * **VAR scale depth** `k in {1..10}` — how much of the residual pyramid is materialised.
# * **DiT NFE** `in {5,10,20,50,100}` — how many denoising steps are taken.
#
# Plus a guidance sweep (`var_cfg`, `dit_cfg`) because CFG strength trades diversity (FID)
# against class fidelity (CLIP) in both paradigms — the classic guidance "FID-U-curve".

# %%
# --- Cell 8.1: quick-metric helper (smaller N, for sweeps) --------------------
SWEEP_N = 20   # per class -> 200 images per sweep point

def sweep_generate(model, tag, **kw):
    out = f"./outputs/sweep/{tag}"
    Path(out).mkdir(parents=True, exist_ok=True)
    for f in glob.glob(out + "/*.png"):
        os.remove(f)
    idx = 0
    for cname in CLASS_NAMES:
        cid, remaining = CLASSES[cname][0], SWEEP_N
        batch = CFG.batch_var if model == "var" else CFG.batch_dit
        while remaining > 0:
            b = min(batch, remaining)
            imgs = var_sample([cid] * b, seed=CFG.seed + idx, **kw) if model == "var" \
                else dit_sample([cid] * b, seed=CFG.seed + idx, **kw)
            dump(imgs, [cname] * b, out, idx)
            idx += b; remaining -= b
    free()
    return out

# %%
# --- Cell 8.2: VAR scale-depth sweep -----------------------------------------
var_sweep = []
for k in range(1, len(PATCH_NUMS) + 1):
    t = benchmark(lambda kk=k: var_sample(ids_v, max_scale=kk, seed=None), CFG.batch_var,
                  n_warmup=1, n_iter=3)
    out = sweep_generate("var", f"var_k{k}", max_scale=k)
    m = dist_metrics(out)
    var_sweep.append({"model": "VAR", "knob": "scale_depth", "value": k,
                      "tokens": sum(p * p for p in PATCH_NUMS[:k]),
                      "ms_per_image": t["ms_per_image"],
                      "peak_vram_mib": t["peak_vram_mib"], **m})
    print(var_sweep[-1])
var_sweep_df = pd.DataFrame(var_sweep)
var_sweep_df.to_csv(f"{CFG.res_dir}/var_scale_sweep.csv", index=False)
display(var_sweep_df.round(2))

# %%
# --- Cell 8.3: DiT NFE sweep --------------------------------------------------
dit_sweep = []
for nfe in [5, 10, 20, 50, 100]:
    t = benchmark(lambda n=nfe: dit_sample(ids_d, nfe=n, seed=None), CFG.batch_dit,
                  n_warmup=1, n_iter=3)
    out = sweep_generate("dit", f"dit_nfe{nfe}", nfe=nfe)
    m = dist_metrics(out)
    dit_sweep.append({"model": "DiT", "knob": "NFE", "value": nfe,
                      "tokens": 256 * nfe,
                      "ms_per_image": t["ms_per_image"],
                      "peak_vram_mib": t["peak_vram_mib"], **m})
    print(dit_sweep[-1])
dit_sweep_df = pd.DataFrame(dit_sweep)
dit_sweep_df.to_csv(f"{CFG.res_dir}/dit_nfe_sweep.csv", index=False)
display(dit_sweep_df.round(2))

# %%
# --- Cell 8.4: guidance-strength sweep ---------------------------------------
guid = []
for g in [1.0, 1.5, 2.0, 3.0, 4.0]:
    out = sweep_generate("var", f"var_cfg{g}", cfg_scale=g)
    m = dist_metrics(out); m["CLIP"] = clip_score(out)
    guid.append({"model": "VAR", "cfg": g, **m}); print(guid[-1])
for g in [1.0, 2.0, 4.0, 6.0, 8.0]:
    out = sweep_generate("dit", f"dit_cfg{g}", cfg_scale=g, nfe=CFG.dit_nfe)
    m = dist_metrics(out); m["CLIP"] = clip_score(out)
    guid.append({"model": "DiT", "cfg": g, **m}); print(guid[-1])
guid_df = pd.DataFrame(guid)
guid_df.to_csv(f"{CFG.res_dir}/guidance_sweep.csv", index=False)
display(guid_df.round(3))

# %%
# --- Cell 8.5: component ablation (CPU offload / precision / smoothing) ------
abl = []

# (a) more_smooth: replaces the hard codebook lookup with a Gumbel-softmax expectation,
#     removing VQ blockiness at the cost of slight over-smoothing.
for ms in [False, True]:
    out = sweep_generate("var", f"var_smooth{int(ms)}", more_smooth=ms)
    m = dist_metrics(out); m["CLIP"] = clip_score(out)
    abl.append({"component": "VAR more_smooth", "setting": str(ms), **m})

# (b) scheduler: DDPM (paper default, ancestral) vs DDIM (deterministic, few-step friendly)
for sch in ["ddpm", "ddim"]:
    out = sweep_generate("dit", f"dit_{sch}", nfe=CFG.dit_nfe, scheduler=sch)
    m = dist_metrics(out); m["CLIP"] = clip_score(out)
    abl.append({"component": "DiT scheduler", "setting": sch, **m})

abl_df = pd.DataFrame(abl)
abl_df.to_csv(f"{CFG.res_dir}/ablation.csv", index=False)
display(abl_df.round(3))

# %%
# --- Cell 8.6: memory-offload ablation ---------------------------------------
# Quantifies the latency tax of `enable_model_cpu_offload()`. Re-loads DiT with the hook
# installed, re-benchmarks, then restores the resident pipeline.
offload_rows = []
offload_rows.append({"mode": "resident (GPU)", **bench[f"DiT-XL/2 (NFE={CFG.dit_nfe})"]})

del dit; free()
dit = DiTPipeline.from_pretrained(CFG.dit_id, torch_dtype=torch.float16)
if CFG.decode_vae_fp32: dit.vae = dit.vae.to(torch.float32)
dit.enable_model_cpu_offload()
dit.set_progress_bar_config(disable=True)
offload_rows.append({"mode": "model_cpu_offload",
                     **benchmark(lambda: dit_sample(ids_d, nfe=CFG.dit_nfe, seed=None),
                                 CFG.batch_dit, n_warmup=1, n_iter=3)})
del dit; free()
dit = DiTPipeline.from_pretrained(CFG.dit_id, torch_dtype=torch.float16)
if CFG.decode_vae_fp32: dit.vae = dit.vae.to(torch.float32)
dit = dit.to("cuda"); dit.set_progress_bar_config(disable=True)

offload_df = pd.DataFrame(offload_rows).round(2)
offload_df.to_csv(f"{CFG.res_dir}/offload_ablation.csv", index=False)
display(offload_df)

# %% [markdown]
# ## 9. Figures
#
# Figure 1 — paradigm progression: VAR's coarse-to-fine **scale** pyramid against DiT's
# iterative **timestep** denoising (x0-estimates), same class, same figure.

# %%
# --- Cell 9.1: progression figure --------------------------------------------
import matplotlib.pyplot as plt
free(); set_seed(CFG.seed)

SHOW_CLASS = "macaw"
cid = CLASSES[SHOW_CLASS][0]
N_COL = 8

_, var_inter = var_sample([cid], seed=7, return_intermediates=True)
_, dit_inter = dit_sample([cid], nfe=50, seed=7, capture=N_COL)

var_pick = np.linspace(0, len(var_inter) - 1, N_COL).round().astype(int)

fig, ax = plt.subplots(2, N_COL, figsize=(2.05 * N_COL, 4.8))
for j, si in enumerate(var_pick):
    ax[0, j].imshow(var_inter[si][0].permute(1, 2, 0).numpy())
    ax[0, j].set_title(f"$k$={si+1}  ({PATCH_NUMS[si]}$\\times${PATCH_NUMS[si]})", fontsize=9)
    ax[0, j].axis("off")
for j in range(N_COL):
    ax[1, j].imshow(dit_inter[j][0].permute(1, 2, 0).numpy())
    ax[1, j].set_title(f"step {int(np.linspace(0,49,N_COL)[j])+1}/50", fontsize=9)
    ax[1, j].axis("off")
ax[0, 0].text(-0.18, 0.5, "VAR\nnext-scale", transform=ax[0, 0].transAxes,
              rotation=90, va="center", ha="center", fontsize=11)
ax[1, 0].text(-0.18, 0.5, "DiT\ndenoising ($\\hat{x}_0$)", transform=ax[1, 0].transAxes,
              rotation=90, va="center", ha="center", fontsize=11)
fig.suptitle(f"Generation trajectories, class = '{SHOW_CLASS.replace('_',' ')}'", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CFG.fig_dir}/fig_progression.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# --- Cell 9.2: latency / quality trade-off -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

a = axes[0]
a.plot(var_sweep_df.ms_per_image, var_sweep_df.FID, "o-", label="VAR (scale depth 1-10)")
a.plot(dit_sweep_df.ms_per_image, dit_sweep_df.FID, "s-", label="DiT (NFE 5-100)")
for _, r in var_sweep_df.iterrows():
    a.annotate(f"k={int(r.value)}", (r.ms_per_image, r.FID), fontsize=7,
               xytext=(3, 3), textcoords="offset points")
for _, r in dit_sweep_df.iterrows():
    a.annotate(f"{int(r.value)}", (r.ms_per_image, r.FID), fontsize=7,
               xytext=(3, 3), textcoords="offset points")
a.set_xscale("log"); a.set_xlabel("Latency (ms / image, T4, log scale)")
a.set_ylabel("FID (vs. custom reference set)")
a.set_title("Quality-latency Pareto front"); a.grid(alpha=.3); a.legend()

b = axes[1]
b.plot(var_sweep_df.tokens, var_sweep_df.KID_x1000, "o-", label="VAR")
b.plot(dit_sweep_df.tokens, dit_sweep_df.KID_x1000, "s-", label="DiT")
b.set_xscale("log"); b.set_xlabel("Transformer token-forwards per image (log scale)")
b.set_ylabel(r"KID $\times 10^3$")
b.set_title("Quality vs. compute"); b.grid(alpha=.3); b.legend()

plt.tight_layout()
plt.savefig(f"{CFG.fig_dir}/fig_tradeoff.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# --- Cell 9.3: qualitative side-by-side grid ---------------------------------
free(); set_seed(CFG.seed)
sel = ["golden_retriever", "macaw", "volcano", "sports_car", "espresso", "daisy"]
ids = [CLASSES[c][0] for c in sel]
gv = var_sample(ids, seed=123)
gd = dit_sample(ids, nfe=CFG.dit_nfe, seed=123)

fig, ax = plt.subplots(2, len(sel), figsize=(2.1 * len(sel), 4.7))
for j, c in enumerate(sel):
    ax[0, j].imshow(gv[j].permute(1, 2, 0).numpy()); ax[0, j].axis("off")
    ax[1, j].imshow(gd[j].permute(1, 2, 0).numpy()); ax[1, j].axis("off")
    ax[0, j].set_title(c.replace("_", " "), fontsize=9)
ax[0, 0].text(-0.15, .5, "VAR", transform=ax[0, 0].transAxes, rotation=90,
              va="center", fontsize=11)
ax[1, 0].text(-0.15, .5, "DiT", transform=ax[1, 0].transAxes, rotation=90,
              va="center", fontsize=11)
plt.tight_layout()
plt.savefig(f"{CFG.fig_dir}/fig_qualitative.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# --- Cell 9.4: guidance FID-U-curve ------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax2 = ax.twinx()
for m, mk in [("VAR", "o"), ("DiT", "s")]:
    g = guid_df[guid_df.model == m]
    ax.plot(g.cfg, g.FID, mk + "-", label=f"{m} FID")
    ax2.plot(g.cfg, g.CLIP, mk + "--", alpha=.55, label=f"{m} CLIP")
ax.set_xlabel("classifier-free guidance scale")
ax.set_ylabel("FID"); ax2.set_ylabel("CLIP-Score")
ax.grid(alpha=.3); ax.set_title("Guidance trades diversity against class fidelity")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
plt.tight_layout()
plt.savefig(f"{CFG.fig_dir}/fig_guidance.png", dpi=180, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Export — LaTeX tables + results bundle
#
# Emits `results.json` and ready-to-paste `\begin{tabular}` bodies so the numbers in the
# report are never transcribed by hand.

# %%
# --- Cell 10.1: export --------------------------------------------------------
def to_latex_rows(df, cols, fmt="{:.2f}"):
    out = []
    for _, r in df.iterrows():
        out.append(" & ".join(fmt.format(r[c]) if isinstance(r[c], (int, float, np.floating))
                              else str(r[c]) for c in cols) + r" \\")
    return "\n".join(out)

results = {
    "config": asdict(CFG),
    "n_real_images": n_real,
    "per_class_real": per_class,
    "benchmark": bench,
    "main": main_tbl.to_dict(),
    "var_scale_sweep": var_sweep_df.to_dict("records"),
    "dit_nfe_sweep": dit_sweep_df.to_dict("records"),
    "guidance": guid_df.to_dict("records"),
    "ablation": abl_df.to_dict("records"),
    "offload": offload_df.to_dict("records"),
}
with open(f"{CFG.res_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

tex = []
tex.append("% ---- Table I: main comparison ----")
tex.append(to_latex_rows(main_tbl.reset_index().rename(columns={"index": "Model"}),
                         ["Model", "NFE", "Latency_ms_img", "PeakVRAM_MiB",
                          "FID", "KID_x1000", "CLIP"]))
tex.append("\n% ---- Table II: VAR scale-depth sweep ----")
tex.append(to_latex_rows(var_sweep_df, ["value", "tokens", "ms_per_image", "FID", "KID_x1000"]))
tex.append("\n% ---- Table III: DiT NFE sweep ----")
tex.append(to_latex_rows(dit_sweep_df, ["value", "tokens", "ms_per_image", "FID", "KID_x1000"]))
latex_blob = "\n".join(tex)
with open(f"{CFG.res_dir}/tables.tex", "w") as f:
    f.write(latex_blob)
print(latex_blob)

get_ipython().system('cd /content && zip -qr ca6127_results.zip outputs/results outputs/figures && ls -lh ca6127_results.zip')
# from google.colab import files; files.download('/content/ca6127_results.zip')
