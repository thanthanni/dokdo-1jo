# 니케 수채화풍 이미지 매일 자동 생성 (Kaggle 스크립트 커널)
# GitHub Actions가 매일 이 파일을 Kaggle에 올려 실행한다. 결과는 /kaggle/working/out/ 에 저장된다.
import os, sys, glob, json, random, subprocess, datetime

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "diffusers", "transformers", "accelerate", "peft", "safetensors"], check=True)

import torch
from PIL import Image, PngImagePlugin
from huggingface_hub import hf_hub_download
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
                       AutoencoderKL, EulerAncestralDiscreteScheduler)

# ===== 설정 =====
IMAGES_PER_DAY = 2
CHAR_WEIGHT = 0.9     # 강도 그리드 테스트 후 조정
STYLE_WEIGHT = 0.7
HIRES_SCALE = 1.5
HIRES_STRENGTH = 0.4

# file: LoRA 데이터셋 안 파일명 / fallback: LoRA가 없을 때 쓸 태그
CHARACTERS = [
    dict(token="mrn_nikke",    file="mrn_nikke.safetensors",    fallback="marian (nikke), goddess of victory: nikke"),
    dict(token="pvmaid_nikke", file="pvmaid_nikke.safetensors", fallback="privaty (nikke), maid, goddess of victory: nikke"),
    dict(token="mrcms_nikke",  file="mrcms_nikke.safetensors",  fallback="marciana (nikke), swimsuit, goddess of victory: nikke"),
]
STYLE = dict(token="nkwc_style", file="nkwc_style.safetensors", fallback="watercolor (medium), traditional media")

POSES = ["upper body, looking at viewer, smile", "portrait, close-up, looking at viewer",
         "full body, standing", "sitting, from side", "cowboy shot, looking back", "upper body, hand up, waving"]
SCENES = ["city street, sky", "cafe, window, sunlight", "flower field, wind", "ruins, overcast",
          "library, bookshelf", "beach, sea, blue sky", "night, city lights", "rooftop, sunset"]
SIZES = [(832, 1216), (832, 1216), (1216, 832), (1024, 1024)]

QUALITY = "masterpiece, high score, great score, absurdres"
NEG = ("lowres, bad anatomy, bad hands, text, error, missing fingers, extra digits, worst quality, "
       "low quality, watermark, signature, nsfw, explicit")

# ===== LoRA 찾기 (Kaggle 입력 데이터셋 어디에 있든) =====
def find_lora(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

style_path = find_lora(STYLE["file"])
for c in CHARACTERS:
    c["path"] = find_lora(c["file"])
print("LoRA:", {c["token"]: bool(c["path"]) for c in CHARACTERS}, "| style:", bool(style_path))

# ===== 모델 =====
assert torch.cuda.is_available(), "GPU가 없습니다"
ckpt = hf_hub_download("cagliostrolab/animagine-xl-4.0", "animagine-xl-4.0-opt.safetensors", cache_dir="/tmp/hf")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)  # T4 fp16 검은 이미지 방지
pipe = StableDiffusionXLPipeline.from_single_file(ckpt, vae=vae, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_tiling()
img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)

loaded = set()
def set_loras(char):
    names, weights = [], []
    for name, path, w in [(char["token"], char["path"], CHAR_WEIGHT), ("style", style_path, STYLE_WEIGHT)]:
        if path:
            if name not in loaded:
                pipe.load_lora_weights(path, adapter_name=name)
                loaded.add(name)
            names.append(name); weights.append(w)
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
    elif loaded:
        pipe.disable_lora()

# ===== 오늘의 조합 (날짜 기반이라 같은 날 재실행해도 같은 결과) =====
today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
rng = random.Random(today.toordinal())
chars = rng.sample(CHARACTERS, k=min(IMAGES_PER_DAY, len(CHARACTERS)))
while len(chars) < IMAGES_PER_DAY:
    chars.append(rng.choice(CHARACTERS))

os.makedirs("/kaggle/working/out", exist_ok=True)
log = []
for i, char in enumerate(chars):
    set_loras(char)
    char_tag = char["token"] if char["path"] else char["fallback"]
    style_tag = STYLE["token"] if style_path else STYLE["fallback"]
    pose, scene, (w, h) = rng.choice(POSES), rng.choice(SCENES), rng.choice(SIZES)
    prompt = f"1girl, {char_tag}, safe, solo, {pose}, {scene}, {style_tag}, {QUALITY}"
    seed = today.toordinal() * 10 + i

    g = torch.Generator("cuda").manual_seed(seed)
    base = pipe(prompt, negative_prompt=NEG, width=w, height=h, num_inference_steps=28,
                guidance_scale=5, generator=g).images[0]
    up = base.resize((int(w * HIRES_SCALE) // 8 * 8, int(h * HIRES_SCALE) // 8 * 8), Image.LANCZOS)
    g = torch.Generator("cuda").manual_seed(seed)
    final = img2img(prompt, negative_prompt=NEG, image=up, strength=HIRES_STRENGTH,
                    num_inference_steps=28, guidance_scale=5, generator=g).images[0]

    info = dict(date=str(today), prompt=prompt, negative=NEG, seed=seed, size=[w, h],
                hires=[HIRES_SCALE, HIRES_STRENGTH], char_lora=bool(char["path"]), style_lora=bool(style_path),
                weights=[CHAR_WEIGHT, STYLE_WEIGHT])
    meta = PngImagePlugin.PngInfo(); meta.add_text("parameters", json.dumps(info, ensure_ascii=False))
    path = f"/kaggle/working/out/{today}_{i + 1}_{char['token']}.png"
    final.save(path, pnginfo=meta)
    log.append(dict(file=os.path.basename(path), **info))
    print("saved", path, "|", prompt)

with open(f"/kaggle/working/out/{today}_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=1)
print("done")
