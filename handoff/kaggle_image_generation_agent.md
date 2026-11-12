# 작업 지시서 — Kaggle에서 니케 수채화풍 이미지 생성

작성일: 2026-11-12 · 받는 사람: 이미지 생성 담당 에이전트

## 1. 목표

학습한 LoRA로 아래 캐릭터들을 **게임 스토리 컷신의 수채화풍**으로 **고품질** 생성한다. 결과물은 **개인 용도**다.

| 캐릭터 LoRA 토큰 | 대상 |
|---|---|
| `mrn_nikke` | 기본 마리안 |
| `pvmaid_nikke` | 프리바티: 언카인드 메이드 |
| `mrcms_nikke` | 마르차나: 마린 스터디 (수영복 버전) |

- 그림체 LoRA 토큰은 `nkwc_style`(수채화 컷신 화풍)이다.
- 생성할 때 **캐릭터 LoRA 1개 + 그림체 LoRA 1개**를 함께 적용한다.

## 2. 현재 상태와 선행 조건

- **LoRA는 아직 학습 전이다.** 이미지 자료를 수집하는 중이고, 학습은 `kaggle/nikke_lora_train.ipynb`로 진행한다.
- LoRA가 나오기 전에도 할 수 있는 일:
  - 생성 노트북 구축
  - 베이스 모델 기준 시험(8절 1단계)
  - 품질 파이프라인 검증
- LoRA 파일 형식: kohya sd-scripts가 출력한 `.safetensors`, U-Net만 학습, rank 32 / alpha 16, fp16 저장
- 파일 이름 규칙: `<토큰>-0000NN.safetensors`(NN = 에폭), 번호가 없는 파일은 마지막 에폭
- 받는 방법: 학습 노트북의 **Output**을 생성 노트북에 **Add Input → Notebook Output**으로 연결한다. 또는 사용자가 고른 파일을 Private 데이터셋으로 올린다.
  - 학습 노트북의 출력 경로: `/kaggle/working/output/<토큰>/`
  - 같은 폴더에 에폭별 샘플 이미지(`sample/`)와 `sample_prompts.txt`가 있다.

## 3. 환경 제약 (Kaggle)

| 항목 | 내용 |
|---|---|
| GPU | **T4 16GB ×2** (Turing). **bf16·fp8을 지원하지 않는다** → fp16 사용 |
| SDXL VAE | fp16에서 NaN(검은 이미지)이 나올 수 있다. VAE를 fp32로 올려서 쓰거나, `madebyollin/sdxl-vae-fp16-fix`를 쓴다 |
| 시간 | 주간 GPU 한도와 세션당 최대 실행 시간이 있다. 시작 전에 현재 값을 확인한다 |
| 저장 | `/kaggle/working`이 출력으로 저장된다(용량 제한 있음). 모델 같은 큰 파일은 `/tmp`에 둔다 |
| 인터넷 | Settings에서 Internet On이 필요하다 |

- **웹 UI를 터널로 외부에 노출하지 않는다**(ComfyUI/WebUI + cloudflared 등). Kaggle 약관을 확인하고 **헤드리스 실행**(diffusers 스크립트, 또는 ComfyUI API를 노트북 안에서만 호출)을 기본으로 한다.
- 두 GPU는 시드나 캐릭터를 나눠 **병렬로 생성**하는 데 쓴다.
- **SFW만 생성한다.**
  - 프롬프트에 `safe` 등급 태그를 넣고, 네거티브에 `nsfw, explicit`을 넣는다.
  - 마르차나는 수영복 디자인이 원본 의상이므로 그대로 두되, 노출을 늘리는 프롬프트는 쓰지 않는다.

## 4. 모델과 기본 생성 설정

- **베이스 모델:** Animagine XL 4.0 Opt
  - `https://huggingface.co/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0-opt.safetensors`
  - **LoRA를 이 모델로 학습했으므로 생성도 같은 모델로 한다.**
- 해상도: 832×1216(세로), 1216×832(가로), 1024×1024
- 샘플러: Euler a(`EulerAncestralDiscreteScheduler`), 28 스텝, CFG 5
  - 4.0 모델 카드 권장값을 다시 확인한다.
- 프롬프트 형식(Animagine 태그 순서): 인원 → 캐릭터 → 등급 → 일반 태그 → 화풍 → 품질
  ```
  1girl, mrn_nikke, safe, solo, upper body, looking at viewer, smile, city street, sky, nkwc_style, masterpiece, high score, great score, absurdres
  ```
- 네거티브:
  ```
  lowres, bad anatomy, bad hands, text, error, missing fingers, extra digits, worst quality, low quality, watermark, signature, nsfw, explicit
  ```
- **트리거 토큰이 캐릭터의 고정 특징을 담고 있다.** 학습 캡션에서 머리색·기본 의상 같은 태그를 지웠기 때문이다. 그러니 그런 특징을 프롬프트에 일부러 적지 않는다. 적으면 학습된 디자인과 충돌할 수 있다.
- 화풍 단어(`watercolor (medium)` 등)도 기본적으로 쓰지 않는다. `nkwc_style`과 그림체 LoRA가 담당한다. 효과를 비교하는 실험에서만 쓴다.

## 5. 구현 권장안 (diffusers)

- `StableDiffusionXLPipeline.from_single_file(...)`로 fp16 로드 → `cuda`
  - VAE 처리는 3절 참고: fp32 업캐스트 또는 fp16-fix VAE
- LoRA 여러 개 적용:
  ```python
  pipe.load_lora_weights(char_path, adapter_name="char")
  pipe.load_lora_weights(style_path, adapter_name="style")
  pipe.set_adapters(["char", "style"], adapter_weights=[w_char, w_style])
  ```
  - kohya 형식 LoRA는 diffusers에서 바로 로드된다. 로드 경고가 나오면 키가 정상적으로 매칭됐는지 확인한다.
- 프롬프트가 77토큰을 넘으면 잘린다. 긴 프롬프트가 필요하면 `compel`을 쓰거나 프롬프트를 줄인다.
- 시드를 고정한다(`torch.Generator("cuda").manual_seed(seed)`). 비교 실험은 반드시 같은 시드로 한다.

## 6. 품질 파이프라인

1. **기본 생성:** 832×1216
2. **고해상도 보정(hires fix):**
   - 1.5배로 업스케일한다(Lanczos 또는 애니용 ESRGAN 계열).
   - 같은 프롬프트와 LoRA로 img2img를 한 번 더 돌린다. denoise 0.35~0.45로 시작한다.
3. **얼굴 보정(선택):**
   - 애니 얼굴 검출 모델로 얼굴 영역을 찾는다.
   - 그 영역을 확대해 inpaint한다(denoise 0.3~0.4).
   - 얼굴이 작은 전신 구도에서 효과가 크다.
4. **선별:** 손·눈 붕괴, 디자인 불일치, 수채화 질감 손실이 있는 이미지는 탈락시킨다.

## 7. LoRA 강도와 에폭 선택

- 먼저 **에폭을 고른다.**
  - 캐릭터 LoRA를 단독으로 적용하고(강도 1.0), 마지막 3~4개 에폭을 같은 시드·프롬프트로 비교한다.
  - 고르는 기준: 닮음, 과적합 여부(배경과 포즈가 학습 이미지에 고정되는지), 이미지 붕괴
- 그다음 **강도 그리드**를 만든다.
  - 캐릭터 `{0.7, 0.85, 1.0}` × 그림체 `{0.5, 0.7, 0.9}`
  - 고정 시드 3개로 생성하고, 캐릭터별로 한 장짜리 비교표(그리드 이미지)를 만든다.
- 평가 기준:
  - 캐릭터 닮음: 얼굴, 머리, 의상 디자인
  - 수채화 질감: 종이 입자, 번짐, 붓 터치
  - 해부학적 오류
  - 프롬프트(포즈·배경)를 따르는지
- 학습 데이터에 없는 포즈와 배경으로 평가한다(예: 앉은 자세, 실내, 야간, 측면).

## 8. 작업 단계

1. **기준 시험 (LoRA 없이)**
   - 다음 프롬프트로 생성한다.
     - A) `1girl, marian (nikke), goddess of victory: nikke, safe, solo, upper body, …품질태그`
     - B) `1girl, safe, solo, upper body, watercolor (medium), traditional media, …품질태그`
     - C) A와 B 결합
   - 확인할 것: 베이스 모델이 캐릭터를 아는지, 기본 수채화 표현이 어떤지
2. **생성 노트북 구축:** 4~6절 설정, 두 GPU 병렬, 출력 정리 기능까지 만든다.
3. **LoRA가 들어오면:** 7절의 에폭 선택 → 강도 그리드
4. **최종 생성:** 캐릭터마다 다양한 구도(얼굴·상반신·전신, 실내·실외)로 생성하고, 6절의 파이프라인을 적용한 뒤 선별한다.
5. 결과와 권장 설정을 보고한다.

## 9. 산출물

```
/kaggle/working/
 ├ baseline/                  ← 8절 1단계 결과
 ├ epoch_compare/<토큰>.png   ← 에폭 비교 그리드
 ├ weight_grid/<토큰>.png     ← 강도 그리드
 ├ final/<토큰>/              ← 최종 선별 이미지 (PNG)
 ├ generation_log.csv         ← 파일, 시드, 프롬프트, 네거티브, LoRA 파일·강도, 해상도, 샘플러·스텝·CFG, hires/face 여부
 └ report.md
```

- 이미지 PNG에는 생성 설정을 메타데이터로 넣는다.
- `report.md`에 넣을 내용:
  - 캐릭터별 추천 에폭과 강도
  - 추천 프롬프트 템플릿
  - 잘 되는 구도와 안 되는 구도
  - 발견한 문제(예: 특정 의상 디테일 누락, 손 붕괴가 잦은 구도)와 원인 추정
  - 재학습이 필요하다면 데이터와 설정을 어떻게 바꿀지에 대한 제안
- 생성 노트북(`.ipynb`)도 함께 남긴다. 사용자가 프롬프트만 바꿔 다시 돌릴 수 있게 만든다.

## 10. 지켜야 할 것

- 데이터셋, 노트북, 결과물을 **Private**로 유지한다. 공개 공유나 배포는 하지 않는다.
- SFW만 생성한다. 실존 인물 묘사나 타인을 사칭하는 용도는 금지한다.
- 확인하지 못한 수치(Kaggle 한도, 모델 권장값 등)는 추측하지 말고 확인 후 기록한다.
- 사용자는 PowerShell과 노트북 조작에 익숙하지 않다. 사용자가 해야 하는 단계(Input 연결, GPU 설정, Save & Run All 등)는 **클릭 순서대로 짧게** 안내한다.
- 사용자 PC(RTX 5060 8GB, RAM 16GB)는 학습용이 아니라 결과 확인용이다. 같은 모델과 LoRA는 집의 ComfyUI에서도 생성할 수 있어야 하므로, **SDXL 범위를 벗어나는 모델로 바꾸지 않는다.**
