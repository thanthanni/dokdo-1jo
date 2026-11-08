# 니케 캐릭터·수채화 LoRA 학습 인수인계 (2차)

작성일: 2026-11-08 (한국 시간) · 이전 인수인계서(1차)를 대체한다.

## 1. 목표 (확정)

| 구분 | 내용 | 트리거 토큰(임시) |
|---|---|---|
| 캐릭터 LoRA ① | **기본 마리안** (카운터스 시절 기본형) | `mrn_nikke` |
| 캐릭터 LoRA ② | **마리안: 퀸** | `mrnqueen_nikke` |
| 캐릭터 LoRA ③ | **라푼젤: 생츄어리** (나중에 진행) | 미정 |
| 그림체 LoRA | 게임 **스토리 컷신의 수채화 질감** 일러스트 | `nkwc_style` |

- 캐릭터마다 LoRA를 따로 만들고, 그림체 LoRA도 따로 만든다. 생성할 때 두 LoRA를 조합한다(예: 수채화풍 마리안).
- 진행 순서: 기본 마리안 → 마리안: 퀸 → 라푼젤: 생츄어리.
- 학습 결과물은 **개인 용도로만** 쓴다. 사용자도 동의했다.
- 의상 범위(기본 의상만인지, 다른 옷도 입힐지)는 **아직 답을 받지 못했다.** 캡션 규칙이 이 답에 따라 달라진다.

## 2. 사용자 PC (실측)

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro 10.0.26300 (한글) |
| RAM | 15.9GB |
| GPU | RTX 5060, VRAM 8151MiB, 드라이버 621.14 (CUDA 13.5 지원) |
| 디스크 | C: 339GB 여유 / E: 204GB 여유 |
| Python | 3.14 (기본값) + **3.10.11** (winget으로 추가 설치) |
| Git | 2.56.0 (winget으로 설치) |

- 한글 Windows라서 `systeminfo | findstr "OS Name"` 같은 명령은 결과가 나오지 않는다. 대신 `Get-CimInstance`를 쓴다.
- 학습할 때는 Chrome, ChatGPT, Claude 앱 등을 꺼야 한다. 이 앱들이 GPU 메모리를 일부 쓰고 있었다.

## 3. 완료된 작업 (확인됨)

1. **Kohya sd-scripts 설치**
   - 위치: `C:\AI\sd-scripts` (git 태그 `v0.12.3`), 가상환경 `venv` (Python 3.10.11)
   - `torch 2.8.0+cu128`, `torchvision 0.23.0`
   - GPU 테스트 출력: `2.8.0+cu128 12.8` / `NVIDIA GeForce RTX 5060 (12, 0)` / `4.0` ✅
   - `pip check`: `No broken requirements found.` ✅
   - 주요 패키지 버전: accelerate 1.15.0, bitsandbytes 0.50.2, diffusers 0.40.0, transformers 5.17.0
2. **accelerate 설정**
   - `accelerate config default --mixed_precision bf16`로 만들었다.
   - 저장 위치: `C:\Users\thant\.cache\huggingface\accelerate\default_config.yaml`
   - 설정 내용: LOCAL_MACHINE, 분산 학습 없음, bf16
3. **베이스 모델 다운로드**
   - `C:\AI\models\animagine-xl-4.0-opt.safetensors` (6,938,350,040 bytes) — 학습용 원본
   - 복사본: `C:\AI\ComfyUI\ComfyUI\models\checkpoints\` (아래 4번 참고)

## 4. 진행 중 / 미확인

- **ComfyUI 데스크톱 앱**
  - 설치는 했다(v0.44.2, NVIDIA, Python 3.13, 설치 위치 `C:\AI\ComfyUI`).
  - 하지만 앱이 계속 **Comfy Cloud**로 연결된 상태다. 로컬 인스턴스로 전환하는 데 **실패했다.**
  - 사용자가 실행 버튼 옆의 "실행 위치" 메뉴를 반복해서 눌렀다. 제목 줄 가운데의 인스턴스 버튼은 찾지 못했다.
  - 앱은 쓰지 않는 방향으로 정했다.
- **대안으로 안내한 방법: ComfyUI 직접 설치 (실행 여부 미확인)**
  - 위치: `C:\AI\ComfyUI-local`
  - git clone → Python 3.10 venv → torch 2.8.0/cu128 + torchaudio 2.8.0 → `requirements.txt` → `python main.py` → 브라우저에서 `http://127.0.0.1:8188`
  - 안내한 명령에 `Move-Item`이 들어 있다. 실행했다면 ComfyUI용 모델 복사본은 `C:\AI\ComfyUI-local\models\checkpoints\`로 **옮겨졌다.**
  - 위험 요소: 최신 ComfyUI가 Python 3.10을 지원하지 않을 수 있다. 오류가 나면 Python 3.12/3.13용 venv를 따로 만들고 그 Python에 맞는 torch 휠을 확인해야 한다.
- **시험용 워크플로 파일 `C:\AI\nikke_test_workflow.json`** (생성 여부 미확인)
  - PowerShell here-string과 `[IO.File]::WriteAllText`로 만들도록 안내했다(BOM 없이 저장).
  - 구성: CheckpointLoaderSimple(animagine-xl-4.0-opt) → 프롬프트 2개 → EmptyLatent 832×1216 → KSampler(seed 12345 fixed, 28 steps, cfg 5, euler_ancestral, normal) → VAEDecode → SaveImage("nikke_test")
  - 프롬프트 A가 미리 들어 있다.

## 5. 다음에 할 일 (순서대로)

1. ComfyUI를 로컬에서 실행한다. 4번의 대안 설치가 성공했는지부터 확인한다.
2. 기준 시험 생성 A·B·C를 한다. 결과는 아직 없다.
   - 네거티브: `lowres, bad anatomy, bad hands, text, error, worst quality, low quality, watermark, signature`
   ```
   A) 1girl, marian (nikke), goddess of victory: nikke, solo, upper body, masterpiece, high score, great score, absurdres
   B) 1girl, solo, upper body, watercolor (medium), traditional media, masterpiece, high score, great score, absurdres
   C) A + watercolor (medium), traditional media
   ```
   - 확인할 것: 모델이 마리안을 이미 아는지(A), 기본 수채화 질감이 어떤지(B), 둘을 합치면 어떻게 나오는지(C)
3. **이미지 캡처 — 기한이 있다:** `MARIA-n` 이벤트가 **2026-11-19 04:59:59 KST에 종료**된다(Codex 조사 결과). 그 전에 MARIA-n과 `FADING AFFECTION`의 수채화 컷을 캡처한다.
4. 의상 범위를 확인하고 캡션 규칙을 확정한다.
5. 이미지 10장 정도로 **100스텝 시험 학습**을 한다. OOM 여부와 스텝당 시간을 측정한다.
6. 본 학습을 한 뒤, 학습에 쓰지 않은 포즈·배경으로 평가한다. 그다음 퀸, 라푼젤 순으로 확장한다.

## 6. 데이터 수집 기준

**캡처 방법**
- PC 클라이언트에서 최고 해상도로 설정하고 PNG로 저장한다. JPG와 유튜브 프레임은 쓰지 않는다.
- 대사창 숨기기 버튼을 쓴다. 숨길 수 없으면 UI 부분을 잘라낸다.
- 다시보기 경로
  - 메인 스토리: 로비 → 전초기지 → 커맨드 센터 → 회상
  - 종료된 이벤트: 회상 → 아카이브 (기억 필름으로 해금)

**찾아볼 위치** (수채화 컷 여부는 직접 보면서 판정해야 한다. Codex도 게임을 실행해 확인하지는 않았다)

| 대상 | 위치 |
|---|---|
| 기본 마리안 | 메인 0~1장, MARIA-n STORY I·II, FADING AFFECTION의 '마리안의 기록' 루트 |
| 마리안: 퀸 | 메인 47~48장(변화 구간), 49~50장(후보), 4주년 공식 자료 |
| 그림체 확장 후보 | OVER ZONE, WHITE MEMORY, RED ASH, OLD TALES |

**공통 기준**
- 같은 CG의 연속 프레임이나 확대본은 1~2장만 남긴다.
- 가로로 긴 컷은 버킷 기능으로 처리하므로 정사각형으로 자르지 않아도 된다.
- 캐릭터 LoRA: 해당 버전 디자인만 쓴다(모더니아·퀸 섞지 않기). 얼굴 30% / 상반신 40% / 전신 30%. 처음에는 피규어 사진을 넣지 않는다. 목표 15~30장.
- 그림체 LoRA: 같은 수채화 화풍의 컷만 쓰고, 여러 인물과 배경을 섞는다. 처음에는 20~50장으로 시작한다.
- 기록용 CSV 열: 파일명, 출처, 챕터/이벤트, 장면, 인물, 의상, 가로, 세로, 용도(스타일/캐릭터/둘다)
- 마리안이 나오는 수채화 컷은 파일명에 `_marian`을 붙여 표시한다.

**폴더 구조 (Kohya 방식, 숫자는 반복 횟수 임시값)**
```
dataset/marian/10_mrn_nikke/
dataset/marian_queen/10_mrnqueen_nikke/
dataset/style_watercolor/10_nkwc_style/
```

**캡션 규칙 (초안, Danbooru 태그 형식)**
- 캐릭터 LoRA: 트리거 토큰을 맨 앞에 두고, 표정·포즈·구도·배경 같은 **바뀌는 요소만** 적는다. 머리색이나 기본 의상 같은 고정 특징은 적지 않는다. 의상을 바꿔 입힐 계획이면 의상 태그를 별도로 적도록 규칙을 바꾼다.
- 그림체 LoRA: `nkwc_style`을 맨 앞에 두고 그림의 **내용**을 자세히 적는다. `watercolor` 같은 화풍 단어는 적지 않는다.

## 7. 8GB용 첫 시험 학습 설정 (초안, 아직 실행 전)

| 항목 | 값 |
|---|---|
| 스크립트 | `sdxl_train_network.py`, `--sdpa` |
| 정밀도·메모리 | `--mixed_precision=bf16 --gradient_checkpointing` |
| 캐시 | `--cache_latents --cache_latents_to_disk --network_train_unet_only --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk` |
| 옵티마이저 | `AdamW8bit` (오류가 나면 Adafactor + `relative_step=False scale_parameter=False warmup_init=False`) |
| rank / alpha | 16 / 8 (메모리가 부족하면 8로 낮춤) |
| 해상도·배치 | 768 기준, 버킷 사용(최대 버킷 작게 제한), 배치 1 |

- 텍스트 인코더 출력을 캐시하면 캡션 셔플·드롭아웃을 쓸 수 없다.
- OOM이 학습 스텝에서 나면 버킷 면적을 줄인다. 캐시를 만드는 중에 나면 캐시 배치를 줄인다.
- 끝났을 때 알림이 오도록 명령 끝에 `[console]::beep(880,400); [console]::beep(1320,600)`을 붙이기로 했다.

**사용자에게 안내한 학습 로그 신호**
- `running training / 学習開始` → 학습 시작
- `steps: x%` 진행 막대 → 진행 중
- `model saved` → 완료
- 실패 신호: `CUDA out of memory`, `no kernel image is available`, 모델 로딩 중 PC 멈춤(RAM 부족)

## 8. 사용자 특성과 진행 방식

- PowerShell이 익숙하지 않다. **복사해서 붙여 넣을 수 있는 명령 블록**을 한 단계씩 주고, 결과는 **스크린샷**으로 받는 방식이 잘 통했다.
- "이렇게 나오면 성공" 같은 **예시 출력을 명령으로 붙여 넣은 적이 있다.** 예시 출력은 명령 블록과 확실히 구분해서 보여 준다.
- `.bat` 더블클릭 파일 방식(install/train_test/train)을 제안했지만 아직 만들지 않았다.
- 사용자가 "컴에 접속해서 해 달라"고 요청했다. 클라우드 세션에서는 PC 화면을 조작할 수 없다. Remote Control 폴더 `클로드`(컴퓨터 `thanthanni`)가 등록되어 있지만 클라우드 세션에서는 쓸 수 없는 상태로 나온다. Claude 앱을 업데이트해야 할 가능성이 있다. 연결되더라도 명령·파일 작업만 가능하고 GUI는 조작할 수 없다.

## 9. 참고 자료

- sd-scripts: https://github.com/kohya-ss/sd-scripts (Windows 설치, `docs/sdxl_train_network.md`, `docs/train_network.md`)
- PyTorch 이전 버전 휠: https://pytorch.org/get-started/previous-versions/
- Animagine XL 4.0: https://huggingface.co/cagliostrolab/animagine-xl-4.0
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- MARIA-n / 4주년 공지: https://game.naver.com/lounge/nikke/board/detail/8369216
- 마리안 등장 정리: https://gamewith.jp/nikke/article/show/584251
- 스토리 아카이브 안내: https://game.naver.com/lounge/nikke/board/detail/5509667
