# 포트폴리오 영상 가이드 — SO-101 모방학습 (STEP 2)

이 저장소의 결과 영상을 포트폴리오에 넣을 때 **어떤 파일을, 어디에, 어떤 문구로** 넣는지 정리한 문서다.
수치는 모두 2026-11-16~17 실측값이다. 아직 측정하지 않은 값은 "미측정"으로 두고 채우지 않는다.

## 0. 간단 표기용 (영상 아래 한두 줄)

> 실물 리더암 텔레옵 시연 50회(학습 40 / 검증 10)로 ACT를 10,000 step 학습 (RTX 4070 Laptop, 1시간 15분). MuJoCo 시뮬레이션, 손목 카메라 입력.
>
> ACT trained for 10k steps on 50 real-leader-arm teleop demos (40 train / 10 val), RTX 4070 Laptop, 1 h 15 min. MuJoCo simulation, wrist-camera input.

- 성공률은 정식 평가(4절) 후에 이 문구 뒤에 덧붙인다. 예: "… 성공률 N/20 (step XXXX)". 평가 전에는 성공률을 적지 않는다.

## 1. 저장 위치와 파일 이름

영상 파일은 저장소에 넣지 않는다(`reports/eval_runs/`는 `.gitignore`). 포트폴리오 폴더 `바탕 화면\포트폴리오\step2\`에 아래 이름으로 둔다.
원본은 `step2\media_raw\`에 두고 포트폴리오에는 올리지 않는다. 각 mp4에는 첫 프레임 썸네일이 같은 이름의 `.png`로 함께 있다.

| 파일 | 상태 | 길이 | 배속 | 용도 |
|---|---|---|---|---|
| `so101_inference_demo.mp4` (+`.png`) | ✅ 있음 | 15.0s | 1× | **학습된 정책이 혼자 집어 놓는 장면** — STEP 2 박스 대표 영상 후보 |
| `so101_dashboard_summary.mp4` (+`.png`) | 미녹화 | — | 최대 2× | 웹 대시보드 4단계(수집→QA→학습→추론) 사용 장면 |
| `so101_teleop_record.mp4` | 미녹화 (선택) | — | 1× | 실물 리더암으로 MuJoCo 팔로워를 조종하며 녹화하는 장면 |

- 웹 규격: 무음, H.264, 가로 1280, `-crf 24 -preset slow -pix_fmt yuv420p -movflags +faststart`. 장면별 5MB 이하, 대표 8MB 이하.
- 웹 페이지에 넣을 때는 png를 poster로 쓴다: `<video src="so101_inference_demo.mp4" poster="so101_inference_demo.png" muted loop playsinline controls>`.

## 2. `so101_inference_demo.mp4` — 무엇이 찍혀 있나

- 화면: MuJoCo 3인칭 렌더, 1280×720, 편집 없는 1 episode(15.0초, 1×). 상단 라벨 `SO-101 ACT policy (step 10000) in MuJoCo | 1x, one unedited episode`.
- 흐름: 쉬는 자세에서 출발 → 빨간 큐브(2.3cm)를 집어 약 14cm 들어 올림 → 흰 접시 위에 놓음(접시 중심에서 1.4cm) → 쉬는 자세로 복귀.
- 정책 입력: **손목 카메라 이미지(240×320) + 관절 6개 상태뿐**이다. 영상의 3인칭 시점은 보여주기용 렌더이고 정책은 보지 않는다.
- 만든 명령:
  ```
  python pipeline/4_inference/eval_success_mujoco.py \
    --checkpoint train_runs/20261116_221202_main_run/checkpoints/010000/pretrained_model \
    --episodes 1 --obj-jitter-cm 0 --video <raw.mp4> --video-width 1280 --video-height 720
  ```
  이후 ffmpeg로 라벨만 얹고 웹 규격으로 다시 인코딩했다(자르기·배속 없음).
- **선별 여부:** 이 녹화 1회가 첫 시도였고 성공했다. 여러 번 찍어 고른 것은 아니다. 다만 물체 위치를 바꾸지 않는 고정 조건이라, 같은 조건에서는 매번 같은 결과가 나온다(결정적). **이 영상 하나로 성공률을 말할 수 없다.**

## 3. 캡션 (그대로 써도 되는 문구)

| 파일 | 한국어 | English |
|---|---|---|
| `so101_inference_demo.mp4` | 실물 리더암 텔레옵 시연 50회로 학습한 ACT 정책이 MuJoCo에서 손목 카메라만 보고 큐브를 집어 접시에 놓는 장면 (편집 없는 1회, 1배속) | An ACT policy trained on 50 real-leader-arm teleop demos picks a cube and places it on the plate in MuJoCo using only the wrist camera (one unedited episode, 1×) |
| `so101_dashboard_summary.mp4` | (녹화 후 작성) | (녹화 후 작성) |

## 4. 함께 적을 수 있는 실측 사실

| 항목 | 값 |
|---|---|
| 데이터 | 실물 SO-101 리더암 → MuJoCo 팔로워 텔레옵 녹화 50 episode, 25,902 프레임, 13.8~21.5초(중앙값 17.1초) |
| QA / 분할 | 대시보드 QA(무결성 문제 0건) 후 스냅샷으로 train 40 / val 10 episode |
| 학습 | ACT, 10,000 step, batch 16, RTX 4070 Laptop (8GB), **1시간 15분**(22:12→23:27), train loss 13.59(step 50) → 0.188(step 10000) |
| 성공률 | **미측정.** 대시보드 ④ "자동 성공률 평가"로 6000/8000/10000 × 20 episode(물체 위치 ±1cm 랜덤)를 돌린 뒤 이 칸을 채운다. 비교한 체크포인트의 성공률을 모두 적는다. |

## 5. 함께 적어야 하는 사실 (빼면 과장이 되는 것)

- **시뮬레이션 결과다.** 실물 팔로워 로봇에서 돌린 것이 아니다. 실물은 리더암(사람이 조종하는 쪽)만이다 — "real-to-sim 텔레옵".
- **성공률은 대표 영상과 별개로 적는다.** 단일 성공 영상만 보여주고 성공률을 생략하면 과장이 된다. 성공률을 아직 못 쟀으면 "평가 진행 중"으로 둔다.
- **평가 조건:** 팔은 학습 데이터의 쉬는 자세에서 시작하고, 한 번 시도 후 쉬는 자세로 돌아오면 episode를 끝낸다(실패 후 재시도 성공은 세지 않음). 물체 위치 변화는 ±1cm로 작다 — 일반화 성능을 주장하지 않는다.
- **시간 압축:** 녹화 루프가 목표 50Hz 대비 약 44Hz로 돌아 프레임 수가 실시간보다 약 13% 적다(데이터 전체에 일관됨).

## 6. 쓰지 말아야 할 표현

- "실물 로봇이 물체를 집는다" — 팔로워는 시뮬레이션이다.
- "성공률 100%" — 단일 영상이나 4회 테스트로 말할 수 없다. 정식 평가 수치만 쓴다.
- "다양한 위치에서 일반화" — 물체 위치는 고정(수집) / ±1cm(평가)뿐이다.
