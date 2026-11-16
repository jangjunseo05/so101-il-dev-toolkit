# 2026-11 개선 기록

이전 학습 모델(13 episodes, 480x640)은 MuJoCo 추론에서 제자리에서 떨기만 하고
pick&place를 완료하지 못했다. 이후 양팔 로봇 프로젝트(HOLD THE FLOW)에서 확인한
내용을 반영해, 같은 조건에서 반복 성공하는 모델과 시연 영상을 확보하기 위해
아래를 바꿨다. 브랜치: `improve/fixed-object-position`.

## 1. 물체 위치 고정 (데이터 수집)
- `run_teleop_real.py`: episode 종료(S) 시 물체를 무작위 위치로 옮기던 동작을
  기본값에서 제외하고, 추론과 같은 scene.xml 스폰 위치로 복귀하게 했다.
  기존 동작은 `--randomize-object`로 켤 수 있다.
- 이유: 위치 일반화보다 같은 조건에서의 성공률을 먼저 확보한다.

## 2. 녹화 가이드 (대시보드 ① 데이터 수집)
- 접어 둔 안내 카드로 시연 원칙을 추가했다: 같은 시작 자세, 멈추지 않고 한 번에,
  비슷한 경로·속도, 실수한 episode는 X로 폐기, 50개 이상.
- 근거: ACT는 현재 장면만 보고 다음 동작 묶음을 예측하므로(과거·경과 시간 정보 없음)
  정지 구간·머뭇거림이 섞이면 같은 장면에 "멈춤"과 "진행" 정답이 섞여 추론 시
  떨림·망설임의 유력한 원인이 된다(양팔 프로젝트에서 시연 방식을 바꿔 해결).

## 3. 손목 카메라 해상도 480x640 -> 240x320
- `pipeline/camera_config.py`의 `WRIST_IMAGE_*`를 변경. 녹화와 추론이 같은 상수를
  쓰므로 두 경로가 함께 바뀐다(`run_teleop_real_to_real.py`도 이 상수를 사용).
- GTX 1650 Ti(4GB) 노트북 벤치마크(합성 데이터 3 episodes, ACT, num_workers=6):

  | 해상도 | batch | 스텝당 | samples/s | GPU 메모리 최대 |
  |---|---|---|---|---|
  | 480x640 | 8 | ~2.1s | ~4 | ~2.6GB |
  | 240x320 | 8 | ~0.65s | ~12 | ~3.1GB |
  | 240x320 | 16 | ~1.2s | ~13 | ~2.0GB |

  480x640은 PNG 디코딩(data_s)이 병목이었다. 240x320 batch 16은 이전 8GB 노트북의
  480x640 처리량(~14 samples/s)과 비슷하다.
- 기존 480x640 데이터셋·체크포인트와는 호환되지 않는다(데이터를 새로 수집).

## 4. 학습 설정 (`config/train_main_run_config.yaml`)
- 이전 데이터셋 전용 episode 목록 제거(전체 사용, QA 스냅샷이 있으면 대시보드가 덮어씀).
- `num_workers` 8 -> 6, `save_freq` 1500 -> 2000(10000 steps에서 체크포인트 5개).

## 5. 자동 성공률 평가 + 시연 영상 (`pipeline/4_inference/eval_success_mujoco.py`)
- 헤드리스로 N episode를 돌려 물체가 접시 위에 1초 이상 놓이면 성공으로 판정,
  성공률을 JSON으로 저장하고 3인칭 장면을 mp4로 기록한다.
- 예: `python pipeline/4_inference/eval_success_mujoco.py --checkpoint <pretrained_model> --episodes 10 --video reports/inference_demo.mp4`

## 이 노트북에서 실행할 때 주의
- **경로에 한글이 있으면 MuJoCo가 scene.xml을 열지 못한다** → 레포를 영문 경로에 둔다
  (현재 `C:\Users\jhelm\so101-il-dev-toolkit`).
- **`lerobot-train`이 체크포인트 저장 후 `checkpoints/last` 심볼릭 링크 생성에서 멈춘다**
  (WinError 1314) → Windows 설정 > 시스템 > 개발자용 > 개발자 모드 켜기(25절과 동일).
- 설정 파일의 한글 주석 때문에 `PYTHONUTF8=1` 환경 변수가 필요할 수 있다
  (cp949 UnicodeDecodeError).
- 환경은 `environment.yml`의 pip 목록을 `pip install --no-deps -r`로 설치했다(문서화된
  datasets/fsspec 버전 경고를 최신 pip가 오류로 처리하기 때문). torch는 cu128.

## 6. 새 컴퓨터 첫 실행 시 대시보드 오류 수정
- 아직 녹화된 데이터셋이 없으면 홈(`app.py`의 헬스 요약, `lib/next_action.py`),
  ① 데이터 수집(데이터셋 현황 카드), ② QA 검증 페이지가 `info.json`을 읽다가
  오류로 멈췄다(QA는 Hugging Face Hub 404). 데이터셋이 없을 때는 안내 문구를
  보여 주도록 고쳤다.
- ③ ACT 학습 페이지: 설정 파일에 `dataset.episodes`가 없을 때 표시가 깨지던 것을 수정.
- Streamlit AppTest로 홈·①~④ 페이지가 데이터 없는 상태에서 예외 없이 그려지는 것을 확인.

## 7. 외부 공개용 체험 데모 (Hugging Face Space)
- `SO101_DEMO=1`이면 데모 모드(`dashboard/lib/demo_mode.py`): 모든 페이지 상단에 "실제 장비와 연결되지 않음" 안내,
  녹화·학습·MuJoCo 추론 버튼은 잠금, 로컬 설정 카드 숨김, 데이터·결과는 `demo_assets/`에서 읽는다.
  ② QA 검증은 데모 서버에서도 실제로 계산한다(이미지 열을 뺀 경량 데이터셋).
- 데모·로컬 공통으로 추가된 화면: ① episode 둘러보기(손목 카메라 미리보기 + 관절 궤적),
  ③ 학습 로그 train loss 곡선, ④ 자동 성공률 평가 결과 카드 + 평가 영상, 홈 헬스 요약이 실제 기록을 읽음.
- 순서:
  1. `python tools/export_demo_assets.py --dataset-root data/so101_teleop_real --train-log logs/jobs/<학습 로그> --eval-report reports/eval_<시각>.json --eval-video reports/inference_demo.mp4 --gpu "GTX 1650 Ti (4GB)" --train-time "<실측 시간>"`
  2. `python tools/build_hf_space.py` -> `build/hf_space/`에 Space 내용(코드+demo_assets) 모음
  3. `python tools/build_hf_space.py --upload --space-id jun-seo/so101-il-dashboard` (공개 주소 생성)
- Space: Docker SDK, CPU, `deploy/hf_space/`(Dockerfile·requirements·README). Linux용 의존성 해석은 pip dry-run으로 확인.
- 이번에 고친 버그: ⑤ 전처리 구조 페이지도 데이터셋이 없으면 Hub 404로 멈추던 것을 안내 문구로 바꿈.
