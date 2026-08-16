# 새 팀원 온보딩 가이드 (SO-101 대시보드 처음 실행하기)

이 문서 하나만 위에서 아래로 따라가면, 자신의 노트북과 자신의 SO-101
리더암으로 데이터 수집 → QA → ACT 학습 → 추론까지 웹 대시보드로 돌릴 수
있다. 예상 소요 시간: 환경 설정 20-40분(다운로드 속도에 따라 다름) +
로봇 연결/캘리브레이션 15-20분.

## 0. 사전 준비물

- Windows PC, NVIDIA GPU (CUDA 12.8 호환 드라이버 — 없어도 ①데이터 수집/
  ②QA는 가능, ③학습/④추론 단계만 GPU 필요)
- SO-101 리더암(USB-시리얼 연결) + 전원 어댑터
- miniconda 또는 anaconda 설치되어 있을 것
- git 설치되어 있을 것

## 1. 코드 받기 (git clone)

이 저장소를 전달받은 경로(로컬 경로 또는 USB로 전달된 bare repo)에서:

```powershell
git clone <전달받은 경로> so101_web
cd so101_web
```

clone한 뒤 원하는 위치로 폴더를 옮겨도 괜찮다 — 프로젝트 코드가
`Path(__file__).resolve().parents[N]` 방식으로 자기 위치 기준 상대경로를
계산하도록 이미 설계되어 있어(`dashboard/lib/local_config.py` 등), 폴더
위치 자체는 어디든 상관없다.

## 2. conda 환경 만들기

```powershell
conda env create -f environment.yml
conda activate mujoco_env
```

`environment.yml`은 실제로 동작 확인된 환경을 그대로 내보낸 것이지,
직접 만든 "이상적인" 목록이 아니다 — 이 파일 그대로 설치하면 이 프로젝트가
검증된 것과 동일한 조합이 된다.

## 3. torch(CUDA) 설치 — conda로 못 받는 부분

`torch`/`torchvision`은 CUDA wheel이라 `environment.yml`에서 의도적으로
제외되어 있다. 아래 명령으로 별도 설치한다 (RTX 5050 기준, CUDA 12.8):

```powershell
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

GPU가 없거나 드라이버가 지원하는 CUDA 버전이 다르면, 위 커맨드의
`cu128` 부분을 자신의 환경에 맞게 조정해야 한다 — 정확한 조합은
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)에서
확인.

설치 확인:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. (선택) 대시보드 전용 패키지 재확인

`environment.yml`에 이미 포함되어 있지만, 혹시 빠졌다면:

```powershell
pip install -r dashboard/requirements-dashboard.txt
```

## 5. 알려진 `pip check` 경고 (무시해도 됨)

설치 후 `pip check`를 돌리면 아래 2건이 뜰 수 있다 — 이미 알려진 상태이고
파이프라인 동작에 지장이 없음이 확인되어 있으므로 고치려 하지 않아도 된다
(자세한 조사 내용은 `CLAUDE.md` 28절 참고):

- `datasets 4.8.5`가 `fsspec<=2026.2.0`을 요구하지만 `fsspec 2026.7.0` 설치됨
- `opencv-python-headless`가 `numpy<2.3.0`을 요구하지만 `numpy 2.4.6` 설치됨

## 6. 리더암 연결 & 캘리브레이션

`docs/REAL_LEADER_SETUP.md`를 처음부터 끝까지 그대로 따라간다 (USB 연결 →
COM 포트 확인 → 단독 연결 테스트 → 캘리브레이션 → 텔레오퍼레이션 동작 확인).
이 단계가 끝나면 자신의 리더암 전용 캘리브레이션 파일이 생긴다.

## 7. 대시보드 실행

프로젝트 루트에서:

```powershell
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 streamlit run dashboard/app.py
```

브라우저가 자동으로 열리며 홈 화면이 뜬다. 홈 화면 상단에 "환경 상태"
섹션이 있는데, 이 시점에는 아직 로컬 설정을 안 넣었으므로 경고가 떠
있는 게 정상이다 (다음 단계에서 채운다).

## 8. 대시보드에서 내 로컬 설정 입력

"① 데이터 수집" 페이지 → "⚙️ 로컬 설정" 펼치기 → COM 포트(2단계에서
확인한 값), 캘리브레이션 id/경로, 데이터 저장 경로 등을 입력 → "이
컴퓨터에 저장". 이 값들은 이 컴퓨터에만 저장되며(`config/local_settings.json`,
git에 포함 안 됨) 다른 팀원에게 영향을 주지 않는다.

## 9. 다 됐는지 확인

- 홈 화면으로 돌아가 "환경 상태" 섹션이 전부 정상(✅)으로 뜨는지 확인
  (GPU가 없는 컴퓨터라면 GPU 항목만 정보성 안내로 남아있어도 정상)
- "① 데이터 수집" 페이지에서 세션 시작 버튼이 눌리는지 확인 (실제 녹화는
  리더암이 연결된 상태에서 진행)

## 10. 문제가 생기면

- `docs/TROUBLESHOOTING.md`
- `docs/CAMERA_TUNING_AND_TROUBLESHOOTING.md`
- `docs/GRIPPER_SLIP_TROUBLESHOOTING.md`
- `docs/REAL_LEADER_SETUP.md`의 "문제 해결" 표
