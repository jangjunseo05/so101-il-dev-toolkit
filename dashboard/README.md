# SO-101 파이프라인 대시보드

부트캠프 팀 내부용, SO-101 모방학습 파이프라인(데이터 수집 → QA/검증 → ACT
학습 → 추론) 개발 키트 겸 대시보드.

> 처음 설정하는 팀원은 `docs/NEW_TEAMMATE_SETUP.md`를 먼저 참고하세요.

## 실행

```
conda activate mujoco_env
pip install -r dashboard/requirements-dashboard.txt   # 최초 1회
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 streamlit run dashboard/app.py
```

프로젝트 루트(`so101_web/`)에서 실행해야 상대 경로(데이터셋, 체크포인트, PNG
등)가 정상적으로 잡힙니다. `PYTHONUTF8`/`PYTHONIOENCODING`은 Windows에서
한글 콘텐츠 처리 시 인코딩 에러를 막기 위함(이 프로젝트에서 여러 번
재발했던 문제).

## 구조

- `app.py` — 홈: 4단계 파이프라인 흐름 + 헬스 요약
- `pages/1~4_*.py` — 단계별 상세 페이지 (Streamlit 멀티페이지 자동 인식)
- `lib/data_sources.py` — 읽기 전용 데이터 로더 (기존 파이프라인 코드는 import만 함, 수정 없음)
- `lib/theme.py` — 공통 색상 상수

## 단계별 구현 상태

4단계(데이터 수집 → QA → ACT 학습 → 추론) 전부 웹에서 직접 실행 가능하도록
구현·실물 하드웨어(GPU/리더암)로 검증 완료됨(CLAUDE.md Stage 2~4 절 참고).

- **① 데이터 수집**: `run_teleop_real.py`를 서브프로세스로 실행/종료, 로컬
  설정(COM 포트 등) 폼 제공.
- **② QA & 검증**: `TeamRobotDataset` 계산을 직접 import해 실행(서브프로세스
  아님, 순수 계산이라 lock 불필요).
- **③ ACT 학습**: `lerobot-train`을 새 학습/이어서 학습 두 트랙으로 서브프로세스
  실행. GPU 자원 lock으로 다른 단계와 충돌 방지.
- **④ 추론**: `run_inference_mujoco.py`를 헤드리스 헬스체크(동기)/뷰어 관찰
  (서브프로세스) 두 트랙으로 실행.

동시 실행 자원 충돌(시리얼 포트/GPU/MuJoCo 뷰어)은 `lib/process_manager.py`의
lock 시스템이 막는다.

## 기존 파이프라인 코드와의 관계

이 대시보드는 `team_robot_dataset.py` 등 기존 코드를 **읽기 전용으로
import**만 하며, 어떤 파일도 수정하지 않습니다. QA 페이지가 계산에 쓰는
`TeamRobotDataset` 메서드들(`detect_outliers`, `check_integrity`,
`get_clean_indices`, `split_episodes` 등)은 전부 비파괴적(non-destructive)
설계라 원본 데이터셋을 건드리지 않습니다.
