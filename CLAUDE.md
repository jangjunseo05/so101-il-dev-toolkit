# TeamRobotDataset 개발 스펙 (v0.3 - 시각화 완료 단계)

## 1. 목적
HuggingFace `lerobot` 라이브러리의 `LeRobotDataset`(v3.0)을 감싸는 팀 전용 데이터셋 클래스를 만든다.
목표는 커스텀 통계(표준편차/분산 등)와 시각화 기능을 base 클래스 위에 얹는 것이며,
base 클래스 소스코드는 절대 수정하지 않는다.

## 2. 데이터 소스 (v0.1 범위) — Phase 0 확인 완료 (2026-08-05)
- 사용 데이터셋: `SO101/datasets/so101_teleop/` (경로: `C:\Users\USER\Desktop\mujoco\SO-ARM100\Simulation\SO101\datasets\so101_teleop`)
- 구조 확인됨: `data/chunk-000/file-*.parquet`, `meta/episodes/chunk-000/file-*.parquet`,
  `meta/stats.json`, `meta/tasks.parquet`, `meta/info.json` (`codebase_version: "v3.0"`) → v3.0 스펙과 일치, 변환 불필요
- **주의**: `videos/` 폴더 없음 — 상태/액션 데이터만 기록됨. 카메라 프레임 기반 기능은
  v0.1 스파이크 범위에서 검증 불가하며, 시각화 스펙(10절)에서 별도 판단 필요
- 실물 하드웨어 데이터(`so101_teleop_real*`, 3개 폴더)는 `meta/info.json`만 있고
  실제 에피소드 데이터 없음(기록 미완료로 추정, 원인 미확인) → v0.1 검증 범위에서 **제외**,
  8절(범위 밖) 항목으로 이관

## 3. 아키텍처 결정: Composition (확정)
```python
class TeamRobotDataset:
    def __init__(self, repo_id_or_path, **kwargs):
        self._base = LeRobotDataset(repo_id_or_path, **kwargs)

    def __getitem__(self, idx):
        return self._base[idx]

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)
```
**근거**: `LeRobotDataset` 내부 구현(private 속성/메서드)에 의존하지 않아
lerobot 라이브러리 버전업에도 안정적. Subclass 방식(`LeRobotLanceDataset` 등 HF 공식 선례 있음)은
개발 속도는 빠르지만 내부 구현 변경에 취약해 이번 프로젝트에서는 채택하지 않음.

**절대 하지 말 것**: `lerobot` 패키지 소스코드(site-packages 내부)를 직접 수정하는 행위.
필요한 커스터마이징은 전부 `TeamRobotDataset` 레이어에서 처리한다.

## 4. Claude Code 작업 시 필수 지침
- `LeRobotDataset`의 실제 속성/메서드는 **절대 추측하지 말 것**. 반드시 다음으로 직접 확인:
  - `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; help(LeRobotDataset)"`
  - 또는 설치된 패키지 소스 직접 열람 (`pip show lerobot`으로 경로 확인 후 `view`)
- 설치된 lerobot 버전이 v3.0 스펙(`meta/stats.json`, `meta/episodes/` chunked parquet)을
  지원하는지 먼저 버전 확인 (`pip show lerobot` → `Version:` 필드, v0.4.0 이상 여부)
- 확인 없이 v2.0 방식(episode-per-file)을 가정한 코드를 작성하지 말 것

## 5. Spike 수락 기준 (이 3개 통과해야 v0.1 완료)
1. **DataLoader 호환성**: `DataLoader(TeamRobotDataset(...), batch_size=8, ...)`로
   배치 로딩이 base와 동일하게 동작하는가 (collate 단계 에러 없음)
2. **메타 위임 검증**: `.meta.stats`, `.meta.tasks`, `.meta.episodes` 등 base가 제공하는
   메타 접근이 `__getattr__` 위임을 통해 정상 작동하는가
3. **성능 확인**: `compute_custom_stats()`에서 프레임 단위로 순회할 때 base 대비
   유의미한 성능 손실(예: 20% 이상 느려짐)이 있는가 — 있다면 원인(위임 오버헤드 vs
   I/O)을 분리해서 보고

## 6. Phase 1 (Spike) 완료 결과 — PASS (2026-08-05)
3개 수락 기준 모두 통과, 재검증까지 완료:
1. **DataLoader 호환성**: PASS — 14 배치 전부 base와 shape 일치
2. **메타 위임**: PASS — `.meta.stats/.tasks/.episodes`가 `__getattr__` 통해 정상 위임 확인
3. **성능**: PASS — 20% 임계값 대비 여유 있게 통과. 단, 최초 단일 측정치(-2.11%/+1.87%)는
   재검증(5회 반복) 결과 표본 편차가 커서(`__getitem__` overhead 표준편차 7.99%p, 범위
   -8.42%~+11.26%) 대표성이 낮았음이 확인됨. 5회 평균 기준 전체 stats 연산 +2.64%,
   `__getitem__` 순회 +3.82% — 결론(PASS)은 유지되나 **2 episodes/109 frames 스모크
   테스트 기준**이며 대규모 데이터셋 성능은 별도 검증 필요
- 산출물: `team_robot_dataset.py`, `test_team_robot_dataset.py`, `phase1_spike_result.md`

## 7. v0.2 스펙: 커스텀 통계 — 관절별 아웃라이어 탐지 (완료)
**목표**: "빠르게 만들고 써보며 개선"하는 원칙에 따라, 최적 설계보다 구현 속도와
추후 교체 용이성을 우선한다.

- **아웃라이어 정의**: z-score 방식 (`|z| > 3`). `compute_custom_stats()`가 이미 계산한
  mean/std를 그대로 재사용 — 추가 연산 최소화가 목적
  - 한계: SG90 포텐셔미터 노이즈처럼 z-score로 안 잡히는 패턴이 나오면, 그때 관절
    물리적 한계값(joint limit) 기반 등 도메인 지식 방식으로 교체. 이 교체 여부 자체가
    실험 근거이므로 지금 최적화하지 않음
- **적용 단위**: 프레임 단위 탐지가 기본. 에피소드 단위 집계는 프레임 결과에서
  "이상 프레임 개수"로 파생 (별도 로직 불필요)
- **범위**: 탐지 + 리포트까지만. 이상 프레임/에피소드를 실제로 제거(필터링)하는 기능은
  **v0.2 범위 밖** — "무엇을 걸러도 되는지" 판단 기준이 별도로 필요해지므로 다음 단계로 이관

## 8. 범위 밖 (다음 단계로 보류)
- 인터랙티브 시각화(plotly 등) — 정적으로 부족하다는 근거가 생기면 재논의
- 카메라 프레임/영상 기반 시각화 — `so101_teleop`에 video 없음이 확인됨.
  별도 데이터 소스 확보 여부에 따라 범위 조정 필요
- 정규화 전략 비교 도구 — 정책 학습(ACT 등) 단계 진입 시점에 재논의
- 실물 하드웨어 노이즈 데이터 통합 — **진단 완료, 가설 수정 (2026-08-05)**:
  dtype 버그(`run_teleop_real.py`의 `np.clip(targets, ctrl_lo, ctrl_hi)`에서
  `ctrl_lo/ctrl_hi`(MuJoCo `mjtNum`, float64)와의 mixed-dtype 연산으로 action이
  float64로 승격되어 LeRobotDataset의 float32 스키마 검증에 매 프레임 실패하던
  문제, `.astype(np.float32)` 한 줄로 수정) 해결 후 5차례 세션을 거쳐 노이즈
  특성을 좁혀감:
  1) 부분관절/미검증 캘리브레이션(204f) → z_thresh=3.0 0건, 2.0에서 75건이나
     "상태 전환 구간"에 몰림(잔진동 아님)
  2) 전체관절 1차, wrist_roll 캘리브레이션 누락(681f) → std 불균일 재확인
  3) wrist_roll 재캘리브레이션 후 전체관절(1283f) → std 균일화(2.7배 격차),
     이전에 의심됐던 지그재그가 그래프에서 사라짐 → "지그재그 원인이
     캘리브레이션 미비였을 가능성" 부상
  4) 손 뗀 정지 상태(56f) → action std 전 관절 정확히 0.0, obs.state도
     대부분 1e-6~1e-9 (순수 물리적 정지 상태, 대조군)
  5) 손으로 잡고 버틴 정지 상태(232f) → action std가 0이 아닌 값(0.0009~
     0.0034)으로 뚜렷이 나타남 — 실제 신호 확인됨. 단 패턴은 "지속적
     고빈도 진동"이 아니라 **대부분 관절에서 평탄 구간 사이를 오가는
     계단형 변화**(사람이 완전한 정지를 유지 못하고 간헐적으로 자세
     미세조정하는 것에 가까움). wrist_roll만 frame 55-70 구간에서
     짧은 이중 진동을 보였으나 에피소드 전체에 지속되지 않음.

  **결론(가설 수정)**: 당초 목표였던 "SG90 포텐셔미터의 지속적 고빈도
  노이즈"는 이번 진단들로 명확히 뒷받침되지 않음. 대신 실물 데이터의
  주된 변동 요인은 (a) 캘리브레이션 미비로 인한 인공적 흔들림(3번에서
  해소됨), (b) 사람이 정지를 유지하지 못해 생기는 간헐적 계단형 자세
  조정(5번에서 확인)으로 좁혀짐. z-score(값 기준) 방식은 두 경우 모두
  "관절이 실제로 크게 움직인 구간의 극값"만 일관되게 탐지하고, 간헐적
  계단형 변화나 짧은 구간 진동은 대부분 놓침 — 7절에 적어둔 "z-score
  한계 시 도메인 기반 교체" 논의가 실물 데이터로 재확인됨.

  실물 데이터 통합은 여기서 진단 단계를 마무리. delta 기반 탐지나
  도메인 기반 방식으로의 전환은, 목표를 "고빈도 진동 탐지"에서
  "간헐적 계단형 전환 탐지"로 재정의한 뒤 착수 예정 (보류).

## 9. v0.2 산출물 요구사항
- `TeamRobotDataset`에 `detect_outliers()` (가칭) 메서드 추가 — z-score 기반,
  기존 `compute_custom_stats()` 결과 재사용
- 프레임 단위 이상치 목록 + 에피소드별 이상 프레임 개수 집계 리포트
- 리포트 저장 파일 (예: `outlier_report.md` 또는 `.json`) — 어떤 프레임/관절이
  왜 이상치로 판단됐는지 근거 포함
- 필터링 기능은 구현하지 말 것 (8절 참고, 범위 밖)

### 9-1. 리포트 확장: 값 기준 + delta 기준 병행 — DONE (2026-08-05)
`run_outlier_detection.py`가 `detect_outliers()`(값 기준)와
`detect_outliers_delta()`(delta 기준)를 모두 실행하도록 확장.
`outlier_report.md`에 두 섹션을 동일 형식으로 병기, 상단에 14절 결론
요약("서로 다른 유형의 이상을 탐지하는 상호보완적 도구") 추가.
결합 오버레이 PNG(`episode_0_trajectory_combined.png`)도 함께 생성.
`team_robot_dataset.py`의 메서드 로직은 미수정 — 리포트 스크립트만 확장.

**so101_teleop(시뮬레이션) 재실행 결과**: 값 기준 0건(기존과 동일) vs delta
기준 47건. 실물 데이터(8절)에서 본 "간헐적 계단형 전환"과는 다른 패턴 —
시뮬레이션의 delta 탐지는 "부드러운 대형 동작의 최고 속도 구간"(S자 곡선
중간 전환부)에 몰림. `timestamp` 필드 탐지는 14절에 이미 기록된 "저분산
필드 과대 탐지" 한계의 재현 사례로 확인(새로운 이상 아님) — 향후 저분산
비관절 필드(timestamp 등)를 탐지 대상에서 제외하는 것도 고려할 만함.

**작업 중 발견한 이슈**: `datasets/so101_teleop/`이 작업 시작 시 완전히
비어 있었음(원인 불명 — `.gitignore` 대상이라 git 히스토리로 추적 불가).
`../SO101.zip`(7/31 생성) 백업에서 복원하여 해결. 재발 시 원인 조사 필요 —
`datasets/`가 gitignore돼 있어 삭제되면 git으로 복구 불가하다는 점 주의.

## 10. v0.3 스펙: 궤적/액션 분포 시각화 (완료)
**목표**: matplotlib 정적 PNG로 빠르게 탐색 가능한 시각화. 인터랙티브 요구가
생기기 전까지는 정적 방식 유지.

- **visualize_episode(episode_idx)**:
  - 해당 에피소드의 action/observation.state 각 차원(관절)을 시간축(frame index)
    대비 라인 플롯으로, subplot으로 배치 (차원 수만큼)
  - detect_outliers() 결과를 재사용해, 해당 에피소드에서 이상치로 판정된 프레임을
    궤적 위에 점으로 오버레이 (별도 재계산 없이 기존 결과 전달받아 사용)
  - PNG로 저장 (예: episode_{idx}_trajectory.png)
- **plot_action_distribution(keys=None)**:
  - 데이터셋 전체 기준, 지정한 key(관절)들의 값 분포를 히스토그램으로
  - keys=None이면 compute_custom_stats()가 쓰는 float 필드 전체 대상
  - PNG로 저장 (예: action_distribution.png)
- **범위 제외**: 카메라 프레임/영상 기반 시각화 (so101_teleop에 videos/ 없음, 2절 참고)
- **범위 제외**: 인터랙티브 시각화 (plotly 등) — 정적으로 부족하다는 근거가 생기면 재논의

## 11. v0.3 산출물 요구사항
- TeamRobotDataset에 visualize_episode(), plot_action_distribution() 메서드 추가
- detect_outliers() 결과는 새로 계산하지 말고 인자로 전달받아 재사용
- PNG 파일 실제 생성 확인 (경로 명시)
- 요약 리포트에 "확인된 사실"/"판단·의견" 구분해서 보고

## 12. v0.3 완료 결과 — DONE (2026-08-05)
`visualize_episode()`, `plot_action_distribution()` 구현 후 `so101_teleop`(2 episodes/109 frames)
대상 실행 완료:
1. **visualize_episode(episode_idx, outlier_results)**: episode 0/1 각각 실행, PNG 생성 확인
   (episode_0: 208,307 bytes, episode_1: 144,523 bytes). `detect_outliers(z_thresh=2.0)`
   결과(75 entries, `{0: 8, 1: 18}`)를 오버레이해서 episode 0의 `shoulder_lift`/`elbow_flex`/
   `wrist_flex` 궤적 중 값이 급격히 꺾이는 구간(프레임 약 48-57)에 이상치로 판정된 프레임이
   빨간 점으로 정확히 겹쳐 표시됨을 확인. `outlier_results=None` 케이스(오버레이 없음)도
   별도 실행해 정상 동작 확인 (`episode_0_trajectory_no_overlay.png`).
2. **plot_action_distribution(keys=None)**: `action`(6) + `observation.state`(6) + `timestamp`(1)
   = 13개 subplot 히스토그램이 3열 그리드로 정상 생성됨 (129,422 bytes).
- **발견된 이슈**: `timestamp` feature가 `meta/info.json`에는 `shape: [1]`로 선언돼 있지만,
  실제 `__getitem__()` 반환값에서는 `torch.Size([])`(0-d 스칼라 텐서)였음(실측 확인).
  `plot_action_distribution()` 내부에서 `np.stack()` 후 1차원 배열이 되어 `.shape[1]` 접근 시
  `IndexError` 발생 → `arr.ndim == 1`이면 `(-1, 1)`로 reshape하는 방어 코드를 추가해 수정.
  (`compute_custom_stats()`/`detect_outliers()` 자체는 0-d 값에서도 에러 없이 동작해 손대지 않음)
- **사용자 확인**: `episode_0_trajectory.png`, `action_distribution.png`를 사용자가 직접
  열람하여 궤적/오버레이/분포가 정상 렌더링됨을 확인함.
- 산출물: `team_robot_dataset.py`(`visualize_episode()`/`plot_action_distribution()` 추가),
  `run_visualization.py`, `episode_0_trajectory.png`, `episode_1_trajectory.png`,
  `episode_0_trajectory_no_overlay.png`, `action_distribution.png`

## 13. 알려진 이슈 (해소됨 — 16절 참고)
- ~~`visualize_episode()`가 `TeamRobotDataset(episodes=[N])`로 필터링 로드된
  상태에서 호출되면 `IndexError` 발생~~ → 16절에서 근본 수정 완료 (2026-08-05)

## 14. v0.4 완료 결과: delta 기반 아웃라이어 탐지 — DONE (2026-08-05)
`detect_outliers_delta(keys=None, z_thresh=3.0)` 구현. 프레임 간 변화량(delta)에
z-score 적용, 에피소드 경계는 delta 계산에서 제외. `detect_outliers()`(값 기준)는
그대로 유지 — 폐기가 아니라 병행 사용.

**검증 결과**:
- episode 0(손 뗀 정지, 대조군): delta 기준도 사실상 탐지 없음(유일한 1건은
  1e-9 스케일, 물리적으로 무의미) — 정상
- episode 1(손으로 잡고 버틴 정지): wrist_roll의 frame 55-70 짧은 이중 진동을
  값 기준은 전혀 탐지 못했으나(재확인 결과 값 기준이 잡은 건 frame 0-26
  초반 정착 구간이었음 — 이전 보고 정정) delta 기준은 방향 전환 지점
  (frame 59, 64, 66)을 정확히 포착. **목표(간헐적 계단형 전환 탐지) 달성 확인**

**결론**: 값 기준과 delta 기준은 서로 다른 이상 유형을 잡는 상호보완적 도구
(값 기준 145건 vs delta 기준 62건, 겹치지 않는 관절 조합) — 하나로 통합하지
않고 둘 다 유지.

**알려진 한계**: 저분산 관절(gripper 등)에서 사소한 변동이 상대적으로
과대 탐지되는 문제가 delta 기준에서도 동일하게 재현됨 — z-score 방식(값/delta
공통)의 근본 한계. 추후 필터링 설계 시 관절별 절대 임계치 보정 고려 필요.

## 15. v0.4 완료 결과: 결합 오버레이 시각화 — DONE (2026-08-05)
`visualize_episode(episode_idx, outlier_results=None, delta_outlier_results=None,
save_path=None)`로 확장 — 값 기준(빨강)과 delta 기준(파랑) 아웃라이어를
동시 오버레이. 하위 호환성 확인(delta_outlier_results 생략 시 기존 출력과
완전히 동일).

**검증**: episode 1(손으로 잡고 버틴 정지)에서 빨간 점(frame 0-26, 초기
정착 구간)과 파란 점(frame 59/64/66 등, 이중 진동 구간)이 서로 다른
구간에 명확히 분리되어 표시됨 — 14절 결론("두 방식은 상호보완적")을
시각적으로 재확인.

**설계 메모**: delta_outlier_results의 파란 점은 delta 값이 아니라 해당
프레임의 실제 궤적 값 위치에 표시(시각화 목적에 맞춤). 단
`frame_outliers`의 `"value"` 필드 자체는 여전히 delta 값이므로, 다른
용도로 재사용 시 혼동 주의.

## 16. 인덱스 불일치 근본 수정 — DONE (2026-08-05)
13절/15절에서 3차례 재발했던 필터링 로드 시 인덱스 불일치 문제를 근본
수정. 원인은 lerobot 라이브러리 설계(`LeRobotDatasetMetadata`가 episodes
필터와 무관하게 항상 전역 인덱스 유지, `_absolute_to_relative_idx`는
private이라 미사용) — `frame_idx`의 의미를 "인스턴스 로컬 행 번호"에서
"에피소드 내 frame_index(필터링 무관 안정값)"로 재정의하여 해결.
`visualize_episode()`는 `meta.episodes` 의존을 제거하고 `episode_index`
직접 스캔 방식으로 변경.

**검증**: 전체 로드/필터링 로드 모두 오프셋 계산 없이 정상 동작, 교차
인스턴스(다른 필터링 상태끼리 결과 전달)도 바이트 단위 동일 PNG로 확인.
탐지 건수 자체는 전체(94/67)와 필터링(145/62) 로드 간 다른데, 이는
버그가 아니라 통계 baseline이 로드 범위마다 다른 원래 특성(2턴 전 이미
확인됨).

**Breaking change 영향 확인**: `frame_idx` 필드를 사용하는 코드는
`run_outlier_detection.py` 한 곳(표시용 문자열 포매팅 2곳)뿐이며, 값을
연산/인덱싱/정렬에 쓰지 않고 그대로 출력만 하므로 코드가 깨지는 곳
없음(확인 완료). 향후 episode_index>0에서 outlier가 잡히면 리포트에
표시되는 frame_idx 숫자의 "의미"만 달라짐(버그 아님).

## 17. 비파괴적 아웃라이어 필터링 — DONE (2026-08-05)
`get_clean_indices(outlier_results=None, delta_outlier_results=None,
exclude_keys=("timestamp",))` 추가. 값 기준 + delta 기준 결과를 합집합으로
모아 이상 프레임을 판정, `exclude_keys`(기본 timestamp)에서만 잡힌 프레임은
후보에서 제외. **비파괴적** — 원본 parquet/데이터셋을 전혀 건드리지 않고
순수 인덱스 계산만 수행, `{"clean_indices", "excluded_indices",
"excluded_count", "total_count"}` 반환. `torch.utils.data.Subset`으로 감싸
`DataLoader`에 바로 사용하는 패턴 채택 (필터링된 새 데이터셋 파일을 만들지
않음 — 3절 원칙 준수).

**검증** (so101_teleop_real episode 1, 232 frames):
- 값 기준(non-timestamp) 62 distinct frame, delta 기준(non-timestamp)
  38 distinct frame → 합집합(중복 제거) 87개, 13개 프레임이 양쪽에서 겹침
  확인
- exclude_keys(timestamp)에서만 잡힌 6개 프레임이 실제로 clean_indices에
  남아있음을 확인
- 손실/중복 없음: `clean(145) + excluded(87) = total(232)` 정확히 일치,
  교집합 0, 합집합이 `range(232)` 전체와 일치
- `Subset(ds, clean_indices)` + `DataLoader(batch_size=8)` 정상 동작
  (19 배치, 145 프레임)

**설계 메모**: `frame_outliers`의 인덱스가 16절 수정으로 "에피소드 내
frame_index"이므로, `get_clean_indices()` 내부에서 이를 `self`의 로컬 행
번호로 재매핑하는 과정을 거침(self를 1회 스캔). `exclude_keys` 기본값은
9-1절 timestamp 과대 탐지 관찰에 근거한 것일 뿐 — 데이터셋/수집 조건이
바뀌면 매번 결과를 재확인해야 하며, timestamp 이상이 항상 무시해도 되는
것은 아님(예: 프레임 드롭의 신호일 수 있음).

## 18. 데이터 무결성 자동 검증 + 성능 최적화 — DONE (2026-08-05)

### 18-1. `check_integrity()` 자동화
`__init__()` 마지막에 `check_integrity()`를 자동 호출, 결과를
`self._integrity_result`에 저장. NaN/Inf 발견 시 `warnings.warn()`으로
경고만 출력(예외 아님) — 학습을 막지 않는 "관대한" 방식 채택(엄격/자동제거
대신). `get_clean_indices()`가 `integrity_results` 파라미터를 받아 기존
`outlier_results`/`delta_outlier_results`와 동일한 합집합 로직으로 병합
가능하도록 확장.

**검증**: NaN 값을 pyarrow로 직접 주입한 임시 복사본(원본 미변경)에서
경고 발생, `_integrity_result` 기록, `get_clean_indices()` 연동,
DataLoader가 막히지 않고 계속 진행되는 것까지 전부 확인.

**주의(docstring 반영)**: `exclude_keys`는 아웃라이어 탐지(통계적 이상)와
같은 의미로 무결성 문제(NaN/Inf, 데이터 손상 신호)에 적용하면 안 될 수
있음 — 실제로 timestamp 등에서 NaN/Inf가 나오면 `exclude_keys=()`로
강제 포함 권장(아직 실증 사례는 없음).

### 18-2. `self[i]` 이미지 디코딩 병목 제거
`compute_custom_stats()`, `detect_outliers()`, `detect_outliers_delta()`,
`check_integrity()` 4개 메서드가 전부 `self[i]`(`__getitem__`)로 프레임을
읽어 float 필드만 필요해도 이미지(`observation.images.wrist_cam`)까지
매번 디코딩하던 것을 확인. 원인은 `LeRobotDataset.__getitem__`이 아니라
`hf_dataset`에 걸린 `set_transform` 콜백이 요청 컬럼 전체를 디코딩하는
구조 — `hf_dataset.select_columns([...])`로 필요한 float 필드만 미리
선택하는 `_light_items(keys)` 헬퍼를 추가해 우회. `__getitem__` 자체와
이미지 로직은 미수정.

**검증(so101_teleop_real, 288 frames, 이미지 필드 보유)**:
- 결과 동일성: 수정 전/후 4개 메서드 전부 바이트 단위로 완전히 동일한
  결과 확인(np.array_equal, entry 단위 필드 비교, 기존 산출 스크립트
  재실행값과도 일치) — **회귀 없음**
- 오버헤드 개선(수정 전 → 후): `check_integrity()` 0.80s→0.028s(29배),
  생성자 무결성 검증분 0.955s→0.023s(41배), `compute_custom_stats()`
  0.77s→0.025s(31배), `detect_outliers()` 1.57s→0.056s(28배),
  `detect_outliers_delta()` 0.76s→0.032s(24배)
- 이미지 필드 없는 `so101_teleop`은 원래도 오버헤드 거의 없었음(수정
  전후 둘 다 <0.03s) — 이번 최적화 효과는 이미지 필드 있는 데이터셋에
  집중

**범위 밖으로 남긴 것**: `visualize_episode()`, `plot_action_distribution()`,
`get_clean_indices()`도 동일 구조의 `self[i]` 병목을 가짐 — 이번 요청
범위(4개 메서드)에 없어 미수정. 필요해지면 동일 방식(`_light_items`)으로
확장 가능. `_light_items()`가 호출마다 `select_columns()`를 새로 하는
것도(비용 자체는 미미) 호출 빈도가 매우 커지면 캐싱 여지 있음.

## 19. 정규화/역정규화 — DONE (2026-08-05)

### 19-1. 설계 결정
ACT 정책 사용 확정에 따라:
- **정규화 기준**: `dataset.meta.stats` 채택 (자체 계산한 `compute_custom_stats()`
  대신) — LeRobot 표준 파이프라인/사전학습 정책과의 호환성 우선. `meta.stats`
  최신성 문제에 대응해 `__init__()`에 `_warn_on_stale_meta_stats()` 추가
  (필드 count와 `total_frames` 불일치 시 경고, 이미지 필드는 서브샘플링이
  정상이라 검사 제외)
- **lerobot `NormalizerProcessorStep` 직접 사용 안 함** — `EnvTransition`/
  `PolicyAction`이라는 policy 파이프라인 전용 타입에 종속돼 있어 3절
  composition 원칙과 안 맞음. 대신 동일 공식(`_apply_transform()`의
  MEAN_STD: `(x-mean)/(std+eps)`, 역변환 `x*std+mean`, `eps=1e-8`을 lerobot
  소스에서 실측 확인)만 재사용해 `normalize()`/`unnormalize()`를 자체 구현
  (비파괴적, 원본 item 미변경)

### 19-2. 이미지 필드: MEAN_STD → IDENTITY 전환
최초 ACT 기본값(`VISUAL: MEAN_STD`)으로 구현 후, 정지 상태 데이터
(so101_teleop_real 56+232f)에서 정규화된 이미지 값이 **-7611~+10917**로
비정상 범위임을 발견. 원인 조사:
- wrist_cam은 실물 카메라가 아니라 MuJoCo 렌더링 화면(팔로워가 시뮬레이션)
- 정지 데이터 이미지 std: 3.6e-05~5.0e-05 (매우 작음)
- 작업 동작 데이터(278f, 팔을 크게 움직임)로 교체 후 std는 28~229배
  커졌으나(0.001~0.01), 정규화 값은 여전히 -320~+385로 비정상 — 원인은
  움직임 크기가 아니라 **렌더링 자체의 특성**(플랫 셰이딩, 텍스처/그림자/
  노이즈 없음 → 대부분 픽셀이 단색 면에 몰림)으로 확정. 데이터를 더
  모아도 자연 이미지 수준(std 0.1~0.3)에는 구조적으로 도달 못 함

**결정**: 이미지 필드만 IDENTITY([0,1] 그대로, SmolVLA/Pi0 선례 따름)로
전환, action/observation.state는 MEAN_STD 유지. ACT의 `norm_map`은
필드별 커스터마이징이 가능하므로 이는 ACT 표준을 벗어나는 게 아니라
설정 범위 안의 조정.

**검증**: 이미지 normalize()가 완전한 항등 변환(`torch.equal` True,
dtype/shape 보존, round-trip 완전 동일) 확인. action/observation.state
round-trip 오차(3.7e-09~3.0e-08)는 변경 전과 동일하게 유지 — 이미지
처리 변경이 다른 필드에 영향 없음 확인.

**후속 확인 필요(범위 밖)**: 실제 ACT 비전 백본을 붙일 때 [0,1] 입력을
기대하는지 재확인 필요. 렌더링을 사실적으로 개선하거나 실물 카메라로
전환되면, 이미지 정규화 방식을 MEAN_STD로 되돌릴지 재검토.

## 20. Action Chunking 조사 (delta_timestamps) — 조사 완료 (2026-08-07)

### 20-1. `delta_timestamps` 동작 방식 (소스 확인)
`dict[str, list[float]]` (초 단위 상대 오프셋). `__getitem__`에서 해당
키만 시퀀스로 확장(`shape (chunk_len, dim)`), `{key}_is_pad` 마스크 동반
생성. 에피소드 경계를 넘는 미래 인덱스는 에러 없이 마지막 프레임 반복 +
`is_pad=True`로 처리(클램핑). ACT는 `action`에만 `chunk_size`(기본 100)
적용, `observation.state`는 `n_obs_steps=1` 고정이라 미적용.
`modeling_act.py`가 `action_is_pad`로 L1 loss를 실제로 마스킹함.

### 20-2. `TeamRobotDataset` 호환성 (실측)
- **안전**: `compute_custom_stats()`/`detect_outliers()`/`detect_outliers_delta()`/
  `check_integrity()` — 18-2절 `_light_items()`가 raw HF 컬럼을 직접 읽어
  `delta_timestamps` 확장 로직을 거치지 않으므로 영향 없음(의도치 않은
  부수 효과로 안전성 확보). `normalize()`도 브로드캐스팅으로 자연히 호환.
- **위험(수정함)**: `visualize_episode()`/`plot_action_distribution()`은
  `self[i]`를 직접 써서, `delta_timestamps` 켜진 상태에서 호출하면
  **에러 없이 조용히 결과가 틀어짐**(관절 차원과 청크 차원 혼동, action
  subplot이 6개가 아니라 chunk_len개로 생성되고 서로 다른 관절이 한
  그래프에 겹쳐 그려짐 — 실측 확인). `self._base.delta_timestamps`가
  설정된 상태에서 대상 키가 겹치면 `warnings.warn()`으로 경고(18-1절과
  동일한 관대한 패턴, 예외 아님) 추가 완료. 하위 호환성(경고 0건) 확인됨.

**결론**: `delta_timestamps={"action": [i/fps for i in range(chunk_size)]}`를
`TeamRobotDataset(**kwargs)`로 그대로 전달하는 방식으로 action chunking
사용 가능. 단 시각화 2개 메서드는 `delta_timestamps` 켜진 인스턴스에서
호출 시 경고를 반드시 확인할 것.

## 21. `so101_teleop` 자동 삭제 방지 — DONE (2026-08-07)

### 배경
`so101_teleop`(시뮬레이션) 데이터셋이 원인 불명으로 최소 2회(9-1절,
2026-08-07) 반복적으로 비워짐. 조사 결과 프로젝트 내에서 이 경로에
쓰기/삭제를 하는 코드는 `run_teleop_simul.py`의 `load_or_create_dataset()`
(`total_episodes==0` 감지 시 자동으로 `shutil.move(root, root+"_corrupted")`
후 새 데이터셋 생성)가 유일함 — `so101_teleop_real`은 동일 코드 패턴을
가지고도 이 분기를 탄 적이 한 번도 없어 반복 재발 패턴과 부합. 직접적
실행 증거(로그, `_corrupted` 백업)는 못 찾았으나 구조적으로 가장 유력한
원인.

**추가 위험 발견**: `LeRobotDatasetMetadata.__init__`은 로컬 meta 로드
실패 시 HF Hub 조회를 시도하고, 성공하면 로컬에 `meta/` 폴더를 새로
만드는 부작용이 있는 구조(소스 확인) — "읽기만 해도" 부작용이 날 수
있는 잠재 위험. 이번엔 401 에러로 중간에 멈춰 실제 피해 없었음.

### 수정
`run_teleop_simul.py`/`run_teleop_real.py`의 `total_episodes==0` 자동
move+recreate 분기를, 콘솔+로그 파일 양쪽에 경고를 남긴 뒤 `input()`으로
사용자 확인(`y`만 진행, 그 외/`EOFError`는 `SystemExit(1)`로 취소)을
요구하도록 변경. `load_or_create_dataset()` 호출 지점이 GUI 뷰어 진입
이전 콘솔 코드 경로임을 확인해 `input()` 사용에 문제없음. move+recreate
실행 시 로그에 "실행됨" 문구 명시.

**검증**: 정상 데이터셋(so101_teleop_real)에서 회귀 없음 확인. 스크래치
폴더로 `total_episodes==0` 상황 인위 재현(원본 미접촉) 후 `n`/`y` 양쪽
입력 경로, `EOFError` 방어 전부 실측 확인.

**주의**: 이 수정으로 재발이 멈추면 원인이 맞았다는 뜻이고, 재발하면
제3의 원인이 있다는 뜻 — 어느 쪽이든 "자동 삭제"라는 위험 요소 자체는
제거됨.

## 22. 에피소드 단위 Train/Val Split — DONE (2026-08-07)

### 설계
`LeRobotDataset`의 `episodes=[...]` 필터링 로드(3절/17절에서 이미 활용
중)를 그대로 재사용하는 방향 — 새 분할 로직 대신 "어느 에피소드가
train/val인지 인덱스 목록만 계산"하는 순수 함수 패턴(18-1절
`get_clean_indices()`와 동일). 실제 `TeamRobotDataset(episodes=[...])`
생성은 호출자가 담당.

`split_episodes(val_ratio=0.2, seed=None)` 인스턴스 메서드로 구현.
`self._base.meta.episodes["episode_index"]`(필터링 여부 무관하게 전체
에피소드 인덱스 보유, 실측 확인)를 `random.Random(seed)`(표준 라이브러리
— 작은 인덱스 순열이라 numpy RNG는 과함)로 셔플 후 `val_ratio`만큼 분리.
`{"train_episodes": [...], "val_episodes": [...]}` 반환. 에피소드 수가
너무 적어 한쪽이 0개가 되면 `warnings.warn()`(18-1절과 동일한 관대한
패턴, 예외 아님) + 최소 1개씩 보장하는 클램프 처리.

### 검증 (so101_teleop_real, 2 episodes)
- `split_episodes(val_ratio=0.2, seed=42)` → `{"train_episodes": [0],
  "val_episodes": [1]}`, 경고 없음(1/1 클램프 정상)
- 반환값으로 `TeamRobotDataset(episodes=train_episodes)`/`(episodes=
  val_episodes)` 실제 생성 → 141/137 frames(합계 278, `meta.total_frames`와
  정확히 일치), `episode_index` 집합 교집합 0건(겹침 없음)
- 같은 `seed=42` 2회 호출 → 완전 동일(재현성 확인)
- `seed=None` 20쌍 호출 → 9/20쌍이 다른 결과(비결정성 확인 — 2 episodes라
  가능한 조합이 2가지뿐이라 정상적인 분포)
- `meta.episodes` 몬키패치로 1-episode 상황 인위 재현(원본 미접촉) →
  `{"train_episodes": [0], "val_episodes": []}` + 경고 발생 확인

**주의**: 현재 데이터(2 episodes)로는 통계적으로 의미 있는 val set 구성이
불가능(1개 vs 1개가 최선). 분할 메커니즘 자체는 검증됐으나, 실제로
의미 있게 쓰려면 에피소드 수가 늘어난 뒤 재적용 필요.

## 23. wrist_roll Clip 문제 진단 및 근본 수정 — DONE (2026-08-12)

### 배경
20-episode pick&place 데이터셋 분석 중, `wrist_roll`의 std가 다른
관절보다 28배 작다는 게 발견됨. 처음엔 "task 특성(손목을 안 씀)"으로
추정했으나, 정밀 진단 결과 **캘리브레이션/매핑 구조 문제**로 판명됨.

### 진단
- 15/20 episode에서 `wrist_roll` action이 소수점 5자리까지 정확히
  동일한 값(-2.74385 rad)으로 고정 — 이 값이 follower
  `actuator_ctrlrange` 하한과 정확히 일치, 즉 clip되어 박혀있는 것으로
  확인
- 원인 규명: leader 캘리브레이션 실측 가동범위(334.24°)가 follower
  MJCF 모델 범위(319.99°)보다 넓은 구조적 불일치. `gripper`를 제외한
  5개 관절이 "leader 도 = follower 도" 1:1 직접 매핑을 쓰는데, 두 범위가
  정확히 안 맞으면 가장자리에서 필연적으로 clip 발생
- 5개 관절 전부 재확인 결과: `wrist_flex`(+14.71°)가 가장 심각,
  `shoulder_lift`(+8.88°)도 유의미한 불일치, `shoulder_pan`/`elbow_flex`는
  leader가 오히려 좁아 안전
- 재캘리브레이션(`lerobot-calibrate`)만으로는 근본 해결 안 됨 확인 —
  "leader 실측범위가 follower 모델범위보다 넓다"는 구조 자체가 안 바뀜

### 수정
`gripper`와 동일한 방식(비례 스케일링)으로 5개 관절 전부 전환.
`leader_action_to_radians()`가 leader 실측 range → follower
`ctrlrange`로 선형 보간하도록 변경. `compute_leader_degree_range()`
헬퍼 추가(lerobot `_normalize()` DEGREES 공식 재현). 검증: 5개 관절
전부 leader range 양 끝값이 follower ctrlrange 양 끝값과 소수점 5자리
일치, `gripper` 로직 영향 없음 확인.

### 부수 발견: 자체 캘리브레이션 스크립트(`205_all_motor_calibration_leader.py`) 폐기
lerobot 공식 절차와 비교해 3가지 문제 확인:
- `Homing_Offset`을 2의 보수로 씀(스펙은 sign-magnitude, 12비트,
  magnitude 최대 2047) — 코드 레벨 버그, 이전에 "Magnitude exceeds 2047"
  에러의 원인으로 추정
- Range 기록이 "Enter 순간 1회 스냅샷" 방식이라 인간 반응속도에 의존
  (lerobot 공식은 계속 polling하며 실제 극값 추적, 더 견고)
- `Present_Position` 12비트 마스킹 — 연속회전 관절(`wrist_roll`)에서
  잠재적 오작동 위험(이번엔 4096 미만 값이라 잠복 상태)

**결정**: 앞으로 `lerobot-calibrate` 공식 CLI만 사용. 자체 스크립트는
폐기.

### 재캘리브레이션 + 검증
`lerobot-calibrate --teleop.type=so101_leader ...`로 재캘리브레이션
진행. `wrist_roll`이 연속회전 관절이라 `range_min=0`/`range_max=4095`로
소프트웨어상 하드코딩됨을 확인(물리적으로는 334° brake 관절 — 이 불일치
자체는 여전히 존재하나 비례 스케일링이 흡수).

재수집 8 episode(20-27) 검증: `wrist_roll` clip 프레임 0%(이전 16/20
episode가 90% 이상 고정이었음), `check_integrity()` 이상 없음(28
episode/9006 frames 전체). 남은 저 std(±0.3°)는 clip이 아니라 진짜
task 특성(그리퍼로 쥘 때만 미세 조정)으로 판단 — 그래프상 gripper
동작 타이밍과 일치 확인.

### 공 위치 랜덤화 기능 추가
S 키(녹화 종료) 시 빨간 공 위치를 하얀 접시 기준 상대offset으로 랜덤
재배치. 범위(`offset_x∈[0.00,0.11]`, `offset_y∈[-0.18,0.01]`)는 임의
설정이 아니라 28-episode 데이터를 forward kinematics로 분석(gripper
닫힘 + 테이블 높이 조건 719 프레임)해 도출. `data.qpos` 직접 대입 +
`mj_forward()` 방식 재사용(기존 `reset_pick_object()` 패턴). 재배치 후
20스텝 안정화 시뮬레이션 추가. 검증: 시드 고정 시 재현성 확인, 2000개
샘플로 균등분포 확인.

## 24. 재캘리브레이션 이후 신규 17-Episode 데이터셋 검증 — DONE (2026-08-12)

### 24-1. wrist_roll Clip 해결 유지 확인
17/17 episode 전부 follower 하한(-2.74385 rad) 고정 프레임 0%. 23절의
수정이 신규 수집에서도 안정적으로 유지됨을 확인.

### 24-2. 랜덤 재배치 정상 동작
로그-데이터 대조로 17/17 offset이 설계 범위 내, 전부 서로 다른 값임을
확인. 단, 이 17개는 접시-제외(rejection sampling) 로직 추가 **이전**에
녹화된 것으로 타임스탬프 대조 확인(로그 마지막 기록 20:36:19 vs 코드
수정 20:38:38) — 3개 episode(11, 15, 16)가 접시 중심에서
0.85~1.65cm 거리로 접시 제외 반경(6.5cm)보다 훨씬 안쪽에서 시작.
현재 코드에는 이미 제외 로직이 반영되어 있어 다음 녹화부터는 해당 없음.

### 24-3. Episode 11 제외 결정
값/delta 기준 아웃라이어 0건, 최단 프레임(225), `wrist_roll` range
0.00°로 유일하게 완전 무변화 — 사용자 확인 결과 **S 키 오조작으로 인한
무효 데모**로 판명. 원본 데이터는 보존(비파괴적 원칙), 학습 시
`episodes` 파라미터로 제외.

**갱신된 `split_episodes()` 결과** (episode 11 제외 16개 기준,
`seed=42`):
- `train_episodes = [0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16]` (13개)
- `val_episodes = [5, 7, 9]` (3개)

### 24-4. wrist_roll std 불균형 재확인 (223배)
Clip이 완전히 사라진 뒤에도 `wrist_roll` std가 여전히 가장 작음(불균형
비율이 오히려 28배→223배로 커짐) — 이는 캘리브레이션 문제가 아니라
**이 조작자/task가 손목을 거의 안 쓴다는 진짜 task 특성**으로 재확인.

### 24-5. 아웃라이어 탐지 패턴
값 기준(z=2.0) 쏠림이 `wrist_roll`→`shoulder_lift`로 이동(둘 다 저분산
필드 과대 탐지의 재현, 새로운 이상 아님). delta 기준(z=3.0)은
`wrist_roll`에서 다시 최다 — 값/delta 두 방식이 서로 다른 신호를
잡는다는 14절 결론이 이번에도 유지됨. Episode 12~14가 delta 기준으로
유독 적어 참고 필요(추가 조치는 보류).

### 다음 단계: ACT 학습 파이프라인 연결 (진행 중)
`lerobot-train` 조사 결과, `TeamRobotDataset`은 학습에 직접 연결되지
않고(공식 CLI가 원본 `LeRobotDataset`을 직접 구성) **학습 전 QA
도구**로 역할이 확정됨(3절 composition 설계 의도와 일치).
`--dataset.episodes`로 `split_episodes()` 결과를 그대로 전달 가능.
`--policy.normalization_mapping.VISUAL=IDENTITY`로 19-2절 결정을
오버라이드 필요(ACT 기본값은 VISUAL도 MEAN_STD임을 소스로 재확인).

GPU 확인: RTX 5050(8GB VRAM, 드라이버 CUDA 13.2) 보유, 현재
`torch==2.10.0+cpu`(CUDA 미지원 빌드) — `cu128`/`cu130` 빌드로 재설치
진행 예정. CPU 스모크테스트 대신 처음부터 GPU로 검증하는 방향으로 전환.

## 25. ACT 학습 파이프라인 GPU 스모크테스트 — DONE (2026-08-12)

### GPU 환경 구축
CPU 전용 `torch==2.10.0+cpu`를 `cu128` CUDA 빌드로 재설치(RTX 5050,
드라이버 CUDA 13.2 지원 확인 후 진행). `torch.cuda.is_available()`
True, `RTX 5050` 정상 인식, GPU 텐서 연산/torchvision C++ 연산 전부
정상 확인. `torchcodec`는 Windows 미지원(현재 비디오 필드 없는 데이터
구성과는 무관, 추후 비디오 데이터 다룰 때 재검토 필요).

### lerobot-train 스모크테스트
`--config_path` YAML 방식으로 실행(`PYTHONIOENCODING=utf-8` 필요,
Windows 콘솔 인코딩 문제 회피). `dataset.episodes`로 24-3절
`train_episodes`(13개, episode 11 제외) 전달, `policy.normalization_mapping.VISUAL=IDENTITY`
반영 확인. 30 스텝 스모크테스트 결과:
- Loss 60.803 → 7.278로 감소, 2회 실행 재현성 확인(GPU 결정론적 동작)
- 데이터 로딩 → 정규화(VISUAL=IDENTITY 반영) → GPU 학습(device=cuda)
  → 체크포인트 저장 → `last` 심볼릭 링크 생성까지 전 과정 에러 없음
- 겪은 문제: Windows 심볼릭 링크 생성 권한 부족 → 개발자 모드 활성화로
  해결(다음 환경 세팅 시 참고)

**결론**: 파이프라인 전체(데이터 로딩부터 체크포인트 저장까지)가
신뢰 가능한 상태로 확인됨.

## 26. MuJoCo 추론 파이프라인 구축 및 검증 — DONE (2026-08-12)

### 조사
`lerobot-eval`은 gymnasium 환경 등록이 필요해 커스텀 MuJoCo 씬에는
과함(우리 목적엔 부적합) — 대신 `run_teleop_real.py`의 제어 루프에서
`leader.get_action()` 자리를 `ACTPolicy.select_action()`으로 교체하는
방식이 훨씬 적은 비용으로 동일 목적(policy가 실제로 팔을 제어하는지
확인) 달성 가능함을 확인.

`ACTPolicy.from_pretrained()` + `make_pre_post_processors()`가
체크포인트 폴더의 `policy_preprocessor.json`(정규화 설정,
`VISUAL=IDENTITY` 포함)을 자동 로드 — 추론 스크립트에서 정규화 설정을
재지정할 필요 없음. Action chunking(`chunk_size=100`)도
`select_action()` 내부 큐(`_action_queue`)가 완전 자동 처리 — 별도
청킹 로직 불필요. Policy 추론은 follower 좌표계로 직접 예측하므로,
23절의 leader→follower 스케일링 변환이 아예 필요 없어져 오히려 teleop
루프보다 단순한 구조가 됨.

### 구현: `run_inference_mujoco.py`
`run_teleop_real.py`는 미수정(참고만). 리더암 연결/녹화 관련 코드
전부 제거, 관측치(qpos, wrist_cam 렌더)는 기존 루프 패턴 재사용.
50Hz 물리 스텝 / 30Hz policy 호출(이미지 렌더와 동일 캐던스) 분리
구조 유지. `ctrl_lo`/`ctrl_hi`(MuJoCo `mjtNum`, float64) dtype 승격
버그(8절 패턴) 재발 방지를 위해 action을 ctrl에 쓰기 직전
`.astype(np.float32)` 명시 적용. `R` 키로 로봇+물체 초기화 및
`policy.reset()` 동시 호출. 물체 위치는 재현 가능한 관찰을 위해 고정
(일반화 성능이 아니라 "의미 있게 움직이는지" 확인이 목적이므로).

**검증**(헤드리스 5초 실행, 30-step 스모크테스트 체크포인트 사용):
물리 스텝 2510회/policy 호출 150회(50Hz/30Hz 캐던스 정확히 일치),
action 전부 유한(NaN/Inf 없음), dtype float32 유지 확인. 뷰어로 직접
관찰한 결과 이 체크포인트(30 스텝)는 예상대로 의미 있는 pick&place
행동을 보이지 않음(학습 부족, 파이프라인 문제 아님) — 배관 자체는
정상 확인됨.

## 27. 본격 학습 실행 — 진행 중 (2026-08-12)

목표: 몇 시간 이내로 끝나는 "감 잡기" 규모 학습(스모크테스트 30 스텝
대비 대폭 확장). `dataset.episodes`=13개 train episodes,
`val_episodes=[5,7,9]`로 평가 구성. `output_dir=./train_main_run_output`,
`wandb` 로깅 시도. 백그라운드 실행 중 — 완료 후 loss 추이, 체크포인트별
성능, `run_inference_mujoco.py`로 재검증 예정.

## 28. 알려진 이슈 — mujoco_env 의존성 버전 충돌 (2026-08-13, 미해결)

대시보드용 `pip install streamlit` 작업 중 `pip check`로 발견된, **대시보드
작업과는 무관한 mujoco_env 전체의 기존 의존성 충돌** 2건을 기록해둔다.
근본 해결(다운그레이드/업그레이드 등)은 이번엔 다루지 않고 조사만 했다 —
파이프라인 안정성에 영향을 주는지 계속 지켜볼 필요가 있음.

### 28-1. fsspec 2026.7.0이 datasets 4.8.5 요구 범위를 벗어남
`pip check` 결과: `datasets 4.8.5 has requirement fsspec[http]<=2026.2.0,
>=2023.1.0, but you have fsspec 2026.7.0.`

파일 타임스탬프로 확인한 설치 시각 — `fsspec-2026.7.0.dist-info`:
2026-07-29 20:23, `datasets-4.8.5.dist-info`: 2026-07-28. 즉 **이번
대시보드 작업(streamlit 설치, 2026-08-13) 3주 전부터 이미 존재하던
문제**이며, streamlit 설치 로그의 "Installing collected packages"
목록에도 fsspec은 없었다(그 설치가 실제로 건드린 건 `pyarrow`
25.0.0→24.0.0뿐 — streamlit이 `pyarrow<25,>=7.0`을 요구해서다. datasets의
`pyarrow>=21.0.0` 요구와는 충돌 없고, `pip check`도 pyarrow는 지적하지
않음).

**현재 상태**: `LeRobotDataset` 로딩, `lerobot-train` 학습(스모크테스트~
10000-step 본학습~7-체크포인트 스윕까지), `ACTPolicy.from_pretrained()`
등 파이프라인 핵심 동작은 이 세션 내내 문제없이 반복 확인됨 — `datasets`가
아직 fsspec의 바뀐 API에 실제로 부딪히는 코드 경로를 안 타고 있는
것으로 추정된다. 다만 요구사항 위반 상태 자체는 남아있어 잠재 리스크로
기록해둔다.

### 28-2. opencv-python-headless가 numpy<2.3.0을 요구하는데 numpy 2.4.6 설치됨
`pip check` 결과: `opencv-python-headless 4.12.0.88 has requirement
numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 2.4.6.`
pyarrow/fsspec과는 무관한 별개의 기존 충돌.

`run_teleop_real.py`의 `CameraPreviewWindow` 독스트링에 "이 환경엔 numpy
호환성 때문에 GUI 기능이 빠진 opencv-python-headless가 깔려 있어(`cv2.imshow`
사용 불가)"라는 기존 기록이 있음 — 즉 opencv-python-headless 자체가 애초에
numpy 호환성 문제 때문에 선택된 것인데, 지금은 그 opencv-python-headless의
numpy 요구사항조차 어긋난 상태. 이 프로젝트에서 opencv를 실제로 쓰는
코드 경로가 이 불일치로 영향을 받는지는 이번 조사에서 확인하지 않음 —
별도 확인 필요.

## 29. 웹 개발 키트 아키텍처 설계 — DONE (2026-08-13)

### 배경
부트캠프 내부용(각자 자기 컴퓨터에서 SO-101 하드웨어를 붙여 돌리는 전제) 웹
개발 키트를 만드는 새 세션(`so101_web` 폴더, 이전 연구 세션과 분리) 시작.
목표는 [데이터 수집] → [QA & 검증] → [학습] → [추론] 4단계를 웹에서 완결시키는
것이되, 위험도가 낮은 단계부터 순서대로 구현: ①읽기 전용 대시보드(완성) →
②데이터 수집+QA 웹 트리거(이번 절 이후 구현) → ③학습+추론 웹 트리거(설계만,
구현은 이후 세션).

### 29-1. 기존 상태 확인
`dashboard/`에 4-page Streamlit 골격이 이미 존재(①만 읽기 전용 완성).
`run_teleop_real.py`/`run_inference_mujoco.py`는 이미 `--port/--id/--repo-id/
--root/--checkpoint/--headless` 등으로 CLI 파라미터화돼 있어(개인 경로/포트
하드코딩 없음) 이번 설계가 그 위에 얹기만 하면 되는 좋은 기반이 있음을 확인.
`run_teleop_real.py`는 이미 stdout/stderr를 `logs/session_*.log`로 자동
tee함. `train_main_run_config.yaml`은 `dataset.root`가 이 컴퓨터 전용
절대경로로 하드코딩돼 있음(3단계 구현 시 해결할 항목으로 기록만, 이번엔
안 건드림). `so101_web` 폴더 자체는 git 저장소가 아님(확인됨) — 로컬 설정
파일을 자유롭게 추가해도 커밋/gitignore 이슈 없음. `psutil`은 이미 설치돼
있어(7.2.2) 새 의존성 없이 프로세스 생존 체크 가능, `streamlit-autorefresh`는
미설치(로그 실시간 갱신용으로 이번에 추가하기로 결정, 29-5절).

### 29-2. 핵심 설계 축 1: Job/Lock 관리 (`dashboard/lib/process_manager.py`)
데이터수집(시리얼포트+MuJoCo뷰어 점유)/학습(GPU)/추론(GPU+MuJoCo뷰어)이
동시에 돌면 자원이 충돌하므로(사용자가 처음부터 우려한 지점) 세 스테이지가
서로를 배제하는 lock이 필요. QA는 순수 계산이라 lock 불필요.

- `state/locks/<stage>.json` — `{pid, started_at, cmd, resources, log_path}`
- `resources` 태그(`mujoco_viewer`, `gpu`, `serial_port`)로 겹치는 자원을
  쓰는 다른 lock이 살아있으면 시작 버튼 비활성화 + 이유 표시
- `psutil.pid_exists(pid)`로 생존 체크 → 비정상 종료로 lock만 남은 경우
  (stale lock) 자동 정리
- subprocess의 stdout/stderr를 process_manager가 직접
  `logs/jobs/<stage>_<timestamp>.log`로 캡처(각 스크립트의 자체 로깅
  유무와 무관하게 항상 동작 — ③④ 스크립트엔 아직 tee 로직이 없으므로
  이렇게 통일해야 나중에 그대로 재사용 가능)
- `st.session_state`가 아니라 **디스크 lock 파일**에 상태를 두는 이유:
  Streamlit 페이지를 새로고침/재시작해도 "지금 뭔가 돌고 있다"는 사실을
  잃지 않아야 하기 때문
- `start_job(stage, cmd, resources)` / `get_job_status(stage)` /
  `stop_job(stage, force)` / `tail_log(log_path, n)` 시그니처는 ②에서
  먼저 쓰지만 ③④가 그대로 재사용하도록 설계 — 3단계 구현 시 이 파일을
  다시 설계할 필요 없게 하는 게 목적. 향후 큐 시스템으로 확장할 때도
  `state/queue.json` 형태로 이 lock 구조 위에 얹을 수 있음

### 29-3. 핵심 설계 축 2: QA → 학습 자동 인식 (`state/qa_snapshots/`)
QA 페이지에 "📌 이 결과를 학습용으로 확정" 버튼 → `state/qa_snapshots/
<repo_id>_<timestamp>.json` 저장. 내용: `repo_id`, `root`,
`total_episodes/frames`, `train_episodes`, `val_episodes`,
`excluded_count`, 사용한 `z_thresh`, 생성 시각.

페이지 로딩마다 자동 계산이 아니라 **명시적 스냅샷**으로 한 이유: 데이터셋에
나중에 episode가 더 추가돼도 "그때 실제로 이 조건으로 학습했다"는 기록이
남아야 재현 가능(24-3절 "episode 11 제외" 같은 판단을 매번 다시 추적하지
않아도 됨). ③ 구현 시 `state/qa_snapshots/`에서 최신 스냅샷을 자동 로드해
"이 스냅샷으로 학습 시작" 버튼이 `--dataset.episodes`를 미리 채우는 방식으로
QA→학습 연결을 완성할 예정.

### 29-4. 로컬 개인화 설정 (`dashboard/lib/local_config.py`, `config/local_settings.json`)
팀원마다 다른 COM 포트/캘리브레이션 id/데이터셋 경로를 페이지 ①의 폼에서
편집·저장. 필드: `leader_port`, `leader_id`, `calibration_dir`, `repo_id`,
`root`. 저장 후 다음 실행부터 기본값으로 자동 채워짐.

### 29-5. 결정된 사항 (사용자 확인)
- 로그 실시간 갱신: `streamlit-autorefresh` 패키지 추가(수동 새로고침 대신
  자동 갱신 — 녹화 중 뷰어 창을 왔다갔다 확인하기 편하게)
- "세션 강제 종료" 버튼: 데이터 수집 페이지에 포함(위험 경고 + 확인 단계
  거치게 설계 — `taskkill`로 강제 종료 시 녹화 중이던 미저장 episode가
  유실될 수 있음을 명시)
- 실물 리더암 연결 검증: 이번 세션엔 리더암 연결 불가 상태라, ②
  구현 후 검증은 더미 명령(process_manager 자체 동작)까지만 하고 실제
  `--port` 연결 검증은 사용자가 나중에 직접 진행하기로 함

### 29-6. 구조적 제약 (사용자에게 먼저 확인, 이견 없이 진행하기로 함)
- **S/X 키보드 녹화 제어는 웹으로 옮기지 않음**: `run_teleop_real.py`의
  episode 녹화 시작/종료가 MuJoCo 뷰어의 `key_callback`(로컬 GUI 창의
  키보드 이벤트)에 묶여 있어, 웹 버튼은 "세션(프로세스) 시작/종료"까지만
  가능 — 실제 녹화 시작/종료는 여전히 뜬 MuJoCo 창에서 사람이 S/X를 눌러야
  함. `key_callback`을 파일/소켓 기반 외부 신호로 바꾸는 추가 작업 없이는
  불가능한데, 검증된 스크립트 로직을 건드리는 리스크 대비 이득이 낮다고
  판단해 이번 범위에서 보류
- **"각자 로컬에서 Streamlit 실행"이 고정 전제**: 웹 버튼 → 로컬 subprocess
  실행이라는 설계 전체가 "브라우저와 Streamlit 서버가 같은 컴퓨터"라는
  전제 위에 있음. 여러 명이 원격 서버 하나를 공유 접속하는 구조로 바뀌면
  이 설계는 성립하지 않음(MuJoCo 뷰어/카메라 창이 서버에서 뜨는 게
  무의미해짐) — 현재는 이 전제가 유효하다고 확인됨
- **tkinter 카메라 미리보기 창도 로컬 GUI**: 위와 같은 이유로 브라우저에
  임베드되지 않고 별도 데스크톱 창으로 계속 뜸

### 29-7. 이번에 구현할 범위 (② 데이터 수집 + QA 웹 트리거)
신규: `dashboard/lib/process_manager.py`, `dashboard/lib/local_config.py`,
`state/`(locks, qa_snapshots), `logs/jobs/`.
수정: `pages/1_데이터_수집.py`(설정 폼 + 시작/종료 버튼 + 상태/로그 뷰어),
`pages/2_QA_검증.py`(z_thresh 재실행 파라미터 + 스냅샷 저장 버튼),
`lib/data_sources.py`(job 종료 후 캐시 무효화 헬퍼).
③④(학습/추론)는 이번엔 웹 트리거를 붙이지 않음(29-2절 이유) — 다만
`process_manager.py`의 시그니처는 재사용 가능하게 설계됨.

## 30. 코드 리뷰 반영 — lock 원자성 + graceful shutdown — DONE (2026-08-14)

29절 구현 이후 리뷰에서 지적된 세 가지 중 위험도가 높은 두 가지(①②)를
바로 수정하고, 나머지 하나(③)는 이유를 남기고 다음 단계로 미룸.

### 30-1. `start_job()` lock 생성의 check-then-act race condition 수정
**문제 확인**: `process_manager.start_job()`이 `get_job_status(stage)["running"]`로
먼저 확인하고 `subprocess.Popen()`을 거쳐 한참 뒤에야 lock 파일을 쓰는
check-then-act 구조였음 — 그 사이(Popen 호출 포함, 수백ms~수초) 동시에
두 호출이 들어오면(예: 웹 버튼 더블클릭) 둘 다 "실행 중 아님"으로 판단해
통과, MuJoCo 뷰어가 두 개 뜰 수 있는 실질적인 race condition이었음(코드
직접 확인, 추측 아님).

**수정**: lock 파일 생성 자체를 `os.open(path, os.O_CREAT | os.O_EXCL |
os.O_WRONLY)`로 원자적 연산으로 바꿈. 먼저 `pid=None, starting=True`인
placeholder를 원자적으로 생성해 "이 stage를 시작할 권리"를 선점하고, 그
다음에 실제 `Popen()`을 호출해 진짜 pid로 lock을 덮어쓰는 2단계 구조로
변경. 두 호출이 동시에 원자적 생성을 시도하면 OS 레벨에서 단 하나만
성공하고 나머지는 `FileExistsError` → `JobConflictError`로 변환됨.
placeholder 상태(`pid=None`)를 stale-lock 정리 로직이 죽은 프로세스로
오판하지 않도록 `_all_locks()`에 특별 처리 추가(단, 부모가 start_job()
도중 죽는 등 placeholder가 60초 넘게 안 풀리면 stale로 간주해 정리).

**검증**: 실제 스레드 8개가 `threading.Barrier`로 동시에 풀려나 같은
stage(`data_collection`)에 대해 `start_job()`을 동시 호출하는 테스트 —
정확히 1개만 시작 성공, 나머지 7개는 `JobConflictError`로 차단됨을 확인
(더미 순차 실행이 아니라 실제 동시성 재현). 종료 후 lock 파일 잔여물 없음.

### 30-2. 강제 종료 시 시리얼 포트 정리 — graceful-then-force로 전환
**문제 확인**: `run_teleop_real.py`에 `signal`/`atexit` import가 아예
없었음(코드 직접 확인). `try/finally`로 `safe_disconnect(leader)`(포트
닫기)와 `dataset.finalize()`를 호출하는 정리 로직 자체는 이미
있었지만(803-807행), 이건 `while viewer.is_running():`이 정상
종료(뷰어 창을 사용자가 직접 닫음)될 때만 실행되는 구조 — 즉 "정리 로직이
없다"가 아니라 "있는데 외부 종료 신호에 연결이 안 돼 있다"가 정확한
진단. 기존 `process_manager.stop_job()`은 `taskkill /F`(하드 kill)를
바로 썼는데, 하드 kill은 OS가 프로세스를 즉시 회수해 Python이 스택을 풀
기회 자체가 없으므로 이 finally가 전혀 실행되지 않았음(포트 안 닫힘,
`dataset.finalize()` 안 됨).

**수정**:
- `run_teleop_real.py`: `signal.SIGINT`/`SIGTERM`/`SIGBREAK`(Windows,
  CTRL_BREAK_EVENT) 핸들러 등록 → `shutdown_requested` 플래그를 세우고,
  `while viewer.is_running() and not shutdown_requested["flag"]:`로 루프
  조건을 바꿔 신호를 받으면 정상 종료 경로(기존 finally)를 그대로 타게
  함. 새 정리 로직을 추가한 게 아니라 있던 걸 신호에 연결한 것.
- `dashboard/lib/process_manager.py`: `start_job()`에서 자식을
  `CREATE_NEW_PROCESS_GROUP`으로 띄워(이미 하고 있었음) 부모가 이 자식만
  선택적으로 겨냥해 `CTRL_BREAK_EVENT`를 보낼 수 있게 함(`CTRL_C_EVENT`는
  콘솔을 공유하는 전체 그룹에만 보낼 수 있어 특정 프로세스 선택 불가 —
  Windows 자식 프로세스를 정상 종료시키는 표준 패턴). `stop_job()`을
  graceful-then-force로 변경: 먼저 `os.kill(pid, CTRL_BREAK_EVENT)`(Windows)
  / `SIGTERM`(그 외)을 보내고 `graceful_timeout_s`(기본 5초) 동안
  프로세스가 스스로 끝나길 기다린 뒤, 그래도 살아있으면 `taskkill /F`로
  강제 종료 escalate.
- `pages/1_데이터_수집.py`: 강제 종료 확인 문구에 "정상 종료를 먼저
  시도합니다(포트 닫기 등 정리 시간 최대 5초)" 추가.

**검증**: 실물 리더암이 없어 lerobot의 실제 `SO101Leader.disconnect()`
내부 동작 자체는 검증 못 함(그건 lerobot이 이미 테스트한 코드) — 대신
`run_teleop_real.py`와 동일한 신호 처리 구조(핸들러 등록 → 플래그 →
루프 → try/finally)를 그대로 복제한 스탠드인 프로세스로 **신호 전달
메커니즘 자체**(process_manager의 graceful stop이 실제로 CTRL_BREAK_EVENT를
전달하고, 그게 핸들러를 거쳐 try/finally의 정리 코드까지 실행시키는가)를
검증:
- 시나리오 1: graceful stop → 0.21초 만에 "포트 close" 마커 파일 생성 +
  로그에 `loop exited normally via shutdown flag` → `port closed` →
  `cleanup finally done` 순서로 정확히 기록됨(5초 timeout 훨씬 이전에
  정상 종료)
- 시나리오 2(대조군): 신호를 명시적으로 무시(`SIG_IGN`)하는 프로세스 →
  3초 timeout까지 정확히 기다린 뒤 강제 종료로 escalate, `psutil`로
  프로세스 실제 종료 확인 — graceful이 안 먹히는 경우의 안전망도 확인됨

### 30-3. 알려진 이슈(미해결, 다음 단계로 보류): 로그 중복 캡처
`dashboard/lib/process_manager.py`가 `run_teleop_real.py`의 stdout/stderr를
`logs/jobs/<stage>_<timestamp>.log`로 직접 캡처하는데, `run_teleop_real.py`
자체도 `_setup_file_logging()`으로 같은 stdout/stderr를
`logs/session_<timestamp>.log`에 자체적으로 tee함(이 스크립트에 원래부터
있던 기능, 29-1절에서 확인됨) — 결과적으로 로그가 두 파일에 중복 기록됨.

**왜 지금 안 급한가**: 로그가 두 군데 남는 것 자체는 기능을 막지 않음
(대시보드는 자기 캡처본만 읽으므로 정상 동작). 다만 두 메커니즘이 같은
OS stdout 핸들을 동시에 다루는 구조라, 실물 하드웨어로 장시간 세션을
돌릴 때 인터리빙(줄이 섞여 깨지는 것) 또는 파일 핸들 경합이 실제로
발생하는지는 이번 세션에서 확인 못함(더미 job으로는 짧은 세션만
테스트했음) — 재확인 필요.

**다음 단계**: ③④(학습/추론) 웹 트리거를 구현할 때, 그 스크립트들에는
애초에 자체 tee 로직이 없으므로 이 시점에 "로깅은 process_manager가
전담하고 각 스크립트의 자체 tee는 제거/비활성화"하는 방향으로 통일
정리하는 게 자연스러움 — 그때 같이 정리하기로 함.

### 30-4. "조용히 죽는" 크래시에 최소한의 경고 배너 추가 — DONE (2026-08-14)
**확인**: `get_job_status()`가 정상 종료와 크래시(포트 없음/import
에러/캘리브레이션 파일 없음 등으로 시작 직후 즉사)를 구분하지 않는지
실제로 재현해서 확인함 — 즉시 크래시하는 더미 프로세스를 띄워보니
`job_status["running"]`이 크래시 직후(첫 폴링에서 이미) `False`로
정확히 감지·정리되긴 하지만, 그 사실 자체를 사용자에게 알리는 경로가
없어 정상 종료와 똑같이 그냥 시작 폼으로 조용히 돌아감(로그 파일엔
`Traceback (most recent call last):`이 정확히 남아있는데 UI가 안 보여줌)
— "조용히 멈춤"이 맞다고 확인됨.

**수정**: 정교한 에러 파싱 대신 최소한의 문자열 휴리스틱만 추가.
`process_manager.looks_like_crash(log_text)` — 로그 tail에서
`Traceback (most recent call last)`/`SerialException`/
`ModuleNotFoundError`/`FileNotFoundError`/`PermissionError`/
`ConnectionError` 같은 흔한 패턴을 대소문자 무시하고 찾기만 함(③④에서도
재사용 가능하도록 process_manager.py에 위치). `1_데이터_수집.py`는 "직전
rerun에선 실행 중이었는데 지금은 아님" 전환을 감지할 때(기존
캐시 무효화 로직과 같은 지점) 마지막 로그 tail에 이 휴리스틱을 적용해
crash로 보이면 `❌ 세션이 예기치 않게 종료됐습니다` 경고 + 로그 스니펫을
`session_state`에 남겨 사용자가 뭔가 조작하기 전까지 계속 보이게 함(새
세션을 시작하면 지움).

**빠른 크래시(수백ms 내 즉사) 케이스 추가 처리**: 위 "직전 rerun엔
실행 중"이라는 전제가, 시작 버튼을 누르자마자 첫 렌더 전에 죽는 경우엔
성립하지 않을 수 있음(오토리프레시가 한 번도 "실행 중"을 못 봄) — 그래서
시작 버튼 핸들러에서 `start_job()` 성공 직후 곧바로 `was_running`/로그
경로를 session_state에 미리 기록해, 다음 rerun에서 바로 전환 감지가
성립하도록 함(인위적인 sleep 없이 처리 — Python `stderr`는 기본
unbuffered라 traceback이 프로세스 종료 전에 이미 로그 파일에 반영됨).

**검증**: ① `looks_like_crash()` 유닛 테스트(양성/음성 케이스), ②
실제로 크래시하는 더미 프로세스를 띄우고 `AppTest`로 "직전엔 실행 중"
상태를 재현해 경고 배너 + 로그 스니펫(Traceback 포함)이 정상 렌더링됨을
확인, ③ 대조군으로 정상 종료(크래시 아님) 더미 프로세스는 경고가 뜨지
않음을 확인(오탐 없음). 시작 버튼 클릭 자체를 통한 즉사 케이스는 실제
`run_teleop_real.py`(mujoco/lerobot import 필요, 하드웨어 의존)를 통해
E2E로 검증하지는 못함 — 코드 리딩으로 로직을 확인한 수준이며, 실물
연결 시 재확인 권장.

### 30-5. 실물 하드웨어 검증 — DONE (2026-08-14)
30-4절에서 남겨둔 실물 검증 체크리스트를 리더암(COM7, USB-Enhanced-SERIAL
CH343, 기존 `full_arm_calibration_leader.json` 캘리브레이션 사용)으로
전부 확인. `config/local_settings.json`에 `leader_port=COM7` 저장해둠
(팀원별로 다르면 대시보드 "① 데이터 수집" 페이지의 로컬 설정 폼에서
각자 수정).

**① 정상 녹화 흐름 — PASS**: 실제 대시보드(`streamlit run dashboard/app.py`)에서
세션 시작 → 실물 리더암으로 S 키 녹화 → S 키 종료 → 뷰어 창 정상 닫기까지
사용자가 직접 수행. `so101_teleop_real` episode 수 17→18, frame 수
9022→9483(+461, 로그의 "461 frames"와 정확히 일치)로 실측 확인. 로그에
`[REC] 녹화 시작 (episode 17)` → `저장 완료: 461 frames (총 18
episodes)` → `공 위치 재배치` → `dataset.finalize() 호출 완료` →
`리더암 연결 해제 완료` 순서로 정상 기록됨(이번엔 신호가 아니라 뷰어
창을 직접 닫는 정상 종료 경로 — 30-2절에서 고친 신호 기반 경로와는
별개로, 기존 정상 종료 경로도 여전히 문제없음을 확인한 셈).

**② graceful stop 시 포트 close — PASS**: `process_manager.stop_job()`으로
실물 세션에 graceful stop 전송 → 1.6초 만에 로그에 `종료 신호(21) 수신`
→ `dataset.finalize() 호출 완료` → `리더암 연결 해제 완료` 순서로 정상
기록됨(30-2절에서 더미로 검증했던 것과 동일한 경로를 실물 리더암으로
재확인).

**③ 강제 종료 후 재시작 — PASS**: ②의 graceful stop 직후 곧바로 같은
COM7로 재시작 → 포트 점유 에러 없이 리더암 재연결 성공(로그에 "could not
open port" 등 에러 없음, 캘리브레이션 로딩까지 정상). 애초에 이 작업을
시작한 동기였던 문제가 실물 기준으로 해결됐음을 확인.

**④ 크래시 배너 — PASS**: 실제 `run_teleop_real.py`를 존재하지 않는
포트(`COM99`)로 실행 → `leader.connect()`에서 `ConnectionError`/
`SerialException` 미처리 예외로 크래시(로그에 전체 traceback 남음) →
`AppTest`로 이 실제 로그를 페이지에 주입해 `❌ 세션이 예기치 않게
종료됐습니다` 배너 렌더링까지 확인. 30-4절에서 더미로만 하던 검증을
실제 스크립트 기준으로 재확인.

**검증 중 실물 기준으로 새로 발견해 그 자리에서 고친 버그 2건**
(둘 다 더미 프로세스 테스트에서는 못 잡았던 것 — 실물 검증의 목적이
정확히 이거였음):
1. **`_send_graceful_signal()`이 `SystemError`를 못 잡음**: 서로 다른
   프로세스에서 각각 `start_job()`/`stop_job()`을 호출하는 방식으로
   테스트하다가(대시보드의 실제 사용 패턴은 아님 — Streamlit은 서버
   하나가 계속 떠 있는 단일 프로세스라 항상 같은 프로세스 안에서
   호출됨) `os.kill(pid, CTRL_BREAK_EVENT)`가 `OSError`를 감싼
   `SystemError`를 던지는 걸 발견. 격리 재현 결과 이 경우에도 신호
   자체는 실제로 전달돼 대상 프로세스가 정상 종료되는 것까지 확인됨 —
   즉 Python의 `os.kill()` 래퍼가 성공한 호출에 대해 스푸리어스하게
   예외를 던지는 경우가 있는 것으로 추정(WinError 87). `except
   SystemError:`도 추가해서 이 경우 `False`로 포기하지 않고 정상
   폴링 단계로 넘어가게 함 — 정말 신호가 안 먹혔으면 어차피 timeout
   후 강제 종료로 escalate되므로 안전. 실제 대시보드 사용 패턴(단일
   프로세스, 위 ②) 재현 테스트에서는 애초 이 예외 자체가 안 남(1.6초
   만에 정상 완료) — 그래도 방어적으로 남겨둠.
2. **자식 프로세스 stdout이 UTF-8이 아니었음**: `process_manager.start_job()`이
   `env`를 안 넘겨서, 자식(`run_teleop_real.py`)이 한국어 Windows의
   콘솔 코드페이지(cp949)로 `print()`를 인코딩해버림 — 우리는 로그
   파일을 `encoding="utf-8"`로 열어 쓰고 읽으므로 한글 로그가 전부
   깨짐(raw bytes를 직접 열어 `utf-8` 디코딩 실패 → `cp949`로는 성공
   확인, 명확한 원인 규명). ASCII만 쓰는 crash 패턴 매칭(`Traceback`,
   `SerialException` 등)은 우연히 영향이 없어서 30-4절 검증 때는
   못 잡혔음. `subprocess.Popen`에 `PYTHONIOENCODING=utf-8`/
   `PYTHONUTF8=1`을 넣은 `env`를 명시적으로 전달해 수정 — 대시보드의
   실시간 로그 뷰어/크래시 스니펫이 이제 한글도 깨지지 않고 정상
   표시됨(수정 후 재실행으로 확인).

## Stage 2 최종 종결 (2026-08-14)
데이터 수집 + QA 웹 트리거(29절 설계 → 30절 리뷰 반영 → 30-5절 실물
하드웨어 검증)를 여기서 최종 마무리한다.

- **체크리스트 4항목 전부 실물 하드웨어(COM7) 기준 PASS** (30-5절)
- `dashboard/lib/process_manager.py`: 원자적 lock 생성, graceful-then-force
  종료(SystemError 방어 포함), crash 감지 휴리스틱, 자식 프로세스
  UTF-8 강제까지 포함해 실물 검증까지 마치고 안정화됨 — ③④(학습/추론)
  웹 트리거 구현 시 그대로 재사용 예정(시그니처/lock 구조 변경 불필요)
- `pages/1_데이터_수집.py` / `pages/2_QA_검증.py`: 실제 구현 + 더미
  검증 + 실물 하드웨어 검증까지 전부 완료
- 남겨둔 known issue: 30-3절 로그 중복 캡처(`process_manager` 캡처
  vs `run_teleop_real.py` 자체 tee) — 기능은 안 막으므로 ③④ 구현
  시점에 로깅 방식을 통일하며 같이 정리하기로 함(변경 없음, 계속 보류)
- ③(학습/추론) 웹 트리거는 29-6절에서 정리한 대로 이번 범위 밖 — 다음
  세션에서 `state/qa_snapshots/`를 읽어 학습 시작 버튼에 자동 반영하는
  것부터 시작하면 됨

## 31. Stage 3(학습 웹 트리거) 설계 전 사전 조사 — DONE (2026-08-14)

Stage 3 설계에 들어가기 전에, lerobot 0.4.4(`miniconda3\envs\mujoco_env\Lib\site-packages\lerobot\`)의
실제 소스를 직접 읽어 resume/SIGINT/체크포인트 구조/W&B 4가지를 확인했다
(추측 없이 코드 인용 기준).

### 31-1. resume 지원 여부 — 지원함
`lerobot/configs/train.py`의 `TrainPipelineConfig`에 `resume: bool = False`,
`checkpoint_path: Path | None = field(init=False, default=None)` 필드가
있음. `validate()`(같은 파일 81-107행)에서:
```python
elif self.resume:
    config_path = parser.parse_arg("config_path")
    if not config_path:
        raise ValueError(...)
    policy_dir = Path(config_path).parent
    if self.policy is not None:
        self.policy.pretrained_path = policy_dir
    self.checkpoint_path = policy_dir.parent
```
그리고 `lerobot_train.py` 318-319행:
```python
if cfg.resume:
    step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
```

**정확한 실행 커맨드**: `lerobot-train --config_path=<checkpoint_dir>/pretrained_model/train_config.json --resume=true`
(`--config_path`는 draccus의 일반 필드가 아니라 `lerobot/configs/parser.py`의
`wrap()`이 가로채는 특수 인자 — 195-238행에서 `config_path_cli`를 먼저
파싱하고, `TrainPipelineConfig.from_pretrained(config_path_cli, cli_args=...)`로
그 JSON 파일 전체(=`dataset`/`policy`/`output_dir`/`batch_size`/`steps`
등 그때 쓰던 설정 전부)를 불러온 뒤 나머지 CLI 인자를 override로 적용).

- `output_dir`은 저장된 config에 이미 들어있어 재지정할 필요 없음 —
  `validate()`의 `if not self.resume and ... self.output_dir.is_dir(): raise
  FileExistsError(...)` 체크가 `resume=True`일 때만 우회되므로, 이어서
  **같은 output_dir**에 계속 저장됨(새 디렉토리 안 생김)
  (`lerobot/configs/train.py` 119-127행)
- `--config_path`가 가리켜야 하는 파일은 특정 step 체크포인트의
  `pretrained_model/train_config.json` — 어느 step에서 재개할지는
  `training_state/training_step.json`에 저장된 값을 그대로 읽어서
  결정됨(사용자가 step을 별도로 지정하는 옵션 없음, 아래 31-3절 구조 참고)
- `checkpoints/last` 심볼릭 링크(아래 31-3절)를 가리키면 "가장 최근
  체크포인트에서 재개"가 됨

### 31-2. SIGINT(Ctrl+C) 처리 — 지원 안 함, 즉시 죽음
`lerobot/scripts/lerobot_train.py` 전체(552줄)를 처음부터 끝까지
읽고 `signal`/`KeyboardInterrupt`/`SIGINT`/`SIGTERM`/`atexit` 문자열을
grep했으나 **단 한 곳도 없음**(직접 확인, 실측). 메인 학습 루프
(407-520행)도 `try/except`나 `finally`로 감싸여 있지 않음 — 체크포인트
저장은 오직 407행 `for` 루프 안의 `is_saving_step`(`step % cfg.save_freq
== 0 or step == cfg.steps`) 조건에서만 발생(453-471행).

**결론**: Ctrl+C(SIGINT)나 강제 종료를 받으면 Python 기본 동작대로
`KeyboardInterrupt`가 그 자리에서 즉시 전파되어 프로세스가 죽고, 마지막
`save_freq` 경계 이후 진행분은 전부 유실됨. `save_freq`가 크면(현재
`train_main_run_config.yaml` 기준 1500 steps) 최대 1499 steps 분량이
날아갈 수 있음. 데이터 수집(30-2절)과 달리, 이건 우리 스크립트가 아니라
lerobot 패키지 자체(`site-packages` 내부)라 **3절 원칙("base 클래스/라이브러리
소스코드는 절대 수정하지 않는다")상 신호 핸들러를 심어 넣는 방식의 수정이
불가능**함 — run_teleop_real.py 때처럼 "있던 정리 로직을 신호에 연결"하는
방식 자체를 못 씀(애초에 정리 로직이 없음).

### 31-3. 체크포인트 디렉토리 구조
`lerobot/utils/train_utils.py`의 `save_checkpoint()` 독스트링(75-90행)과
`get_step_checkpoint_dir()`(42-45행), `lerobot/utils/constants.py`(46-51행)
기준:
```
<output_dir>/checkpoints/<step_identifier>/
├── pretrained_model/
│   ├── config.json / model.safetensors / train_config.json
│   ├── policy_preprocessor*.json, policy_postprocessor*.json (있으면)
└── training_state/
    ├── optimizer_state.safetensors, scheduler_state.json
    ├── rng_state.safetensors, training_step.json
<output_dir>/checkpoints/last  (가장 최근 체크포인트를 가리키는 심볼릭 링크)
```
`step_identifier = f"{step:0{max(6, len(str(total_steps)))}d}"` — 실제
프로젝트의 `checkpoints/010000/`(6자리 zero-pad)과 정확히 일치함(부록/25절
결과와 대조 확인). `last` 심볼릭 링크는 `update_last_checkpoint()`(같은
파일 57-62행)가 매 저장마다 갱신 — 25절에서 겪은 Windows 심볼릭 링크
권한 문제가 여기서도 그대로 적용됨(개발자 모드 필요, 이미 해결된 이슈).

### 31-4. W&B 연동 — 이미 설정돼 있음
`lerobot/configs/default.py`의 `WandBConfig`(`enable`/`project`/`entity`/
`notes`/`run_id`/`mode` 필드, 41-49행)를 확인. 그리고 현재
`config/train_main_run_config.yaml`을 다시 읽어보니 **이미 wandb 섹션이
설정돼 있었음**(57-65행, 27절 작업 때 이미 넣어둔 것):
```yaml
wandb:
  enable: true
  project: so101-pickplace-act
  mode: offline   # WANDB_API_KEY 미설정이라 online 시 인증 대기로 hang 위험 -> offline 강제
```
별도 CLI 플래그나 신규 설정 불필요 — Stage 3 웹 트리거가 이 YAML을
그대로 넘기기만 하면 W&B(오프라인) 로깅이 자동으로 켜짐. 온라인 업로드가
필요해지면 `wandb sync <output_dir 아래 run 경로>`를 나중에 별도 실행하면
됨(27절에 이미 기록됨, 변경 없음).

### 31-5. Stage 3 설계에 대한 판단
**resume 기능은 구현 가치가 있음 — 다만 "무손실 graceful stop"은 아니라는
전제를 명확히 UI에 표시해야 함.**

- resume 자체(체크포인트에서 이어 학습)는 lerobot이 정식 지원하는 기능이고
  (31-1절), 우리가 만들 웹 트리거는 `--config_path=<선택한 체크포인트>/pretrained_model/train_config.json
  --resume=true`를 조합해서 실행하기만 하면 됨 — 추가 구현 부담이 크지 않음
- 다만 31-2절 때문에 "웹에서 학습 중지 버튼 → 그 순간까지 저장 → 나중에
  이어서" 같은 **무손실 graceful stop은 불가능**함(우리가 못 고치는 lerobot
  내부 문제). 실제로 가능한 건 두 가지뿐:
  1. **의도된 재개**: 예정된 `save_freq` 경계까지 정상적으로 도달한 뒤
     자연 종료되거나, 사용자가 다음 세션에서 이어 돌리고 싶을 때 마지막
     체크포인트부터 재개(손실 없음 — 이건 정상적인 "이어하기" 시나리오)
  2. **사고 복구**: 크래시/정전/실수로 강제종료된 경우, 마지막 저장된
     체크포인트로 되돌아가 재개(마지막 `save_freq` 경계 이후 진행분은
     확정적으로 유실)
- 웹 UI에는 이 구분을 그대로 반영해서, "강제 종료" 버튼을 누를 때
  데이터 수집 페이지(30절)처럼 "정상 종료를 먼저 시도"하는 문구가 아니라
  **"학습은 중간 저장 지점 없이 즉시 종료되며, 마지막 체크포인트
  (최대 {save_freq} step) 이후 진행은 유실됩니다"**처럼 다르게 경고해야
  함 — 데이터 수집과 같은 패턴(graceful-then-force)을 그대로 재사용하면
  사용자에게 잘못된 기대(정상 종료될 거라는)를 주게 됨
- `state/qa_snapshots/`(29-3절)에서 train/val episodes를 자동으로
  채우는 것과 동일한 패턴으로, 체크포인트 목록(`checkpoints/*/`)을
  스캔해 "이어서 학습" 드롭다운에 자동으로 채우는 구조로 설계하면 됨
  (이미 `dashboard/lib/data_sources.py`의 `get_checkpoint_list_df()`가
  이 스캔 로직을 갖고 있어 재사용 가능 — 신규 구현 아님)

## 32. Stage 3(학습 웹 트리거) 구현 — 더미 검증 + 실물 GPU 검증 전부 PASS (2026-08-14)

31절 조사 결과를 바탕으로 학습 웹 트리거를 구현했다. 실제 `lerobot-train`
실행(GPU 점유, 긴 시간)은 사용자 승인 전까지 보류 — 이번 절은 그 전
단계인 더미 프로세스 검증까지의 기록.

### 32-1. 신규/수정 파일
- **`dashboard/lib/training_run.py`(신규)**: `state/training_runs/<run_id>.json`에
  학습 세션 기록(run_id, output_dir, dataset_snapshot_path, cmd, status,
  last_checkpoint_step, started_at, ended_at). `create_run()`/`get_run()`/
  `list_runs()`/`list_interrupted_runs()`/`update_last_checkpoint_step()`/
  `finalize_run()`/`mark_running()`/`delete_run()`. "이어서 학습"은 새
  레코드를 만들지 않고 기존 interrupted 레코드를 `mark_running()`으로
  되돌려 재사용(같은 output_dir로 이어지는 같은 run이라는 게 이유 —
  31-1절: resume은 저장된 config의 output_dir을 그대로 씀).
  `finalize_run()`은 로그에 `lerobot_train.py:529`의 `"End of training"`
  마커가 있으면 completed, 없으면(크래시든 사용자의 강제 종료든 구분 없이)
  interrupted로 기록 — 정교한 파싱 대신 Stage 2 `looks_like_crash()`와
  같은 단순 문자열 매칭 패턴.
- **`dashboard/lib/process_manager.py`(수정)**: `stop_job()`에 `graceful: bool
  = True` 파라미터 추가. `graceful=False`면 신호 시도 자체를 건너뛰고
  곧바로 강제 종료 — `lerobot-train`은 SIGINT/SIGBREAK 핸들러가 전혀 없는
  서드파티 코드라(31-2절) 신호를 보내봐야 의미 없이 대기만 하다 결국
  강제 종료로 끝나기 때문. 데이터 수집(`pages/1_데이터_수집.py`)은
  `graceful=True`를 명시적으로 넘기도록 호출부를 수정(디폴트도 True라
  동작은 그대로, 명시성만 높임).
- **`pages/3_ACT_학습.py`(수정, 신규 파일 아님)**: `pages/3_학습.py`라는
  새 파일 대신 기존 파일을 확장 — 새 파일을 만들면 `STAGE_PAGES`에 "3."로
  시작하는 페이지가 두 개가 돼 사이드바가 깨짐(Stage 2 때도 `1_데이터_수집.py`/
  `2_QA_검증.py`를 새로 안 만들고 기존 스텁을 확장한 것과 같은 패턴).
  기존 읽기 전용 콘텐츠(체크포인트별 loss 차트, 저장된 체크포인트 표)는
  그대로 두고 하단에 "학습 세션 제어" 섹션을 추가:
  - `state/qa_snapshots/`의 최신 스냅샷을 자동으로 읽어 `train_episodes`를
    `--dataset.episodes`에 미리 채움(29-3절에서 예고한 것)
  - "🚀 새 학습 시작" / "▶️ 이어서 학습" 두 탭으로 분리
  - 새 학습: `--config_path=config/train_main_run_config.yaml`을 베이스로,
    `--output_dir=train_runs/<run_id>`(매번 새 디렉터리 -- YAML에 박힌
    `output_dir`을 그대로 재사용하면 두 번째 실행부터
    `FileExistsError`가 남, 31-1절 소스 확인), `--dataset.root`(YAML에
    박힌 절대경로가 이 컴퓨터 것이 아니라는 걸 28/31절에서 이미 알고
    있었으므로 `ds.DATASET_ROOT`로 명시 override)를 CLI 오버라이드로 붙임
  - 이어서 학습: `state/training_runs/`에서 status=interrupted 목록을
    보여주고, 선택하면 `_resolve_latest_checkpoint_config()`가
    `output_dir/checkpoints/last`(심볼릭 링크) 우선, 없으면 숫자
    디렉터리 중 최대 step으로 폴백해서 `--config_path=.../train_config.json
    --resume=true`를 자동 구성
  - `lerobot-train` 실행은 `lerobot-train` 콘솔 스크립트(PATH 의존)
    대신 `[sys.executable, "-m", "lerobot.scripts.lerobot_train", ...]`로
    호출 -- 다른 모든 스크립트와 동일하게 PATH에 의존하지 않는 절대
    인터프리터 경로 방식으로 통일(팀원마다 conda 활성화 상태가 다를 수
    있음을 감안)
  - 강제 종료 경고 문구를 데이터 수집과 다르게: "⚠️ 학습은 즉시
    종료되며 마지막 체크포인트 이후 진행은 유실됩니다" + 실제 마지막
    체크포인트 step을 동적으로 표시
  - 로그 뷰어는 Stage 2와 동일한 `streamlit-autorefresh` 패턴(2초 간격)
    재사용. tqdm 출력에서 `(\d+)/(\d+)\s*\[` 정규식으로 진행률 추출 —
    실패하면 `st.progress()` 없이 raw 로그만 표시(fallback, 정교한
    파싱 안 함)
  - W&B는 이미 `train_main_run_config.yaml`에 켜져 있으므로(31-4절)
    별도 CLI 플래그 없이 "로그는 로컬 `wandb/` 폴더에 저장됩니다"
    안내만 추가

### 32-2. 검증 (더미 프로세스만 — 실제 `lerobot-train`은 미실행)
**① graceful 파라미터 회귀 없음**: `graceful=False`(즉시 강제종료, 정리
로직 미실행 확인) / `graceful=True`(기존과 동일하게 신호→정리 로직 실행)
/ 파라미터 생략 시 디폴트(=True와 동일, 하위호환) 세 시나리오 전부
실제 스탠드인 프로세스(30-2절에서 쓴 것과 동일한 신호 처리 구조)로 확인.
페이지1(데이터 수집)도 `AppTest`로 재렌더해 예외 없음 확인(회귀 없음).

**② 페이지3 베이스라인**: job 없음/스냅샷 없음 상태에서 예외 없이 렌더,
탭 2개("🚀 새 학습 시작", "▶️ 이어서 학습") 정상 표시.

**③ QA 스냅샷 자동 반영**: `qa_snapshot.save_snapshot()`으로 스냅샷을
만들어두면 페이지3이 "최신 QA 스냅샷 자동 반영" 메시지와 함께
train/val episode 개수를 정확히 보여줌을 확인.

**④ 실행 중 상태(더미 프로세스, tqdm 형식 로그 흉내)**: 진행률
"1500/10000"이 프로그레스바에 정확히 반영, `run_id`/`PID`/경과시간
메트릭 정상, 강제종료 경고 문구에 실제 마지막 체크포인트 step이 동적
삽입됨, `training_run` 기록의 `last_checkpoint_step`이 로그에서 자동
추출돼 저장됨을 확인.

**⑤ 종료 후 3가지 분기**: 크래시(Traceback 남김) → `❌` 배너 +
interrupted 기록, 사용자 강제종료 느낌(크래시 패턴 없이 그냥 죽음) →
`⏸️` 배너("마지막 체크포인트: step N") + interrupted 기록 + "이어서
학습" 탭 후보 목록에 정확히 나타남, 정상 완료(`"End of training"` 로그
포함) → `✅` 배너 + completed 기록. 세 경우 모두 실제 재현해서 확인.

**검증 중 발견해 그 자리에서 고친 버그 1건**: 처음 테스트에서 "학습
중 페이지를 한 번도 안 보다가 끝난 뒤에 처음 열람" 시나리오를
재현했더니 `last_checkpoint_step`이 계속 `None`으로 나옴 — 원인은
`update_last_checkpoint_step()`이 오직 페이지가 "실행 중" 상태로
렌더링되는 동안(자동갱신 루프 안)에만 호출돼서, 실행 중에 페이지를 한
번도 안 열었으면 종료 시점에 이미 다 갖고 있는 로그에서도 step을
못 뽑고 있었음. finalize 블록에서 `tr.update_last_checkpoint_step()`을
한 번 더 호출하도록 수정해 해결(같은 로그 데이터를 종료 시점에도
재사용하는 것뿐이라 추가 비용 없음).

**⑥ `_resolve_latest_checkpoint_config()` 단위 테스트**: 버튼을 눌러야만
타는 경로라(누르면 실제 `lerobot-train`이 실행돼버림) 별도 실행 없이,
페이지 소스에서 이 함수 정의만 그대로 추출해(재구현 아님) 임시
디렉터리로 4가지 케이스 확인 — checkpoints/ 자체가 없음(None),
`last` 심볼릭 링크 정상(해당 파일 반환, 실제로 심볼릭 링크 생성 자체가
이 환경에서 여전히 되는 것도 재확인), `last` 없이 숫자 디렉터리만
있을 때 최대 step(010000)을 정확히 선택하는 폴백, 숫자 디렉터리는
있지만 `train_config.json`이 없는 손상 케이스(None). 4개 전부 통과.

### 32-3. 실물 GPU 스모크 테스트 — PASS (2026-08-14)
사용자 승인 후 실제 `lerobot-train`으로 "새 학습 시작" → 강제 종료 →
"이어서 학습" end-to-end 검증 시작. 사전 점검: `nvidia-smi` GPU
0MiB/8151MiB 사용(완전 유휴, 좀비 프로세스 없음 확인), 디스크 여유
169GB(체크포인트 1개당 25절 기준 ~591MB이므로 여유 충분).

**스모크 테스트 파라미터(임시, `train_main_run_config.yaml` 파일은
전혀 수정하지 않음 — `--config_path=config/train_main_run_config.yaml`에
`--steps=100 --save_freq=20`을 CLI 오버라이드로만 얹어서 실행, draccus가
파일 위에 override 적용. 파일에 되돌릴 게 없음)**:
- `--steps=100`(YAML 기본 10000 대신) — 파이프라인이 도는지만 보는 게
  목적이라 GPU 점유를 몇 분 안으로 제한
- `--save_freq=20`(YAML 기본 1500 대신) — step 20/40/60/80/100에 체크포인트
  5개, "이어서 학습" 재개 테스트에 필요한 체크포인트 2개 이상 확보
- `--output_dir=train_runs/<run_id>`, `--job_name=<run_id>`,
  `--dataset.root=<ds.DATASET_ROOT>`는 32-1절에 설명한 대로 페이지의
  "새 학습 시작"이 실제로 구성하는 값과 동일(테스트도 같은 코드 경로 사용)

**체크포인트 완료 판정 방식(1차 시도, 실패)**: 로그의 `"Checkpoint
policy after step N"`은 `save_checkpoint()` 호출 *직전*에 찍히는
문구라(소스 확인, `lerobot_train.py:455-457`) 그 시점에 죽이면 저장
도중일 위험이 있음 — 그래서 `checkpoints/last` 심볼릭 링크가 해당
step을 가리키도록 갱신되는지 + `training_state/scheduler_state.json`
파일 크기가 1초 간격으로 두 번 재확인해도 안 변하는지로 "완전히 저장
끝남"을 판정하려 했다. **이 판정 조건이 영원히 참이 되지 않아 1차
시도는 step 40에서 못 멈추고 100 step 전체가 그냥 완주해버림** —
원인을 파헤쳐보니 `scheduler_state.json`은 `save_training_state()`가
`scheduler is not None`일 때만 쓰는데, ACT의 `ACTConfig.
get_scheduler_preset()`이 소스에서 명시적으로 `None`을 반환함
(`configuration_act.py:160`, 확인 완료) — 즉 ACT는 애초에 LR
스케줄러를 안 써서 이 파일이 **영원히 생성되지 않음**. 내가 "항상
생기는 파일"이라고 잘못 가정한 게 원인.

의도치 않게 100 step을 완주해버린 이 1차 실행도 버리지 않고 그대로
활용: `finalize_run()`을 이 실제 로그에 돌려서 completed 판정과
`last_checkpoint_step=100` 추출이 실제 lerobot-train 출력 기준으로도
정확함을 확인(더미 텍스트가 아닌 진짜 로그로 재검증한 셈). loss는
step:50에서 13.736 → step:100에서 4.167로 정상적으로 감소(실제 학습이
일어남을 확인).

**체크포인트 완료 판정 방식(2차 시도, 성공)**: 파일 안정성 폴링 대신
로그 마일스톤을 순차로 기다리는 방식으로 단순화 — ①`"Checkpoint
policy after step 40"` 로그 확인 → ②`"41/100"` 이상 진행 로그 확인(이게
찍힌다는 것 자체가 저장이 전부 끝나고 다음 스텝으로 넘어갔다는 증거,
`save_checkpoint()`가 동기적으로 루프를 블로킹하므로) → ③`"step:50"`
loss 로그까지 확인(확실히 학습 도중이고, baseline loss 값도 같이 확보).
`checkpoints/last -> 000040` 정확히 확인, `training_state/
optimizer_state.safetensors` 존재 + 412,817,652 bytes 확인 후
`stop_job(graceful=False)`로 강제종료(0.49초 만에 완료).

**강제종료 직후 검증**:
- `nvidia-smi`: 0 MiB / 8151 MiB — GPU 완전 해제 확인
- `pm.looks_like_crash(log)` → `False`(강제종료는 크래시 패턴을 안
  남기므로 정상)
- `tr.finalize_run()` → `interrupted`, `last_checkpoint_step=40` —
  실제 강제종료 로그로 재검증(더미 테스트와 동일 결과)
- 체크포인트 40 무결성: `model.safetensors`(234개 텐서)/
  `optimizer_state.safetensors`(459개 텐서)/`rng_state.safetensors`(8개
  텐서) 전부 `safetensors`로 헤더 파싱 성공, 모든 JSON 파일 파싱
  성공, `training_step.json` = `{"step": 40}` 정확 — 강제종료로 인한
  파일 손상 없음 확인
- `checkpoints/` 디렉터리에 `000060`이 없음을 확인 — 의도한 대로
  "저장 사이"에 정확히 죽었음

**"이어서 학습" 재개 검증**: `checkpoints/last/pretrained_model/
train_config.json`(=`_resolve_latest_checkpoint_config()`와 동일 경로)로
`--resume=true` 실행:
- **tqdm 총 개수가 60으로 시작**(=100-40) — step 0부터 다시 세는 게
  아니라 정확히 step 40에서 이어받았다는 구조적 증거
- **최종 loss(step 100) = 4.176** vs 1차 완주 실행의 최종 loss
  `4.167` — 거의 완벽히 일치(차이 0.009). optimizer/model state가
  진짜로 복원 안 됐다면 중단 지점부터 60 step만으로 이 정도까지
  다시 수렴하는 건 사실상 불가능하므로, 이 일치가 optimizer state
  복원이 실제로 작동한다는 가장 강력한 증거
- step:50 loss는 재개 전(13.737) vs 재개 후 재계산(5.275)이 다르게
  나옴 — 이건 실패 신호가 아니라고 판단: `num_workers=8` 멀티프로세스
  dataloader는 rng state 복원과 별개로 워커별 배치 순서까지 완벽히
  재현하지 않을 수 있어(각 워커의 시드 초기화가 메인 프로세스 rng
  복원과 독립적일 가능성), 같은 step 번호라도 다른 배치를 볼 수 있음.
  batch_size=16의 소규모 배치 단위 loss 분산을 감안하면 값 차이 자체는
  자연스러운 범위 — 위 최종 loss 수렴 일치가 훨씬 결정적인 증거이므로
  이 단일 지점 차이로 결론을 뒤집지 않음
- 체크포인트 디렉터리에 `000060/000080/000100`이 같은 `output_dir`
  안에 이어서 정상 생성됨(새 디렉터리로 안 갈라짐), `last -> 000100`
- `wandb/` 오프라인 로그도 `train_runs/<run_id>/wandb/`에 재개 전/후
  두 offline-run이 정상 기록됨(같은 W&B run ID로 이어짐) — 31-4절
  설계대로 별도 설정 없이 W&B 오프라인 로깅 작동 확인
- 재개 완료 후 `nvidia-smi` 다시 0 MiB 확인(GPU 정상 해제)

**부수 발견**: 재개 실행이 자연 종료된 뒤 아무도 `get_job_status()`를
다시 호출하지 않아 `state/locks/training.json`이 스테일 상태로 잠깐
남아있었음 — 실제 죽은 프로세스를 대상으로 `get_job_status()`를 한 번
호출하니 `_all_locks()`의 stale-lock 정리 로직이 정확히 작동해 자동
삭제됨을 확인(30-1절에서 구현한 로직의 실전 재확인).

**정리**: 스모크 테스트용 `train_runs/`(체크포인트 포함 런당 ~2.9GB,
optimizer state까지 저장하는 실제 학습이라 더미보다 훨씬 큼 — 사용자가
미리 지적한 대로) 2개와 `state/training_runs/`의 해당 레코드 2개를
삭제. `logs/jobs/training_*.log`는 다른 단계 검증 로그와 같은 관행으로
남겨둠. `train_main_run_config.yaml`은 애초에 건드리지 않았으므로
되돌릴 것 없음.

**결론**: Stage 3(학습 웹 트리거)의 핵심 경로 — 새 학습 시작(CLI
오버라이드 구성) → 체크포인트 저장 → 강제 종료(신호 없이 즉시,
파일 손상 없음) → interrupted 기록 → 이어서 학습(정확한 step에서
재개, optimizer state 복원) → 완료 기록 — 전부 실물 GPU로 end-to-end
검증 완료.

## 33. Stage 4(추론 웹 트리거) 설계·구현 — 더미 검증 + 실물 GPU 검증 전부 PASS (2026-08-14)

### 33-1. 사전 조사: 기존 추론 파이프라인의 형태와 소요 시간
`pipeline/4_inference/run_inference_mujoco.py`(543줄, 단일 스크립트,
모듈화 안 됨 — `run_teleop_real.py`와 같은 이유로 top-level 부작용 있어
대시보드에서 import 안 하고 subprocess로만 다룸)를 직접 읽고 실행해서
확인. **두 가지 완전히 다른 실행 모드가 있다는 게 이번 조사의 핵심
발견**:

- **`--headless --duration N`**: 뷰어 없이 N초 시뮬레이션만 최대한 빨리
  돌리고 종료. 실측(`--duration 5`): **벽시계 기준 11.47초**(체크포인트
  로딩+CUDA 초기화가 대부분, exit code 0, NaN/Inf 없음, GPU 종료 후
  0MiB로 정상 해제 확인). 시각 피드백/성공 판정 없이 순수 헬스체크만.
  **동기 처리 가능한 짧은 작업** — 학습처럼 백그라운드+진행률이 필요
  없음.
- **기본(뷰어) 모드**: `mujoco.viewer.launch_passive()`로 GUI 창이 뜨고
  `while viewer.is_running():`으로 **사용자가 창을 닫을 때까지 무기한
  루프**. R 키로 episode 리셋은 되지만 프로세스 자체를 끝내는
  episode 경계/성공 판정 로직이 스크립트에 아예 없음 — "1회 롤아웃"
  개념 자체가 없음. 구조적으로 ①(데이터 수집)의 무기한 MuJoCo 뷰어
  세션과 동일.

이 두 모드의 근본적인 차이(동기 가능 vs 무기한 세션) 때문에, Stage 4는
Stage 3(진행률 바 하나)처럼 단일 패턴으로 못 묶고 **트랙 A/B로 분리**하는
쪽으로 설계 확정.

### 33-2. 구현
`pages/4_추론.py`(기존 읽기 전용 페이지 확장, 새 파일 안 만듦 — Stage
2/3와 동일한 이유), `dashboard/lib/training_run.py`에 `list_completed_runs()`
추가(기존 `list_interrupted_runs()`와 동일 패턴).

- **체크포인트 선택**: `training_run.list_completed_runs()`(신규 조회
  로직 작성 안 하고 기존 함수 재사용)로 ③에서 완료된 학습 run 목록을
  드롭다운에 채움. `_resolve_checkpoint_dir()`는 `pages/3_ACT_학습.py`의
  `_resolve_latest_checkpoint_config()`와 같은 `checkpoints/last` 우선 +
  숫자 디렉터리 최대값 폴백 패턴이지만, 반환값이 `train_config.json`이
  아니라 `run_inference_mujoco.py --checkpoint`가 기대하는
  `pretrained_model` 디렉터리 자체라는 점이 다름. 완료된 run이 하나도
  없으면 스크립트 자체의 기본값(`checkpoints/010000/pretrained_model`)으로
  폴백.
- **트랙 A(⚡ 빠른 헬스체크)**: `subprocess.run(...)` **블로킹** 호출(`st.spinner`로
  대기 표시, `timeout=max(120, duration+60)`). `_judge_healthcheck_output()`
  함수로 판정 — 스크립트가 실제로 찍는 `"NaN 포함: {bool}, Inf 포함:
  {bool}"` 문구를 문자열 매칭(정교한 파싱 아님, 33-1절에서 실측한 실제
  출력 형식 그대로). `process_manager`의 lock/start_job을 안 씀(동기
  호출이라 필요 없음) — 대신 `pm.check_conflict("inference")`로 다른
  stage(학습 등)와의 GPU 충돌만 시작 전에 확인.
- **트랙 B(🖥️ 시뮬레이션 관찰)**: `pages/1_데이터_수집.py`(①)에서 쓴
  `process_manager.py` 패턴을 그대로 재사용 — `resources=["gpu",
  "mujoco_viewer"]`는 29-2절에서 이미 예약해둔 태그라 신규 설정 불필요.
  실시간 로그 뷰어(`streamlit-autorefresh`, 2초 간격), crash 감지
  (`looks_like_crash()`, ①과 동일 패턴), 강제 종료 버튼. `run_inference_mujoco.py`엔
  (`run_teleop_real.py`와 달리) 신호 핸들러가 없으므로 `stop_job(STAGE,
  graceful=False)` 사용(31-2절의 학습 stage와 같은 이유) — 다만 추론은
  저장하는 데이터가 없어 데이터 수집처럼 "유실 위험" 경고는 불필요,
  문구도 그에 맞게 순화.
- **성공/실패 정량 판정은 스코프 밖**: 물체를 실제로 집었는지 같은
  task-success 판정 로직은 이번에 만들지 않음. 트랙 A는 NaN/Inf·크래시
  유무만, 트랙 B는 사람이 뷰어로 직접 관찰하는 용도. 필요해지면
  헤드리스 모드에 episode 종료 조건 + 성공 기준(예: 물체-그리퍼 거리,
  목표 위치 도달 여부)을 추가하는 방향(옵션 2)으로 확장 가능 — 이번
  범위에서는 하지 않음.

### 33-3. 검증 (더미 프로세스만 — 실제 GPU 추론은 이번엔 실행 안 함)
- 베이스라인 렌더(완료 run 없음): `checkpoints/010000` 기본 체크포인트로
  정상 폴백, 탭 2개 정상 렌더
- 가짜 완료 run + 가짜 체크포인트 디렉터리(`model.safetensors` 존재만
  충족)로 `_checkpoint_options()`가 드롭다운에 정확히 반영함을 확인
- 트랙 A의 `_judge_healthcheck_output()`을 페이지 소스에서 그대로
  추출(재구현 아님)해 33-1절에서 실측한 **실제** 성공 출력 형식 +
  NaN/Inf/크래시(exit code≠0)/복합 케이스 5가지 전부 검증
- 트랙 B: 더미 프로세스로 실행 중 상태(메트릭/로그/종료 버튼), 크래시
  감지(Traceback 남기고 죽음 → 배너), 학습(training) stage가 GPU를
  잡고 있을 때 트랙 B 시작이 막히는지까지 확인

**검증 중 발견해 그 자리에서 고친 버그 1건**: 트랙 B가 실행 중인 상태에서
AppTest로 페이지를 다시 로드해보니 **트랙 A의 "빠른 헬스체크 실행"
버튼이 안 막혀 있었음** — `pm.check_conflict("inference")`는 자기
자신(inference)과의 충돌은 검사 대상에서 제외하는 구조라(`if
other_stage == stage: continue`), "트랙 B가 이미 이 stage를 쓰고
있는 경우"를 안 걸렀던 것. `job_status["running"]`을 추가로 확인해
트랙 B 실행 중엔 트랙 A 버튼도 비활성화되도록 수정, 재현 테스트로
확인 완료. lock 없이 동기 호출만 하는 트랙 A를 설계하면서 생긴 사각지대라
— 앞으로 비슷하게 "락 없는 동기 트랙 + 락 있는 비동기 트랙"을 같은
stage로 묶을 때 재발 가능성이 있는 패턴으로 기록해둠.

### 33-4. 실물 GPU 검증 — PASS (2026-08-14)
더미 검증 이후 실제 버튼 클릭(`AppTest`가 진짜 `subprocess.run`/
`pm.start_job`을 태우는 방식 — Stage 3 때와 동일하게 페이지 코드 경로
그대로 실행, 재구현 아님)으로 6단계 전부 확인.

**① 사전 상태**: `nvidia-smi` 0MiB/8151MiB(완전 유휴, 컴퓨트 프로세스
없음), `state/training_runs/`가 비어있어(Stage 3 스모크 테스트 산출물을
이미 정리해서) **완료된 run이 하나도 없는 상태** — 계획대로 이게
오히려 "완료 run 없을 때 기본 체크포인트 폴백" 경로를 검증하는 기회가
됨.

**② 트랙 A 정상 케이스**: 폴백 체크포인트(`checkpoints/010000`)로 실제
버튼 클릭 → **11.53초** 소요, `exit code 0`, NaN/Inf 없음, "✅ 통과"
배지 정상 표시. 33-1절 사전 조사에서 스크립트를 직접 실행했을 때의
실측치(11.47초)와 사실상 동일 — 대시보드를 거쳐도 오버헤드 거의 없음.

**③ 트랙 A 손상 케이스**: `_checkpoint_options()`가 `model.safetensors`
존재 여부로 이미 필터링해서, 아예 없는 경로는 애초에 드롭다운에 안 뜨는
걸 먼저 확인(설계상 정상). 그래서 "드롭다운 필터는 통과하지만 실제
로딩은 실패하는" 케이스 — `model.safetensors`는 있는데(가짜 바이트)
`config.json`이 없는 체크포인트를 임시로 만들어 테스트. 실제
`ACTPolicy.from_pretrained()`가 draccus `ParsingError`("Expected a
dict with a 'type' key for ... PreTrainedConfig")로 진짜 실패 →
`exit code 1` → "❌ 실패" 배지 정상 표시(6.5초 소요). 가짜 성공으로
잘못 판정하지 않음을 확인.

**④ 트랙 B 시작**: 실제 버튼 클릭으로 `pm.start_job()` 호출 →
`get_job_status()`로 `running=True`, PID 확인. **`Get-Process -Id
<pid> | Select MainWindowTitle`로 `"MuJoCo : scene"` 확인** — 뷰어
창이 실제로 떠 있다는 직접적 증거(스크린샷 대신 윈도우 타이틀 조회로
확인). `nvidia-smi`도 844MiB/22% util로 실제 추론이 도는 중임을 재확인.
**이번에 고친 버그의 실물 재현**: 트랙 B가 이 상태로 살아있는 동안
페이지를 다시 로드하니 트랙 A의 "빠른 헬스체크 실행" 버튼이 정확히
비활성화되고 "트랙 B(시뮬레이션 관찰) 세션이 이미 실행 중이라..."
경고가 뜸 — 더미로 검증했던 수정이 실물에서도 그대로 작동.

**⑤ 트랙 B 강제종료**: 실제 버튼 클릭 → `stop_job(graceful=False)`가
0.25초 만에 완료, `psutil.pid_exists(pid)`로 프로세스가 실제로
종료됐음을 확인(단순 lock 파일 삭제가 아니라 진짜 프로세스 죽음).
`nvidia-smi` 다시 0MiB로 확인. 종료 후 페이지 재로딩 시 트랙 A 버튼도
정상적으로 다시 활성화됨.

**⑥ 반대 방향 상호배제**: 트랙 B(`resources=["gpu","mujoco_viewer"]`)를
다시 시작한 상태에서 `pages/3_ACT_학습.py`(학습, `resources=["gpu"]`)를
로드하니 "🚀 새 학습 시작" 버튼이 정확히 비활성화되고 경고 사유에
`'inference'가 GPU를 사용 중`이 정확히 표시됨 — 29-2절에서 태그만
예약해두고 실물로는 처음 확인한 조합인데, `gpu` 태그 하나만 겹쳐도
`check_conflict()`가 정확히 잡아냄을 확인(집합 교집합 방식이라 태그가
1개만 겹쳐도 충돌로 잡히는 게 의도한 설계와 일치).

검증 후 GPU 0MiB, `state/locks`/`state/training_runs`/`train_runs`
전부 빈 상태로 정리 완료.

**결론**: Stage 4(추론 웹 트리거)의 핵심 경로 — 체크포인트 선택(완료
run/폴백 둘 다) → 트랙 A 동기 헬스체크(성공/실패 둘 다) → 트랙 B
뷰어 세션(시작/창 렌더/강제종료/GPU 해제) → 두 방향 자원 배제(트랙
A↔B, 트랙 B↔학습) — 전부 실물 GPU + 실제 MuJoCo 창으로 end-to-end
검증 완료. 이걸로 4단계(데이터 수집→QA→학습→추론) 웹 개발 키트 전체가
실물 하드웨어 기준으로 한 번씩은 검증된 상태.

## 부록. Phase 0 확인 결과 (2026-08-05)
- lerobot 버전: 0.4.4 (환경: conda `mujoco_env`, 경로:
  `miniconda3\envs\mujoco_env\Lib\site-packages\lerobot\`)
- `LeRobotDataset(torch.utils.data.Dataset)` — `__init__(repo_id, root=None, episodes=None,
  image_transforms=None, delta_timestamps=None, tolerance_s=1e-4, revision=None,
  force_cache_sync=False, download_videos=True, video_backend=None, batch_encoding_size=1,
  vcodec="libsvtav1", streaming_encoding=False, encoder_queue_maxsize=30, encoder_threads=None)`
- `__len__` → `self.num_frames`, `__getitem__` → dict 반환 (delta_timestamps/video/task 병합)
- `meta` 접근: `dataset.meta.stats`, `dataset.meta.tasks`, `dataset.meta.episodes`
  (top-level 속성이 아닌 `LeRobotDatasetMetadata` 하위 객체) →
  composition 위임(`__getattr__`) 구조와 호환 확인됨
