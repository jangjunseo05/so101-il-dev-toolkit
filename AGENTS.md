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

## 4. Codex 작업 시 필수 지침
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

## 34. 웹 대시보드 구조 요약 + 새 팀원 온보딩 기반 작업 — DONE (2026-08-16)

### 34-1. 대시보드 구조 (SSOT)

`dashboard/app.py`(홈)는 파이프라인 흐름 네비게이션 + 두 개의 요약 섹션만
갖는 진입점이고, `dashboard/pages/1~4_*.py`는 전부 "분석/메타 정보 표시 +
실제 작업을 트리거하는 이벤트 기반 버튼"의 조합이다. 단, 버튼이 작업을
수행하는 **방식**은 단계마다 다르다:

| 단계 | 분석/메타 정보 | 버튼이 하는 일 | 실행 방식 |
|---|---|---|---|
| 홈(app.py) | 환경 상태(34-2절) + 파이프라인 헬스 요약(기존) | (버튼 없음, 읽기 전용) | - |
| ① 데이터 수집 | 로컬 설정 폼 | `run_teleop_real.py` 세션 시작/종료 | 서브프로세스 (`process_manager`, lock: mujoco_viewer/serial_port) |
| ② QA 검증 | 아웃라이어/무결성 분석 결과 | `TeamRobotDataset` 메서드 직접 호출 + QA 스냅샷 저장 | **인프로세스 직접 호출** (서브프로세스 아님, lock 불필요) |
| ③ ACT 학습 | 체크포인트별 loss 차트 | `lerobot-train` 새 학습/이어서 학습/강제종료 | 서브프로세스 (`process_manager`, lock: gpu) |
| ④ 추론 | 체크포인트 선택 | 트랙A 헤드리스 헬스체크, 트랙B MuJoCo 뷰어 세션 | 트랙A 동기 서브프로세스, 트랙B 비동기 서브프로세스 (lock: gpu+mujoco_viewer) |

공통 인프라: `dashboard/lib/process_manager.py`(서브프로세스를 쓰는
①③④가 공유하는 lock/job 관리), `dashboard/lib/local_config.py`(이
컴퓨터 전용 설정, `config/local_settings.json`).

### 34-2. 홈 화면 두 섹션의 구현/검증 상태 (혼동 방지용 명시)

- **"파이프라인 헬스 요약"**: 이번 세션 **이전부터 존재하던 기존 기능**
  (과거 실행 결과의 episode 수/QA 이상치 건수/최종 loss 등을 `ds.get_dataset_info()`
  등으로 읽어 표시). 이번 세션에서 새로 만들지 않았고, 표시되는 수치 자체를
  다시 검증하지도 않았음 — 아래 AppTest 확인 시 크래시 없이 같이 로드되는
  것만 재확인됨.
- **"환경 상태"**: **이번 세션에서 신규 추가**(`local_config.load_local_settings()`로
  `leader_port`/캘리브레이션 파일 존재 여부 확인, `torch.cuda.is_available()`로
  GPU 확인). `streamlit.testing.v1.AppTest`(이 프로젝트가 32-33절에서 이미
  쓰던 것과 같은 검증 패턴)로 두 상태 모두 직접 실행해 확인함:
  - 설정 완료 상태(이 컴퓨터의 실제 `config/local_settings.json` 기준):
    `at.exception` 없음, ✅ 3개(로컬 설정 저장됨/캘리브레이션 파일 있음/GPU
    사용 가능) 정상 렌더링
  - 설정 미완료 상태(`local_settings.json`을 백업 후 임시 제거 →
    `local_config.DEFAULTS` 폴백 유도 → 확인 후 원본 복원): `at.exception`
    없음, ⚠️ "리더암 포트 미설정" + 🧭 "NEW_TEAMMATE_SETUP.md 참고" 안내
    정상 렌더링
  - **주의(정정)**: 처음엔 `Invoke-WebRequest`로 정적 HTTP GET 200 응답 +
    stderr에 traceback 없음을 근거로 "크래시 없이 확인됨"이라 판단했으나,
    이는 부정확한 근거였음 — Streamlit은 정적 HTTP GET만으로는 앱 스크립트를
    실행하지 않고 브라우저가 WebSocket 세션을 열 때 실행되므로, 그 확인은
    스크립트가 실제로 실행됐다는 증거가 못 됨. AppTest 기반 재확인으로
    정정함.

### 34-3. git 전환 + 새 팀원 온보딩 기반 작업 (30-32절 참고, 새로 추가)

29~33절에서 검증된 대시보드를 실제로 다른 팀원의 컴퓨터에서 처음부터
돌릴 수 있게 하는 작업. 상세 근거/조사 과정은 별도 세션 대화 기록 참고,
결과만 요약:

- **git 전환**: `git init`(로컬만, 원격 없음) + `.gitignore`(`data/`,
  `checkpoints/`, `logs/`, `state/`, `train_runs/`, `*.log`,
  `config/local_settings.json`, `config/full_arm_calibration_leader.json`(물리
  리더암 전용 캘리브레이션 — 다른 로봇팔 값을 새 팀원이 그대로 물려받는
  안전 문제로 판단해 제외), `graphify-out/`, `.Codex/` 등 제외) + 첫 커밋
  76개 파일. 작업 중 이전 세션에서 실수로 `C:\Users\USER\Desktop\.git`(바탕화면
  최상위)에 빈 저장소가 잘못 생성돼 있던 것을 발견(커밋 0개, 완전히
  비어있음 확인 후 삭제) — 올바른 위치(`so101_web/`)에 재초기화.
- **환경 재현**: `environment.yml`(프로젝트 루트) — 실제 `mujoco_env`를
  `conda env export`로 조사해서 생성, torch/torchvision(CUDA wheel이라
  conda 불가)은 별도 설치 단계로 분리, 이 세션에서 우연히 mujoco_env에
  깔린 graphify 관련 패키지(`graphifyy`, `uv`, tree-sitter류 26개, /graphify
  스킬 실행 중 설치된 것으로 확인됨)는 로봇 파이프라인과 무관해 제외.
  `docs/reference-pip-freeze-2026-08-16.txt`에 필터링 전 원본 보관.
- **온보딩 문서**: `docs/NEW_TEAMMATE_SETUP.md`(신규) — clone부터 대시보드
  실행까지 선형 안내. `docs/REAL_LEADER_SETUP.md`가 프로젝트 이동 전 옛
  경로(`...mujoco\SO-ARM100\Simulation\SO101`)와 옛 폴더명(`datasets/`)을
  가리키고 있던 것을 확인해 전부 수정(위치 비의존적 안내로 교체).
- **의도적으로 안 건드린 것**: `config/train_main_run_config.yaml`의
  `dataset.root` 절대경로 값 자체는 그대로 둠(대시보드의 "새 학습 시작"이
  항상 `--dataset.root`로 덮어써서 값 자체를 고쳐도 다음 팀원에게 또
  stale해질 뿐 — 대신 주석으로 "이 컴퓨터 전용, 대시보드 경로에서는
  무시됨" 명시).

## 35. 직관 UX 개선: 카드 위계 재정리 + 홈 CTA — DONE (2026-08-17)

### 배경
목표: "처음 쓰는 사람도 웹이 유도하는 대로만 따라가면 4단계 파이프라인을
헤매지 않고 완주할 수 있게" — 기능 추가가 아니라 기존 홈+4단계 페이지의
정보 위계/시각적 위계를 재정리하는 작업. 사용자가 진행 순서를 4단계로
직접 지정: ① 카드 스타일 확정(승인 후 확산) → ② 위계 재정리(버튼→설정→
결과→메타, 차단 시 예외) → ③ "다음 단계로" 링크 → ④ 홈 CTA + 진행 표시.

### 35-1. 공통 컴포넌트: `dashboard/lib/ui_components.py` (신규)
`render_card_header(icon, title, role)` — role은 `action`/`settings`/
`success`/`warning` 4개로 고정, 각 role의 배경/글자색은 이 함수 안에서만
정의(페이지 코드에 색 하드코딩 금지). 기존 `st.container(border=True)`
패턴 위에 첫 줄로 색상 헤더만 추가하는 최소 변경 방식.

**role 적용 규칙(4개 페이지 공통, 실제 적용하며 확정)**:
- **사전조건 경고**(자원 충돌/필수 입력 누락 — "이 행동을 지금 할 수
  있는가"): 별도 카드가 아니라 action 카드 헤더 자체를 조건부로
  action→warning 전환. 사유 텍스트는 헤더 바로 아래 캡션으로 흡수(기존
  `st.warning()` 박스 제거) — 사유가 항상 버튼보다 위에 오는 게 헤더
  위치 자체로 보장됨.
- **사후 사건**(크래시/중단/정상완료 — "이미 벌어진 일"): action 카드와
  분리된 독립 카드, action 카드보다 위에 배치. role은 결과에 따라
  warning(크래시/중단) 또는 success(정상완료)로 전환 — 둘 다 warning
  하나로 묶지 않고 명확히 구분(사용자 확정).
- **결과(results)**: role=success, 페이지 레벨 카드.
- 4개 role만 쓰고 5번째("danger" 등) role은 만들지 않기로 확정 — 사용자가
  "danger role"이라고 표현했던 크래시 카드도 기존 warning role 재사용으로
  해석(대화 중 명시적으로 확인).

**페이지 3(학습)/페이지 4(추론)의 탭 비대칭 처리**: 두 페이지 다 탭
구조(③: 새 학습/이어서 학습, ④: 헬스체크/시뮬레이션 관찰)를 유지하되,
③은 두 탭의 차단 조건이 동일(`check_conflict(STAGE)` 공유)해서 outer
action 카드 헤더를 조건부 전환해도 무방했지만, ④는 트랙 A(헬스체크)가
자원 충돌뿐 아니라 트랙 B 실행 중일 때도 추가로 막히는 비대칭 구조라
outer 헤더를 공유 전환하면 한쪽 탭 상황이 다른 탭에 오도(誤導)됨 —
사용자 확인 후 ④는 outer 헤더를 항상 action 고정, 각 탭이 자기 차단
사유만 캡션으로 표시하도록 설계. ④ 트랙 A의 매번 다른 통과/실패 결과는
독립 카드로 만들지 않고 기존 `st.success`/`st.error` 한 줄로 유지(정적
"헤드리스 검증 결과(이전 세션 기록)" success 카드와 혼동 방지, 사용자
확정).

**버튼이 자기 아래 위젯의 최신 값을 못 읽는 문제**: "행동(버튼) → 설정"
순서로 재배치하면서, 버튼이 코드 실행 순서상 더 아래에 있는 위젯(설정)의
값을 참조해야 하는 문제가 QA 페이지(②)에서 처음 발생 → 위젯 렌더 전에
`st.session_state.setdefault(key, 기본값)`으로 미리 seed하고 버튼 로직은
session_state에서 읽는 패턴으로 해결. ③④는 처음부터 이 패턴으로 구현.
③의 "이어서 학습" selectbox는 매 rerun마다 옵션 목록이 바뀔 수 있어,
session_state에 저장된 이전 선택값이 새 옵션 목록에 없으면(라벨에 포함된
`last_checkpoint_step`이 바뀌는 등) 첫 번째 옵션으로 자동 clamp하는 방어
로직 추가 — AppTest로 신선한 세션/stale 세션 값 두 경우 다 확인.

### 35-2. 페이지별 최종 위계
- **① 데이터 수집**: [크래시 카드] → 행동(녹화 세션 제어) → 설정(로컬
  설정) → 결과(현재 데이터셋 현황, success) → 메타(코드 구조/데이터
  패킷/실행 파라미터, 접힘)
- **② QA 검증**: 행동(스냅샷 저장) → 설정(z_thresh + 라이브 QA 수치 —
  "저장 전에 보고 판단하는" 용도라 결과가 아니라 설정 소속으로 분류,
  사용자 확정) → 결과(스냅샷 저장 완료 요약, 있을 때만 — 없으면 카드
  자체 생략) → 메타(코드 구조/이상치 사례/시각화/스냅샷 전체 이력, 접힘)
- **③ ACT 학습**: [사후 사건 카드: 크래시/중단=warning, 완료=success] →
  행동(탭: 새 학습/이어서 학습, 각 탭 내부 버튼→설정 순) → 결과(loss
  차트+체크포인트 목록, success) → 메타(yaml 설정값/정적 PNG, 접힘)
- **④ 추론**: [크래시 카드, 트랙 B 전용] → 행동(outer 헤더 고정,
  탭: 헬스체크/시뮬레이션 관찰, 각 탭 독립 차단 표시) → 결과(헤드리스
  검증 결과-이전 세션 기록, success) → 메타(코드 구조/실행 파라미터, 접힘)

**Streamlit 제약 재발견**: expander는 다른 expander 안에 중첩 불가 —
②(예전 outlier_report.md)와 ③(정적 PNG)에서 기존에 독립 expander였던
섹션을 메타정보 접힘 안으로 옮기며 둘 다 일반 서브섹션(subheader +
상시 표시)으로 전환.

### 35-3. "다음 단계로" 링크 — 최초 구현 시 누락, 재확인 후 추가
①②③에 `st.page_link()`로 추가(④는 마지막 단계라 해당 없음): ①은
결과 카드에 episode>0일 때, ②는 결과 카드(스냅샷 저장 완료) 안에,
③은 "학습 완료" 사후 사건 카드 안에. **AppTest 함정 발견**: `st.page_link()`를
페이지 스크립트 안에서 호출하면 AppTest(`AppTest.from_file()`)는
`KeyError: 'url_pathname'`으로 크래시함 — AppTest가 단일 페이지 스크립트만
격리 실행해 멀티페이지 앱의 페이지 레지스트리가 없기 때문으로 추정(실제
`streamlit run` 앱에서는 정상 동작, `dashboard-screenshot` 스킬로 브라우저
직접 확인해 검증함 — AppTest만으로는 `page_link` 관련 코드를 검증할 수
없다는 게 이번에 새로 확인된 한계, 다음에도 참고).

### 35-4. `dashboard-screenshot` 프로젝트 스킬 (신규, `.Codex/skills/`)
`/run-skill-generator`는 모델이 직접 호출 못 하는 사용자 전용 명령이라
(호출 시도 시 차단 확인됨), 대신 직접 스킬 파일을 작성. `.tools/screenshot/`
(gitignore 대상, node_modules 포함)에 Playwright를 한 번만 설치해두면
이후 세션에서 재설치 불필요 — 브라우저 바이너리는 `~/AppData/Local/ms-playwright/`
전역 캐시에 남음. `shot.js`는 포트/사이드바 링크 텍스트/출력 경로를
인자로 받는 범용 스크립트, Streamlit 스피너(`[data-testid="stSpinner"]`)가
사라질 때까지 최대 30초 대기 + 추가 버퍼(6초)로 QA/학습/추론 페이지의
느린 인프로세스 계산(TeamRobotDataset 스캔)과 차트 렌더링을 둘 다
커버하도록 이번 작업 중 점진적으로 개선함. `.Codex/`를 통째로
gitignore하면 이 스킬 파일 자체가 팀원에게 안 보이는 문제를 발견해
`.Codex/scheduled_tasks.lock`만 정확히 제외하도록 34절에서 이미 수정된
상태(재확인).

### 35-5. 홈 페이지 동적 CTA + 진행 표시: `dashboard/lib/next_action.py` (신규)
**`get_next_action()`** — 6개 규칙, 첫 매치 채택(사용자 확정 우선순위):
1. `state/locks/`에 살아있는 lock → "진행 중: {단계} 세션 (보러가기)"
2. `training_runs`에 status=interrupted 존재 → "이어서 학습할 수 있는
   run이 있습니다 → ③ 학습"
3. status=completed인 학습 run 존재 → "다음: ④ 추론으로 확인하기"
4. qa_snapshot 존재(completed run 없음) → "다음: ③ 학습 시작하기"
5. 데이터셋에 episode 존재(snapshot 없음) → "다음: ② QA 검증하기"
6. 아무 state도 없음 → "① 데이터 수집부터 시작하세요"

**규칙 3 vs 4 충돌 케이스**(completed run과 그보다 새로운 qa_snapshot이
동시에 존재 — 예: 학습 후 episode를 더 모아 재QA한 경우) 처리 방향을
사용자에게 먼저 확인: recency 비교 없이 **항상 규칙 3(순서 그대로)이
우선** — 완료된 run이 있으면 무조건 "④ 추론" 안내, 새 스냅샷 존재
여부는 이 판단에 영향 없음(사용자 확정, AppTest로 실제 두 state 파일을
동시에 만들어 검증 완료).

**`get_stage_progress()` + `render_stage_progress()`**: `[①,②,③,④]`
각 단계 완료 여부(episode 존재/qa_snapshot 존재/completed run 존재)만
체크하는 watermark 방식 — "처음으로 완료 안 된 단계"가 🔵(현재), 그 전은
✅, 그 후는 ⚪. **④는 항상 False 고정** — 추론은 트랙 A/B 둘 다 결과를
디스크에 남기지 않는 세션형 작업이라(페이지 4 설계 시 이미 확인된 사실)
"완료" 개념 자체가 성립하지 않음, 거짓 신호를 만들지 않기로 함 —
①②③이 다 끝나면 ④가 자연히 영구적인 "현재 단계"로 남는 것으로 설계.
홈+①②③④ 5개 페이지 전부 제목 바로 아래에 `render_stage_progress()`
호출(사이드바가 아니라 각 페이지 상단 — Streamlit 멀티페이지 앱은
사이드바 커스텀 콘텐츠가 페이지 간 자동 공유되지 않아 각 파일에서
개별 호출 필요).

**검증**: AppTest로 4가지 state 조합 확인 — (A) 실제 현재 상태(① 완료,
②③④ 미완료 → 규칙 5, "①✅②🔵③⚪④⚪"), (B) 데이터셋 mock으로 완전
빈 상태(→ 규칙 6, "①🔵②⚪③⚪④⚪"), (C) `data_collection` lock mock으로
진행 중 상태(→ 규칙 1, "진행 중: ① 데이터 수집 세션"), (D) completed
run + 그보다 새로운 snapshot을 실제 파일로 동시 생성(→ 규칙 3 승리,
"①✅②✅③✅④🔵" — 확정된 우선순위 그대로 동작 확인). 홈 페이지 실제
브라우저 스크린샷으로 최종 렌더링(CTA 카드 + 진행 표시 + 기존 환경
상태/헬스 요약 섹션과의 배치)까지 확인.

**후속 제거(2026-08-17)**: 사용자가 스크린샷으로 "①✅②✅③🔵④⚪" 진행
표시 자체가 불필요하다고 판단, `render_stage_progress()` 호출을 홈 +
①②③④ 5개 페이지에서 전부 제거(4개 페이지의 이제 안 쓰는
`from lib import next_action` import도 같이 제거, `next_action.py`
파일과 `get_next_action()`/홈의 CTA 카드는 그대로 유지 — 별개 기능).

### 35-6. 범위 밖 — 향후 고려사항(보류)
로그인 기능과 DB 도입은 이번 스코프에 포함하지 않음 — 지금 아키텍처
전제("각자 로컬에서 Streamlit 실행")와 맞지 않고, `state/` 파일 기반
저장으로 지금 규모엔 충분하다고 판단(사용자 확정, 작업 시작 시점에
명시적으로 범위 밖 처리).

## 36. 브랜드 헤더(로고+DAPIER) — DONE (2026-08-17)

부트캠프 로고(`dashboard/assets/logo.png`, 원본은 사용자가 채팅에 첨부한
jpg를 PIL로 변환)와 "DAPIER" 텍스트를 5개 페이지(홈+①②③④) 최상단에
공통 표시. `dashboard/lib/ui_components.py`의 `render_brand_header()` +
`get_logo_icon()`(브라우저 탭 파비콘용, `st.set_page_config(page_icon=...)`에
PIL.Image로 전달 — 실제 서빙되는 `/favicon.png` 응답으로 확인).

**카드 role 색상 팔레트와의 분리(사용자가 명시적으로 강조한 제약)**:
로고 자체가 다색(주황/청록/보라)이라 기존 4-role 색상 시스템과 섞이면
"색상은 의미에 고정" 원칙이 깨짐 → 브랜드 헤더는 배경색 블록을 아예 안
쓰고 로고 이미지+텍스트만, 얇은 회색 hr로 아래 콘텐츠와 구분(카드
헤더처럼 `background-color` div로 만들지 않음). 스크린샷으로 홈(파란
CTA 카드 바로 위)과 페이지 1(파랑/회색/초록 카드 3개 연속 위) 둘 다
확인 — 로고의 짙은 남색 사각 타일이 카드들과 명확히 다른 시각 언어로
분리됨.

**정리 필요(사용자 확인 대기)**: 사용자가 채팅에 첨부한 원본 jpg
(`702162282_18099803617965003_5567386938952634743_n.jpg`)가 프로젝트
루트에 그대로 남아있음 — `dashboard/assets/logo.png`로 이미 변환/복사
완료했으므로 원본은 삭제해도 안전하지만, 사용자 첨부 파일이라 임의로
지우지 않고 그대로 둠.

## 37. 최초 접속 스플래시 — DONE (2026-08-17, 범위 2차례 정정)

### 37-1. 범위가 두 번 바뀐 과정 (셋 다 실제로 구현/검증까지 했다가 폐기)
1. **디스크 저장** (`config/local_settings.json`의 `welcome_seen: bool`):
   `streamlit run`을 몇 번을 새로 실행해도(서버를 껐다 켜도) 다시 안 뜸
   -- "서버 재시작하면 다시 봐야 한다"는 요구와 반대라 폐기.
2. **`st.cache_resource`** (서버 프로세스 전체가 공유하는 메모리 캐시):
   서버 재시작 시 다시 뜨는 것까지는 맞았으나, **같은 서버에 다른 사람이
   새로 접속해도 안 뜸**(먼저 접속한 누군가가 이미 봤으면 그걸로 전
   서버가 "다 봤음" 처리됨) -- "각 개인별로 처음 들어갔을 때는 봐야
   한다"는 요구와 어긋나서 폐기.
3. **`st.session_state`** (최종, 세션=접속당 독립): 세션은 브라우저
   탭이 서버와 맺는 WebSocket 연결 단위 -- 페이지 이동(멀티페이지 앱 내
   sidebar 네비게이션)은 같은 연결을 재사용해 세션이 유지되지만, 서버
   재시작이나 다른 사람의 새 접속은 새 연결=새 세션이라 자동으로
   원하는 3가지 조건(서버 재시작 시 뜸/페이지 이동 시 안 뜸/다른 사람
   접속 시 뜸)을 동시에 만족한다.

**알아두고 확정한 트레이드오프**: 실제 브라우저로 테스트해보니 **본인이
자기 탭을 새로고침(F5)해도 새 세션으로 잡혀 스플래시가 다시 뜬다**
(Streamlit이 새로고침을 세션 유지가 아니라 새 WebSocket 연결로 처리함,
직접 확인). 이는 "다른 사람이 새로 접속하면 뜬다"는 요구와 매커니즘이
완전히 같아서 서버 쪽에서 분리할 방법이 없음(브라우저 identity를 저장할
persistent 저장소가 없는 한). 사용자에게 이 트레이드오프를 명시하고
확인받음 -- **1.8초짜리 부담이라 그냥 감수하기로 확정**(이후 1.3초로 단축,
37-4절), localStorage
기반의 더 복잡한 대안(브라우저별 영구 식별)은 채택 안 함.

### 37-2. 최종 동작
`app.py`가 `st.set_page_config()` 직후, `render_brand_header()`보다
먼저 체크: `"splash_shown" not in st.session_state`면
`render_splash_screen()`(전체화면 덮개, 로고 코너 픽셀 실측 남색
`#051644` + 흰 텍스트, 카드 role 팔레트와 무관) 렌더 →
`time.sleep(1.0)`(Streamlit 네이티브 방식, 클릭 스킵은 범위 밖 — 최초
1.8초에서 사용자 피드백으로 1.0초까지 단계적으로 단축, 37-4절) →
`st.session_state["splash_shown"] = True` → `st.rerun()`.

사이드바의 "🔄 인트로 다시 보기" 버튼은 37-4절에서 사용자 요청으로
제거함 — 필요 없는 기능으로 판단됨.

### 37-3. 검증
**AppTest 한계**: `st.rerun()`을 자동으로 따라가 최종 정착 상태만
보여줘서 스플래시가 뜬 "중간 상태" 자체는 AppTest로 못 잡는다(35-3절
`page_link` 한계와 같은 종류) -- 로직 정합성(예외 없음, elapsed time)
정도만 AppTest로 보고, 실제 시각 확인은 전부 실제 브라우저로 함.

**실제 브라우저 5단계 시나리오(session_state 버전 최종 검증, 전부 확인
완료)**:
1. 서버 최초 기동 → 접속 → 스플래시 뜸
2. 같은 세션에서 다른 페이지로 이동 → 안 뜸
3. 같은 세션에서 홈으로 복귀 → 안 뜸
4. 같은 탭 새로고침(F5) → **다시 뜸**(37-1절 트레이드오프, 사용자 확인 후 확정)
5. 다른 브라우저 컨텍스트(=다른 사람)가 같은 서버에 새로 접속 → 뜸

서버 재시작 시나리오(이전 `st.cache_resource` 버전에서 별도 검증,
`st.session_state`로 바꿔도 이 부분의 메커니즘은 동일 -- 새 프로세스는
당연히 모든 세션이 새로 시작): 서버 종료 → 다른 포트로 재기동 → 접속 →
스플래시 다시 뜨는 것 확인.

### 37-4. 노출 시간 단축 + 인트로 재보기 버튼 제거 — DONE (2026-08-17)
사용자 피드백("생각보다 시간이 긴데 0.5초만 줄여줘") 반영, `time.sleep(1.8)`
→ `time.sleep(1.3)`으로 1차 조정 → 이후 값 자체가 `1.0`으로 다시 조정됨
(외부 편집). F5 재로딩 시 다시 뜨는 트레이드오프(37-1절)는 그대로 유지
— 노출 시간만 줄었을 뿐 재노출 조건 자체는 변경 없음.

이어서 사용자가 "사이드바 인트로 다시 보기 버튼이 불필요하다"고 판단해
제거 요청 — `app.py` 하단의 `st.sidebar` 블록(`splash_shown`을 False로
되돌리는 버튼) 전체 삭제. 같은 세션 안에서 스플래시를 재확인할 방법은
이제 없음(37-2절 문구도 함께 정정) — 필요해지면 새로고침(F5)이 어차피
같은 효과를 낸다는 점(37-1절)이 재추가하지 않기로 한 근거.

## 38. HuggingFace/LeRobot 호환성 실측 검증 — DONE (2026-08-17)

`TeamRobotDataset`이 "LeRobot/HuggingFace와 호환된다"는 주장을 애매하게
두지 않기 위해, 실제 `so101_teleop_real`(18 episodes/9483 frames) 데이터로
3가지 축을 각각 코드 실행으로 검증했다(추측 없음). 스크립트는
`scratch_compat_check.py`(검증 후 삭제, 결과만 이 절에 기록)로 1회성
작성 — 재사용 가치가 없는 진단 스크립트라 산출물은 이 문서뿐.

### 38-1. 저장 포맷 호환 — **된다** (실측 확인)
- `LeRobotDataset("local/so101_teleop_real", root=".../data/so101_teleop_real")`를
  **`TeamRobotDataset` 없이 직접** 생성 → `len=9483`, `num_episodes=18`,
  `fps=30`, `plain[0]`의 키가 `action/observation.state/
  observation.images.wrist_cam/timestamp/frame_index/episode_index/index/
  task/task_index`로 정상 반환 확인. 당연한 결과이지만 근거는 명확함:
  `team_robot_dataset.py` 전체(1028줄)를 다시 읽어봐도 `add_frame`/
  `create`/`save_episode`/`finalize` 등 디스크에 쓰는 메서드가 단 하나도
  없음(모두 `_light_items()`처럼 읽기 전용 스캔, 또는 `_base`로의 위임) —
  즉 디스크 포맷은 애초에 **`TeamRobotDataset`이 관여할 수 없는 영역**이고,
  실제로 그 디스크에 쓰는 주체는 `run_teleop_real.py`가 호출하는 순정
  `LeRobotDataset`뿐이므로 포맷이 어긋날 방법 자체가 없음.
- 한 단계 더 내려가 원본 parquet도 직접 확인: `data/chunk-000/*.parquet`을
  순정 `datasets.load_dataset("parquet", data_files=...)`(lerobot 무관,
  일반 HF 로더)로 열어 `num_rows=9483`, 컬럼 8개가 그대로 조회됨 — lerobot
  래퍼조차 없이도 표준 HF 파케이 로더로 열리는 순수 HF 포맷.
- 이미지 저장 방식도 확인(`images/` 폴더는 실제로 **비어있음**, 파일 0개)
  → `pyarrow.parquet.read_schema()`로 실제 컬럼 타입을 보니
  `observation.images.wrist_cam: struct<bytes: binary, path: string>` —
  HF `datasets`의 표준 `Image` feature 인코딩 그대로 parquet 안에 바이트로
  내장돼 있음(video가 아니라 image 모드, `meta/info.json`의
  `"video_path": null`과 일치). 외부 이미지 파일 의존이 전혀 없는
  완전 자체완결(self-contained) 포맷.
- **결론**: 포맷 호환은 설계상 자명함(TeamRobotDataset이 storage를 아예
  건드리지 않으므로) — 이번 검증은 "그럴 것이다"가 아니라 실제로 열어서
  숫자까지 맞춰 확인한 것.

### 38-2. Hub 업로드 호환 — **된다(구조적으로), 단 실제 push는 미실행**
- `hasattr(team, "push_to_hub")` → `True`. 더 중요한 확인:
  `team.push_to_hub.__self__ is team._base` → **`True`** — 즉
  `TeamRobotDataset`이 `push_to_hub`를 자체 구현/래핑한 게 전혀 아니라,
  `__getattr__`을 통해 `self._base`(순정 `LeRobotDataset`)의 **바로 그
  bound method**를 그대로 돌려주는 것. `team.push_to_hub(...)`을 호출하는
  것과 `LeRobotDataset(repo_id, root=...).push_to_hub(...)`을 직접 호출하는
  것 사이에 코드 경로상 차이가 전혀 없음(같은 객체의 같은 메서드) —
  "추가로 뭘 더 해야" 하는 게 구조적으로 없다는 뜻.
- 시그니처도 실측: `push_to_hub(self, branch=None, tags=None,
  license='apache-2.0', tag_version=True, push_videos=True, private=False,
  allow_patterns=None, upload_large_folder=False, **card_kwargs)`.
- **실제 네트워크 push는 이번 검증에서 실행하지 않음**(HF Hub 인증 토큰
  설정 여부 불명, 실제로 리포지토리에 업로드하는 부수효과가 있는 위험한
  호출이라 이번 "호환성 확인" 범위에서 제외 — push_to_hub 자체가 실제로
  성공하는지는 lerobot 라이브러리가 이미 테스트했을 영역이지 이 래퍼가
  새로 검증할 대상이 아님). 검증한 것은 정확히 "TeamRobotDataset을 거쳐도
  push_to_hub 호출이 순정 LeRobotDataset과 완전히 동일한 경로로 간다"는
  구조적 사실.

### 38-3. 인터페이스 호환 — **함수적으로는 된다, 두 가지 실측된 예외 지점 있음**
- **`getattr()` 기준 완전 위임 확인**: 순정 `LeRobotDataset`의 public
  속성/메서드 37개(`dir()` 기준, `_`로 시작하는 것 제외 — `add_frame`,
  `meta`, `push_to_hub`, `hf_dataset`, `features`, `episodes`, `fps`,
  `num_episodes` 등 전부 포함) 전부를 `getattr(team, name)`으로 하나씩
  실제 호출 → **실패 0건**. `__getattr__` 위임(`team_robot_dataset.py:109-113`)이
  주장대로 완전히 동작함을 37개 전수 확인.
- **예외 1(경미, 이번에 새로 발견됨 — 발견 직후 저비용이라 바로 수정,
  38-5절)**: `dir(team)`은 **10개만 보여줌**(TeamRobotDataset 자신이
  정의한 메서드 + object 기본 속성뿐). `__getattr__`은 속성 접근 실패
  시에만 호출되는 훅이라 `dir()`/IDE 자동완성/`help(team)`에는 관여하지
  못하는 게 Python 자체의 근본 한계 — `__dir__`을 별도로 오버라이드하지
  않는 한 구조적으로 항상 이렇게 됨. **기능적으로는 문제 없었음**(방금
  확인한 대로 `getattr()` 자체는 37/37 성공) — 하지만 "LeRobotDataset과
  인터페이스가 똑같다"를 `dir(team)`이나 탭 자동완성으로 탐색하려는
  사람에게는 실제로 안 보이므로 오도될 수 있음. **설계 의도가 아니라
  놓친 부분**으로 판단(3절 composition 설계 자체는 옳으나 `__dir__`
  보완이 빠짐) — 저위험 수정이라 발견 직후 바로 고침, 결과는 38-5절.
- **예외 2(설계 의도, 문제 아님 — 17절에서 이미 선택된 트레이드오프의
  실측 재확인)**: `get_clean_indices()` → `torch.utils.data.Subset(team,
  clean_indices)` 패턴(17절 공식 사용법) 자체를 실행해 확인한 결과,
  `Subset`은 **`__getattr__`을 위로 전달하지 않음** — 실측:
  `hasattr(subset, "meta")` → `False`, `hasattr(subset, "push_to_hub")` →
  `False`. 즉 QA 필터링 결과를 `Subset`으로 감싸는 순간
  `TeamRobotDataset`/`LeRobotDataset` 인터페이스 자체를 잃음(이건 lerobot도
  TeamRobotDataset도 아니라 **`torch.utils.data.Subset`이라는 PyTorch
  표준 클래스의 범용 동작** — 애초에 어떤 `Dataset`을 감싸도 똑같이
  일어남). 사용자가 우려했던 "필터링된 인덱스가 원본과 안 맞는 경우"는
  **이미 16절에서 근본 수정되어 실제로는 문제없음**(`frame_idx`가
  필터링 여부와 무관하게 안정적인 에피소드-로컬 `frame_index` 기준이라
  이번 재확인 대상에서도 제외) — 대신 실제로 존재하는 것은 인덱스
  불일치가 아니라 **"필터링 후 껍데기(Subset)가 원본 메타/Hub 메서드를
  안 가진다"는 것**. `get_clean_indices()`의 docstring이 애초에
  `DataLoader(Subset(...))`용으로만 설계된 것이었고(정규화가 필요하면
  원본 `team.meta.stats`/`team.normalize()`를 별도로 계속 쓰라는 전제,
  17절), `DataLoader` 자체는 `__len__`/`__getitem__`만 있으면 되므로
  이 용도에서는 실측으로도 정상 동작 확인(`DataLoader(subset,
  batch_size=4)`의 첫 배치가 `action.shape=(4,6)`으로 정상 반환) — **설계
  의도대로 동작하는 트레이드오프이지, 놓친 버그가 아님**.

### 38-4. 종합 판단
1. 저장 포맷 호환: **된다** (구조적으로 자명 + 실측 확인)
2. Hub 업로드 호환: **된다** (구조적으로 자명 + 실측 확인, 실제 네트워크
   push 자체는 미실행 — lerobot 자체가 이미 검증한 영역이므로 범위 밖)
3. 인터페이스 호환: **함수적으로는 된다**(`getattr()` 37/37 성공) — 단
   `dir()` 미노출은 **놓친 부분**이었으나 즉시 수정 완료(38-5절),
   `Subset` 사용 시 인터페이스 소실은 **의도된 설계**(17절 트레이드오프,
   실사용 패턴에서는 문제 없음).

### 38-5. `__dir__` 오버라이드 추가 — DONE (2026-08-17)
38-3절에서 발견한 `dir(team)` 미노출을 저비용 수정으로 판단해 바로
반영. `team_robot_dataset.py`의 `__getattr__` 바로 아래에
`__dir__(self)` 추가:
```python
def __dir__(self):
    return sorted(set(super().__dir__()) | set(dir(self._base)))
```
`super().__dir__()`(TeamRobotDataset 자신이 정의한 메서드 + object 기본
속성)과 `dir(self._base)`(위임 대상인 순정 LeRobotDataset의 전체 표면)의
합집합 — 교체가 아니라 합집합이라 `detect_outliers`/`normalize` 등
TeamRobotDataset 자체 메서드도 그대로 같이 보임.

**재검증(동일한 so101_teleop_real 인스턴스로 재실행)**:
- `dir(team)`의 public 속성 수: 10개 → **47개**로 증가
- 38-3절에서 확인했던 순정 LeRobotDataset의 public 속성 37개(`meta`,
  `push_to_hub`, `hf_dataset`, `features`, `episodes`, `fps`,
  `num_episodes`, `add_frame` 등)가 **전부(37/37) `dir(team)`에 포함됨**
  확인(빠진 것 0개)
- TeamRobotDataset 자체 메서드 10개(`compute_custom_stats`,
  `detect_outliers`, `detect_outliers_delta`, `check_integrity`,
  `get_clean_indices`, `split_episodes`, `normalize`, `unnormalize`,
  `visualize_episode`, `plot_action_distribution`)도 전부 `dir(team)`에
  여전히 존재 확인(합집합이 실제로 한쪽을 안 지웠는지 회귀 확인)
- `getattr(team, name)` 37개 전수 재확인 — 실패 0건(이번 수정이 기존
  `__getattr__` 위임 동작 자체를 건드리지 않았는지 회귀 확인)

이걸로 38-3절의 유일한 흠이었던 `dir()` 미노출도 해소됨 — 38-4절 결론이
"인터페이스 호환: 함수적으로는 된다"에서 "인터페이스 호환: 된다(함수적
동작 + `dir()` 노출 모두 확인)"로 갱신됨.

**전체 결론**: `TeamRobotDataset`을 "HuggingFace/LeRobot과 호환된다"고
말하는 것은 정확한 표현임 — 1절 목적(3절: base 클래스 위에 커스텀
레이어를 얹되 절대 수정하지 않는 순수 composition)이 애초에 포맷/Hub
호환을 깨뜨릴 수 없는 구조를 강제했고, 실측으로도 어긋나는 지점이
없음을 확인함. 유일하게 새로 발견한 흠은 `dir()` 미노출(기능에는 영향
없는 introspection 한계)뿐.

## 39. 정규화(MEAN_STD/IDENTITY) "결정 vs 적용" 위치 확인 — DONE (2026-08-17)

38절과 같은 틀(코드/데이터를 직접 보고 판단, 추측 금지)로, `TeamRobotDataset`의
`normalize()`/`unnormalize()`(19절)가 실제로 어디서 "적용"되는지 확인했다.
`team_robot_dataset.py`(이미 읽은 소스, 재인용)와 설치된 lerobot 0.4.4
소스(`miniconda3/envs/mujoco_env/Lib/site-packages/lerobot/`) 양쪽을
직접 읽어 대조했다.

### 39-1. `team.__getitem__()`이 반환하는 값 — **raw(미정규화)**
```python
def __getitem__(self, idx):
    return self._base[idx]
```
(`team_robot_dataset.py:103-104`) — `normalize()` 호출이 전혀 없음.
`self._base[idx]`는 순정 `LeRobotDataset.__getitem__`이 반환하는 그대로의
텐서(raw 값)이고, `normalize()`/`unnormalize()`는 **`__getitem__`과 무관한
별도 메서드**라 호출자가 명시적으로 `team.normalize(team[i])`처럼 불러야만
적용됨. 자동으로 걸리는 경로가 없음.

### 39-2. 정규화 통계가 계산/저장되는 위치
- `normalize()`/`unnormalize()`가 읽는 통계는 `self._base.meta.stats`
  (`team_robot_dataset.py:727`, `:769`) — **`compute_custom_stats()`(7절,
  아웃라이어 탐지용 자체 통계)와는 완전히 다른 값**이며 서로 참조하지
  않음(19-1절에서 이미 "정규화 기준: dataset.meta.stats 채택 — 자체 계산한
  compute_custom_stats() 대신"으로 명시된 결정, 이번에 소스로 재확인).
- `self._base.meta.stats`는 디스크의 `meta/stats.json` 파일에서 옴. 이
  파일을 **누가 언제 쓰는지** lerobot 소스로 확인: `LeRobotDataset.save_episode()`
  (`lerobot_dataset.py:1225`)가 내부적으로 `self.meta.save_episode()`를
  호출하고(`:1342`), `LeRobotDatasetMetadata.save_episode()`
  (`lerobot_dataset.py:399-425`) 안에서
  `self.stats = aggregate_stats([self.stats, episode_stats])` →
  `write_stats(self.stats, self.root)`로 **에피소드 저장(녹화 세션의 S/X
  키 종료) 시점마다 lerobot 자신이 직접 계산·갱신**함. `TeamRobotDataset`은
  이 파일을 만들지도, 갱신하지도 않음 — 오직 읽기만 함(`_warn_on_stale_meta_stats()`도
  이 파일의 신선도를 "검사"만 할 뿐 재계산하지 않음, 18-1/19-1절과 일치).
- `TeamRobotDataset` 인스턴스 자신의 속성으로 별도 캐싱도 안 함 —
  `normalize()`/`unnormalize()`를 호출할 때마다 `self._base.meta.stats`를
  매번 다시 읽음(base의 `LeRobotDatasetMetadata` 객체가 이미 속성으로
  들고 있는 값이라 캐싱할 이유가 없음).

### 39-3. `lerobot-train`이 이 통계를 실제로 쓰는 방식 — **TeamRobotDataset을 전혀 거치지 않음**
코드베이스 전수 검색(`grep -rn ".normalize(\|.unnormalize("`) 결과,
`team_robot_dataset.py` 자기 자신의 정의부(`normalize`/`unnormalize`
메서드 선언 그 자체) 외에는 **호출자가 0건**(테스트 파일 포함, `test_team_robot_dataset.py`에도
"normalize" 문자열 자체가 없음). 대신 `lerobot-train`은 아래 경로로
완전히 독립적인 자기 자신의 정규화를 수행한다(lerobot 소스로 확인):

1. `lerobot_train.py:218/224` `dataset = make_dataset(cfg)` →
   `lerobot/datasets/factory.py:93` `LeRobotDataset(...)` — **순정
   LeRobotDataset을 직접 새로 생성**(TeamRobotDataset이 전혀 개입 안 함,
   24절 "TeamRobotDataset은 학습에 직접 연결되지 않는다" 결론과 일치).
2. `lerobot_train.py:238` `policy = make_policy(..., ds_meta=dataset.meta)`
   → `policies/factory.py:477` `kwargs["dataset_stats"] = ds_meta.stats`로
   정책 생성 시점에 정규화 레이어 내부 버퍼로 주입.
3. `lerobot_train.py:263-266, 282-286`:
   ```python
   processor_kwargs["preprocessor_overrides"] = {
       "normalizer_processor": {
           "stats": dataset.meta.stats,
           "features": {**policy.config.input_features, **policy.config.output_features},
           "norm_map": policy.config.normalization_mapping,
       },
   }
   preprocessor, postprocessor = make_pre_post_processors(...)
   ```
   → `NormalizerProcessorStep`(`lerobot/processor/normalize_processor.py:402`)
   생성 — 체크포인트 저장 시 `policy_preprocessor.json`/`policy_postprocessor.json`으로
   재직렬화됨(31-4절/26절에서 이미 확인된 사실과 일치).
4. **실제 텐서 변환이 일어나는 지점**: 학습 루프 안의 `batch = preprocessor(batch)`
   (`run_val_loss_check.py` 주석에서도 동일 패턴 확인됨: "체크포인트에
   저장된 정규화 설정 자동 적용") → `NormalizerProcessorStep.__call__`
   (`normalize_processor.py:445`) → `_apply_transform()`(`:278-390`)이
   MEAN_STD 공식 `(x-mean)/(std+eps)`, `eps=1e-8`을 실제로 적용. 이 공식은
   `TeamRobotDataset.normalize()`가 재구현한 것과 정확히 동일(19-1절에서
   이미 소스 대조로 확인된 사실의 재확인일 뿐, 새로운 발견 아님).

즉 **같은 `meta/stats.json` 파일을 읽지만, 읽는 주체도 다르고 적용 지점도
다르다**: 학습 시점엔 `lerobot-train`이 `dataset.meta.stats`를 (같은 파일,
다른 객체 인스턴스로) 독자적으로 다시 읽어 자기 자신의
`NormalizerProcessorStep`으로 적용한다. `TeamRobotDataset.normalize()`/
`unnormalize()`는 이 경로 어디에도 연결돼 있지 않다.

### 39-4. 그럼 `normalize()`/`unnormalize()`는 왜 존재하는가
현재 코드베이스 어디서도 호출되지 않는 상태(39-3절 grep 결과, 0건) —
19-2절 작업 당시("이미지 필드 MEAN_STD → IDENTITY 전환" 판단 근거를
만들기 위해 직접 정규화를 재현해 이미지 정규화 값이 `-7611~+10917`로
비정상임을 발견) 진단 도구로 만들어졌고, 그 진단 목적은 이미 달성되어
결론(`VISUAL=IDENTITY`)이 실제 학습에는 `--policy.normalization_mapping.VISUAL=IDENTITY`
CLI 오버라이드(25/31-4절)로 별도 반영됨 — 즉 "학습에 실제로 쓰이는
정규화 로직"이 아니라 "학습에 넘길 정규화 설정이 맞는지 검증하기 위해
lerobot의 공식을 손으로 재현해본 진단용 유틸리티"였고, 그 이후로는
호출될 필요가 없어져 코드만 남아있는 상태로 판단됨. 이번 요청 범위(위치
확인)를 벗어나므로 코드 자체는 그대로 둠 — 다만 "학습 파이프라인이 이
메서드를 쓴다"는 오해는 없어야 함을 이 절에 명시해둔다.

### 39-5. 종합: "결정(통계 계산) vs 적용(텐서 변환)"

| 구분 | 통계 계산 | 통계 저장 | 실제 적용(텐서 변환) |
|---|---|---|---|
| `meta/stats.json` 자체 | **lerobot 자신**(`LeRobotDatasetMetadata.save_episode()`, 에피소드 저장 시마다) | `meta/stats.json` 파일(디스크) | (여기선 적용 안 함, 값만 계산·저장) |
| `TeamRobotDataset.normalize()`/`unnormalize()` | 안 함(위 파일을 그대로 읽기만) | 안 함(캐싱도 없음) | 호출 시점에 직접 적용 — 단 `__getitem__` 경로엔 없고, 현재 아무도 호출하지 않음(39-4절) |
| `lerobot-train` 실제 학습 | 안 함(같은 `meta/stats.json`을 자기 객체로 다시 읽음) | 체크포인트의 `policy_preprocessor.json`/`policy_postprocessor.json`으로 재직렬화 | `NormalizerProcessorStep.__call__`(`preprocessor(batch)`), 학습 루프 매 배치마다 — `TeamRobotDataset`과 완전히 독립적인 lerobot 자체 파이프라인 |

**결론**: "정규화 결정(통계 계산)"은 `TeamRobotDataset`도 `lerobot-train`도
아니라 **lerobot 자신이 녹화 시점에 이미 다 해놓은 것**이고, 두 소비자
(`TeamRobotDataset.normalize()`, `lerobot-train`)는 그 결과 파일을 각자
독립적으로 다시 읽어 쓸 뿐이다. "실제 적용(텐서 변환)"은 학습 경로에서는
**`TeamRobotDataset`이 전혀 관여하지 않고 lerobot 자체
`NormalizerProcessorStep`에서 일어난다** — `TeamRobotDataset.normalize()`는
학습 경로와 분리된, 과거 진단 목적으로 쓰인 뒤 현재는 호출되지 않는
유틸리티다.

## 40. 모방학습 표준 전처리 4항목 실태 확인 — DONE (2026-08-17)

39절과 같은 틀("결정 vs 적용" 위치를 코드로 직접 대조)로, 로봇 모방학습에서
흔히 필요한 4가지 전처리(이미지 리사이즈/증강, 액션 델타/청킹, 프레임
스태킹, 센서 동기화)가 이 파이프라인의 어디서(① `TeamRobotDataset`/QA
계층, ② ACT policy 자체, ③ `lerobot-train` CLI 옵션, ④ 아무 데도 없음)
처리되는지 확인했다. `team_robot_dataset.py`(전체 재확인), 설치된 lerobot
0.4.4 소스(`configuration_act.py`, `modeling_act.py`, `datasets/transforms.py`,
`datasets/factory.py`, `processor/delta_action_processor.py`), 그리고
`pipeline/1_data_collection/run_teleop_real.py`(녹화 루프)를 직접 대조했다.

### 40-1. 이미지 리사이즈 — **④ 아무 데도 없음(단, 우연히 필요가 없는 상태)**
- `team_robot_dataset.py` 전체(1028줄)에 리사이즈 관련 코드 없음(재확인,
  38/39절에서 이미 전체를 읽었던 파일).
- ACT 자체(`configuration_act.py`, `modeling_act.py`)에도 `resize`/`crop`/
  `image_size` 관련 코드가 **0건**(grep 확인) — 대신 `input_features`가
  `dataset_to_policy_features(ds_meta.features)`(정책 생성 시점, `policies/factory.py`)로
  **데이터셋이 보고하는 실제 shape를 그대로** 받아옴. 카메라 특징의
  위치 임베딩도 `ACTSinusoidalPositionEmbedding2d`(`modeling_act.py:361`)가
  **매 forward마다 실제 백본 출력 feature map 크기(`cam_features`)를 보고
  동적으로 계산**(`modeling_act.py:474`, `create_sinusoidal_pos_embedding`류의
  고정 크기 버퍼가 아님)함 — 즉 ACT 구조 자체가 특정 고정 해상도를
  가정하지 않음(어떤 입력 크기든 구조적으로 받아들임).
- `lerobot-train` CLI에도 리사이즈 옵션이 없음(`ImageTransformsConfig`에
  crop/resize 항목 없음, 40-2절 참고).
- **왜 지금 문제가 안 되는가**: `run_teleop_real.py:190-191`에서
  `IMAGE_HEIGHT=480, IMAGE_WIDTH=640`으로 MuJoCo 렌더러 자체를 이 해상도로
  고정해서 캡처하고, 이 값이 그대로 `meta/info.json`의
  `observation.images.wrist_cam.shape=[480,640,3]`로 저장돼(2절에서 이미
  확인) 학습까지 변형 없이 그대로 흘러감 — "리사이즈가 필요 없어서 안
  만든 것"이 아니라 **"캡처 단계에서 이미 원하는 해상도로 고정 촬영해서
  결과적으로 리사이즈 불필요"**인 상태. 카메라를 바꾸거나 원본 해상도가
  달라지면(예: 실물 카메라로 전환, 19-2/25절에서 이미 언급된 미래
  과제) 이 빈 자리가 실제로 드러날 수 있음 — **진짜로 빠진 부분**으로
  기록해둠(지금은 우연히 안전한 상태).

### 40-2. 이미지 증강 — **③ lerobot-train CLI 옵션으로 가능, 현재 미사용(꺼짐)**
- `lerobot/datasets/transforms.py`에 `ImageTransformsConfig`
  (`enable: bool = False` 기본값, `torchvision.transforms.v2` 기반
  brightness/contrast/saturation/hue/sharpness/affine 6종, `RandomSubsetApply`로
  매 프레임 무작위 부분집합 적용)가 이미 구현돼 있고, `dataset.image_transforms.*`
  CLI 오버라이드로 켤 수 있음.
- `TeamRobotDataset`/`run_teleop_real.py` 어디에도 이 기능을 사용하는
  코드 없음(grep 0건), **`config/train_main_run_config.yaml`에
  `image_transforms` 섹션 자체가 없음** → 기본값 `enable=False` 그대로
  적용됨 → **현재 프로젝트는 이미지 증강을 전혀 쓰고 있지 않음**(꺼진
  상태, 코드가 없어서가 아니라 옵션을 안 켠 것).
- lerobot이 이미 표준 제공하는 기능이라 `TeamRobotDataset`/ACT 쪽에서
  따로 구현할 이유가 없음(있는 걸 안 켠 상태 — "빠진 부분"이 아니라
  "미사용 옵션").

### 40-3. 액션 델타(상대 표현) — **③ lerobot에 빌딩 블록은 있으나 ACT 기본 경로엔 미배선, ④ 이 프로젝트엔 사실상 해당 없음**
- `lerobot/processor/delta_action_processor.py`에
  `MapDeltaActionToRobotActionStep`/`MapTensorToDeltaActionDictStep`이
  존재하지만, `processor/__init__.py`에서 재수출되는 것 외에는
  **`policies/factory.py`/`configuration_act.py`/`lerobot_train.py`
  어디에서도 참조되지 않음**(grep 확인) — ACT의 기본 학습/추론 경로에
  전혀 배선돼 있지 않은, RL/원격조작 환경(`gym_manipulator.py` 등)용
  별도 빌딩 블록으로 판단됨.
- 우리 프로젝트의 `action` 필드는 처음부터 **절대 관절 라디안 값**(23절에서
  이미 확인: leader→follower 비례 스케일링으로 변환된 절대 위치, 델타가
  아님) — 애초에 "액션을 델타로 표현할지"를 선택하는 지점 자체가 이
  파이프라인 설계에 없고, ACT도 절대 액션 그대로 MEAN_STD 정규화만
  적용(39절)해서 씀. 즉 "빠진 부분"이라기보다 **이 정책/이 액션 표현
  방식(절대 관절제어)에는 애초에 필요하지 않은 옵션**.

### 40-4. 액션 청킹 — **① QA 계층은 경고만, ③ lerobot-train이 자동 구성, 실제 적용은 순정 LeRobotDataset**
(20절에서 이미 조사된 내용의 재확인 + 소스 라인 보강)
- **결정**: `ACTConfig.chunk_size: int = 100`(`configuration_act.py:86`),
  `get_optimizer_delta_indices()`(`:173`)가 `list(range(chunk_size))` 반환
  → `train_main_run_config.yaml`에 `policy.chunk_size`를 별도로 안
  적어도 ACT 기본값 100이 자동 적용됨.
- **적용**: `lerobot/datasets/factory.py`의 `resolve_delta_timestamps()`류
  로직이 이 델타 인덱스를 `delta_timestamps={"action": [i/fps for i in
  range(chunk_size)]}` 형태로 변환해 **순정 `LeRobotDataset`**(`make_dataset(cfg)`가
  만드는 것, 39-3절) 생성자에 전달 → 실제 시퀀스 확장(청크 차원 만들기)은
  `LeRobotDataset.__getitem__` 내부에서 일어남(에피소드 경계를 넘으면
  마지막 프레임 반복 + `action_is_pad=True`로 클램핑, 20-1절에서 이미
  확인).
- **`TeamRobotDataset`의 역할**: 청킹을 스스로 구현하지 않음 — 대신
  `_warn_on_delta_timestamps_keys()`(`team_robot_dataset.py:790-824`)가
  `self._base.delta_timestamps`가 설정된 채로 `visualize_episode()`/
  `plot_action_distribution()`이 호출되면 "청크 차원이 조인트 차원과
  섞여 결과가 조용히 틀어질 수 있다"고 경고만 함(20-2절). QA 계층
  4개 스캔 메서드(`compute_custom_stats` 등)는 `_light_items()`가 원본
  HF 컬럼을 직접 읽어 애초에 `delta_timestamps` 확장을 거치지 않으므로
  영향이 없음(20-2절, 부수적으로 안전).

### 40-5. 프레임 스태킹(관측 이력) — **② ACT 자체가 명시적으로 미지원(구조적 제약)**
`configuration_act.py:85` `n_obs_steps: int = 1`, `:149-151`에서
```python
if self.n_obs_steps != 1:
    raise ValueError(
        f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
    )
```
— **ACT는 관측 프레임 스태킹을 지원하지 않는다는 게 lerobot의 ACT 구현
자체에 하드코딩된 제약**(`n_obs_steps`를 1이 아닌 값으로 주면 정책
생성 시점에 즉시 `ValueError`). 우리 파이프라인이 놓친 게 아니라
**정책 선택(ACT) 자체의 구조적 한계** — 다른 policy 타입(예: diffusion)은
`n_obs_steps>1`을 지원할 수 있으나 이 프로젝트는 ACT를 쓰므로 해당
없음. `TeamRobotDataset`/`lerobot-train` CLI 어느 쪽도 이걸 우회할
방법을 제공하지 않음(제공할 이유도 없음 — 정책이 안 받는데 데이터
쪽에서 스택을 만들어봐야 무의미).

### 40-6. 센서 동기화 — **④ 별도 메커니즘 없음(구조적으로 필요 자체가 없음)**
`run_teleop_real.py`의 단일 제어 루프(785-810행)를 직접 확인:
```python
frame_accum += loop_dt
if frame_accum >= frame_interval:
    frame_accum -= frame_interval
    renderer.update_scene(data, camera=WRIST_CAMERA)
    wrist_image = renderer.render()
    ...
    if recording:
        dataset.add_frame({
            "observation.state": observation_state,  # 같은 루프 반복에서 읽은 값
            "action": targets,                          # 같은 루프 반복에서 읽은 값
            f"observation.images.{WRIST_CAMERA}": wrist_image,  # 같은 루프 반복에서 렌더
            "task": args.task,
        })
```
물리 스텝은 `loop_dt`(=`1/args.hz`, 예: 50Hz) 주기로 매번 돌지만, 카메라
렌더/`add_frame()`은 `frame_interval`(=`1/RECORD_FPS`, 예: 30Hz) 주기로만
서브샘플링됨(누산기 `frame_accum` 방식) — 그런데 이 서브샘플링된 매
순간에도 `observation_state`/`targets`/`wrist_image`가 **전부 그 순간의
같은 루프 반복(iteration) 안에서 나온 값**이라, 타임스탬프 정렬/보간
같은 별도 동기화 알고리즘이 필요 없음. 이건 별도 스레드/프로세스에서
비동기로 도착하는 여러 센서 스트림을 사후에 짝짓는 구조가 **아니라**,
단일 스레드가 "이 순간의 상태·액션·이미지"를 한 번에 같이 만들어내는
구조라서 동기화 문제 자체가 애초에 발생하지 않음. `TeamRobotDataset`/
ACT/`lerobot-train` 어느 쪽도 이 부분에 관여하지 않는데, 관여할 대상
자체가 없기 때문(관여가 빠진 게 아니라 필요가 없는 것).

### 40-7. 종합 표

| 항목 | ① QA 계층(`TeamRobotDataset`) | ② ACT policy 자체 | ③ `lerobot-train` CLI | ④ 안 됨/해당없음 |
|---|---|---|---|---|
| 이미지 리사이즈 | 안 함 | 특정 해상도 비가정(구조적으로 수용) | 옵션 없음 | **빠짐 — 캡처 해상도(480x640) 고정으로 우연히 회피 중** |
| 이미지 증강 | 안 함 | (해당 없음, 데이터 레벨) | **있음(`image_transforms`), 기본 꺼짐** | 현재 프로젝트는 미사용(옵션 존재, 안 켬) |
| 액션 델타(상대 표현) | 안 함 | 절대 액션만 사용(MEAN_STD 정규화, 39절) | 빌딩블록 있으나 ACT 경로 미배선 | 이 프로젝트 액션 표현(절대 관절값)엔 불필요 |
| 액션 청킹 | 경고만(`_warn_on_delta_timestamps_keys`) | **`chunk_size=100` 결정** | 자동 구성(dataset factory) | 실제 적용은 순정 `LeRobotDataset.__getitem__` |
| 프레임 스태킹 | 관여 없음 | **`n_obs_steps=1` 강제(지원 안 함)** | 관여 없음 | 정책 자체의 구조적 미지원 |
| 센서 동기화 | 관여 없음 | 관여 없음 | 관여 없음 | **필요 자체가 없음**(녹화 루프의 단일 반복 구조) |

**결론**: 6개 항목 중 정말로 "구멍"이라 부를 만한 것은 **이미지
리사이즈** 하나뿐 — 지금은 캡처 해상도를 미리 맞춰놓아 문제가 드러나지
않지만, 카메라/해상도가 바뀌면 즉시 필요해질 수 있는 잠재 공백. 나머지
(증강, 액션 델타)는 "없는 게 아니라 안 쓰는 것"(옵션은 존재), 청킹은
정상적으로 lerobot 표준 경로로 이미 처리되고 있음(20절에서 이미 검증),
프레임 스태킹은 ACT 자체가 지원하지 않는 것이 정책 선택에 따른 당연한
제약, 센서 동기화는 애초에 별도 처리가 필요 없는 구조.

## 41. 40절 빈 자리(이미지 리사이즈) 해소 + 정규화 죽은 코드 처리 — DONE (2026-08-17)

40절에서 발견한 유일한 "진짜 빈 자리"(이미지 리사이즈)와, 39절에서 발견한
`normalize()`/`unnormalize()` 죽은 코드(호출자 0건)에 대해 각각 제안 →
사용자 승인 → 구현 → 실측 검증까지 완료했다.

### 41-1. 리사이즈 갭: 옵션 1(lerobot `image_transforms`) vs 옵션 2(명시적 상수 + 방어 코드) — **옵션 2 채택**
제안 단계에서 옵션 1을 기각한 근거를 실제 코드로 재확인:
- `image_transforms`는 `LeRobotDataset(image_transforms=...)`를 통해
  **학습 시 데이터 로딩 단계에서만** 적용되는 opt-in 변환(40-2절)이고,
  `run_inference_mujoco.py`는 `LeRobotDataset`을 아예 만들지 않고
  `renderer.render()` 결과를 정책에 바로 먹인다 — 학습 쪽에만 리사이즈를
  걸면 "학습은 리사이즈된 이미지를 봤는데 추론은 원본을 본다"는 train/inference
  불일치가 새로 생김(지금 상태보다 더 위험).
- 결정적 근거: `run_inference_mujoco.py`도 `IMAGE_HEIGHT=480, IMAGE_WIDTH=640`을
  `run_teleop_real.py`와 **완전히 독립적으로 재선언**하고 있었음(수정 전
  기준 두 파일 모두 190-191행, grep으로 확인) — 지금은 우연히 같은 값이지만
  둘 중 하나만 바뀌어도 아무 경고 없이 어긋나는 구조. 진짜 위험은 "ACT가
  특정 해상도를 못 받는다"가 아니라 "두 파일의 하드코딩된 상수가 조용히
  드리프트한다"는 것이었음.

### 41-2. 구현: `pipeline/camera_config.py`(신규) + 양쪽 스크립트 방어 코드
- **`pipeline/camera_config.py`**: `WRIST_IMAGE_HEIGHT=480`, `WRIST_IMAGE_WIDTH=640`
  단일 정의. `pipeline/`은 패키지가 아니라(`__init__.py` 없음) 각 스크립트가
  독립 실행되는 구조라, 두 스크립트 모두 이미 계산해두던
  `PROJECT_ROOT = Path(__file__).resolve().parents[2]`를 이용해
  `sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))` 후
  `from camera_config import WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH`로 가져옴
  (기존 `IMAGE_HEIGHT`/`IMAGE_WIDTH` 지역 상수는 이 값을 대입받는 형태로
  유지 — 다른 코드에서 참조하는 이름을 바꾸지 않기 위함). 두 파일 모두
  `sys`가 함수 내부에만 지역 임포트돼 있었고 모듈 레벨엔 없어서
  `import sys`를 최상단 import 블록에 추가.
- **방어 assert**: 두 스크립트 모두 `renderer.render()` 직후, 실제
  사용(`add_frame()`/policy 입력) 직전에
  `assert wrist_image.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH)` 추가.
  단순히 넣기만 하면 무의미했을 지점: 두 스크립트 모두 이 렌더 호출을
  감싸는 `try: ... except Exception: print(...); traceback.print_exc()`
  (해당 프레임만 건너뛰고 계속 진행하는 관대한 패턴, transient 카메라
  오류용)가 이미 있어서, 그 안에 assert를 그냥 넣으면 매 프레임 계속
  실패하며 조용히 삼켜지기만 하고 세션은 안 멈추는(결과적으로 프레임을
  하나도 못 쓰는데 로그만 계속 찍히는) 상태가 됐을 것 — "명시적 에러"라는
  요청 의도와 어긋남. `except AssertionError: raise`를 `except Exception:`
  **앞에** 추가해 해상도 불일치만 예외적으로 다시 던지도록 분리(그 외
  transient 오류는 기존과 동일하게 관대히 처리).
  - `run_teleop_real.py`: 바깥쪽이 `try: ... finally: dataset.finalize() ...
    finally: renderer.close(); safe_disconnect(leader)` 구조(바깥에 이
    AssertionError를 삼키는 `except`가 없음, 코드 확인)라 재던짐 →
    `finally` 정리(포트 닫기/`dataset.finalize()`)는 그대로 실행된 뒤
    →`main()` 밖으로 크게 실패. 이미 녹화된 에피소드는 유실 없음.
  - `run_inference_mujoco.py`: 뷰어 모드는 동일한 구조로 재던짐 처리.
    헤드리스 모드(`--headless`, `_select_action()` 호출부)는 애초에
    감싸는 try/except가 전혀 없어 assert가 그대로 프로세스를
    nonzero exit code로 죽이는데, 이는 대시보드 트랙A 헬스체크의 기존
    실패 판정(exit code 기반, 33-4절)과 자연히 맞아떨어짐(추가 배선
    불필요).

### 41-3. 검증
- **정상 케이스 회귀**: 두 스크립트 모두 `import`만으로
  `IMAGE_HEIGHT/IMAGE_WIDTH`가 `camera_config`를 통해 정확히 480/640으로
  해석됨을 실측(`python -c "import run_teleop_real; print(...)"` 등,
  mujoco_env). **`run_inference_mujoco.py --headless --duration 5`를
  실제로 재실행**해 26절/33-1절의 기존 베이스라인과 정확히 일치하는
  결과(물리 스텝 2510회 / policy 호출 150회, action NaN/Inf 없음,
  dtype float32 유지)를 재확인 — 새 assert가 정상 경로에서 아무 영향도
  주지 않음.
- **실패 케이스 격리 검증**: 실제 스크립트를 건드려 재현하는 대신(위험),
  동일한 assert 로직만 분리한 임시 스크립트로 `(480,640,3)` 정상 케이스는
  통과, `(240,320,3)` 불일치 케이스는 정확히
  `AssertionError: wrist_cam render shape (240, 320) != expected (480, 640)`로
  실패함을 확인 후 삭제.

### 41-4. 정규화 죽은 코드: 삭제 대신 유지 + 회귀 테스트 추가로 결정
제안 단계 근거(3절 composition 설계상 `TeamRobotDataset`은 애초에 QA/진단
도구지 학습 파이프라인 컴포넌트가 아님 — `detect_outliers()`/
`compute_custom_stats()`와 같은 범주) 그대로 승인됨. 다만 "테스트 커버리지
0"이라는 약점을 보완하는 조건으로, `pipeline/2_qa/tests/test_team_robot_dataset.py`에
`test_normalize_roundtrip()`(4번째 테스트) 추가:
- `action`/`observation.state`: `normalize()`가 실제로 값을 바꾸는지(no-op
  버그 방지) + `unnormalize(normalize(x))`가 원본을 float32 오차 범위로
  복원하는지
- `observation.images.wrist_cam`: 19-2절 IDENTITY 설계대로 `normalize()`가
  **바이트 단위로 완전한 no-op**인지(`torch.equal`)

**실측 결과(so101_teleop_real, 9483 frames, mujoco_env)**: 기존 3개 테스트
전부 PASS(회귀 없음, `compute_custom_stats` 오버헤드 -98.00%/`__getitem__`
오버헤드 +8.86%, 20% 임계값 이내) + 신규 테스트 PASS —
action/observation.state 라운드트립 `max_err=2.42e-08`(float32 정밀도
범위 내), 이미지 IDENTITY 완전 무손실 확인. `OVERALL: ALL PASS`.

**결론**: `normalize()`/`unnormalize()`는 삭제하지 않고 QA 계층의 진단
유틸리티로 유지 — 39절의 "학습 파이프라인은 이 메서드를 안 쓴다"는
사실 자체는 변하지 않지만(여전히 lerobot-train과는 완전히 독립),
이제 최소한의 자동 회귀 테스트가 붙어 향후 `_NORM_EPS`나 `meta.stats`
접근 방식이 바뀌어도 조용히 깨지지 않음.

## 42. QA 페이지 → "전처리 구조" 상세 뷰 신규 — DONE (2026-08-17)

39/40절 조사 결과를 대시보드에서 직접 확인할 수 있게, `TeamRobotDataset`에
`describe_preprocessing()`을 추가하고 QA(②) 페이지 메타정보 섹션에서
연결되는 별도 상세 페이지를 만들었다. **핵심 요구사항**: 이 페이지가
AGENTS.md 프로즈를 손으로 옮겨 적은 게 아니라, 코드에서 직접 읽어오는
구조여야 함 — 아래처럼 구현.

### 42-1. `TeamRobotDataset.describe_preprocessing()`(신규)
`team_robot_dataset.py` 맨 끝에 추가. 읽기 전용, `self`/파일 어느 쪽도
변경하지 않음. 두 부분을 반환:
- `normalization`: 요약 문구(39절 "결정은 lerobot, 적용은 학습 경로에서
  독립적으로" 내용을 코드에 박아둔 것) + **이 데이터셋 인스턴스의 실제
  `self._base.meta.stats`값**(action/observation.state 각 관절의
  mean/std) — 코드에서 매번 다시 읽으므로 데이터셋이 바뀌면 이 페이지도
  자동으로 최신값을 보여줌(수치를 하드코딩하지 않음).
- `items`: 40-7절 6항목 표와 동일한 6개 딕셔너리 리스트
  (`qa_layer`/`act_policy`/`lerobot_train_cli`/`status`/`status_label`/`note`).
  `status`는 `gap`(이미지 리사이즈만)/`unused_option`(이미지 증강)/
  `not_applicable`(액션 델타, 센서 동기화)/`handled`(액션 청킹)/
  `policy_constraint`(프레임 스태킹) 5가지 중 하나 — UI가 이 값으로
  역할(role)을 정하는 게 아니라(요청대로 "이미지 리사이즈만" 별도
  강조), 페이지 코드가 `status_label`만 그대로 표시.
- 이미지 리사이즈 항목의 `note`는 41절에서 실제로 무슨 방어 코드를
  넣었는지(`pipeline/camera_config.py`, shape assert)까지 포함 — "아직
  안 고쳐진 빈 자리"라는 사실과 "불일치 시 조용히 안 넘어가게는 만들어
  뒀다"는 사실을 둘 다 정확히 전달하도록 문구를 다듬음.

**검증**: 실제 `so101_teleop_real`(9483 frames)로 스모크 테스트 —
`items` 6개, `normalization.samples` 2개(action/observation.state) 정상
반환, action의 `wrist_roll` std=0.0231(24-4절에서 언급된 "다른 관절보다
훨씬 작은 std"와 수치가 일치함을 재확인) 등 실측값이 그대로 나옴.

### 42-2. `dashboard/lib/data_sources.py`: `get_preprocessing_description()`(신규)
기존 `get_qa_results()`와 동일한 패턴(`@st.cache_data`,
`_load_team_robot_dataset()` 재사용) — `describe_preprocessing()`을
그대로 감싸기만 함. `invalidate_dataset_caches()`에도 추가(데이터셋이
갱신되면 이 캐시도 같이 무효화).

### 42-3. `dashboard/pages/5_전처리_구조.py`(신규 페이지)
행동(버튼)이 없는 순수 정보 페이지라 요청대로 행동 카드 없이 정보 카드만
나열(카드 위계 4단계 규칙은 행동이 있는 페이지에만 적용되는 것이므로
예외 아님):
- 정규화 카드(`settings`, 회색) — summary + 실제 stats 표(관절별
  mean/std, `st.dataframe`)
- 6항목 표 카드(`settings`, 회색) — `st.dataframe`으로 항목/QA계층/ACT
  policy/lerobot-train CLI/상태 5열
- 이미지 리사이즈 전용 카드(`warning`, 주황) — 유일하게 강조되는 항목,
  `render_card_header()`의 기존 4-role 팔레트 그대로 재사용(새 색 안 만듦)
- 나머지 5항목 개별 설명은 접힌 `st.expander`로

`dashboard/pages/2_QA_검증.py`의 메타정보 expander 맨 위에
`st.page_link("pages/5_전처리_구조.py", ...)` 추가(링크 텍스트에 6항목
키워드 요약 포함).

`pages/`는 이 프로젝트가 Streamlit 구버전 자동 스캔 방식(명시적
`st.navigation`/`st.Page` API 미사용, 34절에 이미 기록된 `lib/theme.py`의
`STAGE_PAGES` 4개가 전부)을 쓰고 있어, 새 파일을 추가하면 사이드바에도
자동으로 나타남("전처리 구조"로 표시) — `lib/theme.py`의
`STAGE_NAMES`/`STAGE_PAGES`(홈 화면 "파이프라인 흐름" 4단계 화살표 전용)와
`next_action.get_stage_progress()`(둘 다 정확히 4개 고정)는 건드리지
않았으므로 이 신규 페이지가 4단계 파이프라인 흐름에 다섯 번째 단계처럼
끼어드는 일은 없음 — 사이드바에 별도 참고 항목으로만 추가됨.

### 42-4. 검증(실제 브라우저, `dashboard-screenshot` 스킬)
헤드리스 대시보드(포트 8503)를 띄워 실제로 확인:
- `/전처리_구조` 페이지: 정규화 카드에 action/observation.state 각 6개
  관절의 mean/std가 정확한 값으로 렌더링됨, 6항목 표 카드, 이미지
  리사이즈 warning 카드(주황 배경, "빈 자리" 레이블 + note 전문)까지
  전부 스크린샷으로 확인
- QA(②) 페이지: 메타정보 expander를 실제로 펼친 상태에서 새 링크
  ("🔍 이 파이프라인의 전처리 구조 보기 (이미지 리사이즈/증강, 액션
  델타/청킹, 프레임 스태킹, 센서 동기화)")가 "코드 구조" 서브헤더보다
  위에 정상 렌더링됨을 확인

**스킬 사용 중 새로 발견한 제약(기록해둠)**: `st.expander`가 접힌
상태에서는 Playwright의 기본 `getByText()`(visible text 매칭)가 내부
텍스트를 못 찾음(타임아웃) — 35-3절 `page_link`/AppTest 제약과는 다른
종류의, "접힌 콘텐츠는 코드는 실행되지만 DOM에서 안 보이는 텍스트로
취급된다"는 스킬 자체의 한계. `[data-testid="stExpander"] summary`를
먼저 클릭해 펼친 뒤 스크린샷하는 방식으로 우회함(이번엔 1회성 스크립트로
처리, 기존 `dashboard-screenshot` 스킬의 `shot.js`는 안 건드림 — 접힌
expander 안 콘텐츠를 확인해야 하는 경우가 앞으로도 반복되면 그때
스킬에 옵션으로 추가하는 게 나을 듯).

## 43. 사이드바 재구성: `st.navigation()` 섹션 그룹핑 전환 — DONE (2026-08-17)

### 43-1. 기술적 가능 여부 사전 확인(추측 금지, 실제 실행) — 전부 실행 확인
구현 전 별도 테스트 앱(`.tools/nav_test*/`, 검증 후 삭제)으로 3가지를
전부 직접 렌더링·클릭해서 확인했다(설치된 Streamlit 1.61.1 기준):

1. **섹션 그룹핑 지원 여부**: `st.navigation({"": [...], "섹션명": [...]})`
   형태의 dict가 `position="sidebar"`(기본값)에서도 실제로 그룹으로
   묶여 렌더링됨 — 공식 docstring은 "각 그룹이 collapsible item이 되는"
   설명을 `position="top"` 문맥에서만 명시하지만, sidebar에서도 동일하게
   되는지는 실행해서 확인해야 했음(문서에 명시 안 된 부분).
2. **진짜 토글인지**: 스크린샷 3장(펼침→클릭 후 접힘→재클릭 후 재펼침)으로
   확인 — 클릭 시 하위 페이지들이 실제로 DOM에서 사라지고 다시 나타남.
3. **아이콘 자동 구분**: 펼친 상태 라벨 옆 아래꺾쇠(⌄), 접힌 상태
   오른꺾쇠(›) — Streamlit이 네이티브로 렌더링, 직접 그릴 필요 없음.

**추가로 확인**(사용자가 "기본 펼침/접힘" 판단 근거로 요청한 것과
별개로, 구현 전 재확인 요청받아 실행): 펼침/접힘 상태의 지속 범위 —
`newContext()`(=새 세션) 2개로 실측: (a) 세션 A에서 처음 로드 시
기본값은 항상 펼침, (b) 세션 A가 섹션을 접은 뒤 같은 세션 안에서 다른
페이지로 이동해도 접힘 상태가 유지됨(사용자가 접은 걸 존중), (c) 세션
B(새 브라우저 컨텍스트)는 세션 A의 상태와 무관하게 다시 펼침 기본값으로
시작 — splash 화면의 `st.session_state` 방식(37절)과 정확히 같은 성격의
세션 격리. "새로 오는 사람마다 접힌 걸 다시 펴야 하는" 우려는 해당 없음.

### 43-2. 사용자 확정 사항
- 섹션 이름: "모방학습 실행" (그대로)
- 기본 상태: 펼침(43-1절 근거 + "처음 오는 사람이 CTA 링크를 따라갈 때
  마찰 없어야 함" 판단, 사용자 승인)
- `app`/`전처리 구조`는 섹션 밖 최상단(`""` 키)에 항상 노출

### 43-3. 구현: `dashboard/app.py`가 `pages/` 자동 스캔 → 명시적 라우터로 전환
- 기존 홈 페이지 본문 전체(스플래시 체크, 브랜드 헤더, CTA 카드, 파이프라인
  흐름, 환경 상태, 헬스 요약)를 `def home() -> None:` 함수로 감쌈 — 로직
  자체는 한 줄도 안 바뀜, 들여쓰기만 변경.
- 파일 끝에 라우터 추가:
  ```python
  pages = {
      "": [
          st.Page(home, title="app", default=True),
          st.Page("pages/5_전처리_구조.py", title="전처리 구조"),
      ],
      "모방학습 실행": [st.Page(STAGE_PAGES[i], title=STAGE_NAMES[i]) for i in range(len(STAGE_NAMES))],
  }
  pg = st.navigation(pages)
  pg.run()
  ```
  `lib/theme.py`의 `STAGE_PAGES`/`STAGE_NAMES`(기존 파이프라인 흐름 화살표 UI가
  이미 쓰던 것)를 그대로 재사용 — 페이지 목록을 두 곳에 따로 유지하지 않음.
- **`pages/1~5_*.py`는 전혀 수정하지 않음**: 각 파일이 자기 자신의
  `st.set_page_config()`를 계속 호출하는 게 `st.navigation()` 하에서도
  안전한지 실행 테스트로 먼저 확인(엔트리 파일에 별도의 전역
  `st.set_page_config()` 호출 없이도, `st.Page(callable_or_file, ...)`로
  등록된 각 page가 실행될 때 자기 자신의 `set_page_config()`를 호출하면
  예외 없이 정상 동작 + 브라우저 탭 제목도 페이지별로 정확히 바뀜을 확인)
  — 그래서 `app.py`에는 전역 `set_page_config()`를 아예 안 둠(`home()` 안에
  있는 것 하나로 충분).
- 기존 `st.page_link()` 호출 전부(app.py의 CTA/파이프라인 흐름, 페이지
  1→2/2→3/3→4/5→2 "다음 단계" 링크)는 **문자열 그대로 안 건드림** —
  전부 `pages/N_*.py` 파일 경로를 가리키고, 홈(callable 기반)을 가리키는
  링크는 프로젝트 전체에 하나도 없음을 미리 grep으로 확인(콜러블 페이지로
  링크하려면 `Page` 객체가 필요하다는 게 `st.page_link` 공식 제약이라,
  이 사실을 먼저 확인 안 했으면 별도 처리가 필요했을 수 있었음).

### 43-4. 구현 후 검증(사용자가 명시적으로 요구한 3가지, 전부 실행 확인)

**① AppTest 5개 페이지 + 라우터 재확인 — 크래시 없음, 방법론 자체를 개선**:
기존 방식(`AppTest.from_file(개별_페이지_파일).run()`, 페이지를 완전히
격리 실행)으로는 `page_link()`가 있는 페이지(1, 2, 5)에서 여전히
`'url_pathname'` KeyError가 재현됨 — 이건 이번 마이그레이션 때문에 새로
생긴 문제가 아니라 35-3절에서 이미 확인된 것과 동일한, "격리 실행에는
페이지 레지스트리가 없다"는 기존 AppTest 한계의 재현(회귀 아님).

**여기서 실제로 개선한 부분**: `AppTest.switch_page(page_path)`라는
API가 있다는 걸 이번에 처음 활용 — `AppTest.from_file("dashboard/app.py").run()`으로
먼저 라우터를 통해 앱을 띄우면(이 시점에 `st.navigation()`이 실행되어
전체 페이지 레지스트리가 만들어짐) 그 뒤 `.switch_page("pages/2_QA_검증.py").run()`
같은 식으로 실제 사용자 내비게이션처럼 페이지를 전환할 수 있고, 이
경로로는 **`page_link()`가 있는 페이지에서도 크래시가 사라짐**(home +
5개 페이지 전부 `switch_page()`로 순회, 예외 0건 — QA 페이지는 최초
1회 120초 타임아웃이 났으나 재실행 시 정상 완료돼 콜드스타트 지연이지
실패가 아님을 확인). **결론**: 이번 마이그레이션이 오히려 이전에는
AppTest로 검증 못 했던 부분(page_link가 있는 페이지의 정상 동작)을
`switch_page()` 경로로 검증 가능하게 만든 셈 — 순수 회귀 없음을 넘어
테스트 방법론 자체가 개선됨. (재사용 가능한 테스트 파일로 남기지는
않음 — 이 프로젝트의 AppTest 검증은 지금까지 전부 1회성 스크립트 작성 →
실행 → 결과를 AGENTS.md에 기록 → 스크립트 삭제 패턴을 따름, 이번도 동일.)

**② `dashboard-screenshot` 스킬 호환성 — 무수정으로 정상 동작**:
기존 `shot.js`(사이드바 링크 텍스트를 `getByText(exact:true)`로 찾아
클릭하는 방식, 수정 안 함)로 "데이터 수집"(= "모방학습 실행" 섹션 안에
중첩된 페이지)을 정상적으로 찾아 클릭·스크린샷까지 완료 — 43-1절에서
확인한 "기본값이 항상 펼침"이라는 사실 덕분에 섹션을 먼저 펼치는 별도
로직이 스킬에 전혀 필요 없었음(만약 기본값이 접힘이었다면 스킬 수정이
필요했을 것 — 43-2절의 "기본 펼침" 결정이 이 호환성에도 실질적으로
기여함).

**③ 다음 단계 링크 체인(①→②→③→④) — 실제 클릭으로 확인**:
- "① 데이터 수집" 페이지의 "다음: ② QA 검증하기" 링크를 실제로 클릭 →
  `h1`이 정확히 "② QA & 검증"으로 바뀜을 확인
- "② QA 검증" 페이지의 "다음: ③ 학습 시작하기 →" 링크를 실제로 클릭 →
  `h1`이 정확히 "③ ACT 학습"으로 바뀜을 확인
- "③→④"(완료된 학습 run이 있어야 뜨는 조건부 링크)는 현재 state의
  학습 run이 `status: "running"`이라 이번엔 링크 자체가 안 뜬 상태였음
  — 이건 마이그레이션과 무관한 기존 조건부 로직이 그대로 작동한 것(버그
  아님, 검증 대상에서 자연 제외됨)

**검증 중 겪은 사소한 함정(기록해둠)**: Playwright의 `getByText(text,
{exact:false})`가 `st.page_link()`의 라벨(아이콘+텍스트가 여러 자식
노드로 나뉘어 렌더링됨)을 못 찾어 처음엔 "링크가 없다"고 오판할 뻔함 —
`page.locator('body').innerText()`로 전체 텍스트에는 포함돼 있음을
먼저 확인한 뒤, `page.locator('a', { hasText: ... })`(Playwright의 텍스트
엔진, 노드 경계를 넘는 부분 문자열 매칭 + 실제 actionability 체크)로
바꾸자 정상적으로 클릭까지 성공 — `dashboard-screenshot` 스킬의 기존
`shot.js`가 쓰는 `getByText().scrollIntoViewIfNeeded()` 패턴도 잠재적으로
같은 함정에 걸릴 수 있다는 걸 시사하지만, 이번엔 스킬 자체를 고치지
않음(스킬은 "스크롤 대상 텍스트"용이지 "클릭 대상"용으로 설계된 적이
없어서 지금 당장 문제가 된 적은 없음 — 필요해지면 그때 반영).

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
