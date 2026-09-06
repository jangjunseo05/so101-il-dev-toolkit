"""4단계: 추론. run_inference_mujoco.py를 두 트랙으로 웹에서 트리거한다.

CLAUDE.md 33절 조사 결과: 이 스크립트는 두 가지 완전히 다른 실행 모드를
갖는다 --
- `--headless --duration N`: 뷰어 없이 N초 시뮬레이션만 최대한 빨리 돌리고
  종료(실측: 5초 분량이 벽시계 11초). 시각 피드백/성공 판정 없음, 순수
  파이프라인 헬스체크. 동기 처리 가능한 짧은 작업이라 트랙 A로 구현.
- 기본(뷰어) 모드: `mujoco.viewer.launch_passive()`로 뷰어 창이 뜨고
  사용자가 창을 닫을 때까지 무기한 루프(episode 경계/성공 판정 없음) --
  구조적으로 ①(데이터 수집)의 무기한 세션과 동일해서 트랙 B는 그 패턴
  (process_manager.py + resources 태그)을 그대로 재사용한다.

성공/실패를 정량적으로 판정하는 로직(예: 물체를 실제로 집었는지)은 이번
스코프 밖이다 -- 33절 참고, 나중에 필요해지면 headless 모드에 episode
종료 조건 + 성공 기준을 추가하는 방향(옵션 2)으로 확장 가능.

정보 위계(2026-08-17, 직관 UX 개선): 트랙 A/B는 블로킹 조건이 비대칭
(트랙 A는 자원 충돌 + 트랙 B 실행 중일 때도 막힘, 트랙 B는 자원 충돌만)
이라 outer action 카드 헤더는 항상 "action"(파랑) 고정 -- 페이지 1/3과
달리 조건부 전환하지 않고, 각 탭이 자기 블로킹 사유만 캡션으로 표시한다.
트랙 A의 매번 다른 통과/실패 결과는 카드로 감싸지 않고 기존처럼
st.success/error 한 줄로 유지(페이지 하단의 "헤드리스 검증 결과(이전
세션 기록)" 정적 카드와 혼동되지 않도록 카드화하지 않기로 결정).
"이미 벌어진 사건"(크래시)은 트랙 B에만 있는 개념(트랙 A는 동기 실행이라
세션 간 크래시 추적이 없음) -- action 카드보다 위, warning role.
"""

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib import dataset_registry
from lib import process_manager as pm
from lib import training_run as tr
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header

# pipeline/camera_config.py의 스트리밍 포트 상수 재사용(§44) -- 1_데이터_수집.py와
# 동일 패턴(data_sources.py가 이미 pipeline/2_qa를 sys.path에 넣는 것도 같은 방식).
sys.path.insert(0, str(ds.PROJECT_ROOT / "pipeline"))
from camera_config import INFERENCE_STREAM_PORT  # noqa: E402

st.set_page_config(page_title="④ 추론", page_icon=get_logo_icon(), layout="wide")

render_brand_header()

# 데이터셋 선택 -- ②/③ 페이지와 st.session_state로 선택을 공유한다. 데이터셋마다
# 추론 스크립트/씬(worldbody)/카메라 구성이 완전히 다르므로(1카메라 640x480 vs
# 2카메라 320x240) 체크포인트뿐 아니라 실행할 스크립트 자체가 바뀐다.
_dataset_options = dataset_registry.dataset_options()
st.session_state.setdefault("selected_dataset_id", dataset_registry.DEFAULT_DATASET_ID)
_current_id = st.session_state["selected_dataset_id"]
_ds_labels = list(_dataset_options.keys())
_current_label = next(label for label, did in _dataset_options.items() if did == _current_id)
_selected_label = st.selectbox("데이터셋", options=_ds_labels, index=_ds_labels.index(_current_label))
dataset_id = _dataset_options[_selected_label]
st.session_state["selected_dataset_id"] = dataset_id
profile = dataset_registry.get_profile(dataset_id)

if dataset_id == "bottle_pick_pour":
    st.warning(
        "이 데이터셋은 실제 웹캠 사진으로 학습됐습니다 — MuJoCo 렌더(합성 이미지)를 입력으로 받으면 "
        "학습 때와 이미지 도메인이 완전히 달라집니다. 여기서 확인하는 건 '집기/따르기를 성공하는지'가 "
        "아니라 '파이프라인이 에러 없이 돌고 액션 값이 정상 범위인지'뿐입니다(스모크테스트).",
        icon="⚠️",
    )

STAGE = "inference"
INFERENCE_SCRIPT = profile["inference_script"]
DEFAULT_CHECKPOINT = profile["default_checkpoint"]


def _resolve_checkpoint_dir(output_dir: Path) -> Path | None:
    """완료된 학습 run의 output_dir에서 추론에 쓸 pretrained_model 디렉터리를
    찾는다. `checkpoints/last`(심볼릭 링크) 우선, 없으면 숫자 디렉터리 중
    최대 step으로 폴백 -- pages/3_ACT_학습.py의 `_resolve_latest_checkpoint_config()`와
    같은 폴백 패턴이지만, 반환값이 `train_config.json`이 아니라
    `run_inference_mujoco.py --checkpoint`가 기대하는 `pretrained_model`
    디렉터리 자체라는 점이 다르다.
    """
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    last_link = checkpoints_dir / "last"
    if last_link.is_dir():
        pretrained_dir = last_link / "pretrained_model"
        if (pretrained_dir / "model.safetensors").is_file():
            return pretrained_dir
    numbered = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not numbered:
        return None
    latest = max(numbered, key=lambda p: int(p.name))
    pretrained_dir = latest / "pretrained_model"
    return pretrained_dir if (pretrained_dir / "model.safetensors").is_file() else None


def _checkpoint_options() -> dict[str, str]:
    """드롭다운에 쓸 {표시 라벨: 체크포인트 디렉터리 경로} -- ③에서 완료된
    학습 run들을(선택된 dataset_id로 필터링) 우선 보여주고, 완료된 run이
    하나도 없으면 이 데이터셋의 기본 체크포인트로 폴백한다.

    dataset_id로 필터링하는 이유: 병/블록 체크포인트는 서로 다른 카메라
    피처 키·해상도를 기대해서 잘못 고르면 추론 스크립트가 곧바로 에러를
    낸다(옆 페이지의 데이터셋 선택과 항상 일치시켜야 함).
    """
    options = {}
    for run in tr.list_completed_runs(dataset_id=dataset_id):
        ckpt_dir = _resolve_checkpoint_dir(Path(run["output_dir"]))
        if ckpt_dir is not None:
            step = run.get("last_checkpoint_step", "?")
            options[f"{run['run_id']} (step {step})"] = str(ckpt_dir)
    if DEFAULT_CHECKPOINT.is_dir():
        options[f"기본 체크포인트 ({DEFAULT_CHECKPOINT.parent.parent.name}/{DEFAULT_CHECKPOINT.parent.name})"] = str(DEFAULT_CHECKPOINT)
    return options


def _judge_healthcheck_output(output: str, returncode: int) -> tuple[bool, list[str]]:
    """`run_inference_mujoco.py --headless`의 출력에서 통과/실패를 판정한다.
    `[run_inference_mujoco] action에 NaN 포함: {bool}, Inf 포함: {bool}` 형식
    문자열 매칭 -- 정교한 파싱 아님, 이 스크립트가 실제로 찍는 문구 그대로.
    """
    has_nan = "NaN 포함: True" in output
    has_inf = "Inf 포함: True" in output
    exit_ok = returncode == 0
    reasons = []
    if not exit_ok:
        reasons.append(f"exit code {returncode}")
    if has_nan:
        reasons.append("NaN 포함")
    if has_inf:
        reasons.append("Inf 포함")
    return exit_ok and not has_nan and not has_inf, reasons


job_status = pm.get_job_status(STAGE)

_was_running_key = f"{STAGE}_was_running"
_last_log_key = f"{STAGE}_last_log_path"
_crash_key = f"{STAGE}_crash_log_tail"

if job_status["running"] and job_status.get("log_path"):
    st.session_state[_last_log_key] = job_status["log_path"]

# job이 이전 rerun에서는 실행 중이었는데 지금은 아니면(=방금 세션이 끝났으면)
# 크래시 여부를 확인해둔다 -- ①(데이터 수집)과 완전히 동일한 패턴
# (CLAUDE.md 30-4절 looks_like_crash()). 이 개념은 트랙 B(세션형) 전용 --
# 트랙 A는 동기 실행이라 세션 간 크래시 추적이 없다.
if st.session_state.get(_was_running_key, False) and not job_status["running"]:
    last_log_path = st.session_state.get(_last_log_key)
    if last_log_path:
        tail = pm.tail_log(last_log_path, n_lines=60)
        if pm.looks_like_crash(tail):
            st.session_state[_crash_key] = tail
st.session_state[_was_running_key] = job_status["running"]

st.title("④ 추론")
st.caption("학습된 ACT 체크포인트로 MuJoCo에서 자율 pick&place 시도 (`run_inference_mujoco.py`)")

# "이미 벌어진 사건"(크래시, 트랙 B 전용) -- action 카드보다 위
_crash_tail = st.session_state.get(_crash_key) if not job_status["running"] else None
if _crash_tail:
    with st.container(border=True):
        render_card_header("❌", "이전 시뮬레이션 관찰 세션 크래시", "warning")
        st.write("세션이 예기치 않게 종료됐습니다 — 로그를 확인하세요.")
        with st.expander("마지막 로그 보기", expanded=True):
            st.code(_crash_tail, language=None)

checkpoint_options = _checkpoint_options()

# 탭 내부에서 행동(버튼)이 설정(위젯)보다 먼저 오도록 재배치하므로, 버튼이
# 그 아래 위젯의 최신 입력값을 읽을 수 있게 session_state를 미리 seed한다
# (③ 학습 페이지와 동일한 패턴).
_option_labels = list(checkpoint_options.keys())
if _option_labels:
    if st.session_state.get("healthcheck_ckpt") not in _option_labels:
        st.session_state["healthcheck_ckpt"] = _option_labels[0]
    if st.session_state.get("sim_ckpt") not in _option_labels:
        st.session_state["sim_ckpt"] = _option_labels[0]
st.session_state.setdefault("healthcheck_duration", 5.0)

# 1. 행동(action) -- 트랙 A/B는 블로킹 조건이 비대칭이라 outer 헤더는
# 항상 action 고정(추론 위계 재정리 결정 사항), 각 탭이 자기 블로킹만 표시.
with st.container(border=True):
    render_card_header("", "추론 세션 제어", "action")

    st.info(
        "트랙 A는 NaN/Inf·에러 유무만, 트랙 B는 사람이 직접 뷰어로 관찰하는 용도입니다. ",
        icon="ℹ️",
    )

    if not checkpoint_options:
        st.warning("사용 가능한 체크포인트가 없습니다 — ③에서 학습을 완료하거나 `checkpoints/010000`이 있어야 합니다.", icon="⚠️")

    tab_a, tab_b = st.tabs(["⚡ A:빠른 헬스체크", "🖥️ B:시뮬레이션 관찰"])

    with tab_a:
        if checkpoint_options:
            # check_conflict()는 "다른" stage와의 충돌만 보므로(자기 자신은 검사에서
            # 제외됨), 트랙 B(같은 inference stage)가 이미 실행 중인 경우까지
            # 막으려면 job_status["running"]도 같이 봐야 한다 -- 안 그러면 트랙 B가
            # GPU/뷰어를 쓰고 있는 동안 트랙 A에서 또 다른 GPU 프로세스를 락 없이
            # 띄워버릴 수 있음(실제 AppTest로 재현해서 발견, CLAUDE.md 33절 참고).
            _conflict_a = pm.check_conflict(STAGE)
            _blocked_by_self = job_status["running"]
            _a_blocked = _conflict_a is not None or _blocked_by_self

            # 행동 먼저 -- session_state["healthcheck_ckpt"/"healthcheck_duration"]은 위에서 이미 seed됨
            if st.button("⚡ 빠른 헬스체크 실행", disabled=_a_blocked, type="primary"):
                healthcheck_ckpt = checkpoint_options[st.session_state["healthcheck_ckpt"]]
                duration = st.session_state["healthcheck_duration"]
                cmd = [
                    sys.executable, str(INFERENCE_SCRIPT),
                    "--checkpoint", healthcheck_ckpt,
                    "--headless",
                    "--duration", str(duration),
                ]
                with st.spinner(f"헤드리스 실행 중... ({duration:.0f}초 분량, 로딩 포함 최대 1분 정도)"):
                    t0 = time.time()
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            cwd=str(ds.PROJECT_ROOT),
                            timeout=max(120.0, duration + 60.0),
                        )
                        elapsed = time.time() - t0
                        output = result.stdout + result.stderr
                        ok, reasons = _judge_healthcheck_output(output, result.returncode)

                        if ok:
                            st.success(f"✅ 통과 — exit code 0, NaN/Inf 없음 ({elapsed:.1f}초 소요)", icon="✅")
                        else:
                            st.error(f"❌ 실패 — {', '.join(reasons)} ({elapsed:.1f}초 소요)", icon="❌")
                        with st.expander("전체 출력 보기"):
                            st.code(output, language=None)
                    except subprocess.TimeoutExpired as e:
                        st.error(f"❌ 시간 초과({e.timeout:.0f}초) — 응답이 없습니다.", icon="❌")

            # 이 탭만의 블로킹 사유 (탭별 독립 -- 트랙 B와 공유하지 않음)
            if _conflict_a is not None:
                st.caption(f"⚠️ '{_conflict_a}'가 겹치는 자원(GPU)을 사용 중이라 지금은 실행할 수 없습니다.")
            elif _blocked_by_self:
                st.caption("⚠️ 트랙 B(시뮬레이션 관찰) 세션이 이미 실행 중이라 지금은 실행할 수 없습니다.")

            # 설정
            st.selectbox("체크포인트", options=_option_labels, key="healthcheck_ckpt")
            st.number_input("시뮬레이션 시간(초)", min_value=1.0, max_value=60.0, step=1.0, key="healthcheck_duration")
            st.caption(
                "체크포인트가 최소한 안 깨졌는지, NaN/Inf 없이 유한한 액션이 나오는지 짧게 확인합니다. "
                "뷰어가 없어 시각적 피드백은 없고(로딩 포함 실측 약 11초/5초 분량), 응답이 올 때까지 페이지가 멈춰 있습니다."
            )

    with tab_b:
        if job_status["running"]:
            elapsed = time.time() - job_status["started_at"]
            c1, c2, c3 = st.columns(3)
            c1.metric("상태", "🖥️ 실행 중")
            c2.metric("PID", job_status["pid"])
            c3.metric("경과 시간", f"{int(elapsed // 60)}분 {int(elapsed % 60)}초")

            # 브라우저 내 라이브 스트리밍(§44) -- ①(데이터 수집)과 동일 패턴.
            # R 키 episode 리셋은 여전히 네이티브 뷰어 창에서만 가능하다.
            st.markdown("**📺 실시간 화면 (3인칭 뷰)**")
            st.caption(
                "MuJoCo 뷰어 창과는 별개로, 브라우저에서도 씬을 볼 수 있습니다(약 15fps). "
                "R 키 episode 리셋은 여전히 뷰어 창에서 해야 합니다. 몇 초가 지나도 안 뜨면 새로고침하세요."
            )
            st.markdown(
                f'<img src="http://127.0.0.1:{INFERENCE_STREAM_PORT}/stream" '
                f'style="width:100%;max-width:640px;border-radius:4px;" />',
                unsafe_allow_html=True,
            )

            st_autorefresh(interval=2000, key="inference_log_autorefresh")
            log_text = pm.tail_log(job_status["log_path"], n_lines=200)
            st.text_area("실시간 로그 (2초마다 자동 갱신)", value=log_text, height=300, key="inference_log_area")

            st.markdown("**⏹ 세션 종료**")
            st.caption("이 버튼은 세션을 즉시 종료합니다 — 추론은 저장하는 데이터가 없어 안전합니다.")
            if st.button("⏹ 세션 종료", type="primary"):
                pm.stop_job(STAGE, graceful=False)
                st.success("세션을 종료했습니다.")
                st.rerun()
        else:
            if checkpoint_options:
                _conflict_b = pm.check_conflict(STAGE)

                # 행동 먼저 -- session_state["sim_ckpt"]는 위에서 이미 seed됨
                if st.button("🖥️ 시뮬레이션 관찰 시작", disabled=(_conflict_b is not None), type="primary"):
                    sim_ckpt = checkpoint_options[st.session_state["sim_ckpt"]]
                    cmd = [
                        sys.executable, str(INFERENCE_SCRIPT),
                        "--checkpoint", sim_ckpt,
                    ]
                    try:
                        pm.start_job(STAGE, cmd)
                        st.session_state.pop(_crash_key, None)
                        st.success("세션을 시작했습니다. 잠시 후 MuJoCo 뷰어 창이 뜹니다.")
                        st.rerun()
                    except pm.JobConflictError as e:
                        st.error(str(e))

                # 이 탭만의 블로킹 사유
                if _conflict_b is not None:
                    st.caption(f"⚠️ '{_conflict_b}'가 겹치는 자원(GPU/MuJoCo 뷰어)을 사용 중이라 지금은 시작할 수 없습니다.")

                # 설정
                st.selectbox("체크포인트", options=_option_labels, key="sim_ckpt")
                st.caption(
                    "MuJoCo 뷰어 창(+카메라 미리보기)이 로컬에 뜨고, policy가 자율로 움직이는 걸 직접 관찰합니다. "
                    "R 키: episode 리셋. 뷰어 창을 닫거나 위 버튼으로 세션을 끝낼 수 있습니다."
                )

if dataset_id == "block_pickplace":
    # 2. 결과(results) -- 정적/과거 기록, 트랙 A의 매번 다른 결과와 혼동되지 않도록
    # 제목에 "이전 세션 기록"을 명시(트랙 A 결과는 카드화하지 않고 인라인 유지)
    with st.container(border=True):
        render_card_header("📋", "헤드리스 검증 결과 (이전 세션 기록)", "success")
        st.caption("`python run_inference_mujoco.py --headless --duration 5` 실행 결과 — 위 트랙 A로 언제든 재현 가능")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("물리 스텝", "2510회", "5초 분량")
        c2.metric("policy 호출", "150회", "30Hz × 5초")
        c3.metric("action dtype", "float32", "끝까지 유지")
        c4.metric("NaN/Inf", "0건", "전부 유한값")

    # 3. 메타정보(접힘) -- 코드 구조 + 실행 파라미터
    with st.expander("(자세히 보기) 코드 구조 · 실행 파라미터 "):
        st.subheader("코드 구조 — 관측치 → 액션 루프")
        st.markdown(
            """
`leader.get_action()` 자리를 `ACTPolicy.select_action()`으로 교체한 구조입니다:

```text
MuJoCo에서 observation(qpos, wrist_cam 렌더) 읽기
  → {"observation.state": tensor, "observation.images.wrist_cam": tensor}
  → preprocessor(obs)          # 체크포인트의 정규화 설정 자동 적용 (VISUAL=IDENTITY 포함)
  → policy.select_action(obs)  # action chunking 내부에서 자동 처리 (100스텝 중 매 100번째 호출만 재추론)
  → postprocessor(action)
  → MuJoCo data.ctrl에 적용 (clip 후 float32로 재캐스팅 — dtype 승격 버그 방지)
```
"""
        )

        st.subheader("실행 파라미터")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("POLICY_HZ", "30", help="카메라 렌더 + policy 호출 주파수 (학습 데이터셋 fps와 동일)")
        p2.metric("CONTROL_HZ", "50", help="MuJoCo 물리 스텝 주파수")
        p3.metric("chunk_size", "100")
        p4.metric("n_action_steps", "100")
else:
    with st.expander("(자세히 보기) 코드 구조"):
        st.markdown(
            """
`run_inference_mujoco_bottle.py`는 카메라가 2대(front+wrist, 320x240)라는 점만
다르고 나머지 루프는 블록 데이터셋용 스크립트와 동일합니다:

```text
MuJoCo에서 observation(qpos, front_cam+wrist_cam 렌더) 읽기
  → {"observation.state": tensor,
     "observation.images.front": tensor, "observation.images.wrist": tensor}
  → preprocessor(obs)          # 체크포인트의 정규화 설정 자동 적용 (VISUAL=MEAN_STD)
  → policy.select_action(obs)  # action chunking 내부에서 자동 처리
  → postprocessor(action)
  → MuJoCo data.ctrl에 적용
```

⚠️ 학습 이미지는 실제 웹캠 사진, 추론 이미지는 MuJoCo 합성 렌더라 도메인이
다릅니다 — action이 유한값으로 나오는지(파이프라인 정상 동작)만 보장하고,
실제로 병을 집거나 따르는 데 성공하는 것은 기대하지 않습니다.
"""
        )
