"""3단계: ACT 학습. lerobot-train CLI를 우리가 만든 YAML 설정으로 구동한
결과(train_main_run_config.yaml, checkpoint_sweep_results.json)를 보여주고,
CLAUDE.md 31절 조사 결과를 바탕으로 새 학습 시작/이어서 학습을 웹에서
직접 트리거한다. 추론(④)은 GPU 자원이 겹쳐 process_manager의
resources=["gpu"]로 자동 배제되므로 동시 실행되지 않는다.

정보 위계(2026-08-16, 직관 UX 개선): "이미 벌어진 사건"(크래시/중단/정상
완료 -- 셋 다) 카드가 action 카드보다 위, role은 결과에 따라 warning
(크래시/중단) 또는 success(정상 완료)로 전환된다. action 카드는 탭
구조(새 학습 시작/이어서 학습)를 그대로 유지하되, 각 탭 내부는
행동(버튼) → 설정(입력) 순서로 재배치했다 -- 버튼이 그 아래 위젯의
최신 입력값을 읽어야 하므로 session_state key 바인딩을 씀(QA 페이지에서
먼저 겪은 문제, 페이지 3은 처음부터 이 패턴으로 구현). action 카드
아래는 학습 결과(success 카드, loss 차트+체크포인트 목록), 맨 아래는
메타정보(접힘: 정적 yaml 설정값 + PNG).
"""

import re
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib import process_manager as pm
from lib import qa_snapshot
from lib import training_run as tr
from lib.theme import RATIO_COLOR, TRAIN_COLOR, VAL_COLOR
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header

st.set_page_config(page_title="③ ACT 학습", page_icon=get_logo_icon(), layout="wide")

render_brand_header()

STAGE = "training"
BASE_TRAIN_CONFIG = ds.PROJECT_ROOT / "config" / "train_main_run_config.yaml"

# tqdm 진행률 라인(예: "Training:  45%|####      | 4500/10000 [02:15<02:45,  3.32it/s]")
# 에서 "4500/10000 ["만 뽑는다. 정교한 파싱이 아니라 실패하면 그냥 None을
# 반환해 raw 로그만 보여주는 fallback으로 감(31절 이후 설계 원칙).
_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[")


def _extract_progress(log_text: str) -> tuple[int, int] | None:
    matches = _PROGRESS_RE.findall(log_text)
    if not matches:
        return None
    current, total = matches[-1]
    return int(current), int(total)


def _resolve_latest_checkpoint_config(output_dir: Path) -> Path | None:
    """output_dir/checkpoints/last(심볼릭 링크)를 우선 쓰되, 없거나 깨져
    있으면(Windows 심볼릭 링크 권한 문제 -- 25절에서 실제로 겪음) checkpoints/
    아래 숫자 이름 디렉터리 중 가장 큰 step을 직접 찾는다.
    """
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    last_link = checkpoints_dir / "last"
    if last_link.is_dir():
        config_path = last_link / "pretrained_model" / "train_config.json"
        if config_path.is_file():
            return config_path
    numbered = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not numbered:
        return None
    latest = max(numbered, key=lambda p: int(p.name))
    config_path = latest / "pretrained_model" / "train_config.json"
    return config_path if config_path.is_file() else None


job_status = pm.get_job_status(STAGE)

_was_running_key = f"{STAGE}_was_running"
_last_log_key = f"{STAGE}_last_log_path"
_run_id_key = f"{STAGE}_current_run_id"
_crash_key = f"{STAGE}_crash_log_tail"
_finalize_msg_key = f"{STAGE}_finalize_message"

if job_status["running"] and job_status.get("log_path"):
    st.session_state[_last_log_key] = job_status["log_path"]
    _run_id = job_status.get("run_id")
    if _run_id:
        st.session_state[_run_id_key] = _run_id
        tr.update_last_checkpoint_step(_run_id, pm.tail_log(job_status["log_path"], n_lines=500))

# job이 이전 rerun에서는 실행 중이었는데 지금은 아니면(=방금 학습이 끝났으면)
# training_run 기록을 completed/interrupted로 확정한다. get_job_status()는
# 정상 종료와 크래시/강제종료를 구분하지 않으므로(30-4절과 동일한 한계),
# training_run.finalize_run()의 "End of training" 로그 마커 매칭 + 기존
# looks_like_crash() 휴리스틱을 함께 써서 세 가지 경우를 구분해 보여준다.
if st.session_state.get(_was_running_key, False) and not job_status["running"]:
    _last_log_path = st.session_state.get(_last_log_key)
    _run_id = st.session_state.get(_run_id_key)
    _log_text = pm.tail_log(_last_log_path, n_lines=500) if _last_log_path else ""
    if _run_id:
        # 사용자가 실행 중에 이 페이지를 한 번도 안 봤다면(자동갱신이 한
        # 번도 안 돌아서) last_checkpoint_step이 아직 안 채워졌을 수 있다 --
        # finalize 시점에 이미 들고 있는 로그로 한 번 더 시도해 채운다.
        tr.update_last_checkpoint_step(_run_id, _log_text)
        _final_status = tr.finalize_run(_run_id, _log_text)
        if _final_status == "interrupted" and pm.looks_like_crash(_log_text):
            st.session_state[_crash_key] = _log_text[-4000:]
            st.session_state[_finalize_msg_key] = ("crash", None)
        elif _final_status == "interrupted":
            _step = (tr.get_run(_run_id) or {}).get("last_checkpoint_step")
            st.session_state[_finalize_msg_key] = ("interrupted", _step)
        else:
            st.session_state[_finalize_msg_key] = ("completed", None)
    st.session_state[_run_id_key] = None
st.session_state[_was_running_key] = job_status["running"]

st.title("③ ACT 학습")
st.caption("`lerobot-train` CLI + `train_main_run_config.yaml` (GPU: RTX 5050, cu128 torch)")

# "이미 벌어진 사건" -- 크래시/중단/정상완료 셋 다 action 카드보다 위,
# role만 다름(경고=warning, 정상완료=success). 실행 중일 땐 해당 없음.
_finalize_msg = st.session_state.get(_finalize_msg_key) if not job_status["running"] else None
if _finalize_msg:
    _kind, _extra = _finalize_msg
    if _kind == "crash":
        with st.container(border=True):
            render_card_header("❌", "학습 크래시", "warning")
            st.write("학습이 예기치 않게 종료됐습니다(크래시로 추정) — 로그를 확인하세요.")
            with st.expander("마지막 로그 보기", expanded=True):
                st.code(st.session_state.get(_crash_key, ""), language=None)
    elif _kind == "interrupted":
        with st.container(border=True):
            render_card_header("⏸️", "학습 중단됨", "warning")
            st.write(f"마지막 체크포인트: step {_extra} — 아래 '이어서 학습'에서 재개할 수 있습니다.")
    else:
        with st.container(border=True):
            render_card_header("✅", "학습 완료", "success")
            st.write("마지막 학습이 정상적으로 완료됐습니다.")
            st.page_link("pages/4_추론.py", label="다음: ④ 추론으로 확인하기 →", icon="➡️")

# 자원 충돌은 STAGE 단위(양쪽 탭 공통)라 한 번만 계산 -- action 카드 헤더
# 색과 두 탭의 버튼 disabled 상태 모두 여기서 파생된다.
conflict = None if job_status["running"] else pm.check_conflict(STAGE)
action_blocked = (not job_status["running"]) and conflict is not None

# 탭 내부에서 "행동(버튼)이 설정(위젯)보다 먼저" 오도록 재배치하면서, 버튼이
# 그 아래 위젯의 최신 입력값을 읽어야 하는 문제가 생긴다(QA 페이지에서 먼저
# 발견됨) -- 위젯을 렌더하기 전에 session_state 기본값을 미리 심어두고
# key 바인딩으로 읽는 방식으로 처음부터 이렇게 구현한다.
st.session_state.setdefault("training_new_job_name", "main_run")
_snapshots = qa_snapshot.list_snapshots()
_latest_snapshot = _snapshots[0] if _snapshots else None
_interrupted_runs = tr.list_interrupted_runs()
_resume_options: dict[str, dict] = {}
_resume_labels: list[str] = []
if _interrupted_runs:
    _resume_options = {
        f"{r['run_id']} (마지막 체크포인트: step {r.get('last_checkpoint_step')})": r for r in _interrupted_runs
    }
    _resume_labels = list(_resume_options.keys())
    # 이전 rerun에서 골랐던 라벨이 이번 rerun의 옵션 목록에 없으면(예: 목록이
    # 바뀜) selectbox가 에러를 내므로, 없으면 첫 번째 옵션으로 되돌린다.
    if st.session_state.get("training_resume_select") not in _resume_labels:
        st.session_state["training_resume_select"] = _resume_labels[0]

# 1. 행동(action) -- 학습 세션 제어
with st.container(border=True):
    render_card_header("🧠", "학습 세션 제어", "warning" if action_blocked else "action")
    if action_blocked:
        st.caption(f"⚠️ '{conflict}'가 GPU를 사용 중이라 지금은 시작할 수 없습니다.")

    st.info(
        "학습은 GPU를 점유합니다 — '④ 추론'과 동시에 실행할 수 없습니다\n\n"
        "W&B 로깅은 이미 설정에 켜져 있습니다— 로그는 로컬 `wandb/` 폴더에 저장되며, "
        "온라인 업로드가 필요하면 나중에 `wandb sync <경로>`를 직접 실행하세요.",
        icon="🧠",
    )

    if job_status["running"]:
        _elapsed = time.time() - job_status["started_at"]
        _run_id = job_status.get("run_id", "?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("상태", "🧠 학습 중")
        c2.metric("run_id", _run_id)
        c3.metric("PID", job_status["pid"])
        c4.metric("경과 시간", f"{int(_elapsed // 60)}분 {int(_elapsed % 60)}초")

        st_autorefresh(interval=2000, key="training_log_autorefresh")
        _log_text = pm.tail_log(job_status["log_path"], n_lines=300)

        _progress = _extract_progress(_log_text)
        if _progress:
            _current, _total = _progress
            st.progress(min(_current / _total, 1.0), text=f"step {_current}/{_total}")
        else:
            st.caption("진행률을 로그에서 추출하지 못했습니다 — 아래 raw 로그를 참고하세요.")

        st.text_area("실시간 로그 (2초마다 자동 갱신)", value=_log_text, height=350, key="training_log_area")

        st.markdown("**⏹ 학습 강제 종료**")
        _last_step = (tr.get_run(_run_id) or {}).get("last_checkpoint_step") if _run_id != "?" else None
        st.warning(
            "⚠️ 학습은 즉시 종료되며 마지막 체크포인트 이후 진행은 유실됩니다 "
            "(`lerobot-train`은 신호를 받아도 정상 종료하는 기능이 없어 데이터 수집처럼 "
            "포트를 정리하고 끝내는 graceful stop이 불가능합니다 — CLAUDE.md 31-2절). "
            f"마지막 저장된 체크포인트: {'step ' + str(_last_step) if _last_step is not None else '아직 없음'}",
            icon="⚠️",
        )
        _confirm = st.checkbox("마지막 체크포인트 이후 진행이 유실될 수 있음을 이해했습니다", key="training_confirm_stop")
        if st.button("⏹ 학습 강제 종료", disabled=not _confirm, type="primary"):
            pm.stop_job(STAGE, graceful=False)
            st.success("학습을 종료했습니다.")
            st.rerun()
    else:
        tab_new, tab_resume = st.tabs(["🚀 새 학습 시작", "▶️ 이어서 학습"])

        with tab_new:
            # 행동 먼저 -- session_state["training_new_job_name"]은 위에서 이미 seed됨
            if st.button("🚀 새 학습 시작", disabled=action_blocked, type="primary"):
                _job_name_input = st.session_state["training_new_job_name"]
                _run_id = tr.generate_run_id(_job_name_input or "run")
                _output_dir = ds.PROJECT_ROOT / "train_runs" / _run_id
                _cmd = [
                    sys.executable,
                    "-m",
                    "lerobot.scripts.lerobot_train",
                    f"--config_path={BASE_TRAIN_CONFIG}",
                    f"--output_dir={_output_dir}",
                    f"--job_name={_run_id}",
                    f"--dataset.root={ds.DATASET_ROOT}",
                ]
                if _latest_snapshot:
                    _cmd.append(f"--dataset.episodes={_latest_snapshot['train_episodes']}")
                _snapshot_path = (
                    str(ds.PROJECT_ROOT / "state" / "qa_snapshots" / _latest_snapshot["file"])
                    if _latest_snapshot
                    else None
                )
                tr.create_run(_run_id, str(_output_dir), dataset_snapshot_path=_snapshot_path, cmd=_cmd)
                try:
                    _started = pm.start_job(STAGE, _cmd, extra_meta={"run_id": _run_id})
                except pm.JobConflictError as e:
                    tr.delete_run(_run_id)
                    st.error(str(e))
                else:
                    st.session_state.pop(_crash_key, None)
                    st.session_state.pop(_finalize_msg_key, None)
                    st.session_state[_run_id_key] = _run_id
                    st.session_state[_was_running_key] = True
                    st.session_state[_last_log_key] = _started["log_path"]
                    st.success(f"학습을 시작했습니다 (run_id={_run_id}).")
                    st.rerun()

            # 설정
            st.text_input("job_name", key="training_new_job_name")
            st.caption("`config/train_main_run_config.yaml`을 기본값으로 쓰고, 데이터셋 경로와 episodes만 최신 상태로 덮어씁니다.")
            if _latest_snapshot:
                st.success(
                    f"최신 QA 스냅샷 자동 반영: `{_latest_snapshot['file']}` "
                    f"(train {len(_latest_snapshot['train_episodes'])}개 / val {len(_latest_snapshot['val_episodes'])}개 episode, "
                    f"{_latest_snapshot['created_at']} 생성)",
                    icon="📌",
                )
            else:
                st.warning(
                    "아직 저장된 QA 스냅샷이 없습니다\n\n"
                    "'② QA & 검증' 페이지에서 먼저 스냅샷을 저장하세요.\n\n"
                    "스냅샷 없이도 시작할 수 있지만 dataset.episodes가 YAML 기본값 그대로 쓰입니다.",
                    icon="⚠️",
                )

        with tab_resume:
            if not _interrupted_runs:
                st.caption("중단된(interrupted) 학습 기록이 없습니다.")
            else:
                # 행동 먼저 -- session_state["training_resume_select"]는 위에서 이미 seed/clamp됨
                if st.button("▶️ 이어서 학습", disabled=action_blocked, type="primary"):
                    _selected = _resume_options[st.session_state["training_resume_select"]]
                    _output_dir = Path(_selected["output_dir"])
                    _config_path = _resolve_latest_checkpoint_config(_output_dir)
                    if _config_path is None:
                        st.error(f"이어받을 체크포인트를 찾지 못했습니다 ({_output_dir}/checkpoints/ 안에 저장된 step이 없습니다).")
                    else:
                        _run_id = _selected["run_id"]
                        _cmd = [
                            sys.executable,
                            "-m",
                            "lerobot.scripts.lerobot_train",
                            f"--config_path={_config_path}",
                            "--resume=true",
                        ]
                        tr.mark_running(_run_id)
                        try:
                            _started = pm.start_job(STAGE, _cmd, extra_meta={"run_id": _run_id})
                        except pm.JobConflictError as e:
                            # 시작 자체가 막혔으니 방금 running으로 되돌린 상태를 다시
                            # interrupted로 롤백한다(빈 로그로 finalize -> 완료 마커가
                            # 없으니 자동으로 interrupted가 됨).
                            tr.finalize_run(_run_id, "")
                            st.error(str(e))
                        else:
                            st.session_state.pop(_crash_key, None)
                            st.session_state.pop(_finalize_msg_key, None)
                            st.session_state[_run_id_key] = _run_id
                            st.session_state[_was_running_key] = True
                            st.session_state[_last_log_key] = _started["log_path"]
                            st.success(f"학습을 이어서 시작했습니다 (run_id={_run_id}, config={_config_path}).")
                            st.rerun()

                # 설정
                st.selectbox("재개할 학습", options=_resume_labels, key="training_resume_select")

# 2. 결과(results) -- 체크포인트 스윕 loss + 저장된 체크포인트 목록
with st.container(border=True):
    render_card_header("📈", "학습 결과", "success")
    st.caption(
        "각 체크포인트(1500~10000 step)로 val episodes=[5,7,9] / train episodes=[0,1]에 대해 "
        "`policy.forward()` loss를 재현 측정한 결과 (같은 원리로 lerobot-train은 val loss를 기본 제공하지 않아 직접 재현)."
    )

    sweep_df = ds.get_checkpoint_sweep_df()
    long_df = sweep_df.melt(id_vars="step", value_vars=["train_loss", "val_loss"], var_name="series", value_name="loss")
    long_df["series"] = long_df["series"].map({"train_loss": "train", "val_loss": "val"})

    KNEE_LO, KNEE_HI = 6000, 7500
    band_df = pd.DataFrame({"x": [KNEE_LO], "x2": [KNEE_HI]})

    band = alt.Chart(band_df).mark_rect(opacity=0.08, color="#52514e").encode(x="x:Q", x2="x2:Q")
    loss_lines = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="training step"),
            y=alt.Y("loss:Q", title="loss"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["train", "val"], range=[TRAIN_COLOR, VAL_COLOR]),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
            tooltip=["step", "series", alt.Tooltip("loss:Q", format=".4f")],
        )
        .properties(height=320)
    )
    st.altair_chart((band + loss_lines).properties(title="train vs val loss"), width='stretch')

    ratio_line = (
        alt.Chart(sweep_df)
        .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2, color=RATIO_COLOR)
        .encode(
            x=alt.X("step:Q", title="training step"),
            y=alt.Y("ratio:Q", title="val / train loss ratio"),
            tooltip=["step", alt.Tooltip("ratio:Q", format=".2f")],
        )
        .properties(height=220)
    )
    rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4], color="#52514e").encode(y="y:Q")
    st.altair_chart((band + ratio_line + rule).properties(title="val/train ratio (1.0 = 과적합 없음)"), width='stretch')

    st.info(
        f"**early stopping 적정 지점: step {KNEE_LO}~{KNEE_HI}** — 이 구간부터 val loss 개선폭이 "
        "직전 구간 대비 5~10배 작아지며 사실상 정체됩니다(0.50→0.48, 이후 0.48→0.465로 거의 안 줄어듦). "
        "반면 train loss는 계속 떨어지고 ratio는 끝까지 단조 증가(1.24x→3.79x) — step 10000까지 다 돌리는 것보다 "
        "이 구간에서 멈추는 게 과적합 대비 유리한 트레이드오프입니다.",
        icon="🎯",
    )

    st.markdown("**저장된 체크포인트**")
    st.dataframe(ds.get_checkpoint_list_df(), width='stretch', hide_index=True)

# 3. 메타정보(접힘) -- 정적 yaml 설정값 + PNG
with st.expander("(자세히 보기) 학습 설정 (train_main_run_config.yaml) · 정적 PNG"):
    cfg = ds.get_train_config()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("steps", cfg["steps"])
    c2.metric("batch_size", cfg["batch_size"])
    c3.metric("num_workers", cfg["num_workers"])
    c4.metric("save_freq", cfg["save_freq"])
    st.write(f"**dataset.episodes** (train, 13개): `{cfg['dataset']['episodes']}`")
    st.write(f"**policy.normalization_mapping**: `{cfg['policy']['normalization_mapping']}`")

    # 주의: st.expander는 다른 expander 안에 중첩할 수 없다(Streamlit 제약) --
    # 이 섹션이 원래 독립 expander였으나, 지금은 위 메타정보 expander 안에
    # 있으므로 일반 섹션(subheader + 상시 표시)으로 바꿨다.
    st.subheader("정적 PNG로도 보기 (다운로드/대조용)")
    png_path = ds.get_asset_path("checkpoint_sweep_loss.png")
    if png_path.exists():
        st.image(str(png_path))
