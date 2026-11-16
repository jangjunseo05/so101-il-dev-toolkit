"""
학습된 ACT 체크포인트를 MuJoCo에서 여러 episode 자동으로 돌려 pick&place
성공률을 재고, 원하면 3인칭 장면을 mp4로 저장하는 평가 스크립트.

run_inference_mujoco.py는 사람이 뷰어로 지켜보는 용도(또는 몇 초짜리 헬스체크)라
"몇 번 중 몇 번 성공했는가"를 셀 수 없었다. 이 스크립트는 그 모듈의 관측 생성/
policy 호출 경로(_select_action)를 그대로 재사용하고, 그 위에 다음만 더한다.

  - episode마다 로봇과 물체를 scene.xml 기본 스폰 위치로 리셋하고 policy.reset()
  - 성공 판정: 물체 중심이 접시(place_plate) 반경 안(xy)에 있고, 접시 위 높이에
    놓여 있으며, 그 상태가 --hold-seconds 동안 유지되면 성공으로 보고 조기 종료
  - --max-seconds 안에 성공하지 못하면 실패
  - --video를 주면 3인칭 카메라 장면을 mp4로 저장(episode를 이어 붙임)
  - 결과 요약을 콘솔과 JSON(--report)으로 남김

사용 예:
  python pipeline/4_inference/eval_success_mujoco.py --checkpoint <pretrained_model 경로> --episodes 10
  python pipeline/4_inference/eval_success_mujoco.py --checkpoint <경로> --episodes 3 --video reports/inference_demo.mp4

물체 위치는 run_inference_mujoco.py와 같이 매 episode 고정 위치에서 시작한다
(데이터 수집도 기본값이 같은 고정 위치이므로 학습·평가 조건이 일치한다).
"""

import argparse
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import run_inference_mujoco as rim  # noqa: E402  (main()은 실행되지 않음)

mujoco = rim.mujoco
np = rim.np
torch = rim.torch

PLATE_BODY = "place_plate"
PLATE_GEOM = "place_plate_geom"
OBJECT_BODY = "pick_object"
OBJECT_GEOM = "pick_object_geom"


def _make_scene_camera(args) -> "mujoco.MjvCamera":
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = args.cam_lookat
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation
    return cam


class _VideoWriter:
    """PyAV(av)로 H.264 mp4를 쓴다. av는 lerobot 의존성으로 이미 설치되어 있다."""

    def __init__(self, path: str, width: int, height: int, fps: int):
        import av

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(path, mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": "23"}
        self._av = av

    def write(self, rgb: "np.ndarray") -> None:
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _object_on_plate(model, data, plate_xy, plate_radius, object_half, plate_top_z) -> bool:
    obj_pos = data.xpos[model.body(OBJECT_BODY).id]
    dist_xy = float(np.linalg.norm(obj_pos[:2] - plate_xy))
    # 물체 중심이 접시 반경 안쪽(물체 반쪽 크기만큼 여유를 뺀 범위)에 있고,
    # 접시 윗면 위에 놓인 높이(공중에 들려 있지 않음)인지 확인한다.
    within = dist_xy <= plate_radius - object_half * 0.5
    resting = obj_pos[2] <= plate_top_z + object_half * 2.5
    return bool(within and resting)


def run_episode(*, ep_idx, args, model, data, renderer, policy, preprocessor, postprocessor,
                follower_qpos_adr, follower_actuator_id, ctrl_lo, ctrl_hi, substeps,
                scene_renderer=None, scene_cam=None, writer=None):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    policy.reset()

    plate_xy = data.xpos[model.body(PLATE_BODY).id][:2].copy()
    plate_radius = float(model.geom(PLATE_GEOM).size[0])
    plate_top_z = float(data.xpos[model.body(PLATE_BODY).id][2] + model.geom(PLATE_GEOM).size[1])
    object_half = float(model.geom(OBJECT_GEOM).size[0])

    loop_dt = 1.0 / args.hz
    frame_interval = 1.0 / args.policy_hz
    video_interval = 1.0 / args.video_fps
    frame_accum = 0.0
    video_accum = video_interval  # 첫 프레임을 바로 기록
    sim_time = 0.0
    on_plate_since = None
    success_time = None

    while sim_time < args.max_seconds:
        frame_accum += loop_dt
        if frame_accum >= frame_interval:
            frame_accum -= frame_interval
            targets = rim._select_action(
                model=model, data=data, renderer=renderer, policy=policy,
                preprocessor=preprocessor, postprocessor=postprocessor,
                follower_qpos_adr=follower_qpos_adr, task=args.task,
                ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, preview=None,
            )
            for act_id, val in zip(follower_actuator_id, targets):
                data.ctrl[act_id] = val

        for _ in range(substeps):
            mujoco.mj_step(model, data)
        sim_time += loop_dt

        if writer is not None:
            video_accum += loop_dt
            if video_accum >= video_interval:
                video_accum -= video_interval
                scene_renderer.update_scene(data, camera=scene_cam)
                writer.write(scene_renderer.render())

        if _object_on_plate(model, data, plate_xy, plate_radius, object_half, plate_top_z):
            if on_plate_since is None:
                on_plate_since = sim_time
            elif sim_time - on_plate_since >= args.hold_seconds:
                success_time = on_plate_since
                break
        else:
            on_plate_since = None

    obj_pos = data.xpos[model.body(OBJECT_BODY).id].copy()
    result = {
        "episode": ep_idx,
        "success": success_time is not None,
        "time_to_place_s": None if success_time is None else round(success_time, 2),
        "sim_seconds": round(sim_time, 2),
        "final_object_xyz": [round(float(v), 4) for v in obj_pos],
        "final_dist_to_plate_xy": round(float(np.linalg.norm(obj_pos[:2] - plate_xy)), 4),
    }
    status = "성공" if result["success"] else "실패"
    print(f"[eval] episode {ep_idx}: {status} (sim {result['sim_seconds']}s, "
          f"접시 중심까지 {result['final_dist_to_plate_xy']*100:.1f}cm)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="ACT 체크포인트 MuJoCo 자동 성공률 평가")
    parser.add_argument("--checkpoint", default=rim.DEFAULT_CHECKPOINT,
                        help="pretrained_model 폴더(config.json + model.safetensors)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=30.0, help="episode당 최대 시뮬레이션 시간")
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="접시 위에 이만큼 유지돼야 성공")
    parser.add_argument("--hz", type=float, default=rim.CONTROL_HZ)
    parser.add_argument("--policy-hz", type=float, default=rim.POLICY_HZ)
    parser.add_argument("--task", default=rim.DEFAULT_TASK)
    parser.add_argument("--video", default=None, help="mp4 저장 경로(생략하면 영상 저장 안 함)")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=[0.15, 0.07, 0.07])
    parser.add_argument("--cam-distance", type=float, default=0.62)
    parser.add_argument("--cam-azimuth", type=float, default=200.0)
    parser.add_argument("--cam-elevation", type=float, default=-30.0)
    parser.add_argument("--report", default=None, help="결과 JSON 저장 경로(생략 시 reports/eval_<시각>.json)")
    args = parser.parse_args()

    print(f"[eval] 체크포인트: {args.checkpoint}")
    policy = rim.ACTPolicy.from_pretrained(args.checkpoint, device=args.device)
    policy.eval()
    preprocessor, postprocessor = rim.make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    model = mujoco.MjModel.from_xml_path(rim.SCENE_XML)
    data = mujoco.MjData(model)
    follower_qpos_adr = [model.jnt_qposadr[model.joint(n).id] for n in rim.JOINT_NAMES]
    follower_actuator_id = [model.actuator(n).id for n in rim.JOINT_NAMES]
    ctrl_lo = model.actuator_ctrlrange[follower_actuator_id, 0]
    ctrl_hi = model.actuator_ctrlrange[follower_actuator_id, 1]
    substeps = max(1, round(1.0 / args.hz / model.opt.timestep))
    renderer = mujoco.Renderer(model, height=rim.IMAGE_HEIGHT, width=rim.IMAGE_WIDTH)

    scene_renderer = scene_cam = writer = None
    if args.video:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.video_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.video_height)
        scene_renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        scene_cam = _make_scene_camera(args)
        writer = _VideoWriter(args.video, args.video_width, args.video_height, args.video_fps)

    results = []
    t0 = time.time()
    try:
        for ep in range(args.episodes):
            results.append(run_episode(
                ep_idx=ep, args=args, model=model, data=data, renderer=renderer, policy=policy,
                preprocessor=preprocessor, postprocessor=postprocessor,
                follower_qpos_adr=follower_qpos_adr, follower_actuator_id=follower_actuator_id,
                ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, substeps=substeps,
                scene_renderer=scene_renderer, scene_cam=scene_cam, writer=writer,
            ))
    finally:
        if writer is not None:
            writer.close()
        renderer.close()
        if scene_renderer is not None:
            scene_renderer.close()

    n_ok = sum(r["success"] for r in results)
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": len(results),
        "successes": n_ok,
        "success_rate": round(n_ok / len(results), 3) if results else 0.0,
        "max_seconds": args.max_seconds,
        "hold_seconds": args.hold_seconds,
        "wall_seconds": round(time.time() - t0, 1),
        "video": args.video,
        "results": results,
    }
    report = Path(args.report) if args.report else rim.PROJECT_ROOT / "reports" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 성공 {n_ok}/{len(results)} ({summary['success_rate']*100:.0f}%) -- 결과: {report}")
    if args.video:
        print(f"[eval] 영상: {args.video}")


if __name__ == "__main__":
    main()
