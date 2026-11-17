/* SO-101 모방학습 대시보드 — 정적 체험 데모.
 * data/manifest.json + data/frames.bin(tools/build_static_demo.py가 생성)을 읽어 화면을 그린다.
 * manifest가 없으면 '결과 준비 중' 상태로 화면 구성만 보여 준다. */
"use strict";

const COLORS = { train: "#2a78d6", val: "#1baf7a", ratio: "#4a3aa7", bar: "#1baf7a" };
const state = { manifest: null, frames: null, W: 13 };

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function metric(label, value, delta, note) {
  return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>` +
    (delta ? `<div class="delta">${esc(delta)}</div>` : "") + (note ? `<div class="note">${esc(note)}</div>` : "") + `</div>`;
}

function chart(el, spec) {
  spec.$schema = "https://vega.github.io/schema/vega-lite/v5.json";
  spec.width = "container";
  spec.config = Object.assign({ font: "Malgun Gothic, system-ui, sans-serif", view: { stroke: null },
    axis: { labelColor: "#52514e", titleColor: "#52514e", gridColor: "#e5e4e0" } }, spec.config || {});
  if (window.vegaEmbed) vegaEmbed(el, spec, { actions: false });
}

/* ---------- 라우팅 ---------- */
function route() {
  const page = (location.hash || "#home").slice(1);
  const known = ["home", "collect", "qa", "train", "infer"];
  const cur = known.includes(page) ? page : "home";
  known.forEach((p) => { $("#page-" + p).hidden = p !== cur; });
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === cur));
  window.scrollTo(0, 0);
  // 숨겨져 있던 차트는 폭을 못 잡으므로 보일 때 그린다
  if (state.manifest && renderers[cur] && !renderers[cur].done) { renderers[cur](); renderers[cur].done = true; }
}

/* ---------- 홈 ---------- */
function renderHome() {
  const m = state.manifest, s = m.summary || {}, ref = m.qa_reference;
  let html = metric("① 데이터 수집", `${m.total_episodes} episodes`, `${m.total_frames} frames`, `fps=${m.fps}`);
  html += metric("② QA & 검증", `${m.total_episodes} episodes 스캔`, `이상 frame ${ref.excluded_count}건 탐지`);
  if (m.sweep) {
    const last = m.sweep.steps[m.sweep.steps.length - 1], r = m.sweep.results[String(last)];
    html += metric("③ ACT 학습", `final train loss ${r.train_loss.toFixed(3)}`, `val ${r.val_loss.toFixed(3)}`, [s.gpu, s.steps_label, s.train_time_label].filter(Boolean).join(" · "));
  } else if (m.train_log && m.train_log.length) {
    const last = m.train_log[m.train_log.length - 1];
    html += metric("③ ACT 학습", `final train loss ${last[1].toFixed(3)}`, `step ${last[0]}`, [s.gpu, s.steps_label, s.train_time_label].filter(Boolean).join(" · "));
  } else html += metric("③ ACT 학습", "기록 없음");
  if (m.eval) html += metric("④ 추론 성공률", `${m.eval.successes} / ${m.eval.episodes}`, `${Math.round(m.eval.success_rate * 100)}% (MuJoCo 자동 평가)`);
  else html += metric("④ 추론", "평가 기록 없음");
  $("#home-metrics").innerHTML = html;
}

/* ---------- ① 데이터 수집 ---------- */
function episodeJointRows(epIndex) {
  const m = state.manifest, ep = m.episodes.find((e) => e.index === epIndex);
  const rows = [];
  for (let k = 0; k < ep.length; k++) {
    const base = (ep.from + k) * state.W;
    m.joint_names.forEach((name, j) => rows.push({ frame: k, joint: name, state: state.frames[base + j] }));
  }
  return rows;
}

function renderCollect() {
  const m = state.manifest;
  if (m.image_shape) $("#schema-img").textContent = `(${m.image_shape.join(", ")})`;
  const opts = m.episodes.map((e) => `<option value="${e.index}">episode ${e.index} (${e.length} frames)</option>`).join("");
  $("#collect-body").innerHTML =
    `<div class="metrics">${metric("총 episode 수", m.total_episodes)}${metric("총 frame 수", m.total_frames)}${metric("fps", m.fps)}</div>` +
    `<b>episode별 길이</b><div id="ep-len" class="chart"></div>` +
    `<p class="caption">길이가 유난히 짧거나 긴 episode는 오조작일 수 있으니 ② QA 검증에서 확인합니다.</p>` +
    `<b>episode 둘러보기</b><label class="field">episode<select id="ep-select">${opts}</select></label>` +
    `<div class="browse" id="browse"><div id="ep-video"></div><div><div id="ep-joints" class="chart"></div>` +
    `<p class="caption">6개 관절의 실제 위치(observation.state) — 매 episode 비슷한 궤적이면 시연이 일관된 것입니다.</p></div></div>`;
  chart("#ep-len", { height: 260, data: { values: m.episodes.map((e) => ({ episode: e.index, frames: e.length })) },
    mark: { type: "bar", color: COLORS.bar }, encoding: { x: { field: "episode", type: "ordinal", title: "episode" },
      y: { field: "frames", type: "quantitative", title: "frame 수" }, tooltip: [{ field: "episode" }, { field: "frames" }] } });
  const show = () => {
    const idx = Number($("#ep-select").value);
    const hasVideo = m.previews.includes(idx);
    $("#browse").style.gridTemplateColumns = hasVideo ? "" : "1fr";
    $("#ep-video").innerHTML = hasVideo
      ? `<video src="media/episodes/episode_${String(idx).padStart(3, "0")}.mp4" autoplay muted loop playsinline controls></video><p class="caption">손목 카메라 (학습 입력 이미지, 30fps)</p>`
      : "";
    $("#ep-video").hidden = !hasVideo;
    chart("#ep-joints", { height: 290, data: { values: episodeJointRows(idx) }, mark: { type: "line", strokeWidth: 1.6 },
      encoding: { x: { field: "frame", type: "quantitative", title: "frame (30fps)" }, y: { field: "state", type: "quantitative", title: "관절각 (rad)" },
        color: { field: "joint", type: "nominal", sort: m.joint_names, scale: { domain: m.joint_names }, legend: { orient: "bottom", title: null } } } });
  };
  $("#ep-select").addEventListener("change", show);
  show();
}

/* ---------- ② QA ---------- */
function countTable(title, counts) {
  const keys = Object.keys(counts).map(Number).sort((a, b) => a - b);
  const rows = keys.length ? keys.map((k) => `<tr><td>${k}</td><td>${counts[k]}</td></tr>`).join("") : `<tr><td>(없음)</td><td>0</td></tr>`;
  return `<div><b>${title}</b><table><tr><th>episode</th><th>이상 frame 수</th></tr>${rows}</table></div>`;
}

function sameCounts(a, b) {
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && a[k] === b[k]);
}

function renderQA() {
  const m = state.manifest;
  $("#qa-body").innerHTML =
    `<div class="controls"><label class="field">값 기준 z_thresh (detect_outliers)<input id="zv" type="number" min="0.5" max="10" step="0.5" value="3"></label>` +
    `<label class="field">delta 기준 z_thresh (detect_outliers_delta)<input id="zd" type="number" min="0.5" max="10" step="0.5" value="3"></label></div>` +
    `<div id="qa-out"></div>`;
  const run = () => {
    const zv = Number($("#zv").value) || 3, zd = Number($("#zd").value) || 3;
    const t0 = performance.now();
    const r = SO101QA.computeQA(state.frames, m, zv, zd);
    const ms = performance.now() - t0;
    let check = "";
    const ref = m.qa_reference;
    if (zv === ref.z_thresh_value && zd === ref.z_thresh_delta) {
      const ok = r.excluded === ref.excluded_count && r.integrityCount === ref.integrity_problem_count &&
        sameCounts(r.valueCounts, ref.value_episode_outlier_counts) && sameCounts(r.deltaCounts, ref.delta_episode_outlier_counts);
      check = ok ? `<div class="banner ok">✅ 브라우저 계산 결과가 Python <code>TeamRobotDataset</code> 결과(z=3)와 일치합니다.</div>`
        : `<div class="banner warn">⚠️ Python 결과와 차이가 있습니다 (Python 제외 frame ${ref.excluded_count}건).</div>`;
    }
    $("#qa-out").innerHTML =
      `<div class="metrics">${metric("스캔한 episode 수", m.total_episodes)}${metric("스캔한 frame 수", r.numFrames)}` +
      `${metric("무결성 이상(NaN/Inf)", r.integrityCount)}${metric("clean frame 비율", `${r.numFrames - r.excluded}/${r.numFrames}`)}</div>` +
      `<p class="caption">이 브라우저에서 ${r.numFrames.toLocaleString()} frame을 ${ms < 1 ? "1ms 미만" : ms.toFixed(0) + "ms"}에 계산했습니다.</p>` + check +
      `<div class="two-col">${countTable("값 기준 아웃라이어 (<code>detect_outliers</code>)", r.valueCounts)}` +
      `${countTable("delta 기준 아웃라이어 (<code>detect_outliers_delta</code>)", r.deltaCounts)}</div>` +
      `<p><b><code>get_clean_indices()</code></b> — timestamp 필드 제외, 값+delta 기준 합집합<br>` +
      `전체 ${r.numFrames} frames 중 <b>${r.excluded}건 제외</b> (${r.numFrames - r.excluded} clean) — <i>비파괴적, 실제로 제거하지 않음</i></p>` +
      `<p><b><code>split_episodes(val_ratio=0.2, seed=42)</code></b><br>train_episodes = [${m.split.train_episodes.join(", ")}]<br>val_episodes = [${m.split.val_episodes.join(", ")}]</p>`;
  };
  $("#zv").addEventListener("change", run);
  $("#zd").addEventListener("change", run);
  run();
  $("#qa-case").hidden = !m.has_wrist_roll_case;
}

/* ---------- ③ 학습 ---------- */
function renderTrain() {
  const m = state.manifest, s = m.summary || {}, c = m.train_config || {};
  if (s.gpu) $("#train-caption").innerHTML += ` (학습 GPU: ${esc(s.gpu)})`;
  let html = "";
  if (m.train_log && m.train_log.length) {
    html += `<b>학습 중 train loss (lerobot-train 로그)</b><div id="loss-log" class="chart"></div>`;
    const bits = [s.gpu, s.steps_label, s.train_time_label].filter(Boolean);
    if (bits.length) html += `<p class="caption">${esc(bits.join(" · "))}</p>`;
  }
  if (m.sweep) html += `<b>체크포인트별 train vs val loss</b><p class="caption">각 체크포인트로 train/val episode에 대해 loss를 다시 측정한 결과 (lerobot-train은 val loss를 기본 제공하지 않아 직접 재현).</p><div id="loss-sweep" class="chart"></div>`;
  if (!html) html = `<p class="empty">아직 공개된 학습 기록이 없습니다.</p>`;
  html += `<h3>학습 설정</h3><div class="metrics">${metric("steps", c.steps ?? "-")}${metric("batch_size", c.batch_size ?? "-")}` +
    `${metric("num_workers", c.num_workers ?? "-")}${metric("save_freq", c.save_freq ?? "-")}</div>`;
  $("#train-body").innerHTML = html;
  if (m.train_log && m.train_log.length) {
    chart("#loss-log", { height: 260, data: { values: m.train_log.map(([step, loss]) => ({ step, loss })) },
      mark: { type: "line", strokeWidth: 2, color: COLORS.train },
      encoding: { x: { field: "step", type: "quantitative", title: "training step" },
        y: { field: "loss", type: "quantitative", title: "train loss", scale: { type: "log" } }, tooltip: [{ field: "step" }, { field: "loss", format: ".4f" }] } });
  }
  if (m.sweep) {
    const rows = [];
    m.sweep.steps.forEach((st) => { const r = m.sweep.results[String(st)]; rows.push({ step: st, series: "train", loss: r.train_loss }, { step: st, series: "val", loss: r.val_loss }); });
    chart("#loss-sweep", { height: 280, data: { values: rows }, mark: { type: "line", point: true, strokeWidth: 2 },
      encoding: { x: { field: "step", type: "quantitative", title: "training step" }, y: { field: "loss", type: "quantitative", title: "loss" },
        color: { field: "series", type: "nominal", scale: { domain: ["train", "val"], range: [COLORS.train, COLORS.val] }, legend: { title: null, orient: "top-right" } } } });
  }
}

/* ---------- ④ 추론 ---------- */
function renderInfer() {
  const m = state.manifest, e = m.eval;
  if (!e) return;
  const times = (e.results || []).filter((r) => r.success && r.time_to_place_s != null).map((r) => r.time_to_place_s);
  const avg = times.length ? `${(times.reduce((a, b) => a + b, 0) / times.length).toFixed(1)}초` : "-";
  const b = m.eval_batch;
  const jitter = e.obj_jitter_cm ?? (b && b.obj_jitter_cm);
  let html = `<p class="caption">학습된 정책이 MuJoCo에서 혼자 물체를 집어 접시에 놓는지 ${e.episodes}번 반복 실행했습니다. ` +
    `팔은 학습 데이터의 쉬는 자세에서 시작하고${jitter ? `, 물체 시작 위치는 매번 ±${jitter}cm 랜덤` : ""}입니다. ` +
    `물체가 접시 위에 ${e.hold_seconds}초 이상 머물면 성공. 한 번 시도한 뒤 팔이 쉬는 자세로 돌아오면 episode를 끝냅니다` +
    `(실패 후 재시도 성공은 세지 않음, 최대 ${e.max_seconds}초).</p>`;
  if (b && b.items && b.items.length > 1) {
    const fails = [...new Set(b.items.flatMap((it) => Object.keys(it.failure_counts || {})))];
    html += `<b>체크포인트 비교 (같은 조건·같은 물체 위치 순서)</b><table><tr><th>체크포인트</th><th>성공</th><th>성공률</th>` +
      fails.map((f) => `<th>${f}</th>`).join("") + `</tr>` +
      b.items.slice().sort((x, y) => x.step - y.step).map((it) => `<tr><td>step ${it.step}</td><td>${it.successes} / ${b.episodes}</td>` +
        `<td>${Math.round(it.success_rate * 100)}%</td>` + fails.map((f) => `<td>${(it.failure_counts || {})[f] || 0}</td>`).join("") + `</tr>`).join("") +
      `</table><p class="caption">실패 유형은 물체 궤적만으로 분류했습니다(0.5cm 미만 이동=미접촉, 1cm 미만 들림=밀기만 함, 그 이상=들었지만 못 놓음). 아래 수치·영상은 가장 좋은 체크포인트(${e.label || ""}) 기준입니다.</p>`;
  }
  html += `<div class="metrics">${metric("성공", `${e.successes} / ${e.episodes}`)}${metric("성공률", `${Math.round(e.success_rate * 100)}%`)}${metric("놓기까지 평균 시간", avg)}</div>`;
  if (m.inference_video) html += `<video src="${m.inference_video}" controls muted playsinline></video><p class="caption">평가 ${e.episodes} episode 전체를 이어 붙인 3인칭 MuJoCo 렌더 (편집·선별 없음, 성공과 실패가 그대로 섞여 있음). 정책 입력은 손목 카메라와 관절값뿐이며 이 3인칭 시점은 보여주기용입니다.</p>`;
  html += `<details><summary>episode별 결과</summary><table><tr><th>episode</th><th>결과</th><th>놓기까지(초)</th><th>최대 들어올림</th><th>접시 중심까지 거리</th></tr>` +
    (e.results || []).map((r) => `<tr><td>${r.episode + 1}</td><td>${r.success ? "성공" : `실패${r.failure_type ? ` (${r.failure_type})` : ""}`}</td><td>${r.time_to_place_s ?? "-"}</td>` +
      `<td>${r.object_max_lift_cm != null ? r.object_max_lift_cm.toFixed(1) + "cm" : "-"}</td><td>${(r.final_dist_to_plate_xy * 100).toFixed(1)}cm</td></tr>`).join("") +
    `</table></details>`;
  $("#infer-body").innerHTML = html;
}

const renderers = { home: renderHome, collect: renderCollect, qa: renderQA, train: renderTrain, infer: renderInfer };

/* ---------- 시작 ---------- */
async function boot() {
  window.addEventListener("hashchange", route);
  try {
    const res = await fetch("data/manifest.json", { cache: "no-cache" });
    if (!res.ok) throw new Error("no manifest");
    state.manifest = await res.json();
    state.W = state.manifest.layout.reduce((a, f) => a + f.dims, 0);
    const bin = await fetch("data/frames.bin");
    state.frames = new Float32Array(await bin.arrayBuffer());
  } catch (err) {
    state.manifest = null;
    $("#pending").hidden = false;
  }
  route();
}
boot();
