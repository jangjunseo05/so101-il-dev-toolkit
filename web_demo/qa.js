/* team_robot_dataset.py의 QA 계산을 브라우저용으로 옮긴 것 (같은 결과를 내도록 맞춤).
 * - detect_outliers():       관절(차원)별 전체 평균·표준편차(모집단, ddof=0)로 z-score, |z| > 기준이면 그 frame 표시
 * - detect_outliers_delta(): 같은 episode 안에서 바로 앞 frame과의 차이로 같은 계산 (episode 경계는 건너뜀)
 * - check_integrity():       NaN/Inf
 * - get_clean_indices():     위 결과의 합집합, 단 timestamp에서만 걸린 frame은 제외하지 않음
 * frames: Float32Array, frame마다 [observation.state(6), action(6), timestamp(1)]
 */
(function (root) {
  function columnKeys(layout) {
    const keys = [];
    layout.forEach((f) => { for (let d = 0; d < f.dims; d++) keys.push(f.key); });
    return keys;
  }

  function meanStd(get, n, width) {
    const mean = new Float64Array(width), m2 = new Float64Array(width);
    for (let i = 0; i < n; i++) for (let c = 0; c < width; c++) mean[c] += get(i, c);
    for (let c = 0; c < width; c++) mean[c] /= n || 1;
    for (let i = 0; i < n; i++) for (let c = 0; c < width; c++) { const d = get(i, c) - mean[c]; m2[c] += d * d; }
    const std = new Float64Array(width);
    for (let c = 0; c < width; c++) std[c] = Math.sqrt(m2[c] / (n || 1));
    return { mean, std };
  }

  function computeQA(frames, manifest, zValue, zDelta) {
    const keys = columnKeys(manifest.layout);
    const W = keys.length, N = frames.length / W;
    const epOf = new Int32Array(N), frOf = new Int32Array(N);
    manifest.episodes.forEach((ep) => { for (let k = 0; k < ep.length; k++) { epOf[ep.from + k] = ep.index; frOf[ep.from + k] = k; } });
    const v = (i, c) => frames[i * W + c];

    const valueAny = new Uint8Array(N), valueReal = new Uint8Array(N);
    const deltaAny = new Uint8Array(N), deltaReal = new Uint8Array(N);
    const integAny = new Uint8Array(N), integReal = new Uint8Array(N);

    // 무결성
    for (let i = 0; i < N; i++) for (let c = 0; c < W; c++) {
      if (!Number.isFinite(v(i, c))) { integAny[i] = 1; if (keys[c] !== "timestamp") integReal[i] = 1; }
    }
    // 값 기준
    const s = meanStd(v, N, W);
    for (let i = 0; i < N; i++) for (let c = 0; c < W; c++) {
      if (s.std[c] > 0 && Math.abs((v(i, c) - s.mean[c]) / s.std[c]) > zValue) {
        valueAny[i] = 1; if (keys[c] !== "timestamp") valueReal[i] = 1;
      }
    }
    // delta 기준 (episode 경계를 넘는 쌍은 제외)
    const pairs = [];
    for (let i = 1; i < N; i++) if (epOf[i] === epOf[i - 1]) pairs.push(i);
    const dget = (p, c) => v(pairs[p], c) - v(pairs[p] - 1, c);
    const ds = meanStd(dget, pairs.length, W);
    for (let p = 0; p < pairs.length; p++) for (let c = 0; c < W; c++) {
      if (ds.std[c] > 0 && Math.abs((dget(p, c) - ds.mean[c]) / ds.std[c]) > zDelta) {
        deltaAny[pairs[p]] = 1; if (keys[c] !== "timestamp") deltaReal[pairs[p]] = 1;
      }
    }

    const perEpisode = (mask) => {
      const out = {};
      for (let i = 0; i < N; i++) if (mask[i]) out[epOf[i]] = (out[epOf[i]] || 0) + 1;
      return out;
    };
    let excluded = 0, integrityCount = 0;
    for (let i = 0; i < N; i++) {
      if (valueReal[i] || deltaReal[i] || integReal[i]) excluded++;
      if (integAny[i]) integrityCount++;
    }
    return {
      numFrames: N,
      valueCounts: perEpisode(valueAny),
      deltaCounts: perEpisode(deltaAny),
      integrityCount,
      excluded,
      frameIndexOf: frOf,
    };
  }

  const api = { computeQA, columnKeys };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SO101QA = api;
})(this);
