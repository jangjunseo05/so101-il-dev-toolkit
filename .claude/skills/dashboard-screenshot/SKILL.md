---
name: dashboard-screenshot
description: "Use whenever verifying visual/UI changes to the so101_web Streamlit dashboard (dashboard/app.py, dashboard/pages/*.py) — color contrast, layout, spacing, card styling. AppTest confirms elements render without exceptions but can't confirm what they actually look like; this skill launches the real app headlessly and captures an actual browser screenshot via Playwright."
---

# dashboard-screenshot

Streamlit's multipage routing does **not** resolve page URLs directly
(e.g. hitting `/1_데이터_수집` returns "Page not found") — the app must be
driven like a real user: load the home page, then click the sidebar nav
link text.

## One-time setup (already done as of 2026-08-16, skip if `.tools/screenshot/node_modules/` exists)

```powershell
cd C:\Users\USER\Desktop\so101_web\.tools\screenshot
npm install playwright@1.62.1
npx playwright install chromium   # downloads the browser binary to
                                   # ~/AppData/Local/ms-playwright/ (global
                                   # cache, persists across projects/sessions)
```

If `.tools/screenshot/node_modules/playwright` already exists, skip
straight to "Usage" below — no reinstall needed.

## Usage

1. Launch the dashboard headlessly on a scratch port (don't reuse 8501 —
   that's the port a real teammate session might be using):

```powershell
conda activate mujoco_env
cd C:\Users\USER\Desktop\so101_web
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Start-Process -FilePath "streamlit" -ArgumentList "run", "dashboard/app.py", "--server.headless", "true", "--server.port", "8503" -RedirectStandardOutput "shot_stdout.log" -RedirectStandardError "shot_stderr.log" -PassThru | Select-Object -ExpandProperty Id | Out-File -FilePath shot.pid -Encoding ascii
Start-Sleep -Seconds 6
Invoke-WebRequest -Uri "http://localhost:8503" -UseBasicParsing -TimeoutSec 5 | Select-Object -ExpandProperty StatusCode
```

2. Run the driver script (bash):

```bash
cd "C:/Users/USER/Desktop/so101_web/.tools/screenshot"
node shot.js <port> "<sidebar link text|app>" "<output path>" ["<scroll-to text>"]
```

- `<sidebar link text>`: exact text of the page in the left nav, e.g.
  `데이터 수집`, `QA 검증`, `ACT 학습`, `추론`. Use `app` for the home page
  (no click needed).
- `<output path>`: absolute path for the PNG.
- `<scroll-to text>` (optional): substring of any visible text to scroll
  to before capturing a clipped 1280x1400 viewport shot instead of a
  full-page shot — use this to target a specific card/section instead of
  screenshotting the whole (possibly very long) page.

Example — screenshot the "녹화 세션 제어" card on the data-collection page:

```bash
node shot.js 8503 "데이터 수집" "/tmp/check.png" "녹화 세션 제어"
```

3. View the result with the Read tool (it supports PNG).

4. Clean up:

```powershell
$procId = Get-Content C:\Users\USER\Desktop\so101_web\shot.pid
Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
Remove-Item C:\Users\USER\Desktop\so101_web\shot.pid, C:\Users\USER\Desktop\so101_web\shot_stdout.log, C:\Users\USER\Desktop\so101_web\shot_stderr.log -ErrorAction SilentlyContinue
```

Also delete any screenshot PNGs left in `.tools/screenshot/` after
reviewing them — they're scratch output, not meant to accumulate
(`.tools/` is gitignored, but no reason to let it grow unbounded).

## Notes

- The script waits for any Streamlit spinner (`[data-testid="stSpinner"]`)
  to disappear before screenshotting, up to 30s -- this covers slow
  in-process computation (e.g. the QA page's live `TeamRobotDataset` scan),
  not just chart rendering. If a page is consistently slower than 30s,
  raise the timeout in `shot.js` rather than adding a longer fixed wait.
- If you need to simulate a specific dashboard *state* (blocked action,
  missing settings, crash banner) before screenshotting, do it via the
  real files/session state, not by editing the page code:
  - Missing local settings: `mv config/local_settings.json config/local_settings.json.bak`,
    screenshot, then `mv config/local_settings.json.bak config/local_settings.json`
    to restore. Never leave it removed.
  - Crash banner / other `st.session_state` driven UI: easier to verify
    via `streamlit.testing.v1.AppTest` with `at.session_state[...] = ...`
    preloaded before `.run()` (see any recent CLAUDE.md entry for the
    exact session_state keys per page) — AppTest can preload state that's
    awkward to trigger through the real UI. Use AppTest for state
    simulation and this skill for the final visual check.
