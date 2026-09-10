"""真实 Tk 工业窗口的短交互检查；完整成功两轮仍属于 G4/G8 长验收。"""

from datetime import datetime, timezone
from pathlib import Path
import json
import time
from ibero.envs.latch_release import LatchReleaseEnv
from ibero.industrial_review import IndustrialReviewer


def main():
    out = Path("artifacts/industrial_core01/latch/viewer") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    env = LatchReleaseEnv()
    app = IndustrialReviewer(env, seed=0, steps=20)
    report = {
        "initial_wait": False,
        "pause": False,
        "single_step": False,
        "previous_frame": False,
        "history_does_not_integrate": False,
        "replay": False,
        "rerun": False,
    }
    state = {"phase": "initial", "started": time.monotonic()}

    def check():
        try:
            if time.monotonic() - state["started"] > 90:
                raise TimeoutError("Industrial viewer short check timed out")
            phase = state["phase"]
            if phase == "initial":
                assert env.data.time == 0 and not app.running
                report["initial_wait"] = True
                app.key("Return")
                state["phase"] = "run"
            elif phase == "run" and app.index >= 3:
                app.key("p")
                state["phase"] = "pause"
            elif phase == "pause" and app.future is None:
                report["pause"] = not app.running
                state.update(phase="single", index=app.index, time=float(env.data.time))
                app.key("n")
            elif (
                phase == "single" and app.future is None and app.index > state["index"]
            ):
                assert app.index == state["index"] + 1
                report["single_step"] = True
                state.update(phase="back", time=float(env.data.time))
                app.key("b")
            elif phase == "back":
                assert app.history and app.index == state["index"]
                assert env.data.time == state["time"]
                report["previous_frame"] = True
                report["history_does_not_integrate"] = True
                app.key("r")
                state["phase"] = "replay"
            elif phase == "replay" and not app.replaying:
                assert env.data.time == state["time"]
                report["replay"] = True
                app.key("Return")
                state["phase"] = "rerun"
            elif (
                phase == "rerun"
                and not app.pending_run
                and not app.resetting
                and env.data.time < state["time"]
            ):
                report["rerun"] = True
                app.last_image.save(out / "window.png")
                app.close()
                return
        except Exception as error:
            report["error"] = str(error)
            app.close()
            return
        app.root.after(100, check)

    app.root.after(200, check)
    app.run()
    env.close()
    report["passed"] = all(report.values()) and "error" not in report
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out, report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
