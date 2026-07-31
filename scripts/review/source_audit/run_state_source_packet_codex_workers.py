from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKET_ROOT = REPO_ROOT / "state" / "review" / "packets" / "source" / "states"


def _tmux_has_session(session_name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session_name], text=True, capture_output=True).returncode == 0


def _find_existing_packet_session(packet_dir: Path) -> Optional[str]:
    proc = subprocess.run(
        ["tmux", "list-panes", "-a", "-F", "#{session_name}\t#{pane_start_command}"],
        text=True,
        capture_output=True,
        check=True,
    )
    prompt_path = str(packet_dir / "codex_worker_prompt.txt")
    packet_str = str(packet_dir)
    rel_packet_str = str(packet_dir.relative_to(REPO_ROOT))
    rel_prompt_path = str((packet_dir / "codex_worker_prompt.txt").relative_to(REPO_ROOT))
    for line in proc.stdout.splitlines():
        try:
            session_name, start_cmd = line.split("\t", 1)
        except ValueError:
            continue
        if (
            prompt_path in start_cmd
            or packet_str in start_cmd
            or rel_prompt_path in start_cmd
            or rel_packet_str in start_cmd
        ):
            return session_name
    return None


def _packet_dirs_from_args(state_abbrs: list[str]) -> list[Path]:
    if not state_abbrs:
        return sorted(p for p in PACKET_ROOT.iterdir() if p.is_dir())
    return [PACKET_ROOT / state.lower() for state in state_abbrs]


def _launch_packet(packet_dir: Path, *, model: str, session_prefix: str, force_restart: bool) -> dict[str, str]:
    state_key = packet_dir.name.lower()
    prompt_path = packet_dir / "codex_worker_prompt.txt"
    run_log = packet_dir / "codex_worker_run.jsonl"
    last_message = packet_dir / "codex_worker_last_message.txt"
    session_name = f"{session_prefix}{state_key}"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    existing_session = _find_existing_packet_session(packet_dir)
    if existing_session and not force_restart:
        return {"state": state_key.upper(), "session_name": existing_session, "status": "already_running"}
    if existing_session and force_restart:
        subprocess.run(["tmux", "kill-session", "-t", existing_session], check=True, text=True, capture_output=True)

    if _tmux_has_session(session_name):
        if force_restart:
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True, text=True, capture_output=True)
        else:
            return {"state": state_key.upper(), "session_name": session_name, "status": "already_running"}

    cmd = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"codex exec --dangerously-bypass-approvals-and-sandbox -C {shlex.quote(str(REPO_ROOT))} "
        f"--json -o {shlex.quote(str(last_message))} "
        f"\"$(cat {shlex.quote(str(prompt_path))})\" "
        f"> {shlex.quote(str(run_log))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd], check=True, text=True, capture_output=True)
    return {"state": state_key.upper(), "session_name": session_name, "status": "started"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="*", default=[])
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--session-prefix", default="packet_state_")
    parser.add_argument("--force-restart", action="store_true")
    args = parser.parse_args()

    packet_dirs = _packet_dirs_from_args(args.states)
    if not packet_dirs:
        raise SystemExit("No state packet directories found.")

    for packet_dir in packet_dirs:
        if not packet_dir.exists():
            print({"state": packet_dir.name.upper(), "session_name": "", "status": "missing_packet_dir"})
            continue
        try:
            print(
                _launch_packet(
                    packet_dir,
                    model=args.model,
                    session_prefix=args.session_prefix,
                    force_restart=args.force_restart,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print({"state": packet_dir.name.upper(), "session_name": "", "status": f"error:{type(exc).__name__}:{exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
