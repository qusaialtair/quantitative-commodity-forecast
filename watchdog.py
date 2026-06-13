import os
import time
import subprocess
import threading
from datetime import datetime, timezone

# The folder we are watching (Current Directory)
WATCH_DIR = "."
# Folders to ignore so we don't scan virtual environments or git
IGNORE_DIRS = {"venv", ".git", "__pycache__", ".claude"}

# Daily LSTM update fires once per day after market close (22:00 UTC ~ 6 PM ET)
DAILY_UPDATE_HOUR_UTC = 22
_lstm_last_update_date = None   # tracks last date the update ran


def get_file_mod_times():
    file_times = {}
    for root, dirs, files in os.walk(WATCH_DIR):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if file.endswith(".py") and file != "watchdog.py":
                filepath = os.path.join(root, file)
                try:
                    file_times[filepath] = os.path.getmtime(filepath)
                except OSError:
                    pass
    return file_times


def run_daily_lstm_update():
    """
    Called in a background thread once per day after market close.
    Runs models/lstm_predictor.py update which fine-tunes on the day's new data.
    """
    print("\n[Watchdog] ── Daily LSTM update triggered ────────────────────────")
    result = subprocess.run(
        ["python3", "models/lstm_predictor.py", "update"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[Watchdog] LSTM daily update complete.")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
    else:
        print("[Watchdog] LSTM daily update FAILED — invoking self-heal.")
        err = result.stderr.strip()
        print(f"  {err[:400]}")
        prompt = (
            "The LSTM daily update (models/lstm_predictor.py update) threw this error:\n\n"
            + err + "\n\nFix the file so the update loop works correctly. No explanation needed."
        )
        subprocess.run(["claude", prompt, "--dangerously-skip-permissions"])


def _check_daily_update():
    """Returns True if we should fire the daily LSTM update right now."""
    global _lstm_last_update_date
    now = datetime.now(timezone.utc)
    today = now.date()
    if now.hour >= DAILY_UPDATE_HOUR_UTC and _lstm_last_update_date != today:
        _lstm_last_update_date = today
        return True
    return False


def main():
    print("Watchdog active. Scanning entire project for changes...")
    print(f"Daily LSTM update scheduled after {DAILY_UPDATE_HOUR_UTC}:00 UTC each day.")
    last_times = get_file_mod_times()

    while True:
        time.sleep(2)
        current_times = get_file_mod_times()

        # ── File-change loop ─────────────────────────────────────────────────
        for filepath, mod_time in current_times.items():
            if filepath not in last_times or mod_time > last_times[filepath]:
                print(f"\n[Watchdog] Detected change in: {filepath}")
                print(f"[Watchdog] Executing {filepath}...")

                # Run the modified script
                result = subprocess.run(["python3", filepath], capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"[Watchdog] Error detected in {filepath}!")
                    error_msg = result.stderr.strip()

                    # Wake up Claude to fix it automatically WITHOUT asking for permission
                    print("[Watchdog] Summoning Claude to self-heal...")
                    prompt = f"The file {filepath} just threw this error when executed:\n\n{error_msg}\n\nAnalyze the error, rewrite the file to fix it, and save it. Do not explain, just fix the code."

                    subprocess.run([
                        "claude",
                        prompt,
                        "--dangerously-skip-permissions"
                    ])
                    print("[Watchdog] Claude has completed the fix attempt. Re-running to verify...")
                    verify = subprocess.run(["python3", filepath], capture_output=True, text=True)
                    if verify.returncode == 0:
                        print(f"[Watchdog] Fix verified! {filepath} runs successfully. Resuming watch...")
                    else:
                        print(f"[Watchdog] Fix attempt failed. Still erroring:\n{verify.stderr.strip()}")
                    # Update mtime so watchdog doesn't re-trigger on the same fix
                    last_times[filepath] = os.path.getmtime(filepath)
                else:
                    print(f"[Watchdog] {filepath} ran successfully!")

        last_times = current_times

        # ── Daily LSTM update (non-blocking background thread) ───────────────
        if _check_daily_update():
            t = threading.Thread(target=run_daily_lstm_update, daemon=True)
            t.start()


if __name__ == "__main__":
    main()
