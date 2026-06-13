import os
import time
import subprocess

# The folder we are watching (Current Directory)
WATCH_DIR = "."
# Folders to ignore so we don't scan virtual environments or git
IGNORE_DIRS = {"venv", ".git", "__pycache__", ".claude"}

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

def main():
    print("Watchdog active. Scanning entire project for changes...")
    last_times = get_file_mod_times()
    
    while True:
        time.sleep(2)
        current_times = get_file_mod_times()
        
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
                    print("[Watchdog] Claude has completed the fix attempt.")
                else:
                    print(f"[Watchdog] {filepath} ran successfully!")
                
        last_times = current_times

if __name__ == "__main__":
    main()