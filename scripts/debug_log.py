import sys
from pathlib import Path

def debug_log():
    log_path = Path('demo_output.txt')
    if not log_path.exists():
        print("Log file not found.")
        return
        
    content = log_path.read_bytes()
    try:
        text = content.decode('utf-16')
    except:
        text = content.decode('utf-8', errors='ignore')
        
    lines = text.splitlines()
    print("--- LAST 50 LINES OF LOG ---")
    for line in lines[-50:]:
        print(line)

if __name__ == "__main__":
    debug_log()
