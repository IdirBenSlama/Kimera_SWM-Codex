import subprocess
import os

# Define the log file paths
stdout_log_path = "launcher_stdout.log"
stderr_log_path = "launcher_stderr.log"

# The command to run
command = ["python", "kimera.py"]

print(f"Executing command: {' '.join(command)}")
print(f"Redirecting stdout to: {stdout_log_path}")
print(f"Redirecting stderr to: {stderr_log_path}")

try:
    # Open log files
    with open(stdout_log_path, 'wb') as stdout_log, open(stderr_log_path, 'wb') as stderr_log:
        # Start the subprocess
        process = subprocess.Popen(
            command,
            stdout=stdout_log,
            stderr=stderr_log,
            # Use os.setsid on Unix-like systems to create a new session
            # This helps in managing the process group
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        print(f"Process started with PID: {process.pid}")
        # The script will exit here, but the subprocess continues to run.
        # This is sufficient to capture startup errors.

except Exception as e:
    print(f"Failed to launch subprocess: {e}") 