"""Stream child-process logs through notebook stdout and preserve failures."""

import subprocess


def run_logged(arguments):
    with subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, arguments)
