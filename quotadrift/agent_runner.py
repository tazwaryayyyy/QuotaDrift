import os
import subprocess
import tempfile


def run_code(code: str, language: str, timeout: int = 5) -> dict:
    """
    Executes code in a subprocess and returns the result.
    Supported languages: python, javascript
    """
    if language.lower() in ["python", "py"]:
        return _run_python(code, timeout)
    elif language.lower() in ["javascript", "js", "node"]:
        return _run_javascript(code, timeout)
    else:
        return {"error": f"Language '{language}' not supported."}


def _run_python(code: str, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(code.encode("utf-8"))
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {timeout} seconds."}
    except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_javascript(code: str, timeout: int) -> dict:
    # Check if node is installed
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"error": "Node.js is not installed on the host."}

    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tmp:
        tmp.write(code.encode("utf-8"))
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["node", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {timeout} seconds."}
    except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
