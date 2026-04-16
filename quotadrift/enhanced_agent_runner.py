"""
Enhanced Agent Runner with Sandboxing and Multi-Language Support.

Features:
- Docker-based sandboxing for security
- Support for Python, JavaScript, Go, Rust, Java, C++
- Resource limits (CPU, memory, network)
- Timeout enforcement
- Secure file handling
- Output capture and error handling
"""

import json
import logging
import os
import pathlib
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger("agent_runner")


@dataclass
class SandboxConfig:
    """Sandbox configuration for code execution."""

    timeout: int = 10  # Execution timeout in seconds
    memory_limit: str = "128m"  # Memory limit
    cpu_limit: str = "0.5"  # CPU limit
    network_access: bool = False  # Network access
    temp_dir_size: str = "10m"  # Temp directory size limit


@dataclass
class ExecutionResult:
    """Result of code execution."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    language: str
    error: str | None = None


class LanguageConfig:
    """Configuration for supported programming languages."""

    LANGUAGES = {
        "python": {
            "extensions": [".py"],
            "dockerfile": """
FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY main.py .
RUN chmod +x main.py
CMD ["python", "main.py"]
            """.strip(),
            "files": ["main.py", "requirements.txt"],
            "entrypoint": "python main.py",
        },
        "javascript": {
            "extensions": [".js"],
            "dockerfile": """
FROM node:18-alpine
WORKDIR /code
COPY package*.json ./
RUN npm install --no-cache
COPY main.js .
RUN chmod +x main.js
CMD ["node", "main.js"]
            """.strip(),
            "files": ["main.js", "package.json"],
            "entrypoint": "node main.js",
        },
        "go": {
            "extensions": [".go"],
            "dockerfile": """
FROM golang:1.21-alpine
WORKDIR /code
COPY main.go .
RUN go mod init main
RUN go mod tidy
RUN go build -o main main.go
RUN chmod +x main
CMD ["./main"]
            """.strip(),
            "files": ["main.go"],
            "entrypoint": "./main",
        },
        "rust": {
            "extensions": [".rs"],
            "dockerfile": """
FROM rust:1.75-alpine
WORKDIR /code
COPY main.rs .
RUN cargo init --name main
RUN cargo build --release
RUN chmod +x target/release/main
CMD ["./target/release/main"]
            """.strip(),
            "files": ["main.rs", "Cargo.toml"],
            "entrypoint": "./target/release/main",
        },
        "java": {
            "extensions": [".java"],
            "dockerfile": """
FROM openjdk:17-slim
WORKDIR /code
COPY Main.java .
RUN javac Main.java
CMD ["java", "Main"]
            """.strip(),
            "files": ["Main.java"],
            "entrypoint": "java Main",
        },
        "cpp": {
            "extensions": [".cpp"],
            "dockerfile": """
FROM gcc:13-alpine
WORKDIR /code
COPY main.cpp .
RUN g++ -o main main.cpp -std=c++17
RUN chmod +x main
CMD ["./main"]
            """.strip(),
            "files": ["main.cpp"],
            "entrypoint": "./main",
        },
    }


class EnhancedAgentRunner:
    """Enhanced agent runner with Docker sandboxing."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self.temp_dir = tempfile.mkdtemp(prefix="quotadrift_sandbox_")

    def detect_language(self, code: str, filename: str | None = None) -> str:
        """Detect programming language from code or filename."""
        if filename:
            ext = pathlib.Path(filename).suffix.lower()
            for lang, config in LanguageConfig.LANGUAGES.items():
                if ext in config["extensions"]:
                    return lang

        # Simple heuristics for language detection
        code_lower = code.lower().strip()

        if code_lower.startswith("def ") or "import " in code_lower:
            return "python"
        elif (
            "function " in code_lower or "const " in code_lower or "let " in code_lower
        ):
            return "javascript"
        elif "package main" in code_lower or "func " in code_lower:
            return "go"
        elif "fn main()" in code_lower or "use std::" in code_lower:
            return "rust"
        elif "public class" in code_lower or "public static void main" in code_lower:
            return "java"
        elif "#include" in code_lower and "int main" in code_lower:
            return "cpp"

        return "python"  # Default fallback

    async def run_code(
        self, code: str, language: str | None, filename: str | None = None
    ) -> ExecutionResult:
        """Execute code in a Docker sandbox."""
        if not language:
            language = self.detect_language(code, filename)

        if language not in LanguageConfig.LANGUAGES:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                execution_time=0,
                language=language,
                error=f"Language '{language}' not supported",
            )

        try:
            # Create sandbox environment
            sandbox_id = str(uuid.uuid4())[:8]
            work_dir = os.path.join(self.temp_dir, sandbox_id)
            os.makedirs(work_dir, exist_ok=True)

            # Prepare files
            await self._prepare_files(work_dir, code, language, filename)

            # Build and run in Docker
            result = await self._run_in_docker(work_dir, language)

            return result

        except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            logger.error("Error executing %s code: %s", language, exc)
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                execution_time=0,
                language=language,
                error=str(exc),
            )
        finally:
            # Cleanup
            try:
                import shutil

                shutil.rmtree(work_dir, ignore_errors=True)
            except OSError:
                pass

    async def _prepare_files(
        self, work_dir: str, code: str, language: str, _filename: str | None = None
    ):
        """Prepare source files for execution."""
        lang_config = LanguageConfig.LANGUAGES[language]

        # Write main source file
        main_file = os.path.join(
            work_dir, "main" + lang_config["extensions"][0])
        with open(main_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Create additional files if needed
        for file_template in lang_config["files"]:
            if file_template == "main.py" and language == "python":
                # Create requirements.txt for Python
                req_file = os.path.join(work_dir, "requirements.txt")
                with open(req_file, "w", encoding="utf-8") as f:
                    f.write("requests\nnumpy\npandas")  # Common packages
            elif file_template == "package.json" and language == "javascript":
                # Create package.json for Node.js
                pkg_file = os.path.join(work_dir, "package.json")
                package_json = {
                    "name": "sandbox-code",
                    "version": "1.0.0",
                    "dependencies": {},
                }
                with open(pkg_file, "w", encoding="utf-8") as f:
                    json.dump(package_json, f, indent=2)
            elif file_template == "Cargo.toml" and language == "rust":
                # Create Cargo.toml for Rust
                cargo_file = os.path.join(work_dir, "Cargo.toml")
                cargo_toml = """
[package]
name = "main"
version = "0.1.0"
edition = "2021"

[dependencies]
                """.strip()
                with open(cargo_file, "w", encoding="utf-8") as f:
                    f.write(cargo_toml)

    async def _run_in_docker(self, work_dir: str, language: str) -> ExecutionResult:
        """Run code in Docker container with resource limits."""
        import time

        lang_config = LanguageConfig.LANGUAGES[language]
        container_name = f"quotadrift_{language}_{os.path.basename(work_dir)}"

        # Create Dockerfile
        dockerfile_path = os.path.join(work_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(lang_config["dockerfile"])

        # Build Docker image
        build_cmd = ["docker", "build", "-t", container_name, work_dir]

        try:
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if build_result.returncode != 0:
                return ExecutionResult(
                    stdout="",
                    stderr=build_result.stderr,
                    exit_code=1,
                    execution_time=0,
                    language=language,
                    error=f"Docker build failed: {build_result.stderr}",
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                execution_time=0,
                language=language,
                error="Docker build timed out",
            )

        # Run container with resource limits
        run_cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--memory",
            self.config.memory_limit,
            "--cpus",
            self.config.cpu_limit,
            "--network",
            "none" if not self.config.network_access else "bridge",
            "-v",
            f"{work_dir}:/code:ro",
            container_name,
            lang_config["entrypoint"],
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                check=False,
            )
            execution_time = time.time() - start_time

            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                language=language,
            )

        except subprocess.TimeoutExpired:
            # Kill the container
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                check=False,
            )
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=124,
                execution_time=self.config.timeout,
                language=language,
                error=f"Execution timed out after {self.config.timeout} seconds",
            )

        finally:
            # Clean up Docker image
            subprocess.run(
                ["docker", "rmi", container_name],
                capture_output=True,
                check=False,
            )

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return list(LanguageConfig.LANGUAGES.keys())

    def get_language_info(self, language: str) -> dict:
        """Get information about a supported language."""
        if language not in LanguageConfig.LANGUAGES:
            return {"error": f"Language '{language}' not supported"}

        config = LanguageConfig.LANGUAGES[language]
        return {
            "language": language,
            "extensions": config["extensions"],
            "supported": True,
            "has_package_manager": language in ["python", "javascript", "rust", "go"],
        }

    def cleanup(self):
        """Clean up temporary directories."""
        try:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError:
            pass


# Fallback for systems without Docker
class SimpleAgentRunner:
    """Simple fallback runner for systems without Docker."""

    def __init__(self):
        from quotadrift import agent_runner  # Import original runner

        self.fallback_runner = agent_runner

    async def run_code(
        self, code: str, language: str | None, filename: str | None = None
    ) -> ExecutionResult:
        """Fallback to simple subprocess execution."""
        if not language:
            language = self.detect_language(code, filename)

        if language not in ["python", "javascript"]:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                execution_time=0,
                language=language,
                error=f"Language '{language}' not supported without Docker",
            )

        # Use original runner for Python and JavaScript
        result = self.fallback_runner.run_code(code, language, timeout=10)

        return ExecutionResult(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            exit_code=result.get("exit_code", 1),
            execution_time=0,  # Not tracked in original
            language=language,
            error=result.get("error"),
        )

    def detect_language(self, code: str, filename: str | None = None) -> str:
        """Simple language detection."""
        if filename:
            ext = pathlib.Path(filename).suffix.lower()
            if ext == ".py":
                return "python"
            elif ext in [".js", ".mjs"]:
                return "javascript"

        code_lower = code.lower()
        if "def " in code_lower or "import " in code_lower:
            return "python"
        elif "function " in code_lower or "const " in code_lower:
            return "javascript"
        return "python"


@lru_cache(maxsize=1)
def get_runner() -> EnhancedAgentRunner | SimpleAgentRunner:
    """Get the appropriate runner instance."""
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"],
                       capture_output=True, check=True)
        logger.info("Using Docker-based sandbox runner")
        return EnhancedAgentRunner()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Docker not available, using simple runner")
        return SimpleAgentRunner()
