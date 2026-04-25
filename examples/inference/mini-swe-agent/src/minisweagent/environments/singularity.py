#!/usr/bin/env python3

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SingularityEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity executable."""
    sandbox_build_retries: int = 3
    """Number of retries for building the sandbox if an error occurs."""
    writable_testbed_dir: str = ""
    """If set, copy /testbed from sandbox to this local directory and bind-mount it.
    This avoids writing to the NFS-cached sandbox, preventing quota issues.
    Defaults to a temp directory under /tmp when APPTAINER_SANDBOX_DIR is set."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self._local_testbed: Path | None = None
        self._is_cached_sandbox = False
        self.sandbox_dir = self._build_sandbox()

    def _build_sandbox(self) -> Path:
        # Check for a pre-built sandbox in the cache directory
        cached = self._find_cached_sandbox()
        if cached is not None:
            self._is_cached_sandbox = True
            self._setup_local_testbed(cached)
            return cached

        # Building the sandbox can fail (very rarely), so we retry it
        max_retries = self.config.sandbox_build_retries
        for attempt in range(max_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
            try:
                subprocess.run(
                    [self.config.executable, "build", "--sandbox", sandbox_dir, self.config.image],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.error(
                    f"Error building image {self.config.image}, stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise

        # Cache the built sandbox for future reuse
        self._cache_sandbox(sandbox_dir)
        return sandbox_dir

    def _find_cached_sandbox(self) -> Path | None:
        cache_dir = os.getenv("APPTAINER_SANDBOX_DIR")
        if not cache_dir:
            return None
        instance_id = self._get_instance_id()
        if not instance_id:
            return None
        cached = Path(cache_dir) / instance_id
        if cached.is_dir():
            return cached
        return None

    def _setup_local_testbed(self, sandbox_dir: Path) -> None:
        """Copy /testbed from cached sandbox to a local directory for bind-mounting."""
        testbed_src = sandbox_dir / "testbed"
        if not testbed_src.is_dir():
            self.logger.warning(f"No /testbed found in cached sandbox {sandbox_dir}, skipping local testbed setup")
            return

        # Determine local directory
        writable_dir = self.config.writable_testbed_dir
        if writable_dir:
            local_dir = Path(writable_dir)
        else:
            local_dir = Path(tempfile.gettempdir())

        local_testbed_dir = local_dir / f"minisweagent-testbed-{uuid.uuid4().hex[:8]}"
        try:
            shutil.copytree(testbed_src, local_testbed_dir / "testbed")
            self._local_testbed = local_testbed_dir
            self.logger.info(f"Copied /testbed to local: {local_testbed_dir} (bind-mount will override /testbed)")
        except Exception as e:
            self.logger.warning(f"Failed to copy /testbed to local directory: {e}")
            self._local_testbed = None

    def _cache_sandbox(self, sandbox_dir: Path) -> None:
        cache_dir = os.getenv("APPTAINER_SANDBOX_DIR")
        if not cache_dir:
            return
        instance_id = self._get_instance_id()
        if not instance_id:
            return
        dest = Path(cache_dir) / instance_id
        if dest.exists():
            return
        try:
            shutil.copytree(sandbox_dir, dest)
            self.logger.info(f"Cached sandbox to {dest}")
        except Exception as e:
            self.logger.warning(f"Failed to cache sandbox: {e}")

    def _get_instance_id(self) -> str | None:
        """Extract instance_id from the image URL.
        e.g. docker://docker.io/swebench/sweb.eval.x86_64.django_1776_django-10914:latest
             -> django__django-10914
        """
        image = self.config.image
        if not image:
            return None
        # Extract the part after the last dot and before :latest
        filename = image.rsplit("/", 1)[-1]  # sweb.eval.x86_64.django_1776_django-10914:latest
        name = filename.rsplit(":", 1)[0]     # sweb.eval.x86_64.django_1776_django-10914
        # Remove prefix "sweb.eval.x86_64."
        prefix = "sweb.eval.x86_64."
        if name.startswith(prefix):
            name = name[len(prefix):]
        # Convert _1776_ back to __
        instance_id = name.replace("_1776_", "__")
        return instance_id

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        cmd = [self.config.executable, "exec"]

        # Do not inherit directories and env vars from host
        cmd.extend(["--contain", "--cleanenv"])

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        # If using a cached sandbox with local testbed, bind-mount it
        if self._local_testbed is not None:
            cmd.extend(["--writable"])
            cmd.extend(["--bind", f"{self._local_testbed / 'testbed'}:/testbed"])
        elif self._is_cached_sandbox:
            # Cached sandbox without local testbed fallback — still need writable
            cmd.extend(["--writable"])
        else:
            # Fresh sandbox in /tmp — writable by default
            cmd.extend(["--writable"])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend([str(self.sandbox_dir), "bash", "-c", command])
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        # Clean up local testbed copy if present
        if self._local_testbed is not None:
            shutil.rmtree(self._local_testbed, ignore_errors=True)
            self._local_testbed = None

        # Don't delete cached sandboxes, only delete newly built ones
        cache_dir = os.getenv("APPTAINER_SANDBOX_DIR", "")
        if cache_dir and str(self.sandbox_dir).startswith(cache_dir):
            return
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
