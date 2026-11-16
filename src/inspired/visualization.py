"""Portable orchestration for external phonon visualization tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PHONON_WEBSITE_URL = "https://henriquemiranda.github.io/phononwebsite/phonon.html"


class VisualizationError(RuntimeError):
    """Raised when visualization data cannot be generated or opened."""


class VisualizationTarget(str, Enum):
    """Visualization destinations supported by INSPIRED."""

    JMOL = "jmol"
    PHONON_WEBSITE = "phonon-website"


@dataclass(frozen=True)
class VisualizationResult:
    """Describe the artifact prepared for a visualization target."""

    target: VisualizationTarget
    artifact: Path
    message: str


class PhononVisualizer:
    """Prepare phonon artifacts and launch a visualization destination.

    External integrations live here instead of in individual calculation
    workers. This keeps DFT and MLFF workflows consistent and makes new
    visualization targets straightforward to add and test.
    """

    BAND_FILENAME = "band.yaml"
    JMOL_FILENAME = "out_gamma.phonon"
    JMOL_JAR_ENV = "INSPIRED_JMOL_JAR"
    CONVERTER_ENV = "INSPIRED_OCLIMAX_CONVERT"

    def __init__(
        self,
        workdir: str | os.PathLike[str],
        *,
        command_runner: Callable[..., object] = subprocess.run,
        application_launcher: Callable[..., object] = subprocess.Popen,
        url_opener: Callable[[str], bool] = webbrowser.open_new_tab,
        executable_finder: Callable[[str], str | None] = shutil.which,
        environment: Mapping[str, str] | None = None,
        converter_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self._run = command_runner
        self._launch = application_launcher
        self._open_url = url_opener
        self._which = executable_finder
        self._environment = os.environ if environment is None else environment
        self._converter_path = Path(converter_path).expanduser() if converter_path else None

    def visualize(
        self,
        target: VisualizationTarget,
        generate_band_yaml: Callable[[], object],
        *,
        source_files: Iterable[str | os.PathLike[str]] = (),
    ) -> VisualizationResult:
        """Prepare current artifacts and open *target*.

        ``generate_band_yaml`` is supplied by the active calculation worker,
        while this class owns caching, conversion, and application discovery.
        """
        if not self.workdir.is_dir():
            raise VisualizationError(f"Working directory does not exist: {self.workdir}")

        sources = tuple(self._resolve_source(path) for path in source_files)
        band_path = self._ensure_band_yaml(generate_band_yaml, sources)

        if target is VisualizationTarget.PHONON_WEBSITE:
            self._open_phonon_website()
            return VisualizationResult(
                target=target,
                artifact=band_path,
                message=f"Phonon Website opened. Upload {band_path} on the page.",
            )

        if target is VisualizationTarget.JMOL:
            jmol_path = self._ensure_jmol_file(band_path)
            self._launch_jmol(jmol_path)
            return VisualizationResult(
                target=target,
                artifact=jmol_path,
                message=f"Opened Gamma-point modes from {jmol_path} in Jmol.",
            )

        raise VisualizationError(f"Unsupported visualization target: {target}")

    def _resolve_source(self, path: str | os.PathLike[str]) -> Path:
        source = Path(path)
        return source if source.is_absolute() else self.workdir / source

    def _ensure_band_yaml(self, generator: Callable[[], object], sources: tuple[Path, ...]) -> Path:
        band_path = self.workdir / self.BAND_FILENAME
        if self._is_stale(band_path, sources):
            print("INFO: Generating band.yaml for phonon visualization. This may take a few moments.")
            try:
                generator()
            except Exception as error:
                raise VisualizationError(f"Could not generate {self.BAND_FILENAME}: {error}") from error
        if not band_path.is_file():
            raise VisualizationError(
                f"{self.BAND_FILENAME} was not created in {self.workdir}. "
                "Complete the phonon calculation and try again."
            )
        return band_path

    def _ensure_jmol_file(self, band_path: Path) -> Path:
        jmol_path = self.workdir / self.JMOL_FILENAME
        if self._is_stale(jmol_path, (band_path,)):
            converter = self._find_converter()
            try:
                self._run(
                    [str(converter), "-yamld", band_path.name, "-p"],
                    cwd=self.workdir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                detail = getattr(error, "stderr", None) or str(error)
                raise VisualizationError(f"OCLIMAX could not create {self.JMOL_FILENAME}: {detail}") from error
        if not jmol_path.is_file():
            raise VisualizationError(f"OCLIMAX did not create {jmol_path}.")
        return jmol_path

    def _find_converter(self) -> Path:
        candidates: list[str | os.PathLike[str] | None] = [
            self._converter_path,
            self._environment.get(self.CONVERTER_ENV),
            self._which("oclimax_convert"),
            Path(__file__).parent / "oclimax" / "oclimax_convert",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).expanduser().is_file():
                return Path(candidate).expanduser().resolve()
        raise VisualizationError(
            "Could not find oclimax_convert. Install INSPIRED with its package data "
            f"or set {self.CONVERTER_ENV} to the executable path."
        )

    def _launch_jmol(self, jmol_path: Path) -> None:
        command = self._find_jmol_command()
        try:
            self._launch([*command, str(jmol_path)], cwd=self.workdir, start_new_session=True)
        except OSError as error:
            raise VisualizationError(f"Could not start Jmol: {error}") from error

    def _find_jmol_command(self) -> list[str]:
        configured_jar = self._environment.get(self.JMOL_JAR_ENV)
        if configured_jar:
            return self._java_jar_command(Path(configured_jar).expanduser(), configured=True)

        jmol_executable = self._which("jmol")
        if jmol_executable:
            return [jmol_executable]

        for jar_path in (Path("/usr/share/java/Jmol.jar"), Path("/usr/share/jmol/Jmol.jar")):
            if jar_path.is_file():
                return self._java_jar_command(jar_path)

        raise VisualizationError(
            "Jmol was not found. Install the 'jmol' command or set "
            f"{self.JMOL_JAR_ENV}=/path/to/Jmol.jar."
        )

    def _java_jar_command(self, jar_path: Path, *, configured: bool = False) -> list[str]:
        if not jar_path.is_file():
            prefix = f"{self.JMOL_JAR_ENV} points to" if configured else "Jmol archive"
            raise VisualizationError(f"{prefix} a missing file: {jar_path}")
        java_executable = self._which("java")
        if not java_executable:
            raise VisualizationError("Java is required to launch the configured Jmol archive.")
        return [java_executable, "-jar", str(jar_path.resolve())]

    def _open_phonon_website(self) -> None:
        try:
            opened = self._open_url(PHONON_WEBSITE_URL)
        except webbrowser.Error as error:
            raise VisualizationError(f"Could not open the default web browser: {error}") from error
        if opened is False:
            raise VisualizationError(
                f"No default web browser is available. Open {PHONON_WEBSITE_URL} manually."
            )

    @staticmethod
    def _is_stale(output: Path, inputs: Iterable[Path]) -> bool:
        if not output.is_file():
            return True
        output_mtime = output.stat().st_mtime_ns
        return any(path.is_file() and path.stat().st_mtime_ns > output_mtime for path in inputs)
