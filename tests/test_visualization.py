"""Tests for the shared phonon visualization workflow."""

from __future__ import annotations

import os

import pytest

from inspired.visualization import (
    PHONON_WEBSITE_URL,
    PhononVisualizer,
    VisualizationError,
    VisualizationTarget,
)


def test_website_generates_band_file_and_opens_default_browser(tmp_path):
    """Generate the missing band artifact before opening the website."""
    opened_urls = []

    def generate_band_yaml():
        (tmp_path / "band.yaml").write_text("phonons", encoding="utf-8")

    visualizer = PhononVisualizer(
        tmp_path,
        url_opener=lambda url: opened_urls.append(url) or True,
    )

    result = visualizer.visualize(VisualizationTarget.PHONON_WEBSITE, generate_band_yaml)

    assert result.artifact == tmp_path / "band.yaml"
    assert opened_urls == [PHONON_WEBSITE_URL]
    assert str(result.artifact) in result.message


def test_band_file_is_regenerated_when_source_is_newer(tmp_path):
    """Regenerate cached data when calculation inputs have changed."""
    source_path = tmp_path / "FORCE_CONSTANTS"
    band_path = tmp_path / "band.yaml"
    source_path.write_text("new forces", encoding="utf-8")
    band_path.write_text("old band", encoding="utf-8")
    os.utime(band_path, ns=(1, 1))
    os.utime(source_path, ns=(2, 2))
    generated = []

    def generate_band_yaml():
        generated.append(True)
        band_path.write_text("new band", encoding="utf-8")

    visualizer = PhononVisualizer(tmp_path, url_opener=lambda _url: True)
    visualizer.visualize(
        VisualizationTarget.PHONON_WEBSITE,
        generate_band_yaml,
        source_files=(source_path.name,),
    )

    assert generated == [True]
    assert band_path.read_text(encoding="utf-8") == "new band"


def test_jmol_conversion_and_launch_are_portable(tmp_path):
    """Convert band data and launch a discovered Jmol executable."""
    converter_path = tmp_path / "oclimax_convert"
    converter_path.touch()
    run_calls = []
    launch_calls = []

    def generate_band_yaml():
        (tmp_path / "band.yaml").write_text("phonons", encoding="utf-8")

    def run(command, **kwargs):
        run_calls.append((command, kwargs))
        (tmp_path / "out_gamma.phonon").write_text("modes", encoding="utf-8")

    def launch(command, **kwargs):
        launch_calls.append((command, kwargs))

    visualizer = PhononVisualizer(
        tmp_path,
        command_runner=run,
        application_launcher=launch,
        executable_finder=lambda name: "/usr/bin/jmol" if name == "jmol" else None,
        converter_path=converter_path,
    )

    result = visualizer.visualize(VisualizationTarget.JMOL, generate_band_yaml)

    assert run_calls[0][0] == [str(converter_path), "-yamld", "band.yaml", "-p"]
    assert run_calls[0][1]["cwd"] == tmp_path
    assert launch_calls[0][0] == ["/usr/bin/jmol", str(tmp_path / "out_gamma.phonon")]
    assert result.artifact == tmp_path / "out_gamma.phonon"


def test_configured_jmol_jar_uses_java(tmp_path):
    """Use Java when the user configures a Jmol archive."""
    jar_path = tmp_path / "Jmol.jar"
    jar_path.touch()
    (tmp_path / "band.yaml").touch()
    (tmp_path / "out_gamma.phonon").touch()
    launches = []

    visualizer = PhononVisualizer(
        tmp_path,
        application_launcher=lambda command, **kwargs: launches.append((command, kwargs)),
        executable_finder=lambda name: "/usr/bin/java" if name == "java" else None,
        environment={"INSPIRED_JMOL_JAR": str(jar_path)},
    )

    visualizer.visualize(VisualizationTarget.JMOL, lambda: None)

    assert launches[0][0] == [
        "/usr/bin/java",
        "-jar",
        str(jar_path),
        str(tmp_path / "out_gamma.phonon"),
    ]


def test_browser_failure_has_actionable_error(tmp_path):
    """Report a useful error when no system browser can be opened."""
    (tmp_path / "band.yaml").touch()
    visualizer = PhononVisualizer(tmp_path, url_opener=lambda _url: False)

    with pytest.raises(VisualizationError, match="default web browser"):
        visualizer.visualize(VisualizationTarget.PHONON_WEBSITE, lambda: None)


def test_generator_must_create_band_file(tmp_path):
    """Reject generators that do not produce the required artifact."""
    visualizer = PhononVisualizer(tmp_path, url_opener=lambda _url: True)

    with pytest.raises(VisualizationError, match="was not created"):
        visualizer.visualize(VisualizationTarget.PHONON_WEBSITE, lambda: None)
