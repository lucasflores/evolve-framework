"""
Tests for ERP standalone examples.

Ensures that all standalone examples execute without errors
and produce expected results.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def run_example(script_name: str, timeout: int = 60) -> tuple[int, str, str]:
    """
    Run an example script and capture output.

    Args:
        script_name: Name of the Python script (e.g., "sexual_selection.py")
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    script_path = EXAMPLES_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Example script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, timeout=timeout
    )

    return result.returncode, result.stdout, result.stderr


class TestSexualSelectionExample:
    """Tests for sexual_selection.py standalone example."""

    def test_sexual_selection_runs_successfully(self):
        """Test that sexual selection example completes without errors."""
        returncode, stdout, stderr = run_example("sexual_selection.py", timeout=120)

        assert returncode == 0, f"Script failed with stderr: {stderr}"
        assert "SEXUAL SELECTION WITH ERP" in stdout
        assert "Example complete!" in stdout

    def test_sexual_selection_shows_results(self):
        """Test that results analysis is printed."""
        returncode, stdout, stderr = run_example("sexual_selection.py", timeout=120)

        assert returncode == 0
        assert "SEXUAL SELECTION RESULTS" in stdout
        assert "Fitness Evolution:" in stdout
        assert "Initial best:" in stdout
        assert "Final best:" in stdout

    def test_sexual_selection_creates_population(self):
        """Test that population creation works."""
        returncode, stdout, stderr = run_example("sexual_selection.py", timeout=120)

        assert returncode == 0
        assert "Creating population with sexual selection..." in stdout
        assert "choosy individuals" in stdout
        assert "eager individuals" in stdout

    def test_sexual_selection_detects_selection_effect(self):
        """Test that analysis detects selection effects."""
        returncode, stdout, stderr = run_example("sexual_selection.py", timeout=120)

        assert returncode == 0
        # Should show either strong or weak effect
        assert (
            "sexual selection effect" in stdout
            or "selection effect" in stdout
            or "Weak selection effect" in stdout
        )


class TestSpeciationDemoExample:
    """Tests for speciation_demo.py standalone example."""

    def test_speciation_runs_successfully(self):
        """Test that speciation demo completes without errors."""
        returncode, stdout, stderr = run_example("speciation_demo.py", timeout=180)

        assert returncode == 0, f"Script failed with stderr: {stderr}"
        assert "SPECIATION VIA ASSORTATIVE MATING" in stdout
        assert "Example complete!" in stdout

    def test_speciation_shows_analysis(self):
        """Test that speciation analysis is printed."""
        returncode, stdout, stderr = run_example("speciation_demo.py", timeout=180)

        assert returncode == 0
        assert "SPECIATION ANALYSIS" in stdout
        assert "Population Diversity:" in stdout
        assert "Assortative Mating Effect:" in stdout

    def test_speciation_uses_cosine_similarity(self):
        """Test that cosine similarity matchability is mentioned."""
        returncode, stdout, stderr = run_example("speciation_demo.py", timeout=180)

        assert returncode == 0
        assert "cosine similarity" in stdout
        assert "similar genomes" in stdout

    def test_speciation_creates_assortative_population(self):
        """Test that assortative population is created."""
        returncode, stdout, stderr = run_example("speciation_demo.py", timeout=180)

        assert returncode == 0
        assert "Creating assortative mating population..." in stdout
        assert "individuals with cosine similarity matchability" in stdout


class TestProtocolEvolutionExample:
    """Tests for protocol_evolution.py standalone example."""

    def test_protocol_evolution_runs_successfully(self):
        """Test that protocol evolution tracking completes without errors."""
        returncode, stdout, stderr = run_example("protocol_evolution.py", timeout=180)

        assert returncode == 0, f"Script failed with stderr: {stderr}"
        assert "PROTOCOL EVOLUTION TRACKING" in stdout
        assert "Example complete!" in stdout

    def test_protocol_evolution_shows_summary(self):
        """Test that evolution summary is printed."""
        returncode, stdout, stderr = run_example("protocol_evolution.py", timeout=180)

        assert returncode == 0
        assert "PROTOCOL EVOLUTION SUMMARY" in stdout
        assert "Matchability Thresholds:" in stdout
        assert "Crossover Type Distribution:" in stdout
        assert "Swap Probabilities" in stdout

    def test_protocol_evolution_shows_timeline(self):
        """Test that timeline is printed."""
        returncode, stdout, stderr = run_example("protocol_evolution.py", timeout=180)

        assert returncode == 0
        assert "PROTOCOL EVOLUTION TIMELINE" in stdout
        assert "Generation 0:" in stdout

    def test_protocol_evolution_tracks_parameters(self):
        """Test that protocol parameters are tracked."""
        returncode, stdout, stderr = run_example("protocol_evolution.py", timeout=180)

        assert returncode == 0
        assert "Threshold:" in stdout
        assert "Swap prob:" in stdout
        assert "Crossovers:" in stdout

    def test_protocol_evolution_shows_fitness(self):
        """Test that fitness evolution is reported."""
        returncode, stdout, stderr = run_example("protocol_evolution.py", timeout=180)

        assert returncode == 0
        assert "FITNESS EVOLUTION" in stdout
        assert "Initial:" in stdout
        assert "Final:" in stdout
        assert "Improvement:" in stdout


class TestAllExamplesTogether:
    """Integration tests for all examples together."""

    @pytest.mark.slow
    def test_all_examples_complete_within_time_limit(self):
        """Test that all examples complete within reasonable time."""
        examples = [
            ("sexual_selection.py", 120),
            ("speciation_demo.py", 180),
            ("protocol_evolution.py", 180),
        ]

        for script, timeout in examples:
            returncode, stdout, stderr = run_example(script, timeout=timeout)
            assert returncode == 0, f"{script} failed: {stderr}"
            assert "Example complete!" in stdout

    def test_all_examples_produce_output(self):
        """Test that all examples produce meaningful output."""
        examples = ["sexual_selection.py", "speciation_demo.py", "protocol_evolution.py"]

        for script in examples:
            returncode, stdout, stderr = run_example(script, timeout=180)
            assert returncode == 0
            assert len(stdout) > 1000, f"{script} produced insufficient output"
            assert "=" * 70 in stdout, f"{script} missing formatted sections"

    def test_no_examples_crash(self):
        """Test that no examples crash with uncaught exceptions."""
        examples = ["sexual_selection.py", "speciation_demo.py", "protocol_evolution.py"]

        for script in examples:
            returncode, stdout, stderr = run_example(script, timeout=180)
            assert returncode == 0, f"{script} crashed: {stderr}"
            # Check for common error indicators
            assert "Traceback" not in stderr
            assert (
                "Error:" not in stderr or "Error" in stdout
            )  # Allow intentional error messages in stdout
