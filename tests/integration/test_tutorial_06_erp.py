"""
Integration test for Tutorial 6: Evolvable Reproduction Protocols.

Ensures the tutorial notebook executes without errors and produces expected outputs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TUTORIAL_PATH = (
    Path(__file__).parent.parent.parent / "docs" / "tutorials" / "07_evolvable_reproduction.ipynb"
)

try:
    import nbconvert  # noqa: F401

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


def execute_notebook(notebook_path: Path, timeout: int = 900) -> tuple[bool, str]:
    """
    Execute a Jupyter notebook and return success status.

    Args:
        notebook_path: Path to .ipynb file
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=" + str(timeout),
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )

        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, f"Execution exceeded {timeout} seconds"
    except Exception as e:
        return False, str(e)


def get_notebook_cell_count(notebook_path: Path) -> int:
    """Count cells in notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)
    return len(nb["cells"])


class TestTutorial06ERP:
    """Tests for Tutorial 6: Evolvable Reproduction Protocols."""

    def test_notebook_exists(self):
        """Test that the notebook file exists."""
        assert TUTORIAL_PATH.exists(), f"Notebook not found at {TUTORIAL_PATH}"

    def test_notebook_has_minimum_cells(self):
        """Test that notebook has expected number of cells."""
        cell_count = get_notebook_cell_count(TUTORIAL_PATH)
        assert cell_count >= 45, f"Expected at least 45 cells, got {cell_count}"

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not JUPYTER_AVAILABLE, reason="jupyter not installed")
    def test_notebook_executes_successfully(self):
        """Test that entire notebook executes without errors."""
        success, error_msg = execute_notebook(TUTORIAL_PATH, timeout=900)

        if not success:
            pytest.fail(f"Notebook execution failed:\n{error_msg}")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not JUPYTER_AVAILABLE, reason="jupyter not installed")
    def test_notebook_execution_time(self):
        """Test that notebook completes within reasonable time."""
        import time

        start = time.time()
        success, error_msg = execute_notebook(TUTORIAL_PATH, timeout=900)
        duration = time.time() - start

        assert success, f"Notebook execution failed: {error_msg}"
        assert duration < 900, f"Execution took {duration:.0f}s, expected < 900s"

        # Log actual time for monitoring
        print(f"\n✓ Tutorial 6 executed in {duration:.1f} seconds")

    def test_notebook_has_required_sections(self):
        """Test that notebook has all required section markers."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        required_sections = [
            "Part 0: Setup and Imports",
            "Part 1: ERP Primer",
            "Part 2: The Three Protocol Components",
            "Part 3: Building Protocols",
            "Part 4: Running ERP Evolution",
            "Part 5: Protocol Evolution",
            "Part 6: Recovery Mechanisms",
            "Part 7: Case Study - Sexual Selection",
            "Part 8: Advanced ERP Capabilities",
            "Part 9: Best Practices",
            "Part 10: Summary",
        ]

        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_notebook_has_erp_imports(self):
        """Test that notebook imports ERP modules."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        required_imports = [
            "evolve.reproduction.engine",
            "evolve.reproduction.protocol",
            "evolve.reproduction.mutation",
        ]

        for imp in required_imports:
            assert imp in content, f"Missing import: {imp}"

    def test_notebook_has_visualizations(self):
        """Test that notebook includes visualization code."""
        with open(TUTORIAL_PATH) as f:
            nb = json.load(f)

        viz_keywords = ["plt.figure", "plt.plot", "plt.show", "fig,", "ax ="]

        viz_cells = []
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if any(keyword in source for keyword in viz_keywords):
                    viz_cells.append(source)

        assert len(viz_cells) >= 5, (
            f"Expected at least 5 visualization cells, found {len(viz_cells)}"
        )

    def test_notebook_has_erp_engine_usage(self):
        """Test that ERPEngine is actually used in notebook."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        assert "ERPEngine(" in content, "ERPEngine not instantiated"
        assert "ERPConfig(" in content, "ERPConfig not instantiated"
        assert "ProtocolMutator(" in content, "ProtocolMutator not instantiated"

    def test_notebook_has_protocol_examples(self):
        """Test that notebook shows protocol creation examples."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        protocol_components = [
            "ReproductionIntentPolicy",
            "MatchabilityFunction",
            "CrossoverProtocolSpec",
            "ReproductionProtocol",
        ]

        for component in protocol_components:
            assert component in content, f"Missing protocol component: {component}"

    @pytest.mark.integration
    def test_notebook_cells_have_outputs(self):
        """Test that code cells have execution outputs."""
        with open(TUTORIAL_PATH) as f:
            nb = json.load(f)

        code_cells_with_output = 0
        total_code_cells = 0

        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                total_code_cells += 1
                if cell.get("outputs"):
                    code_cells_with_output += 1

        # At least 60% of code cells should have been executed
        if total_code_cells > 0:
            execution_rate = code_cells_with_output / total_code_cells
            assert execution_rate >= 0.6, f"Only {execution_rate:.1%} of cells executed"


class TestTutorialContent:
    """Tests for specific tutorial content."""

    def test_sexual_selection_example(self):
        """Test that sexual selection case study is present."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        assert "sexual selection" in content.lower(), "Sexual selection not covered"
        assert "choosy" in content.lower() or "selective" in content.lower(), (
            "Choosy mating not demonstrated"
        )

    def test_recovery_mechanisms_covered(self):
        """Test that recovery mechanisms are explained."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        assert "recovery" in content.lower(), "Recovery mechanisms not covered"
        assert "enable_recovery" in content, "Recovery configuration not shown"

    def test_protocol_mutation_covered(self):
        """Test that protocol mutation is explained."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        assert "protocol_mutation" in content.lower(), "Protocol mutation not covered"
        assert "MutationConfig" in content, "MutationConfig not used"

    def test_best_practices_included(self):
        """Test that best practices section exists."""
        with open(TUTORIAL_PATH) as f:
            content = f.read()

        best_practice_keywords = [
            "best practices",
            "pitfall",
            "common mistakes",
            "debugging",
        ]

        matches = sum(1 for keyword in best_practice_keywords if keyword in content.lower())
        assert matches >= 2, "Best practices section insufficient"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not JUPYTER_AVAILABLE, reason="jupyter not installed")
def test_tutorial_performance_benchmark():
    """Benchmark tutorial execution time for monitoring."""
    import time

    start = time.time()
    success, error_msg = execute_notebook(TUTORIAL_PATH, timeout=900)
    duration = time.time() - start

    assert success, f"Benchmark failed: {error_msg}"

    # Log performance metrics
    print(f"\n{'=' * 60}")
    print("Tutorial 6 Performance Benchmark")
    print(f"{'=' * 60}")
    print(f"Execution time: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
    print(f"Cell count: {get_notebook_cell_count(TUTORIAL_PATH)}")

    if duration < 300:
        print("✓ Excellent: Under 5 minutes")
    elif duration < 600:
        print("✓ Good: Under 10 minutes")
    elif duration < 900:
        print("⚠ Acceptable: Under 15 minutes")
    else:
        print("❌ Too slow: Over 15 minutes")

    print(f"{'=' * 60}\n")
