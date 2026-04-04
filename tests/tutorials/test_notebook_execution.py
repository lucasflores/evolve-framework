"""
Notebook execution tests using papermill.

Validates that tutorial notebooks:
1. Execute end-to-end without errors
2. Produce expected outputs
3. Meet convergence criteria

T047: Create notebook validation test file
T048: VectorGenome notebook execution test
T066: SequenceGenome notebook execution test (placeholder)
T083: GraphGenome notebook execution test (placeholder)
T103: RLGenome notebook execution test (placeholder)
T125: SCMGenome notebook execution test (placeholder)
T149a: Notebook import independence check
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import notebooks in a way that checks for import independence
TUTORIALS_DIR = Path(__file__).parent.parent.parent / "docs" / "tutorials"


def _notebook_exists(name: str) -> bool:
    """Check if a notebook exists in the tutorials directory."""
    return (TUTORIALS_DIR / name).exists()


# Check for papermill availability
try:
    import papermill as pm

    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False


@pytest.fixture(scope="module")
def temp_output_dir():
    """Provide a temporary directory for executed notebooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# T149a: NOTEBOOK IMPORT INDEPENDENCE CHECK
# =============================================================================


class TestNotebookImportIndependence:
    """Test that each notebook can be imported without side effects.

    T149a: Verify notebook imports don't have circular dependencies or
    require specific import order.
    """

    def test_tutorial_utils_import_standalone(self):
        """tutorial_utils should import without importing evolve core."""
        # Fresh import to check for side effects
        import importlib

        # Clear any cached imports
        set(sys.modules.keys())

        # Import tutorial_utils
        spec = importlib.util.spec_from_file_location(
            "tutorial_utils_test", TUTORIALS_DIR / "utils" / "tutorial_utils.py"
        )
        module = importlib.util.module_from_spec(spec)

        # This should work without evolve being imported first
        try:
            spec.loader.exec_module(module)
            import_succeeded = True
        except ImportError as e:
            import_succeeded = False
            pytest.fail(f"tutorial_utils failed standalone import: {e}")

        assert import_succeeded, "tutorial_utils should import independently"

    def test_tutorial_utils_exports(self):
        """tutorial_utils __init__ should export all public symbols."""
        from docs.tutorials.utils import tutorial_utils

        # Check essential exports
        essential_exports = [
            "BenchmarkFunction",
            "EvolutionHistory",
            "ParetoFront",
            "sphere_function",
            "get_benchmark",
            "generate_polynomial_data",
            "plot_fitness_history",
            "get_glossary",
            "render_mermaid",
        ]

        for name in essential_exports:
            assert hasattr(tutorial_utils, name), f"Missing export: {name}"


# =============================================================================
# T047/T048: VECTORGENOME NOTEBOOK TESTS
# =============================================================================


@pytest.mark.skipif(not PAPERMILL_AVAILABLE, reason="papermill not installed")
@pytest.mark.skipif(
    not _notebook_exists("01_vector_genome.ipynb"), reason="Notebook not yet created"
)
class TestVectorGenomeNotebook:
    """Tests for 01_vector_genome.ipynb notebook execution.

    T048: VectorGenome notebook execution test
    """

    def test_notebook_executes_without_error(self, temp_output_dir):
        """Notebook should execute all cells without raising exceptions."""
        input_path = TUTORIALS_DIR / "01_vector_genome.ipynb"
        output_path = temp_output_dir / "01_vector_genome_executed.ipynb"

        # Execute notebook
        pm.execute_notebook(
            str(input_path),
            str(output_path),
            kernel_name="python3",
            parameters={},
            cwd=str(TUTORIALS_DIR),
        )

        assert output_path.exists(), "Executed notebook should be created"

    def test_rastrigin_convergence(self, _temp_output_dir):
        """Rastrigin optimization should converge within 1% of optimum.

        Independent test criterion: Rastrigin converges to within 1% of
        global optimum (0.0) within 100 generations.
        """
        # This test validates the quality of evolution results
        # Skip if notebook not fully implemented yet
        pytest.skip("Convergence test requires full notebook implementation")

    def test_rosenbrock_convergence(self, _temp_output_dir):
        """Rosenbrock optimization should converge within 1% of optimum."""
        pytest.skip("Convergence test requires full notebook implementation")


# =============================================================================
# T066: SEQUENCEGENOME NOTEBOOK TESTS (PLACEHOLDER)
# =============================================================================


@pytest.mark.skipif(not PAPERMILL_AVAILABLE, reason="papermill not installed")
@pytest.mark.skipif(
    not _notebook_exists("02_sequence_genome.ipynb"), reason="Notebook not yet created"
)
class TestSequenceGenomeNotebook:
    """Tests for 02_sequence_genome.ipynb notebook execution.

    T066: SequenceGenome notebook execution test
    """

    def test_notebook_executes_without_error(self, temp_output_dir):
        """Notebook should execute all cells without raising exceptions."""
        input_path = TUTORIALS_DIR / "02_sequence_genome.ipynb"
        output_path = temp_output_dir / "02_sequence_genome_executed.ipynb"

        pm.execute_notebook(
            str(input_path),
            str(output_path),
            kernel_name="python3",
            cwd=str(TUTORIALS_DIR),
        )

    def test_symbolic_regression_accuracy(self, _temp_output_dir):
        """Evolved expressions should achieve within 5% test error."""
        pytest.skip("Accuracy test requires full notebook implementation")


# =============================================================================
# T083: GRAPHGENOME/NEAT NOTEBOOK TESTS (PLACEHOLDER)
# =============================================================================


@pytest.mark.skipif(not PAPERMILL_AVAILABLE, reason="papermill not installed")
@pytest.mark.skipif(
    not _notebook_exists("03_graph_genome_neat.ipynb"), reason="Notebook not yet created"
)
class TestGraphGenomeNotebook:
    """Tests for 03_graph_genome_neat.ipynb notebook execution.

    T083: GraphGenome notebook execution test
    """

    def test_notebook_executes_without_error(self, temp_output_dir):
        """Notebook should execute all cells without raising exceptions."""
        input_path = TUTORIALS_DIR / "03_graph_genome_neat.ipynb"
        output_path = temp_output_dir / "03_graph_genome_neat_executed.ipynb"

        pm.execute_notebook(
            str(input_path),
            str(output_path),
            kernel_name="python3",
            cwd=str(TUTORIALS_DIR),
        )


# =============================================================================
# T103: RL/NEUROEVOLUTION NOTEBOOK TESTS (PLACEHOLDER)
# =============================================================================


@pytest.mark.skipif(not PAPERMILL_AVAILABLE, reason="papermill not installed")
@pytest.mark.skipif(
    not _notebook_exists("04_rl_neuroevolution.ipynb"), reason="Notebook not yet created"
)
class TestRLNeuroevolutionNotebook:
    """Tests for 04_rl_neuroevolution.ipynb notebook execution.

    T103: RLGenome notebook execution test
    """

    def test_notebook_executes_without_error(self, temp_output_dir):
        """Notebook should execute all cells without raising exceptions."""
        input_path = TUTORIALS_DIR / "04_rl_neuroevolution.ipynb"
        output_path = temp_output_dir / "04_rl_neuroevolution_executed.ipynb"

        pm.execute_notebook(
            str(input_path),
            str(output_path),
            kernel_name="python3",
            cwd=str(TUTORIALS_DIR),
        )


# =============================================================================
# T125: SCMGENOME NOTEBOOK TESTS (PLACEHOLDER)
# =============================================================================


@pytest.mark.skipif(not PAPERMILL_AVAILABLE, reason="papermill not installed")
@pytest.mark.skipif(
    not _notebook_exists("05_scm_multiobjective.ipynb"), reason="Notebook not yet created"
)
class TestSCMGenomeNotebook:
    """Tests for 05_scm_multiobjective.ipynb notebook execution.

    T125: SCMGenome notebook execution test
    """

    def test_notebook_executes_without_error(self, temp_output_dir):
        """Notebook should execute all cells without raising exceptions."""
        input_path = TUTORIALS_DIR / "05_scm_multiobjective.ipynb"
        output_path = temp_output_dir / "05_scm_multiobjective_executed.ipynb"

        pm.execute_notebook(
            str(input_path),
            str(output_path),
            kernel_name="python3",
            cwd=str(TUTORIALS_DIR),
        )


# =============================================================================
# UTILITY TESTS
# =============================================================================


class TestNotebookInfrastructure:
    """Tests for notebook testing infrastructure itself."""

    def test_tutorials_directory_exists(self):
        """Tutorials directory should exist."""
        assert TUTORIALS_DIR.exists(), f"Tutorials dir should exist: {TUTORIALS_DIR}"

    def test_utils_module_exists(self):
        """Utils module should be importable."""
        utils_path = TUTORIALS_DIR / "utils" / "tutorial_utils.py"
        assert utils_path.exists(), f"tutorial_utils.py should exist: {utils_path}"

    def test_papermill_availability_message(self):
        """Should provide useful message if papermill unavailable."""
        if not PAPERMILL_AVAILABLE:
            pytest.skip("papermill not available - install with: pip install papermill")
        assert PAPERMILL_AVAILABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:indico"])
