"""
Mermaid diagram rendering utilities for Jupyter notebooks.

Provides functions to render beautiful mermaid diagrams using mermaid-cli (mmdc)
with fallback support for markdown code blocks.

Requirements:
    - @mermaid-js/mermaid-cli (install with: npm install -g @mermaid-js/mermaid-cli)
    - puppeteer (install with: npm install -g puppeteer)

Usage:
    from docs.tutorials.utils.mermaid_renderer import show_mermaid, DIAGRAM_DIR

    diagram_code = '''
    graph LR
        A[Start] --> B[End]
    '''
    show_mermaid(diagram_code, "my_diagram", title="My Flow Chart")
"""

import subprocess
import tempfile
from pathlib import Path

from IPython.display import Image, Markdown, display

# Create diagrams directory for storing rendered images
DIAGRAM_DIR = Path("./diagrams")
DIAGRAM_DIR.mkdir(exist_ok=True)


def render_mermaid_cli(
    mermaid_code: str,
    filename: str,
    theme: str = "default",
    background: str = "transparent",
    scale: int = 2,
) -> str | None:
    """
    Render mermaid diagram using mermaid-cli (mmdc) for beautiful output.

    Args:
        mermaid_code: Mermaid diagram code (e.g., "graph LR\\n A-->B")
        filename: Output filename (without extension)
        theme: Mermaid theme - 'default', 'forest', 'dark', 'neutral'
        background: Background color - 'transparent', 'white', '#ffffff', etc.
        scale: Scale factor for output image (1-4, default 2 for retina quality)

    Returns:
        Path to saved PNG file, or None if rendering failed

    Example:
        >>> code = "graph LR\\n A[Start] --> B[End]"
        >>> path = render_mermaid_cli(code, "flow_diagram", theme="forest")
        >>> if path:
        ...     print(f"Rendered to: {path}")
    """
    output_path = DIAGRAM_DIR / f"{filename}.png"

    # Write mermaid code to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False) as f:
        f.write(mermaid_code)
        input_path = f.name

    try:
        # Use mermaid-cli (mmdc) for beautiful rendering
        result = subprocess.run(
            [
                "mmdc",
                "-i",
                input_path,
                "-o",
                str(output_path),
                "-t",
                theme,
                "-b",
                background,
                "-s",
                str(scale),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and output_path.exists():
            return str(output_path)
        else:
            print(f"⚠️ mmdc error: {result.stderr}")
            return None

    except FileNotFoundError:
        print("⚠️ mermaid-cli (mmdc) not found.")
        print("   Install with: npm install -g @mermaid-js/mermaid-cli puppeteer")
        return None

    except subprocess.TimeoutExpired:
        print("⚠️ Rendering timed out after 30 seconds")
        return None

    except Exception as e:
        print(f"⚠️ Error rendering diagram: {e}")
        return None

    finally:
        # Clean up temporary file
        Path(input_path).unlink(missing_ok=True)


def show_mermaid(
    mermaid_code: str, filename: str, title: str = "", theme: str = "default", width: int = 700
):
    """
    Render and display a beautiful mermaid diagram in a Jupyter notebook.

    Args:
        mermaid_code: Mermaid diagram code
        filename: Output filename (without extension)
        title: Optional title to display above the diagram
        theme: Mermaid theme - 'default', 'forest', 'dark', 'neutral'
        width: Display width in pixels (default 700)

    Example:
        >>> diagram = '''
        ... graph TD
        ...     A[Start] --> B{Decision}
        ...     B -->|Yes| C[Action 1]
        ...     B -->|No| D[Action 2]
        ... '''
        >>> show_mermaid(diagram, "decision_flow", title="Decision Process", theme="forest")
    """
    if title:
        print(f"📊 {title}\n")

    path = render_mermaid_cli(mermaid_code, filename, theme)

    if path:
        # Successfully rendered - display the PNG image
        display(Image(filename=path, width=width))
    else:
        # Fallback to markdown code block
        print("⚠️ Falling back to markdown display\n")
        display(Markdown(f"```mermaid\n{mermaid_code.strip()}\n```"))


def check_mermaid_cli() -> bool:
    """
    Check if mermaid-cli is installed and available.

    Returns:
        True if mmdc command is available, False otherwise

    Example:
        >>> if not check_mermaid_cli():
        ...     print("Please install mermaid-cli")
    """
    try:
        result = subprocess.run(["mmdc", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Convenience function for initialization
def setup_mermaid(check_installation: bool = True):
    """
    Setup mermaid rendering for the notebook.

    Args:
        check_installation: If True, check that mmdc is installed

    Returns:
        True if setup succeeded, False if mmdc not found

    Example:
        >>> if not setup_mermaid():
        ...     print("Install with: npm install -g @mermaid-js/mermaid-cli puppeteer")
    """
    # Ensure diagrams directory exists
    DIAGRAM_DIR.mkdir(exist_ok=True)

    if check_installation:
        if check_mermaid_cli():
            print("✅ Mermaid CLI ready")
            return True
        else:
            print("❌ Mermaid CLI not found")
            print("Install with: npm install -g @mermaid-js/mermaid-cli puppeteer")
            return False

    return True
