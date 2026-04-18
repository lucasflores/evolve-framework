# Governance — tests/agents.md

## Test Rules

- Mirror `evolve/` structure: `tests/unit/representation/`, `tests/integration/`
- Use pytest fixtures in `conftest.py` for shared setup
- Test file naming: `test_<module>.py`
- Each test function tests one behavior; descriptive names (`test_decode_uniform_cppn_produces_no_hidden_neurons`)
- Deterministic: use explicit seeds for any RNG
- No external network calls; mock where needed
- Integration tests in `tests/integration/`; unit tests in `tests/unit/`
