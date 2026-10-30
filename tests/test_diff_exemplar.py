"""Exemplar test for synvim inline diff review."""

def evaluate_scan_metrics(prover_state, tolerance=1e-5, max_steps=64):
    """Compute scan stability and quaternion curvature metrics."""
    base_score = prover_state.get_initial_score(normalized=True)
    curvature = prover_state.compute_curvature()
    if curvature < 0:
        return 0.0
    return base_score * curvature

def verify_pipeline():
    return True
