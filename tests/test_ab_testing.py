import math
import numpy as np
from services.ab_testing import ABTestingService, Status, Group, Experiment

def test_ab_testing_service_lifecycle():
    """Verifies experiment creation, starting, stopping, and basic user routing."""
    service = ABTestingService()
    
    # 1. Test creation
    exp = service.create(exp_id="test_exp", description="Validation test", control_frac=0.4, min_samples=50)
    assert exp.exp_id == "test_exp"
    assert exp.status == Status.DRAFT
    assert exp.groups["control"].fraction == 0.4
    assert exp.groups["treatment"].fraction == 0.6
    
    # Routing assignments should be None while in DRAFT
    assert service.assign(user_id=1, exp_id="test_exp") is None

    # 2. Test start
    service.start("test_exp")
    assert exp.status == Status.RUNNING
    assert exp.started_at is not None

    # Routing assignments should now assign group control/treatment deterministically
    group1 = service.assign(user_id=42, exp_id="test_exp")
    group2 = service.assign(user_id=42, exp_id="test_exp")
    assert group1 in ["control", "treatment"]
    assert group1 == group2, "Deterministic assignment failed!"

    # 3. Test record
    service.record("test_exp", group="control", event="impression")
    service.record("test_exp", group="control", event="click")
    service.record("test_exp", group="control", event="watch_complete", watch_time=45.5)
    
    control_g = exp.groups["control"]
    assert control_g.impressions == 1
    assert control_g.clicks == 1
    assert control_g.watch_completions == 1
    assert control_g.total_watch_time == 45.5

    # 4. Test stop
    service.stop("test_exp")
    assert exp.status == Status.DONE
    assert exp.ended_at is not None

def test_ab_testing_service_ztest():
    """Verifies mathematical two-proportion Z-test calculations and decision verdicts."""
    service = ABTestingService()
    
    # Underpowered or equal proportions
    res_continue = service._ztest(sA=5, nA=100, sB=6, nB=100, alpha=0.05)
    assert res_continue["verdict"] == "CONTINUE"
    assert res_continue["significant"] is False

    # Significant treatment lift
    res_ship = service._ztest(sA=10, nA=200, sB=60, nB=200, alpha=0.05)
    assert res_ship["verdict"] == "SHIP treatment"
    assert res_ship["significant"] is True
    assert res_ship["relative_lift"] > 0

    # Significant control retention (treatment did worse)
    res_keep = service._ztest(sA=60, nA=200, sB=10, nB=200, alpha=0.05)
    assert res_keep["verdict"] == "KEEP control"
    assert res_keep["significant"] is True
    assert res_keep["relative_lift"] < 0

def test_ab_testing_service_simulation():
    """Verifies full experiment simulation runs and analysis report generations."""
    service = ABTestingService()
    
    # Run simulation with 500 users
    report = service.simulate(
        exp_id="sim_exp",
        n_users=500,
        ctrl_ctr=0.05,
        treat_ctr=0.15,
        ctrl_completion=0.10,
        treat_completion=0.25
    )
    
    assert report["exp_id"] == "sim_exp"
    assert report["status"] == Status.DONE
    assert "control" in report
    assert "treatment" in report
    
    # Ensure Z-tests are executed because n_users (500) > min_samples (100)
    assert "ctr_test" in report
    assert "completion_test" in report
    assert "message" not in report
