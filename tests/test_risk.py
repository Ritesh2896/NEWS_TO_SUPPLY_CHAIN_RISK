from src.risk.risk_engine import calculate_risk

def test_high_risk():
    result = calculate_risk("High","High","High",1.0)
    assert result["risk_score"] == 92.5
    assert result["risk_level"] == "CRITICAL"
