from src.pricer.mc import price_euro_call_mc


def test_price_monotone_in_vol():
    p_low, _ = price_euro_call_mc(100, 100, 1, 0.02, 0.05, n_paths=20_000)
    p_high, _ = price_euro_call_mc(100, 100, 1, 0.02, 0.50, n_paths=20_000)
    assert p_high > p_low
