from src.pricer.mc import price_option


def test_price_monotone_in_vol():
    p_low, _ = price_option(100, 100, 1, 0.02, 0.05, 0, 20_000)
    p_high, _ = price_option(100, 100, 1, 0.02, 0.50, 0, 20_000)
    assert p_high > p_low
