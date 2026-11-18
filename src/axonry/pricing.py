"""Fees: a run pays its GPU hours and its agent's model use, from a virtual account kept in USD.

An account is the sum of a coldkey's deposits to the treasury, each converted to USD at its own moment (TAO/USD at
the deposit's block time; alpha through the subnet's alpha price at the deposit's block), minus the run charges.
Before review the account must cover a run's maximum: every GPU hour up to the deadline plus the agent budget
(run.max_agent_usd). fee.slippage only pads the transfer the miner tool suggests, for the price moving before its
block; it is never required.
alpha price (TAO per alpha) comes from chain state (exact, reproducible).
TAO/USD has no on-chain source: median of the external sources that answer (>= 2); refuse when they disagree.
Amounts are integer micro-USD.
"""
import json
import statistics
import time
import urllib.request

RAO = 10**9
MUSD = 10**6


class PriceError(RuntimeError):
    pass


def deposit_musd(amount_rao: int, asset: str, tao_usd: float, alpha_tao: float | None = None) -> int:
    """The USD value of a deposit at its own moment's prices."""
    if tao_usd <= 0:
        raise PriceError("bad TAO/USD price")
    tao = amount_rao / RAO
    if asset == "alpha":
        if not alpha_tao or alpha_tao <= 0:
            raise PriceError("bad alpha price")
        tao *= alpha_tao
    elif asset != "tao":
        raise PriceError(f"unknown asset {asset}")
    return int(tao * tao_usd * MUSD)


def hour_musd(rules) -> int:
    """One hour of a run: the GPU-hour price times the run's GPUs."""
    return int(round(rules["fee"]["usd_per_gpu_hour"] * rules["run"]["gpus"] * MUSD))


def gpu_max_musd(rules) -> int:
    """Every GPU hour up to the deadline."""
    return int(round(hour_musd(rules) * rules["run"]["hours"]))


def agent_max_musd(rules) -> int:
    return int(round(rules["run"]["max_agent_usd"] * MUSD))


def run_cost_musd(rules) -> int:
    """What a run holds from its account until it ends: its GPU hours and agent budget."""
    return gpu_max_musd(rules) + agent_max_musd(rules)


def used_musd(rules, seconds: float) -> int:
    """The GPU charge for a run of this length (capped at every hour up to the deadline)."""
    return min(int(hour_musd(rules) * max(seconds, 0) / 3600), gpu_max_musd(rules))


def review_musd(rules) -> int:
    """fee.review_usd: charged when an entry goes to review; kept if it never runs, otherwise part of its run's cost."""
    return int(round(rules["fee"].get("review_usd", 0) * MUSD))


def agent_musd(rules, usd: float | None) -> int:
    """The agent charge: what its model use cost, capped at the agent budget."""
    return min(int(max(usd or 0, 0) * MUSD), agent_max_musd(rules))


# ---- TAO/USD sources ------------------------------------------------------------------------------------------
def _get_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "axonry-price/1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def coingecko(ts: int) -> float:
    """Price closest to unix time ts (5-minute granularity for recent ranges)."""
    data = _get_json("https://api.coingecko.com/api/v3/coins/bittensor/market_chart/range"
                     f"?vs_currency=usd&from={ts - 900}&to={ts + 900}")
    pts = data.get("prices") or []
    if not pts:
        raise PriceError("coingecko: no data")
    return float(min(pts, key=lambda p: abs(p[0] / 1000 - ts))[1])


def binance(ts: int) -> float:
    rows = _get_json(f"https://api.binance.com/api/v3/klines?symbol=TAOUSDT&interval=1m&startTime={(ts - 60) * 1000}&limit=3")
    if not rows:
        raise PriceError("binance: no data")
    row = min(rows, key=lambda r: abs(r[0] / 1000 - ts))
    return float(row[4])


def kraken(ts: int) -> float:
    data = _get_json(f"https://api.kraken.com/0/public/OHLC?pair=TAOUSD&interval=1&since={ts - 180}")
    rows = next((v for k, v in (data.get("result") or {}).items() if k != "last"), None)
    if not rows:
        raise PriceError(f"kraken: {data.get('error') or 'no data'}")
    return float(min(rows, key=lambda r: abs(r[0] - ts))[4])


def coinbase(ts: int) -> float:
    iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))  # noqa: E731
    rows = _get_json(f"https://api.exchange.coinbase.com/products/TAO-USD/candles?granularity=60"
                     f"&start={iso(ts - 600)}&end={iso(ts + 60)}")
    if not rows:
        raise PriceError("coinbase: no data")
    return float(min(rows, key=lambda r: abs(r[0] - ts))[4])


# Any two that answer and agree suffice; several, because a source may be unreachable from where the orchestrator
# runs (e.g. Binance and Bybit refuse some regions).
SOURCES = {"coingecko": coingecko, "kraken": kraken, "coinbase": coinbase, "binance": binance}


def tao_usd_at(ts: int, sources=tuple(SOURCES), max_spread=0.05, fetch=None) -> tuple[float, dict]:
    """-> (median price, {source: price}). Raises if fewer than 2 sources answer or they disagree by > max_spread."""
    fetch = fetch or SOURCES
    got = {}
    for name in sources:
        try:
            got[name] = fetch[name](ts)
        except Exception:  # one source down is fine as long as two remain
            continue
    if len(got) < 2:
        raise PriceError(f"need 2 price sources, got {sorted(got)}")
    med = statistics.median(got.values())
    if (max(got.values()) - min(got.values())) / med > max_spread:
        raise PriceError(f"price sources disagree: {got}")
    return med, got
