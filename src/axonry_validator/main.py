"""Standalone validator. Reads the subnet's published weights.json, verifies the owner signature, maps uid->hotkey
against the chain and sets weights with its own wallet. URL, owner and netuid are the subnet's published constants
(network.py), not settings. Needs no storage keys, database or eval data. Does not import the `axonry` package.

  axonry-validator --wallet NAME --hotkey HOTKEY [--network finney] [--once]
"""
import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

from . import network as published

log = logging.getLogger("axonry-validator")
SCHEMA = "axonry.weights/1"
TEMPO = 360              # blocks between the subnet's epochs, when the chain does not say
RETRY = 60               # seconds before trying again after a failed set (a node hiccup), within the same window
BLOCK_S = 12             # seconds per block, to sleep until the next window


def window(block: int, tempo: int) -> int:
    """Weights are set once per window of tempo // 2 blocks counted from block 0: at least once in every epoch (the
    chain uses only the last weights before it), and further apart than the rate limit (100 blocks)."""
    return block // max(tempo // 2, 1)


def tempo_of(subtensor, netuid) -> int:
    try:
        import bittensor as bt  # noqa: PLC0415
        return int(subtensor.query(bt.storage.SubtensorModule.Tempo, [netuid])) or TEMPO
    except Exception:  # noqa: BLE001 - the default until the chain answers
        return TEMPO
MAX_AGE = 7 * 86400      # warn when the published weights are older than this
VERSION_KEY = 0


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(doc: dict, owner: str) -> bool:
    from bittensor_wallet import Keypair  # noqa: PLC0415

    sig = doc.get("signature")
    if doc.get("signer") != owner or not isinstance(sig, str) or not sig.startswith("0x"):
        return False
    body = canonical({k: v for k, v in doc.items() if k != "signature"})
    try:
        return bool(Keypair(ss58_address=owner).verify(body, bytes.fromhex(sig[2:])))
    except Exception:
        return False


def check(doc: dict, owner: str, netuid: int, last_sequence: int) -> str | None:
    """-> reason the document is unusable, or None."""
    if doc.get("schema") != SCHEMA:
        return f"schema {doc.get('schema')!r}"
    if not verify(doc, owner):
        return "bad signature"
    if doc.get("netuid") != netuid:
        return f"netuid {doc.get('netuid')} != {netuid}"
    if not isinstance(doc.get("sequence"), int) or doc["sequence"] < last_sequence:
        return f"sequence {doc.get('sequence')} below last seen {last_sequence}"
    ws = doc.get("weights")
    if not isinstance(ws, list) or not ws or any(not isinstance(w.get("uid"), int) or w.get("weight", -1) < 0 for w in ws):
        return "malformed weights"
    return None


def resolve(doc: dict, uid_hotkeys: dict) -> dict:
    """{uid: weight}; an entry whose hotkey no longer sits at its uid goes to the fallback uid (uid0 = burn)."""
    fallback = doc.get("fallback", {}).get("uid", 0)
    out = {}
    for w in doc["weights"]:
        uid = w["uid"]
        if w.get("hotkey") is not None and uid_hotkeys.get(uid) != w["hotkey"]:
            uid = fallback
        out[uid] = out.get(uid, 0.0) + float(w["weight"])
    return out


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "axonry-validator/1", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class State:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"sequence": -1, "doc": None}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)


def set_weights(subtensor, wallet, netuid, weights: dict, version_key: int):
    import bittensor as bt  # noqa: PLC0415

    uids = sorted(weights)
    res = subtensor.execute(bt.SetWeights(netuid=netuid, uids=uids, weights=[weights[u] for u in uids],
                                          version_key=version_key), wallet)
    if hasattr(res, "raise_for_failure"):
        res.raise_for_failure()
    return res


def cycle(a, state, subtensor, wallet, fetcher=fetch, setter=set_weights, now=time.time):
    try:
        doc = fetcher(a.url)
        why = check(doc, a.owner, a.netuid, state.data["sequence"])
        if why:
            log.warning("weights.json rejected: %s; keeping last valid", why)
        else:
            state.data.update(sequence=doc["sequence"], doc=doc)
            state.save()
    except Exception as e:  # network / JSON errors: keep the last valid document
        log.warning("fetch failed: %s; keeping last valid", e)
    doc = state.data.get("doc")
    if not doc:
        log.warning("no valid weights document yet; not setting weights")
        return None
    if now() - doc.get("issued_at", 0) > a.max_age:
        log.warning("weights.json older than %ss; still using it", a.max_age)
    import bittensor as bt  # noqa: PLC0415
    uid_hotkeys = {int(k): str(v) for k, v in subtensor.query_map(bt.storage.SubtensorModule.Keys, [a.netuid])}
    weights = resolve(doc, uid_hotkeys)
    setter(subtensor, wallet, a.netuid, weights, a.version_key)
    log.info("set weights %s (sequence %s)", weights, doc["sequence"])
    return weights


def main(argv=None):
    ap = argparse.ArgumentParser(description="Set the subnet's published, owner-signed weights")
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--hotkey", required=True)
    ap.add_argument("--wallet-path")
    ap.add_argument("--network", default="finney", choices=sorted(published.NETWORKS))
    ap.add_argument("--state", default="~/.axonry-validator/state.json")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    net = published.get(a.network)
    run = argparse.Namespace(url=published.weights_url(a.network), owner=net["owner"], netuid=net["netuid"],
                             max_age=MAX_AGE, version_key=VERSION_KEY)
    import bittensor as bt  # noqa: PLC0415
    from bittensor_wallet import Wallet  # noqa: PLC0415

    wallet = Wallet(name=a.wallet, hotkey=a.hotkey, **({"path": a.wallet_path} if a.wallet_path else {}))
    subtensor = bt.Subtensor(network=a.network)
    state = State(a.state)
    log.info("axonry netuid %s on %s, weights from %s signed by %s", run.netuid, a.network, run.url, run.owner)
    done = None                  # the window weights were set in
    while True:
        try:
            tempo, block = tempo_of(subtensor, run.netuid), int(subtensor.block)
        except Exception as e:  # noqa: BLE001 - the node is unreachable for a moment
            log.error("reading the chain failed: %s (retrying in %s s)", e, RETRY)
            time.sleep(RETRY)
            continue
        if window(block, tempo) != done:
            try:
                cycle(run, state, subtensor, wallet)
                done = window(block, tempo)
            except Exception as e:
                log.error("cycle failed: %s (retrying in %s s)", e, RETRY)
        if a.once:
            return 0
        half = max(tempo // 2, 1)
        time.sleep(RETRY if window(block, tempo) != done else min((half - block % half) * BLOCK_S + 1, 3600))


if __name__ == "__main__":
    raise SystemExit(main())
