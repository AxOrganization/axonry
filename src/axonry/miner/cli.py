"""axonry-miner: everything a miner needs, from the public side only.

  axonry-miner board [MINER] [--watch 60] [-n 20] [--no-color]
        no MINER: the subnet at a glance - champion and queue (runs live: time left, GPU use, the agent's progress),
        entries not in the queue and why, recent matches with per-benchmark thresholds;
        MINER (a uid, hotkey, coldkey or submission id): that miner in full - account, entry, its run live (each
        GPU), timeline, admission and the result per benchmark
  axonry-miner price                                    what a run costs at most, in USD and in the accepted asset now
  axonry-miner deposit [--amount X] [--wallet W] [--hotkey H] [--yes]
        transfer from the coldkey owning H to the subnet's treasury (from the published rules); credited in USD at
        that block's price to that coldkey's account, which pays H's run. Without --amount: what the account still
        lacks for a run, at the current price
  axonry-miner commit --dir DIR --github LOGIN [--wallet W] [--hotkey H] [--yes]
        the entry of hotkey H, once: set_commitment "<digest>:<GitHub id of LOGIN>" signed by H (registered on the
        subnet; from the rules' open_block on); only a PR by LOGIN, in H's directory, opened after the commitment,
        with exactly this content can carry it (and, if the rules set reveal_hours, opened within that time)
  W and H are bittensor wallet and hotkey names (default: "default", as in btcli).
One registration is one entry: only a hotkey's first commitment counts, and the content and GitHub account it names
can never change. Commit first and push only after the command returns: a content belongs to its earliest commitment,
and content pushed before it is committed can be copied and committed first by others. The run is paid by the
GPU-hour plus its agent's model use, from the account of the coldkey owning the hotkey: deposit from that coldkey,
as plain transfers (this tool), not via proxies or multisig. The account must cover a whole run before review
(`price`); going to review costs the review fee (once), which counts towards the run, or is kept if the entry never
runs; each full GPU hour is charged as it passes, the rest and the agent's use when the run ends.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from ..common import layout
from ..common.digest import content_digest
from ..common.ids import make_commitment


def public_url(network="finney"):
    """The subnet's published data location (axonry_validator.network); AXONRY_PUBLIC_URL overrides it for testing."""
    from axonry_validator import network as published  # noqa: PLC0415
    return os.environ.get("AXONRY_PUBLIC_URL") or published.get(network)["public_url"]


class PublicReader:
    def __init__(self, base_url=None, store=None):
        self.base, self.store = (base_url or "").rstrip("/"), store

    def get(self, key):
        if self.store is not None:
            data = self.store.get(key)
            return json.loads(data) if data else None
        try:
            req = urllib.request.Request(f"{self.base}/{key}", headers={"User-Agent": "axonry-miner/1",
                                                                       "Cache-Control": "no-cache"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def rules(self, base=None):
        """The rules version in force of `base`'s line, else of the primary line (index.json names them)."""
        idx = self.get(layout.INDEX)
        if not idx:
            raise SystemExit("the subnet has not published its rules yet")
        lines = idx.get("lines") or [{"base": idx["base"], "rules": idx["rules"]}]
        line = lines[0] if base is None else next((x for x in lines if x["base"] == base), None)
        if line is None:
            raise SystemExit(f"no line of base {base} is in force (lines: {', '.join(x['base'] for x in lines)})")
        return self.get(line["rules"])


def cmd_price(r, network, base=None):
    """What a run costs, and what that is in TAO / alpha at this moment's price."""
    from .. import pricing  # noqa: PLC0415
    rules = r.rules(base)
    print(f"base {rules['base']}")
    run, top = rules["run"], pricing.run_cost_musd(rules) / pricing.MUSD
    pad = 1 + rules["fee"].get("slippage", 0)          # what to send: room for the price moving before the block
    print(f"a run: {run['gpus']} x {run['gpu_model']}, at most {run['hours']} h")
    print(f"  GPUs     {rules['fee']['usd_per_gpu_hour']} USD per GPU-hour, at most "
          f"{pricing.gpu_max_musd(rules) / pricing.MUSD:.2f} USD")
    print(f"  agent    its model use, at most {run['max_agent_usd']:.2f} USD")
    if pricing.review_musd(rules):
        print(f"  review   {pricing.review_musd(rules) / pricing.MUSD:.2f} USD when it goes to review, "
              + ("kept" if pricing.review_kept(rules) else "given back when it runs"))
    print(f"  held     {top:.2f} USD: your account must hold this before review")
    try:
        usd, note = asset_usd(network, rules)
        print("  send     " + " or ".join(f"{top * pad / v:.6f} {a}" for a, v in usd.items())
              + f" now: {pad - 1:.0%} more than held is recommended, for the price moving before the block ({note})")
    except Exception as e:  # noqa: BLE001 - the price sources are unreachable from here: the USD figures stand
        print(f"  now      no price available here ({str(e)[:120]})")
    review = rules["fee"].get("review_usd")
    if review:
        print(f"  review   {review:.2f} USD when the entry goes to review: part of the run, kept if it never runs")
    print("Each full GPU hour is charged as it passes; the rest of the GPU time and the agent's use when the run ends.\n"
          "A deposit counts in USD at the price of its block.")


def asset_usd(network, rules) -> tuple[dict, str]:
    """USD per unit of each accepted asset now (TAO/USD from public sources, alpha via the subnet's alpha price)."""
    from .. import pricing  # noqa: PLC0415
    tao_usd, sources = pricing.tao_usd_at(int(time.time()))
    usd = {"tao": tao_usd}
    if "alpha" in rules["fee"]["assets"]:
        usd["alpha"] = tao_usd * _alpha_price(network, rules)
    return {a: usd[a] for a in rules["fee"]["assets"]}, f"TAO/USD {tao_usd:.2f} from {', '.join(sources)}"


def _alpha_price(network, rules):
    import bittensor as bt  # noqa: PLC0415

    from ..chain.subtensor import Chain  # noqa: PLC0415
    return Chain(subtensor=bt.Subtensor(network=network)).alpha_price(rules["netuid"], None)


def _prepare(a, r):
    """-> (subtensor, wallet, hotkey, rules, commitment, digest) after checking the directory against the rules."""
    import bittensor as bt  # noqa: PLC0415
    from bittensor_wallet import Wallet  # noqa: PLC0415

    for f in ("entry.json", "METHOD.md"):
        if not os.path.isfile(os.path.join(a.dir, f)):
            raise SystemExit(f"{f} missing in {a.dir}")
    entry = json.load(open(os.path.join(a.dir, "entry.json")))
    rules = r.rules(entry.get("base"))           # the line of the base it names
    wallet = Wallet(name=a.wallet, hotkey=a.hotkey, **({"path": a.wallet_path} if a.wallet_path else {}))
    hk = wallet.hotkey.ss58_address
    if f"submissions/{hk}/" not in os.path.abspath(a.dir).replace(os.sep, "/") + "/":
        raise SystemExit(f"the directory must be submissions/{hk}/<entry>/")
    digest = content_digest(a.dir)
    sub = bt.Subtensor(network=a.network)
    from ..chain.subtensor import Chain  # noqa: PLC0415
    chain = Chain(a.network, subtensor=sub)
    if chain.uid_of(hk, rules["netuid"], None) is None:
        raise SystemExit(f"{hk} is not registered on subnet {rules['netuid']}: only a registered hotkey can commit")
    from axonry_validator.network import NETWORKS  # noqa: PLC0415
    owner, opens = NETWORKS.get(a.network, {}).get("owner"), rules.get("open_block", 0)
    if hk != owner and int(sub.block) < opens:
        raise SystemExit(f"entries open at block {opens} (now {int(sub.block)}): an earlier commitment does not count")
    if hk != owner:
        try:
            reg, reg_owner = chain.registration_owner(hk, rules["netuid"], int(sub.block))
        except Exception as e:  # noqa: BLE001 - the archive node is unreachable: say so rather than commit blind
            raise SystemExit(f"cannot read the hotkey's registration from an archive node ({str(e)[:120]}); try again")
        if reg is not None and reg_owner != chain.owner_of(hk, None):
            raise SystemExit(f"{hk}: its owner changed since its registration (block {reg}), so its entry would be "
                             "refused (HOTKEY_MOVED); register a hotkey under the coldkey that pays")
    if hk != owner and chain.has_committed(hk, rules["netuid"]):
        raise SystemExit(f"{hk} has committed on subnet {rules['netuid']} before: one registration is one entry "
                         "(only its first commitment counts); use a new hotkey")
    gid = github_id(a.github)
    print(f"GitHub {a.github} = user id {gid}")
    return sub, wallet, hk, rules, make_commitment(digest, gid), digest


def github_id(login):
    """The numeric id of a GitHub account (public API, no token): commitments name it, since logins can change."""
    req = urllib.request.Request(f"https://api.github.com/users/{login}", headers={"User-Agent": "axonry-miner"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(json.loads(r.read())["id"])
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot look up GitHub user {login!r}: {e}") from None


def cmd_commit(a, r):
    from ..chain.subtensor import submit_commitment  # noqa: PLC0415

    sub, wallet, hk, rules, commitment, digest = _prepare(a, r)
    print(f"digest {digest}\ncommitting {commitment}\nThis is the one entry of hotkey {hk}: the content and the GitHub "
          "account can never change afterwards.")
    if not a.yes and input("proceed? [y/N] ").strip().lower() != "y":
        return
    blk, i = submit_commitment(sub, wallet, rules["netuid"], commitment)
    print(f"committed in block {blk} extrinsic {i} (this content's priority). Now push it to your PR, opened by "
          f"{a.github}. Never push content before its commitment is in a block: whoever commits it first owns it.")
    if rules.get("reveal_hours") is not None:
        print(f"Open the PR within {float(rules['reveal_hours']):g} h of this block: an entry whose content no PR "
              "carries by then is void (REVEAL_EXPIRED).")


def cmd_deposit(a, r):
    """Pay for a hotkey: a transfer from the coldkey that owns it (that hotkey's run is charged to the account of its
    owning coldkey; a deposit from any other coldkey does not count for it)."""
    import bittensor as bt  # noqa: PLC0415
    from bittensor_wallet import Wallet  # noqa: PLC0415

    from .. import pricing  # noqa: PLC0415
    from ..chain.subtensor import Chain, submit_deposit  # noqa: PLC0415

    rules = r.rules(a.base)
    a.asset = a.asset or rules["fee"]["assets"][0]
    if a.asset not in rules["fee"]["assets"]:
        raise SystemExit(f"{a.asset} is not accepted (accepted: {', '.join(rules['fee']['assets'])})")
    wallet = Wallet(name=a.wallet, hotkey=a.hotkey, **({"path": a.wallet_path} if a.wallet_path else {}))
    sub = bt.Subtensor(network=a.network)
    hk, ck = wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address
    owner = Chain(subtensor=sub).owner_of(hk, None)
    if owner != ck:
        raise SystemExit(f"hotkey {hk} is owned by {owner}, not by this wallet's coldkey {ck}: pay from its owner")
    if a.amount is None:
        top = pricing.run_cost_musd(rules) / pricing.MUSD
        have = ((r.get(layout.STATUS) or {}).get("accounts", {}).get(ck) or {}).get("available_usd", 0)
        if have >= top:
            raise SystemExit(f"the account of {ck} already covers a run ({have:.2f} of {top:.2f} USD available)")
        try:
            per = asset_usd(a.network, rules)[0][a.asset]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"no price available here ({str(e)[:120]}): give --amount") from None
        pad = 1 + rules["fee"].get("slippage", 0)       # room for the price moving before the block
        a.amount = round((top * pad - have) / per + 5e-7, 6)
        print(f"the account has {have:.2f} of the {top:.2f} USD a run holds: {a.amount} {a.asset} covers the rest")
    treasury, amount = rules["fee"]["treasury"], int(round(a.amount * pricing.RAO))
    staked = a.stake_hotkey or hk
    print(f"depositing {a.amount} {a.asset} from coldkey {ck} to {treasury}" +
          (f" (stake on {staked})" if a.asset == "alpha" else "") +
          f"\ncredited to the account of {ck}, which pays the runs of its hotkeys (incl. {hk})")
    if not a.yes and input("proceed? [y/N] ").strip().lower() != "y":
        return
    blk, i = submit_deposit(sub, wallet, treasury, rules["netuid"], a.asset, amount,
                            staked_hotkey=staked if a.asset == "alpha" else None)
    print(f"finalized in block {blk} extrinsic {i} (see `axonry-miner board {ck}`)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="axonry-miner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="finney")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("board", help="the subnet at a glance, or one miner in full")
    b.add_argument("miner", nargs="?", help="a uid, hotkey, coldkey or submission id: that miner in full")
    b.add_argument("-n", type=int, default=20, help="recent matches to show")
    b.add_argument("--no-color", action="store_true")
    b.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="redraw every SECONDS (e.g. 60)")
    pr = sub.add_parser("price", help="what a run costs at most")
    c = sub.add_parser("commit", help="the hotkey's one entry: commit the content on chain, then push the PR")
    c.add_argument("--dir", required=True, help="submissions/<your hotkey>/<entry>/")
    c.add_argument("--github", required=True, help="GitHub login that opens (or opened) the pull request")
    d = sub.add_parser("deposit", help="fund the account of the coldkey that owns the hotkey (sent to the "
                                       "treasury named in the published rules)")
    d.add_argument("--amount", type=float, help="default: what the account still lacks for a run, at the current price")
    d.add_argument("--asset", choices=("tao", "alpha"), help="default: the asset the rules accept")
    d.add_argument("--stake-hotkey", help="alpha: the hotkey (ss58) the stake to move sits on (default: --hotkey's)")
    for s in (pr, d):
        s.add_argument("--base", help="the line of this base model (default: the first line in force)")
    for s in (c, d):
        s.add_argument("--wallet", default="default", help="bittensor wallet name (default: default)")
        s.add_argument("--hotkey", default="default", help="the miner hotkey's name in that wallet (default: default)")
        s.add_argument("--wallet-path", help="wallets directory (default: bittensor's)")
        s.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 0
    r = PublicReader(public_url(a.network))
    if a.cmd == "board":
        from .board import cmd_board  # noqa: PLC0415
        cmd_board(r, a.n, False if a.no_color else None, a.watch, a.miner)
    elif a.cmd == "price":
        cmd_price(r, a.network, a.base)
    elif a.cmd == "deposit":
        cmd_deposit(a, r)
    elif a.cmd == "commit":
        cmd_commit(a, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
