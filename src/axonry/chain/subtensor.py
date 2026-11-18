"""Chain access through the bittensor 11 SDK (`bittensor.Subtensor`). Read-only for the orchestrator; the miner CLI
also uses `submit_commitment` and `submit_deposit`. Only finalized blocks are ever read for intake.
"""
from .extrinsics import parse_extrinsic

ZERO_ACCOUNT = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"   # default of SubtensorModule.Owner


class Chain:
    def __init__(self, network="finney", subtensor=None):
        if subtensor is None:
            import bittensor as bt  # noqa: PLC0415
            subtensor = bt.Subtensor(network=network)
        self.sub = subtensor

    @property
    def _S(self):
        import bittensor as bt  # noqa: PLC0415
        return bt.storage.SubtensorModule

    def finalized_head(self) -> int:
        return int(self.sub.finalized_block)

    def owner_of(self, hotkey, block):
        v = self.sub.query(self._S.Owner, [hotkey], block=block)
        v = str(v) if v else None
        return None if v in (None, ZERO_ACCOUNT) else v

    def registered_at(self, hotkey, netuid, block=None):
        """The block of the hotkey's current registration on the subnet (as of `block`), or None."""
        uid = self.uid_of(hotkey, netuid, block)
        return None if uid is None else int(self.sub.query(self._S.BlockAtRegistration, [netuid, uid], block=block))

    def uid_of(self, hotkey, netuid, block=None):
        v = self.sub.query(self._S.Uids, [netuid, hotkey], block=block)
        return None if v is None else int(v)

    def alpha_price(self, netuid, block):
        """TAO per alpha from the swap runtime API at `block` (authoritative under the swap pallet)."""
        p = self.sub.prices.alpha_price(netuid, block=block)
        return float(p["tao_per_alpha"])

    def uid_hotkeys(self, netuid, block=None) -> dict:
        return {int(k): str(v) for k, v in self.sub.query_map(self._S.Keys, [netuid], block=block)}

    def has_committed(self, hotkey, netuid) -> bool:
        """Whether this hotkey ever set a commitment on the subnet (the storage keeps its latest one)."""
        import bittensor as bt  # noqa: PLC0415
        return self.sub.query(bt.storage.Commitments.CommitmentOf, [netuid, hotkey]) is not None

    def items_in_block(self, block, treasury, netuid) -> list:
        """The commitments and deposits of one block, in extrinsic order."""
        import bittensor as bt  # noqa: PLC0415

        info = self.sub.block_info(block)
        ts = int(info.timestamp.timestamp())
        status = {}
        for e in self.sub.query(bt.storage.System.Events, block=block):
            if e.get("module_id") == "System" and e.get("event_id") in ("ExtrinsicSuccess", "ExtrinsicFailed"):
                status[e["extrinsic_idx"]] = e["event_id"] == "ExtrinsicSuccess"
        out = []
        for idx, xt in enumerate(info.extrinsics):
            if not xt or not xt.get("address"):
                continue
            out += parse_extrinsic(xt["call"], signer=str(xt["address"]), block=block, extrinsic_index=idx,
                                   timestamp=ts, success=status.get(idx, False), treasury=treasury, netuid=netuid)
        return out


def commitment_call(netuid: int, text: str):
    import bittensor as bt  # noqa: PLC0415
    raw = text.encode()
    return bt.calls.Commitments.set_commitment(netuid=netuid, info={"fields": [{f"Raw{len(raw)}": "0x" + raw.hex()}]})


def submit_commitment(subtensor, wallet, netuid: int, text: str):
    """Miner side: a content commitment, signed by the hotkey, which must be registered on the subnet.
    -> (block, index)."""
    res = subtensor.submit_call(commitment_call(netuid, text), wallet, signer="hotkey")
    res.raise_for_failure()
    blk, idx = map(int, res.extrinsic_id.split("-"))
    return blk, idx


def submit_deposit(subtensor, wallet, treasury, netuid, asset: str, amount_rao: int, staked_hotkey: str | None = None):
    """Miner side: a plain transfer to the treasury, credited to this coldkey's account. -> (block, index)."""
    import bittensor as bt  # noqa: PLC0415

    if asset == "tao":
        call = bt.calls.Balances.transfer_keep_alive(dest=treasury, value=int(amount_rao))
    elif asset == "alpha":
        call = bt.calls.SubtensorModule.transfer_stake(destination_coldkey=treasury, hotkey=staked_hotkey,
                                                       origin_netuid=netuid, destination_netuid=netuid,
                                                       alpha_amount=int(amount_rao))
    else:
        raise ValueError(asset)
    res = subtensor.submit_call(call, wallet)
    res.raise_for_failure()
    blk, idx = map(int, res.extrinsic_id.split("-"))
    return blk, idx
