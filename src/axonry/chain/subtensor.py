"""Chain access through the bittensor 11 SDK (`bittensor.Subtensor`). Read-only for the orchestrator; the miner CLI
also uses `submit_commitment` and `submit_deposit`. Only finalized blocks are ever read for intake.

A read of the state at a given block goes to an archive node (a lite node keeps only about the last 300 blocks, so
intake could not catch up after a longer pause); reads of the current state go to the network's own node.
"""
from .extrinsics import parse_extrinsic

ZERO_ACCOUNT = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"   # default of SubtensorModule.Owner


class BlockUnreadable(Exception):
    """A node returned a block incomplete (e.g. right after it reconnected): read it again later."""


class Chain:
    def __init__(self, network="finney", subtensor=None):
        if subtensor is None:
            import bittensor as bt  # noqa: PLC0415
            subtensor = bt.Subtensor(network=network)
        self.sub, self.network, self._archive = subtensor, network, None

    def _at(self, block):
        """The node to read the state at `block` from: the archive for a past block, else the network's node."""
        if block is None:
            return self.sub
        if self._archive is None:
            import bittensor as bt  # noqa: PLC0415
            self._archive = bt.Subtensor(network="archive" if self.network == "finney" else self.network)
        return self._archive

    @property
    def _S(self):
        import bittensor as bt  # noqa: PLC0415
        return bt.storage.SubtensorModule

    def finalized_head(self) -> int:
        return int(self.sub.finalized_block)

    def owner_of(self, hotkey, block):
        v = self._at(block).query(self._S.Owner, [hotkey], block=block)
        v = str(v) if v else None
        return None if v in (None, ZERO_ACCOUNT) else v

    def registration_owner(self, hotkey, netuid, block):
        """(block of the hotkey's registration on the subnet as of `block`, its owning coldkey at that block);
        (None, None) if the hotkey had no uid at `block`. Raises when the chain cannot be read."""
        uid = self.uid_of(hotkey, netuid, block)
        if uid is None:
            return None, None
        reg = int(self._at(block).query(self._S.BlockAtRegistration, [netuid, uid], block=block))
        return reg, self.owner_of(hotkey, reg)

    def uid_of(self, hotkey, netuid, block=None):
        v = self._at(block).query(self._S.Uids, [netuid, hotkey], block=block)
        return None if v is None else int(v)

    def alpha_price(self, netuid, block):
        """TAO per alpha from the swap runtime API at `block` (authoritative under the swap pallet)."""
        p = self._at(block).prices.alpha_price(netuid, block=block)
        return float(p["tao_per_alpha"])

    def uid_hotkeys(self, netuid, block=None) -> dict:
        return {int(k): str(v) for k, v in self._at(block).query_map(self._S.Keys, [netuid], block=block)}

    def has_committed(self, hotkey, netuid) -> bool:
        """Whether this hotkey ever set a commitment on the subnet (the storage keeps its latest one)."""
        import bittensor as bt  # noqa: PLC0415
        return self.sub.query(bt.storage.Commitments.CommitmentOf, [netuid, hotkey]) is not None

    def items_in_block(self, block, treasury, netuid) -> list:
        """Raises BlockUnreadable when the node returns the block incomplete (no timestamp: its extrinsics cannot be
        trusted either); the block is read again later."""
        return self._items_in_block(block, treasury, netuid)

    def _items_in_block(self, block, treasury, netuid) -> list:
        """The commitments and deposits of one block, in extrinsic order."""
        import bittensor as bt  # noqa: PLC0415

        node = self._at(block)
        info = node.block_info(block)
        if info is None or info.timestamp is None:     # every block carries its timestamp: this read is incomplete
            raise BlockUnreadable(f"block {block}: the node returned it incomplete")
        ts = int(info.timestamp.timestamp())
        status = {}
        for e in node.query(bt.storage.System.Events, block=block):
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
