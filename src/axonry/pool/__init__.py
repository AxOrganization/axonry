"""All GPU machines: the hosts we own first (pool/owned.py), then machines rented on vast.ai (pool/vast.py) when that is
enabled. One table holds both; each row is driven by the pool it came from. How many machines are in use is decided
by the orchestrator (runs at once, evaluation machines), never per source: a purpose an owned host cannot take goes to
vast.ai."""
from .machine import MachineError, is_eval, sample
from .owned import OWNED


class NoMachine(MachineError):
    """No machine is free (the owned hosts are full and vast.ai is disabled): wait for one, nothing failed."""


class Machines:
    def __init__(self, db, key: str, owned=None, vast=None, items=None):
        """items: () -> the benchmark items' pack, built before an evaluation machine is rented (so that our own
        failure to build it never ends up blamed on a machine)."""
        self.db, self.key, self.owned, self.vast, self.items = db, key, owned, vast, items

    def _of(self, row):
        return self.owned if row["provider_id"].startswith(OWNED) else self.vast

    def get(self, purpose) -> dict | None:
        return next((m for m in self.db.machines() if m["name"] == purpose), None)

    def active(self) -> list:
        return self.db.machines()

    def machine(self, row):
        return self._of(row).machine(row)

    def rent(self, purpose: str, gpu_name: str, gpus: int):
        if is_eval(purpose) and self.items:
            self.items()
        if self.owned and self.owned.rent(purpose, gpu_name, gpus):
            return
        if self.vast is None:
            raise NoMachine(f"no owned machine free for {purpose} ({gpus} x {gpu_name}) and vast.ai is disabled")
        try:
            self.vast.rent(purpose, gpu_name, gpus)
        except MachineError as e:
            if "LowCredit" in type(e).__name__ or "insufficient_credit" in str(e):
                raise NoMachine(str(e)) from e     # the account, not the entry: the queue waits, nothing backs off
            raise

    def advance(self, purpose) -> str:
        row = self.get(purpose)
        return "gone" if row is None else self._of(row).advance(purpose)

    def release(self, purpose, note="released"):
        row = self.get(purpose)
        if row is not None:
            self._of(row).release(purpose, note)

    def hand_over(self, purpose, public_key: str, deadline: int):
        return self._of(self.get(purpose)).hand_over(purpose, public_key, deadline)

    def _side_by_side(self, purposes, call) -> dict:
        """call(purpose, row) for each machine at once: {purpose: its result, or the exception it raised}. The rows
        are read here (the DB stays in this thread); only the remote rounds run in parallel."""
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        rows = {p: row for p in purposes if (row := self.get(p)) is not None}

        def one(item):
            try:
                return call(*item)
            except Exception as e:  # noqa: BLE001 - one machine's failure is that machine's result only
                return e
        with ThreadPoolExecutor(max_workers=max(1, len(rows))) as ex:
            return dict(zip(rows, ex.map(one, rows.items())))

    def sample_all(self, purposes) -> dict:
        """Each machine's sample: {purpose: its sample, or the exception reading it raised}."""
        return self._side_by_side(purposes, lambda p, row: sample(self.machine(row), self.key))

    def vanished_all(self, purposes) -> dict:
        """{purpose: whether its machine vanished, or the exception checking it raised}."""
        return self._side_by_side(purposes, lambda p, row: self._of(row).vanished(p, row))

    def probe(self, purpose) -> dict:
        return self._of(self.get(purpose)).probe(purpose)

    def vanished(self, purpose) -> bool:
        row = self.get(purpose)
        return row is not None and self._of(row).vanished(purpose)
