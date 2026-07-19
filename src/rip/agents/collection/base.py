"""Collection agent contract.

Collection agents FETCH ONLY. They emit RawDocument records (a schema with no
fields for interpretation) plus a RunManifest. Politeness (delays, retries,
backoff) lives here so every source agent inherits it.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterable

from rich.console import Console

from ...core.models import RawDocument, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()


class BaseCollector(ABC):
    name: str = "base"

    def __init__(self, store: Store, params: dict):
        self.store = store
        self.params = params
        self.manifest = RunManifest(
            agent=self.name,
            run_id=new_run_id(self.name),
            started_at=utcnow(),
            params={k: v for k, v in params.items() if "secret" not in k.lower()},
        )

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(can_run, reason). Missing credentials => (False, why)."""

    @abstractmethod
    def fetch(self) -> Iterable[RawDocument]:
        """Yield raw documents. No summarizing, no inference."""

    def run(self) -> RunManifest:
        ok, reason = self.available()
        if not ok:
            self.manifest.status = "skipped"
            self.manifest.notes.append(reason)
            self.manifest.finished_at = utcnow()
            self.store.save_manifest(self.manifest)
            console.print(f"[yellow]⊘ {self.name}: skipped — {reason}[/yellow]")
            return self.manifest

        fetched = written = errors = 0
        try:
            batch: list[RawDocument] = []
            for doc in self.fetch():
                fetched += 1
                batch.append(doc)
                if len(batch) >= 200:
                    written += self.store.add_raw(batch)
                    batch = []
            written += self.store.add_raw(batch)
            self.manifest.status = "ok"
        except Exception as e:  # partial results are still saved above
            errors += 1
            self.manifest.status = "partial" if written else "failed"
            self.manifest.notes.append(f"{type(e).__name__}: {e}")
            console.print(f"[red]✗ {self.name}: {e}[/red]")
        finally:
            self.manifest.counts = {"fetched": fetched, "written_new": written, "errors": errors}
            self.manifest.finished_at = utcnow()
            self.store.save_manifest(self.manifest)
            console.print(
                f"[green]✓ {self.name}[/green] fetched={fetched} new={written} status={self.manifest.status}"
            )
        return self.manifest

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
