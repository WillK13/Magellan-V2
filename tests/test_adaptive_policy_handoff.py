from __future__ import annotations

from datetime import datetime, timezone

from magellan.config.policy_models import AdaptivePolicy, ObjectiveWeights
from magellan.models.types import ActionType, RawActionEstimate, TaskProfile
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore


def test_policy_snapshot_merges_on_destination(tmp_path) -> None:
    source_store = AdaptivePolicyStore(tmp_path / "source")
    service = AdaptivePolicyService(
        AdaptivePolicy(),
        ObjectiveWeights(time=0.25, carbon=0.5, cost=0.25),
        source_store,
    )
    task = TaskProfile(
        task_id="moving-task",
        workload_type="counter",
        current_node_id="boston",
        power_kw=0.1,
        checkpoint_bytes=10,
        estimated_remaining_seconds=60,
    )
    service.prepare(
        task,
        [
            RawActionEstimate(
                action=ActionType.CONTINUE,
                source_node_id="boston",
                time_seconds=60,
                carbon_grams=10,
                cost_usd=1,
            )
        ],
        datetime.now(timezone.utc),
    )
    snapshot = source_store.get("moving-task")
    assert snapshot is not None

    destination = AdaptivePolicyStore(tmp_path / "destination")
    assert destination.merge(snapshot) is True
    restored = destination.get("moving-task")
    assert restored is not None
    assert restored.normalization.time.samples
    assert restored.effective_weights == snapshot.effective_weights
