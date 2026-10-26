"""Observe training microbatches without changing their objective or RNG.

The pinned author trainer logs only the last microbatch at a logging boundary.
These additional local metrics describe all main-process training microbatches
since the previous training log. They are scalar averages, not gradient shares,
global distributed averages, or necessarily one optimizer update.
"""

import hashlib
import math
from pathlib import Path
from collections import Counter


def metric_window_manifest():
    return dict(schema='microduck_navigation_training_metric_window_v1',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        aggregation='main_process_training_microbatches_since_previous_training_log',
        scalar_weighting='equal_weight_per_microbatch', validation_included=False,
        distributed_global_average=False, gradient_contributions_measured=False,
        original_train_metrics='unchanged_author_last_microbatch_metrics',
        affects_objective=False, checkpoint_state_required=False)


class MicrobatchMetricWindow:
    def __init__(self):
        self.sample_catalog = None
        self._clear()

    def _clear(self):
        self.microbatches = self.samples = self.retained_items = self.eligible_microbatches = 0
        self.totals = dict(loss=0., loss_video=0., loss_action=0., loss_retention=0.)
        self.sample_counts = Counter()

    def bind_sample_catalog(self, catalog):
        if self.sample_catalog is not None or self.microbatches:
            raise ValueError('sample catalog must be bound once before observing training')
        self.sample_catalog = catalog

    def add(self, loss, details, *, batch_items, sample_ids=None):
        if type(batch_items) is not int or batch_items < 1 or not isinstance(details, dict):
            raise ValueError('metric window requires a nonempty training microbatch')
        retained = details.get('retention_items', 0)
        if type(retained) is not int or not 0 <= retained <= batch_items:
            raise ValueError('invalid retained-item count in metric window')
        values = dict(loss=loss, loss_video=details.get('loss_video'),
                      loss_action=details.get('loss_action'), loss_retention=details.get('loss_retention', 0.))
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ValueError('metric window requires finite reported loss scalars')
        counts = self.sample_catalog.counts(sample_ids, batch_items) if self.sample_catalog is not None else {}
        if self.sample_catalog is None and sample_ids is not None:
            raise ValueError('sample IDs have no bound training catalog')
        # Validate before changing the counters, so rejected records cannot
        # contaminate later logs. Preserve scalar weighting across short batches.
        self.microbatches += 1
        self.samples += batch_items
        self.retained_items += retained
        self.eligible_microbatches += int(retained > 0)
        for name, value in values.items():
            self.totals[name] += value
        self.sample_counts.update(counts)

    def flush(self):
        if not self.microbatches:
            raise ValueError('training log has no observed training microbatches')
        result = dict(microbatches=self.microbatches, samples=self.samples,
                      retained_items=self.retained_items, eligible_microbatches=self.eligible_microbatches)
        result.update({name+'_microbatch_mean': total/self.microbatches for name, total in self.totals.items()})
        result.update(self.sample_counts)
        self._clear()
        return {'train_window/'+key: value for key, value in result.items()}


def observe_training_loss(window, loss, details, *, batch_items, main_process, sample_ids=None):
    """Read detached scalars on the logging rank; validation never enters it."""
    import torch

    if type(main_process) is not bool:
        raise ValueError('metric ownership must explicitly identify the logging rank')
    if not main_process or not torch.is_grad_enabled() or not loss.requires_grad:
        return
    window.add(float(loss.detach()), details, batch_items=batch_items, sample_ids=sample_ids)
