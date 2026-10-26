"""Attribute actual training microbatch use to the immutable weighted dataset.

An item index travels through DataLoader/Accelerate as audit metadata and is
removed before invoking the model. Prefetched, validation, and other-rank items
are not included in the logging rank's counts. Counts describe sample uses,
not unique examples, gradient contributions, or a changed sampling policy.
"""
from collections import Counter
import hashlib
import json
import operator
from pathlib import Path
import re


SAMPLE_INDEX_KEY = '_microduck_sample_index'


def _identity(dataset, index):
    entry, start = dataset.samples[index]
    task, digest = entry.get('task'), entry.get('sha256')
    if (type(start) is not int or start < 0 or not isinstance(task, str)
            or re.fullmatch(r'[a-z][a-z0-9_]{0,63}', task) is None
            or not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None
            or not isinstance(entry.get('path'), str) or not entry['path']):
        raise ValueError('sample usage requires an indexed task, tensor hash, path, and window start')
    return (str(Path(entry.get('_root', dataset.root)).resolve()), entry['path'], digest, start, task)


class TrainingSampleCatalog:
    def __init__(self, dataset):
        if not len(dataset) or len(dataset.samples) != len(dataset):
            raise ValueError('sample usage requires the complete ordered weighted sample list')
        self.identities = tuple(_identity(dataset, i) for i in range(len(dataset)))
        roots = sorted({row[0] for row in self.identities})
        self.source_ids = {root: f'source_{i:03d}' for i, root in enumerate(roots)}
        self.labels = tuple((self.source_ids[row[0]], row[4]) for row in self.identities)

    def manifest(self):
        inventory = Counter(self.labels)
        return dict(schema='microduck_actual_training_sample_usage_v1',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            complete_sampling_sha256=hashlib.sha256(json.dumps(
                self.identities, separators=(',', ':')).encode()).hexdigest(),
            weighted_items=len(self.identities),
            sources={source: dict(root=root, weighted_inventory_by_task={task: count
                for (sid, task), count in sorted(inventory.items()) if sid == source})
                for root, source in self.source_ids.items()},
            counting='actual_main_process_grad_enabled_loss_microbatches',
            unit='sample_uses_including_repeated_weighted_items',
            interval='since_previous_training_log', validation_included=False,
            prefetch_counted=False, optimizer_warmup_counted=False, distributed_global_counts=False,
            gradient_contributions_measured=False, metadata_removed_before_model=True,
            resampled=False, tensor_values_changed=False,
            resume='new_log_window; historical counts remain in their original metrics records')

    def counts(self, sample_ids, batch_items):
        import torch

        if (not isinstance(sample_ids, torch.Tensor) or sample_ids.dtype != torch.int64
                or sample_ids.ndim != 1 or tuple(sample_ids.shape) != (batch_items,)):
            raise ValueError('training sample IDs must be a batched int64 tensor')
        ids = sample_ids.detach().cpu().tolist()
        if any(not 0 <= i < len(self.identities) for i in ids):
            raise ValueError('training sample ID is outside the bound weighted dataset')
        result = Counter()
        for i in ids:
            source, task = self.labels[i]
            result['source_samples/'+source] += 1
            result['task_samples/'+task] += 1
            result['source_task_samples/'+source+'/'+task] += 1
        return result


class IndexedTrainingView:
    """Attach an index without modifying samples, RNG, or loader ordering."""
    def __init__(self, dataset, catalog):
        if tuple(_identity(dataset, i) for i in range(len(dataset))) != catalog.identities:
            raise ValueError('sample usage catalog differs from the training dataset')
        self.dataset, self.catalog = dataset, catalog

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.catalog.identities)

    def __getitem__(self, index):
        if isinstance(index, bool):
            raise IndexError('boolean training index is ambiguous')
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError('training index is outside the bound dataset')
        if (len(self.dataset) != len(self) or len(self.dataset.samples) != len(self)
                or _identity(self.dataset, index) != self.catalog.identities[index]):
            raise ValueError('training sample identity changed after usage binding')
        sample = dict(self.dataset[index])
        if SAMPLE_INDEX_KEY in sample:
            raise ValueError('underlying sample already contains usage metadata')
        sample[SAMPLE_INDEX_KEY] = index
        return sample


def model_sample_and_usage(sample):
    """Keep audit identity out of the vision/action model and retention teacher."""
    if SAMPLE_INDEX_KEY not in sample:
        return sample, None
    model_sample = dict(sample)
    ids = model_sample.pop(SAMPLE_INDEX_KEY)
    return model_sample, ids
