"""Attach an explicit, source-bound retention mask without resampling data."""
import hashlib
import json
import operator
from pathlib import Path


def _identity(dataset, index):
    entry, start = dataset.samples[index]
    if type(start) is not int or start < 0:
        raise ValueError("retention source has an invalid window start")
    return (str(Path(entry.get('_root', dataset.root)).resolve()), entry['path'],
            entry['sha256'], start)


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def batch_validation_retention(sample, batch):
    """Preserve the explicit validation exclusion across the author's batching.

    The upstream converter copies a fixed set of model inputs and drops metadata.
    This adapter carries only the authenticated false mask, without adding any
    teacher supervision or modifying the author's tensors.
    """
    import torch

    mask = sample.get('retention_is_retained')
    action = batch.get('action')
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise ValueError('validation requires a batched action tensor')
    size = action.shape[0]
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape not in ((), (size,)) or bool(mask.any())):
        raise ValueError('validation requires an explicit false retention mask')
    if mask.ndim == 0 and size != 1:
        raise ValueError('one validation mask cannot describe multiple samples')
    if 'retention_is_retained' in batch:
        raise ValueError('validation batching already supplied a retention decision')
    return dict(batch, retention_is_retained=mask.reshape(size))


class RetainedTrainingView:
    """Wrap the already audited rehearsal dataset, preserving its sample clock.

    The underlying loader authenticates the retained recipe and constructs its
    complete, weighted prefix. New obstacle/correction items receive no parent
    constraint. Validation never supplies retention supervision. The wrapper
    neither adds samples nor changes images, labels, normalization or padding.
    """
    def __init__(self, dataset, *, split, retained_recipe_sha256,
                 expected_samples, expected_retained_samples):
        if split not in ('train', 'validation'):
            raise ValueError("retention view supports training and validation only")
        if (type(expected_samples) is not int or type(expected_retained_samples) is not int
                or not 0 < expected_retained_samples < expected_samples
                or len(dataset) != expected_samples or len(dataset.samples) != expected_samples
                or dataset.retained_samples != expected_retained_samples
                or dataset.retained_recipe_sha256 != retained_recipe_sha256):
            raise ValueError("retained dataset membership differs from its binding")
        if (not isinstance(retained_recipe_sha256, str) or len(retained_recipe_sha256) != 64
                or set(retained_recipe_sha256) - set('0123456789abcdef')):
            raise ValueError("invalid retained recipe hash")
        self.dataset, self.split = dataset, split
        self._identities = [_identity(dataset, i) for i in range(expected_samples)]
        self._retained_count = expected_retained_samples
        for entry, _ in dataset.samples[:expected_retained_samples]:
            if entry.get('_extended_teacher_episode') or entry.get('_guarded_recovery_episode'):
                raise ValueError("new obstacle items cannot enter the retained prefix")
        self.retention_binding = dict(schema='microduck_retained_training_item_mask_v1', split=split,
            retained_recipe_sha256=retained_recipe_sha256, total_samples=expected_samples,
            retained_prefix_samples=expected_retained_samples,
            eligible_retention_samples=expected_retained_samples if split == 'train' else 0,
            retained_sampling_sha256=_digest(self._identities[:expected_retained_samples]),
            complete_sampling_sha256=_digest(self._identities), resampled=False,
            tensor_values_changed=False, teacher_targets_added=False)

    def __len__(self):
        return len(self._identities)

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        import torch

        if isinstance(index, bool):
            raise IndexError("boolean item index is ambiguous")
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError("retention item index is outside the bound sample list")
        if (len(self.dataset.samples) != len(self)
                or _identity(self.dataset, index) != self._identities[index]):
            raise ValueError("sample membership changed after retention binding")
        sample = dict(self.dataset[index])
        if 'retention_is_retained' in sample:
            raise ValueError("underlying data already contains a retention decision")
        sample['retention_is_retained'] = torch.tensor(
            self.split == 'train' and index < self._retained_count, dtype=torch.bool)
        return sample
