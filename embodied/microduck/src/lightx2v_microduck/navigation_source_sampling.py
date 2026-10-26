"""Explicit source exposure for new training stages; legacy order is unchanged.

Integer repetition changes how often existing clips are sampled, never their
observations or labels. All included roots must be named, and resume receipts
bind both these multipliers and each source index.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
from pathlib import Path


def canonical_repeats(value, expected_roots):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('source repetition needs an explicit nonempty mapping')
    result = {}
    for root, count in value.items():
        if not isinstance(root, str) or not root or type(count) is not int or count < 1:
            raise ValueError('source repeats require path strings and positive integer counts')
        key = str(Path(root).resolve())
        if key in result:
            raise ValueError('duplicate canonical source path')
        result[key] = count
    if set(result) != {str(Path(p).resolve()) for p in expected_roots}:
        raise ValueError('repeat mapping must name every included source exactly once')
    return dict(sorted(result.items()))


def repeat_samples(samples, root, split, repeats):
    if repeats is None:
        return samples, None  # Preserve the existing sampler's exact index order.
    if split != 'train':
        raise ValueError('source repetition is training-only')
    base = str(Path(root).resolve())
    sources = [str(Path(entry.get('_root', base)).resolve()) for entry, _ in samples]
    weights = canonical_repeats(repeats, set(sources))
    expanded = [sample for sample, source in zip(samples, sources) for _ in range(weights[source])]
    before = Counter(sources)
    exposure = {source: {'repeats': weights[source], 'original_samples': before[source],
                         'weighted_samples': before[source] * weights[source]}
                for source in weights}
    return expanded, exposure


def training_source_repeat_contract(train, validation):
    if validation.get('sample_source_repeats') is not None:
        raise ValueError('validation source repetition is forbidden')
    value = train.get('sample_source_repeats')
    if value is None:
        return None
    if train.get('augmentation_roots') or train.get('retarget_roots'):
        raise ValueError('source repetition currently supports full teachers and recoveries')
    roots = [train['root'], *train.get('qualified_teacher_roots', []), *train.get('recovery_roots', [])]
    weights = canonical_repeats(value, roots)
    return {'schema': 'microduck_training_source_repeats_v1', 'sources': [
        {'root': root, 'repeats': count,
         'index_sha256': hashlib.sha256((Path(root) / 'index.json').read_bytes()).hexdigest()}
        for root, count in weights.items()]}


def verify_resume_source_repeats(receipt, contract):
    if receipt.get('sample_source_repeats') != contract:
        raise ValueError('resume source exposure changed; initialize an explicit new training stage')
