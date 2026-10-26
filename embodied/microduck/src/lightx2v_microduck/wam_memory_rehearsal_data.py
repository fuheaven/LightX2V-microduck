"""An explicit new input stage over unchanged, source-bound rehearsal samples.

The student sees RGB target memory. Its frozen retention parent receives the
original fixed camera history for the exact same sample. Neither view changes
the action targets, state, text, normalization, sample order or source weights.
"""
from .navigation_history import HistoryGridSpec, history_spec_from_contract
from .wam_data import FastWAMNavigationDataset
from .wam_rehearsal_history_data import FastWAMNavigationRehearsalHistoryDataset


MEMORY_REHEARSAL_TARGET = (
    'lightx2v_microduck.wam_memory_rehearsal_data.FastWAMNavigationMemoryRehearsalDataset')


def memory_rehearsal_spec(contract):
    if contract is None:
        raise ValueError('memory rehearsal requires its explicit new input contract')
    contract = dict(contract)
    spec = history_spec_from_contract(contract)
    if contract['schema'] != 'microduck_rgb_target_keyframe_v2':
        raise ValueError('memory rehearsal requires the verified v2 fallback contract')
    return spec


class FastWAMNavigationMemoryRehearsalDataset(FastWAMNavigationRehearsalHistoryDataset):
    def __init__(self, root, split='train', *, memory_input_contract,
                 correction_parent_recipe_path=None, correction_parent_recipe_sha256=None,
                 correction_sources=None, **kwargs):
        spec = memory_rehearsal_spec(memory_input_contract)
        from .navigation_correction_tail import append_correction_tail, correction_tail_contract

        config = dict(root=str(root), split=split, _target_=MEMORY_REHEARSAL_TARGET,
                      memory_input_contract=memory_input_contract, **kwargs)
        tail = dict(correction_parent_recipe_path=correction_parent_recipe_path,
                    correction_parent_recipe_sha256=correction_parent_recipe_sha256,
                    correction_sources=correction_sources)
        contract = correction_tail_contract(dict(config, **tail))
        # Existing rehearsal validation authenticates the retained recipe and
        # all appended sources before any input adaptation takes place.
        super().__init__(root, split, **kwargs)
        append_correction_tail(self, contract, config)
        original = HistoryGridSpec((300, 150, 50, 0))
        if self.history_contract != original.as_dict():
            raise ValueError('memory rehearsal requires the original fixed six-second input')
        self.retention_history_spec = original
        self.retention_history_contract = original.as_dict()
        self.history_spec = spec
        self.history_contract = spec.as_dict()
        self.history_target_keyframe = True

    def __getitem__(self, index):
        # Read and normalize once; both views use the identical physical time,
        # episode prefix and instruction. Only camera selection can differ.
        sample = FastWAMNavigationDataset.__getitem__(self, index)
        parent = self.compose_sample_history(index, dict(sample), self.retention_history_spec,
                                              target_keyframe=False)
        student = self.compose_sample_history(index, sample, self.history_spec,
                                               target_keyframe=True)
        for key in ('video', 'history_source_ticks', 'history_is_pad'):
            student['retention_' + key] = parent[key]
        return student
