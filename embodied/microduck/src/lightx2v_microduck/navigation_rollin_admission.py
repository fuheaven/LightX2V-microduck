"""Admit student recordings only on pinned, released obstacle training layouts.

The guarded release has its own fixed layout split. A numeric seed range or the
legacy hash split cannot authorize these cases. This module admits recording,
not teacher labels, optimizer input or an acceptance result.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT
from .navigation_augmentation import contained, sha256
from .navigation_extended_qualification import TASKS, layout_sha256, require
from .navigation_extended_release import load_extended_release


@dataclass(frozen=True)
class GuardedTrainingCase:
    root: str
    index_sha256: str
    task: str
    seed: int
    path: str
    metadata_sha256: str
    navigation_sha256: str
    layout_sha256: str

    def binding(self):
        return dict(schema='guarded_training_rollin_admission_v1',
                    release_root=self.root, release_index_sha256=self.index_sha256,
                    task=self.task, seed=self.seed, split='train',
                    scene='room_curriculum_one', profile='official_bam',
                    source_path=self.path, source_metadata_sha256=self.metadata_sha256,
                    source_navigation_sha256=self.navigation_sha256,
                    layout_sha256=self.layout_sha256,
                    action_contract=FORWARD04_NAVIGATION_CONTRACT)

    def validate_request(self, *, task, seed, scene, profile, contract, record_only):
        require(type(record_only) is bool and (task, seed, scene, profile, contract, record_only) ==
                (self.task, self.seed, 'room_curriculum_one', 'official_bam',
                 FORWARD04_NAVIGATION_CONTRACT, True),
                'guarded admission only permits this training case and record-only execution')

    def _initial_observation(self):
        root = Path(self.root)
        require(sha256(root/'index.json') == self.index_sha256, 'guarded release index changed')
        folder = contained(root, self.path)
        require(sha256(folder/'episode.json') == self.metadata_sha256
                and sha256(folder/'navigation.npz') == self.navigation_sha256,
                'admitted teacher observation changed')
        meta = json.loads((folder/'episode.json').read_text())
        require((meta['task'], meta['seed'], meta['split'], meta['layout_sha256']) ==
                (self.task, self.seed, 'train', self.layout_sha256),
                'admitted source metadata changed')
        with np.load(folder/'navigation.npz', allow_pickle=False) as data:
            return data['rgb'][0].copy(), data['proprio'][0].copy(), meta['instruction']

    def verify_initial(self, env):
        require(env.tick == 0 and layout_sha256(env.evaluator_state()) == self.layout_sha256,
                'actual student initialization differs from its qualified training geometry')
        rgb, proprio, instruction = self._initial_observation()
        env.render_enabled = True
        observation = env.observe()
        require(observation.tick == observation.camera_tick == 0
                and env.task.instruction == instruction,
                'student initialization camera or instruction changed')
        np.testing.assert_array_equal(observation.rgb, rgb)
        np.testing.assert_array_equal(observation.state[:48], proprio)

    def verify_recorded_start(self, meta, recording):
        require(meta.get('training_admission') == self.binding()
                and (meta.get('task'), meta.get('seed'), meta.get('split')) ==
                (self.task, self.seed, 'train')
                and meta['evaluator']['scene']['name'] == 'room_curriculum_one',
                'student recording does not match its admitted training case')
        rgb, proprio, instruction = self._initial_observation()
        # Historical fixed-history recordings did not save the text. New
        # recordings bind it to the qualified task, including memory selection.
        if 'instruction' in meta:
            require(meta['instruction'] == instruction,
                    'recorded instruction differs from its qualified training case')
        np.testing.assert_array_equal(recording['rgb'][0], rgb)
        np.testing.assert_array_equal(recording['proprio48'][0], proprio)


def admit_guarded_cases(root, base_dataset, expected_index_sha256, evaluation_protocol,
                        seeds, tasks):
    """Verify the existing complete release, then select only its train members."""
    require(bool(seeds) and all(type(s) is int for s in seeds)
            and len(seeds) == len(set(seeds)) and bool(tasks)
            and len(tasks) == len(set(tasks)) and set(tasks) <= TASKS,
            'unique supported training seeds and tasks are required')
    reserved = (set(evaluation_protocol['development_seeds'])
                | set(evaluation_protocol['independent_final_seeds'])
                | set(range(10000, 10200)))
    require(not reserved.intersection(seeds), 'development or reserved layout cannot enter roll-in')
    entries = load_extended_release(root, base_dataset, expected_index_sha256)
    train = {(e['task'], e['seed']): e for e in entries
             if e['split'] == 'train' and e['eligible'] is True}
    requested = {(task, seed) for seed in seeds for task in tasks}
    require(requested <= set(train), 'every requested case must belong to the qualified train release')
    return {(task, seed): GuardedTrainingCase(
                str(Path(root).resolve()), expected_index_sha256, task, seed,
                train[task, seed]['path'], train[task, seed]['metadata_sha256'],
                train[task, seed]['sha256'], train[task, seed]['layout_sha256'])
            for task, seed in sorted(requested)}
