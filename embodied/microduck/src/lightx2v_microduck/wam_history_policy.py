"""History-grid navigation adapter. Old single-frame artifacts cannot opt in silently."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from .contracts import Observation
from .navigation import DECISION_TICKS
from .navigation_action_contracts import action_contract, core_action_contract, verify_artifact_action_contract
from .navigation_history import CausalCameraHistory, HistoryGridSpec, history_spec_from_contract
from .navigation_visual_memory import CausalTargetCameraHistory, TargetMemoryGridSpec
from .wam_policy import FastWAMNavigationPolicy, artifact_sha256


@dataclass(frozen=True)
class HistoryNavigationRequest:
    observation: Observation
    spec: HistoryGridSpec
    source_ticks: tuple[int, ...]
    padding: tuple[bool, ...]
    session_generation: int


class FastWAMNavigationHistoryPolicy(FastWAMNavigationPolicy):
    def __init__(self, core, *, model_id, artifact_sha256, validity_ticks=25, contract=None,
                 record_model_inputs=False):
        history_contract = getattr(core, 'history_contract', None) or getattr(core, 'config', {}).get('navigation_history')
        if not isinstance(history_contract, dict):
            raise ValueError('a single-frame core is not a history model')
        spec = history_spec_from_contract(history_contract)
        if ((core.policy_profile, core.action_dim, core.robot_state_dim, core.camera_size)
                != ('microduck_navigation', 5, 48, spec.output_size) or core.gripper_postprocess
                or (validity_ticks is not None and
                    (type(validity_ticks) is not int or not 1 <= validity_ticks <= 256 * DECISION_TICKS))):
            raise ValueError('history core dimensions, action semantics or validity do not match')
        self.core, self.validity_ticks = core, validity_ticks
        self.contract = core_action_contract(core, contract)
        self.history = CausalTargetCameraHistory(spec) if isinstance(spec, TargetMemoryGridSpec) else CausalCameraHistory(spec)
        self.episode_id = ''
        self._history_generation = 0
        self.metadata = {'model_id': model_id, 'artifact_sha256': artifact_sha256,
                         'contract': self.contract, 'hardware_validated': False,
                         'input': 'head_rgb_history_grid+instruction+proprio48',
                         'history_contract': history_contract, 'uses_privileged_state': False}
        from .navigation_model_inputs import configure_recording
        configure_recording(self, record_model_inputs)

    @classmethod
    def from_artifact(cls, path, *, device='cuda'):
        path = Path(path).resolve()
        manifest = json.loads((path / 'navigation_model.json').read_text())
        action_contract(manifest['contract'])
        for name in ('model.pt', 'dataset_stats.json', 'lightx2v_config.json'):
            if artifact_sha256(path / name) != manifest['files'][name]:
                raise ValueError('history artifact checksum mismatch')
        config = json.loads((path / 'lightx2v_config.json').read_text())
        contract = verify_artifact_action_contract(manifest, config,
            json.loads((path / 'dataset_stats.json').read_text()))
        if not manifest.get('history_contract') or manifest['history_contract'] != config.get('navigation_history'):
            raise ValueError('artifact is not trained/exported for this history contract')
        config.update(adapter_model_path=str(path / 'model.pt'), dataset_stats_path=str(path / 'dataset_stats.json'), device=device)
        import lightx2v_platform.set_ai_device  # noqa: F401
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
        return cls(FastWAMPolicy.from_config(config), model_id=manifest['model_id'],
                   artifact_sha256=manifest['files']['model.pt'], contract=contract)

    def reset(self, episode_id):
        super().reset(episode_id)
        self.history.reset(self.episode_id)
        self._history_generation += 1

    def observe(self, observation):
        """Can run at sensor cadence between model predictions; never step physics."""
        if observation.rgb is None or observation.camera_tick != observation.tick:
            raise ValueError('history requires a fresh measured head-camera frame')
        if isinstance(self.history, CausalTargetCameraHistory):
            self.history.append(observation.episode_id, observation.tick, observation.rgb, instruction=observation.instruction)
        else:
            self.history.append(observation.episode_id, observation.tick, observation.rgb)

    def prepare_request(self, observation):
        """Run on the sensor thread even when an older inference is in flight.

        Observation copies make this request independent of later frames,
        pruning and replacement of pending inference requests.
        """
        if not observation.instruction:
            raise ValueError('history navigation requires an instruction')
        self.observe(observation)
        grid, ticks, padding = self.history.compose(observation.tick)
        return HistoryNavigationRequest(replace(observation, rgb=grid), self.history.spec,
                                        tuple(ticks.tolist()), tuple(padding.tolist()), self._history_generation)

    def predict_prepared(self, request):
        """Inference reads a snapshot; it never mutates the live camera history."""
        if (not isinstance(request, HistoryNavigationRequest) or request.spec != self.history.spec
                or request.session_generation != self._history_generation
                or request.observation.episode_id != self.episode_id):
            raise ValueError('prepared history request belongs to another contract or reset session')
        chunk = super().predict(request.observation)
        if (request.session_generation != self._history_generation
                or request.observation.episode_id != self.episode_id):
            raise ValueError('history inference completed after its session was reset')
        chunk.diagnostics.update(history_source_ticks=list(request.source_ticks),
                                 history_padding=list(request.padding))
        return chunk

    def predict(self, observation):
        return self.predict_prepared(self.prepare_request(observation))
