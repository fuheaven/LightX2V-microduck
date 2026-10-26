"""Rebuild hybrid actor inputs and skill execution from independent physics.

No endpoint or learned model is called. Neural actions are taken from their
original recordings, never claimed as recomputed. Supervisor replies, each
skill's camera history, feedback, memory, horizons and physical outcomes are
independently checked. Historical timing and capability acceptance stay separate.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from .navigation import DECISION_TICKS
from .navigation_history import history_spec_from_contract
from .navigation_hybrid import HybridNavigationPolicy, HYBRID_PROTOCOL, NAVIGATION_INSTRUCTIONS, validate_skill_request
from .navigation_model_inputs import RECORDING_CONTRACT
from .navigation_outcomes import read, require, sha
from .navigation_recording_replay import replay_recording
from .planner_response import decode_framed_object
from .planner_validation_feedback import VALIDATION_FEEDBACK_PROTOCOL
from .vlm_service import canonical, parse_request
from .wam_history_policy import FastWAMNavigationHistoryPolicy
from .wam_policy import FastWAMNavigationPolicy


class OfflineHybridObserver:
    protocol = HYBRID_PROTOCOL
    families = ("VLM+WAM",)
    policy_class = HybridNavigationPolicy

    def initialize_session(self):
        self.policy.reset(self.episode_id)

    def policy_options(self, metadata):
        return {}

    def __init__(self, episode, requests, expected_identity):
        self.requests, self.files = Path(requests), {}
        meta = episode['policy']
        self.validation_feedback = meta.get('validation_feedback_protocol') == VALIDATION_FEEDBACK_PROTOCOL
        require('validation_feedback_protocol' not in meta or self.validation_feedback,
                'unsupported hybrid correction protocol')
        require(episode['mode'] == 'offline' and meta.get('family') in self.families
                and meta.get('protocol') == self.protocol, 'unsupported hybrid audit protocol')
        service = self.read_file(self.requests / 'service.json')
        self.identity = hashlib.sha256(canonical(service['identity'])).hexdigest()
        require(self.identity == expected_identity == service['identity_sha256']
                == meta['expected_service_identity'], 'hybrid service identity differs')
        require(service['identity']['model_id'] == meta['supervisor_model'], 'hybrid model differs')
        self.episode_id = episode['evaluator']['episode_id']
        require(re.fullmatch(r'find_(red|blue|yellow)_ball', episode['task']), 'unknown hybrid goal')
        self.goal_skill = 'navigate_' + episode['task'][5:-5]
        self.original_generation = None
        self.current, self.prepared = None, None
        self.chunks, self.request_ids = {}, []
        self.held_ticks, self.failed_requests = [], []
        self.learned_calls, self.selected_tools = Counter(), Counter()
        self.predictions, self.execution_ticks, self.wrong_goal_delegations = 0, 0, 0
        skills = {}
        for name, binding in meta['learned_skills'].items():
            policy = binding['policy']
            require(binding['instruction'] == NAVIGATION_INSTRUCTIONS.get(name)
                    and policy.get('model_input_recording') == RECORDING_CONTRACT,
                    'learned skill needs its explicit input recording contract')
            history = policy.get('history_contract')
            size = history_spec_from_contract(history).output_size if history else 256
            core = SimpleNamespace(policy_profile='microduck_navigation', action_dim=5,
                robot_state_dim=48, camera_size=size, gripper_postprocess=False,
                config={'navigation_contract': meta['contract']}, history_contract=history,
                reset=lambda: None, predict_action_chunk=self.recorded_actions(name))
            cls = FastWAMNavigationHistoryPolicy if history else FastWAMNavigationPolicy
            skills[name] = cls(core, model_id=policy['model_id'], artifact_sha256=policy['artifact_sha256'],
                contract=meta['contract'], validity_ticks=episode['execution_clock']['max_age_ticks'],
                record_model_inputs=True)
            require(skills[name].metadata == policy, 'reconstructed learned skill metadata differs')
        self.policy = self.policy_class(skills=skills, endpoint='http://127.0.0.1:1/v1',
            model=meta['supervisor_model'], contract=meta['contract'], context_mode=meta['context_mode'],
            expected_service_identity=self.identity, validation_feedback=self.validation_feedback,
            **self.policy_options(meta))
        require(self.policy.metadata == meta, 'reconstructed hybrid source or policy differs')
        self.initialize_session()
        if self.validation_feedback:
            self.policy.client.request_raw = self.recorded_raw_reply
        else:
            self.policy.client.request_json = self.recorded_reply
        self.expected_holds = episode['scheduled_prediction_hold_ticks']

    def read_file(self, path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), 'missing regular evidence file: ' + str(path))
        digest = sha(path)
        require(path not in self.files or self.files[path] == digest, 'hybrid evidence changed')
        self.files[path] = digest
        return read(path)

    def recorded_actions(self, name):
        def replay(**arguments):
            require(self.current is not None
                    and self.current['diagnostics']['hybrid']['call']['tool'] == name,
                    'replay invoked a different learned skill')
            require(arguments['task_description'] == NAVIGATION_INSTRUCTIONS[name],
                    'delegate received a different instruction')
            self.learned_calls[name] += 1
            # This is recorded, already bounded output. Input receipts below
            # are independently regenerated; neural outputs are NOT recomputed.
            return np.asarray(self.current['submitted_actions'], np.float32).copy()
        return replay

    def request_record(self, identifier, payload):
        require(isinstance(identifier, str) and re.fullmatch(r'[0-9a-f]{32}', identifier)
                and identifier not in self.request_ids, 'invalid or duplicated hybrid request')
        directory = self.requests / identifier
        require(not directory.is_symlink(), 'request directory cannot be a symlink')
        original, receipt = (self.read_file(directory / n) for n in ('request.json', 'input.json'))
        digest = hashlib.sha256(canonical(original)).hexdigest()
        require(canonical(original) == canonical(payload) and digest == receipt['request_sha256']
                and receipt['identity_sha256'] == self.identity,
                'supervisor input differs from replayed images, sensors, goal, memory or feedback')
        require(parse_request(payload, self.policy.metadata['supervisor_model'])['images'] == receipt['images'],
                'supervisor received different images')
        return directory, digest

    def recorded_reply(self, payload):
        original = self.current['diagnostics']['hybrid']
        evidence = original['serving_evidence']
        identifier = evidence.get('request_id')
        directory, digest = self.request_record(identifier, payload)
        stored = self.read_file(directory / 'result.json')
        raw, response = stored['result'], stored['response']
        expected = {'request_id': identifier, 'request_sha256': digest,
                    'identity_sha256': self.identity, 'model_id': self.policy.metadata['supervisor_model']}
        require(raw['status'] == 'complete' and response['id'] == identifier
                and response['model'] == expected['model_id']
                and response['choices'][0]['message']['content'] == raw['content']
                and all(response['microduck_evidence'].get(k) == v for k, v in expected.items()),
                'raw supervisor output or its identity differs')
        value, framing = decode_framed_object(raw['content'])
        expected.update(framing)
        require(expected == evidence, 'raw supervisor framing evidence differs')
        value = self.policy.validate_call(value)
        require(value == original['call'], 'recorded skill differs from the raw model choice')
        self.request_ids.append(identifier)
        self.selected_tools[value['tool']] += 1
        return value, expected

    def observe_sensors(self, observation, control, predictions):
        if observation.tick % DECISION_TICKS:
            require(not predictions, 'hybrid prediction outside sensor cadence')
            return
        require(len(predictions) <= 1, 'multiple offline predictions for one source')
        captured = predictions[0]['source_monotonic_ns'] if predictions else control['observation_capture_monotonic_ns']
        observation = replace(observation, episode_id=self.episode_id, monotonic_ns=captured)
        self.prepared = self.policy.prepare_request(observation)
        required = self.policy.prediction_required(observation)
        require(required == bool(predictions), 'recorded hybrid prediction/hold schedule differs')
        if not required:
            self.held_ticks.append(observation.tick)

    def observe_prediction(self, observation, prediction):
        require(prediction['error'] is None and prediction['submitted_actions'] is not None
                and self.prepared is not None and self.prepared.observation.tick == observation.tick,
                'unprepared or failed hybrid prediction requires separate evidence')
        original = prediction['diagnostics']['hybrid']
        generation = original['generation']
        require(type(generation) is int and generation > 0
                and (self.original_generation is None or self.original_generation == generation),
                'hybrid generation changed within the recording')
        self.original_generation = generation
        self.current = prediction
        before_calls = sum(self.learned_calls.values())
        chunk = self.policy.predict_prepared(self.prepared)
        self.current = None
        measured = chunk.diagnostics['hybrid']
        # Session counter and measured durations cannot be remeasured from a
        # CPU replay. Decision sequence and every semantic source field can.
        ignored = {'generation', 'decision_key', 'supervisor_http_elapsed_ms', 'delegate_inference_elapsed_ms'}
        require({k: v for k, v in measured.items() if k not in ignored}
                == {k: v for k, v in original.items() if k not in ignored},
                'hybrid skill, source, finite window or delegate binding differs')
        require(original['decision_key'] == f"{generation}:{measured['decision_key'].partition(':')[2]}",
                'hybrid decision sequence differs')
        learned = original['execution_kind'] == 'learned_navigation'
        require(sum(self.learned_calls.values()) - before_calls == int(learned),
                'learned inference call inventory differs')
        for key in ('model_core_inputs', 'history_source_ticks', 'history_padding'):
            require(chunk.diagnostics.get(key) == prediction['diagnostics'].get(key),
                    'delegated model input differs: ' + key)
        if learned and original['call']['tool'] != self.goal_skill:
            self.wrong_goal_delegations += 1
        np.testing.assert_array_equal(chunk.actions, np.asarray(prediction['submitted_actions'], np.float32))
        require(chunk.valid_until_tick == prediction['valid_until_tick']
                and chunk.action_interval_ticks == prediction['action_interval_ticks']
                and chunk.contract == prediction['action_contract']
                and chunk.source_monotonic_ns == prediction['source_monotonic_ns'],
                'hybrid submitted chunk differs')
        require(chunk.source_monotonic_ns not in self.chunks, 'duplicate hybrid source capture')
        self.chunks[chunk.source_monotonic_ns] = chunk
        self.predictions += 1

    def observe_execution(self, before, after, intent, control):
        before = replace(before, episode_id=self.episode_id,
                         monotonic_ns=control['observation_capture_monotonic_ns'])
        after = replace(after, episode_id=self.episode_id)
        source = control['checked_plan_source_monotonic_ns']
        require(source is None or source in self.chunks, 'execution refers to an unaudited hybrid plan')
        self.policy.observe_execution(before, after, intent, control, self.chunks.get(source))
        self.execution_ticks += 1

    def observe_failure(self, observation, failure):
        require(failure['tick'] == observation.tick and failure['phase'] == 'execution',
                'terminal hybrid failure belongs to another observation')
        observation = replace(observation, episode_id=self.episode_id)
        request = self.policy.prepare_request(observation)
        require(self.policy.decision is None or observation.tick >= self.policy.decision['until_tick'],
                'delegate-only terminal failure requires separate model error evidence')
        payload = self.policy.request_payload(request)
        matches = []
        for path in self.requests.glob('*/request.json'):
            candidate = read(path)
            metadata = candidate.get('metadata', {})
            if (metadata.get('episode_id') == self.episode_id
                    and metadata.get('source_tick') == str(observation.tick)):
                matches.append(path.parent.name)
        require(len(matches) == 1, 'failed hybrid source lacks a unique original request')
        directory, digest = self.request_record(matches[0], payload)
        outcomes = {name: self.read_file(directory / name) for name in ('result.json', 'error.json')
                    if (directory / name).is_file()}
        require(outcomes, 'failed hybrid request lacks a terminal receipt')
        if 'result.json' in outcomes:
            stored = outcomes['result.json']
            raw, response = stored['result'], stored['response']
            expected = {'request_id': directory.name, 'request_sha256': digest,
                        'identity_sha256': self.identity, 'model_id': self.policy.metadata['supervisor_model']}
            require(response['id'] == directory.name and response['model'] == expected['model_id']
                    and response['choices'][0]['message']['content'] == raw['content']
                    and all(response['microduck_evidence'].get(k) == v for k, v in expected.items()),
                    'failed hybrid response belongs to another request')
        if failure['error_type'] in ('ValueError', 'JSONDecodeError'):
            raw = outcomes.get('result.json', {}).get('result', {})
            require(raw.get('status') == 'complete', 'validation failure lacks completed raw output')
            try:
                value, _ = decode_framed_object(raw['content'])
                self.policy.validate_call(value)
            except ValueError:
                pass
            else:
                raise ValueError('original hybrid validation error did not recur')
        else:
            require(failure['error_type'] in ('HTTPError', 'URLError', 'TimeoutError'),
                    'unsupported historical hybrid failure')
        self.failed_requests.append({'request_id': directory.name, 'source_tick': observation.tick,
            'client_error_type': failure['error_type'], 'service_outcomes': outcomes,
            'failure_cause_independently_verified': False, 'actions_executed': 0})

    def finish(self):
        require(self.held_ticks == self.expected_holds, 'hybrid held sensor cadence differs')
        require(all(sha(p) == digest for p, digest in self.files.items()), 'hybrid evidence changed during replay')
        return {'status': 'complete', 'exact_supervisor_requests': len(self.request_ids),
            'request_ids': self.request_ids, 'submitted_predictions_verified': self.predictions,
            'learned_core_inputs_verified': sum(self.learned_calls.values()),
            'learned_calls_by_skill': dict(self.learned_calls), 'selected_tools': dict(self.selected_tools),
            'delegate_goal_mismatch_predictions': self.wrong_goal_delegations,
            'exact_held_sensor_ticks': len(self.held_ticks), 'execution_feedback_ticks': self.execution_ticks,
            'failed_unexecuted_requests': self.failed_requests,
            'files_sha256': {str(p.relative_to(self.requests)): d for p, d in self.files.items()},
            'full_supervisor_input_regenerated': True, 'learned_core_inputs_regenerated': True,
            'learned_actions_reused_from_recording': True, 'neural_outputs_recomputed': False,
            'latency_remeasured': False, 'historical_runtime_identity_verified': False,
            'warmup_requests_outside_scope': True, 'formal_acceptance': False, 'hardware_validated': False}

    def close(self):
        self.policy.close()


def audit_offline_hybrid_episode(folder, case, requests):
    episode = read(Path(folder) / 'report.json')
    observer_type = OfflineHybridObserver
    if episode['policy'].get('validation_feedback_protocol') is not None:
        from .navigation_hybrid_feedback_audit import FeedbackHybridObserver
        observer_type = FeedbackHybridObserver
    observer = observer_type(episode, requests, case['service_identity_sha256'])
    try:
        physics = replay_recording(folder, case, policy_observer=observer)
        model_sources = observer.finish()
        require(model_sources['submitted_predictions_verified'] == physics['exact_prediction_source_observations']
                and model_sources['execution_feedback_ticks'] == physics['exact_control_ticks'],
                'hybrid model-source and physics coverage differs')
        return {'schema': 'microduck_offline_hybrid_physical_audit_v1', 'status': 'complete',
            'physics': physics, 'model_sources': model_sources, 'audit_source_sha256': sha(__file__),
            'formal_acceptance': False, 'hardware_validated': False}
    finally:
        observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-folder', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--case-sha256', required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and sha(args.case) == args.case_sha256, 'case changed or output exists')
    result = audit_offline_hybrid_episode(args.episode_folder, read(args.case), args.requests)
    require(sha(args.case) == args.case_sha256, 'case changed during hybrid replay')
    result['case_sha256'] = args.case_sha256
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'control_ticks': result['physics']['exact_control_ticks'],
        'supervisor_requests': result['model_sources']['exact_supervisor_requests'],
        'learned_inputs': result['model_sources']['learned_core_inputs_verified'], 'formal_acceptance': False}))


if __name__ == '__main__':
    main()
