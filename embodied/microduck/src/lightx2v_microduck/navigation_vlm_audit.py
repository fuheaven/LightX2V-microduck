"""Bind recorded VLM HTTP decisions to independently replayed robot observations.

Offline only: regenerate the complete planner input, including execution
feedback and model memory, from physical replay. Read the raw stored model
output without repairing it, then reconstruct the submitted action chunk.
This does not rerun model inference or establish capability acceptance.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .navigation_outcomes import read, require, sha
from .navigation_recording_replay import replay_recording
from .navigation_vlm import VLMNavigationPolicy
from .skill_planner import validate_plan
from .vlm_service import canonical, parse_request


class OfflineVLMObserver:
    def __init__(self, episode, requests, expected_identity):
        self.requests = Path(requests)
        self.files = {}
        service = self.read_file(self.requests / 'service.json')
        require(episode['mode'] == 'offline' and episode['policy'].get('family') == 'VLM',
                'requires an offline VLM recording')
        digest = hashlib.sha256(canonical(service['identity'])).hexdigest()
        require(digest == expected_identity == service['identity_sha256']
                == episode['policy']['expected_service_identity'], 'service identity differs')
        policy = episode['policy']
        require(service['identity']['model_id'] == policy['model'], 'service model differs')
        self.identity = digest
        self.episode_id = episode['evaluator']['episode_id']
        # Never open the endpoint. The client is used only to regenerate the
        # exact input contract; outputs come from immutable HTTP records.
        self.policy = VLMNavigationPolicy(
            endpoint='http://127.0.0.1:1/v1', model=policy['model'], contract=policy['contract'],
            context_mode=policy['context_mode'], expected_service_identity=digest,
            prompt_profile=policy.get('prompt_profile', 'legacy_v1'))
        self.policy.reset(self.episode_id)
        self.chunks = {}
        self.request_ids = []
        self.failed_requests = []
        self.executed_control_ticks = 0

    def read_file(self, path):
        path = Path(path)
        require(not path.is_symlink() and path.is_file(), 'missing regular evidence file: ' + str(path))
        digest = sha(path)
        if path in self.files:
            require(self.files[path] == digest, 'evidence changed during VLM replay')
        self.files[path] = digest
        return read(path)

    def observe_prediction(self, observation, prediction):
        require(prediction['submitted_actions'] is not None and prediction['error'] is None,
                'failed inference needs a separate failure-receipt audit')
        diagnostics = prediction['diagnostics']
        evidence = diagnostics.get('serving_evidence')
        require(isinstance(evidence, dict) and evidence.get('identity_sha256') == self.identity
                and evidence.get('model_id') == self.policy.metadata['model'],
                'prediction has no matching service evidence')
        identifier = evidence.get('request_id')
        require(isinstance(identifier, str) and re.fullmatch(r'[0-9a-f]{32}', identifier)
                and identifier not in self.request_ids, 'invalid or duplicate model request')
        directory = self.requests / identifier
        require(not directory.is_symlink(), 'request directory cannot be a symlink')
        payload = self.read_file(directory / 'request.json')
        receipt = self.read_file(directory / 'input.json')
        stored = self.read_file(directory / 'result.json')
        digest = hashlib.sha256(canonical(payload)).hexdigest()
        require(digest == evidence['request_sha256'] == receipt['request_sha256']
                and receipt['identity_sha256'] == self.identity, 'request receipt differs')
        parsed = parse_request(payload, self.policy.metadata['model'])
        require(parsed['images'] == receipt['images'], 'received images differ from their receipt')
        observation = replace(observation, episode_id=self.episode_id,
                              monotonic_ns=prediction['source_monotonic_ns'])
        request = self.policy.prepare_request(observation)
        expected = self.policy.client.request_payload(observation, context=request.context,
                                                      previous=request.previous)
        require(canonical(payload) == canonical(expected),
                'model input differs from replayed camera, state, instruction or execution feedback')
        raw, response = stored['result'], stored['response']
        require(raw['status'] == 'complete'
                and response['choices'][0]['message']['content'] == raw['content']
                and response['id'] == identifier and response['model'] == self.policy.metadata['model']
                and all(response['microduck_evidence'].get(k) == v for k, v in evidence.items()),
                'raw output or serving identity differs')
        # json.loads is intentionally strict about Markdown/text wrappers.
        plan = validate_plan(json.loads(raw['content']), navigation_contract=self.policy.contract)
        require(plan == diagnostics['plan'], 'recorded plan differs from the actual raw model output')
        self.policy.client.request_plan = lambda *args, **kwargs: {**plan, '_serving_evidence': evidence}
        chunk = self.policy.predict_prepared(request)
        np.testing.assert_array_equal(chunk.actions, np.asarray(prediction['submitted_actions'], np.float32))
        require(chunk.valid_until_tick == prediction['valid_until_tick']
                and chunk.action_interval_ticks == prediction['action_interval_ticks']
                and chunk.contract == prediction['action_contract'], 'VLM action horizon or contract differs')
        require(chunk.source_monotonic_ns not in self.chunks, 'duplicate source capture clock')
        self.chunks[chunk.source_monotonic_ns] = chunk
        self.request_ids.append(identifier)

    def observe_execution(self, before, after, intent, control):
        before = replace(before, episode_id=self.episode_id,
                         monotonic_ns=control['observation_capture_monotonic_ns'])
        after = replace(after, episode_id=self.episode_id)
        source = control['checked_plan_source_monotonic_ns']
        require(source is None or source in self.chunks, 'execution refers to an unaudited VLM plan')
        self.policy.observe_execution(before, after, intent, control, self.chunks.get(source))
        self.executed_control_ticks += 1

    def observe_failure(self, observation, failure):
        """Verify the unexecuted terminal input without inventing an action.

        Historical client timing cannot be replayed. Preserve the client error
        and service outcome separately, rather than claiming a latency/cause
        measurement from this CPU reconstruction.
        """
        require(failure['tick'] == observation.tick and failure['phase'] == 'execution',
                'failure belongs to another observation')
        matches = []
        for path in self.requests.glob('*/request.json'):
            value = read(path)
            metadata = value.get('metadata', {})
            if (metadata.get('episode_id') == self.episode_id
                    and metadata.get('source_tick') == str(observation.tick)):
                matches.append(path.parent)
        require(len(matches) == 1, 'failed source has no unique recorded HTTP request')
        directory = matches[0]
        require(not directory.is_symlink() and re.fullmatch(r'[0-9a-f]{32}', directory.name)
                and directory.name not in self.request_ids, 'invalid failed request identity')
        payload = self.read_file(directory / 'request.json')
        receipt = self.read_file(directory / 'input.json')
        require(hashlib.sha256(canonical(payload)).hexdigest() == receipt['request_sha256']
                and receipt['identity_sha256'] == self.identity, 'failed request receipt differs')
        parsed = parse_request(payload, self.policy.metadata['model'])
        require(parsed['images'] == receipt['images'], 'failed input images differ')
        observation = replace(observation, episode_id=self.episode_id)
        request = self.policy.prepare_request(observation)
        expected = self.policy.client.request_payload(observation, context=request.context,
                                                      previous=request.previous)
        require(canonical(payload) == canonical(expected), 'failed input differs from replayed observation')
        outcome = {}
        for filename in ('result.json', 'error.json'):
            if (directory / filename).exists():
                outcome[filename] = self.read_file(directory / filename)
        require(outcome, 'failed HTTP request lacks a terminal service record')
        self.failed_requests.append(dict(request_id=directory.name, source_tick=observation.tick,
                                         client_error_type=failure['error_type'], service_outcome=outcome,
                                         actions_executed=0, failure_cause_independently_verified=False))

    def finish(self):
        require(all(sha(path) == digest for path, digest in self.files.items()),
                'HTTP evidence changed during replay')
        return {
            'schema': 'microduck_offline_vlm_http_replay_v1', 'status': 'complete',
            'service_identity_sha256': self.identity, 'request_ids': self.request_ids,
            'exact_model_requests': len(self.request_ids),
            'failed_unexecuted_requests': self.failed_requests,
            'execution_feedback_ticks': self.executed_control_ticks,
            'request_files_sha256': {str(p.relative_to(self.requests)): d for p, d in self.files.items()},
            'full_received_payload_regenerated': True, 'raw_output_to_actions_verified': True,
            'model_inference_replayed': False, 'fresh_realtime_measurement': False,
            'historical_runtime_identity_verified': False,
            'formal_acceptance': False, 'hardware_validated': False,
        }

    def close(self):
        self.policy.close()


def audit_offline_vlm_episode(folder, case, requests):
    episode = read(Path(folder) / 'report.json')
    observer = OfflineVLMObserver(episode, requests, case['service_identity_sha256'])
    try:
        physics = replay_recording(folder, case, policy_observer=observer)
        http = observer.finish()
        require(http['exact_model_requests'] == physics['exact_prediction_source_observations']
                and http['execution_feedback_ticks'] == physics['exact_control_ticks'],
                'VLM and physics audit coverage differs')
        return {'schema': 'microduck_offline_vlm_physical_audit_v1', 'status': 'complete',
                'physics': physics, 'http': http, 'formal_acceptance': False,
                'hardware_validated': False, 'audit_source_sha256': sha(__file__)}
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
    result = audit_offline_vlm_episode(args.episode_folder, read(args.case), args.requests)
    require(sha(args.case) == args.case_sha256, 'case changed during replay')
    result['case_sha256'] = args.case_sha256
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'requests': result['http']['exact_model_requests'],
                      'control_ticks': result['physics']['exact_control_ticks'],
                      'strict_success': result['physics']['strict_success'], 'formal_acceptance': False}))


if __name__ == '__main__':
    main()
