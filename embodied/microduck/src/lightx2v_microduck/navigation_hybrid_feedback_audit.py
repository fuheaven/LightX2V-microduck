"""Replay hybrid tool corrections from original HTTP receipts and real physics.

The policy regenerates each correction from the rejected response. Neither a
supervisor nor a learned navigation model is called by this audit. Historical
transport failures are preserved, not claimed as remeasured network behavior.
"""

from dataclasses import replace
import hashlib
import urllib.error

from .navigation_hybrid_audit import OfflineHybridObserver
from .navigation_outcomes import read, require
from .planner_transport_evidence import raise_recorded_transport_error, validate_recorded_transport_error
from .vlm_service import canonical


class HybridFeedbackEvidenceMismatch(BaseException):
    """An audit fault must never masquerade as the expected policy exception."""


class FeedbackHybridObserver(OfflineHybridObserver):
    def __init__(self, episode, requests, expected_identity):
        super().__init__(episode, requests, expected_identity)
        require(self.validation_feedback, 'correction audit requires an enabled policy')
        self.active_trace, self.trace_cursor, self.active_failure = None, 0, None
        self.validated_decisions = self.corrected_decisions = self.rejected_calls = 0
        self.rejections = []

    def begin_trace(self, trace, failure=None):
        require(isinstance(trace, list) and 1 <= len(trace) <= 2,
                'hybrid decision lacks its bounded HTTP attempt trace')
        self.active_trace, self.trace_cursor, self.active_failure = trace, 0, failure

    def recorded_raw_reply(self, payload):
        try:
            return self._recorded_raw_reply(payload)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            raise
        except Exception as error:
            raise HybridFeedbackEvidenceMismatch(str(error)) from error

    def _recorded_raw_reply(self, payload):
        require(self.active_trace is not None and self.trace_cursor < len(self.active_trace),
                'replayed hybrid decision has extra HTTP attempts')
        row = self.active_trace[self.trace_cursor]
        digest = hashlib.sha256(canonical(payload)).hexdigest()
        require(row['attempt'] == self.trace_cursor and row['request_sha256'] == digest
                and row['source_tick'] == payload['metadata']['source_tick']
                and row['camera_tick'] == payload['metadata']['camera_tick']
                and row['action_executed'] is False,
                'hybrid attempt differs from regenerated observation or rejection feedback')
        evidence = row.get('serving_evidence') or {}
        identifier = evidence.get('request_id') or row.get('transport_error', {}).get('request_id')
        if identifier is None:
            # A transport exception may occur before the client receives an ID.
            # Never guess among duplicate requests (for example warmup calls).
            matches = []
            for path in self.requests.glob('*/request.json'):
                if path.parent.name not in self.request_ids and canonical(read(path)) == canonical(payload):
                    matches.append(path.parent.name)
            require(len(matches) == 1, 'failed hybrid attempt lacks a unique original request')
            identifier = matches[0]
        directory, verified_digest = self.request_record(identifier, payload)
        require(verified_digest == digest, 'hybrid attempt digest differs')
        terminal = {name: self.read_file(directory / name) for name in ('result.json', 'error.json')
                    if (directory / name).is_file()}
        require(terminal, 'hybrid attempt lacks a terminal HTTP receipt')
        stored = terminal.get('result.json', {})
        raw, response = stored.get('result', {}), stored.get('response', {})
        expected = dict(request_id=identifier, request_sha256=digest,
                        identity_sha256=self.identity, model_id=self.policy.metadata['supervisor_model'])
        if stored:
            require(response.get('id') == identifier and response.get('model') == expected['model_id']
                    and response.get('choices', [{}])[0].get('message', {}).get('content') == raw.get('content')
                    and all(response.get('microduck_evidence', {}).get(k) == v for k, v in expected.items()),
                    'hybrid raw response or serving identity differs')
        self.request_ids.append(identifier)
        self.trace_cursor += 1
        if row['status'] == 'request_started':
            require(self.active_failure is not None, 'successful decision contains a transport failure')
            kind = self.active_failure['error_type']
            transport = row.get('transport_error')
            if transport is not None:
                validate_recorded_transport_error(transport, payload, identifier, self.identity, terminal)
                require(transport['exception_type'] == kind, 'transport type differs from terminal failure')
                raise_recorded_transport_error(transport)
            if kind == 'HTTPError':
                raise urllib.error.HTTPError('recorded', 504, 'recorded client failure', {}, None)
            if kind == 'URLError':
                raise urllib.error.URLError('recorded client failure')
            if kind == 'TimeoutError':
                raise TimeoutError('recorded client failure')
            raise ValueError('unsupported historical hybrid transport failure')
        require(raw.get('status') == 'complete' and raw['content'] == row['raw_content'],
                'hybrid tool trace differs from the completed raw response')
        return raw['content'], expected

    def complete_trace(self):
        require(self.trace_cursor == len(self.active_trace), 'recorded hybrid HTTP attempts were omitted')
        for row in self.active_trace:
            if row['status'] == 'tool_rejected':
                self.rejected_calls += 1
                self.rejections.append(row['rejection'])

    def observe_prediction(self, observation, prediction):
        original = prediction['diagnostics']['hybrid']
        requested = original['supervisor_requested_this_prediction']
        if requested:
            self.begin_trace(original.get('request_trace'))
        else:
            self.active_trace = None
        super().observe_prediction(observation, prediction)
        if requested:
            # The inherited comparison includes the full reconstructed trace.
            self.complete_trace()
            self.validated_decisions += 1
            self.corrected_decisions += int(len(self.active_trace) == 2)
            self.selected_tools[original['call']['tool']] += 1

    def observe_failure(self, observation, failure):
        require(failure['tick'] == observation.tick and failure['phase'] == 'execution',
                'terminal hybrid failure belongs to another observation')
        observation = replace(observation, episode_id=self.episode_id)
        request = self.policy.prepare_request(observation)
        require(self.policy.decision is None or observation.tick >= self.policy.decision['until_tick'],
                'delegate-only failure requires separate learned model error evidence')
        self.begin_trace(failure.get('planner_request_trace'), failure)
        start = len(self.request_ids)
        try:
            self.policy.predict_prepared(request)
        except Exception as error:
            require(type(error).__name__ == failure['error_type']
                    and canonical(getattr(error, 'planner_request_trace', None)) == canonical(self.active_trace),
                    'terminal hybrid error or full attempt trace did not recur')
        else:
            raise ValueError('original hybrid failure was not reproduced')
        self.complete_trace()
        for identifier in self.request_ids[start:]:
            self.failed_requests.append(dict(request_id=identifier, source_tick=observation.tick,
                client_error_type=failure['error_type'], actions_executed=0,
                failure_cause_independently_verified=failure['error_type'] in ('ValueError', 'JSONDecodeError')))

    def finish(self):
        result = super().finish()
        result.update(exact_http_attempts=len(self.request_ids),
                      validated_supervisor_decisions=self.validated_decisions,
                      corrected_supervisor_decisions=self.corrected_decisions,
                      rejected_tool_calls=self.rejected_calls,
                      rejected_tool_records=self.rejections,
                      correction_requests_regenerated=True)
        return result
