"""Diagnostic capture at the real VAE and proprioception-encoder boundaries.

This synchronizes and copies tensors. Run separately from latency experiments;
an input match is neither task acceptance nor a new realtime measurement.
"""
from contextlib import ExitStack
import hashlib

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def expected_state_tensor(state, stats):
    """Independently apply the pinned global min/max proprio48 convention."""
    import torch

    row = stats['state']['default']
    low = torch.as_tensor(row['global_min'], dtype=torch.float32).clone()
    high = torch.as_tensor(row['global_max'], dtype=torch.float32).clone()
    require(low.shape == high.shape == (48,) and torch.isfinite(low).all()
            and torch.isfinite(high).all() and (high >= low).all(), 'invalid state normalizer')
    width = high-low
    constant = width < 1e-4
    width[constant] = 2.
    scale = 2./width
    offset = -1.-scale*low
    offset[constant] = -low[constant]
    return (torch.from_numpy(state.copy())*scale+offset).clamp(-5., 5.)


def _replace(stack, owner, name, wrapper):
    """Restore both instance attributes and class-provided bound methods."""
    existed = name in vars(owner)
    old_attribute = vars(owner).get(name)
    original = getattr(owner, name)
    setattr(owner, name, wrapper(original))

    def restore():
        if existed:
            setattr(owner, name, old_attribute)
        else:
            delattr(owner, name)
    stack.callback(restore)


def capture_prediction(core, backend, rgb, state, instruction, stats, *, seed=0, require_cuda=True):
    """Observe actual encoder inputs while preserving the entire prediction call.

Reference captures the upstream VAE entry and its nn.Module proprio encoder;
native captures VAE.encode and the native proprio weight's apply method.
No preprocessing method is replaced and no tensor is changed by the hooks.
"""
    import torch

    rgb, state = np.asarray(rgb), np.asarray(state)
    require(backend in ('reference', 'native'), 'unsupported backend')
    require(rgb.dtype == np.uint8 and rgb.shape == (core.camera_size, core.camera_size, 3),
            'wrong RGB contract')
    require(state.dtype == np.float32 and state.shape == (48,) and np.isfinite(state).all(),
            'wrong proprio48 contract')
    require(isinstance(instruction, str) and instruction, 'instruction required')
    expected_image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1.)
    expected_state = expected_state_tensor(state, stats)
    before = hashlib.sha256(rgb.tobytes()+state.tobytes()+instruction.encode()).hexdigest()
    captures = {}
    metadata = {}

    def capture(name, tensor, shape, expected):
        require(name not in captures, 'encoder input was captured more than once: '+name)
        require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape,
                'unexpected encoder input shape: '+name)
        require(tensor.dtype == torch.bfloat16, 'diagnostic requires the deployed BF16 input: '+name)
        require(not require_cuda or tensor.device.type == 'cuda', 'encoder input is not on CUDA: '+name)
        actual = tensor.detach().to(device='cpu').clone().reshape(expected.shape)
        target = expected.to(dtype=tensor.dtype)
        require(torch.equal(actual, target), 'normalized encoder input differs: '+name)
        value = actual.float().numpy()
        captures[name] = value
        metadata[name] = dict(shape=list(tensor.shape), canonical_shape=list(value.shape),
            dtype=str(tensor.dtype), device=str(tensor.device),
            canonical_float32_sha256=hashlib.sha256(value.tobytes()).hexdigest(), exact_expected_match=True)

    with ExitStack() as stack:
        if backend == 'reference':
            def image_wrapper(original):
                def call(*args, **kwargs):
                    image = kwargs['input_image'] if 'input_image' in kwargs else args[0]
                    capture('image', image, (1, 3, core.camera_size, core.camera_size), expected_image)
                    return original(*args, **kwargs)
                return call
            _replace(stack, core.model, '_encode_input_image_latents_tensor', image_wrapper)

            def state_hook(module, args):
                require(len(args) == 1, 'unexpected reference proprio encoder call')
                # Upstream inserts a token axis immediately before its encoder.
                capture('state', args[0], (1, 1, 48), expected_state)
            handle = core.model.proprio_encoder.register_forward_pre_hook(state_hook)
            stack.callback(handle.remove)
        else:
            def image_wrapper(original):
                def call(*args, **kwargs):
                    require(len(args) >= 1, 'missing native VAE input')
                    capture('image', args[0], (1, 3, 1, core.camera_size, core.camera_size), expected_image)
                    return original(*args, **kwargs)
                return call
            _replace(stack, core.vae, 'encode', image_wrapper)

            def state_wrapper(original):
                def call(*args, **kwargs):
                    require(len(args) >= 1, 'missing native proprio encoder input')
                    capture('state', args[0], (1, 48), expected_state)
                    return original(*args, **kwargs)
                return call
            _replace(stack, core.model.pre_weight.proprio_encoder, 'apply', state_wrapper)
        actions = np.asarray(core.predict_action_chunk(
            {'head_camera': rgb}, state, instruction, seed=seed), dtype=np.float32)
    require(set(captures) == {'image', 'state'}, 'prediction did not visit both encoder boundaries')
    require(actions.ndim == 2 and actions.shape[1] == 5 and len(actions) > 0
            and np.isfinite(actions).all(), 'invalid prediction result')
    require(hashlib.sha256(rgb.tobytes()+state.tobytes()+instruction.encode()).hexdigest() == before,
            'prediction mutated raw arguments')
    return dict(schema='microduck_network_input_capture_v1', backend=backend,
        tensors=metadata, raw_arguments_sha256=before, action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        prediction_returned=True, normalized_encoder_inputs_match=True,
        all_inputs_on_cuda=all(v['device'].startswith('cuda') for v in metadata.values()),
        diagnostic_changes_timing=True, fresh_realtime_measurement=False, formal_acceptance=False), captures, actions
