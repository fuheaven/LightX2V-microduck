"""Optional fixed-shape CUDA capture of the unchanged action flow loop.

Every new observation replaces all context/video/state buffers before replay.
Noise and timesteps are created by the normal scheduler outside capture. The
caller must warm up on its main thread before starting concurrent simulation.
"""

import threading

import torch


def clone_inputs(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_inputs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_inputs(item) for item in value)
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"unsupported graph input {type(value).__name__}")


def copy_inputs(destination, source):
    """Reject changed contracts instead of replaying stale or misbound buffers."""
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor) or (destination.shape, destination.dtype, destination.device) != (source.shape, source.dtype, source.device):
            raise ValueError("CUDA graph tensor shape, dtype or device changed; create a new policy")
        destination.copy_(source)
    elif isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise ValueError("CUDA graph condition fields changed; create a new policy")
        for key in destination:
            copy_inputs(destination[key], source[key])
    elif isinstance(destination, (list, tuple)):
        if type(source) is not type(destination) or len(destination) != len(source):
            raise ValueError("CUDA graph condition structure changed")
        for target, value in zip(destination, source):
            copy_inputs(target, value)
    elif type(destination) is not type(source) or destination != source:
        raise ValueError("CUDA graph static parameter changed")


class FastWAMActionGraph:
    def __init__(self, model, inputs, noise, timesteps, deltas):
        if noise.device.type != "cuda":
            raise ValueError("FastWAM action graphs require CUDA")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("warm up FastWAM CUDA graphs on the main thread before starting simulation workers")
        self.inputs = clone_inputs(inputs)
        self.noise, self.timesteps, self.deltas = noise.clone(), timesteps.clone(), deltas.clone()
        # The original implementation materializes this CPU constant inside
        # each step. Transfer it before capture; its numerical values stay fixed.
        model.pre_infer.action_freqs = model.pre_infer.action_freqs.to(noise.device)

        def denoise():
            latents = self.noise
            for step in range(len(self.timesteps)):
                prediction = model.predict_action_noise(self.inputs, latents, self.timesteps[step])
                # Preserve exactly the native scheduler's BF16 operation order.
                latents = latents + prediction * self.deltas[step]
            return latents

        stream = torch.cuda.Stream(device=noise.device)
        stream.wait_stream(torch.cuda.current_stream(noise.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                denoise()
        torch.cuda.current_stream(noise.device).wait_stream(stream)
        torch.cuda.synchronize(noise.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = denoise()

    def replay(self, inputs, noise, timesteps, deltas):
        copy_inputs(self.inputs, inputs)
        copy_inputs(self.noise, noise)
        copy_inputs(self.timesteps, timesteps)
        copy_inputs(self.deltas, deltas)
        self.graph.replay()
        return self.output
