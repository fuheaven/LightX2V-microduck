"""Build the unchanged navigation model inside its owning spawned process."""
from pathlib import Path


def build_navigation_core(*, backend, model_config, checkpoint, backbone, stats,
                          history_contract, navigation_contract, normalization_binding,
                          inference_steps, compile_action, cache_prompt,
                          cache_action_context, action_cuda_graph, action_rms_one_pass):
    import torch
    from .wam_policy import artifact_sha256, native_navigation_config

    history = None
    if history_contract is not None:
        from .navigation_history import history_spec_from_contract
        history = history_spec_from_contract(history_contract)
    if backend == 'reference':
        if cache_action_context or action_cuda_graph or action_rms_one_pass:
            raise ValueError('native optimizations cannot be applied to the reference model')
        import inspect
        import fastwam
        from .wam_reference import ReferenceNavigationCore

        core = ReferenceNavigationCore.from_checkpoint(
            model_config, checkpoint, stats, compile_action_infer=compile_action,
            cache_prompt=cache_prompt, inference_steps=inference_steps,
            history_spec=history, navigation_contract=navigation_contract,
            normalization_binding=normalization_binding)
        reference_root = Path(fastwam.__file__).resolve().parent
        core.reference_provenance = {
            'source_sha256': {str(p.relative_to(reference_root.parent)): artifact_sha256(p)
                              for p in sorted(reference_root.rglob('*.py'))},
            'model_source': str(Path(inspect.getfile(type(core.model))).resolve().relative_to(reference_root.parent)),
            'adapter_sha256': artifact_sha256(Path(inspect.getfile(ReferenceNavigationCore))),
        }
    elif backend == 'native':
        if compile_action:
            raise ValueError('native compilation is not implemented')
        import lightx2v_platform.set_ai_device  # noqa: F401
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy

        config = native_navigation_config(backbone, checkpoint, stats, history_spec=history,
            contract=navigation_contract, normalization_binding=normalization_binding)
        config.update(checkpoint_mmap=True, action_infer_steps=inference_steps,
                      fastwam_cache_prompt=cache_prompt,
                      fastwam_cache_action_context=cache_action_context,
                      fastwam_action_cuda_graph=action_cuda_graph,
                      fastwam_action_rms_one_pass=action_rms_one_pass)
        core = FastWAMPolicy.from_config(config)
        core.resolved_config = config
    else:
        raise ValueError('unknown navigation model backend')
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    core.runtime_provenance = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        gpu_name=device.name, gpu_uuid=str(getattr(device, 'uuid', '')))
    core.runtime_metrics = lambda: dict(peak_memory_bytes=torch.cuda.max_memory_allocated())
    return core
