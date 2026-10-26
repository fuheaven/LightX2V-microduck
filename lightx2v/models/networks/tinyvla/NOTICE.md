TinyVLA implementation attribution

The model, flow expert, Qwen semantic path and embodiment modules are adapted
from AlexWortega/tinyvla, tinyvla_b200/src/tinyvla, revision
603f5045b45126af011418befc899e3c67d04346:
https://huggingface.co/AlexWortega/tinyvla/tree/603f5045b45126af011418befc899e3c67d04346/tinyvla_b200/src/tinyvla

The source repository model card declares Apache-2.0. The Apache-2.0 license
provided at this repository's LICENSE applies to these adaptations. Qwen
backbone artifacts, model weights, demonstration data and robot mesh assets
remain separately sourced artifacts; no robot mesh or private checkpoint is
included here.

Changes: remove LeRobot base/config runtime dependency; expose deterministic
flow noise/time, explicit precision/vision compilation selection, local
checkpoint loading/saving, artifact verification, normalization, session
lifecycle and standalone model serving. Upstream state_dict names are retained.

Only the public 50-step architecture is implemented. This is not the private
Stage 2.5 model and does not imply reproduction of its thirteen demos.

The unified LM path explicitly supplies the sequential 1D positional indices
used by Transformers 5.5.4 when the upstream implementation runs on a fresh
model with inputs_embeds only. This prevents stale rope_deltas from affecting
later sessions without changing to a different M-RoPE training convention.
The image processor's JSON constants are read directly; the native patchifier
is unchanged and does not require torchvision's fast processor class.

Partial language-layer unfreezing is explicitly serialized in configuration.
Vision fine-tuning uses the HF tower to preserve positional-embedding gradients;
the author's detached fixed-grid cache is retained for inference only.
