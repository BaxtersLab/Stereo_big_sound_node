# Stereo_big_sound_node

A ComfyUI custom audio node for stereo enhancement intended to be used at the end of an audio chain.

## Modes

- `doubler`: time-domain widening (delay/mod/detune style)
- `spectral`: frequency-domain stereoizer using spectral masks

## Notes

- Accepts mono or stereo input; always outputs stereo.
- Pure-Python DSP (uses NumPy; Torch optional when provided by ComfyUI).
- This repo intentionally does not ship or create virtual environments.
