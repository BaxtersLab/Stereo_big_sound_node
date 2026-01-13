# Stereo_big_sound_node

A ComfyUI custom audio node for stereo enhancement intended to be used at the end of an audio chain.

## Modes

- `doubler`: time-domain widening (delay/mod/detune style)
- `spectral`: frequency-domain stereoizer using spectral masks

## Notes

- Accepts mono or stereo input; always outputs stereo.
- Pure-Python DSP (uses NumPy; Torch optional when provided by ComfyUI).
- This repo intentionally does not ship or create virtual environments.

## GUI (Fixed Lane Order)

This node intentionally uses a fixed 16-lane layout.

- Lane 0: `SHARED CONTROLS` (text-only)
- Lane 1: `width_amount`
- Lane 2: `dry_wet`
- Lane 3: `mode` (`doubler` / `spectral`)
- Lane 4: `DOUBLER CONTROLS` (text-only)
- Lane 5: `delay_ms`
- Lane 6: `pitch_cents`
- Lane 7: `mod_rate_hz`
- Lane 8: `mod_depth`
- Lane 9: `mono_safe_doubler`
- Lane 10: `SPECTRAL CONTROLS` (text-only)
- Lane 11: `mask_mode`
- Lane 12: `fft_size`
- Lane 13: `mono_safe_spectral`
- Lane 14: `energy_match`
- Lane 15: `USE THIS LAST IN CHAIN!` (text-only)
