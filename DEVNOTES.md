# DEVNOTES — Stereo Big Sound Node
> Carry this file into every future session. It is the handoff document.

---

## Current Status: BUG FIXED — AWAITING FIRST TEST

Session 2 fixed a mono-input crash and cleaned up housekeeping. DSP is solid.

---

## What This Node Does

Stereo enhancer intended as the **last node in an audio chain**. Two modes:

### Doubler (time-domain)
- Delays the signal slightly on L/R (quadrature LFO for movement)
- Applies small pitch detune per channel (opposite cents on L vs R)
- `mono_safe=True`: processes side channel only → mono sum is preserved

### Spectral (frequency-domain)
- STFT overlap-add: splits spectral bins into L and R via a mask pattern
- Modes: `even_odd` (alternating bins), `random` (seed=0, deterministic), `band_split_2` (low/high), `band_split_4` (four bands interleaved)
- `mono_safe=True`: uses split as side around original mono mid
- `energy_match=True`: normalises output RMS to match input

### Shared post-processing
1. **Width control** — scales side signal (0=mono, 1=unchanged, 2=double-wide)
2. **Dry/wet blend**
3. **Soft clip** — `tanh(x × 1.5) / tanh(1.5)` for analog-style saturation; transparent at low levels, limiting above ±1

Always outputs **stereo** regardless of input channel count.

---

## What Changed (Session 2 — May 2026)

| File | Change |
|---|---|
| `Stereo_big_sound_node.py` | **Fixed mono input crash** — `out = np.zeros_like(wf_bct)` allocated the output array with the same channel count as the input. For mono input `(B,1,T)`, the subsequent `out[b, 0:2, :] = mixed` tried to write a `(2,T)` array into a `(1,T)` slot, raising a NumPy shape mismatch. Fixed to `np.zeros((B, 2, T))`. Removed redundant `[:, 0:2, :]` slice on final pack. |
| `__init__.py` | Added `BigSound_` key prefix, readable display name. |
| `requirements.txt` | Created. |
| `DEVNOTES.md` | Created this file. |

---

## DSP Notes

**Doubler modulation:** L and R use 90° phase-offset LFOs (`phi_r = π/2`) so the stereo image sweeps in a circular motion rather than pumping in and out.

**Pitch detune:** `f = 2^(cents/1200)`. L is resampled at rate `f` (up), R at `1/f` (down). This creates a classic "double-tracked" widening.

**STFT stereoizer:** Overlap-add with Hann window and 75% overlap (hop = fft_size/4). Window correction divides by `win²` accumulated per sample. The spectral mask is applied in the frequency domain before IFFT, so each channel contains complementary frequency content. The `random` mask uses `rng_seed=0` — always deterministic.

**Soft clip:** `tanh(x × d) / tanh(d)` where `d=1.5`. At x=1.0, output=1.0. At x=0.1, output≈0.166 (slight boost for "big" character). Clips smoothly above ±1.

---

## What Still Needs Doing

### Priority 1 — Test

- [ ] **Test mono input** — connect a mono AUDIO, confirm output is stereo without crash (this was the bug fixed this session)
- [ ] **Test stereo input** — both modes, all mask_mode options
- [ ] **Test dry_wet=0** — should pass audio through with only soft-clip coloring
- [ ] **Test width_amount=0** — should produce mono output (side=0)
- [ ] **Test width_amount=2** — should produce extra-wide stereo

### Priority 2 — Improvements

- [ ] **Bypass soft clip when dry_wet=0** — currently the tanh soft clip is always applied, even when dry_wet=0 (pure dry). This adds subtle saturation coloring to the dry pass-through, which may be unexpected. Consider applying soft clip only when dry_wet > 0.
- [ ] **Expose soft clip drive as widget** — currently hardcoded at 1.5. A slider (0.5–3.0) would give control over saturation character.
- [ ] **Add `load_preset` / `save_preset`** — no preset management, unlike the Flanger node. Useful for saving favourite width/mode combinations.

### Priority 3 — Polish

- [ ] **Spectral random mask per-run option** — currently `rng_seed=0` so random is always the same. Add an optional `random_seed` INT widget for varied stereo images.
- [ ] **Update display name in README** — README title still uses the old format.

---

## Known Risks / Watch Out For

- **Use LAST in chain** (as labelled in GUI lane 15) — the soft clip and stereo widening are mastering-style effects. Applying them mid-chain will colour downstream processing.
- **Spectral mode + large FFT** — `fft_size=8192` with long audio runs many STFT frames. Expect 1–3s processing time for 30s audio.
- **Doubler pitch detune** — `_linear_resample_same_length` is a simple linear-interpolation pitch shifter. It introduces aliasing above ~8kHz at large `pitch_cents` values. For subtle widening (±5–15 cents) this is inaudible.
- **Key renamed** — `Stereo_big_sound_node` → `BigSound_StereoEnhancer`. Update any existing workflow JSON references.

---

## Node Key Names

| Key (in workflow JSON `type` field) | Display name |
|---|---|
| `BigSound_StereoEnhancer` | Stereo Big Sound Enhancer |

---

## Session Log

| Date | What happened |
|---|---|
| ~4 months ago | Initial build. Well-implemented DSP — two modes, MS processing, STFT stereoizer. Mono input crash bug present. |
| 2026-05-18 | Fixed mono input crash (output always allocated as stereo). Key prefix added. |
