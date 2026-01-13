import math
from typing import Any, Optional, Tuple

import numpy as np

try:
	import torch
except Exception:  # torch is expected in ComfyUI, but keep import-safe
	torch = None


def _clamp(x: float, lo: float, hi: float) -> float:
	return float(min(max(float(x), float(lo)), float(hi)))


def _softclip_tanh(x: np.ndarray, drive: float = 1.5) -> np.ndarray:
	d = float(max(drive, 1e-6))
	return np.tanh(x * d) / math.tanh(d)


def _ensure_bct(waveform_np: np.ndarray) -> np.ndarray:
	# Normalize to (batch, channels, samples)
	wf = np.asarray(waveform_np, dtype=np.float32)
	if wf.ndim == 1:
		wf = wf[None, None, :]
	elif wf.ndim == 2:
		# Prefer (channels, samples)
		if wf.shape[0] <= 8 and wf.shape[1] > wf.shape[0]:
			wf = wf[None, :, :]
		else:
			wf = wf.T[None, :, :]
	elif wf.ndim == 3:
		if not (wf.shape[1] <= 8 and wf.shape[2] > wf.shape[1]):
			wf = np.transpose(wf, (0, 2, 1))
	else:
		wf = wf.reshape(wf.shape[0], -1, wf.shape[-1])
	return wf.astype(np.float32, copy=False)


def _mid_side(stereo_ct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	# stereo_ct: (2, T)
	l = stereo_ct[0]
	r = stereo_ct[1]
	mid = 0.5 * (l + r)
	side = 0.5 * (l - r)
	return mid, side


def _from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
	l = mid + side
	r = mid - side
	return np.stack([l, r], axis=0)


def _linear_resample_same_length(x: np.ndarray, rate: float) -> np.ndarray:
	# Very small detune helper; keeps length identical.
	# rate>1 shifts pitch up (reads faster); rate<1 shifts pitch down.
	n = int(x.shape[0])
	if n <= 1:
		return x.astype(np.float32, copy=False)
	rate = float(max(rate, 1e-6))
	pos = np.arange(n, dtype=np.float64) * rate
	pos = np.clip(pos, 0.0, float(n - 1))
	i0 = np.floor(pos).astype(np.int64)
	i1 = np.minimum(i0 + 1, n - 1)
	frac = (pos - i0).astype(np.float32)
	return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32, copy=False)


def _fractional_delay_variable(x: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
	# x shape (T,)
	# delay_samples shape (T,) in samples, >=0
	n = int(x.shape[0])
	if n <= 1:
		return x.astype(np.float32, copy=False)
	idx = np.arange(n, dtype=np.float64) - delay_samples.astype(np.float64)
	idx = np.clip(idx, 0.0, float(n - 1))
	i0 = np.floor(idx).astype(np.int64)
	i1 = np.minimum(i0 + 1, n - 1)
	frac = (idx - i0).astype(np.float32)
	return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32, copy=False)


def _stft_stereoizer(
	x_mono: np.ndarray,
	fs: int,
	fft_size: int,
	mask_mode: str,
	rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
	# Returns (L, R) signals, each shape (T,)
	n = int(x_mono.shape[0])
	fft_size = int(fft_size)
	fft_size = max(256, fft_size)
	hop = fft_size // 4
	win = np.hanning(fft_size).astype(np.float32)

	# Pad to fit frames
	pad = fft_size
	x = np.pad(x_mono.astype(np.float32, copy=False), (pad, pad), mode="constant")
	out_len = x.shape[0]

	n_frames = 1 + (out_len - fft_size) // hop

	rng = np.random.default_rng(int(rng_seed))
	bins = fft_size // 2 + 1
	if mask_mode == "even_odd":
		mask_l = np.zeros(bins, dtype=np.float32)
		mask_l[::2] = 1.0
	elif mask_mode == "random":
		mask_l = (rng.random(bins) > 0.5).astype(np.float32)
	elif mask_mode == "band_split_2":
		mask_l = np.zeros(bins, dtype=np.float32)
		mask_l[: bins // 2] = 1.0
	elif mask_mode == "band_split_4":
		mask_l = np.zeros(bins, dtype=np.float32)
		q = bins // 4
		mask_l[:q] = 1.0
		mask_l[2 * q : 3 * q] = 1.0
	else:
		# default
		mask_l = np.zeros(bins, dtype=np.float32)
		mask_l[::2] = 1.0

	mask_r = 1.0 - mask_l

	y_l = np.zeros(out_len, dtype=np.float32)
	y_r = np.zeros(out_len, dtype=np.float32)
	norm = np.zeros(out_len, dtype=np.float32)

	for i in range(int(n_frames)):
		start = i * hop
		frame = x[start : start + fft_size] * win
		spec = np.fft.rfft(frame)
		spec_l = spec * mask_l
		spec_r = spec * mask_r
		frame_l = np.fft.irfft(spec_l, n=fft_size).astype(np.float32)
		frame_r = np.fft.irfft(spec_r, n=fft_size).astype(np.float32)
		y_l[start : start + fft_size] += frame_l * win
		y_r[start : start + fft_size] += frame_r * win
		norm[start : start + fft_size] += win * win

	norm = np.maximum(norm, 1e-8)
	y_l = y_l / norm
	y_r = y_r / norm

	# Remove padding
	y_l = y_l[pad : pad + n]
	y_r = y_r[pad : pad + n]
	return y_l.astype(np.float32, copy=False), y_r.astype(np.float32, copy=False)


class Stereo_big_sound_node:
	RETURN_TYPES = ("AUDIO",)
	RETURN_NAMES = ("audio",)
	FUNCTION = "process"
	CATEGORY = "audio/processing"

	@classmethod
	def INPUT_TYPES(cls):
		ro = {"multiline": False, "readonly": True, "display": "text"}
		return {
			"required": {
				"audio": ("AUDIO",),

				# Lane 0 — Text Only
				"lane0": ("STRING", {"default": "SHARED CONTROLS", **ro}),

				# Lane 1 — Control
				"width_amount": (
					"FLOAT",
					{"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"},
				),

				# Lane 2 — Control
				"dry_wet": (
					"FLOAT",
					{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
				),

				# Lane 3 — Control
				"mode": (["doubler", "spectral"], {"default": "doubler"}),

				# Lane 4 — Text Only
				"lane4": ("STRING", {"default": "DOUBLER CONTROLS", **ro}),

				# Lane 5 — Control
				"delay_ms": (
					"FLOAT",
					{"default": 12.0, "min": 0.0, "max": 40.0, "step": 0.1, "display": "slider"},
				),

				# Lane 6 — Control
				"pitch_cents": (
					"FLOAT",
					{"default": 8.0, "min": -50.0, "max": 50.0, "step": 0.5, "display": "slider"},
				),

				# Lane 7 — Control
				"mod_rate_hz": (
					"FLOAT",
					{"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.01, "display": "slider"},
				),

				# Lane 8 — Control
				"mod_depth": (
					"FLOAT",
					{"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"},
				),

				# Lane 9 — Control
				"mono_safe_doubler": ("BOOLEAN", {"default": True}),

				# Lane 10 — Text Only
				"lane10": ("STRING", {"default": "SPECTRAL CONTROLS", **ro}),

				# Lane 11 — Control
				"mask_mode": (
					["even_odd", "random", "band_split_2", "band_split_4"],
					{"default": "even_odd"},
				),

				# Lane 12 — Control
				"fft_size": ([1024, 2048, 4096, 8192], {"default": 2048}),

				# Lane 13 — Control
				"mono_safe_spectral": ("BOOLEAN", {"default": True}),

				# Lane 14 — Control
				"energy_match": ("BOOLEAN", {"default": True}),

				# Lane 15 — Text Only
				"lane15": ("STRING", {"default": "USE THIS LAST IN CHAIN!", **ro}),
			}
		}

	def _extract_audio(self, audio: Any) -> Tuple[np.ndarray, int, Optional[dict], Optional[Any]]:
		audio_dict = audio if isinstance(audio, dict) else None
		waveform = audio_dict.get("waveform") if audio_dict is not None else audio
		sample_rate = int(audio_dict.get("sample_rate", 44100)) if audio_dict is not None else 44100

		torch_device = None
		if torch is not None and isinstance(waveform, torch.Tensor):
			try:
				torch_device = waveform.device
			except Exception:
				torch_device = None
			wf = waveform.detach()
			if not wf.is_floating_point():
				wf = wf.float()
			wf_np = wf.cpu().numpy()
		else:
			wf_np = np.asarray(waveform, dtype=np.float32)

		wf_bct = _ensure_bct(wf_np)
		return wf_bct, sample_rate, audio_dict, torch_device

	def _pack_audio(self, wf_bct: np.ndarray, sample_rate: int, audio_dict: Optional[dict], torch_device: Optional[Any]):
		if torch is not None:
			wf_t = torch.from_numpy(np.asarray(wf_bct, dtype=np.float32))
			if torch_device is not None:
				try:
					wf_t = wf_t.to(torch_device)
				except Exception:
					pass
			return {"waveform": wf_t, "sample_rate": int(sample_rate)}
		return {"waveform": np.asarray(wf_bct, dtype=np.float32), "sample_rate": int(sample_rate)}

	def process(
		self,
		audio,
		lane0,
		width_amount,
		dry_wet,
		mode,
		lane4,
		delay_ms,
		pitch_cents,
		mod_rate_hz,
		mod_depth,
		mono_safe_doubler,
		lane10,
		mask_mode,
		fft_size,
		mono_safe_spectral,
		energy_match,
		lane15,
	):
		wf_bct, fs, audio_dict, torch_device = self._extract_audio(audio)

		width_amount = _clamp(float(width_amount), 0.0, 2.0)
		dry_wet = _clamp(float(dry_wet), 0.0, 1.0)

		out = np.zeros_like(wf_bct, dtype=np.float32)
		for b in range(int(wf_bct.shape[0])):
			ct = wf_bct[b]
			# Ensure stereo
			if int(ct.shape[0]) == 1:
				st = np.repeat(ct, 2, axis=0)
			else:
				st = ct[:2]

			dry_st = st.astype(np.float32, copy=False)

			if str(mode) == "spectral":
				proc_st = self._process_spectral(
					dry_st,
					fs=fs,
					mask_mode=str(mask_mode),
					fft_size=int(fft_size),
					mono_safe=bool(mono_safe_spectral),
					energy_match=bool(energy_match),
				)
			else:
				proc_st = self._process_doubler(
					dry_st,
					fs=fs,
					delay_ms=float(delay_ms),
					pitch_cents=float(pitch_cents),
					mod_rate_hz=float(mod_rate_hz),
					mod_depth_ms=float(mod_depth),
					mono_safe=bool(mono_safe_doubler),
				)

			# Shared width control applied to processed branch
			mid, side = _mid_side(proc_st)
			side = side * width_amount
			proc_st = _from_mid_side(mid, side)

			# Shared dry/wet
			mixed = (1.0 - dry_wet) * dry_st + dry_wet * proc_st
			mixed = _softclip_tanh(mixed, drive=1.5)

			out[b, 0:2, :] = mixed

		# Always return stereo (B,2,T)
		audio_out = self._pack_audio(out[:, 0:2, :], fs, audio_dict, torch_device)
		return (audio_out,)

	def _process_doubler(
		self,
		stereo_ct: np.ndarray,
		fs: int,
		delay_ms: float,
		pitch_cents: float,
		mod_rate_hz: float,
		mod_depth_ms: float,
		mono_safe: bool,
	) -> np.ndarray:
		delay_ms = _clamp(delay_ms, 0.0, 40.0)
		mod_rate_hz = _clamp(mod_rate_hz, 0.0, 5.0)
		mod_depth_ms = _clamp(mod_depth_ms, 0.0, 10.0)
		pitch_cents = _clamp(pitch_cents, -50.0, 50.0)

		t = stereo_ct.shape[1]
		n = int(t)
		ts = (np.arange(n, dtype=np.float32) / float(fs)).astype(np.float32)

		base_delay_samp = float(delay_ms) * float(fs) / 1000.0
		depth_samp = float(mod_depth_ms) * float(fs) / 1000.0

		if mod_rate_hz <= 0.0 or depth_samp <= 0.0:
			delays_l = np.full(n, base_delay_samp, dtype=np.float32)
			delays_r = np.full(n, base_delay_samp, dtype=np.float32)
		else:
			phi_l = 0.0
			phi_r = math.pi / 2.0
			delays_l = (base_delay_samp + depth_samp * np.sin(2.0 * math.pi * mod_rate_hz * ts + phi_l)).astype(np.float32)
			delays_r = (base_delay_samp + depth_samp * np.sin(2.0 * math.pi * mod_rate_hz * ts + phi_r)).astype(np.float32)
			delays_l = np.maximum(delays_l, 0.0)
			delays_r = np.maximum(delays_r, 0.0)

		# Small detune factors
		f = float(2.0 ** (pitch_cents / 1200.0))
		f_inv = float(1.0 / f) if f != 0.0 else 1.0

		if mono_safe:
			mid, side = _mid_side(stereo_ct)
			# Process side only; mid passes through to preserve mono sum.
			side_l = _fractional_delay_variable(side, delays_l)
			side_r = _fractional_delay_variable(side, delays_r)
			# Slightly different detune per channel for widening
			side_l = _linear_resample_same_length(side_l, f)
			side_r = _linear_resample_same_length(side_r, f_inv)
			wet_side = 0.5 * (side_l + side_r)
			return _from_mid_side(mid, wet_side)

		# Non-mono-safe: process L/R independently
		l = stereo_ct[0]
		r = stereo_ct[1]
		wet_l = _fractional_delay_variable(l, delays_l)
		wet_r = _fractional_delay_variable(r, delays_r)
		wet_l = _linear_resample_same_length(wet_l, f)
		wet_r = _linear_resample_same_length(wet_r, f_inv)
		return np.stack([wet_l, wet_r], axis=0).astype(np.float32, copy=False)

	def _process_spectral(
		self,
		stereo_ct: np.ndarray,
		fs: int,
		mask_mode: str,
		fft_size: int,
		mono_safe: bool,
		energy_match: bool,
	) -> np.ndarray:
		# Analyze mono for determinism
		mono = 0.5 * (stereo_ct[0] + stereo_ct[1])

		# Build spectral split L/R from mono
		l_s, r_s = _stft_stereoizer(mono, fs=fs, fft_size=int(fft_size), mask_mode=str(mask_mode), rng_seed=0)

		if mono_safe:
			# Use split as SIDE around original mono MID, so mono sum remains stable.
			mid = mono
			side = 0.5 * (l_s - r_s)
			out = _from_mid_side(mid, side)
		else:
			out = np.stack([l_s, r_s], axis=0)

		if energy_match:
			in_rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
			out_mono = 0.5 * (out[0] + out[1])
			out_rms = float(np.sqrt(np.mean(out_mono * out_mono) + 1e-12))
			if out_rms > 1e-9:
				out = out * (in_rms / out_rms)

		return out.astype(np.float32, copy=False)
