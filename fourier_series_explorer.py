#!/usr/bin/env python3
"""Interactive, cross-platform Fourier-series explorer (NumPy + Matplotlib).

Samples cover [-2, 2), with period 4. FFT coefficients are normalized by N.
Space advances from DC (k=0) through Nyquist (k=N/2), grouping conjugate
frequency bins into one real-valued harmonic. Final synthesis interpolates
all N original samples (up to floating-point error).

Run: python3 fourier_series_explorer.py
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

X_MIN, X_MAX = -2.0, 2.0
PERIOD = X_MAX - X_MIN
DEFAULT_SAMPLES = 128
MIN_SAMPLES, MAX_SAMPLES = 32, 512
DENSE_POINTS = 2049


@dataclass(frozen=True)
class SignalDefinition:
    name: str
    key: str
    adjustable: str


SIGNALS = (
    SignalDefinition('rectangle', '1', 'pulse width'),
    SignalDefinition('triangle', '2', 'triangle width'),
    SignalDefinition('sine', '3', 'frequency'),
    SignalDefinition('gaussian', '4', 'standard deviation'),
    SignalDefinition('sawtooth', '5', 'period count'),
    SignalDefinition('two rectangles', '6', 'pulse width; [/] separation'),
    SignalDefinition('pulse train', '7', 'duty cycle; [/] repetition count'),
)
SIGNAL_BY_KEY = {item.key: item for item in SIGNALS}
SIGNAL_BY_NAME = {item.name: item for item in SIGNALS}


def normalized_settings(settings: 'Settings') -> 'Settings':
    """Constrain all sample-dependent parameters in one place."""
    s = settings
    if s.n < MIN_SAMPLES or s.n > MAX_SAMPLES or (s.n & (s.n - 1)):
        raise ValueError('Sample count must be a supported power of two')
    if s.kind not in SIGNAL_BY_NAME:
        raise ValueError(f'Unknown signal type: {s.kind}')
    dx = PERIOD / s.n
    s.half_width = float(np.clip(s.half_width, dx, PERIOD / 2 - dx))
    s.pulse_half_width = float(np.clip(s.pulse_half_width, dx, PERIOD / 2 - dx))
    s.pulse_separation = float(np.clip(s.pulse_separation, 0, PERIOD / 2))
    s.sine_cycles = int(np.clip(s.sine_cycles, 1, s.n // 2 - 1))
    s.saw_cycles = int(np.clip(s.saw_cycles, 1, s.n // 4))
    s.gaussian_sigma = float(np.clip(s.gaussian_sigma, dx, PERIOD / 2))
    allowed = [2 ** k for k in range(int(np.log2(s.n // 4)) + 1)]
    s.train_cycles = min(allowed, key=lambda v: abs(v - s.train_cycles))
    s.train_duty = float(np.clip(s.train_duty, 1 / (s.n // s.train_cycles),
                                 1 - 1 / (s.n // s.train_cycles)))
    s.stage = int(np.clip(s.stage, -1, s.n // 2))
    return s


def circular_sample_distance(indices: np.ndarray, center: int, n: int) -> np.ndarray:
    """Signed shortest grid displacement on an N-point circular domain."""
    return (indices - center + n // 2) % n - n // 2


def rectangular_pulse(indices: np.ndarray, center: int, n: int,
                      half_width_samples: int) -> np.ndarray:
    """Rectangular pulse with half-height samples at grid-aligned boundaries."""
    distance = np.abs(circular_sample_distance(indices, center, n))
    return (distance < half_width_samples).astype(float) + 0.5 * (distance == half_width_samples)


def periodic_gaussian(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth periodization of a Gaussian, normalized to unit peak at x=0.

    The periodic sum is truncated only after distant images are negligible.
    """
    # Even at maximum allowed sigma=2, 8 images each way puts omitted
    # contributions well below machine precision.
    offsets = PERIOD * np.arange(-8, 9, dtype=float)
    values = np.exp(-0.5 * ((distance[..., None] + offsets) / sigma) ** 2).sum(axis=-1)
    peak = np.exp(-0.5 * (offsets / sigma) ** 2).sum()
    return values / peak


@dataclass
class Settings:
    n: int = DEFAULT_SAMPLES
    kind: str = 'rectangle'
    shift: float = 0.0           # physical x units, maintained when resampling
    half_width: float = 1.0      # physical x units
    sine_cycles: int = 2         # cycles per period
    gaussian_sigma: float = 0.45 # Gaussian standard deviation, in x units
    saw_cycles: int = 2          # sawtooth periods per 4-unit domain
    pulse_half_width: float = 0.30 # half-width of each of two pulses
    pulse_separation: float = 1.6 # distance between their centers
    train_cycles: int = 4        # power-of-two periods in the sampled domain
    train_duty: float = 0.5      # fraction of one period that is high
    stage: int = -1              # -1: no harmonics, 0: DC ... n/2: Nyquist
    dense: bool = False


def sample_signal(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Return a periodic, discretely sampled waveform at N spatial locations."""
    settings = normalized_settings(settings)
    n = settings.n
    x = np.linspace(X_MIN, X_MAX, n, endpoint=False)
    dx = PERIOD / n
    # Round the requested physical displacement/width to the sampling grid.
    center = (n // 2 + round(settings.shift / dx)) % n
    indices = np.arange(n)
    delta = circular_sample_distance(indices, center, n)
    dist = np.abs(delta)
    half = max(1, min(n // 2 - 1, round(settings.half_width / dx)))
    if settings.kind == 'rectangle':
        y = rectangular_pulse(indices, center, n, half)
    elif settings.kind == 'triangle':
        y = np.maximum(0.0, 1.0 - dist / half)
    elif settings.kind == 'sine':
        y = np.sin(2 * np.pi * settings.sine_cycles * (indices - center) / n)
    elif settings.kind == 'gaussian':
        y = periodic_gaussian(delta * dx, settings.gaussian_sigma)
    elif settings.kind == 'sawtooth':
        # Integer numbers of periods tile the sampled four-unit domain exactly.
        phase = settings.saw_cycles * (indices - center) / n
        y = 2 * ((phase + 0.5) % 1.0) - 1
    elif settings.kind == 'two rectangles':
        radius = max(1, min(n // 2 - 1, round(settings.pulse_half_width / dx)))
        half_separation = round(settings.pulse_separation / (2 * dx))
        left_center = (center - half_separation) % n
        right_center = (center + half_separation) % n
        y = np.maximum(rectangular_pulse(indices, left_center, n, radius),
                       rectangular_pulse(indices, right_center, n, radius))
    elif settings.kind == 'pulse train':
        # Integer sample-aligned periods prevent artificial boundary leakage.
        samples_per_period = n // settings.train_cycles
        high_samples = int(np.clip(round(settings.train_duty * samples_per_period),
                                   1, samples_per_period - 1))
        phase_samples = (indices - center + high_samples // 2) % samples_per_period
        y = (phase_samples < high_samples).astype(float)
    else:
        raise ValueError(f'Unknown signal type: {settings.kind}')
    return x, y


def decompose(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized DFT, real harmonic contributions, and all partial sums."""
    n = len(y)
    nyquist = n // 2
    c = np.fft.fft(y) / n
    index = np.arange(n)
    k = np.arange(1, nyquist)[:, None]
    terms = np.empty((nyquist + 1, n), dtype=float)
    terms[0, :] = c[0].real
    terms[1:nyquist, :] = 2 * np.real(c[1:nyquist, None] *
                                           np.exp(2j * np.pi * k * index / n))
    # The Nyquist bin is its own conjugate; do not double it.
    terms[nyquist, :] = np.real(c[nyquist] * (-1.0) ** index)
    partial = np.cumsum(terms, axis=0)
    return c, terms, partial


def dense_partial(coeff: np.ndarray, stage: int, x: np.ndarray) -> np.ndarray:
    """Evaluate the continuous trigonometric interpolant between sample points."""
    result = np.zeros_like(x)
    if stage < 0:
        return result
    n = len(coeff)
    nyquist = n // 2
    t = (x - X_MIN) / PERIOD
    result[:] = coeff[0].real
    if stage > 0:
        upper = min(stage, nyquist - 1)
        k = np.arange(1, upper + 1)
        result += np.sum(2 * np.real(coeff[k, None] *
                                     np.exp(2j * np.pi * k[:, None] * t)), axis=0)
    if stage >= nyquist:
        result += np.real(coeff[nyquist] * np.exp(2j * np.pi * nyquist * t))
    return result


def significant_coefficients(coeff: np.ndarray) -> np.ndarray:
    """Mask out undefined phases at numerically zero Fourier coefficients."""
    return np.abs(coeff) > max(1e-11, float(np.max(np.abs(coeff))) * 1e-9)


class FourierExplorer:
    """Manage UI state, cached signal analysis, and reusable plot artists."""

    def __init__(self) -> None:
        self.settings = normalized_settings(Settings())
        self.x = np.empty(0)
        self.signal = np.empty(0)
        self.coeff = np.empty(0, dtype=complex)
        self.components = np.empty((0, 0))
        self.partial = np.empty((0, 0))
        self.dense_x = np.linspace(X_MIN, X_MAX, DENSE_POINTS)
        self.reanalyze()

        # Match the companion Fourier Filtering Explorer's natural window
        # size and header layout. Window sizing stays backend-independent.
        self.fig, axes = plt.subplots(3, 2, figsize=(14, 9.1))
        self.fig.subplots_adjust(top=0.79, bottom=0.09, hspace=0.72,
                                 wspace=0.30, left=0.085, right=0.97)
        self.fig.canvas.manager.set_window_title('Fourier Series Explorer')
        (self.ax_input, self.ax_mag, self.ax_phase,
         self.ax_current, self.ax_previous, self.ax_sum) = axes.flat
        self.axes = axes.flat
        self.fig.text(0.5, 0.975, 'Fourier Series Explorer',
                      ha='center', va='top', fontsize=19, weight='bold')
        self.instructions = self.fig.text(
            0.5, 0.931,
            '1–7 function  |  Space/Backspace terms  |  ←/→ shift  |  '
            'w/W size/frequency  |  [/] gap/period\n'
            '+/- samples  |  d dense reconstruction  |  r reset  |  q quit',
            ha='center', va='top', fontsize=10, linespacing=1.6,
            family='monospace')
        self.status = self.fig.text(0.5, 0.023, '', ha='center', va='center',
                                    family='monospace', fontsize=10)
        for ax in axes.flat:
            ax.grid(alpha=0.20)
            ax.tick_params(labelsize=8)
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            ax.title.set_fontsize(9)
        for ax in (self.ax_input, self.ax_current, self.ax_previous, self.ax_sum):
            ax.set(xlim=(X_MIN, X_MAX), xlabel='x', ylabel='f(x)')
        self.ax_mag.set(xlabel='Frequency (cycles per x-unit)',
                        ylabel=r'$|\mathcal{F}\{f(x)\}|$')
        self.ax_phase.set(xlabel='Frequency (cycles per x-unit)',
                          ylabel='Phase (radians)',
                          ylim=(-np.pi - 0.25, np.pi + 0.25),
                          yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                          yticklabels=['−π', '−π/2', '0', 'π/2', 'π'])
        self.input_line, = self.ax_input.plot([], [], 'k.-', lw=1.4, ms=3)
        self.mag_stems = None
        self.mag_selected, = self.ax_mag.plot([], [], 'o', color='tab:red', ms=7)
        self.phase_dots, = self.ax_phase.plot([], [], '.', color='tab:purple', ms=5)
        self.phase_selected, = self.ax_phase.plot([], [], 'o', color='tab:red', ms=7)
        self.current_line, = self.ax_current.plot([], [], color='tab:blue', lw=1.6)
        self.ax_current.axhline(0, color='0.6', lw=0.7)
        self.previous_lines = []
        self.previous_data_valid = False
        self.target_line, = self.ax_sum.plot([], [], 'k.', ms=4, label='Target samples')
        self.sum_line, = self.ax_sum.plot([], [], color='tab:orange', lw=2,
                                           label='Reconstruction (samples)')
        self.dense_line, = self.ax_sum.plot([], [], color='tab:green', lw=1.2,
                                             alpha=0.9, label='Continuous interpolation')
        self.ax_sum.legend(loc='upper right', fontsize=7)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_signal_plots()
        self.update_reconstruction()

    def reanalyze(self) -> None:
        normalized_settings(self.settings)
        self.x, self.signal = sample_signal(self.settings)
        self.previous_data_valid = False
        self.coeff, self.components, self.partial = decompose(self.signal)

    def bins_for_stage(self) -> list[int]:
        k = self.settings.stage
        n = self.settings.n
        if k < 0:
            return []
        if k == 0 or k == n // 2:
            return [k]
        return [k, n - k]

    def update_signal_plots(self) -> None:
        s = self.settings
        n = s.n
        self.input_line.set_data(self.x, self.signal)
        dx = PERIOD / n
        width = 2 * max(1, min(n // 2 - 1, round(s.half_width / dx))) * dx
        shift = round(s.shift / dx) * dx
        if s.kind in ('rectangle', 'triangle'):
            desc = f'width={width:g}'
        elif s.kind == 'sine':
            desc = f'{s.sine_cycles} cycle(s) / 4 units, f={s.sine_cycles / PERIOD:g}'
        elif s.kind == 'gaussian':
            desc = f'σ={s.gaussian_sigma:.4g}'
        elif s.kind == 'sawtooth':
            desc = f'{s.saw_cycles} period(s) / 4 units'
        elif s.kind == 'pulse train':
            samples_per_period = n // s.train_cycles
            high_samples = int(np.clip(round(s.train_duty * samples_per_period),
                                       1, samples_per_period - 1))
            desc = (f'{s.train_cycles} cycles / 4 units, period={PERIOD / s.train_cycles:g}'
                    f', duty={100 * high_samples / samples_per_period:g}%')
        else:
            desc = (f'pulse width={2 * round(s.pulse_half_width / dx) * dx:g}'
                    f', separation≈{2 * round(s.pulse_separation / (2 * dx)) * dx:g}')
        self.ax_input.set_title(f'{s.kind.capitalize()} | {desc} | shift={shift:+.4g}')
        f = np.fft.fftfreq(n, d=dx)
        order = np.fft.fftshift(np.arange(n))
        mag = np.abs(self.coeff[order])
        if self.mag_stems is not None:
            self.mag_stems.remove()
        self.mag_stems = self.ax_mag.vlines(f[order], 0, mag, color='0.5', lw=0.9)
        limits = (-n / (2 * PERIOD) - 0.3, n / (2 * PERIOD) + 0.3)
        self.ax_mag.set(xlim=limits, ylim=(0, max(0.05, mag.max() * 1.15)))
        self.ax_phase.set_xlim(*limits)
        good = significant_coefficients(self.coeff)[order]
        self.phase_dots.set_data(f[order][good], np.angle(self.coeff[order][good]))
        self.ax_mag.set_title(r'Magnitude $|\mathcal{F}\{f(x)\}|$  (normalized)')
        self.ax_phase.set_title(r'Phase $\angle\mathcal{F}\{f(x)\}$  (undefined at zero magnitude)')
        extent = max(1.15, float(np.max(np.abs(self.signal))) * 1.2)
        self.ax_input.set_ylim(-extent if s.kind in ('sine', 'sawtooth') else -0.2, extent)
        self.ax_sum.set_ylim(-extent if s.kind in ('sine', 'sawtooth') else -max(0.2, extent * 0.2), extent)
        limit = max(0.55, float(np.max(np.abs(self.components))) * 1.12)
        self.ax_current.set_ylim(-limit, limit)
        self.ax_previous.set_ylim(-limit, limit)
        self.target_line.set_data(self.x, self.signal)

    def update_reconstruction(self) -> None:
        s = self.settings
        k = s.stage
        n = s.n
        bins = self.bins_for_stage()
        freq = np.fft.fftfreq(n, d=PERIOD / n)
        self.mag_selected.set_data(freq[bins], np.abs(self.coeff[bins]))
        good = significant_coefficients(self.coeff)
        selected = [i for i in bins if good[i]]
        self.phase_selected.set_data(freq[selected], np.angle(self.coeff[selected]))
        self.current_line.set_data(self.x,
                                   self.components[k] if k >= 0 else np.zeros(n))
        self.sum_line.set_data(self.x,
                               self.partial[k] if k >= 0 else np.zeros(n))
        # Reuse artists for all previously included harmonics; never remove/recreate on a step.
        for index in range(k + 1):
            if index == len(self.previous_lines):
                line, = self.ax_previous.plot([], [], lw=0.85, alpha=0.55)
                self.previous_lines.append(line)
            if not self.previous_data_valid or not self.previous_lines[index].get_xdata().__len__():
                self.previous_lines[index].set_data(self.x, self.components[index])
            self.previous_lines[index].set_visible(True)
        for index in range(k + 1, len(self.previous_lines)):
            self.previous_lines[index].set_visible(False)
        self.previous_data_valid = True
        if k < 0:
            self.ax_current.set_title('Current Fourier component: none')
        elif k == 0:
            self.ax_current.set_title(f'Current: DC (mean = {self.coeff[0].real:.5g})')
        else:
            amp = abs(self.coeff[k]) if k == n // 2 else 2 * abs(self.coeff[k])
            note = ' | Nyquist (unpaired)' if k == n // 2 else ''
            if amp < 1e-11:
                note += ' | zero amplitude'
            self.ax_current.set_title(f'Current harmonic {k}/{n // 2} | '
                                      f'f={k / PERIOD:g} | Amplitude={amp:.4g}{note}')
        self.ax_previous.set_title(f'Previously included components: {max(0, k + 1)}')
        reconstructed = self.partial[k] if k >= 0 else np.zeros(n)
        error = float(np.max(np.abs(self.signal - reconstructed)))
        self.ax_sum.set_title('Cumulative reconstruction')
        self.dense_line.set_visible(s.dense)
        if s.dense:
            self.dense_line.set_data(self.dense_x, dense_partial(self.coeff, k, self.dense_x))
        # Match the Filtering Explorer's 10-point monospace status typography
        # and generous double-space padding around each separator.
        progress = f'{k + 1}/{n // 2 + 1}'
        error_text = f'{error:.2e}'
        dense_text = 'ON' if s.dense else 'OFF'
        if k == n // 2:
            self.status.set_text(
                f'COMPLETE  |  N = {n}  |  Components = {progress}  |  '
                f'Max sample error = {error_text}  |  Dense = {dense_text}')
        else:
            stage = str(k) if k >= 0 else 'none'
            self.status.set_text(
                f'N = {n}  |  Stage = {stage}/{n // 2}  |  '
                f'Components = {progress}  |  Max sample error = {error_text}'
                f'  |  Dense = {dense_text}')
        self.fig.canvas.draw_idle()

    @staticmethod
    def normalize_key(key: str) -> str:
        """Normalize backend-specific aliases without changing upper/lowercase intent."""
        return {'space': ' ', 'plus': '+', 'minus': '-', 'shift+=': '+',
                'shift++': '+', 'shift+-': '_', 'shift+w': 'W',
                'esc': 'escape', 'return': 'enter',
                'bracketleft': '[', 'bracketright': ']'}.get(key, key)

    def apply_key(self, key: str) -> str:
        """Apply an action without touching the GUI; return change category.

        Categories are 'signal', 'stage', 'view', 'quit', or 'none'.
        """
        s = self.settings
        key = self.normalize_key(key)
        action = 'none'
        if key in ('+', '='):
            if s.n < MAX_SAMPLES:
                s.n *= 2
                s.stage = -1
                action = 'signal'
        elif key in ('-', '_'):
            if s.n > MIN_SAMPLES:
                s.n //= 2
                s.stage = -1
                action = 'signal'
        elif key == ' ':
            s.stage = min(s.n // 2, s.stage + 1)
            action = 'stage'
        elif key in ('backspace', 'down'):
            s.stage = max(-1, s.stage - 1)
            action = 'stage'
        elif key in ('left', 'right'):
            s.shift += (1 if key == 'right' else -1) * PERIOD / s.n
            action = 'signal'
        elif key in ('W', 'w'):
            direction = 1 if key == 'W' else -1
            dx = PERIOD / s.n
            if s.kind == 'sine':
                s.sine_cycles += direction
            elif s.kind == 'sawtooth':
                s.saw_cycles += direction
            elif s.kind == 'gaussian':
                s.gaussian_sigma += direction * dx
            elif s.kind == 'pulse train':
                samples_per_period = s.n // s.train_cycles
                s.train_duty = (round(s.train_duty * samples_per_period) + direction) / samples_per_period
            elif s.kind == 'two rectangles':
                s.pulse_half_width += direction * dx
            else:
                s.half_width += direction * dx
            action = 'signal'
        elif key in ('[', ']'):
            if s.kind == 'pulse train':
                s.train_cycles *= 2 if key == ']' else 0.5
                s.train_cycles = int(s.train_cycles)
                action = 'signal'
            elif s.kind == 'two rectangles':
                s.pulse_separation += (-1 if key == '[' else 1) * 2 * PERIOD / s.n
                action = 'signal'
        elif key in SIGNAL_BY_KEY:
            s.kind = SIGNAL_BY_KEY[key].name
            action = 'signal'
        elif key in ('r', 'home'):
            s.stage = -1
            action = 'stage'
        elif key in ('d', 'D'):
            s.dense = not s.dense
            action = 'view'
        elif key in ('q', 'escape'):
            return 'quit'
        if action == 'signal':
            normalized_settings(s)
        return action

    def on_key(self, event) -> None:
        action = self.apply_key(event.key)
        if action == 'quit':
            plt.close(self.fig)
        elif action != 'none':
            if action == 'signal':
                self.reanalyze()
                self.update_signal_plots()
            self.update_reconstruction()


def main() -> None:
    plt.rcParams.update({'font.size': 9})
    # Matplotlib uses weak references for bound callbacks; retain this object.
    explorer = FourierExplorer()
    plt.show()


if __name__ == '__main__':
    main()
