"""Regression tests for fourier_series_explorer.py; uses only unittest and NumPy."""
import gc
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

import numpy as np

from fourier_series_explorer import (
    PERIOD, Settings, SIGNALS, SIGNAL_BY_KEY, normalized_settings, periodic_gaussian,
    decompose, dense_partial, sample_signal, significant_coefficients, FourierExplorer, main,
)


class FourierSeriesTests(unittest.TestCase):
    def test_exact_reconstruction_all_signals(self):
        for n in (32, 64, 128, 256, 512):
            for kind in ('rectangle', 'triangle', 'sine', 'gaussian', 'sawtooth', 'two rectangles', 'pulse train'):
                with self.subTest(n=n, kind=kind):
                    _, y = sample_signal(Settings(n=n, kind=kind))
                    c, components, partial = decompose(y)
                    self.assertEqual(components.shape, (n // 2 + 1, n))
                    self.assertLess(np.max(np.abs(partial[-1] - y)), 3e-13)
                    self.assertTrue(np.allclose(c, np.fft.fft(y) / n))

    def test_translation_preserves_magnitude_and_changes_phase(self):
        n = 128
        dx = PERIOD / n
        for kind in ('rectangle', 'triangle', 'sine', 'gaussian', 'sawtooth', 'two rectangles', 'pulse train'):
            _, original = sample_signal(Settings(n=n, kind=kind))
            _, shifted = sample_signal(Settings(n=n, kind=kind, shift=5 * dx))
            c0 = decompose(original)[0]
            c1 = decompose(shifted)[0]
            expected = c0 * np.exp(-2j * np.pi * np.arange(n) * 5 / n)
            self.assertTrue(np.allclose(np.abs(c0), np.abs(c1), atol=1e-13))
            self.assertTrue(np.allclose(c1, expected, atol=1e-13))

    def test_nyquist_is_not_doubled(self):
        n = 64
        y = (-1.0) ** np.arange(n)
        coeff, components, partial = decompose(y)
        self.assertAlmostEqual(coeff[n // 2].real, 1.0)
        self.assertTrue(np.allclose(components[-1], y))
        self.assertTrue(np.allclose(partial[-1], y))

    def test_dense_interpolation_agrees_with_samples(self):
        n = 128
        x, y = sample_signal(Settings(n=n, kind='rectangle'))
        c, _, _ = decompose(y)
        self.assertLess(np.max(np.abs(dense_partial(c, n // 2, x) - y)), 3e-13)
        self.assertTrue(np.array_equal(dense_partial(c, -1, x), np.zeros(n)))

    def test_keyboard_callbacks_survive_garbage_collection(self):
        """Exercise the real key-event path while main() is inside show()."""
        def exercise_window():
            gc.collect()
            figure = plt.gcf()
            send = lambda key: KeyEvent('key_press_event', figure.canvas, key=key)._process()
            send('space')
            self.assertIn('Current: DC', figure.axes[3].get_title())
            send('space')
            self.assertIn('harmonic 1/', figure.axes[3].get_title())
            send('right')
            self.assertIn('shift=+', figure.axes[0].get_title())
            send('2')
            self.assertIn('triangle', figure.axes[0].get_title())
            send('3')
            self.assertIn('sine', figure.axes[0].get_title())
            send('4')
            self.assertIn('gaussian', figure.axes[0].get_title())
            send('5')
            self.assertIn('sawtooth', figure.axes[0].get_title())
            send('6')
            self.assertIn('two rectangles', figure.axes[0].get_title())
            old_title = figure.axes[0].get_title()
            send(']')
            self.assertNotEqual(old_title, figure.axes[0].get_title())
            send('[')
            self.assertEqual(old_title, figure.axes[0].get_title())
            send('7')
            self.assertIn('pulse train', figure.axes[0].get_title())
            initial_title = figure.axes[0].get_title()
            send('W')
            self.assertNotEqual(initial_title, figure.axes[0].get_title())
            send('w')
            self.assertEqual(initial_title, figure.axes[0].get_title())
            send(']')
            self.assertIn('8 cycles', figure.axes[0].get_title())
            send('[')
            self.assertIn('4 cycles', figure.axes[0].get_title())
            send('3')
            send('W')
            self.assertIn('3 cycle(s)', figure.axes[0].get_title())
            send('w')
            self.assertIn('2 cycle(s)', figure.axes[0].get_title())
            send('+')
            self.assertIn('N=256', figure.texts[-1].get_text())
            send('-')
            self.assertIn('N=128', figure.texts[-1].get_text())
            send('d')
            self.assertIn('dense: ON', figure.texts[-1].get_text())
            send('backspace')
            self.assertIn('Current Fourier component: none', figure.axes[3].get_title())
            send('r')
            self.assertIn('Current Fourier component: none', figure.axes[3].get_title())

        try:
            with mock.patch('fourier_series_explorer.plt.show', side_effect=exercise_window):
                main()
        finally:
            plt.close('all')

    def test_gaussian_width_and_spectrum(self):
        narrow = sample_signal(Settings(kind='gaussian', gaussian_sigma=0.20))[1]
        broad = sample_signal(Settings(kind='gaussian', gaussian_sigma=0.65))[1]
        self.assertGreater(broad.sum(), narrow.sum())
        self.assertGreater(np.count_nonzero(narrow > 0.5), 0)
        # Smooth, sufficiently broad Gaussians suppress high-frequency content.
        spectrum = abs(decompose(broad)[0])
        self.assertLess(spectrum[20], spectrum[1])

    def test_sawtooth_periodicity_and_nonzero_high_harmonics(self):
        a = sample_signal(Settings(kind='sawtooth', saw_cycles=2))[1]
        self.assertTrue(np.allclose(a[:64], a[64:]))
        self.assertGreater(abs(decompose(a)[0][12]), 0.005)
        self.assertTrue(np.all(a >= -1) and np.all(a <= 1))

    def test_two_pulses_separation_changes_spectrum(self):
        near = sample_signal(Settings(kind='two rectangles', pulse_separation=1.0))[1]
        far = sample_signal(Settings(kind='two rectangles', pulse_separation=1.75))[1]
        self.assertFalse(np.allclose(abs(decompose(near)[0]), abs(decompose(far)[0])))

    def test_pulse_train_periodicity_duty_cycle_and_harmonics(self):
        n = 128
        s = Settings(n=n, kind='pulse train', train_cycles=4, train_duty=0.5)
        _, y = sample_signal(s)
        m = n // s.train_cycles
        self.assertTrue(np.array_equal(y[:m], y[m:2 * m]))
        self.assertEqual(int(y.sum()), n // 2)
        c, _, parts = decompose(y)
        self.assertLess(np.max(np.abs(parts[-1] - y)), 3e-13)
        # Four periods: non-DC coefficients occur only at multiples of four.
        self.assertLess(np.max(np.abs(c[1:4])), 1e-12)
        # At 50% duty cycle, even multiples of the fundamental vanish.
        for k in (8, 16, 24, 32, 40, 48, 56):
            self.assertLess(abs(c[k]), 1e-12)
        self.assertGreater(abs(c[4]), 0.1)
        narrower = sample_signal(Settings(n=n, kind='pulse train',
                                          train_cycles=4, train_duty=0.25))[1]
        self.assertEqual(int(narrower.sum()), n // 4)
        _, shifted = sample_signal(Settings(n=n, kind='pulse train',
                                           train_cycles=4, shift=3 * PERIOD / n))
        self.assertTrue(np.array_equal(shifted, np.roll(y, 3)))

    def test_pulse_train_all_sample_counts_and_repetition_controls(self):
        for n in (32, 64, 128, 256, 512):
            for cycles in (1, 2, 4, 8):
                with self.subTest(n=n, cycles=cycles):
                    y = sample_signal(Settings(n=n, kind='pulse train',
                                               train_cycles=cycles))[1]
                    self.assertTrue(np.array_equal(y, np.roll(y, n // cycles)))
                    self.assertLess(np.max(np.abs(decompose(y)[2][-1] - y)), 3e-13)

    def test_registry_is_complete_and_unique(self):
        self.assertEqual([entry.key for entry in SIGNALS], list('1234567'))
        self.assertEqual(len({entry.name for entry in SIGNALS}), len(SIGNALS))
        for entry in SIGNALS:
            self.assertEqual(SIGNAL_BY_KEY[entry.key].name, entry.name)
            self.assertEqual(len(sample_signal(Settings(kind=entry.name))[1]), 128)

    def test_parameter_normalization_and_resampling_limits(self):
        for n in (32, 64, 128, 256, 512):
            s = Settings(n=n, kind='pulse train', half_width=100,
                         pulse_half_width=-1, pulse_separation=100,
                         gaussian_sigma=-1, train_cycles=4096,
                         train_duty=0, sine_cycles=10000, saw_cycles=-3,
                         stage=10000)
            normalized_settings(s)
            self.assertLessEqual(s.half_width, PERIOD / 2 - PERIOD / n)
            self.assertGreaterEqual(s.pulse_half_width, PERIOD / n)
            self.assertLessEqual(s.pulse_separation, PERIOD / 2)
            self.assertGreaterEqual(s.gaussian_sigma, PERIOD / n)
            self.assertLessEqual(s.train_cycles, n // 4)
            self.assertEqual((s.train_cycles & (s.train_cycles - 1)), 0)
            self.assertGreater(s.train_duty, 0)
            self.assertEqual(s.stage, n // 2)
            self.assertLess(np.max(np.abs(decompose(sample_signal(s)[1])[2][-1] - sample_signal(s)[1])), 3e-13)
        with self.assertRaises(ValueError):
            normalized_settings(Settings(n=100))

    def test_periodic_gaussian_has_continuous_periodic_boundaries(self):
        sigma = 1.4
        epsilon = 1e-5
        left = periodic_gaussian(np.array([-PERIOD / 2 + epsilon]), sigma)[0]
        right = periodic_gaussian(np.array([PERIOD / 2 - epsilon]), sigma)[0]
        self.assertAlmostEqual(left, right, places=10)
        self.assertAlmostEqual(periodic_gaussian(np.array([0.]), sigma)[0], 1.0)
        self.assertTrue(np.allclose(periodic_gaussian(np.array([-.4, .2]), sigma),
                                    periodic_gaussian(np.array([-.4 + PERIOD, .2 - PERIOD]), sigma),
                                    atol=1e-14))

    def test_apply_key_updates_state_without_redrawing(self):
        app = FourierExplorer()
        try:
            with mock.patch.object(app, 'update_reconstruction', side_effect=AssertionError('draw')):
                self.assertEqual(app.apply_key('space'), 'stage')
                self.assertEqual(app.settings.stage, 0)
                self.assertEqual(app.apply_key('4'), 'signal')
                self.assertEqual(app.settings.kind, 'gaussian')
                self.assertEqual(app.apply_key('d'), 'view')
                self.assertTrue(app.settings.dense)
                self.assertEqual(app.apply_key('unused'), 'none')
                self.assertEqual(app.apply_key('q'), 'quit')
        finally:
            plt.close(app.fig)

    def test_cached_previous_lines_are_not_reset_on_each_stage(self):
        app = FourierExplorer()
        try:
            app.apply_key('space')
            app.update_reconstruction()
            line = app.previous_lines[0]
            with mock.patch.object(line, 'set_data', side_effect=AssertionError('unnecessary update')):
                app.apply_key('space')
                app.update_reconstruction()
            app.apply_key('left')
            app.reanalyze()
            app.update_signal_plots()
            app.update_reconstruction()
            self.assertIs(line, app.previous_lines[0])
        finally:
            plt.close(app.fig)

    def test_unknown_signal_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_signal(Settings(kind='unknown'))

    def test_zero_magnitude_phase_is_masked(self):
        c = np.array([2.0 + 0j, 0j, 1e-14 + 0j])
        self.assertEqual(significant_coefficients(c).tolist(), [True, False, False])


if __name__ == '__main__':
    unittest.main()
