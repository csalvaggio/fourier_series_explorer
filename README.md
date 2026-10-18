# Fourier Series Explorer

An interactive classroom demonstration of Fourier-series synthesis, written in Python with **NumPy** and **Matplotlib**. The program displays six synchronized plots and lets you build a signal from its Fourier components, one harmonic at a time.

It runs on **macOS and Linux** using Matplotlib's available graphical backend; the code contains no platform-specific window-management logic. Resize or maximize the window using your desktop's normal window controls.

## Requirements and installation

- Python 3.10 or later
- NumPy
- Matplotlib
- A graphical Matplotlib backend for the interactive application (for example, a working Qt or macOS backend)

Install the packages in your preferred Python environment:

```bash
python3 -m pip install numpy matplotlib
```

## Run the explorer

```bash
python3 fourier_series_explorer.py
```

Click inside the figure window to give it keyboard focus. The instructions at the top of the figure summarize the most common controls.

### Keyboard controls

| Key | Action |
| --- | --- |
| **Space** | Add the next Fourier component (DC first, then harmonics through Nyquist) |
| **Backspace** or **↓** | Remove the most recently added component |
| **←** / **→** | Shift the input signal one sample to the left / right |
| **W** (Shift+w) | Increase rectangle/triangle width, Gaussian standard deviation, individual pulse width, pulse-train duty cycle, sine frequency, or sawtooth period count |
| **w** | Decrease the corresponding width, standard deviation, frequency, or period count |
| **1** | Select rectangle |
| **2** | Select triangle |
| **3** | Select sine |
| **4** | Select Gaussian |
| **5** | Select sawtooth |
| **6** | Select two rectangles |
| **7** | Select periodic pulse train (square wave at 50% duty cycle) |
| **[** / **]** | Function 6: decrease/increase pulse separation; function 7: halve/double the number of repetitions (increase/decrease repetition period) |
| **+** or **=** | Double the number of samples (up to 512) |
| **-** | Halve the number of samples (down to 32) |
| **d** | Show or hide dense continuous Fourier-series interpolation |
| **r** or **Home** | Restart component-by-component synthesis for the current input |
| **q** or **Esc** | Close the figure |

**Default:** 128 equally spaced samples on **[-2, 2)**, a period of 4, a centered rectangle extending from -1 to 1, and no components added yet.

The available sample counts are **32, 64, 128, 256, and 512**. Changing the sample count recomputes the transform and resets the synthesis stage. Changing function type, position, width, period count, pulse separation, or sine frequency recomputes the transform while retaining the current stage. The dense-curve toggle does not change the Fourier coefficients.

## Understanding the six plots

1. **Input function:** the selected sampled function.
2. **Magnitude spectrum:** magnitudes of the normalized, two-sided DFT coefficients; selected frequency bins are highlighted.
3. **Phase spectrum:** phase angles in radians; phase is omitted where the coefficient is effectively zero, because it is undefined there.
4. **Current component:** the real-valued sinusoid (or DC term) being added at the current stage.
5. **Previously included components:** all included real harmonic contributions plotted separately.
6. **Cumulative reconstruction:** the sum so far compared with the target samples, with the maximum sample error displayed. Press **d** to show the reconstructed *continuous trigonometric interpolant* between the samples.

The bottom status line displays the sample count, current stage, total available stages, and whether dense reconstruction is enabled. It reports **COMPLETE** at the final (Nyquist) stage.

## Fourier convention and teaching notes

The input is sampled over **[-2, 2)**, so one period has length **4**. The script calculates normalized discrete Fourier coefficients,

\[
C_k=\frac{1}{N}\sum_{n=0}^{N-1} f(x_n)e^{-i2\pi kn/N}.
\]

These coefficients describe the Fourier series of a **periodic, sampled signal**. The plot labels use the familiar Fourier-transform notation \(\mathcal{F}\{f(x)\}\), and the spectrum titles explicitly identify the values as normalized DFT bins. The frequency increment is **1/4 cycle per unit x**.

Each press of **Space** adds one *real* contribution: DC at stage 0, a conjugate positive/negative-frequency pair for each interior harmonic, and finally the **unpaired Nyquist term** at stage \(N/2\). For even \(N\), the full sequence therefore has \(N/2+1\) stages (65 stages at the default 128 samples).

When all terms have been included, the reconstruction agrees with **every sampled value** to floating-point precision. This is not a claim that the interpolated curve equals an ideal discontinuous rectangle at every position between samples. The **d** toggle makes that distinction visible, including ringing near transitions. Rectangle samples exactly on a jump are assigned the midpoint value, **0.5**.

### Available functions

| Key | Function | Principal observation | `w` / `W` |
| --- | --- | --- | --- |
| **1** | Rectangle | Sinc-like spectral envelope, sharp edges, and Gibbs ringing | Width |
| **2** | Triangle | Faster high-frequency decay than a rectangle | Width |
| **3** | Sine | Concentrated harmonics for integer cycles | Cycles per period |
| **4** | Gaussian | Narrower spatial profile means broader spectral content | Gaussian standard deviation (σ) |
| **5** | Sawtooth | Repeated discontinuities and slowly decaying harmonics | Periods over the four-unit domain |
| **6** | Two rectangles | Frequency-domain interference caused by pulse separation | Individual pulse width |
| **7** | Periodic pulse train | Harmonic comb, duty cycle, repetition frequency and odd-harmonic structure at 50% duty | Duty cycle (one sample per repetition period) |

For **two rectangles**, use **[** and **]** to move the pulse centers closer together or farther apart. Both pulse widths and their separation are quantized to the sample grid; the pulses combine by taking their maximum where they overlap rather than adding their heights. The Gaussian is **smoothly periodized** by summing shifted Gaussian copies separated by the four-unit domain period, then normalizing its peak to 1. Thus its value and slope match across the periodic boundary. On a finite sampled domain it still produces normalized DFT coefficients rather than the continuous, infinite-domain Gaussian transform. The sawtooth includes an integer number of periods in the domain, avoiding a mismatched-boundary example unless the signal is modified.

For **pulse train (`7`)**, `w` and `W` vary the number of high samples in each period (the duty cycle). `[` halves the number of repetitions, increasing the period, while `]` doubles repetitions, decreasing the period. The allowed repetition counts are powers of two (from 1 through at most `N/4`) to ensure every period occupies an integer number of samples. The title displays repetition count, period in x units, and **actual quantized duty cycle**. At 50% duty cycle, the **0-to-1** pulse train has a DC term and only odd multiples of the pulse-repetition fundamental (apart from numerical roundoff); this is different from a bipolar −1-to-1 square wave, which has zero DC. Adjusting duty cycle redistributes harmonic amplitudes. The phase of the harmonics changes when you translate the pulse train.

**Frequency-domain terminology:** all displayed spectral values are normalized DFT coefficients for periodically sampled signals, not literal samples of the infinite-domain continuous transform. In particular, the sampled, periodized Gaussian should not be confused with the exact continuous Fourier transform of an infinite-domain Gaussian.

Shifting a signal offers a particularly useful demonstration: its magnitude spectrum stays the same for integer-sample circular shifts, while its Fourier phase changes. Changing the shape or width provides complementary examples of changes in frequency content.

## Unit tests

The companion file **`test_fourier_series_explorer.py`** uses Python's built-in `unittest` module. Place it beside `fourier_series_explorer.py` and run:

```bash
python3 -m unittest -v test_fourier_series_explorer.py
```

Or discover all test modules in the current directory:

```bash
python3 -m unittest discover -v
```

The application keeps signal definitions in a compact central registry, enforces sample-dependent parameter limits in one normalization function, and separates keyboard state changes from plot refreshes. Cached harmonic lines are updated only when the sampled signal changes. These changes do not introduce any new dependencies.

The tests use Matplotlib's **Agg** (noninteractive) backend, so they can run without displaying a window. They check:

- Exact sampled reconstruction for all seven shapes at 32, 64, 128, 256, and 512 samples.
- Magnitude preservation and the expected phase change under integer-sample translation.
- Correct handling of the Nyquist term without doubling its amplitude.
- Agreement of the dense interpolation with the original sample values at the final stage.
- Masking of phase values where the Fourier coefficient is numerically zero.
- Keyboard callbacks remaining active after garbage collection, including all seven function selections, two-pulse separation, stepping, moving, resizing, sample counts, dense-curve toggling, and resetting.
- Gaussian width and smooth periodization, sawtooth periodicity and harmonics, spectral changes from pulse separation, pulse-train periodicity/duty cycle/harmonic zeros, and error handling for unknown functions.
- Function-registry consistency, parameter normalization across sample counts, GUI-independent keyboard actions, and unchanged previously plotted harmonic artists during ordinary stepping.

The tests exercise mathematics and noninteractive Matplotlib event handling. For a full check after GUI or layout changes, also launch the application and try the keyboard controls on the actual macOS or Linux desktop.

## Project files

```text
fourier_series_explorer.py       # interactive application
test_fourier_series_explorer.py  # regression tests
requirements.txt                 # module requirements file
LICENSE                          # license file
README.md                        # this guide
```

The explorer is intentionally a **single-file application** with no custom GUI framework or operating-system-specific dependencies.

## License

This project is licensed under the GNU General Public License v3.0.
See `LICENSE` for details.

## Contact

### Author

Carl Salvaggio, Ph.D.  
Professor of Imaging Science  
Director, Digital Imaging and Remote Sensing (DIRS) Laboratory

### E-mail

carl.salvaggio@rit.edu

### Organization

Chester F. Carlson Center for Imaging Science  
Rochester Institute of Technology  
Rochester, New York, 14623  
United States