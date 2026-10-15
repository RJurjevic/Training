# Training

A small C# utility for parsing Nodchip trainer logs and generating Mathematica plots for Vafra NNUE training progress.

## What it does

This tool reads training log output containing repeated `PROGRESS (calc_loss)` blocks, extracts:

- epoch
- validation loss
- move accuracy

and generates a Wolfram Mathematica script and notebook for visualizing the run.

At the moment the generated plots include:

- validation loss vs epoch
- move accuracy vs epoch
- the lowest observed validation loss
- a smoothed trend line
- a trend-supported validation loss suggestion

## Background

This utility is used in work around my Cfish clone **Vafra**, mainly for inspecting NNUE training runs produced with a slightly modified **Nodchip trainer**.

The current workflow is:

1. run training and save the trainer log to a text file
2. run this utility on that log
3. open the generated Mathematica script or notebook
4. inspect the resulting plots and candidate validation-loss checkpoints

## Usage

Build and run the program with the trainer log file as the argument:

```bash
Training.exe 15000.txt
```

For example, from the Release build directory:

```bat
Training.exe "C:\Users\Hostmaster\OneDrive\Chess\20050.txt"
```

The utility writes `plot_progress.wl` beside the input log, replacing an existing file of that name. It prints the number of parsed samples and the output path.

In Mathematica, evaluate:

```mathematica
Get["C:/Users/Hostmaster/OneDrive/Chess/plot_progress.wl"]
```

The script opens a notebook containing the plots and saves it as `plot_progress.nb` in the script's directory. The executable generates the script; Mathematica generates the notebook when the script is evaluated.

## Orange trend curve

The orange dashed curve uses robust local linear smoothing (LOWESS-style). Previously it used a 101-record moving average, whose first and last estimates omitted the outer 50 records.

For each recorded epoch, the new display smoother:

- Uses observations within 1,000 epochs on either side: a 2,000-epoch neighbourhood in the interior. For sparse data it expands to at least five records, or all available records if fewer exist.
- Fits a local straight line with tricube distance weights, giving nearby observations more influence.
- Applies up to two bisquare residual reweightings after the initial fit, reducing the influence of isolated spikes.
- Uses the available one-sided neighbourhood at the beginning and end, so the curve reaches the latest recorded epoch. Endpoint estimates have less support than interior estimates.

The small local fits join into the displayed curved trend. The calculation centres epoch and loss locally for numerical stability, with weighted-mean fallbacks for degenerate fits. It runs in C#; Mathematica plots the resulting points.

This is smoothing within the observed epoch range, with no forecast beyond it. It can retain local bends; it is not constrained to decrease monotonically. A wider neighbourhood would produce more smoothing while hiding shorter-term changes.

Only the orange curve changed. The checkpoint-selection rules, raw data, markers, plot styling and axes remain unchanged.

## Lowest and trend-supported validation loss

Both suggestions identify actual parsed epoch/loss observations, not values taken from the orange curve. They refer to candidate networks only if the corresponding checkpoints were saved by the trainer; this utility does not save or verify network files.

**Lowest validation loss** selects the smallest raw validation loss across all parsed samples, choosing the earliest epoch on a tie.

**Trend-supported validation loss** retains the original selection rule:

1. Calculate the average epoch and average loss for every complete window of 101 consecutive records.
2. Find the window with the smallest average loss; use its average epoch as a cutoff. Ties choose the earlier cutoff.
3. Among all raw samples at or before that cutoff, select the smallest validation loss, choosing the earliest epoch on a tie.

With fewer than 101 samples, it selects the overall raw minimum instead.

Thus, the trend-supported candidate is a record-low observation within the eligible prefix of the log. It need not be the overall minimum, a strict local minimum, or a checkpoint explicitly marked accepted by the trainer. It can match the overall minimum when that observation lies within the eligible prefix.

For example, one run reports an overall minimum of 0.0291517 at epoch 31820 and a trend-supported candidate of 0.0291756 at epoch 31550. The later, lower observation is outside the eligible prefix for that run.

The name describes this cutoff heuristic; it does not establish statistical significance or stronger chess play. Compare candidate networks with engine matches to assess playing strength.

The orange display curve and this selection calculation intentionally use different smoothers. Extending the orange curve to the final epoch does not change either candidate. Results may change when additional log records are parsed.

## Version 2.0.0.0

The display curve now uses robust local linear smoothing and reaches both ends of the observed data. The existing checkpoint-selection logic is preserved.
