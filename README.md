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