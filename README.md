# Impulsive Noise Removal from Music Recordings

This project implements an **adaptive autoregressive (AR) model** with **Exponentially Weighted Least Squares (EW-LS)** to remove impulsive noise (clicks/pops) from `.wav` files.

---

## Description

The algorithm:
1. Models the audio using a 4th-order AR model.
2. Estimates coefficients adaptively using EW-LS with forgetting factor λ = 0.99.
3. Detects impulsive noise based on a dynamic threshold (η·δₑ).
4. Removes corrupted samples using linear interpolation.

---

## Requirements

Install dependencies:

pip install numpy scipy matplotlib
