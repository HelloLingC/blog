---
title: "Lecture 02: First-Order Optimization & Gradient Descent"
date: 2026-10-28T10:30:00+08:00
weight: 2
---

## Overview

In Lecture 02, we cover first-order optimization algorithms.

### Gradient Descent Update Rule

The parameters are iteratively updated in the direction of steepest descent:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

where $\eta > 0$ represents the learning rate (step size).

### Variants

- **Batch Gradient Descent (BGD)**: Computes the true gradient over the entire dataset.
- **Stochastic Gradient Descent (SGD)**: Approximates the gradient using a single random sample.
- **Mini-batch SGD**: Uses a subset of $B$ samples for improved computational efficiency and variance reduction.
