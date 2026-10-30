---
title: "Lecture 01: Introduction & Problem Formulation"
date: 2026-10-28T10:00:00+08:00
weight: 1
---

## Overview

Welcome to Lecture 01 of the course notes.

In this lecture, we introduce the core formulation of optimization problems in machine learning and deep learning.

### Objective

Given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, we want to find model parameters $\theta$ that minimize empirical risk:

$$
\min_{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i; \theta), y_i)
$$

### Key Takeaways

1. Optimization forms the computational backbone of modern AI models.
2. Loss functions quantify prediction discrepancies.
