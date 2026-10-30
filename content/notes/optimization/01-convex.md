---
title: "01: Convex Sets"
date: 2026-10-28T10:55:00+08:00
weight: 1
---

# Convex Sets

A set \(C\) is convex if and only if the line connecting any two points \(x, y \in C\) is also entirely in \(C\):

$$
\alpha x + (1-\alpha)y \in C, \quad \alpha \in [0,1]
$$

# Inequality Constraints


Intersections of Convex Sets are convex





## Linear Program Formulation

$$
\min_{\mathbf{x}} f(\mathbf{x}) = \mathbf{c}^T\mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \le \mathbf{b}
$$