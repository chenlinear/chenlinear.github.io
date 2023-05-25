---
author: "Chen Li"
title: "LIGO O4: multi-messenger"
date: "2023-05-14"
tags: 
- Physics
math: true
---

LIGO is back on track with Observing Run 4 (O4)[^1]. And I got this idea:

Active Galactic Nucleus (AGN) is considered as a Gravitational Wave (GW) candidate that can be detected by LIGO-Virgo-Kagra[^2]. And IceCube recently proposed a method to combine neutrino detection with radio detection in the observation of AGN[^3]. So maybe these three messengers (GW, neutrino, radio) can work together to observe AGN. And maybe we should write a program that can synchronize these observations.

It's exciting that we can make use of basically all the messengers that we know of. Except cosmic rays, but galactic cosmic rays mostly can't reach earth anyway (because of their charge). However, their are few things we should consider:

1. We don't observe many GWs[^4], so we can just do it by hand.
2. A radio telescope can turn around, so it's kind of urgent to focus on the target. But based on the way IceCube works, there's no rush. IceCube, the basic part of which is a $1 \space \text{km}^3$ ice cube, can't turn around. LIGO can't turn around either.

[^1]: See [their news website](https://www.ligo.org/news/index.php#ER15) or [_Gravitational wave detectors prepare for next observing run_](https://www.ligo.org/news/images/ER15-newsitem.pdf).
[^2]: See [_Optical Emission Model for Binary Black Hole Merger Remnants Travelling through Discs of Active Galactic Nucleus_](https://arxiv.org/abs/2304.10567). I later found out Virgo is delayed, see [_Virgo postpones entry into O4 observing run_](https://www.virgo-gw.eu/news/virgo-postpones-entry-into-o4-observing-run/).
[^3]: See [_Search for correlations of high-energy neutrinos detected in IceCube with radio-bright AGN and gamma-ray emission from blazars_](https://arxiv.org/abs/2304.12675).
[^4]: See [Detection of gravitational waves](https://www.ligo.org/detections.php).
