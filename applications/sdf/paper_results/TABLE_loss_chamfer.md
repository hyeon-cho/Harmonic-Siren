# Loss & Chamfer vs iteration (Armadillo SDF, 3 seeds, mean ± std)

| metric | method | iter 100 | iter 200 | iter 500 | iter 1000 | iter 1500 | iter 2000 |
|---|---|---|---|---|---|---|---|
| Loss | SIREN | 15.97 ± 1.14 | 14.20 ± 1.39 | 9.37 ± 0.75 | 7.60 ± 0.20 | 6.66 ± 0.10 | 6.36 ± 0.62 |
| Loss | Ours (R=3, lr 1e-4) | 16.70 ± 1.59 | 12.54 ± 0.38 | 8.25 ± 0.39 | 6.51 ± 0.20 | 5.67 ± 0.31 | 6.13 ± 0.95 |
| Chamfer x1e-6 | SIREN | 17070 ± 18915 | 8384 ± 7819 | 2286 ± 323 | 1201 ± 626 | 663 ± 456 | 484 ± 327 |
| Chamfer x1e-6 | Ours (R=3, lr 1e-4) | 3467 ± 439 | 2076 ± 820 | 1176 ± 705 | 421 ± 180 | 208 ± 27 | 373 ± 132 |

- Loss = SIREN SDF objective (|sdf|*3e3 + inter*1e2 + normal*1e2 + eikonal*5e1),
  trailing 10-epoch mean at each mark to smooth the random off-surface batch noise.
- Chamfer measured at 128^3 marching-cubes resolution (curve protocol); headline numbers
  at 256^3 are slightly lower (e.g., Ours best 196 @ep1300 on the mean curve).
- Fixed-batch loss decomposition at final ckpt (same points for both): SIREN 5.66 vs Ours 5.46,
  with the largest gap in the normal-alignment term (0.540 vs 0.381).
