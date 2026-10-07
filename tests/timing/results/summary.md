# Timing test summary

## Test R² (seed-averaged; land = LF ≥ 0.5)

| Model | current (all) | current (land) | current (ocean) | concurrent (all) | concurrent (land) | concurrent (ocean) |
|---|---|---|---|---|---|---|
| NN-BL | 0.286 | 0.229 | 0.323 | 0.275 | 0.211 | 0.319 |
| NN-GAUSS | 0.522 | 0.409 | 0.609 | 0.501 | 0.389 | 0.589 |
| SR-BL | 0.293 | 0.224 | 0.341 | 0.275 | 0.211 | 0.319 |
| SR-ATM | 0.336 | 0.301 | 0.355 | 0.321 | 0.290 | 0.338 |
| SR-SFC | 0.397 | 0.307 | 0.464 | 0.386 | 0.298 | 0.451 |
| SR-ALL | 0.426 | 0.304 | 0.519 | 0.408 | 0.292 | 0.498 |
| SR-ALL-PC | 0.428 | 0.306 | 0.522 | – | – | – |

Timesteps in split: current = 2208, concurrent = 2202

## NN-GAUSS kernels (seed mean)

| Variant | Predictor | Peak level | Spread | Seeds |
|---|---|---|---|---|
| current | rh | 0.50 | 0.235 | 3 |
| current | thetae | 1.00 | 0.133 | 3 |
| current | thetaestar | 0.75 | 0.118 | 3 |
| concurrent | rh | 0.50 | 0.225 | 3 |
| concurrent | thetae | 1.00 | 0.136 | 3 |
| concurrent | thetaestar | 0.75 | 0.125 | 3 |

## PySR equations

| Variant | Run | Seed | Ref. complexity equation | Elbow (complexity) | Elbow R² all / land / ocean |
|---|---|---|---|---|---|
| current | sr_bl | 42 | `cube(bl - -0.298459) + 0.13681741` | `cube(bl - -0.33914208)` (5) | 0.296 / 0.218 / 0.351 |
| current | sr_bl | 72 | `cube(bl + 0.29321024) + 0.144626` | `cube(bl + 0.33623663)` (5) | 0.295 / 0.219 / 0.348 |
| current | sr_bl | 102 | `cube(bl + 0.29592127) + 0.14504439` | `cube(bl + 0.3392554)` (5) | 0.296 / 0.218 / 0.351 |
| current | sr_atm | 42 | `cube(max(rh, (thetae - 0.36875376) - (thetaestar * 1.4769912))) * 1.566091` | `cube((thetae - (thetaestar * 1.454325)) - 0.18724935)` (11) | 0.219 / 0.155 / 0.262 |
| current | sr_atm | 72 | `cube(max(thetae - ((thetaestar * 1.4674053) - -0.22039457), rh + 0.17831568))` | `cube((thetae + -0.18664525) - (thetaestar * 1.4474149))` (11) | 0.219 / 0.155 / 0.262 |
| current | sr_atm | 102 | `None` | `((thetae + -0.93484056) * 4.1842575) - (thetaestar * 5.93664)` (11) | 0.224 / 0.161 / 0.265 |
| current | sr_sfc | 42 | `None` | `sr_atm_eq * (0.9885615 - (shf * (lf + -0.9659257)))` (10) | -0.035 / 0.294 / -0.324 |
| current | sr_sfc | 72 | `(sr_atm_eq + (lhf * 0.21017148)) + (shf * (0.7556408 - lf))` | `sr_atm_eq + ((0.94573975 - lf) * (sr_atm_eq * shf))` (10) | -0.010 / 0.294 / -0.279 |
| current | sr_sfc | 102 | `((shf * (0.7224154 - lf)) + sr_atm_eq) - (lhf * -0.2118274)` | `(sr_atm_eq * (shf * (0.93809164 - lf))) + sr_atm_eq` (10) | 0.011 / 0.294 / -0.239 |
| current | sr_all | 42 | `(lhf * 0.12917784) + (sr_atm_eq - ((shf * sr_atm_eq) * (lf + -0.8810418)))` | `sr_atm_eq - (((lf + -0.95535153) * shf) * sr_atm_eq)` (10) | -0.039 / 0.294 / -0.332 |
| current | sr_all | 72 | `(sr_atm_eq + (((0.8819423 - lf) * sr_atm_eq) * shf)) + (lhf * 0.14020321)` | `sr_atm_eq + ((0.94580233 - lf) * (sr_atm_eq * shf))` (10) | -0.010 / 0.294 / -0.279 |
| current | sr_all | 102 | `sr_atm_eq + ((thetae - (shf * -5.6251073)) * cube(0.6989902 - lf))` | `sr_atm_eq * ((shf * (0.9264647 - lf)) + 0.9804427)` (10) | 0.088 / 0.296 / -0.098 |
| concurrent | sr_bl | 42 | `cube(bl + 0.29477403) - -0.13524452` | `cube(bl - -0.33630553)` (5) | 0.279 / 0.212 / 0.326 |
| concurrent | sr_bl | 72 | `0.121017836 - cube(-0.3067702 - bl)` | `cube(bl - -0.342788)` (5) | 0.283 / 0.213 / 0.333 |
| concurrent | sr_bl | 102 | `cube(bl + 0.30968282) + 0.12672548` | `cube(bl + 0.34721392)` (5) | 0.286 / 0.214 / 0.337 |
| concurrent | sr_atm | 42 | `(cube(max((thetae * 0.7296798) - thetaestar, rh)) - 0.15575238) * 1.9550837` | `((thetae * 4.3837457) - (thetaestar * 6.296659)) - 4.1419654` (11) | 0.210 / 0.146 / 0.253 |
| concurrent | sr_atm | 72 | `None` | `(((thetaestar * 1.4292991) - thetae) + 0.94630796) * -4.473951` (11) | 0.215 / 0.151 / 0.259 |
| concurrent | sr_atm | 102 | `cube(max(thetae - ((thetaestar - -0.24442424) * 1.4854615), rh) * 1.1706634)` | `cube(thetae - (thetaestar * 1.4379362)) + -0.64704067` (11) | 0.238 / 0.160 / 0.292 |
| concurrent | sr_sfc | 42 | `(sr_atm_eq - ((lf - 0.7873303) * shf)) + (lhf * 0.21874899)` | `sr_atm_eq - ((shf * (lf - 0.9867445)) * sr_atm_eq)` (10) | 0.007 / 0.282 / -0.239 |
| concurrent | sr_sfc | 72 | `(shf - ((lf * (shf - -0.2234778)) + -1.0956119)) * sr_atm_eq` | `(shf * ((0.94862664 - lf) * sr_atm_eq)) + sr_atm_eq` (10) | 0.099 / 0.289 / -0.074 |
| concurrent | sr_sfc | 102 | `(lhf * 0.23208448) + (sr_atm_eq - (shf * (lf - 0.75228435)))` | `sr_atm_eq - (sr_atm_eq * (shf * (lf - 0.9755824)))` (10) | 0.036 / 0.284 / -0.186 |
| concurrent | sr_all | 42 | `None` | `sr_atm_eq + ((0.98698103 - lf) * (shf * sr_atm_eq))` (10) | 0.006 / 0.282 / -0.240 |
| concurrent | sr_all | 72 | `None` | `(((shf + 0.21329264) * (0.9036174 - lf)) + 0.8952705) * sr_atm_eq` (12) | -0.018 / 0.262 / -0.269 |
| concurrent | sr_all | 102 | `None` | `sr_atm_eq - ((sr_atm_eq * shf) * (lf - 0.9752584))` (10) | 0.037 / 0.284 / -0.184 |
| concurrent | sr_all_k1 | 42 | `sr_atm_eq - ((shf * (lf + min(-0.86987007, lhf))) * (sr_atm_eq + 0.28648353))` | `sr_atm_eq - ((shf * (lf + -0.9861786)) * sr_atm_eq)` (10) | 0.008 / 0.282 / -0.236 |
| concurrent | sr_all_k1 | 72 | `sr_atm_eq - (cube(0.7529247 - lf) * ((shf + (thetae * 0.13330923)) * -4.9089847))` | `sr_atm_eq * (((shf + 0.24784805) * (0.9162716 - lf)) + 0.8755642)` (12) | -0.112 / 0.252 / -0.433 |
| concurrent | sr_all_k1 | 102 | `sr_atm_eq - (((cube(lf - 1.2734146) * (shf + 0.26428914)) * thetae) - -0.18790826)` | `sr_atm_eq - ((lf - 0.9754822) * (shf * sr_atm_eq))` (10) | 0.036 / 0.284 / -0.185 |
