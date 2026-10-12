# Corrected schedule and controlled experiments

The default schedule interpolates **bounded base bits** instead of multiplying
them by a temperature and clipping. With warm-up 20, transition endpoint 50:

```
r(t) = clip((t - 20) / 30, 0, 1)
b(t) = (1 - r(t)) * 8 + r(t) * b_base
target(t) = (1 - r(t)) * 8 + r(t) * 4
```

Quantization is bypassed through epoch 20. Fractional QAT starts at epoch 21;
the derivative with respect to the base bits is r(t), positive from that epoch.
The feasible interval is [8-6r, 8], which contains target=8-4r throughout.
At epoch 50 the target is exactly 4 and the learned policy has its full [2,8]
range. The budget is a soft loss, so feasibility does not guarantee attainment.
Inference rounds to integer bits; early rounded inference maps may still be all
8 even though continuous QAT maps differ. `legacy_multiply` explicitly retains
the historical plateau and target mismatch for comparison. It is not the default.

## Run the controls

Use the same local student and teacher checkpoints for every variant. The YAML
controls optimizer, augmentation defaults, image size, batch size and epochs.
The data YAML supplies `path`, `train`, `val`, and `names` in YOLO format.

```bash
python -m mcaq_yolo.experiments --config configs/train_config.yaml \
  --data /data/cse/data.yaml --student /models/student_initial.pt \
  --teacher /models/teacher.pt --device cuda:0 \
  --output-dir outputs/schedule_controls --seeds 0 1 2 3 4 5 6 7 8 9
```

Add `--prepare-only` to write all resolved run configs without training.
Choose a fresh output directory for a new run; existing histories are not
silently overwritten. A single prepared config can be run using
`python -m mcaq_yolo.train --config <run>/config.yaml`.

| Variant | Intervention relative to `both` |
| --- | --- |
| `both` | sample admission and corrected precision schedule |
| `neither` | all samples and final precision from epoch 1 |
| `sample_only` | sample admission; final precision from epoch 1 |
| `precision_only` | all samples; corrected precision schedule |
| `uniform` | spatially constant final target bits (4 by default) |
| `shuffle` | permute base bits within equal-area tiles of each level |
| `gate_off` | set m(p)=1 while retaining adaptive bits |
| `layerwise` | replace each image/level map by its element-weighted mean |
| `std` | channel-mean feature standard deviation per tile, normalized by its image maximum, followed by the same bit mapper |
| `legacy` | original multiplicative temperature and budget schedule |

KD, initialization, batch size, optimizer and training duration are inherited.
The four curriculum variants isolate the two switches. Precision-off also removes
its high-precision warm-up and smoothness ramp. Gate-off resets the data RNG after
model construction so omitted gate parameters do not change augmentation seeds.
Sample admission necessarily changes the number of optimizer updates; the history
records samples and epochs. This is an equal-epoch comparison, not an assertion
of equal sample exposure or equal wall-clock cost.

Shuffle uses a local seeded generator and equal-area groups, preserving both the
tile histogram and element cost of each incoming map, including 20x20 features
partitioned into 8x8 tiles. Independently trained controls can still reach different
budgets. Layerwise rounding can change element cost. Compare the **measured** costs,
not just the shared target; repeat runs at a target grid if matched achieved costs
are needed. Do not label these separately trained models as exact budget matches.

For an exact-budget **paired intervention**, load a single adaptive checkpoint:

```bash
python -m mcaq_yolo.experiments \
  --config outputs/schedule_controls/both/seed_0/resolved_config.yaml \
  --paired-checkpoint outputs/schedule_controls/both/seed_0/best.pt \
  --output-dir outputs/paired_seed_0 --device cuda:0
```

All maps are computed once on the original forward, permuted within equal-area
groups, and replayed together. This prevents changed upstream features from
altering later maps. `paired_result.json` stores baseline/shuffled mAP and actual
bit costs. The learned gate remains active and can respond to changed maps, so
this tests placement within the gated system. It is an inference perturbation,
distinct from retraining with shuffle. The reference model must be adaptive.

## Saved evidence

Every run saves `manifest.json` (configuration, package versions, source hashes,
available input hashes), `resolved_config.yaml`, `history.jsonl`, `best.pt`,
`last.pt`, and `result.json` (selected checkpoint hash and evaluation at the final
inference policy). Selection uses final-stage validation mAP50. Short smoke runs
are marked through `final_policy_trained`; they do not establish accuracy.

`history.jsonl` includes actual per-level minimum/mean/maximum bits, continuous
8-bit saturation fraction, rounded integer histograms, policy mean, element mean,
target gap, and descriptor/combined-score means and population standard deviations.
The fractal `fallback_value_fraction` counts normalized D_f/2 == 0.5; it includes
natural values equal to the fallback and is **not an instrumented branch count**.
Contour `zero_fraction` identifies the observed degenerate outputs. Descriptor
statistics describe the actual feature tiles, not object ROIs or Table IV's
historical category aggregates.

The ideal bit-payload byte estimate excludes maps, scales, gates and alignment and
does not imply packed storage. Actual QDQ feature/map tensor bytes are reported
separately per image; sums of hook outputs are not peak GPU memory. This path has
floating-point weights/activations and does not reproduce a W4 deployment engine.

```bash
python -m mcaq_yolo.experiments --output-dir outputs/schedule_controls --collect-only
```

The collector writes seed-level `summary.csv` and `aggregate.json` (mean and sample
standard deviation). mAP50 is in [0,1] and uses the reference evaluator's all-point
AP at IoU 0.5. It does not report BCa intervals or manufacture deployment FPS.
The historical paper metrics have not been remeasured with this corrected code;
the original benchmark dataset, checkpoints and deployment engine are required
before associating new runs with those table rows.

```bash
python -m mcaq_yolo.experiments --output-dir outputs/schedule_controls --plot-only
```

This renders `bit_diagnostics.png` per run: per-level continuous means/min/max,
maximum-bit saturation, target/achieved means, and final-training versus selected
endpoint inference histograms. These plots come from saved measurements.

`legacy` checkpoints are selected by endpoint-inference validation mAP, with
scheduled validation recorded separately. `final_policy_trained` is false for
the legacy annealing path, which does not reach multiplier one; `schedule_completed`
indicates completion of its configured duration. The collector includes completed
schedules and keeps both flags in seed-level records. Copy `resolved_config.yaml`
beside `best.pt` when moving a trained control: inference automatically reloads
its allocation/gate/seed settings and requires a strict state match.
