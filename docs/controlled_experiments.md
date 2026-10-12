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
Mosaic defaults to 1.0 and MixUp to 0.0 even when the imported teacher checkpoint
contains other values; set `augmentation` explicitly to override them. Other
augmentation parameters inherit Ultralytics/teacher settings unless overridden.
The full effective detection-augmentation settings are saved as
`resolved_augmentation` in the resolved config and manifest.
The data YAML supplies `path`, `train`, `val`, optional `test`, and `names` in YOLO
format. Validation is used for checkpoint selection. A configured test split is
evaluated only after training and loading the selected checkpoint. Set `data.test`
in the training config to override the data YAML's test path, or set it to `null`
to explicitly disable test evaluation. `--data` takes all split paths from the new
data YAML. Test image paths must be disjoint from both train and validation;
resolved path overlap is rejected (this is not an image-content duplicate check).
Without a test split, final results are explicitly marked `validation_only`.

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
Sample admission changes training effort. Each history row records observed
`processed_samples`, actual `optimizer_updates`, and `training_wall_seconds` under
`train`, plus cumulative totals under `training_effort`. Samples count batch images
actually processed, including partial batches; mosaic source images are not separate
sample exposures. Optimizer post-step hooks exclude AMP overflow skips from updates.
The wall timer synchronizes CUDA/MPS before and after each training epoch and includes
training data loading and host work; it excludes validation, checkpoint I/O, initial
complexity scoring, and final test evaluation. CPU timing uses the same wall clock.
Totals are also saved in `result.json` and the collector. This is an equal-epoch
comparison, not an assertion of equal sample exposure, update count, or elapsed cost.

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
inference policy). `dataset_index.json` includes every configured split, including
test, and label hashes. Selection uses final-stage validation mAP50. The result records
`selection_split=val`, selection method, `evaluation_split`, `evaluation_scope`, and
the evaluator `protocol`/`protocol_details`. If the selection regime is never reached,
the final-epoch fallback is labeled explicitly. Short smoke runs are marked through
`final_policy_trained`; they do not establish accuracy. `evaluation.detection_loss`
names the selected split's loss; `val_loss` is retained as a compatibility alias.

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
standard deviation, with per-metric contributing counts). Aggregates separate
variant, evaluation split/scope, protocol, and protocol details, preventing test
and validation-only results or incompatible metric definitions from being pooled.
Old result files are labeled validation-only/reference-VOC; unavailable historical
effort and size metrics remain null. The collector reports AP50, AP50:95, size AP,
and measured training effort. It does not report BCa intervals or deployment FPS.

`evaluation.protocol: reference_voc` is the default. It retains the existing
all-point interpolated reference evaluator and returns AP50 and the mean over
IoUs 0.50:0.05:0.95, both in [0,1]. It is not the official COCO evaluator; size AP
is null. `evaluation.protocol: coco_bbox` requires `pip install pycocotools` and
uses official COCOeval matching, 101 recall samples, standard area bins, and
maximum detections [1,10,100]. Both paths share the decoded predictions (confidence
threshold 0.001, NMS IoU 0.65, at most 300 predictions before COCOeval's limit).
For COCO evaluation, predictions and targets are restored from letterboxed input
coordinates to original-image pixels using `ori_shape` and `ratio_pad`; missing
geometry fails explicitly. Size AP therefore uses original-image bbox area.
Undefined size bins remain null. These are converted ordinary YOLO bounding-box
annotations: original COCO crowd/ignore flags and segmentation areas are absent.
The metrics are labeled `coco_bbox`, not a reproduction of a full-annotation COCO
benchmark score. Matching the original dataset and annotation protocol still
requires the original evaluation assets.
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
