# Native MimicGen generation

Robot Stack calls the official NVIDIA `DataGenerator` at commit
[`72bd767`](https://github.com/NVlabs/mimicgen/tree/72bd767c255545f462e7ccfb2731f2e5d4c1d9bb).
The first supported task is **Panda Lift in robosuite 1.5.2**, using a world-frame
OSC controller. This is a task-specific interface; it does not imply MimicGen
support for every simulator adapter.

## Reproduce

Use a separate Linux Python 3.10 environment, because the other benchmarks have
different MuJoCo/NumPy requirements. Install PyTorch for your machine before the
optional extra. Upstream source and assets retain their own licenses.

```bash
pip install -e '.[mimicgen]'
export MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m robosuite_collector.collect --output outputs/lift-source \
  --episodes 2 --seed-start 100 --max-steps 600
python -m robot_stack.mimicgen prepare \
  --sources outputs/lift-source/ep_0100.hdf5 outputs/lift-source/ep_0101.hdf5 \
  --output outputs/mimicgen-source.hdf5
python -m robot_stack.mimicgen generate --source outputs/mimicgen-source.hdf5 \
  --output outputs/mimicgen-lift --episodes 10 --seed-start 400
python -m robot_stack.mimicgen replay \
  --trajectories outputs/mimicgen-lift/*.hdf5 \
  --output outputs/mimicgen-lift/replay.json
```

For video, add `--video-dir outputs/mimicgen-lift/videos` to replay. This requires
EGL and FFmpeg. Video samples every fourth action; state verification still checks
every action.

Preparation physically replays each source from its seeded reset and rejects
state divergence, unsuccessful sources, missing grasp transitions and action/frame
conversion inconsistencies. It derives end-effector and cube poses, controller
target poses, gripper actions and a latched native grasp boundary. Two object-relative
subtasks cover approach/grasp and lift. It never substitutes stored simulator states.

Generation starts from new reset seeds and uses the upstream object-relative
trajectory transformation and interpolation. Every attempted trajectory is retained,
including failures. Each HDF5 has T pre-action states, T actions, a final state,
post-action success flags, source labels and provenance. This is MimicGen's source
layout, distinct from the correction engine's T+1-state layout.

Final success requires native Lift success, two-sided grasp and at least 10 cm of
height gain, held for the final ten control steps. Upstream's *ever succeeded* flag
is recorded separately. A transient success followed by a dropped object is not a
successful output. Independent replay verifies every recorded state with absolute
tolerance 1e-7 and verifies the complete stable-success history.

The current generator records `split_group` (source-dataset SHA256); the first smoke archive predates this field and records `source_sha256` instead. Use these fields, plus source hashes in the prepared dataset,
to keep source demonstrations and their descendants together across train/test splits.
Do not randomly split generated episodes that share source demonstrations.

## Measured scope

The first fixed check used two successful scripted sources (seeds 100–101) and ten
new resets (400–409): **7/10 successful**, three retained failures, zero generation
exceptions. All ten passed independent native action replay. Generation took 29.17
seconds after imports/source loading; this is a small shared-server smoke check,
not a throughput benchmark or an official MimicGen score.

These are augmented demonstrations, **not newly verified correction pairs**.
MimicGen transforms source segments; it does not automatically create and validate
an error/control/recovery triplet. The existing correction protocol remains the
required gate for that label. Extending this interface to recovery data requires
appropriate recovery subtask boundaries; extending it to another task requires its
object frames, action conversion and success predicate.
