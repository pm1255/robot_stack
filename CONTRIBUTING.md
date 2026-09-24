# Contributing

A useful contribution adds reproducible evidence, not just another simulator name.

1. Open an issue describing the task, simulator version, action space and native success oracle.
2. Keep simulator imports inside the adapter. The core and its tests must run without a GPU or proprietary runtime.
3. Add a bounded smoke command, fixed seeds, dependency versions and an output schema example.
4. Save failed attempts. Do not search for successful seeds and then report them as an unbiased success rate.
5. Re-execute saved actions before claiming replay support. State-video playback is not dynamics replay.
6. Keep upstream licenses and asset provenance. Do not commit checkpoints, full datasets, private paths or credentials.

## Adding a task

Implement the `Adapter` protocol in `robot_stack/core.py`, or supply an existing native environment through the RoboDojo bridge. Reset, action semantics, numeric state and success belong to the adapter. The collection engine handles branching, budgets, evidence and paired labels. A new task still needs a capable policy and a task-appropriate perturbation; the framework does not synthesize an expert from a task name.

`error_features()` must describe relevant physical state, excluding clocks and counters. Document units and thresholds. `perturbation_actions()` must return executable actions; it must not reset, teleport, attach objects or install a successful state. The state reader must include enough information to justify the claimed replay coverage.

Run `python -m unittest discover -s tests -v`. Include native smoke output in the PR when available; otherwise label the integration as unverified. Keep all descendants of one `source_demo_id` in the same dataset split.
