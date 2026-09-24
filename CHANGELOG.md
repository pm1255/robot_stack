# Changelog

## 0.2.0 — 2026-09-24

- Add a shared three-branch correction collector with action budgets, source provenance, verified prefix reconstruction and physical recovery.
- Verify paired correction collection on five native task instances each for MetaWorld push and pick-place.
- Add schema audit and independent action replay commands; group identical initial states to reduce train/validation leakage.
- Add experimental RoboCasa navigation and AI2-THOR PointNav adapters, plus an explicit native RoboDojo EvalEnv bridge.
- Preserve the earlier robosuite Lift, MetaWorld, ManiSkill and RoboTwin collection backends and their measured limitations.
- Add installable core, CI contracts, contributor guidance, structured issue templates and a static evidence dashboard.

- Verified 100 MetaWorld correction pairs (300 trajectories); task-definition caching preserved all states/actions and reduced observed wall time by 3.4–3.6× on the shared test server.
- Verified three RoboCasa NavigateKitchen correction pairs and independent three-branch action replay; fixed native base-frame control and friction-induced stalling without changing physics.
- Published the GitHub Pages dashboard with real correction videos.
