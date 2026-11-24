# Baseline implementation and measured execution matrix

Generated from actual finalized campaigns/trainer results by `scripts/collect_baseline_evidence.py`. Counts include failed and zero-action episodes; checkpoint existence alone is not training PASS.

| Method | Implementation identity | Core code | Actual training chain | Heldout physical retrieval |
|---|---|---|---|---|
| Grasp Only | FROM_SCRATCH selector / reused GraspNet | `apfg/baselines/heuristic.py` | NOT_RUN (no learned parameters) | 1/2; FAILED |
| Heuristic | FROM_SCRATCH | `apfg/baselines/heuristic.py` | NOT_RUN (no learned parameters) | 1/2; FAILED |
| Random valid | FROM_SCRATCH | `apfg/baselines/heuristic.py` | NOT_RUN (no learned parameters) | 0/2; FAILED |
| Candidate Double DQN | FROM_SCRATCH; not SB3 fixed-slot DQN | `apfg/baselines/dqn.py` | PASS | 2/2; PASS |
| Candidate BC | FROM_SCRATCH | `apfg/baselines/bc.py` | PASS | 1/2; FAILED |
| High-level Diffusion Policy | ADAPTED official U-Net + local DDPM | `apfg/baselines/dp_adapted.py` | PASS | 0/2; FAILED |

Official DP network source: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy), commit `5ba07ac6661db573af695b419a7947ecb704690f`; source/license records are in `apfg/baselines/_official_dp/`. Only the stated network is adapted. Observation/action representation, candidate projection, scheduler and training data differ from native DP.

| Native published method | Execution identity/status | Reason |
|---|---|---|
| Native Diffusion Policy | NOT_RUN | No native continuous EE/gripper action-chunk training/rollout in this shared candidate experiment. |
| Native DP3 | NOT_RUN | No native synchronized high-frequency point-cloud/continuous-action dataset and executor; high-level DP is not DP3. |
| GAPG / GOMP / MS-MEM / Uni-PNP / SPIN | NOT_RUN | Source/task audit only; see [primary-source review](BASELINE_SOURCE_AUDIT.md). No heuristic impersonation. |

E1 qualification and next-scale readiness are separate from optimizer/checkpoint availability. Actual per-episode outcomes, no-go reasons, losses, hashes and split audits are in `evidence/baselines/summary.json`; full observations, videos, replay and model weights remain local under `outputs/baselines/`. Static candidate agreement is not physical retrieval success.
