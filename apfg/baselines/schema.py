"""Frozen shared tensor contract for Track A baseline development."""
OBS_DIM=32
CAND_DIM=24
MAX_CANDIDATES=24
TYPES=('GRASP','ACTIVE','PUSH','FLIP')
SCHEMA_VERSION='baseline_candidate_v1'
# Policy state: obs[32], candidates[24,24], mask[24]. A transition NPZ has
# obs/candidates/mask/action/reward/next_obs/next_candidates/next_mask/done.
# Manifest rows provide transition_path, episode_id, scene_id, split,
# success, method, demonstrator_source, object_id, seed, action_type,
# step_index, data_sha256. Only NPZ numeric sensor-derived fields enter models.
