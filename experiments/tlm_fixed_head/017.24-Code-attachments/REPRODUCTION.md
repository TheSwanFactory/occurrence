# 017.24 Code attachments — reproduction

Pinned manifest SHA256: `81a4700e8cf1bac0f66e473d144c4ccb003fe26f8cb7175108fe8884effd107e`

```bash
cd experiments/tlm_fixed_head
uv run python -m unittest test_01721_phi_and_controls test_01723_final_audit -v
uv run python learned_admissibility_01723_final_audit.py
```

No new training. Uses frozen Rec_ray dens (`01717_artifacts/checkpoints/seed*_Rec_ray.pt`),
saved rewired masks (`01721_artifacts/rewired_masks.json`), and rematerialized exact
benchmark (SHA must match).

Deps: topographo 0.8.2, torch 2.14.0, numpy 2.5.1, networkx 3.6.1. Host: thebeast.

Full per-example GroupedFirst records: `../01723_artifacts/grouped_first_per_example.jsonl`
(sample only in this attachment directory).
