# 017.22 Code attachments — reproduction

```bash
cd experiments/tlm_fixed_head
uv run python -m unittest test_01721_phi_and_controls -v
uv run python learned_admissibility_01721_structural_controls.py
```

manifest_sha256: `81a4700e8cf1bac0f66e473d144c4ccb003fe26f8cb7175108fe8884effd107e`
verdict: `both`
Pinned dens: `01717_artifacts/checkpoints/seed{0,1,2}_Rec_ray.pt`
Policy budget: 800 steps, lr=0.05, Policy(16→32→2)
Phi adapter applied in tensor forward + exact target construction.
