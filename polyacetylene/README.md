# Polyacetylene walltime scaling

Walltime scaling:

```python
times_u = np.loadtxt("times_u.txt")
times_r = np.loadtxt("times_r.txt")
n_units = times_u[:, 0]
nelec = times_u[:, 1]
norb = times_u[:, 2]
uccsd_times = times_u[:, 3] / 60
mask_uccsd = uccsd_times != 0
uccsdpt_times = times_u[:, 4] / 60
mask_uccsdpt = uccsdpt_times != 0
afqmc_ucisd_times = times_u[:, 5] / 60
mask_afqmc_ucisd = afqmc_ucisd_times != 0
ccsd_times = times_r[:, 3] / 60
mask_ccsd = ccsd_times != 0
ccsdpt_times = times_r[:, 4] / 60
mask_ccsdpt = ccsdpt_times != 0
afqmc_cisd_times = times_r[:, 5] / 60
mask_afqmc_cisd = afqmc_cisd_times != 0
```

NB: `n_ene_blocks` in the following line ([appears here](https://github.com/ankit76/ad_afqmc/blob/f8e16c95c036545703383f824b90f7a88e5b8c73/ad_afqmc/driver.py#L91)) was set to 1 during these calculations:

```python
sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)
```

Unfortunately, there is currently no input option for doing this.
