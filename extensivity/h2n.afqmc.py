from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc
options = {
    "dt": 0.005,
    "n_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 400,
    "n_walkers": 400,
    "n_batch": 2,
    "walker_type": "rhf",
    "trial": "rhf"
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

