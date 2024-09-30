import numpy as np
from pyscf import gto, scf, cc
from ad_afqmc import pyscf_interface

norb_frozen = 1
atomstring = f'''
O   0.000000   0.000000   0.000000
'''

mol = gto.M(atom = atomstring, verbose=4, basis='ccpvdz', spin=2)
mf = scf.RHF(mol)
mf.chkfile = 'mf.chk'
mf.max_cycle = 500
mf.level_shift = 0.4
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()

umf = scf.UHF(mol)
mf.level_shift = 0.4
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
umf.stability()

mycc = cc.CCSD(umf)
mycc.frozen = norb_frozen
mycc.run()
et = mycc.ccsd_t()
print(mycc.e_tot + et)

pyscf_interface.prep_afqmc(mycc)

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc

options = {
    "dt": 0.005,
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 200,
    "n_walkers": 200,
    "walker_type": "rhf",
    "trial": "ucisd"
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

