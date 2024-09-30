import sys, os
import numpy as np
from pyscf import gto, scf, cc, lib
import basis_set_exchange as bse
from ad_afqmc import pyscf_interface


atomstring = '''
Fe -0.64147387529051 0.51990405379180 0.11450483185168
O 0.33915564253394 2.18819453520299 -0.84159903476570
H 0.65823736209818 2.99799985286513 -0.40575899522591
O -1.29914529040912 -0.20924466129350 -1.80322550106302
H -2.11419692217693 0.03267722517314 -2.27701147100162
O 0.01237770373627 1.24396717111997 2.03533113813669
H 0.83097217272839 1.00756306178418 2.50577204385612
O -1.62284672685044 -1.15048542622041 1.07007991103536
H -1.94592790838151 -1.95766677806096 0.63237073377438
O -2.41866907277644 1.66373032346654 0.25658821238371
H -2.55381601393967 2.56024091790599 -0.09930153080408
O 1.13404022498401 -0.62623224778001 -0.02436092147139
H 1.26370551457750 -1.52400168391753 0.33034524187694
H -3.28110369794222 1.35873477131317 0.59094893393907
H -0.81032015579443 -0.82248643308581 -2.37966496653734
H 1.99795010138314 -0.32692539271158 -0.36001715946551
H 0.57998322602479 2.26642229799559 -1.78148703231078
H -0.47668197075682 1.85673194452426 2.61208351578631
H -1.85653031374812 -1.23356353207299 2.011352050005
'''

mol = gto.M(atom = atomstring, basis = {
    'default': bse.get_basis('cc-pvtz-dk', fmt='nwchem'),
    'Fe': bse.get_basis('cc-pwcvtz-dk', fmt='nwchem'),
    },
    verbose=4, unit='angstrom', symmetry=0, charge=2, spin=4)

umf = scf.UHF(mol).x2c()
umf.chkfile = 'umf.chk'
umf.level_shift = 0.5
umf.max_cycle = 50
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
umf.stability()
umf.analyze()


norb_frozen = 11
mycc = cc.CCSD(umf)
mycc.frozen = norb_frozen
mycc.run()

pyscf_interface.prep_afqmc(mycc)

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc
options = {
    "dt": 0.005,
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 500,
    "n_walkers": 200,
    "n_batch": 1,
    "walker_type": "rhf",
    "trial": "uhf"
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

