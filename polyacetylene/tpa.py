n = 6
import sys, os
import numpy as np
from pyscf import gto, scf, cc, lib
import basis_set_exchange as bse
from ad_afqmc import pyscf_interface
import time


# bond lengths and angles (assuming uniform c=c, c-c and c-h bond lengths, and all angles to be 120 degrees)
b1 = 1.45
b2 = 1.34
bh = 1.08
t = 2 * np.pi / 3

if n % 2 == 0:
    # vector displacements on one side
    c0 = np.array([b1 / 2, 0.0, 0.0])
    cs = np.array([b1, 0.0, 0.0])
    cd = np.array([b2 / 2, b2 * 3**0.5 / 2, 0.0])
    ch = np.array([bh / 2, -bh * 3**0.5 / 2, 0.0])
    cht = np.array([bh, 0.0, 0.0])

    atomString = "C {} {} {};\nC {} {} {};\n".format(
        c0[0], c0[1], c0[2], -c0[0], -c0[1], -c0[2]
    )
    currentC = c0
    for i in range(n - 1):
        if i % 2 == 0:
            newC = currentC + cd
            newH = currentC + ch
        else:
            newC = currentC + cs
            newH = currentC - ch
        atomString += "C {} {} {};\nC {} {} {};\n".format(
            newC[0], newC[1], newC[2], -newC[0], -newC[1], -newC[2]
        )
        atomString += "H {} {} {};\nH {} {} {};\n".format(
            newH[0], newH[1], newH[2], -newH[0], -newH[1], -newH[2]
        )
        currentC = newC

    # terminal h's
    th1 = currentC + cht
    th2 = currentC - ch
    atomString += "H {} {} {};\nH {} {} {};\n".format(
        th1[0], th1[1], th1[2], -th1[0], -th1[1], -th1[2]
    )
    atomString += "H {} {} {};\nH {} {} {};\n".format(
        th2[0], th2[1], th2[2], -th2[0], -th2[1], -th2[2]
    )

else:
    # vector displacements on one side
    c0 = np.array([b2 / 2, 0.0, 0.0])
    cd = np.array([b2, 0.0, 0.0])
    cs = np.array([b1 / 2, b1 * 3**0.5 / 2, 0.0])
    ch = np.array([bh / 2, -bh * 3**0.5 / 2, 0.0])
    cht = np.array([bh / 2, bh * 3**0.5 / 2, 0.0])

    atomString = "C {} {} {};\nC {} {} {};\n".format(
        c0[0], c0[1], c0[2], -c0[0], -c0[1], -c0[2]
    )
    currentC = c0
    for i in range(n - 1):
        if i % 2 == 0:
            newC = currentC + cs
            newH = currentC + ch
        else:
            newC = currentC + cd
            newH = currentC - ch
        atomString += "C {} {} {};\nC {} {} {};\n".format(
            newC[0], newC[1], newC[2], -newC[0], -newC[1], -newC[2]
        )
        atomString += "H {} {} {};\nH {} {} {};\n".format(
            newH[0], newH[1], newH[2], -newH[0], -newH[1], -newH[2]
        )
        currentC = newC

    # terminal h's
    th1 = currentC + cht
    th2 = currentC + ch
    atomString += "H {} {} {};\nH {} {} {};\n".format(
        th1[0], th1[1], th1[2], -th1[0], -th1[1], -th1[2]
    )
    atomString += "H {} {} {};\nH {} {} {};\n".format(
        th2[0], th2[1], th2[2], -th2[0], -th2[1], -th2[2]
    )

mol = gto.M(
    atom=atomString,
    basis="aug-cc-pvdz",
    unit="angstrom",
    verbose=4,
    max_memory=10000,
)
norb_frozen = 2 * n

mf = scf.RHF(mol)
# chkfile = 'mf.chk'
# mf.__dict__.update(lib.chkfile.load(chkfile, "scf"))
mf.chkfile = 'mf.chk'
mf.level_shift = 0.5
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()
mf.analyze()

start_time = time.time()
mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.run()
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken for CCSD calculation: {time_taken:.2f} seconds")

pyscf_interface.prep_afqmc(mycc)

start_time = time.time()
et = mycc.ccsd_t()
end_time = time.time()
print(f"CCSD(T) energy: {mycc.e_tot + et}")
time_taken = end_time - start_time
print(f"Time taken for CCSD(T) calculation: {time_taken:.2f} seconds")

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc
options = {
    "dt": 0.005,
    "n_eql": 2 * n,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 20 * n,
    "n_walkers": 200,
    "n_batch": 1,
    "walker_type": "rhf",
    "trial": "cisd"
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

