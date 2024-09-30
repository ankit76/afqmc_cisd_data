import numpy as np
from pyscf import gto, scf, cc
from ad_afqmc import pyscf_interface

norb_frozen = 10
f = 1.
# bond lengths
cu_cu = 2.8 + f * 0.8
o_o = 2.3 - f * 0.9

atomString = f'Cu {-cu_cu/2} 0. 0.; Cu {cu_cu/2} 0. 0.; O 0. {o_o/2} 0.; O 0. {-o_o/2} 0.'

mol = gto.M(atom = atomString,
    basis = {'O': gto.basis.parse('''
O    S
 105374.95                   0.459000000E-04             -0.105000000E-04            0.900000000E-05             -0.109000000E-04
  15679.240                  0.360700000E-03             -0.825000000E-04            0.705000000E-04             -0.822000000E-04
   3534.5447                 0.191980000E-02             -0.441200000E-03            0.375700000E-03             -0.467500000E-03
    987.36516                0.820670000E-02             -0.188640000E-02            0.161460000E-02             -0.184650000E-02
    315.97875                0.297257000E-01             -0.695400000E-02            0.593400000E-02             -0.755850000E-02
    111.65428                0.904558000E-01             -0.217208000E-01            0.187866000E-01             -0.210868000E-01
     42.699451               0.217405400E+00             -0.568513000E-01            0.494683000E-01             -0.667511000E-01
     17.395596               0.368765700E+00             -0.113963500E+00            0.103039900E+00             -0.109367300E+00
      7.4383090              0.337279800E+00             -0.162020100E+00            0.162058600E+00             -0.273143100E+00
      3.2228620              0.967505000E-01             -0.333800000E-01            0.936700000E-03             0.209713700E+00
      1.2538770              0.256740000E-02             0.365506800E+00             -0.822425100E+00            0.120348070E+01
       .49515500             0.137460000E-02             0.552003100E+00             -0.101790200E+00            -0.677469400E+00
       .19166500             -0.141000000E-03            0.223639300E+00             0.425393900E+00             -0.142988400E+01
       .06708300             0.683000000E-04             0.657450000E-02             0.687702800E+00             0.148910680E+01
O    P
    200.00000                0.893300000E-03            -0.838400000E-03            0.126180000E-02
     46.533367               0.736900000E-02            -0.684910000E-02            0.111628000E-01
     14.621809               0.349392000E-01            -0.328505000E-01            0.518316000E-01
      5.3130640              0.115298500E+00            -0.110006000E+00            0.197884500E+00
      2.1025250              0.258323100E+00            -0.313526300E+00            0.570765200E+00
       .85022300             0.369623100E+00            -0.319601100E+00            -0.178929100E+00
       .33759700             0.323878900E+00            0.221724300E+00             -0.898207700E+00
       .12889200             0.146798900E+00            0.562261600E+00             0.266664300E+00
       .04511200             0.336127000E-01            0.301325100E+00             0.625899400E+00
O    D
      3.7500000              0.12849340            -0.21820550
      1.3125000              0.52118840            -0.48176950
       .45937500             0.43457840             0.13575950
       .16078100             0.14574090             0.82977340
    '''),
    'Cu': gto.basis.parse('''
Cu    S
     27.6963200              0.2311320              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     13.5053500             -0.6568110              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      8.8153550             -0.5458750              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      2.3808050              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.9526160              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.1126620              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.0404860              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0100000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Cu    P
     93.5043270              0.0228290              0.0000000              0.0000000              0.0000000              0.0000000
     16.2854640             -1.0095130              0.0000000              0.0000000              0.0000000              0.0000000
      5.9942360              0.0000000              0.2464500              0.0000000              0.0000000              0.0000000
      2.5368750              0.0000000              0.7920240              0.0000000              0.0000000              0.0000000
      0.8979340              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.1317290              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0308780              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Cu    D
     41.2250060              0.0446940              0.0000000              0.0000000
     12.3432500              0.2121060              0.0000000              0.0000000
      4.2019200              0.4534230              0.0000000              0.0000000
      1.3798250              0.5334650              0.0000000              0.0000000
      0.3834530              0.0000000              1.0000000              0.0000000
      0.1000000              0.0000000              0.0000000              1.0000000
    ''')},
    ecp = {'Cu': gto.basis.parse_ecp('''
Cu nelec 10
Cu ul
2       1.000000000            0.000000000
Cu S
2      30.220000000          355.770158000
2      13.190000000           70.865357000
Cu P
2      33.130000000          233.891976000
2      13.220000000           53.947299000
Cu D
2      38.420000000          -31.272165000
2      13.260000000           -2.741104000
    ''')},
    verbose=4, unit='angstrom', symmetry=0, spin=0, charge=2)
# mf = scf.RHF(mol)
# mf.level_shift = 0.5
# mf.kernel()

umf = scf.UHF(mol)
umf.chkfile = 'umf.chk'
umf.level_shift = 0.5
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
umf.stability()

mycc = cc.CCSD(umf)
mycc.frozen = norb_frozen
mycc.max_cycle = 100
mycc.run()
et = mycc.ccsd_t()
print(mycc.e_tot + et)

pyscf_interface.prep_afqmc(mycc, chol_cut=1e-5)

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc

options = {
    "dt": 0.005,
    "n_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 200,
    "n_walkers": 200,
    "walker_type": "rhf",
    "trial": "ucisd"
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

