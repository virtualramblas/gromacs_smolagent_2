"""
Synthetic GROMACS log snippets used across Phase 4B tests.
Each constant mirrors the exact format GROMACS writes to .log files.
Kept as module-level strings so tests stay readable.
"""

# ---------------------------------------------------------------------------
# EM logs
# ---------------------------------------------------------------------------

EM_CONVERGED_LOG = """\
                      Step           Time
                         0        0.00000

   Energies (kJ/mol)
           Bond          Angle    Proper Dih.  Improper Dih.
    1.23456e+03    2.34567e+02    1.23456e+01    0.00000e+00

   Potential        Kinetic En.   Total Energy  Temperature
  -4.56789e+05    0.00000e+00   -4.56789e+05    0.00000e+00

Steepest Descents converged to Fmax < 1000 in 847 steps
Potential Energy  = -4.56789e+05
Maximum force     =  9.87654e+02

GROMACS reminds you: "Keep computing!" 

Performance:
   Mnbf/s    GFlops  ns/day   hour/ns
  1234.56    567.89   12.34     1.94
"""

EM_NOT_CONVERGED_LOG = """\
                      Step           Time
                      1000        0.00000

   Potential        Kinetic En.   Total Energy
  -2.34567e+05    0.00000e+00   -2.34567e+05

Steepest Descents did not converge to Fmax < 1000 in 1000 steps
Potential Energy  = -2.34567e+05
Maximum force     =  1.52340e+04

integrator              = steep
nsteps                  = 1000
"""

EM_EXPLODED_LOG = """\
                      Step           Time
                         0        0.00000

   Potential        Kinetic En.   Total Energy
    2.34567e+07    0.00000e+00    2.34567e+07

Steepest Descents did not converge to Fmax < 1000 in 100 steps
Potential Energy  =  2.34567e+07
Maximum force     =  9.87654e+08

integrator              = steep
nsteps                  = 100
"""

EM_NAN_LOG = """\
                      Step           Time
                        10        0.00000

   Potential        Kinetic En.   Total Energy
          nan    0.00000e+00           nan

Fatal error:
NaN detected in the force on atom 42

integrator              = steep
nsteps                  = 5000
"""

EM_LINCS_ERROR_LOG = """\
                      Step           Time
                       100        0.20000

LINCS ERROR
Too many LINCS warnings (1000)

Fatal error:
Too many LINCS warnings

integrator              = steep
nsteps                  = 5000
"""

# ---------------------------------------------------------------------------
# NVT logs
# ---------------------------------------------------------------------------

NVT_STABLE_LOG = """\
title                   = NVT Equilibration
integrator              = md
nsteps                  = 50000
ref-t                   = 300

                      Step           Time
                      1000        2.00000

   Energies (kJ/mol)
   Potential        Kinetic En.   Total Energy  Temperature
  -4.56789e+05    1.23456e+04   -4.44443e+05   299.87

                      Step           Time
                     25000       50.00000

   Potential        Kinetic En.   Total Energy  Temperature
  -4.57123e+05    1.23789e+04   -4.44744e+05   300.12

                      Step           Time
                     50000      100.00000

   Potential        Kinetic En.   Total Energy  Temperature
  -4.57234e+05    1.23901e+04   -4.44844e+05   300.05

Performance:
   Mnbf/s    GFlops  ns/day   hour/ns
  1234.56    567.89   45.67     0.53
"""

NVT_TEMP_UNSTABLE_LOG = """\
title                   = NVT Equilibration
integrator              = md
nsteps                  = 50000
ref-t                   = 300

                      Step           Time
                      1000        2.00000

   Temperature
   250.12

                      Step           Time
                     25000       50.00000

   Temperature
   340.87

                      Step           Time
                     50000      100.00000

   Temperature
   280.33
"""

# ---------------------------------------------------------------------------
# NPT logs
# ---------------------------------------------------------------------------

NPT_STABLE_LOG = """\
title                   = NPT Equilibration
integrator              = md
nsteps                  = 50000
ref-t                   = 300

                      Step           Time
                     50000      100.00000

   Potential        Kinetic En.   Total Energy  Temperature
  -4.57234e+05    1.23901e+04   -4.44844e+05   300.05

   Pressure (bar)
    1.023

Performance:
   Mnbf/s    GFlops  ns/day   hour/ns
  1234.56    567.89   45.67     0.53
"""

# ---------------------------------------------------------------------------
# Error / warning logs
# ---------------------------------------------------------------------------

SETTLE_ERROR_LOG = """\
integrator              = md
nsteps                  = 50000

                      Step           Time
                       500        1.00000

SETTLE: can't settle atom 1234 of molecule 456
SETTLE error

Fatal error:
Too many SETTLE warnings
"""

PARTICLE_ESCAPED_LOG = """\
integrator              = md
nsteps                  = 50000

                      Step           Time
                      2000        4.00000

Fatal error:
1 particles are outside of the box
particle 4521 flew away
"""

GPU_ERROR_LOG = """\
integrator              = md
nsteps                  = 50000

GPU error: CUDA error on device 0
Fatal error:
GPU error detected
"""

MPI_ERROR_LOG = """\
integrator              = md
nsteps                  = 50000

MPI error: rank 0 failed
Fatal error:
MPI error detected
"""

DISK_FULL_LOG = """\
integrator              = md
nsteps                  = 500000

Fatal error:
No space left on device
write error on file md.xtc
"""

MISSING_PARAMS_LOG = """\
integrator              = steep
nsteps                  = 5000

Fatal error:
No default Lennard-Jones parameter for combination
missing parameters for atom type LIG
"""

CHARGE_WARNING_LOG = """\
integrator              = steep
nsteps                  = 5000

WARNING: The net charge of your system is not zero (net charge = 4e)
Total charge of the system is not zero
"""

LINCS_WARNING_LOG = """\
integrator              = md
nsteps                  = 50000

LINCS WARNING: bond angle rotation too large
lincs_warnangle exceeded
"""

TOPOLOGY_ERROR_LOG = """\
integrator              = steep
nsteps                  = 5000

Fatal error:
topology error: inconsistent topology detected
"""

PERFORMANCE_LOG = """\
integrator              = md
nsteps                  = 500000

Performance:
   Mnbf/s    GFlops  ns/day   hour/ns
  2345.67    678.90   78.90     0.30
"""

DRIFT_LOG = """\
title                   = Production MD
integrator              = md
nsteps                  = 500000

Total energy drift = 25.34 kJ/mol/ps
"""

MULTI_WARNING_LOG = """\
integrator              = steep
nsteps                  = 5000

WARNING: first warning here
WARNING: second warning here
WARNING: third warning here
NOTE: this is a note
NOTE: another note
"""