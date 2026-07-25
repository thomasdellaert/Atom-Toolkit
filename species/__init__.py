"""
Species library: one module per isotope, each building a concrete :class:`Atom`.

Run any of these as ``__main__`` to regenerate its ``.json`` snapshot alongside it::

    cd species && python Ca_II_43.py

Every module loads energy levels from a cached NIST ASD CSV in ``resources/`` (so the
build is reproducible offline), then layers hyperfine coefficients, nuclear g-factors
and measured transition data on top, with sources cited inline.

Available species
-----------------
Trapped ions, alkaline-earth-like (valence s electron over a closed shell):
    Be_II_9      I=3/2   Mg_II_24    I=0     Mg_II_25    I=5/2
    Ca_II_40     I=0     Ca_II_43    I=7/2
    Sr_II_87     I=9/2   Sr_II_88    I=0
    Ba_II_133    I=1/2   Ba_II_137   I=3/2   Ba_II_138   I=0
    Ra_II_225    I=1/2
    Yb_II_171    I=1/2   Yb_II_173   I=5/2   Yb_II_174   I=0

Trapped ions, group 12 (d-hole states above the ground state rather than below the P levels):
    Zn_II_64     I=0     Zn_II_67    I=5/2
    Cd_II_111    I=1/2
    Hg_II_199    I=1/2   Hg_II_202   I=0

Alkali metals (neutral):
    Li_I (7Li)   I=3/2   Li_I_6      I=1
    Na_I_23      I=3/2
    K_I_39       I=3/2   K_I_40      I=4     K_I_41      I=3/2
    Rb_I_85      I=5/2   Rb_I_87     I=3/2
    Cs_I_133     I=7/2
    Fr_I_211     I=9/2

MOT-trapped neutrals, alkaline-earth-like (1S0 ground state, two-stage broad/narrow MOT):
    Mg_I_24      I=0     Ca_I_40     I=0
    Sr_I_88      I=0     Sr_I_87     I=9/2   -- 698 nm lattice clock
    Yb_I_174     I=0     Yb_I_171    I=1/2   -- 578 nm lattice clock, tweezer qubit
                         Yb_I_173    I=5/2
    Hg_I_202     I=0     Hg_I_199    I=1/2
    Cd_I_112     I=0

MOT-trapped neutrals, strongly magnetic (dipolar quantum gases):
    Dy_I_164     I=0     Dy_I_161    I=5/2   -- mu = 10 mu_B, the most magnetic neutral atom
    Er_I_166     I=0     Er_I_167    I=7/2   -- mu = 7 mu_B
    Cr_I_52      I=0     -- mu = 6 mu_B, the first dipolar BEC
    Tm_I_169     I=1/2   -- 1.14 um inner-shell clock transition

MOT-trapped metastable noble gases (NOTE: the trapped state is a metastable *excited*
state, not the ground state -- reference it explicitly, e.g. He4.levels['1s.2s 3S1']):
    He_I_4       I=0     -- metastable 1s.2s 3S1, cooled at 1083 nm
    Ne_I_20      I=0     -- metastable 3s 2[3/2]*2, cooled at 640 nm

Other:
    Tm_II_169    I=1/2   -- lanthanide ion with a transition list (the *ion*; see Tm_I_169
                            for the MOT-trapped neutral)
    Ho_I_165     I=7/2   -- deliberately difficult: J=15/2 ground state, 128 ground-manifold
                            Zeeman sublevels, JK/JJ-coupled excited configurations, and a
                            hyperfine B larger than its A. Use it as a stress test.

Conventions and caveats
-----------------------
* **Short names.** Each module exposes its Atom as element+mass (``Sr88``, ``Dy164``). Where a
  neutral and an ion of the *same isotope* both exist, the neutral takes a trailing ``n`` to
  keep the two apart: ``Sr_II_87.Sr87`` is the ion, ``Sr_I_87.Sr87n`` the neutral. Import the
  module rather than the bare name and the distinction never bites you.
* **Forbidden clock transitions.** ``1S0 -> 3P*0`` is J=0 -> J=0, forbidden for every
  multipole, and only weakly allowed in odd isotopes via hyperfine mixing. It is therefore
  never auto-populated and is added by hand in Sr_I_87 and Yb_I_171, where its
  ``allowed_types`` correctly reads all-False. The same applies to the ions' E2 clock lines,
  which are forbidden only at E1 and so also need adding explicitly.
* ``atom.gI`` is the *signed nuclear g-factor in nuclear magneton units*. The alkalis use
  measured, diamagnetically shielded atomic g_I values (converted from the Bohr-magneton
  convention used by Steck's alkali data documents). The ions mostly use the bare
  ``mu_I / I`` ratio, which is typically ~0.5% larger; that is enough to move a magic
  field by a few mG, so replace it with a shielded value before doing precision work.
* Levels whose term symbols the parser cannot handle (deeply nested JJ/JK couplings in
  Hg II and Ho I) are skipped at build time with a warning rather than aborting the build.
* ``ALLOWED_TRANSITIONS = (True, False, False)`` populates E1 transitions only, so
  quadrupole clock transitions (Ca+ 729 nm, Sr+ 674 nm, Hg+ 282 nm) are added explicitly.
* Several files carry a "Data still needed" block listing hyperfine coefficients that are
  not yet filled in.
"""

# TODO: Look more deeply into namespace packaging and see if it's possible to include this
#  package in the namespace of atomtoolkit in a nice way

# TODO: A registry so callers don't need to know about file paths --
#  atomtoolkit.available_species() -> list of names
#  atomtoolkit.load_species('Yb II 171') -> Atom, loaded from the prebuilt JSON snapshot
#  See the matching TODO in atomtoolkit/IO.py.

# TODO: Maybe if you try to import something that doesn't exist it could just throw a warning
#  and try to do a quick NIST import?
