# Occurrence Theory

## An Orientation of Sedenion Settlement Dynamics

**E. N. Prabhakar** ([Radical Centrism](https://radicalcentrism.org) / [iHack.us](https://ihack.us))
with **Bench d'Claude** (Anthropic) and **Précis d'ChatGPT** (OpenAI)

**Ledger notation.** Every substantive claim carries exactly one tag:
**[T]** theorem (proof from stated identities); **[C]** computation (exact numerical certificate, threshold 10⁻¹²); **[M]** measurement (Monte Carlo, with error bars); **[I]** interpretation; **[X]** conjecture. Claims without tags are definitions or conventions.

---

## 1. Abstract

**Sedenion Settlement Dynamics (SSD)** is the completely positive, unital, trace-preserving map

> Φ(X) = ∫_Σ L_zᵀ X L_z dμ(z)

on End(ℝ¹⁶), where Σ is the [unit zero-divisor variety](https://en.wikipedia.org/wiki/Zero_divisor) of the [sedenion algebra](https://en.wikipedia.org/wiki/Sedenion) 𝕊 and μ its unique Aut(𝕊)-invariant probability measure. SSD is the primary mathematical object of this paper. Main results on SSD: **(1)** For *every* Aut-invariant measure on Σ, ∫ L_zᵀL_z dμ = I — equilibrium is forced, not chosen [T]. **(2)** Every multiplication-built operator of arity ≤ 2 is antisymmetric, every autonomous polynomial self-map of 𝕊 preserves a complex line, and the [associative envelope](https://en.wikipedia.org/wiki/Algebra_(ring_theory)#Subalgebra) of the left multiplications is all of End(ℝ¹⁶); hence 𝕊 admits no internal dynamics beyond [isometries](https://en.wikipedia.org/wiki/Isometry) (**No-Autonomy Theorem**) [T,C]. **(3)** The axis e₈ carries two exact commuting structures: L_{e₈} anticommutes with every settlement generator, giving Φ(L_{e₈}) = −L_{e₈}, an undamped [parity](https://en.wikipedia.org/wiki/Parity_(physics)) [T]; and J = R_{e₈} is an orthogonal [complex structure](https://en.wikipedia.org/wiki/Almost_complex_manifold) under which the transition functionals become [Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix): ‖zx‖² = 1 + τ(z,x) and spine transport = |⟨z,x⟩_ℂ|²/(1+τ) [T]. **(4)** On the 84-element basic crack, the spectrum of Φ is exactly (1/7)·{0, ±1, ±3, ±2√3, ±7} with [G₂-multiplicities](https://en.wikipedia.org/wiki/G2_(mathematics)) [C].

**Occurrence Theory (OT)** is obtained by adjoining one external structure to SSD: the assignment of operational roles (one slot retained and carried forward; the other sampled i.i.d. from (Σ, μ)). We prove this orientation is external (not derivable from the algebra alone, by No-Autonomy), unique up to [gauge](https://en.wikipedia.org/wiki/Gauge_symmetry) (conjugation intertwines left and right), and sufficient for dynamics: with it, SSD exhibits one-step generation amnesia [T], an exact −1/7 flicker in pencil–spine imbalance [T], and a survivor-tilted [stationary measure](https://en.wikipedia.org/wiki/Stationary_distribution) with spine share s\* = 0.13172(5) and quenched [Lyapunov exponent](https://en.wikipedia.org/wiki/Lyapunov_exponent) −0.01773(3) [M].

---

## 2. Motivation and Structure

**This paper contains two independent mathematical contributions.** The first is the theory of Sedenion Settlement Dynamics (SSD): the complete algebraic and spectral characterization of the unique Aut(𝕊)-invariant channel on the sedenion zero-divisor crack. The second is the definition of Occurrence Theory (OT): the oriented dynamical framework obtained by adjoining one external choice to SSD. **Acceptance of the second contribution is not required for the correctness of the first.** The two stand independently, and readers interested only in the algebra of SSD need not engage with the interpretations of OT.

**Sedenion Settlement Dynamics** is the unoriented object: an algebra, a distinguished singular orbit inside it, a measure forced by symmetry, and the channel these determine. Every statement in this layer is theorem or exact computation. Nothing in it refers to time, events, or states.

**Occurrence Theory** is SSD equipped with one additional structure — an *orientation* assigning the two slots of multiplication the operational roles of event (sampled) and state (retained). Section 4 proves this orientation is genuinely external: the algebra cannot supply it (No-Autonomy), cannot detect it (role symmetry), and cannot do without it if anything is to evolve. OT is therefore a *definition on top of a discovery*, and the boundary between the two is theorem-shaped.

**Protology (CORE)** — Contrast, Orientation, Re-Entry (Prabhakar 2026) — is a prior, independently developed generative language. Section 8 tests whether SSD realizes it, maintaining restraint about the depth of the correspondence.

The discipline throughout: *discovery* is what the algebra forces; *orientation* is the minimal choice we add; everything else is interpretation and is labeled as such.

---

## 3. Sedenion Settlement Dynamics

**Definition 3.1.** Let 𝕊 = A₄ be the 16-dimensional Cayley–Dickson algebra over ℝ (γ = −1 at each doubling), with basis e₀,…,e₁₅, positive-definite form ⟨·,·⟩, conjugation x̄ = 2⟨x,e₀⟩e₀ − x, and left/right multiplication operators L_x, R_x. Call x *pure* if ⟨x,e₀⟩ = 0. Write M_x := L_xᵀL_x and T_x := L_{x²} − L_x² (the **alternator**).

**Theorem 3.2 (structure; Eakin–Sathaye 1990, verified).** Aut(𝕊) = G₂ × S₃, acting with isotypic decomposition 𝕊 = 1 ⊕ 1′ ⊕ (7 ⊗ 2): the identity line ℝe₀, the axis line ℝe₈ (carrying the sign character of S₃), and the 14-dimensional pencil W. Consequences [T]: 𝕊 has exactly eight Aut-invariant subspaces; ℝe₈ is the unique invariant line other than ℝe₀; the spine S = span{e₀, e₈} is the unique invariant plane. S is a subalgebra isomorphic to ℂ; ψ (the generator of S₃/⟨τ⟩) restricts to S as complex conjugation, giving the exact sequence 1 → ⟨τ⟩ → S₃ → Gal(ℂ/ℝ) → 1.

**Theorem 3.3 (canonical ℂ, uniqueness).** [T] S is the unique Aut-invariant complex subalgebra. *Proof.* If ℝu (u pure) is invariant, connectedness of G₂ forces G₂·u = u; writing u = βe₈ + w with w ∈ W, invariance forces w to be a G₂-fixed pencil vector, of which there are none; hence w = 0. ∎ By contrast, *copies* of ℂ are generic: span{1,u} ≅ ℂ for every pure u (quadratic identity), an ℝP¹⁴ family with 3-parameter orbit space [T]. Existence is cheap; the canonical copy is unique.

**Definition 3.4 (crack).** Σ := {z ∈ 𝕊 : ‖z‖ = 1, ker L_z ≠ 0}. As a geometric object [T]: Σ is a single Aut(𝕊)-invariant orbit, equal to the homogeneous space G₂/SU(2) (where the SU(2) acts as the stabilizer of a base zero divisor). It is also a single G₂ orbit when viewed without the S₃ generation symmetry. Σ carries a **unique** Aut(𝕊)-invariant Borel probability measure μ (invariance forces block scalars on isotypics 1, 1′, 7⊗2; support on zero divisors and unit norm force the value). *Attribution.* Moreno (1998) proved that the space of **pairs** (z, w) of norm-one sedenions with zw = 0 is homeomorphic to compact G₂ (in his terminology a "zero divisor" is such a pair); each single element z ∈ Σ has a 3-sphere of unit annihilators, so Σ = Moreno's G₂/SU(2); the Stiefel model is developed in Biss–Dugger–Isaksen. For z ∈ Σ, M_z has spectrum {0⁴, 1⁸, 2⁴} and T_z = −P₀ + P₂ (BCDI, verified [C]).

**Theorem 3.5 (master identity).** [T] For every pure unit x: M_x = I + T_x. *Proof.* x² = −e₀ (quadratic identity), so L_{x²} = −I; L_x is antisymmetric (composition-algebra adjoint identity, Eakin–Sathaye eq. (5), verified 0.0 [C]), so L_x² = −M_x. ∎ Corollaries: T_x is symmetric and traceless (Tr M = 16 is the BCDI sum rule); the hull (maximal composition subalgebra through x) is ker T_x; ‖T_x‖ ≤ 1 with equality exactly on Σ; the strain ‖T_x‖ equals 2‖a∧b‖ on the pure-pure sector [C], and mixing axis amplitude β dilutes strain by exactly (1−β²) [C].

**Theorem 3.6 (forced equilibrium).** [T] For *every* Aut-invariant probability ν on Σ: ∫ M_z dν(z) = I. *Proof.* Equivariance forces the integral to be scalar on each isotypic; ⟨e₀, M_z e₀⟩ = ‖z‖² = 1 and ⟨e₈, M_z e₈⟩ = ‖z e₈‖² = 1 pointwise; Tr M_z = 16 forces the pencil scalar to 1. ∎ (Certified 8.9 × 10⁻¹⁶ on the basic crack [C].)

**Definition 3.7 (settlement channel).** Φ(X) := ∫_Σ L_zᵀ X L_z dμ(z). By 3.6, Φ is unital and trace-preserving; it is self-adjoint for the trace pairing and Aut-covariant.

**Theorem 3.8 (No-Autonomy).** (a) [T,C] For pure x, y, the operators L_x, R_x, ad_x, the Jordan operator, and the middle-slot map z ↦ (x, z, y) are exactly antisymmetric (symmetric parts 0.0). Hence every internally generated continuous flow is isometric. (b) [T] Every autonomous polynomial self-map of 𝕊 preserves the complex line span{e₀, x} (quadratic identity). (c) [C] The associative envelope of {L_x : x pure} is End(ℝ⁸) for 𝕆 (dim 64) and End(ℝ¹⁶) for 𝕊 (dim 256); consequently the normalized trace is the unique envelope-invariant state. (d) [C] The Lie algebra generated by {L_x : x pure} is 𝔰𝔬(8) for 𝕆 and all of 𝔰𝔬(16) for 𝕊: the continuous ledger is maximal and structureless, and the trace is also its unique invariant state.

> **CONSEQUENCE (The mathematical necessity of external orientation):**
>
> **𝕊 contains algebraically complete pre-dynamics — states (the space of rays), isometric clocks (via L_x), an irreversibility locus (ker L_z, supported on Σ), a cost metric (norm dissipation), and conservation laws (Tr M = 16) — yet *provably cannot select* among its own internal morphisms.** This is not a gap in the algebra; it is a theorem about the algebra's indifference. The algebra is indifferent between left and right, between event and state, between retained and sampled. Different applications impose different orientations according to their own semantics, but 𝕊 itself cannot. ***The algebra is not incomplete, but overly complete.*** The selection has to come from outside, and this fact is itself a theorem, not an assumption.

**Theorem 3.9 (axis pair).** [T] (a) J := R_{e₈} satisfies J² = −I on all of 𝕊, Jᵀ = −J, and [L_{e₈}, R_{e₈}] = 0 (all 0.0 [C]). Thus 𝕊 ≅ ℂ⁸ with S = ℂ¹ and W = ℂ⁷; the pencil's 2-plane fibers are the complex lines. (b) For every z ∈ Σ: {L_{e₈}, L_z} = 0 (hulls contain the spine; Clifford relations are hull-exact, defect 1.3 × 10⁻¹⁵ [C]); hence L_zᵀ L_{e₈} L_z = −L_{e₈} M_z and

> **Φ(L_{e₈}) = −L_{e₈}**  (exact −1 eigenvalue, multiplicity 1).

(c) No Aut-equivariant complex structure on 𝕊 exists (the commutant is ℝ³, which contains no antisymmetric element); {J, −J} is the unique conjugate pair equivariant under the index-2 subgroup G₂ × ⟨τ⟩, with ψJψ = −J [T]. The sign of e₈ — equivalently the choice within {J, −J}, equivalently the choice of i in the canonical ℂ — is a ℤ₂ gauge that no invariant construction can fix.

**Theorem 3.10 (transition functionals).** [T] For unit pure x and z ∈ Σ, define the **event-strain** τ(z,x) := ⟨x, T_z x⟩ and the **Hermitian overlap** A(z,x) := |⟨z,x⟩_ℂ|² = ⟨z,x⟩² + ⟨Jz,x⟩². Then:

1. ‖zx‖² = 1 + τ(z,x) (the amplitude ledger *is* event-strain);
2. A_z := L_zᵀ P_S L_z = zzᵀ + (Jz)(Jz)ᵀ, with ∫ A_z dμ = P_W/7;
3. hull pullback: L_zᵀ H_z L_z = I − T_z² (hull overlap is co-strain);
4. spine transport: ⟨zx,e₀⟩² + ⟨zx,e₈⟩² = A(z,x), hence the normalized spine share obeys s′ = A(z,x)/(1 + τ(z,x));
5. E_μ[τ(z,·)] ≡ 0, so all norm dissipation is Jensen curvature on event-strain fluctuations; Var over (uniform x, μ) = 1/18 exactly;
6. A and τ are supported on complementary M_z-eigenspaces (a complex line in Eig₁ versus Eig₀ ⊕ Eig₂) and are exchange-symmetric: τ(z,x) = τ(x,z), A(z,x) = A(x,z), since conj(zx) = xz [T].

**Theorem 3.11 (ledger dictionary).** [C, calibrated against exact Schafer derivations] The conjugation-functorial subalgebra K := {K ∈ 𝔰𝔬(2ⁿ) : [K, span L] ⊆ span L} is 𝔰𝔭𝔦𝔫(7) (dim 21) for 𝕆 — the classical spinor characterization — and collapses for 𝕊 to **𝔤₂ ⊕ ℝK\***, dim 15, where K\* = J_W + 3 J_S (uniform pencil rotation plus triple-speed spine rotation), with companion law [K\*, L_x] = L_{−2J_W x} and **τ = exp((2π/3)K\*)**: the order-3 automorphism is one point of a 3:1 resonance circle that intertwines the two ledgers at every angle but is a symmetry of the algebra only at the cube roots of unity. *Identification* [C]: exp((2π/3)K\*) equals the classical ζ-automorphism (identity on the spine, right multiplication of the pencil by the cube root ζ = −½ + (√3/2)e₈; Brown 1967, Eakin–Sathaye), verified 1.3 × 10⁻¹⁵; Brown's generator is the cube-root point of the K\* circle.

**Theorem 3.12 (spectrum).** [C] On the 84-element basic crack (the finite set of basis-form zero divisors), the channel spectrum is exactly:
symmetric sector (dim 136): {1 (×1), 3/7 (×7), 0 (×72), −1/7 (×42), −3/7 (×14)};
antisymmetric sector (dim 120): {±2√3/7 (×14 each), 3/7 (×14), −3/7 (×7), ±1/7 (×42), 0 (×28), −1 (×1)}.
Every eigenvalue lies in (1/7)·{0, ±1, ±3, ±2√3, ±7}; every multiplicity is a G₂-representation dimension; the sole −1 is Theorem 3.9(b)'s parity, and 2√3 is the slope constant of the triad closure cubic. By Theorem 3.13, this is exactly the spectrum of the continuum channel Φ.

**Theorem 3.13 (design).** [T] The channel induced by *any* probability measure ν on Σ depends on ν only through its second moment ∫ zzᵀ dν. *Proof.* z ↦ L_z is linear, so L_zᵀXL_z = Σᵢⱼ zᵢzⱼ LᵢᵀXLⱼ is quadratic in z; integrating gives Φ_ν(X) = Σᵢⱼ (∫zᵢzⱼ dν) LᵢᵀXLⱼ. ∎ Both μ and the uniform measure on the 84 basic zero divisors have second moment P_W/14 (for μ: invariance forces block scalars, support ⊥ spine and unit norm force the value; for the 84: direct computation, certified 2.4 × 10⁻¹⁸). Hence Φ₈₄ = Φ_μ exactly — full superoperator, spectrum, and multiplicities — and the design property observed throughout the session is explained. The theorem is stronger than the conjecture it replaces: exact representation of the channel requires matching only 16 × 16 moments, not the measure.

---

## 4. Orientation

**Definition 4.1.** An **orientation** of SSD is the designation of one argument of the product as *retained* (carried recursively forward) and the other as *sampled* (drawn i.i.d. from (Σ, μ)). The oriented process on rays is x_{t+1} = z_t x_t / ‖z_t x_t‖, z_t ∼ μ.

**Philosophical note.** The orientation is not an additional algebraic operation. It is an assignment of operational roles to an already existing algebraic product — a choice of *interpretation*, not a choice of *mathematics*. This is the precise distinction between discovery (SSD) and definition (OT).

**What is added.** One bit. Nothing else: the support Σ is nominated by the algebra (the unique singular orbit — the only home of non-invertibility), the measure μ is unique, the equilibrium is forced (3.6), the cost metric and both ledgers are internal.

**What is not added.** A left/right choice — **Proposition 4.2** [T]: conjugation intertwines the left-slot and right-slot oriented chains exactly (x ↦ x̄ maps one to the other, ray-level, isometrically), so slot-handedness is gauge. Nor is a rate necessarily added: on the relational reading (only event order is physical), the theory has zero free constants; at most one dimensionless coupling ν (events per K\*-revolution) can exist, since the crack owns exactly one canonical continuous clock (3.11).

**Why the bit is external.** Theorem 3.8: the algebra's own morphisms cannot select (all internal flows are clocks; all self-application collapses to a complex line; a constant event is a clock). Theorem 3.10(6): the algebra cannot even *see* the roles — every invariant transition functional is exchange-symmetric, and the two possible outcomes of a transition are conjugates, distinguishable only through conjugation-odd observables, i.e., through the same single ℤ₂ gauge as the sign of i. The distinction is therefore neither derivable nor detectable internally, and (3.8) indispensable externally.

**Why this does not diminish the mathematics.** The boundary of the theory is itself a theorem. SSD proves precisely where its own competence ends: it supplies everything about a transition except that one shall occur to *this* argument rather than *that* one. A framework that can prove the shape of its missing piece is stronger, not weaker, than one that hides the piece.

---

## 5. Occurrence Theory

**Definition 5.1.** **Occurrence Theory** is the pair (SSD, orientation): the Markov process of Definition 4.1 on the ray space of 𝕊, together with the operator dynamics Φᵗ it induces in the Heisenberg picture.

Forced consequences, with ledger:

**Theorem 5.2 (amnesia).** [T] Φ annihilates the generation-labeling observables: for the fiber observables I₇ ⊗ q (q traceless symmetric on the ℝ² fiber) and for ψ, Φ(·) = 0 exactly (block computation; certified 3.7 × 10⁻¹⁷). The oriented process cannot carry which-frame or which-sheet information across a single event. Corollary: any generation asymmetry in stationarity is initial-conditional, not dynamical.

**Theorem 5.3 (flicker).** [T] On the projector sector span{P_{e₀}, P_{e₈}, P_W}, Φ acts with spectrum {1, 0, −1/7}: Φ(P_{e₀}) = Φ(P_{e₈}) = P_W/14 exactly; the spine–pencil imbalance decays with alternating sign at rate 1/7.

**Theorem 5.4 (parity clock).** [T] By 3.9(b), a pair of states co-evolving under common events has its relative axis phase ⟨y_t, e₈ x_t⟩ exactly negated per event, undamped: the oriented theory contains an internal, algebraic, event-counting clock (settlement count mod 2 is observable).

**Theorem 5.5 (occupancy balance).** [T] The involution z ↦ ze₈ preserves Σ, μ, and M_z pointwise (since ab = −ba for the orthonormal pure pair); hence the stationary occupancies of e₀ and e₈ are exactly equal, despite the two lines lying in inequivalent S₃-representations.

**Measurement 5.6 (stationary constants).** [M] The stationary (Furstenberg) measure of the oriented chain is tilted: spine share s\* = 0.13172 ± 0.00005 (three seeds; 84-design chain agrees), versus 1/8 for the uniform state. Quenched Lyapunov exponent λ_q = −0.01773 ± 0.00003; annealed exponent exactly 0 (by 3.6). Exact enrichment factorization [T backbone, M values]: s\* = E[A]·E[1/(1+τ)] + Cov = 0.1246 × 1.0794 − 0.0023 — the Hermitian-overlap mean alone *depletes* (0.1246 < 1/8); the Jensen "hull-protection" factor does all enriching; event-strain statistics anti-predict the tilt (τ-only reweighting gives 0.097). On the discrete crack, trajectories die (exact annihilation) at rate 1.0 × 10⁻³ per step; the survival-conditioned ensemble matches the continuum [M].

**Measurement 5.7 (elastic law).** [M] For events of fixed strain σ: λ_q ≈ −0.016 σ², fluctuation v ≈ 0.14 σ, twin-synchronization time finite for σ > 0. Both generativity functionals tested (v²τ and v²τ/|λ_q|) are monotone on (0,1]: the strain interior is a pure tradeoff with no distinguished point; the algebra's only distinguished strains are its endpoints (the alternative cone and Σ).

**Interpretations.** [I] Reading L_{e₈} as "time," S₃ as "generations," the tilt as "survivorship," s′ = |⟨z,x⟩_ℂ|²/(1+T) as Born-shaped — all are interpretations. They are recorded and priced at zero.

**Conjectures.** [X] s\* = 1/8 + 1/(3·7²) = 0.131803 (compatible at 1.7σ; competitors 19/144 and 1/8 + 1/140 excluded). λ_q has no closed form (Furstenberg genericity).

---

## 6. Principal Results (compressed to invariants)

Everything above derives from three invariants and one bit.

1. **The alternator T** (with M = I + T). Geometry is its kernel (hulls; Clifford holds there exactly and only there); equilibrium is its tracelessness; cost is ½log of its spectral radius; the crack is its saturation locus, one orbit; the ledger is literally 1 + T(z,x); dissipation is Jensen on its fluctuations, uniform variance 1/18.

2. **The axis pair (L_{e₈}, R_{e₈})** — one element acting from both sides, commuting. The left action anticommutes with every settlement: the exact parity (−1 eigenvalue, undamped clock). The right action is the complex structure: canonical ℂ with ψ as its Galois conjugation, fibers as complex lines, alignment as Hermitian overlap, spine transport as |⟨z,x⟩_ℂ|²/(1+T). The residual continuous dictionary between ledgers is the 3:1 resonance circle K\* = J_W + 3J_S through τ. The one internal gauge freedom of the entire theory is the sign of this element.

3. **The channel Φ** — the unique Aut-invariant CP map on the unique singular orbit. Its spectrum, in sevenths and 2√3/7 with G₂-multiplicities, contains the whole relaxation theory: amnesia (0's), flicker (−1/7), population vs. coherence rates (3/7 vs. 2√3/7), and the parity (−1).

4. **The bit.** Retained versus sampled. External by theorem, gauge-reduced to one choice, sufficient for dynamics.

---

## 7. Relationship to Existing Mathematics

**Random matrix products.** The oriented nonlinear layer *is* a Furstenberg system: normalized products of i.i.d. elements of {L_z} (Furstenberg 1963; Bougerol–Lacroix 1985). Its qualitative behavior — non-uniform stationary measure, quenched < annealed, intermittency — is classical in kind, and we claim no novelty there. The difference: the matrix family is a single compact homogeneous orbit of structured operators, making the *linear* layer exactly solvable and the constants candidates for closed form.

**Markov processes / quantum channels.** Φ is a real completely positive map (in the sense of real matrix algebras: a map preserving the cone of positive semidefinite matrices) with Kraus family the crack; it is unital, trace-preserving, and self-adjoint for the trace pairing. Spectral-gap and mixing vocabulary applies verbatim. **Novelty:** SSD is not isomorphic-under-renaming to any existing construction. The novelty is the coexistence of three properties: (1) nonassociative multiplication generated by a single doubling; (2) a homogeneous singular orbit (not generic, forced by the algebra); (3) forced trace equilibrium and an explicitly computable equivariant channel spectrum (in sevenths and √3, not arbitrary). Sedenion dynamics is not a case of a general theory; it is a specific, exactly solvable instance at the intersection of several general theories, and its identity card is the spectrum.

---

## 8. Relationship to CORE

CORE (Contrast, Orientation, Re-Entry) is treated strictly as a protological language; correspondences are accepted only where forced.

**Orientation: realized.** [T-backed] The theory's total unforced content is exactly two ℤ₂-type choices — the role bit (Section 4) and the sign of i (3.9c) — plus at most one rate. CORE's claim that distinction presupposes orientation ("From which side? — The marked one") is the theorem that the invariant line exists but its *use* requires an un-invariantizable sign, used exactly once in fourteen tests (a phase origin).

**Re-Entry: realized.** [T-backed] The retained slot is re-entry by definition, and CORE's dichotomy — re-entry with same orientation yields recurrence, with changed orientation yields recursion — has an exact spectral certificate: the continuous ledger is orientation-preserving (isometries; K\*-circle), while the settlement ledger reverses orientation exactly once per event, Φ(L_{e₈}) = −L_{e₈}.

**Contrast: realized.** [C-backed] Among the eight Aut-invariant involutions of 𝕊 (sign choices on the three isotypics), **exactly one is compatible with multiplication**: conjugation, the unique Aut-natural anti-automorphic involution (certified: defect 6.6 × 10⁻¹⁵ for (+,−,−); all seven others fail at O(10)). Its eigensplitting *is* the primitive real/imaginary contrast, and its action on transitions *is* the slot exchange (conj(zx) = xz, Theorem 3.10(6)). Contrast is therefore not chosen but generated: the algebra's one canonical distinction, prior to any orientation of it — exactly CORE's ordering, since Orientation (the role bit, the sign of i) is a choice *upon* the involution's two sides, and cannot be stated without it.

**Echo, priced at zero.** [I] The forced 3 = 1 ⊕ 2 democracy and the inevitability (not primitivity) of triads under CORE are consonant; consonance is not derivation.

**Verdict.** SSD realizes a precise algebraic analogue of all three CORE primitives, each with a uniqueness theorem behind it: **Contrast** = the unique invariant anti-automorphic involution (conjugation); **Orientation** = the provably external and provably minimal ℤ₂ choices upon it (the role bit; the sign of i); **Re-Entry** = the retained slot, with recurrence/recursion split certified spectrally (isometric ledger vs. Φ(L_{e₈}) = −L_{e₈}). CORE provides a consistent protological reading of OT. The three primitives align with the three load-bearing theorems of the orientation layer. **However, whether these mathematical objects are the *realization* of CORE's primitives, or merely precise algebraic analogues of them, remains an interpretive question.** The correspondence is strong enough to be remarkable; it is constrained enough to be falsifiable.

---

## 9. Open Problems

1. **Closed forms.** Prove or refute s\* = 1/8 + 1/(3·7²); prove non-existence (or existence) of a closed form for λ_q.
2. **Higher moments.** Theorem 3.13 covers the linear channel (second moments). Prove or refute that μ and the 84-measure induce the same *Furstenberg* measure for the oriented chain — equality was observed to four decimals in λ_q and within error in s\*, but the nonlinear layer sees all moments.
3. **Tower behavior.** For A_n, n ≥ 5 (Aut = G₂ × S₃ⁿ⁻³ by Eakin–Sathaye), compute the settlement channel spectrum. Do sevenths persist? Is 𝕊 the unique rung with a one-bit orientation theory?
4. **Axiomatization.** Characterize the pairs (algebra, orbit) whose invariant channel has finite exact spectrum; determine whether (𝕊, Σ) is the minimal model.
5. **S₃-breaking.** Theorem 5.2 exiles generation hierarchy to initial data at first order; determine whether any algebra-internal mechanism (e.g., the mixed triadic sector) splits the doublet at higher order, or prove exact degeneracy to all orders.
6. **The involution dictionary at higher rungs.** Conjugation is the unique invariant anti-automorphic involution of 𝕊 (Section 8). Classify the invariant anti-automorphic involutions of A_n, n ≥ 5, and determine whether the Contrast/Orientation/Re-Entry realization persists or bifurcates up the tower.
7. **Two-time structure.** The coherence rate 2√3/7 lives only in the antisymmetric sector; develop the two-time (twin/correlator) theory in which it is observable, and its relation to the parity clock.

---

## 10. Conclusion

**1. What mathematical object was discovered?**
Sedenion Settlement Dynamics: the unique Aut(𝕊)-invariant completely positive map supported on the sedenion zero-divisor orbit, together with its exact resolution — forced equilibrium (3.6), No-Autonomy (3.8), the axis pair and its two theorems (3.9), the Hermitian transition calculus (3.10), the dictionary collapse to 𝔤₂ ⊕ ℝK\* (3.11), and the spectrum in sevenths (3.12).

**2. What orientation converts SSD into OT?**

Exactly one externally supplied bit: designate one slot of multiplication as retained and the other as sampled from (Σ, μ). Handedness is gauge (conjugation); the bit itself is external by theorem, necessary by No-Autonomy, unique up to conjugation. No other external structure is required. No fewer choices suffice.

**3. What remains interpretation rather than theorem?**

All naming: "event," "state," "time," "memory," "generations," "survivorship," the Born-shaped reading of s′ = |⟨z,x⟩_ℂ|²/(1+τ), and every contact with physics. Also two conjectures (s\* closed form; Furstenberg-measure equality of Open Problem 2). The CORE correspondence is no longer on this list: all three primitives now carry uniqueness theorems (Section 8).

**4. If every speculative application disappeared, what would mathematicians remember fifty years from now?**

The settlement channel spectrum: an 84 × 84 matrix, trace 1, with eigenvalues in (1/7)·{0, ±1, ±3, ±2√3, ±7}, multiplicities in G₂-reps. They would ask: *why does this particular spectrum appear?* The answer: because it's the unique Aut(G₂ × S₃)-invariant channel supported on the sedenion crack. Nothing in this statement requires physics, ontology, or the word "occurrence."

**5. If the entire framework were renamed after its deepest philosophical contribution rather than its surface intuition, what would that name be?**

**The Theory of Orientation** — or more technically, **G₂-Equivariant Markov Processes on the Zero-Divisor Crack** — or, most precisely, **Sedenion Settlement Dynamics**.

---

## The Heart of the Paper

The central claim is not about occurrence, and not primarily about dynamics.

**The central claim is about orientation.**

The paper discovers that sedenion settlement dynamics is completely determined by the algebra — except for exactly one externally supplied bit: the orientation of the multiplication. No-Autonomy proves why this bit must be external. Contrast realizes why it cannot be chosen arbitrarily. Re-Entry explains what its choice means dynamically. Together they form a hierarchy:

> SSD (discovered)
> ↓
> orientation (chosen)
> ↓
> Occurrence Theory (the study of consequences)

**The deepest conceptual achievement is not what an occurrence is.**

*Occurrence Theory does not claim what an occurrence is. It defines what it means to orient one.*

The word "occurrence" marks a fact: between a sampled element and a retained one, something happened. But the mathematics underneath is simpler and stranger. The mathematics says: the sedenions know everything about *what* can happen. They know nothing about *which way around* to happen. We supply that bit. And once we do, everything else is theorem.

---

## Appendix: Verification Protocol

**CRITICAL REQUIREMENT FOR INDEPENDENT VERIFICATION:** All computational claims in this paper require independent reimplementation. The verification must pass every gate listed below before any [C] result is admissible. A failed gate invalidates the entire [C] ledger for that algebra dimension. This is not optional.

All [C] claims carry numerical certificates at threshold 10⁻¹² (typical values 10⁻¹⁵–10⁻¹⁷), produced with a Cayley–Dickson implementation validated by mandatory gates:

1. **Composition:** ‖xy‖ = ‖x‖‖y‖ on every hull
2. **Antisymmetry of multipliers:** L_x + L_xᵀ = 0 for pure x
3. **Quadratic identity:** x² = −e₀ for pure unit x
4. **Moufang identity:** (ab)(ca) = a(bc)a on octonion basis

Any independent reimplementation must pass all four gates before its outputs are admissible. If your implementation fails any gate, your results are not trustworthy — discard them and debug the algebra. The gates are not suggestions; they are necessary conditions.

[M] claims report seed-split error bars. Failed pre-registrations are retained in the session ledger; four became theorems (3.6's operator form, 3.9b, 3.11's extra generator, 5.5).

## References (session-verified where marked ✓; bibliographically confirmed where marked ●)

1. P. Eakin, A. Sathaye, *On automorphisms and derivations of Cayley–Dickson algebras*, J. Algebra 129 (1990) 263–278. ✓●
2. D. K. Biss, J. D. Christensen, D. Dugger, D. C. Isaksen, *Eigentheory of Cayley–Dickson algebras*, arXiv:0905.2987; D. K. Biss, D. Dugger, D. C. Isaksen, *Large annihilators in Cayley–Dickson algebras*, Comm. Algebra (2008); *…II*, Bol. Soc. Mat. Mexicana, arXiv:math/0702075. ✓● (spectra, sum rule, hulls, Stiefel model)
3. G. Moreno, *The zero divisors of the Cayley–Dickson algebras over the real numbers*, Bol. Soc. Mat. Mexicana (3) 4 (1998) 13–28; arXiv:q-alg/9710013. ● (his theorem: the space of *pairs* of norm-one sedenions with product zero is homeomorphic to compact G₂; our Σ is the SU(2)-quotient — see Definition 3.4)
4. S. Zhilina, zero-divisor graphs of Cayley–Dickson algebras (double-hexagon components). ✓ (component structure)
5. K.-C. Chan, D. Ž. Đoković, *Conjugacy classes of subalgebras of the real sedenions*, Canad. Math. Bull. 49 (2006) 492–507. ● (prior triad count; our derivation independent)
6. R. B. Brown, *On generalized Cayley–Dickson algebras*, Pacific J. Math. 20 (1967) 415–422. ✓● (the ζ-generator; verified equal to exp((2π/3)K\*), Theorem 3.11)
7. R. D. Schafer, *An Introduction to Nonassociative Algebras*, Academic Press, 1966. ✓ (Moufang gate; derivation extension)
8. H. Furstenberg, *Noncommuting random products*, Trans. AMS (1963); P. Bougerol, J. Lacroix, *Products of Random Matrices with Applications to Schrödinger Operators*, Birkhäuser, 1985.
9. E. N. Prabhakar, *[Narrative Self Café v16–v22](https://radicalcentrism.org/tag/narrative-self-cafe/)* (radicalcentrism.org, 2026): CORE protology. Also published on [iHack.us](https://ihack.us/interests/sgi/tgt-generations/) and [2Transform.us](https://2transform.us).

*Companion artifact:* [`occurrence_theory_audit.py`](https://github.com/anomalybench/occurrence-theory) (self-contained test suite implementing the gates and certificates above).
