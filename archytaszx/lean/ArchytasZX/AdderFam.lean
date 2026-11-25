/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Closed
import ArchytasZX.ChainInduct

/-!
# Adder families: carry-chain telescoping and the Fourier shift

Lemmas of the every-`n` proofs of `draper_adder` and `fourier_constant_adder` in the boxed form
`C⁻¹ P C`, copies indexed MSB first (copy `k` reads root order `d ^ (k + 1)`), boundary
interleaved copy-major. A carry chain is `x k + e (k + 1) = x' k + d * e k` for `k < n`: copy `k`
takes carry-in `e (k + 1)` and gives carry-out `e k`.

* `om_tower`, `om_tower_mul`: `om (d ^ (m + 1)) ^ d = om (d ^ m)`.
* `digit_carry`, `carry_lt_two`, `chain_digit`: the digit-carry identity of one copy.
* `telescope`: the copy phases of a carry chain collapse to the two end terms.
* `chain_value`: the chain adds its values, `digVal x + e n = e 0 * d ^ n + digVal x'`.
* `fourier_shift`: `F³ · diag (om d ^ (y s)) · F` is the shift by `s`.
* `AdderIface`, `AdderIface.fam`: the family theorem by `Nat.rec` against an interface the engine
  export fills; `fam_8`, `fam_24` instantiate it.
* `fou_three`: `F · F · F = conj F` entrywise.
* `glueNet`, `Fits`, `glue_fits`, `glue_equal`, `netIface`, `netIface_fam`: the `AdderIface`
  over networks with scalars, `glue` attaching one copy; `trans` and `congr` proved, `base`,
  `peel`, `close` hypotheses; `glue n` attaches `copy n` shifted to level `n` (a `chainNet` step).
* `glueAt`, `FitsAt`, `CopyAt`, `glueAt_fits`, `glueAt_equal`, `netIfaceAt`, `netIfaceAt_fam`:
  the same with copy `n` joining labels of every earlier level through `cut n`.
* `telescope_off`: `telescope` with root orders `d ^ (k + r + 1)`, end terms at `d ^ r` and
  `d ^ (n + r)`.
* `cell`, `carrySum`, `carrySum_closed`: the carry chain summed over its inner carries is
  `[digVal x + e_in = e_out d ^ n + digVal x']`.
* `om_pow_up`, `phase_digVal`, `telescope_value`, `carrySum_phase`: the copy phases read the
  chain value, so the chain sum moves the phases of `x`, `e_in` to those of `x'`, `e_out`.
-/

namespace ArchytasZX

open Complex Finset

/-- The root-of-unity tower step. -/
theorem om_tower (d m : ℕ) (hd : 0 < d) : om (d ^ (m + 1)) ^ d = om (d ^ m) :=
  by
  unfold om
  rw [← Complex.exp_nat_mul]
  congr 1
  have h1 : ((d : ℂ)) ≠ 0 := by exact_mod_cast hd.ne'
  push_cast
  rw [pow_succ]
  field_simp

theorem om_tower_mul (d k y c : ℕ) (hd : 0 < d) :
    om (d ^ (k + 1)) ^ (y * (d * c)) = om (d ^ k) ^ (y * c) := by
  rw [show y * (d * c) = d * (y * c) by ring, pow_mul, om_tower d k hd]

/-- The digit-carry identity. -/
theorem digit_carry (d x c : ℕ) : x + c = (x + c) % d + d * ((x + c) / d) :=
  (Nat.mod_add_div _ _).symm

theorem carry_lt_two (d x c : ℕ) (hx : x < d) (hc : c ≤ d) : (x + c) / d < 2 := by
  rw [Nat.div_lt_iff_lt_mul (by omega)]; omega

/-- Digits and carries from `x` and a carry-in `e n`, `e k = (x k + e (k + 1)) / d`, form a
carry chain. -/
theorem chain_digit (d n : ℕ) (x e : ℕ → ℕ) (he : ∀ k < n, e k = (x k + e (k + 1)) / d) :
    ∀ k < n, x k + e (k + 1) = (x k + e (k + 1)) % d + d * e k := fun k hk => by
  rw [he k hk]; exact digit_carry d _ _

/-- The copy phases of a carry chain: `∏ om (d^(k+1))^(y x k) · om (d^n)^(y e n)` equals
`om 1 ^ (y e 0) · ∏ om (d^(k+1))^(y x' k)`. -/
theorem telescope (d n y : ℕ) (hd : 0 < d) (x x' e : ℕ → ℕ)
    (hc : ∀ k < n, x k + e (k + 1) = x' k + d * e k) :
    (∏ k ∈ range n, om (d ^ (k + 1)) ^ (y * x k)) * om (d ^ n) ^ (y * e n) =
      om (d ^ 0) ^ (y * e 0) * ∏ k ∈ range n, om (d ^ (k + 1)) ^ (y * x' k) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h := hc n (Nat.lt_succ_self n)
    rw [prod_range_succ, prod_range_succ, mul_assoc, ← pow_add, ← mul_add, h, mul_add, pow_add,
      om_tower_mul d n y _ hd]
    linear_combination om (d ^ (n + 1)) ^ (y * x' n) * ih fun k hk => hc k (Nat.lt_succ_of_lt hk)

/-- The base-`d` value of `n` copies, MSB first. -/
def digVal (d : ℕ) (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => d * digVal d x n + x n

/-- A carry chain adds: `digVal x + e n = e 0 * d ^ n + digVal x'`. -/
theorem chain_value (d n : ℕ) (x x' e : ℕ → ℕ)
    (hc : ∀ k < n, x k + e (k + 1) = x' k + d * e k) :
    digVal d x n + e n = e 0 * d ^ n + digVal d x' n := by
  induction n with
  | zero => simp [digVal]
  | succ n ih =>
    have h := hc n (Nat.lt_succ_self n)
    have ih' := ih fun k hk => hc k (Nat.lt_succ_of_lt hk)
    simp only [digVal]
    calc d * digVal d x n + x n + e (n + 1) = d * (digVal d x n + e n) + x' n := by
          rw [add_assoc, h]; ring
      _ = e 0 * d ^ (n + 1) + (d * digVal d x' n + x' n) := by rw [ih']; ring

/-- `F³ · diag (om d ^ (y s)) · F` is the shift `i ↦ i + s` mod `d`; `F³ = F⁻¹` has entry
`conj (fou d o y)`. -/
theorem fourier_shift (d s o i : ℕ) (hd : 0 < d) :
    ∑ y ∈ range d, (starRingEnd ℂ) (fou d o y) * om d ^ (y * s) * fou d y i =
      if (s + i) % d = o % d then 1 else 0 := by
  have hr : (starRingEnd ℂ) (rt d) = rt d := Complex.conj_ofReal _
  have hterm : ∀ y, (starRingEnd ℂ) (fou d o y) * om d ^ (y * s) * fou d y i =
      (rt d * rt d)⁻¹ * (om d ^ (y * (s + i)) * (starRingEnd ℂ) (om d) ^ (y * o)) := by
    intro y
    simp only [fou, map_div₀, map_pow]
    change (starRingEnd ℂ) (om d) ^ (o * y) / (starRingEnd ℂ) (rt d) * om d ^ (y * s) *
      (om d ^ (y * i) / rt d) = _
    rw [hr, mul_add, pow_add, mul_comm o y]
    field_simp
  simp only [hterm, ← mul_sum, char_sum d hd, rt_sq]
  split_ifs
  · exact nat_inv_mul d hd
  · simp

/-- The interface of an every-`n` adder family `C⁻¹ P C`: `lhs n` the boxed circuit, `rhs n`
the carry chain, `rel n` equality of their semantics, `peel` the copy-0 cut of the boxed side
around `glue n`, `close` the step identity (telescope, then the Fourier shift on copy 0). -/
structure AdderIface (α : ℕ → Type*) where
  lhs : (n : ℕ) → α n
  rhs : (n : ℕ) → α n
  rel : (n : ℕ) → α n → α n → Prop
  trans : ∀ n a b c, rel n a b → rel n b c → rel n a c
  glue : (n : ℕ) → α n → α (n + 1)
  base : rel 0 (lhs 0) (rhs 0)
  peel : ∀ n, rel (n + 1) (lhs (n + 1)) (glue n (lhs n))
  congr : ∀ n a b, rel n a b → rel (n + 1) (glue n a) (glue n b)
  close : ∀ n, rel (n + 1) (glue n (rhs n)) (rhs (n + 1))

/-- The family theorem: `lhs n` and `rhs n` are related at every `n`. -/
theorem AdderIface.fam {α : ℕ → Type*} (I : AdderIface α) : ∀ n, I.rel n (I.lhs n) (I.rhs n)
  | 0 => I.base
  | n + 1 => I.trans _ _ _ _ (I.peel n)
      (I.trans _ _ _ _ (I.congr n _ _ (AdderIface.fam I n)) (I.close n))

theorem AdderIface.fam_8 {α : ℕ → Type*} (I : AdderIface α) : I.rel 8 (I.lhs 8) (I.rhs 8) :=
  I.fam 8

theorem AdderIface.fam_24 {α : ℕ → Type*} (I : AdderIface α) : I.rel 24 (I.lhs 24) (I.rhs 24) :=
  I.fam 24

/-! ## The Fourier box cubed -/

/-- `F³ = conj F` entrywise, in matrix order, at every `o`, `i`. -/
theorem fou_three (d o i : ℕ) (hd : 0 < d) :
    ∑ a ∈ range d, ∑ b ∈ range d, fou d o a * fou d a b * fou d b i =
      (starRingEnd ℂ) (fou d o i) := by
  set a0 := (d - i % d) % d with ha0
  have hi : i % d < d := Nat.mod_lt _ hd
  have h0 : (a0 + i) % d = 0 := by
    rw [Nat.add_mod, ha0, Nat.mod_mod, Nat.mod_add_mod, Nat.sub_add_cancel hi.le, Nat.mod_self]
  have huniq : ∀ a < d, (a + i) % d = 0 → a = a0 := by
    intro a ha h
    have h1 : a + i ≡ a0 + i [MOD d] := by
      show (a + i) % d = (a0 + i) % d
      rw [h, h0]
    have h2 := Nat.ModEq.add_right_cancel' i h1
    unfold Nat.ModEq at h2
    rwa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt (Nat.mod_lt _ hd)] at h2
  simp only [mul_assoc, ← mul_sum, fou_sq d hd _ i, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_eq_single a0 (fun a ha hne =>
      ite_eq_right_iff.2 fun h => absurd (huniq a (Finset.mem_range.1 ha) h) hne)
      (fun h => absurd (Finset.mem_range.2 (Nat.mod_lt _ hd)) h), ite_eq_left_iff.2
      fun h => absurd h0 h]
  have hr : (starRingEnd ℂ) (rt d) = rt d := Complex.conj_ofReal _
  have hone : om d ^ (o * a0) * om d ^ (o * i) = 1 := by
    rw [← pow_add, ← mul_add, om_pow_mod d hd, Nat.mul_mod, h0, mul_zero, Nat.zero_mod, pow_zero]
  simp only [fou, map_div₀, map_pow, conj_om]
  change om d ^ (o * a0) / rt d = ((om d)⁻¹) ^ (o * i) / (starRingEnd ℂ) (rt d)
  rw [hr, inv_pow, eq_inv_of_mul_eq_one_left hone]

/-! ## A network instance -/

/-- `N` with one more copy attached at level `k`: `copy` shifted by `k W`, plugged through the
labels `cut` shifted by `k W`, which become summed. -/
def glueNet (W : ℕ) (cut : List ℕ) (copy : Net) (k : ℕ) (N : Net) : Net :=
  let C := copy.mapL (· + k * W)
  let cs := cut.map (· + k * W)
  ⟨C.nodes ++ N.nodes, C.int ++ cs ++ N.int, N.bnd.filter (· ∉ cs) ++ C.bnd.filter (· ∉ cs)⟩

theorem chainNet_succ (W : ℕ) (cut : List ℕ) (base : Net) (copy : ℕ → Net) (k : ℕ) :
    chainNet W cut base copy (k + 1) = glueNet W cut (copy k) k (chainNet W cut base copy k) :=
  rfl

/-- Every label of `N` below `k W + W`, no summed one a cut label shifted by `k W`. -/
def Fits (W : ℕ) (cut : List ℕ) (k : ℕ) (N : Net) : Prop :=
  (∀ x ∈ labelsOf N.nodes ++ N.int ++ N.bnd, x < k * W + W) ∧
    ∀ x ∈ N.int, ∀ c ∈ cut, x ≠ c + k * W

theorem glue_fits (W : ℕ) (cut : List ℕ) (cp N : Net) (k : ℕ) (hcut : ∀ c ∈ cut, c < W)
    (hc : copyOK W cut cp = true) (hN : Fits W cut k N) :
    Fits W cut (k + 1) (glueNet W cut cp k N) := by
  simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hc
  obtain ⟨hcl, hci⟩ := hc
  obtain ⟨ih1, ih2⟩ := hN
  have hsucc : (k + 1) * W = k * W + W := by rw [Nat.succ_mul]
  refine ⟨fun x hx => ?_, fun x hx c hcm e => ?_⟩
  · rw [hsucc]
    have hx' : x ∈ labelsOf (cp.mapL (· + k * W)).nodes ++ (cp.mapL (· + k * W)).int ++
        (cp.mapL (· + k * W)).bnd ∨ x ∈ cut.map (· + k * W) ∨
        x ∈ labelsOf N.nodes ++ N.int ++ N.bnd := by
      simp only [glueNet, labelsOf, List.flatMap_append, List.mem_append, List.mem_filter] at hx ⊢
      tauto
    rcases hx' with h | h | h
    · obtain ⟨y, hy, rfl⟩ := mem_labels_mapL (· + k * W) cp x h
      rcases hcl y hy with h' | ⟨_, h'⟩
      · have := hcut y h'; omega
      · omega
    · obtain ⟨c', hc', rfl⟩ := List.mem_map.1 h
      have := hcut c' hc'; omega
    · have := ih1 x h; omega
  · rw [hsucc] at e
    have hcW := hcut c hcm
    simp only [glueNet, List.mem_append] at hx
    rcases hx with (hx | hx) | hx
    · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 (show x ∈ cp.int.map (· + k * W) from hx)
      obtain ⟨_, hy2⟩ := hci y hy
      exact hy2 (List.mem_map.2 ⟨c, hcm, by omega⟩)
    · obtain ⟨c', hc', rfl⟩ := List.mem_map.1 hx
      have := hcut c' hc'; omega
    · have := ih1 x (mem_all_of_int hx); omega

/-- Attaching the same copy to both sides keeps an equation. -/
theorem glue_equal (E : Env) (W : ℕ) (cut : List ℕ) (cp : Net) (k : ℕ) {s t : ℂ} {A B : Net}
    (hc : copyOK W cut cp = true) (hA : Fits W cut k A) (hB : Fits W cut k B)
    (h : Equal E s A t B) : Equal E s (glueNet W cut cp k A) t (glueNet W cut cp k B) := by
  have hc' := hc
  simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hc'
  have A1 := plug E (cp.mapL (· + k * W)).nodes A.nodes B.nodes
    ((cp.mapL (· + k * W)).int ++ cut.map (· + k * W)) A.int B.int
    (A.bnd.filter (· ∉ cut.map (· + k * W)) ++
      (cp.mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W)))
    A.bnd _ _ (Equal.toLocal E _ _ h)
    (copy_off W k cut A cp hA.1 hA.2 hc'.1) (copy_off W k cut B cp hB.1 hB.2 hc'.1)
    (face_left _ _ _ _)
  exact A1.reorder E (B' := glueNet W cut cp k B) (List.Perm.refl _) (List.Perm.refl _)
    ((h.1.filter _).append (List.Perm.refl _))

/-- The `AdderIface` over networks with scalars: `α n = ℂ × Net`, `rel n` is `Fits` on both
sides and `Equal E`, `glue n` attaches `copy n` at level `n`. -/
def netIface (E : Env) (W : ℕ) (cut : List ℕ) (copy : ℕ → Net) (lhs rhs : ℕ → ℂ × Net)
    (hcut : ∀ c ∈ cut, c < W) (hcopy : ∀ n, copyOK W cut (copy n) = true)
    (fL : ∀ n, Fits W cut n (lhs n).2) (fR : ∀ n, Fits W cut n (rhs n).2)
    (base : Equal E (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2)
    (peel : ∀ n, Equal E (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
      (glueNet W cut (copy n) n (lhs n).2))
    (close : ∀ n, Equal E (rhs n).1 (glueNet W cut (copy n) n (rhs n).2) (rhs (n + 1)).1
      (rhs (n + 1)).2) :
    AdderIface (fun _ => ℂ × Net) where
  lhs := lhs
  rhs := rhs
  rel n a b := Fits W cut n a.2 ∧ Fits W cut n b.2 ∧ Equal E a.1 a.2 b.1 b.2
  trans _ _ _ _ h h' := ⟨h.1, h'.2.1, h.2.2.trans h'.2.2⟩
  glue n a := (a.1, glueNet W cut (copy n) n a.2)
  base := ⟨fL 0, fR 0, base⟩
  peel n := ⟨fL (n + 1), glue_fits W cut _ _ n hcut (hcopy n) (fL n), peel n⟩
  congr n _ _ h := ⟨glue_fits W cut _ _ n hcut (hcopy n) h.1,
    glue_fits W cut _ _ n hcut (hcopy n) h.2.1, glue_equal E W cut _ n (hcopy n) h.1 h.2.1 h.2.2⟩
  close n := ⟨glue_fits W cut _ _ n hcut (hcopy n) (fR n), fR (n + 1), close n⟩

/-- The network family theorem: `lhs n` equals `rhs n` with their scalars at every `n`. -/
theorem netIface_fam (E : Env) (W : ℕ) (cut : List ℕ) (copy : ℕ → Net) (lhs rhs : ℕ → ℂ × Net)
    (hcut : ∀ c ∈ cut, c < W) (hcopy : ∀ n, copyOK W cut (copy n) = true)
    (fL : ∀ n, Fits W cut n (lhs n).2) (fR : ∀ n, Fits W cut n (rhs n).2)
    (base : Equal E (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2)
    (peel : ∀ n, Equal E (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
      (glueNet W cut (copy n) n (lhs n).2))
    (close : ∀ n, Equal E (rhs n).1 (glueNet W cut (copy n) n (rhs n).2) (rhs (n + 1)).1
      (rhs (n + 1)).2) (n : ℕ) :
    Equal E (lhs n).1 (lhs n).2 (rhs n).1 (rhs n).2 :=
  ((netIface E W cut copy lhs rhs hcut hcopy fL fR base peel close).fam n).2.2

/-- `Fits` of a chain family at every `k`. -/
theorem chainNet_fits (W : ℕ) (cut : List ℕ) (base : Net) (copy : ℕ → Net)
    (hb : baseOK W cut base = true) (hc : ∀ i, copyOK W cut (copy i) = true) (k : ℕ) :
    Fits W cut k (chainNet W cut base copy k) :=
  chain_bounds W cut base copy hb hc k

/-! ## A network instance with fan-out copies -/

/-- `N` with `cp` attached through the labels `cut`, which become summed. -/
def glueAt (cut : List ℕ) (cp N : Net) : Net :=
  ⟨cp.nodes ++ N.nodes, cp.int ++ cut ++ N.int, N.bnd.filter (· ∉ cut) ++ cp.bnd.filter (· ∉ cut)⟩

/-- Every label of `N` below `B n`, no summed one in `cut m` for `m ≥ n`. -/
def FitsAt (B : ℕ → ℕ) (cut : ℕ → List ℕ) (n : ℕ) (N : Net) : Prop :=
  (∀ x ∈ labelsOf N.nodes ++ N.int ++ N.bnd, x < B n) ∧ ∀ m, n ≤ m → ∀ x ∈ N.int, x ∉ cut m

/-- Copy `n` reads its cut labels (below `B n`) and owns labels in `[B n, B (n + 1))`; later cuts
avoid its summed labels and its cut. -/
structure CopyAt (B : ℕ → ℕ) (cut : ℕ → List ℕ) (cp : ℕ → Net) : Prop where
  mono : ∀ n, B n ≤ B (n + 1)
  cutLt : ∀ n, ∀ x ∈ cut n, x < B n
  own : ∀ n, ∀ x ∈ labelsOf (cp n).nodes ++ (cp n).int ++ (cp n).bnd,
    x ∈ cut n ∨ (B n ≤ x ∧ x < B (n + 1))
  intOff : ∀ n m, n < m → ∀ x ∈ (cp n).int, x ∉ cut m
  cutOff : ∀ n m, n < m → ∀ x ∈ cut n, x ∉ cut m

theorem glueAt_fits {B : ℕ → ℕ} {cut : ℕ → List ℕ} {cp : ℕ → Net} (hC : CopyAt B cut cp)
    (n : ℕ) {N : Net} (hN : FitsAt B cut n N) :
    FitsAt B cut (n + 1) (glueAt (cut n) (cp n) N) := by
  have hm := hC.mono n
  refine ⟨fun x hx => ?_, fun m hm' x hx => ?_⟩
  · have hx' : x ∈ labelsOf (cp n).nodes ++ (cp n).int ++ (cp n).bnd ∨ x ∈ cut n ∨
        x ∈ labelsOf N.nodes ++ N.int ++ N.bnd := by
      simp only [glueAt, labelsOf, List.flatMap_append, List.mem_append, List.mem_filter] at hx ⊢
      tauto
    rcases hx' with h | h | h
    · rcases hC.own n x h with h' | ⟨_, h'⟩
      · have := hC.cutLt n x h'; omega
      · exact h'
    · have := hC.cutLt n x h; omega
    · have := hN.1 x h; omega
  · simp only [glueAt, List.mem_append] at hx
    rcases hx with (hx | hx) | hx
    · exact hC.intOff n m (by omega) x hx
    · exact hC.cutOff n m (by omega) x hx
    · exact hN.2 m (by omega) x hx

/-- Attaching copy `n` to both sides keeps an equation. -/
theorem glueAt_equal (E : Env) {B : ℕ → ℕ} {cut : ℕ → List ℕ} {cp : ℕ → Net}
    (hC : CopyAt B cut cp) (n : ℕ) {s t : ℂ} {A A' : Net} (hA : FitsAt B cut n A)
    (hA' : FitsAt B cut n A') (h : Equal E s A t A') :
    Equal E s (glueAt (cut n) (cp n) A) t (glueAt (cut n) (cp n) A') := by
  have off : ∀ {N : Net}, FitsAt B cut n N → ∀ x ∈ labelsOf (cp n).nodes, x ∉ N.int :=
    fun hN x hx hxi => by
      rcases hC.own n x (mem_all_of_nodes hx) with h' | ⟨h', _⟩
      · exact hN.2 n le_rfl x hxi h'
      · have := hN.1 x (mem_all_of_int hxi); omega
  have A1 := plug E (cp n).nodes A.nodes A'.nodes ((cp n).int ++ cut n) A.int A'.int
    (A.bnd.filter (· ∉ cut n) ++ (cp n).bnd.filter (· ∉ cut n)) A.bnd _ _
    (Equal.toLocal E _ _ h) (off hA) (off hA') (face_left _ _ _ _)
  exact A1.reorder E (B' := glueAt (cut n) (cp n) A') (List.Perm.refl _) (List.Perm.refl _)
    ((h.1.filter _).append (List.Perm.refl _))

/-- The `AdderIface` over networks with scalars whose copy `n` may join every earlier level:
`rel n` is `FitsAt` on both sides and `Equal E`, `glue n` is `glueAt (cut n) (cp n)`. -/
def netIfaceAt (E : Env) (B : ℕ → ℕ) (cut : ℕ → List ℕ) (cp : ℕ → Net) (lhs rhs : ℕ → ℂ × Net)
    (hC : CopyAt B cut cp) (fL : ∀ n, FitsAt B cut n (lhs n).2)
    (fR : ∀ n, FitsAt B cut n (rhs n).2)
    (base : Equal E (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2)
    (peel : ∀ n, Equal E (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
      (glueAt (cut n) (cp n) (lhs n).2))
    (close : ∀ n, Equal E (rhs n).1 (glueAt (cut n) (cp n) (rhs n).2) (rhs (n + 1)).1
      (rhs (n + 1)).2) :
    AdderIface (fun _ => ℂ × Net) where
  lhs := lhs
  rhs := rhs
  rel n a b := FitsAt B cut n a.2 ∧ FitsAt B cut n b.2 ∧ Equal E a.1 a.2 b.1 b.2
  trans _ _ _ _ h h' := ⟨h.1, h'.2.1, h.2.2.trans h'.2.2⟩
  glue n a := (a.1, glueAt (cut n) (cp n) a.2)
  base := ⟨fL 0, fR 0, base⟩
  peel n := ⟨fL (n + 1), glueAt_fits hC n (fL n), peel n⟩
  congr n _ _ h := ⟨glueAt_fits hC n h.1, glueAt_fits hC n h.2.1,
    glueAt_equal E hC n h.1 h.2.1 h.2.2⟩
  close n := ⟨glueAt_fits hC n (fR n), fR (n + 1), close n⟩

/-- The fan-out network family theorem. -/
theorem netIfaceAt_fam (E : Env) (B : ℕ → ℕ) (cut : ℕ → List ℕ) (cp : ℕ → Net)
    (lhs rhs : ℕ → ℂ × Net) (hC : CopyAt B cut cp) (fL : ∀ n, FitsAt B cut n (lhs n).2)
    (fR : ∀ n, FitsAt B cut n (rhs n).2)
    (base : Equal E (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2)
    (peel : ∀ n, Equal E (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
      (glueAt (cut n) (cp n) (lhs n).2))
    (close : ∀ n, Equal E (rhs n).1 (glueAt (cut n) (cp n) (rhs n).2) (rhs (n + 1)).1
      (rhs (n + 1)).2) (n : ℕ) :
    Equal E (lhs n).1 (lhs n).2 (rhs n).1 (rhs n).2 :=
  ((netIfaceAt E B cut cp lhs rhs hC fL fR base peel close).fam n).2.2

/-- `telescope` with every root order raised by `d ^ r`: the end terms read `d ^ r` and
`d ^ (n + r)`. -/
theorem telescope_off (d n y r : ℕ) (hd : 0 < d) (x x' e : ℕ → ℕ)
    (hc : ∀ k < n, x k + e (k + 1) = x' k + d * e k) :
    (∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x k)) * om (d ^ (n + r)) ^ (y * e n) =
      om (d ^ r) ^ (y * e 0) * ∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x' k) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h := hc n (Nat.lt_succ_self n)
    rw [show n + 1 + r = n + r + 1 by omega, prod_range_succ, prod_range_succ, mul_assoc,
      ← pow_add, ← mul_add, h, mul_add, pow_add, om_tower_mul d (n + r) y _ hd]
    linear_combination om (d ^ (n + r + 1)) ^ (y * x' n) *
      ih fun k hk => hc k (Nat.lt_succ_of_lt hk)

/-! ## The carry chain in closed form -/

/-- The cell `[x + e_in = x' + d e_out]`. -/
def cell (d x ein x' eout : ℕ) : ℂ := if x + ein = x' + d * eout then 1 else 0

/-- The carry chain of `n` copies, carry-in `ein` of copy `n - 1`, carry-out `eout` of copy `0`,
every inner carry summed over `range C`. -/
noncomputable def carrySum (d C : ℕ) (x x' : ℕ → ℕ) : ℕ → ℕ → ℕ → ℂ
  | 0, ein, eout => if ein = eout then 1 else 0
  | n + 1, ein, eout => ∑ c ∈ range C, cell d (x n) ein (x' n) c * carrySum d C x x' n c eout

/-- The carry chain sum is `[digVal x + e_in = e_out d ^ n + digVal x']`. -/
theorem carrySum_closed (d C : ℕ) (hd : 0 < d) (hC : 2 ≤ C) (x x' : ℕ → ℕ)
    (hx : ∀ k, x k < d) (hx' : ∀ k, x' k < d) :
    ∀ n ein eout, ein ≤ d → carrySum d C x x' n ein eout =
      if digVal d x n + ein = eout * d ^ n + digVal d x' n then 1 else 0 := by
  intro n
  induction n with
  | zero => intro ein eout _; simp [carrySum, digVal]
  | succ n ih =>
    intro ein eout hein
    rw [carrySum, show digVal d x (n + 1) = d * digVal d x n + x n from rfl,
      show digVal d x' (n + 1) = d * digVal d x' n + x' n from rfl]
    have hP : eout * d ^ (n + 1) = d * (eout * d ^ n) := by ring
    rw [hP]
    have hxn := hx n
    have hxn' := hx' n
    by_cases hm : (x n + ein) % d = x' n
    · set c0 := (x n + ein) / d
      have hc0 : x n + ein = x' n + d * c0 := by rw [← hm]; exact digit_carry d _ _
      have hc02 : c0 < 2 := carry_lt_two d _ _ hxn hein
      have hcell : ∀ c, cell d (x n) ein (x' n) c = if c = c0 then 1 else 0 := by
        intro c
        unfold cell
        congr 1
        apply propext
        constructor
        · intro h
          exact Nat.eq_of_mul_eq_mul_left hd (by omega)
        · rintro rfl; exact hc0
      simp only [hcell, ite_mul, one_mul, zero_mul]
      have hcC : c0 ∈ range C := Finset.mem_range.2 (by omega)
      rw [Finset.sum_ite_eq' (range C) c0, ite_eq_left_iff.2 fun h => absurd hcC h, ih c0 eout (by omega)]
      congr 1
      apply propext
      constructor
      · intro h
        have h1 := congrArg (d * ·) h
        simp only [mul_add] at h1
        linarith
      · intro h
        have h1 : d * (digVal d x n + c0) = d * (eout * d ^ n + digVal d x' n) := by
          rw [mul_add, mul_add]; linarith
        exact Nat.eq_of_mul_eq_mul_left hd h1
    · have hcell : ∀ c, cell d (x n) ein (x' n) c = 0 := by
        intro c
        unfold cell
        refine ite_eq_right_iff.2 fun h => absurd ?_ hm
        rw [h, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hxn']
      simp only [hcell, zero_mul, Finset.sum_const_zero]
      split_ifs with h
      · exfalso
        apply hm
        have h2 : (d * digVal d x n + (x n + ein)) % d =
            (d * (eout * d ^ n + digVal d x' n) + x' n) % d := by
          congr 1; rw [mul_add]; linarith
        rwa [Nat.mul_add_mod, Nat.mul_add_mod, Nat.mod_eq_of_lt hxn'] at h2
      · rfl

/-- `om (d ^ (n + r)) ^ (c d ^ n) = om (d ^ r) ^ c`. -/
theorem om_pow_up (d r c : ℕ) (hd : 0 < d) :
    ∀ n, om (d ^ (n + r)) ^ (c * d ^ n) = om (d ^ r) ^ c := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    have h := om_tower_mul d (n + r) 1 (c * d ^ n) hd
    rw [show n + 1 + r = n + r + 1 by omega, show c * d ^ (n + 1) = 1 * (d * (c * d ^ n)) by ring,
      h, one_mul, ih]

/-- The copy phases read the value: `∏ om (d^(k+r+1))^(y x k) = om (d^(n+r))^(y digVal x)`. -/
theorem phase_digVal (d r y : ℕ) (hd : 0 < d) (x : ℕ → ℕ) :
    ∀ n, ∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x k) =
      om (d ^ (n + r)) ^ (y * digVal d x n) := by
  intro n
  induction n with
  | zero => simp [digVal]
  | succ n ih =>
    rw [prod_range_succ, ih, show digVal d x (n + 1) = d * digVal d x n + x n from rfl,
      ← om_tower_mul d (n + r) y _ hd, ← pow_add, show n + 1 + r = n + r + 1 by omega]
    ring_nf

/-- `telescope_off` from the value equation of the chain alone. -/
theorem telescope_value (d n y r : ℕ) (hd : 0 < d) (x x' : ℕ → ℕ) (ein eout : ℕ)
    (hv : digVal d x n + ein = eout * d ^ n + digVal d x' n) :
    (∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x k)) * om (d ^ (n + r)) ^ (y * ein) =
      om (d ^ r) ^ (y * eout) * ∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x' k) := by
  rw [phase_digVal d r y hd x n, phase_digVal d r y hd x' n, ← pow_add, ← mul_add, hv, mul_add,
    pow_add, ← om_pow_up d r (y * eout) hd n]
  ring_nf

/-- The carry chain sum carries the copy phases of `x` and `e_in` to those of `x'` and
`e_out`. -/
theorem carrySum_phase (d C n y r : ℕ) (hd : 0 < d) (hC : 2 ≤ C) (x x' : ℕ → ℕ)
    (hx : ∀ k, x k < d) (hx' : ∀ k, x' k < d) (ein eout : ℕ) (hein : ein ≤ d) :
    carrySum d C x x' n ein eout *
        ((∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x k)) * om (d ^ (n + r)) ^ (y * ein)) =
      carrySum d C x x' n ein eout *
        (om (d ^ r) ^ (y * eout) * ∏ k ∈ range n, om (d ^ (k + r + 1)) ^ (y * x' k)) := by
  rw [carrySum_closed d C hd hC x x' hx hx' n ein eout hein]
  split_ifs with hv
  · rw [telescope_value d n y r hd x x' ein eout hv]
  · simp

end ArchytasZX
