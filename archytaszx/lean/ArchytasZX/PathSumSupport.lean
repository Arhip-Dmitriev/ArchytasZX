/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumAmp

/-!
# Reduced sums with a leftover support

`finalSOK` holds for two reduced sums with no path variable, one list of constraints (the
leftover support indicator after `gather` and `bcanon`), one list of rows and one phase up to
the constant term. `finalS_sound` compares such sums; `afinalS_perm` does so with atoms equal up
to order and no held variable.
-/

namespace ArchytasZX
namespace PSem

open Complex

/-- Whether both sums have no path variable and the same constraints, rows and phase up to the
constant term. -/
def finalSOK (sL sR : Data) : Bool :=
  decide (sL.path = []) && decide (sR.path = []) && decide (sL.cons = sR.cons) &&
    decide (sL.rows = sR.rows) && decide (strip sL.phase = strip sR.phase)

theorem finalS_sound (M : ℕ) (_hM : 0 < M) (cL cR : ℂ) (sL sR : Data) (σ : ℕ → ZMod 2)
    (kL kR : ℕ) (h : finalSOK sL sR = true) (hkL : constK sL.phase = kL)
    (hkR : constK sR.phase = kR) (hc : cL * om M ^ kL = cR * om M ^ kR) :
    val M cL sL σ = val M cR sR σ := by
  subst hkL hkR
  simp only [finalSOK, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨hpL, hpR⟩, hcons⟩, hrows⟩, hph⟩ := h
  unfold val summand
  rw [hpL, hpR]
  simp only [bsum]
  rw [hcons, hrows]
  split_ifs
  · rw [pn_strip σ sL.phase, pn_strip σ sR.phase, hph]
    push_cast
    rw [add_div, add_div, ex_add, ex_add, ← om_eq_ex, ← om_eq_ex]
    linear_combination ex (((pn σ (strip sR.phase) : ℤ) : ℝ) / M) * hc
  · rfl

/-- Two reduced sums with a leftover support, no held variable and atoms equal up to order,
compared. -/
theorem afinalS_perm (M : ℕ) (hM : 0 < M) (cL cR : ℂ) (A : ℕ → ℂ) (tL tR : AData)
    (σ : ℕ → ZMod 2) (kL kR : ℕ) (hL : tL.held = []) (hR : tR.held = [])
    (ha : decide (tL.atoms.Perm tR.atoms) = true) (h : finalSOK tL.s tR.s = true)
    (hkL : constK tL.s.phase = kL) (hkR : constK tR.s.phase = kR)
    (hc : cL * om M ^ kL = cR * om M ^ kR) :
    valA M cL A tL σ = valA M cR A tR σ := by
  simp only [valA, hL, hR, bsum]
  rw [aval_perm A σ (of_decide_eq_true ha),
    finalS_sound M hM cL cR tL.s tR.s σ kL kR h hkL hkR hc]

end PSem
end ArchytasZX
