/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFam
import ArchytasZX.ChainMixed

/-!
# Adder families over a growing label list

`netIfaceAt` with the environment `envAt E0 L n = { E0 with labs := L n }` at copy count `n`:
the label dimensions of level `n` and below are read from `L n`, and `L (n + 1)` agrees with
`L n` below `B n`.

* `envAt`, `LabsMono`: the environment family and its agreement condition.
* `netIfaceAtE`, `netIfaceAtE_fam`: the `AdderIface` over it and its family theorem.
-/

namespace ArchytasZX

/-- `E0` with the label list `L n`. -/
def envAt (E0 : Env) (L : ℕ → List (ℕ × ℕ)) (n : ℕ) : Env := { E0 with labs := L n }

/-- `L (n + 1)` gives every label below `B n` the dimension `L n` gives it. -/
def LabsMono (E0 : Env) (L : ℕ → List (ℕ × ℕ)) (B : ℕ → ℕ) : Prop :=
  ∀ n x, x < B n → labDim (envAt E0 L (n + 1)) x = labDim (envAt E0 L n) x

theorem Equal.envAt_succ {E0 : Env} {L : ℕ → List (ℕ × ℕ)} {B : ℕ → ℕ} {cut : ℕ → List ℕ}
    (hL : LabsMono E0 L B) (n : ℕ) {s t : ℂ} {A A' : Net} (hA : FitsAt B cut n A)
    (hA' : FitsAt B cut n A') (h : Equal (envAt E0 L n) s A t A') :
    Equal (envAt E0 L (n + 1)) s A t A' :=
  Equal.relabs (E := envAt E0 L n) (L (n + 1)) h
    (fun l hl => hL n l (hA.1 l (by simp only [List.mem_append] at hl ⊢; tauto)))
    (fun l hl => hL n l (hA'.1 l (mem_all_of_int hl)))

/-- The `AdderIface` of `netIfaceAt` with `rel n` read in `envAt E0 L n`. -/
def netIfaceAtE (E0 : Env) (L : ℕ → List (ℕ × ℕ)) (B : ℕ → ℕ) (cut : ℕ → List ℕ)
    (cp : ℕ → Net) (lhs rhs : ℕ → ℂ × Net) (hL : LabsMono E0 L B) (hC : CopyAt B cut cp)
    (fL : ∀ n, FitsAt B cut n (lhs n).2) (fR : ∀ n, FitsAt B cut n (rhs n).2)
    (base : Equal (envAt E0 L 0) (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2)
    (peel : ∀ n, Equal (envAt E0 L (n + 1)) (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
      (glueAt (cut n) (cp n) (lhs n).2))
    (close : ∀ n, Equal (envAt E0 L (n + 1)) (rhs n).1 (glueAt (cut n) (cp n) (rhs n).2)
      (rhs (n + 1)).1 (rhs (n + 1)).2) :
    AdderIface (fun _ => ℂ × Net) where
  lhs := lhs
  rhs := rhs
  rel n a b := FitsAt B cut n a.2 ∧ FitsAt B cut n b.2 ∧ Equal (envAt E0 L n) a.1 a.2 b.1 b.2
  trans _ _ _ _ h h' := ⟨h.1, h'.2.1, h.2.2.trans h'.2.2⟩
  glue n a := (a.1, glueAt (cut n) (cp n) a.2)
  base := ⟨fL 0, fR 0, base⟩
  peel n := ⟨fL (n + 1), glueAt_fits hC n (fL n), peel n⟩
  congr n _ _ h := ⟨glueAt_fits hC n h.1, glueAt_fits hC n h.2.1,
    glueAt_equal _ hC n h.1 h.2.1 (Equal.envAt_succ hL n h.1 h.2.1 h.2.2)⟩
  close n := ⟨glueAt_fits hC n (fR n), fR (n + 1), close n⟩

/-- The family theorem of `netIfaceAtE`. -/
theorem netIfaceAtE_fam (E0 : Env) (L : ℕ → List (ℕ × ℕ)) (B : ℕ → ℕ) (cut : ℕ → List ℕ)
    (cp : ℕ → Net) (lhs rhs : ℕ → ℂ × Net) (hL : LabsMono E0 L B) (hC : CopyAt B cut cp)
    (fL : ∀ n, FitsAt B cut n (lhs n).2) (fR : ∀ n, FitsAt B cut n (rhs n).2)
    (base : Equal (envAt E0 L 0) (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2)
    (peel : ∀ n, Equal (envAt E0 L (n + 1)) (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
      (glueAt (cut n) (cp n) (lhs n).2))
    (close : ∀ n, Equal (envAt E0 L (n + 1)) (rhs n).1 (glueAt (cut n) (cp n) (rhs n).2)
      (rhs (n + 1)).1 (rhs (n + 1)).2) (n : ℕ) :
    Equal (envAt E0 L n) (lhs n).1 (lhs n).2 (rhs n).1 (rhs n).2 :=
  ((netIfaceAtE E0 L B cut cp lhs rhs hL hC fL fR base peel close).fam n).2.2

end ArchytasZX
