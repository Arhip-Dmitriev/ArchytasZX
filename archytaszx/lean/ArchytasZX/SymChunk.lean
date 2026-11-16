/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymNet

/-!
# Symbolic normal forms checked node by node

`trStepB` is one node's step of `trNodesB` on the tail's expression. `trNodesB_cons` reduces
`trNodesB` on a node list to that step, so an export states each intermediate expression and
checks each step with a `decide` of its own. `symCheck_of` gives `symCheck` from both
networks' stated expressions.
-/

namespace ArchytasZX.Sym

/-- One node's step of `trNodesB` on the tail's expression `b`. -/
def trStepB (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node) (b : List STm) :
    Option (List STm) :=
  match trNode dt pt rl (OFF + i) n with
  | some a =>
    if exMulOK a b then some (nfEx fl (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)))
    else none
  | none => none

theorem trNodesB_nil (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) :
    trNodesB fl dt pt rl labs int pre i [] = some [⟨CMon.ofQ 1, [], [], []⟩] := rfl

theorem trNodesB_cons (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node)
    (ns : List Node) (b : List STm)
    (h : trNodesB fl dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns = some b) :
    trNodesB fl dt pt rl labs int pre i (n :: ns) = trStepB fl dt pt rl labs int pre i n b := by
  unfold trNodesB trStepB
  rw [h]
  cases trNode dt pt rl (OFF + i) n <;> rfl

theorem trNet_of (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (N : Net) (ns : List Node) (b : List STm)
    (hn : N.nodes = ns) (h : trNodesB fl dt pt rl labs N.int [] 0 ns = some b) :
    trNet fl dt pt rl labs N =
      some (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)) b) := by
  unfold trNet
  rw [hn, h]
  subst hn
  rfl

theorem symCheck_of (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (IX : List ℕ) (N0 N1 : Net)
    (S0 S1 e0 e1 : List STm) (h0 : trNet fl dt pt rl labs N0 = some e0)
    (h1 : trNet fl dt pt rl labs N1 = some e1)
    (h : (N0.bnd.isPerm N1.bnd && netChk fl dt pt rl labs IX N0 && netChk fl dt pt rl labs IX N1 &&
      exConst S0 && exConst S1 && (exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
      (nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty)) = true) :
    symCheck fl dt pt rl labs IX N0 N1 S0 S1 = true := by
  unfold symCheck checkWith
  rw [h0, h1]
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h ⊢
  obtain ⟨⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, f⟩ := h
  exact ⟨⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, Or.inl f⟩

end ArchytasZX.Sym
