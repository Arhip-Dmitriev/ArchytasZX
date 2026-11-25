/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainLabel
import ArchytasZX.BoxFree
import ArchytasZX.AdderFamFcaEng
import ArchytasZX.CopyBridge

/-!
# A diagram's entry as a network under a port labelling

`eval_toNet_dim`: `eval_toNet` with each wire label's dimension given by `labDim` in place of
`check`. `eval_netBy`: when `lab` reads slot `i` at boundary port `i`, both ends of a wire alike,
distinct wires apart and above the slots, every leg on the boundary or a wire, and each wire's
label of its dimension, `D.eval E σ` is the entry of `D.netBy lab` (through `toNet` over `E`
with the wire labels `o + j` above every `lab` value added to `E.labs`).
-/

namespace ArchytasZX

namespace Diagram

variable (D : Diagram)

theorem eval_toNet_dim (E : Env) (off : ℕ) (hoff : D.bnd.length ≤ off) (hok : D.portsOK)
    (hd : ∀ j < D.wires.length, labDim E (off + j) = E.dimv (D.wireDim j)) (σ : ℕ → ℕ) :
    ArchytasZX.eval E (D.toNet off) σ = D.eval E σ := by
  refine sumOver_shift _ _ off D.wires.length _ _ σ (fun a b hab ha => ?_) _ (fun j hj => ?_)
    σ (fun _ => 0) (fun j hj hn => absurd (List.mem_range.2 hj) hn) (fun _ _ => rfl)
  · simp only [prodVal, toNet, List.map_map]
    congr 1
    refine List.map_congr_left fun n hn => ?_
    have hn' := hok n hn
    show val E a (n.node (D.label off)) = val E id (n.node (D.value σ b))
    rw [← Function.id_comp a, ← val_rename, node_rename]
    simp only [DNode.node]
    congr 2
    · exact List.map_congr_left fun k hk =>
        D.okPort_label hoff (hn'.1 k (List.mem_range.1 hk)) hab ha
    · exact List.map_congr_left fun k hk =>
        D.okPort_label hoff (hn'.2 k (List.mem_range.1 hk)) hab ha
  · exact hd j (List.mem_range.1 hj)

/-- `E` with the labels `o + j` of dimension index `D.wireDim j`. -/
def envW (E : Env) (o : ℕ) : Env :=
  { E with labs := ((List.range D.wires.length).map fun j => (o + j, D.wireDim j)) ++ E.labs }

theorem lookup_append_none {l l' : List (ℕ × ℕ)} {x : ℕ} (h : l.lookup x = none) :
    (l ++ l').lookup x = l'.lookup x := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : x = k
    · subst hk; simp [List.lookup] at h
    · have : (x == k) = false := by simpa using hk
      simp only [List.cons_append, List.lookup, this] at h ⊢
      exact ih h

theorem lookup_append_some {l l' : List (ℕ × ℕ)} {x v : ℕ} (h : l.lookup x = some v) :
    (l ++ l').lookup x = some v := by
  induction l with
  | nil => simp at h
  | cons p l ih =>
    obtain ⟨k, w⟩ := p
    by_cases hk : x = k
    · subst hk; simp only [List.cons_append, List.lookup, beq_self_eq_true] at h ⊢; exact h
    · have : (x == k) = false := by simpa using hk
      simp only [List.cons_append, List.lookup, this] at h ⊢
      exact ih h

theorem labDim_envW_lo (E : Env) (o : ℕ) {x : ℕ} (hx : x < o) : labDim (D.envW E o) x = labDim E x := by
  unfold labDim envW
  rw [lookup_append_none (Fca.lookup_none _ _ (by simp; omega))]

theorem labDim_envW_hi (E : Env) (o : ℕ) {j : ℕ} (hj : j < D.wires.length) :
    labDim (D.envW E o) (o + j) = E.dimv (D.wireDim j) := by
  unfold labDim envW
  rw [lookup_append_some (FcaEng.lookup_shift _ _ _ _ hj)]
  rfl

theorem le_sum {l : List ℕ} {x : ℕ} (h : x ∈ l) : x ≤ l.sum := by
  induction l with
  | nil => simp at h
  | cons a l ih =>
    rcases List.mem_cons.1 h with rfl | h
    · simp
    · simp only [List.sum_cons]; have := ih h; omega

/-- `D.eval E σ` is the entry of `D.netBy lab`. -/
theorem eval_netBy (E : Env) (lab : Port → ℕ)
    (hb : ∀ i (hi : i < D.bnd.length), lab D.bnd[i] = i)
    (hw : ∀ w ∈ D.wires, lab w.1 = lab w.2) (hok : D.portsOK)
    (hnd : (D.wires.map fun w => lab w.1).Nodup)
    (hlo : ∀ w ∈ D.wires, D.bnd.length ≤ lab w.1)
    (hdim : ∀ j (hj : j < D.wires.length), labDim E (lab D.wires[j].1) = E.dimv (D.wireDim j))
    (hbd : ∀ i < D.bnd.length, labDim E i = E.dimv (D.bdims.getD i 0))
    (σ : ℕ → ℕ) (hσ : D.InRange E σ) : D.eval E σ = ArchytasZX.eval E (D.netBy lab) σ := by
  set o := D.bnd.length + (D.wires.map fun w => lab w.1).sum + 1
  have hlt : ∀ w ∈ D.wires, lab w.1 < o := fun w hw' => by
    have := le_sum (List.mem_map_of_mem (f := fun w => lab w.1) hw'); omega
  have hv : ∀ τ (x : Node), val (D.envW E o) τ x = val E τ x := fun τ x =>
    val_envAt E (D.envW E o) rfl rfl rfl τ x rfl
  have e1 : D.eval E σ = D.eval (D.envW E o) σ := by
    simp only [Diagram.eval, hv]; rfl
  rw [e1, ← D.eval_toNet_dim (D.envW E o) o (by omega) hok
    (fun j hj => D.labDim_envW_hi E o hj) σ]
  have hEq := D.equal_netBy (D.envW E o) o lab (by omega) hb hw hok hnd hlo (fun j hj => by
    rw [D.labDim_envW_hi E o hj, D.labDim_envW_lo E o (hlt _ (List.getElem_mem hj)), hdim j hj])
  have hin : ArchytasZX.InRange (D.envW E o) (D.toNet o).bnd σ := fun l hl => by
    have hl' : l < D.bnd.length := List.mem_range.1 hl
    rw [D.labDim_envW_lo E o (by omega), hbd l hl']
    exact hσ l hl'
  have := hEq.2 σ hin
  simp only [one_mul] at this
  rw [this]
  unfold ArchytasZX.eval
  have hp : prodVal (D.envW E o) (D.netBy lab).nodes = prodVal E (D.netBy lab).nodes := by
    have hv' : val (D.envW E o) = val E := funext fun τ => funext fun x => hv τ x
    unfold prodVal; rw [hv']
  rw [hp]
  refine sumOver_dim_congr _ _ _ (fun x hx => ?_) _ σ
  obtain ⟨w, hw', rfl⟩ := List.mem_map.1 hx
  exact D.labDim_envW_lo E o (hlt w hw')

end Diagram

end ArchytasZX
