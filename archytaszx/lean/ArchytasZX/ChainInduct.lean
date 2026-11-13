/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bundle

/-!
# Chain families by induction on the copy count

`chainNet W cut base copy k` is the network of a chain box at multiplicity `k`: `base` at
`k = 0`, and at `k + 1` copy `k` (`copy k` shifted by `k W`) plugged onto the network at `k`
through the labels `cut` shifted by `k W`, which become summed. `chain_equal` proves both
families equal at every `k` from the base case and the equation of each copy pair, the scalars
multiplied over the copies; `perIdx` reads copy `i` from a table of period `|tab|`.
-/

namespace ArchytasZX

open Complex Finset

/-- A network with every label mapped through `f`. -/
def Net.mapL (f : ℕ → ℕ) (N : Net) : Net :=
  ⟨N.nodes.map (Node.rename f), N.int.map f, N.bnd.map f⟩

theorem eval_mapL (E : Env) (f : ℕ → ℕ) (hf : Function.Injective f)
    (hd : ∀ x, labDim E (f x) = labDim E x) (N : Net) (σ : ℕ → ℕ) :
    eval E (N.mapL f) σ = eval E N (σ ∘ f) := by
  simp only [eval, Net.mapL]
  rw [show prodVal E (N.nodes.map (Node.rename f)) = fun τ => prodVal E N.nodes (τ ∘ f) from
    funext fun τ => prodVal_rename E f τ N.nodes]
  exact sumOver_map _ f hf N.int (fun x _ => hd x) _ σ

theorem Equal.mapL (E : Env) (f : ℕ → ℕ) (hf : Function.Injective f)
    (hd : ∀ x, labDim E (f x) = labDim E x) {s s' : ℂ} {A B : Net} (h : Equal E s A s' B) :
    Equal E s (A.mapL f) s' (B.mapL f) := by
  refine ⟨by simpa [Net.mapL] using h.1.map f, fun σ hσ => ?_⟩
  rw [eval_mapL E f hf hd, eval_mapL E f hf hd]
  refine h.2 _ fun l hl => ?_
  have := hσ (f l) (by simp only [Net.mapL, List.mem_map]; exact ⟨l, hl, rfl⟩)
  simpa [Function.comp, hd] using this

theorem labelsOf_rename (f : ℕ → ℕ) (ns : List Node) :
    labelsOf (ns.map (Node.rename f)) = (labelsOf ns).map f := by
  simp [labelsOf, Node.labels, Node.rename, List.map_flatMap, List.flatMap_map]

theorem eval_perm_net (E : Env) (A B : Net) (hn : A.nodes.Perm B.nodes) (hi : A.int.Perm B.int)
    (σ : ℕ → ℕ) : eval E A σ = eval E B σ := by
  unfold eval
  rw [sumOver_perm _ hi]
  congr 1
  funext τ
  exact (hn.map _).prod_eq

theorem Equal.reorder (E : Env) {s s' : ℂ} {A B B' : Net} (h : Equal E s A s' B)
    (hn : B.nodes.Perm B'.nodes) (hi : B.int.Perm B'.int) (hb : B.bnd.Perm B'.bnd) :
    Equal E s A s' B' :=
  ⟨h.1.trans hb, fun σ hσ => by rw [h.2 σ hσ, eval_perm_net E B B' hn hi σ]⟩

theorem Equal.reorderL (E : Env) {s s' : ℂ} {A A' B : Net} (h : Equal E s A s' B)
    (hn : A.nodes.Perm A'.nodes) (hi : A.int.Perm A'.int) (hb : A.bnd.Perm A'.bnd) :
    Equal E s A' s' B :=
  (h.symm.reorder E hn hi hb).symm

theorem Equal.congr_scalar {E : Env} {a a' b b' : ℂ} {X Y : Net} (h : Equal E a X b Y)
    (ha : a = a') (hb : b = b') : Equal E a' X b' Y := ha ▸ hb ▸ h

/-- Every label has dimension 2 when `Δ k = 2 ^ max k 1` and every listed digit count is at
most 1. -/
theorem labDim_two (E : Env) (hE : ∀ k, E.dimv k = 2 ^ max k 1)
    (h : E.labs.all (fun p => decide (p.2 ≤ 1)) = true) : ∀ x y, labDim E x = labDim E y := by
  have key : ∀ x, labDim E x = 2 := by
    intro x
    unfold labDim
    rw [hE]
    have hle : (E.labs.lookup x).getD 0 ≤ 1 := by
      cases hl : E.labs.lookup x with
      | none => simp
      | some v =>
        have := List.all_eq_true.1 h _ (lookup_mem hl)
        simpa using this
    rw [Nat.max_eq_right hle]
    norm_num
  intro x y
  rw [key, key]

/-! ## The family -/

/-- The network of a chain at `k` copies, copy `i` at level `i`. -/
def chainNet (W : ℕ) (cut : List ℕ) (base : Net) (copy : ℕ → Net) : ℕ → Net
  | 0 => base
  | k + 1 =>
    let C := (copy k).mapL (· + k * W)
    let cs := cut.map (· + k * W)
    ⟨C.nodes ++ (chainNet W cut base copy k).nodes,
      C.int ++ cs ++ (chainNet W cut base copy k).int,
      (chainNet W cut base copy k).bnd.filter (· ∉ cs) ++ C.bnd.filter (· ∉ cs)⟩

/-- Every base label below `W`, the cut labels below `W` and not summed. -/
def baseOK (W : ℕ) (cut : List ℕ) (base : Net) : Bool :=
  (labelsOf base.nodes ++ base.int ++ base.bnd).all (fun x => decide (x < W)) &&
    cut.all (fun x => decide (x < W)) && base.int.all (fun x => decide (x ∉ cut))

/-- Every copy label a cut label or in `[W, 2 W)`, the summed ones off the cut labels and their
shifts by `W`. -/
def copyOK (W : ℕ) (cut : List ℕ) (copy : Net) : Bool :=
  (labelsOf copy.nodes ++ copy.int ++ copy.bnd).all
      (fun x => decide (x ∈ cut ∨ (W ≤ x ∧ x < 2 * W))) &&
    copy.int.all (fun x => decide (W ≤ x ∧ x ∉ cut.map (· + W)))

/-- Entry `i mod |tab|` of a table. -/
def perIdx {α : Type} (tab : List α) (d : α) (i : ℕ) : α := tab.getD (i % tab.length) d

theorem perIdx_all {α : Type} (tab : List α) (d : α) (P : α → Prop) (hn : 0 < tab.length)
    (h : ∀ j < tab.length, P (tab.getD j d)) : ∀ i, P (perIdx tab d i) :=
  fun i => h _ (Nat.mod_lt _ hn)

theorem mem_labels_mapL (f : ℕ → ℕ) (N : Net) (x : ℕ)
    (h : x ∈ labelsOf (N.mapL f).nodes ++ (N.mapL f).int ++ (N.mapL f).bnd) :
    ∃ y ∈ labelsOf N.nodes ++ N.int ++ N.bnd, x = f y := by
  simp only [Net.mapL, labelsOf_rename, List.mem_append, List.mem_map] at h
  rcases h with (⟨y, hy, rfl⟩ | ⟨y, hy, rfl⟩) | ⟨y, hy, rfl⟩
  · exact ⟨y, by simp [hy], rfl⟩
  · exact ⟨y, by simp [hy], rfl⟩
  · exact ⟨y, by simp [hy], rfl⟩

theorem mem_all_of_nodes {N : Net} {x : ℕ} (h : x ∈ labelsOf N.nodes) :
    x ∈ labelsOf N.nodes ++ N.int ++ N.bnd := List.mem_append_left _ (List.mem_append_left _ h)

theorem mem_all_of_int {N : Net} {x : ℕ} (h : x ∈ N.int) :
    x ∈ labelsOf N.nodes ++ N.int ++ N.bnd := List.mem_append_left _ (List.mem_append_right _ h)

theorem mem_all_of_bnd {N : Net} {x : ℕ} (h : x ∈ N.bnd) :
    x ∈ labelsOf N.nodes ++ N.int ++ N.bnd := List.mem_append_right _ h

theorem mem_chain_succ (W : ℕ) (cut : List ℕ) (base : Net) (copy : ℕ → Net) (k x : ℕ)
    (hx : x ∈ labelsOf (chainNet W cut base copy (k + 1)).nodes ++
      (chainNet W cut base copy (k + 1)).int ++ (chainNet W cut base copy (k + 1)).bnd) :
    x ∈ labelsOf ((copy k).mapL (· + k * W)).nodes ++ ((copy k).mapL (· + k * W)).int ++
        ((copy k).mapL (· + k * W)).bnd ∨ x ∈ cut.map (· + k * W) ∨
      x ∈ labelsOf (chainNet W cut base copy k).nodes ++ (chainNet W cut base copy k).int ++
        (chainNet W cut base copy k).bnd := by
  simp only [chainNet, labelsOf, List.flatMap_append, List.mem_append, List.mem_filter] at hx ⊢
  tauto

/-- Every label of the network at `k` is below `k W + W`; no summed one is a cut label
shifted by `k W`. -/
theorem chain_bounds (W : ℕ) (cut : List ℕ) (base : Net) (copy : ℕ → Net)
    (hb : baseOK W cut base = true) (hc : ∀ i, copyOK W cut (copy i) = true) (k : ℕ) :
    (∀ x ∈ labelsOf (chainNet W cut base copy k).nodes ++ (chainNet W cut base copy k).int ++
      (chainNet W cut base copy k).bnd, x < k * W + W) ∧
      ∀ x ∈ (chainNet W cut base copy k).int, ∀ c ∈ cut, x ≠ c + k * W := by
  simp only [baseOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hb
  obtain ⟨⟨hbl, hcut⟩, hbi⟩ := hb
  induction k with
  | zero =>
    refine ⟨fun x hx => by simpa using hbl x hx, fun x hx c hcm e => ?_⟩
    simp only [zero_mul, add_zero] at e
    exact hbi x hx (e ▸ hcm)
  | succ k ih =>
    obtain ⟨ih1, ih2⟩ := ih
    have hck := hc k
    simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hck
    obtain ⟨hcl, hci⟩ := hck
    have hsucc : (k + 1) * W = k * W + W := by rw [Nat.succ_mul]
    refine ⟨fun x hx => ?_, fun x hx c hcm e => ?_⟩
    · rw [hsucc]
      rcases mem_chain_succ W cut base copy k x hx with h | h | h
      · obtain ⟨y, hy, rfl⟩ := mem_labels_mapL (· + k * W) (copy k) x h
        rcases hcl y hy with h' | ⟨_, h'⟩
        · have := hcut y h'; omega
        · omega
      · obtain ⟨c', hc', rfl⟩ := List.mem_map.1 h
        have := hcut c' hc'; omega
      · have := ih1 x h; omega
    · rw [hsucc] at e
      have hcW := hcut c hcm
      simp only [chainNet, List.mem_append] at hx
      rcases hx with (hx | hx) | hx
      · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 (show x ∈ (copy k).int.map (· + k * W) from hx)
        obtain ⟨_, hy2⟩ := hci y hy
        exact hy2 (List.mem_map.2 ⟨c, hcm, by omega⟩)
      · obtain ⟨c', hc', rfl⟩ := List.mem_map.1 hx
        have := hcut c' hc'; omega
      · have := ih1 x (mem_all_of_int hx); omega

theorem copy_off (W k : ℕ) (cut : List ℕ) (N cp : Net)
    (h1N : ∀ x ∈ labelsOf N.nodes ++ N.int ++ N.bnd, x < k * W + W)
    (h2N : ∀ x ∈ N.int, ∀ c ∈ cut, x ≠ c + k * W)
    (hl : ∀ x ∈ labelsOf cp.nodes ++ cp.int ++ cp.bnd, x ∈ cut ∨ (W ≤ x ∧ x < 2 * W)) :
    ∀ x ∈ labelsOf (cp.mapL (· + k * W)).nodes, x ∉ N.int := by
  intro x hx hxi
  obtain ⟨y, hy, rfl⟩ := mem_labels_mapL (· + k * W) cp x (mem_all_of_nodes hx)
  rcases hl y hy with h | ⟨h, _⟩
  · exact h2N _ hxi y h rfl
  · have := h1N _ (mem_all_of_int hxi); omega

theorem net_off (W k : ℕ) (cut : List ℕ) (N cp : Net)
    (h1N : ∀ x ∈ labelsOf N.nodes ++ N.int ++ N.bnd, x < k * W + W)
    (hci : ∀ x ∈ cp.int, W ≤ x ∧ x ∉ cut.map (· + W)) :
    ∀ x ∈ labelsOf N.nodes, x ∉ (cp.mapL (· + k * W)).int := by
  intro x hx hxi
  obtain ⟨y, hy, rfl⟩ := List.mem_map.1 (show x ∈ cp.int.map (· + k * W) from hxi)
  have := h1N _ (mem_all_of_nodes hx)
  have := (hci y hy).1
  omega

theorem face_left (A : List ℕ) (cs : List ℕ) (iC B : List ℕ) :
    ∀ x ∈ A, x ∈ iC ++ cs ∨ x ∈ A.filter (· ∉ cs) ++ B := by
  intro x hx
  by_cases hm : x ∈ cs
  · exact Or.inl (List.mem_append_right _ hm)
  · exact Or.inr (List.mem_append_left _ (List.mem_filter.2 ⟨hx, by simpa using hm⟩))

theorem face_right (A : List ℕ) (cs : List ℕ) (iC B : List ℕ) :
    ∀ x ∈ A, x ∈ cs ++ iC ∨ x ∈ B ++ A.filter (· ∉ cs) := by
  intro x hx
  by_cases hm : x ∈ cs
  · exact Or.inl (List.mem_append_left _ hm)
  · exact Or.inr (List.mem_append_right _ (List.mem_filter.2 ⟨hx, by simpa using hm⟩))

/-- Two chain families agree at every `k` from the base case and the equation of each copy
pair, copy `i` scaled by `a i` and `b i`. -/
theorem chain_equal (E : Env) (W : ℕ) (cut : List ℕ) (bO : Net) (cO : ℕ → Net) (bX : Net)
    (cX : ℕ → Net) (hD : ∀ x y, labDim E x = labDim E y)
    (hbO : baseOK W cut bO = true) (hcO : ∀ i, copyOK W cut (cO i) = true)
    (hbX : baseOK W cut bX = true) (hcX : ∀ i, copyOK W cut (cX i) = true)
    (s t : ℂ) (a b : ℕ → ℂ) (h0 : Equal E s bO t bX) (h1 : ∀ i, Equal E (a i) (cO i) (b i) (cX i)) :
    ∀ k, Equal E (s * ∏ i ∈ range k, a i) (chainNet W cut bO cO k)
      (t * ∏ i ∈ range k, b i) (chainNet W cut bX cX k) := by
  intro k
  induction k with
  | zero => simpa [chainNet] using h0
  | succ k ih =>
    have bO' := chain_bounds W cut bO cO hbO hcO k
    have bX' := chain_bounds W cut bX cX hbX hcX k
    have hcOk := hcO k
    have hcXk := hcX k
    simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hcOk hcXk
    have hinj : Function.Injective (· + k * W) := fun x y h => by simpa using h
    have hcc : Equal E (a k) ((cO k).mapL (· + k * W)) (b k) ((cX k).mapL (· + k * W)) :=
      Equal.mapL E _ hinj (fun x => hD _ _) (h1 k)
    have A1 := plug E ((cO k).mapL (· + k * W)).nodes (chainNet W cut bO cO k).nodes
      (chainNet W cut bX cX k).nodes (((cO k).mapL (· + k * W)).int ++ cut.map (· + k * W))
      (chainNet W cut bO cO k).int (chainNet W cut bX cX k).int
      ((chainNet W cut bO cO k).bnd.filter (· ∉ cut.map (· + k * W)) ++
        ((cO k).mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W)))
      (chainNet W cut bO cO k).bnd _ _ (Equal.toLocal E _ _ ih)
      (copy_off W k cut _ (cO k) bO'.1 bO'.2 hcOk.1) (copy_off W k cut _ (cO k) bX'.1 bX'.2 hcOk.1)
      (face_left _ _ _ _)
    have A2 := plug E (chainNet W cut bX cX k).nodes ((cO k).mapL (· + k * W)).nodes
      ((cX k).mapL (· + k * W)).nodes (cut.map (· + k * W) ++ (chainNet W cut bX cX k).int)
      ((cO k).mapL (· + k * W)).int ((cX k).mapL (· + k * W)).int
      ((chainNet W cut bO cO k).bnd.filter (· ∉ cut.map (· + k * W)) ++
        ((cO k).mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W)))
      ((cO k).mapL (· + k * W)).bnd _ _ (Equal.toLocal E _ _ hcc)
      (net_off W k cut _ (cO k) bX'.1 hcOk.2) (net_off W k cut _ (cX k) bX'.1 hcXk.2)
      (face_right _ _ _ _)
    have hn : ∀ (A B : List Node), (A ++ B).Perm (B ++ A) := fun A B => List.perm_append_comm
    have hi : ∀ (A B C : List ℕ), (A ++ B ++ C).Perm (C ++ A ++ B) := fun A B C =>
      List.perm_append_comm.trans (List.Perm.of_eq (List.append_assoc _ _ _).symm)
    have A2' : Equal E (a k)
        ⟨((cO k).mapL (· + k * W)).nodes ++ (chainNet W cut bX cX k).nodes,
          ((cO k).mapL (· + k * W)).int ++ cut.map (· + k * W) ++ (chainNet W cut bX cX k).int,
          (chainNet W cut bO cO k).bnd.filter (· ∉ cut.map (· + k * W)) ++
            ((cO k).mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W))⟩ (b k)
        ⟨((cX k).mapL (· + k * W)).nodes ++ (chainNet W cut bX cX k).nodes,
          ((cX k).mapL (· + k * W)).int ++ cut.map (· + k * W) ++ (chainNet W cut bX cX k).int,
          (chainNet W cut bO cO k).bnd.filter (· ∉ cut.map (· + k * W)) ++
            ((cO k).mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W))⟩ :=
      (A2.reorderL E (A' := ⟨((cO k).mapL (· + k * W)).nodes ++ (chainNet W cut bX cX k).nodes,
          ((cO k).mapL (· + k * W)).int ++ cut.map (· + k * W) ++ (chainNet W cut bX cX k).int,
          (chainNet W cut bO cO k).bnd.filter (· ∉ cut.map (· + k * W)) ++
            ((cO k).mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W))⟩)
        (hn _ _) (hi _ _ _) (List.Perm.refl _)).reorder E
        (B' := ⟨((cX k).mapL (· + k * W)).nodes ++ (chainNet W cut bX cX k).nodes,
          ((cX k).mapL (· + k * W)).int ++ cut.map (· + k * W) ++ (chainNet W cut bX cX k).int,
          (chainNet W cut bO cO k).bnd.filter (· ∉ cut.map (· + k * W)) ++
            ((cO k).mapL (· + k * W)).bnd.filter (· ∉ cut.map (· + k * W))⟩)
        (hn _ _) (hi _ _ _) (List.Perm.refl _)
    have hb := (ih.1.filter (fun x => decide (x ∉ cut.map (· + k * W)))).append
      (hcc.1.filter (fun x => decide (x ∉ cut.map (· + k * W))))
    have A3 := (A1.mul_trans A2').reorder E (B' := chainNet W cut bX cX (k + 1))
      (List.Perm.refl _) (List.Perm.refl _) hb
    exact A3.congr_scalar (a' := s * ∏ i ∈ range (k + 1), a i)
      (b' := t * ∏ i ∈ range (k + 1), b i)
      (by rw [Finset.prod_range_succ]; ring) (by rw [Finset.prod_range_succ]; ring)

end ArchytasZX
