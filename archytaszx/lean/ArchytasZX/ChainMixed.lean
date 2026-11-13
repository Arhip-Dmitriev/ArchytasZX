/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainInduct

/-!
# Chain families over mixed wire dimensions

`chain_equal_le` is `chain_equal` up to a copy count `K`, each copy label's dimension kept by
the shift to its level instead of every label sharing one dimension. `lvEnv E W C B K` is `E`
with the labels of `C` (in `[W, 2 W)`) at the levels `0 .. K - 1`, then those of `B` (below
`W`); `lv_hD` gives the shift hypothesis for it and `Equal.lvK` moves an equation on labels
below `2 W` from `K = 1` to any `K ≥ 1`. `Equal.relabs` moves an equation to another label
list agreeing on its labels' dimensions.
-/

namespace ArchytasZX

open Complex Finset

theorem sumOver_dims (D D' : ℕ → ℕ) (f : (ℕ → ℕ) → ℂ) :
    ∀ (ls : List ℕ), (∀ l ∈ ls, D l = D' l) → ∀ σ, sumOver D ls f σ = sumOver D' ls f σ
  | [], _, _ => rfl
  | l :: ls, h, σ => by
    simp only [sumOver]
    rw [h l (by simp)]
    exact Finset.sum_congr rfl fun j _ =>
      sumOver_dims D D' f ls (fun x hx => h x (by simp [hx])) _

theorem tensor_relabs (E : Env) (L : List (ℕ × ℕ)) (k : Kind) :
    ∀ d ph os is, tensor { E with labs := L } k d ph os is = tensor E k d ph os is := by
  induction k with
  | Bun m b ih => intro d ph os is; simp only [tensor, copyOuts, copyIns, ih]
  | _ =>
    intro d ph os is
    rcases os with _ | ⟨o, _ | ⟨o', os⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨i', is⟩⟩ <;>
      simp [tensor]

theorem val_relabs (E : Env) (L : List (ℕ × ℕ)) (σ : ℕ → ℕ) (n : Node) :
    val { E with labs := L } σ n = val E σ n := by
  simp only [val, tensor_relabs]

theorem prodVal_relabs (E : Env) (L : List (ℕ × ℕ)) :
    prodVal { E with labs := L } = prodVal E := by
  funext ns σ
  have h : val { E with labs := L } σ = val E σ := funext (val_relabs E L σ)
  simp only [prodVal, h]

/-- `Equal` under the label list `L`, given the labels' dimensions agree on both networks. -/
theorem Equal.relabs {E : Env} (L : List (ℕ × ℕ)) {s s' : ℂ} {A B : Net}
    (h : Equal E s A s' B)
    (hA : ∀ l ∈ A.int ++ A.bnd, labDim { E with labs := L } l = labDim E l)
    (hB : ∀ l ∈ B.int, labDim { E with labs := L } l = labDim E l) :
    Equal { E with labs := L } s A s' B := by
  refine ⟨h.1, fun σ hσ => ?_⟩
  have hσ' : InRange E A.bnd σ := fun l hl => by
    rw [← hA l (List.mem_append_right _ hl)]
    exact hσ l hl
  have key := h.2 σ hσ'
  unfold eval at key ⊢
  rw [prodVal_relabs, prodVal_relabs, sumOver_dims _ (labDim E) _ A.int (fun l hl => hA l (List.mem_append_left _ hl)),
    sumOver_dims _ (labDim E) _ B.int hB]
  exact key

theorem eval_mapL_on (E : Env) (f : ℕ → ℕ) (hf : Function.Injective f) (N : Net)
    (hd : ∀ x ∈ N.int, labDim E (f x) = labDim E x) (σ : ℕ → ℕ) :
    eval E (N.mapL f) σ = eval E N (σ ∘ f) := by
  simp only [eval, Net.mapL]
  rw [show prodVal E (N.nodes.map (Node.rename f)) = fun τ => prodVal E N.nodes (τ ∘ f) from
    funext fun τ => prodVal_rename E f τ N.nodes]
  exact sumOver_map _ f hf N.int hd _ σ

/-- `Equal.mapL` with the dimensions kept on the summed and boundary labels only. -/
theorem Equal.mapL_on (E : Env) (f : ℕ → ℕ) (hf : Function.Injective f) {s s' : ℂ} {A B : Net}
    (hd : ∀ x ∈ A.int ++ B.int ++ A.bnd, labDim E (f x) = labDim E x)
    (h : Equal E s A s' B) : Equal E s (A.mapL f) s' (B.mapL f) := by
  refine ⟨by simpa [Net.mapL] using h.1.map f, fun σ hσ => ?_⟩
  rw [eval_mapL_on E f hf A (fun x hx => hd x (by simp [hx])),
    eval_mapL_on E f hf B (fun x hx => hd x (by simp [hx]))]
  refine h.2 _ fun l hl => ?_
  have := hσ (f l) (by simp only [Net.mapL, List.mem_map]; exact ⟨l, hl, rfl⟩)
  simpa [Function.comp, hd l (by simp [hl])] using this

/-- `Equal.mapL2` with the dimensions kept on the summed and boundary labels only. -/
theorem Equal.mapL2_on (E : Env) (f g : ℕ → ℕ) (hf : Function.Injective f)
    (hg : Function.Injective g) {s s' : ℂ} {A B : Net}
    (hdf : ∀ x ∈ A.int ++ A.bnd, labDim E (f x) = labDim E x)
    (hdg : ∀ x ∈ B.int, labDim E (g x) = labDim E x) (h : Equal E s A s' B)
    (hfg : ∀ x ∈ A.bnd, f x = g x) (hcB : ∀ l ∈ labelsOf B.nodes, l ∈ B.int ∨ l ∈ B.bnd) :
    Equal E s (A.mapL f) s' (B.mapL g) := by
  have hmap : A.bnd.map f = A.bnd.map g := List.map_congr_left hfg
  refine ⟨?_, fun σ hσ => ?_⟩
  · simp only [Net.mapL]
    rw [hmap]
    exact h.1.map g
  rw [eval_mapL_on E f hf A (fun x hx => hdf x (List.mem_append_left _ hx)),
    eval_mapL_on E g hg B hdg]
  have hσA : InRange E A.bnd (σ ∘ f) := fun l hl => by
    have := hσ (f l) (by simp only [Net.mapL, List.mem_map]; exact ⟨l, hl, rfl⟩)
    simpa [Function.comp, hdf l (List.mem_append_right _ hl)] using this
  rw [h.2 _ hσA]
  congr 1
  unfold eval
  apply sumOver_ext (labDim E) B.int (labelsOf B.nodes) _ (prodVal_reads E B.nodes)
  intro x hx hn
  rcases hcB x hx with hi | hb
  · exact absurd hi hn
  · simp only [Function.comp]
    rw [hfg x (h.1.mem_iff.2 hb)]

/-- `chain_equal` for `k ≤ K`, each copy below `K` keeping its labels' dimensions under the
shift to its level. -/
theorem chain_equal_le (E : Env) (W : ℕ) (cut : List ℕ) (bO : Net) (cO : ℕ → Net) (bX : Net)
    (cX : ℕ → Net) (K : ℕ)
    (hDO : ∀ i < K, ∀ x ∈ labelsOf (cO i).nodes ++ (cO i).int ++ (cO i).bnd,
      labDim E (x + i * W) = labDim E x)
    (hDX : ∀ i < K, ∀ x ∈ labelsOf (cX i).nodes ++ (cX i).int ++ (cX i).bnd,
      labDim E (x + i * W) = labDim E x)
    (hbO : baseOK W cut bO = true) (hcO : ∀ i, copyOK W cut (cO i) = true)
    (hbX : baseOK W cut bX = true) (hcX : ∀ i, copyOK W cut (cX i) = true)
    (s t : ℂ) (a b : ℕ → ℂ) (h0 : Equal E s bO t bX)
    (h1 : ∀ i < K, Equal E (a i) (cO i) (b i) (cX i)) :
    ∀ k ≤ K, Equal E (s * ∏ i ∈ range k, a i) (chainNet W cut bO cO k)
      (t * ∏ i ∈ range k, b i) (chainNet W cut bX cX k) := by
  intro k hk
  induction k with
  | zero => simpa [chainNet] using h0
  | succ k ih =>
    have hkK : k < K := by omega
    have ih := ih (by omega)
    have bO' := chain_bounds W cut bO cO hbO hcO k
    have bX' := chain_bounds W cut bX cX hbX hcX k
    have hcOk := hcO k
    have hcXk := hcX k
    simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hcOk hcXk
    have hinj : Function.Injective (· + k * W) := fun x y h => by simpa using h
    have hcc : Equal E (a k) ((cO k).mapL (· + k * W)) (b k) ((cX k).mapL (· + k * W)) := by
      refine Equal.mapL_on E _ hinj (fun x hx => ?_) (h1 k hkK)
      simp only [List.mem_append] at hx
      rcases hx with (hx | hx) | hx
      · exact hDO k hkK x (mem_all_of_int hx)
      · exact hDX k hkK x (mem_all_of_int hx)
      · exact hDO k hkK x (mem_all_of_bnd hx)
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

/-! ## Labels by level -/

/-- `C` shifted to level `i`. -/
def lvl (W : ℕ) (C : List (ℕ × ℕ)) (i : ℕ) : List (ℕ × ℕ) := C.map fun p => (p.1 + i * W, p.2)

/-- Levels `0 .. K - 1` of `C`. -/
def lvls (W : ℕ) (C : List (ℕ × ℕ)) : ℕ → List (ℕ × ℕ)
  | 0 => []
  | K + 1 => lvls W C K ++ lvl W C K

/-- `E` with the labels of `C` at levels `0 .. K - 1`, then those of `B`. -/
def lvEnv (E : Env) (W : ℕ) (C B : List (ℕ × ℕ)) (K : ℕ) : Env :=
  { E with labs := lvls W C K ++ B }

theorem lookup_none_of (l : List (ℕ × ℕ)) (x : ℕ) (h : ∀ p ∈ l, p.1 ≠ x) : l.lookup x = none := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    have hk : k ≠ x := h _ List.mem_cons_self
    have : (x == k) = false := by simpa [beq_iff_eq] using hk.symm
    simp only [List.lookup, this]
    exact ih fun q hq => h q (List.mem_cons_of_mem _ hq)

theorem lookup_app_none (l₁ l₂ : List (ℕ × ℕ)) (x : ℕ) (h : l₁.lookup x = none) :
    (l₁ ++ l₂).lookup x = l₂.lookup x := by
  induction l₁ with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : x = k
    · subst hk
      simp [List.lookup] at h
    · have : (x == k) = false := by simpa [beq_iff_eq] using hk
      simp only [List.cons_append, List.lookup, this] at h ⊢
      exact ih h

theorem lookup_app_some (l₁ l₂ : List (ℕ × ℕ)) (x v : ℕ) (h : l₁.lookup x = some v) :
    (l₁ ++ l₂).lookup x = some v := by
  induction l₁ with
  | nil => simp [List.lookup] at h
  | cons p l ih =>
    obtain ⟨k, w⟩ := p
    by_cases hk : x = k
    · subst hk
      simpa [List.lookup] using h
    · have : (x == k) = false := by simpa [beq_iff_eq] using hk
      simp only [List.cons_append, List.lookup, this] at h ⊢
      exact ih h

theorem lookup_lvl (W i x : ℕ) (C : List (ℕ × ℕ)) :
    (lvl W C i).lookup (x + i * W) = C.lookup x := by
  induction C with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : x = k
    · subst hk
      simp [lvl, List.lookup]
    · have h1 : (x + i * W == k + i * W) = false := by simpa [beq_iff_eq] using hk
      have h2 : (x == k) = false := by simpa [beq_iff_eq] using hk
      simp only [lvl, List.map_cons, List.lookup, h1, h2] at ih ⊢
      exact ih

theorem mem_lvls (W : ℕ) (C : List (ℕ × ℕ)) :
    ∀ K, ∀ p ∈ lvls W C K, ∃ q ∈ C, ∃ j < K, p.1 = q.1 + j * W
  | 0, p, hp => by simp [lvls] at hp
  | K + 1, p, hp => by
    simp only [lvls, List.mem_append] at hp
    rcases hp with hp | hp
    · obtain ⟨q, hq, j, hj, e⟩ := mem_lvls W C K p hp
      exact ⟨q, hq, j, by omega, e⟩
    · simp only [lvl, List.mem_map] at hp
      obtain ⟨q, hq, rfl⟩ := hp
      exact ⟨q, hq, K, by omega, rfl⟩

theorem step_le (W j K : ℕ) (h : j < K) : j * W + W ≤ K * W := by
  have := Nat.mul_le_mul_right W (show j + 1 ≤ K by omega)
  rwa [Nat.succ_mul] at this

theorem lookup_lvls (W : ℕ) (C : List (ℕ × ℕ)) (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W) (x : ℕ)
    (hx : W ≤ x ∧ x < 2 * W) : ∀ K, ∀ i < K, (lvls W C K).lookup (x + i * W) = C.lookup x
  | 0, i, hi => absurd hi (Nat.not_lt_zero _)
  | K + 1, i, hi => by
    simp only [lvls]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with h | h
    · cases hc : C.lookup x with
      | some v =>
        exact lookup_app_some _ _ _ _ ((lookup_lvls W C hC x hx K i h).trans hc)
      | none =>
        rw [lookup_app_none _ _ _ ((lookup_lvls W C hC x hx K i h).trans hc)]
        refine lookup_none_of _ _ fun p hp => ?_
        simp only [lvl, List.mem_map] at hp
        obtain ⟨q, hq, rfl⟩ := hp
        have := hC q hq
        have := step_le W i K h
        simp only
        omega
    · subst h
      rw [lookup_app_none _ _ _ (lookup_none_of _ _ fun p hp => ?_)]
      · exact lookup_lvl W i x C
      obtain ⟨q, hq, j, hj, e⟩ := mem_lvls W C i p hp
      have := hC q hq
      have := step_le W j i hj
      omega

theorem lv_hi (E : Env) (W : ℕ) (C B : List (ℕ × ℕ)) (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W)
    (hB : ∀ p ∈ B, p.1 < W) (x : ℕ) (hx : W ≤ x ∧ x < 2 * W) (K i : ℕ) (hi : i < K) :
    labDim (lvEnv E W C B K) (x + i * W) = E.dimv ((C.lookup x).getD 0) := by
  simp only [labDim, lvEnv]
  cases hc : C.lookup x with
  | some v => rw [lookup_app_some _ _ _ _ ((lookup_lvls W C hC x hx K i hi).trans hc)]
  | none =>
    rw [lookup_app_none _ _ _ ((lookup_lvls W C hC x hx K i hi).trans hc),
      lookup_none_of B _ fun p hp => by have := hB p hp; omega]

theorem lv_lo (E : Env) (W : ℕ) (C B : List (ℕ × ℕ)) (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W)
    (x : ℕ) (hx : x < W) (K : ℕ) :
    labDim (lvEnv E W C B K) x = E.dimv ((B.lookup x).getD 0) := by
  simp only [labDim, lvEnv]
  rw [lookup_app_none _ _ _ (lookup_none_of _ _ fun p hp => ?_)]
  obtain ⟨q, hq, j, _, e⟩ := mem_lvls W C K p hp
  have := hC q hq
  omega

/-- Each label below `2 W` has the same dimension at every `K ≥ 1`. -/
theorem lv_one (E : Env) (W : ℕ) (C B : List (ℕ × ℕ)) (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W)
    (hB : ∀ p ∈ B, p.1 < W) (K : ℕ) (hK : 1 ≤ K) (x : ℕ) (hx : x < 2 * W) :
    labDim (lvEnv E W C B K) x = labDim (lvEnv E W C B 1) x := by
  by_cases hl : x < W
  · rw [lv_lo E W C B hC x hl K, lv_lo E W C B hC x hl 1]
  · have h := lv_hi E W C B hC hB x ⟨by omega, hx⟩ K 0 (by omega)
    have h' := lv_hi E W C B hC hB x ⟨by omega, hx⟩ 1 0 (by omega)
    simp only [zero_mul, add_zero] at h h'
    rw [h, h']

/-- An equation on labels below `2 W` at `K = 1` holds at every `K ≥ 1`. -/
theorem Equal.lvK {E : Env} {W : ℕ} {C B : List (ℕ × ℕ)} (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W)
    (hB : ∀ p ∈ B, p.1 < W) {s s' : ℂ} {X Y : Net} (h : Equal (lvEnv E W C B 1) s X s' Y)
    (hX : ∀ l ∈ X.int ++ X.bnd, l < 2 * W) (hY : ∀ l ∈ Y.int, l < 2 * W) (K : ℕ)
    (hK : 1 ≤ K) : Equal (lvEnv E W C B K) s X s' Y :=
  h.relabs (lvls W C K ++ B) (fun l hl => lv_one E W C B hC hB K hK l (hX l hl))
    (fun l hl => lv_one E W C B hC hB K hK l (hY l hl))

/-- The shift hypothesis of `chain_equal_le` for `lvEnv`, each cut label `c` and its image
`c + W` of one dimension. -/
theorem lv_hD (E : Env) (W : ℕ) (C B : List (ℕ × ℕ)) (cut : List ℕ) (c : ℕ → Net)
    (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W) (hB : ∀ p ∈ B, p.1 < W)
    (hcut : ∀ x ∈ cut, x < W ∧ labDim (lvEnv E W C B 1) (x + W) = labDim (lvEnv E W C B 1) x)
    (hc : ∀ i, copyOK W cut (c i) = true) (K : ℕ) :
    ∀ i < K, ∀ x ∈ labelsOf (c i).nodes ++ (c i).int ++ (c i).bnd,
      labDim (lvEnv E W C B K) (x + i * W) = labDim (lvEnv E W C B K) x := by
  intro i hi x hx
  have hci := hc i
  simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hci
  rcases hci.1 x hx with hcx | hw
  · obtain ⟨hxW, hd⟩ := hcut x hcx
    rcases Nat.eq_zero_or_pos i with h0 | hpos
    · subst h0
      simp
    · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
      have e : x + (j + 1) * W = (x + W) + j * W := by rw [Nat.succ_mul]; ring
      rw [e, lv_hi E W C B hC hB (x + W) ⟨by omega, by omega⟩ K j (by omega),
        lv_lo E W C B hC x hxW K]
      have h1 := lv_hi E W C B hC hB (x + W) ⟨by omega, by omega⟩ 1 0 (by omega)
      have h2 := lv_lo E W C B hC x hxW 1
      simp only [zero_mul, add_zero] at h1
      rw [← h1, ← h2]
      exact hd
  · rw [lv_hi E W C B hC hB x hw K i hi]
    have h := lv_hi E W C B hC hB x hw K 0 (by omega)
    simp only [zero_mul, add_zero] at h
    rw [h]

theorem copyOK_lt {W : ℕ} {cut : List ℕ} {c : Net} (hc : copyOK W cut c = true)
    (hcut : ∀ x ∈ cut, x < W) : ∀ l ∈ labelsOf c.nodes ++ c.int ++ c.bnd, l < 2 * W := by
  simp only [copyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hc
  intro l hl
  rcases hc.1 l hl with h | h
  · have := hcut l h
    omega
  · omega

theorem baseOK_lt {W : ℕ} {cut : List ℕ} {b : Net} (hb : baseOK W cut b = true) :
    ∀ l ∈ labelsOf b.nodes ++ b.int ++ b.bnd, l < W := by
  simp only [baseOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hb
  exact hb.1.1

theorem mem_all_of_ib {N : Net} {x : ℕ} (h : x ∈ N.int ++ N.bnd) :
    x ∈ labelsOf N.nodes ++ N.int ++ N.bnd := by
  simp only [List.mem_append] at h ⊢
  tauto

/-- Two base networks' equation at `K = 1` at every `K`. -/
theorem Equal.lvBase {E : Env} {W : ℕ} {C B : List (ℕ × ℕ)} {cut : List ℕ}
    (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W) {s s' : ℂ} {X Y : Net}
    (hX : baseOK W cut X = true) (hY : baseOK W cut Y = true)
    (h : Equal (lvEnv E W C B 1) s X s' Y) (K : ℕ) : Equal (lvEnv E W C B K) s X s' Y :=
  h.relabs (lvls W C K ++ B)
    (fun l hl => (lv_lo E W C B hC l (baseOK_lt hX l (mem_all_of_ib hl)) K).trans
      (lv_lo E W C B hC l (baseOK_lt hX l (mem_all_of_ib hl)) 1).symm)
    (fun l hl => (lv_lo E W C B hC l (baseOK_lt hY l (mem_all_of_int hl)) K).trans
      (lv_lo E W C B hC l (baseOK_lt hY l (mem_all_of_int hl)) 1).symm)

/-- Two copies' equation at `K = 1` at every `K ≥ 1`. -/
theorem Equal.lvCopy {E : Env} {W : ℕ} {C B : List (ℕ × ℕ)} {cut : List ℕ}
    (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W) (hB : ∀ p ∈ B, p.1 < W) (hcut : ∀ x ∈ cut, x < W)
    {s s' : ℂ} {X Y : Net} (hX : copyOK W cut X = true) (hY : copyOK W cut Y = true)
    (h : Equal (lvEnv E W C B 1) s X s' Y) (K : ℕ) (hK : 1 ≤ K) :
    Equal (lvEnv E W C B K) s X s' Y :=
  h.lvK hC hB (fun l hl => copyOK_lt hX hcut l (mem_all_of_ib hl))
    (fun l hl => copyOK_lt hY hcut l (mem_all_of_int hl)) K hK

/-- Two chain families over `lvEnv` agree at every `k`, from the base and copy equations at
`K = 1`. -/
theorem chain_equal_lv (E : Env) (W : ℕ) (cut : List ℕ) (bO : Net) (cO : ℕ → Net) (bX : Net)
    (cX : ℕ → Net) (C B : List (ℕ × ℕ)) (hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W)
    (hB : ∀ p ∈ B, p.1 < W)
    (hcut : ∀ x ∈ cut, x < W ∧ labDim (lvEnv E W C B 1) (x + W) = labDim (lvEnv E W C B 1) x)
    (hbO : baseOK W cut bO = true) (hcO : ∀ i, copyOK W cut (cO i) = true)
    (hbX : baseOK W cut bX = true) (hcX : ∀ i, copyOK W cut (cX i) = true)
    (s t : ℂ) (a b : ℕ → ℂ) (h0 : Equal (lvEnv E W C B 1) s bO t bX)
    (h1 : ∀ i, Equal (lvEnv E W C B 1) (a i) (cO i) (b i) (cX i)) :
    ∀ k, Equal (lvEnv E W C B k) (s * ∏ i ∈ range k, a i) (chainNet W cut bO cO k)
      (t * ∏ i ∈ range k, b i) (chainNet W cut bX cX k) := fun k =>
  chain_equal_le (lvEnv E W C B k) W cut bO cO bX cX k
    (lv_hD E W C B cut cO hC hB hcut hcO k) (lv_hD E W C B cut cX hC hB hcut hcX k)
    hbO hcO hbX hcX s t a b (h0.lvBase hC hbO hbX k)
    (fun i hi => (h1 i).lvCopy hC hB (fun x hx => (hcut x hx).1) (hcO i) (hcX i) k (by omega))
    k le_rfl

end ArchytasZX
