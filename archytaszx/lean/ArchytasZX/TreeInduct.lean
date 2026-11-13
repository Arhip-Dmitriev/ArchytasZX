/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainInduct

/-!
# Tree families by plugging one copy at a time

`treeNet W r p base copy n` is a tree box's network after `n` copies: `base` at `n = 0`, and at
`n + 1` copy `n` (labels shifted by `n W`, its root label `r` sent to the plug label `p n`)
plugged onto the network at `n` through `p n`, which becomes summed. With `heapPlug`, copy `n`
is plugged onto a child label of copy `(n - 1) / b` (heap order), so the tree of depth `k` is
`treeNet` at `(b ^ k - 1) / (b - 1)` copies and its leaf level is the last `b ^ (k - 1)`
copies. `tree_equal` proves both families equal at every `n` from the base case and the
equation of each copy pair, given `PlugOK` for both; `heap_plugOK` gives `PlugOK` for
`heapPlug` from checks on the child labels.
-/

namespace ArchytasZX

open Complex Finset

/-- Copy `t`'s labels shifted by `t W`, then its root `r + t W` swapped with the plug `p t`. -/
def plugMap (W r : ℕ) (p : ℕ → ℕ) (t : ℕ) (x : ℕ) : ℕ :=
  Equiv.swap (r + t * W) (p t) (x + t * W)

/-- The network of a tree box after `n` copies, copy `t` plugged at the label `p t`. -/
def treeNet (W r : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net) : ℕ → Net
  | 0 => base
  | t + 1 =>
    let C := (copy t).mapL (plugMap W r p t)
    ⟨C.nodes ++ (treeNet W r p base copy t).nodes,
      C.int ++ [p t] ++ (treeNet W r p base copy t).int,
      (treeNet W r p base copy t).bnd.filter (· ∉ [p t]) ++ C.bnd.filter (· ∉ [p t])⟩

/-- Every base label below `W`. -/
def tbaseOK (W : ℕ) (base : Net) : Bool :=
  (labelsOf base.nodes ++ base.int ++ base.bnd).all fun x => decide (x < W)

/-- The root label below `W`, every other copy label in `[W, 2 W)`, the summed ones too. -/
def tcopyOK (W r : ℕ) (copy : Net) : Bool :=
  decide (r < W) &&
    (labelsOf copy.nodes ++ copy.int ++ copy.bnd).all
      (fun x => decide (x = r ∨ (W ≤ x ∧ x < 2 * W))) &&
    copy.int.all fun x => decide (W ≤ x ∧ x < 2 * W)

/-- The plug labels: below the labels of later copies, off the base's summed labels, distinct,
and off every copy's summed labels. -/
structure PlugOK (W : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net) : Prop where
  lt : ∀ t, p t < t * W + W
  base : ∀ t, p t ∉ base.int
  ne : ∀ t s, s < t → p t ≠ p s
  own : ∀ t s, ∀ x ∈ (copy s).int, p t ≠ x + s * W

theorem plugMap_root (W r : ℕ) (p : ℕ → ℕ) (t : ℕ) : plugMap W r p t r = p t := by
  simp [plugMap]

theorem plugMap_own (W r : ℕ) (p : ℕ → ℕ) (t y : ℕ) (hr : r < W) (hy : W ≤ y)
    (hp : p t < t * W + W) : plugMap W r p t y = y + t * W := by
  unfold plugMap
  exact Equiv.swap_apply_of_ne_of_ne (by omega) (by omega)

theorem plugMap_injective (W r : ℕ) (p : ℕ → ℕ) (t : ℕ) : Function.Injective (plugMap W r p t) :=
  fun x y h => by
    have := (Equiv.swap (r + t * W) (p t)).injective h
    omega

theorem tcopy_labels {W r : ℕ} {c : Net} (h : tcopyOK W r c = true) :
    r < W ∧ (∀ x ∈ labelsOf c.nodes ++ c.int ++ c.bnd, x = r ∨ (W ≤ x ∧ x < 2 * W)) ∧
      ∀ x ∈ c.int, W ≤ x ∧ x < 2 * W := by
  simp only [tcopyOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

theorem mem_tree_succ (W r : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net) (n x : ℕ)
    (hx : x ∈ labelsOf (treeNet W r p base copy (n + 1)).nodes ++
      (treeNet W r p base copy (n + 1)).int ++ (treeNet W r p base copy (n + 1)).bnd) :
    x ∈ labelsOf ((copy n).mapL (plugMap W r p n)).nodes ++
        ((copy n).mapL (plugMap W r p n)).int ++ ((copy n).mapL (plugMap W r p n)).bnd ∨
      x = p n ∨
      x ∈ labelsOf (treeNet W r p base copy n).nodes ++ (treeNet W r p base copy n).int ++
        (treeNet W r p base copy n).bnd := by
  simp only [treeNet, labelsOf, List.flatMap_append, List.mem_append, List.mem_filter,
    List.mem_singleton] at hx ⊢
  tauto

/-- Every label of the network after `n` copies is below `n W + W`; each summed one is a base
summed label, an earlier plug label or an earlier copy's shifted summed label. -/
theorem tree_bounds (W r : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net)
    (hb : tbaseOK W base = true) (hc : ∀ i, tcopyOK W r (copy i) = true)
    (hlt : ∀ t, p t < t * W + W) (n : ℕ) :
    (∀ x ∈ labelsOf (treeNet W r p base copy n).nodes ++ (treeNet W r p base copy n).int ++
      (treeNet W r p base copy n).bnd, x < n * W + W) ∧
      ∀ x ∈ (treeNet W r p base copy n).int, x ∈ base.int ∨ (∃ s < n, x = p s) ∨
        ∃ s < n, ∃ y ∈ (copy s).int, x = y + s * W := by
  simp only [tbaseOK, List.all_eq_true, decide_eq_true_eq] at hb
  induction n with
  | zero => exact ⟨fun x hx => by simpa using hb x hx, fun x hx => Or.inl hx⟩
  | succ n ih =>
    obtain ⟨ih1, ih2⟩ := ih
    obtain ⟨hr, hcl, hci⟩ := tcopy_labels (hc n)
    have hsucc : (n + 1) * W = n * W + W := by rw [Nat.succ_mul]
    have hpn := hlt n
    refine ⟨fun x hx => ?_, fun x hx => ?_⟩
    · rw [hsucc]
      rcases mem_tree_succ W r p base copy n x hx with h | h | h
      · obtain ⟨y, hy, rfl⟩ := mem_labels_mapL (plugMap W r p n) (copy n) x h
        rcases hcl y hy with hyr | ⟨h1, h2⟩
        · rw [hyr, plugMap_root]; omega
        · rw [plugMap_own W r p n y hr h1 hpn]; omega
      · omega
      · have := ih1 x h; omega
    · simp only [treeNet, List.mem_append, List.mem_singleton] at hx
      rcases hx with (hx | hx) | hx
      · obtain ⟨y, hy, rfl⟩ := List.mem_map.1
          (show x ∈ (copy n).int.map (plugMap W r p n) from hx)
        refine Or.inr (Or.inr ⟨n, Nat.lt_succ_self n, y, hy, ?_⟩)
        exact plugMap_own W r p n y hr (hci y hy).1 hpn
      · exact Or.inr (Or.inl ⟨n, Nat.lt_succ_self n, hx⟩)
      · rcases ih2 x hx with h | ⟨s, hs, h⟩ | ⟨s, hs, y, hy, h⟩
        · exact Or.inl h
        · exact Or.inr (Or.inl ⟨s, Nat.lt_succ_of_lt hs, h⟩)
        · exact Or.inr (Or.inr ⟨s, Nat.lt_succ_of_lt hs, y, hy, h⟩)

theorem plug_off_int (W r : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net)
    (hb : tbaseOK W base = true) (hc : ∀ i, tcopyOK W r (copy i) = true)
    (hp : PlugOK W p base copy) (n : ℕ) : p n ∉ (treeNet W r p base copy n).int := by
  intro hx
  rcases (tree_bounds W r p base copy hb hc hp.lt n).2 _ hx with h | ⟨s, hs, h⟩ |
    ⟨s, _, y, hy, h⟩
  · exact hp.base n h
  · exact hp.ne n s hs h
  · exact hp.own n s y hy h

theorem tcopy_off (W r : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net)
    (hb : tbaseOK W base = true) (hc : ∀ i, tcopyOK W r (copy i) = true)
    (hp : PlugOK W p base copy) (n : ℕ) (cp : Net) (hcp : tcopyOK W r cp = true) :
    ∀ x ∈ labelsOf (cp.mapL (plugMap W r p n)).nodes, x ∉ (treeNet W r p base copy n).int := by
  intro x hx hxi
  obtain ⟨hr, hcl, _⟩ := tcopy_labels hcp
  obtain ⟨y, hy, rfl⟩ := mem_labels_mapL (plugMap W r p n) cp x (mem_all_of_nodes hx)
  rcases hcl y hy with hyr | ⟨h1, _⟩
  · rw [hyr, plugMap_root] at hxi
    exact plug_off_int W r p base copy hb hc hp n hxi
  · rw [plugMap_own W r p n y hr h1 (hp.lt n)] at hxi
    have := (tree_bounds W r p base copy hb hc hp.lt n).1 _ (mem_all_of_int hxi)
    omega

theorem tnet_off (W r : ℕ) (p : ℕ → ℕ) (base : Net) (copy : ℕ → Net)
    (hb : tbaseOK W base = true) (hc : ∀ i, tcopyOK W r (copy i) = true)
    (hlt : ∀ t, p t < t * W + W) (n : ℕ) (cp : Net) (hcp : tcopyOK W r cp = true) :
    ∀ x ∈ labelsOf (treeNet W r p base copy n).nodes, x ∉ (cp.mapL (plugMap W r p n)).int := by
  intro x hx hxi
  obtain ⟨hr, _, hci⟩ := tcopy_labels hcp
  obtain ⟨y, hy, rfl⟩ := List.mem_map.1 (show x ∈ cp.int.map (plugMap W r p n) from hxi)
  have h1 := (tree_bounds W r p base copy hb hc hlt n).1 _ (mem_all_of_nodes hx)
  rw [plugMap_own W r p n y hr (hci y hy).1 (hlt n)] at h1
  have := (hci y hy).1
  omega

/-- Two tree families agree after every number of copies from the base case and the equation
of each copy pair, copy `i` scaled by `a i` and `b i`. -/
theorem tree_equal (E : Env) (W r : ℕ) (p : ℕ → ℕ) (bO : Net) (cO : ℕ → Net) (bX : Net)
    (cX : ℕ → Net) (hD : ∀ x y, labDim E x = labDim E y)
    (hbO : tbaseOK W bO = true) (hcO : ∀ i, tcopyOK W r (cO i) = true)
    (hbX : tbaseOK W bX = true) (hcX : ∀ i, tcopyOK W r (cX i) = true)
    (hpO : PlugOK W p bO cO) (hpX : PlugOK W p bX cX)
    (s t : ℂ) (a b : ℕ → ℂ) (h0 : Equal E s bO t bX) (h1 : ∀ i, Equal E (a i) (cO i) (b i) (cX i)) :
    ∀ n, Equal E (s * ∏ i ∈ range n, a i) (treeNet W r p bO cO n)
      (t * ∏ i ∈ range n, b i) (treeNet W r p bX cX n) := by
  intro n
  induction n with
  | zero => simpa [treeNet] using h0
  | succ k ih =>
    set f := plugMap W r p k with hf
    have hinj : Function.Injective f := plugMap_injective W r p k
    have hcc : Equal E (a k) ((cO k).mapL f) (b k) ((cX k).mapL f) :=
      Equal.mapL E _ hinj (fun x => hD _ _) (h1 k)
    have A1 := plug E ((cO k).mapL f).nodes (treeNet W r p bO cO k).nodes
      (treeNet W r p bX cX k).nodes (((cO k).mapL f).int ++ [p k])
      (treeNet W r p bO cO k).int (treeNet W r p bX cX k).int
      ((treeNet W r p bO cO k).bnd.filter (· ∉ [p k]) ++
        ((cO k).mapL f).bnd.filter (· ∉ [p k]))
      (treeNet W r p bO cO k).bnd _ _ (Equal.toLocal E _ _ ih)
      (tcopy_off W r p bO cO hbO hcO hpO k (cO k) (hcO k))
      (tcopy_off W r p bX cX hbX hcX hpX k (cO k) (hcO k))
      (face_left _ _ _ _)
    have A2 := plug E (treeNet W r p bX cX k).nodes ((cO k).mapL f).nodes
      ((cX k).mapL f).nodes ([p k] ++ (treeNet W r p bX cX k).int)
      ((cO k).mapL f).int ((cX k).mapL f).int
      ((treeNet W r p bO cO k).bnd.filter (· ∉ [p k]) ++
        ((cO k).mapL f).bnd.filter (· ∉ [p k]))
      ((cO k).mapL f).bnd _ _ (Equal.toLocal E _ _ hcc)
      (tnet_off W r p bX cX hbX hcX hpX.lt k (cO k) (hcO k))
      (tnet_off W r p bX cX hbX hcX hpX.lt k (cX k) (hcX k))
      (face_right _ _ _ _)
    have hn : ∀ (A B : List Node), (A ++ B).Perm (B ++ A) := fun A B => List.perm_append_comm
    have hi : ∀ (A B C : List ℕ), (A ++ B ++ C).Perm (C ++ A ++ B) := fun A B C =>
      List.perm_append_comm.trans (List.Perm.of_eq (List.append_assoc _ _ _).symm)
    have A2' : Equal E (a k)
        ⟨((cO k).mapL f).nodes ++ (treeNet W r p bX cX k).nodes,
          ((cO k).mapL f).int ++ [p k] ++ (treeNet W r p bX cX k).int,
          (treeNet W r p bO cO k).bnd.filter (· ∉ [p k]) ++
            ((cO k).mapL f).bnd.filter (· ∉ [p k])⟩ (b k)
        ⟨((cX k).mapL f).nodes ++ (treeNet W r p bX cX k).nodes,
          ((cX k).mapL f).int ++ [p k] ++ (treeNet W r p bX cX k).int,
          (treeNet W r p bO cO k).bnd.filter (· ∉ [p k]) ++
            ((cO k).mapL f).bnd.filter (· ∉ [p k])⟩ :=
      (A2.reorderL E (A' := ⟨((cO k).mapL f).nodes ++ (treeNet W r p bX cX k).nodes,
          ((cO k).mapL f).int ++ [p k] ++ (treeNet W r p bX cX k).int,
          (treeNet W r p bO cO k).bnd.filter (· ∉ [p k]) ++
            ((cO k).mapL f).bnd.filter (· ∉ [p k])⟩)
        (hn _ _) (hi _ _ _) (List.Perm.refl _)).reorder E
        (B' := ⟨((cX k).mapL f).nodes ++ (treeNet W r p bX cX k).nodes,
          ((cX k).mapL f).int ++ [p k] ++ (treeNet W r p bX cX k).int,
          (treeNet W r p bO cO k).bnd.filter (· ∉ [p k]) ++
            ((cO k).mapL f).bnd.filter (· ∉ [p k])⟩)
        (hn _ _) (hi _ _ _) (List.Perm.refl _)
    have hb := (ih.1.filter (fun x => decide (x ∉ [p k]))).append
      (hcc.1.filter (fun x => decide (x ∉ [p k])))
    have A3 := (A1.mul_trans A2').reorder E (B' := treeNet W r p bX cX (k + 1))
      (List.Perm.refl _) (List.Perm.refl _) hb
    exact A3.congr_scalar (a' := s * ∏ i ∈ range (k + 1), a i)
      (b' := t * ∏ i ∈ range (k + 1), b i)
      (by rw [Finset.prod_range_succ]; ring) (by rw [Finset.prod_range_succ]; ring)

/-! ## Heap order -/

/-- The plug of copy `0` at `leaf0`, of copy `t + 1` at child label `t mod b` of copy
`t / b`, `b` the number of child labels. -/
def heapPlug (W leaf0 : ℕ) (kids : List ℕ) : ℕ → ℕ
  | 0 => leaf0
  | t + 1 => kids.getD (t % kids.length) 0 + t / kids.length * W

/-- At least one child label and one copy; the child labels in `[W, 2 W)`, distinct and off
every copy's summed labels; `leaf0` below `W` and off the base's summed labels. -/
def heapOK (W leaf0 : ℕ) (kids : List ℕ) (base : Net) (copies : List Net) : Bool :=
  decide (0 < kids.length) && decide (0 < copies.length) && decide kids.Nodup &&
    kids.all (fun x => decide (W ≤ x ∧ x < 2 * W)) && decide (leaf0 < W) &&
    decide (leaf0 ∉ base.int) && copies.all fun c => kids.all fun x => decide (x ∉ c.int)

theorem kid_mem (kids : List ℕ) (hk : 0 < kids.length) (t : ℕ) :
    kids.getD (t % kids.length) 0 ∈ kids := by
  rw [List.getD_eq_getElem _ _ (Nat.mod_lt _ hk)]
  exact List.getElem_mem _

theorem div_W (W x q : ℕ) (h1 : W ≤ x) (h2 : x < 2 * W) : (x + q * W) / W = q + 1 := by
  apply Nat.div_eq_of_lt_le
  · rw [Nat.add_mul, one_mul]; omega
  · rw [Nat.add_mul, Nat.add_mul, one_mul]; omega

theorem heap_plugOK (W r leaf0 : ℕ) (kids : List ℕ) (base : Net) (copies : List Net) (d : Net)
    (hb : tbaseOK W base = true) (hc : ∀ i, tcopyOK W r (perIdx copies d i) = true)
    (h : heapOK W leaf0 kids base copies = true) :
    PlugOK W (heapPlug W leaf0 kids) base (perIdx copies d) := by
  simp only [heapOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨hk, hcl⟩, hnd⟩, hkw⟩, hl0⟩, hlb⟩, hci⟩ := h
  simp only [tbaseOK, List.all_eq_true, decide_eq_true_eq] at hb
  have kw : ∀ t, W ≤ kids.getD (t % kids.length) 0 ∧ kids.getD (t % kids.length) 0 < 2 * W :=
    fun t => hkw _ (kid_mem kids hk t)
  refine ⟨fun t => ?_, fun t => ?_, fun t s hst => ?_, fun t s x hx => ?_⟩
  · cases t with
    | zero => simpa [heapPlug] using hl0
    | succ t =>
      simp only [heapPlug]
      have h1 := kw t
      have h2 := Nat.mul_le_mul_right W (Nat.div_le_self t kids.length)
      rw [Nat.succ_mul]
      omega
  · cases t with
    | zero => simpa [heapPlug] using hlb
    | succ t =>
      intro hm
      have h1 := hb _ (mem_all_of_int hm)
      have h2 := kw t
      simp only [heapPlug] at h1
      omega
  · cases t with
    | zero => omega
    | succ t =>
      cases s with
      | zero =>
        simp only [heapPlug]
        have := kw t
        omega
      | succ s =>
        simp only [heapPlug]
        intro he
        have ht := kw t
        have hs := kw s
        have hq : t / kids.length = s / kids.length := by
          have e1 := div_W W _ (t / kids.length) ht.1 ht.2
          have e2 := div_W W _ (s / kids.length) hs.1 hs.2
          rw [he] at e1
          omega
        rw [hq] at he
        have hg : kids.getD (t % kids.length) 0 = kids.getD (s % kids.length) 0 := by omega
        rw [List.getD_eq_getElem _ _ (Nat.mod_lt _ hk),
          List.getD_eq_getElem _ _ (Nat.mod_lt _ hk)] at hg
        have hm := (List.Nodup.getElem_inj_iff hnd).1 hg
        have e3 := Nat.div_add_mod t kids.length
        have e4 := Nat.div_add_mod s kids.length
        rw [hq, hm] at e3
        omega
  · have hxw := ((tcopy_labels (hc s)).2.2 x hx)
    have hxc : x ∉ kids := by
      intro hm
      have hs := Nat.mod_lt s hcl
      have := hci _ (List.getElem_mem hs) x hm
      simp only [perIdx, List.getD_eq_getElem _ _ hs] at hx
      exact this hx
    cases t with
    | zero =>
      simp only [heapPlug]
      have := Nat.zero_le (s * W)
      omega
    | succ t =>
      simp only [heapPlug]
      intro he
      have ht := kw t
      have hq : t / kids.length = s := by
        have e1 := div_W W _ (t / kids.length) ht.1 ht.2
        have e2 := div_W W x s hxw.1 hxw.2
        rw [he] at e1
        omega
      rw [hq] at he
      have hg : kids.getD (t % kids.length) 0 = x := by omega
      exact hxc (hg ▸ kid_mem kids hk t)

end ArchytasZX
