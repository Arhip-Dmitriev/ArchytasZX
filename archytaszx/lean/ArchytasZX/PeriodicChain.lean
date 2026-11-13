/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Periodic

/-!
# Periodic chain families

`famNet W u tl cp hd bd k` is a closed chain family: a fixed end `tl` on the link labels `u`,
`k` copies of `cp` (copy `j` raised `j` levels of `W` by `sh`), and the moving end `hd` raised
`k` levels; labels below `W` are never raised. `fam_step` gives the family at `k + p` equal to
`c` times the family at `k` from one equation, `p` copies then `hd` equal to `c` times `hd`
(`blkNet`, `headNet`), plugged in place of the last `p` copies and `hd`. `fam_periodic` gives
the family equal to a fixed network `G` at every `k` from the pairs at `0 .. p - 1`, by
`Equal.periodic` with the identity relabelling.
-/

namespace ArchytasZX

/-- `x` kept below `W`, else raised by `k` levels of `W`. -/
def sh (W k x : ℕ) : ℕ := if x < W then x else x + k * W

theorem sh_lt {W k x : ℕ} (h : x < W) : sh W k x = x := by simp [sh, h]

theorem sh_ge {W k x : ℕ} (h : W ≤ x) : sh W k x = x + k * W := by simp [sh, Nat.not_lt.2 h]

theorem sh_sh (W k j x : ℕ) : sh W k (sh W j x) = sh W (k + j) x := by
  by_cases h : x < W
  · rw [sh_lt h, sh_lt h, sh_lt h]
  · have h' : W ≤ x := Nat.le_of_not_lt h
    rw [sh_ge h', sh_ge (by omega), sh_ge h']
    ring

theorem sh_zero (W x : ℕ) : sh W 0 x = x := by simp [sh]

theorem sh_inj (W k : ℕ) : Function.Injective (sh W k) := by
  intro x y h
  by_cases hx : x < W <;> by_cases hy : y < W
  · rwa [sh_lt hx, sh_lt hy] at h
  · rw [sh_lt hx, sh_ge (Nat.le_of_not_lt hy)] at h; omega
  · rw [sh_ge (Nat.le_of_not_lt hx), sh_lt hy] at h; omega
  · rw [sh_ge (Nat.le_of_not_lt hx), sh_ge (Nat.le_of_not_lt hy)] at h; omega

/-- Two labels in `[W, 2 W)` raised by `j` and `m` levels meet only at the same level and
label. -/
theorem lev_eq {W a b j m : ℕ} (ha : W ≤ a ∧ a < 2 * W) (hb : W ≤ b ∧ b < 2 * W)
    (h : a + j * W = b + m * W) : j = m ∧ a = b := by
  have hW : 0 < W := by omega
  have e1 : (a + j * W) / W = 1 + j := by
    rw [Nat.add_mul_div_right _ _ hW,
      Nat.div_eq_of_lt_le (k := 1) (by omega : 1 * W ≤ a) (by omega : a < (1 + 1) * W)]
  have e2 : (b + m * W) / W = 1 + m := by
    rw [Nat.add_mul_div_right _ _ hW,
      Nat.div_eq_of_lt_le (k := 1) (by omega : 1 * W ≤ b) (by omega : b < (1 + 1) * W)]
  have : j = m := by rw [h] at e1; omega
  subst this
  exact ⟨rfl, by omega⟩

theorem rename_rename (f g : ℕ → ℕ) (n : Node) :
    Node.rename f (Node.rename g n) = Node.rename (f ∘ g) n := by
  simp [Node.rename, List.map_map]

theorem mapL_mapL (f g : ℕ → ℕ) (N : Net) : (N.mapL g).mapL f = N.mapL (f ∘ g) := by
  simp [Net.mapL, List.map_map, Function.comp_def, rename_rename]

theorem mapL_id (N : Net) : N.mapL id = N := by
  have : Node.rename id = id := funext fun n => by cases n; simp [Node.rename]
  simp [Net.mapL, this]

theorem sh_comp (W k j : ℕ) : sh W k ∘ sh W j = sh W (k + j) := funext (sh_sh W k j)

theorem rename_sh (W k j : ℕ) :
    Node.rename (sh W k) ∘ Node.rename (sh W j) = Node.rename (sh W (k + j)) :=
  funext fun n => by simp [rename_rename, sh_comp]

/-- A closed chain family at `k` copies: the tail `tl` on the link labels `u`, copy `j`
(`cp` raised `j` levels) from `u` at level `j` to `u` at level `j + 1`, the head `hd` raised
`k` levels; labels below `W` are never raised. -/
def famNet (W : ℕ) (u : List ℕ) (tl cp hd : Net) (bd : List ℕ) (k : ℕ) : Net :=
  ⟨tl.nodes ++ (List.range k).flatMap (fun j => (cp.mapL (sh W j)).nodes) ++
      (hd.mapL (sh W k)).nodes,
    tl.int ++ u.filter (· ∉ bd) ++
      (List.range k).flatMap (fun j => cp.int.map (sh W j) ++ u.map (sh W (j + 1))) ++
      hd.int,
    bd⟩

/-- `p` copies and the head raised `p` levels, open on `u` and the head's boundary. -/
def blkNet (W : ℕ) (u : List ℕ) (cp hd : Net) (p : ℕ) : Net :=
  ⟨(List.range p).flatMap (fun j => (cp.mapL (sh W j)).nodes) ++ (hd.mapL (sh W p)).nodes,
    (List.range p).flatMap (fun j => cp.int.map (sh W j) ++ u.map (sh W (j + 1))) ++ hd.int,
    u ++ hd.bnd⟩

/-- The head open on `u` and its boundary. -/
def headNet (u : List ℕ) (hd : Net) : Net := ⟨hd.nodes, hd.int, u ++ hd.bnd⟩

/-- The label conditions of `famNet`: `u` and the copy's summed labels in `[W, 2 W)`, the
copy's legs on `u`, `u` one level up, or its summed labels; the head's and tail's legs on `u` or
below `W`, the head's summed and boundary labels below `W`, the tail's labels off the head's
summed ones, the head's boundary in `bd`. -/
def famOK (W : ℕ) (u : List ℕ) (tl cp hd : Net) (bd : List ℕ) : Bool :=
  u.all (fun x => decide (W ≤ x ∧ x < 2 * W)) &&
  cp.int.all (fun x => decide (W ≤ x ∧ x < 2 * W ∧ x ∉ u)) &&
  (labelsOf cp.nodes).all (fun x => decide (x ∈ u ∨ x ∈ u.map (· + W) ∨ x ∈ cp.int)) &&
  (labelsOf hd.nodes).all (fun x => decide (x ∈ u ∨ x < W)) &&
  (hd.int ++ hd.bnd).all (fun x => decide (x < W)) &&
  (labelsOf tl.nodes ++ tl.int).all (fun x => decide ((x ∈ u ∨ x < W) ∧ x ∉ hd.int)) &&
  hd.bnd.all (fun x => decide (x ∈ bd))

section step

variable {W : ℕ} {u : List ℕ} {tl cp hd : Net} {bd : List ℕ}

theorem famOK_spec (h : famOK W u tl cp hd bd = true) :
    (∀ x ∈ u, W ≤ x ∧ x < 2 * W) ∧ (∀ x ∈ cp.int, W ≤ x ∧ x < 2 * W ∧ x ∉ u) ∧
      (∀ x ∈ labelsOf cp.nodes, x ∈ u ∨ x ∈ u.map (· + W) ∨ x ∈ cp.int) ∧
      (∀ x ∈ labelsOf hd.nodes, x ∈ u ∨ x < W) ∧ (∀ x ∈ hd.int ++ hd.bnd, x < W) ∧
      (∀ x ∈ labelsOf tl.nodes ++ tl.int, (x ∈ u ∨ x < W) ∧ x ∉ hd.int) ∧
      ∀ x ∈ hd.bnd, x ∈ bd := by
  simp only [famOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1.1.1.1, h.1.1.1.1.1.2, h.1.1.1.1.2, h.1.1.1.2, h.1.1.2, h.1.2, h.2⟩

theorem mem_ctx (k x : ℕ)
    (hx : x ∈ labelsOf (tl.nodes ++ (List.range k).flatMap fun j => (cp.mapL (sh W j)).nodes)) :
    x ∈ labelsOf tl.nodes ∨ ∃ j < k, ∃ y ∈ labelsOf cp.nodes, x = sh W j y := by
  simp only [labelsOf, List.flatMap_append, List.mem_append, List.mem_flatMap, List.mem_range,
    Net.mapL, List.mem_map] at hx
  rcases hx with h | ⟨n, ⟨j, hj, m, hm, rfl⟩, hx⟩
  · exact Or.inl (by simpa [labelsOf, List.mem_flatMap] using h)
  · right
    refine ⟨j, hj, ?_⟩
    simp only [Node.labels, Node.rename, List.mem_append, List.mem_map] at hx
    rcases hx with ⟨y, hy, rfl⟩ | ⟨y, hy, rfl⟩
    · exact ⟨y, List.mem_flatMap.2 ⟨m, hm, by simp [Node.labels, hy]⟩, rfl⟩
    · exact ⟨y, List.mem_flatMap.2 ⟨m, hm, by simp [Node.labels, hy]⟩, rfl⟩

theorem ctx_off_blk (h : famOK W u tl cp hd bd = true) (k p x : ℕ)
    (hx : x ∈ labelsOf (tl.nodes ++ (List.range k).flatMap fun j => (cp.mapL (sh W j)).nodes)) :
    x ∉ (List.range p).flatMap (fun j => cp.int.map (sh W (k + j)) ++ u.map (sh W (k + j + 1))) ++
      hd.int := by
  obtain ⟨hu, hci, hcl, -, hhb, htl, -⟩ := famOK_spec h
  intro hm
  simp only [List.mem_append, List.mem_flatMap, List.mem_range, List.mem_map] at hm
  rcases mem_ctx k x hx with hx | ⟨j, hj, y, hy, rfl⟩
  · have ht := htl x (List.mem_append_left _ hx)
    rcases hm with ⟨i, hi, ⟨c, hc, he⟩ | ⟨v, hv, he⟩⟩ | hm
    · obtain ⟨c1, c2, c3⟩ := hci c hc
      rw [sh_ge c1] at he
      rcases ht.1 with hxu | hxw
      · have := lev_eq (j := 0) (m := k + i) (hu x hxu) ⟨c1, c2⟩ (by rw [he]; ring)
        exact c3 (this.2 ▸ hxu)
      · omega
    · obtain ⟨v1, v2⟩ := hu v hv
      rw [sh_ge v1] at he
      rcases ht.1 with hxu | hxw
      · have := lev_eq (j := 0) (m := k + i + 1) (hu x hxu) ⟨v1, v2⟩ (by rw [he]; ring)
        omega
      · omega
    · exact ht.2 hm
  · rcases hcl y hy with hyu | hyw | hyc
    · obtain ⟨y1, y2⟩ := hu y hyu
      rw [sh_ge y1] at hm
      rcases hm with ⟨i, hi, ⟨c, hc, he⟩ | ⟨v, hv, he⟩⟩ | hm
      · obtain ⟨c1, c2, c3⟩ := hci c hc
        rw [sh_ge c1] at he
        have := lev_eq ⟨c1, c2⟩ ⟨y1, y2⟩ he
        omega
      · obtain ⟨v1, v2⟩ := hu v hv
        rw [sh_ge v1] at he
        have := lev_eq ⟨v1, v2⟩ ⟨y1, y2⟩ he
        omega
      · have := hhb _ (List.mem_append_left _ hm); omega
    · obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hyw
      obtain ⟨y1, y2⟩ := hu v hv
      rw [sh_ge (by omega)] at hm
      rcases hm with ⟨i, hi, ⟨c, hc, he⟩ | ⟨v', hv', he⟩⟩ | hm
      · obtain ⟨c1, c2, c3⟩ := hci c hc
        rw [sh_ge c1] at he
        have := lev_eq (j := k + i) (m := j + 1) ⟨c1, c2⟩ ⟨y1, y2⟩ (by rw [he]; ring)
        exact c3 (this.2 ▸ hv)
      · obtain ⟨v1, v2⟩ := hu v' hv'
        rw [sh_ge v1] at he
        have := lev_eq (j := k + i + 1) (m := j + 1) ⟨v1, v2⟩ ⟨y1, y2⟩ (by rw [he]; ring)
        omega
      · have := hhb _ (List.mem_append_left _ hm); omega
    · obtain ⟨y1, y2, -⟩ := hci y hyc
      rw [sh_ge y1] at hm
      rcases hm with ⟨i, hi, ⟨c, hc, he⟩ | ⟨v, hv, he⟩⟩ | hm
      · obtain ⟨c1, c2, c3⟩ := hci c hc
        rw [sh_ge c1] at he
        have := lev_eq ⟨c1, c2⟩ ⟨y1, y2⟩ he
        omega
      · obtain ⟨v1, v2⟩ := hu v hv
        rw [sh_ge v1] at he
        have := lev_eq ⟨v1, v2⟩ ⟨y1, y2⟩ he
        omega
      · have := hhb _ (List.mem_append_left _ hm); omega

theorem ctx_off_head (h : famOK W u tl cp hd bd = true) (k x : ℕ)
    (hx : x ∈ labelsOf (tl.nodes ++ (List.range k).flatMap fun j => (cp.mapL (sh W j)).nodes)) :
    x ∉ hd.int := by
  have := ctx_off_blk h k 0 x hx
  simpa using this

theorem famNet_add (k p : ℕ) : famNet W u tl cp hd bd (k + p) =
    ⟨(tl.nodes ++ (List.range k).flatMap fun j => (cp.mapL (sh W j)).nodes) ++
        ((List.range p).flatMap (fun j => (cp.mapL (sh W (k + j))).nodes) ++
          (hd.mapL (sh W (k + p))).nodes),
      (tl.int ++ u.filter (· ∉ bd) ++
          (List.range k).flatMap fun j => cp.int.map (sh W j) ++ u.map (sh W (j + 1))) ++
        ((List.range p).flatMap (fun j => cp.int.map (sh W (k + j)) ++ u.map (sh W (k + j + 1))) ++
          hd.int),
      bd⟩ := by
  simp only [famNet, List.range_add, List.flatMap_append, List.flatMap_map, List.append_assoc,
    Nat.add_assoc]

theorem blk_mapL (h : famOK W u tl cp hd bd = true) (k p : ℕ) :
    (blkNet W u cp hd p).mapL (sh W k) =
      ⟨(List.range p).flatMap (fun j => (cp.mapL (sh W (k + j))).nodes) ++
          (hd.mapL (sh W (k + p))).nodes,
        (List.range p).flatMap (fun j => cp.int.map (sh W (k + j)) ++ u.map (sh W (k + j + 1))) ++
          hd.int,
        u.map (sh W k) ++ hd.bnd⟩ := by
  obtain ⟨-, -, -, -, hhb, -, -⟩ := famOK_spec h
  have hi : hd.int.map (sh W k) = hd.int :=
    (List.map_congr_left (g := id) fun x hx => sh_lt (hhb x (List.mem_append_left _ hx))).trans
      (List.map_id _)
  have hb : hd.bnd.map (sh W k) = hd.bnd :=
    (List.map_congr_left (g := id) fun x hx => sh_lt (hhb x (List.mem_append_right _ hx))).trans
      (List.map_id _)
  simp only [blkNet, Net.mapL, List.map_append, List.map_flatMap, List.map_map, hi, hb,
    rename_sh, sh_comp, Nat.add_assoc]

theorem head_mapL (h : famOK W u tl cp hd bd = true) (k : ℕ) :
    (headNet u hd).mapL (sh W k) = ⟨(hd.mapL (sh W k)).nodes, hd.int, u.map (sh W k) ++ hd.bnd⟩ := by
  obtain ⟨-, -, -, -, hhb, -, -⟩ := famOK_spec h
  have hi : hd.int.map (sh W k) = hd.int :=
    (List.map_congr_left (g := id) fun x hx => sh_lt (hhb x (List.mem_append_left _ hx))).trans
      (List.map_id _)
  have hb : hd.bnd.map (sh W k) = hd.bnd :=
    (List.map_congr_left (g := id) fun x hx => sh_lt (hhb x (List.mem_append_right _ hx))).trans
      (List.map_id _)
  simp only [headNet, Net.mapL, List.map_append, hi, hb]

/-- `p` more copies scale the family by `c` when the head after `p` copies is `c` times the
head. -/
theorem fam_step (E : Env) (hD : ∀ x y, labDim E x = labDim E y)
    (h : famOK W u tl cp hd bd = true) (p : ℕ) (c : ℂ)
    (h0 : Equal E 1 (blkNet W u cp hd p) c (headNet u hd)) (k : ℕ) :
    Equal E 1 (famNet W u tl cp hd bd (k + p)) c (famNet W u tl cp hd bd k) := by
  obtain ⟨hu, -, -, -, -, -, hbd⟩ := famOK_spec h
  have hk := Equal.mapL E (sh W k) (sh_inj W k) (fun x => hD _ _) h0
  rw [blk_mapL h k p, head_mapL h k] at hk
  rw [famNet_add]
  refine plug E _ _ _ _ _ _ bd _ 1 c (Equal.toLocal E _ _ hk) (fun x hx => ctx_off_blk h k p x hx)
    (fun x hx => ctx_off_head h k x hx) fun x hx => ?_
  rcases List.mem_append.1 hx with hx | hx
  · obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hx
    cases k with
    | zero =>
      rw [sh_zero]
      by_cases hb : v ∈ bd
      · exact Or.inr hb
      · left; simp [hv, hb]
    | succ k =>
      left
      simp only [List.mem_append, List.mem_flatMap, List.mem_range, List.mem_map]
      exact Or.inr ⟨k, by omega, Or.inr ⟨v, hv, rfl⟩⟩
  · exact Or.inr (hbd x hx)

/-- A table read periodically, times `A` per period. -/
noncomputable def perS (tab : List ℂ) (A : ℂ) (p k : ℕ) : ℂ := tab.getD (k % p) 0 * A ^ (k / p)

theorem perS_step (tab : List ℂ) (A : ℂ) (p : ℕ) (hp : 0 < p) (k : ℕ) :
    perS tab A p (k + p) = A * perS tab A p k := by
  simp only [perS, Nat.add_mod_right, Nat.add_div_right _ hp, pow_succ]
  ring

theorem perS_lt (tab : List ℂ) (A : ℂ) (p r : ℕ) (hr : r < p) :
    perS tab A p r = tab.getD r 0 := by
  simp [perS, Nat.mod_eq_of_lt hr, Nat.div_eq_of_lt hr]

/-- The family equals the fixed network `G` at every `k`, from the pairs at `0 .. p - 1` and
the head after `p` copies `c` times the head; the scalars grow by `A` and `A c` per period. -/
theorem fam_periodic (E : Env) (hD : ∀ x y, labDim E x = labDim E y)
    (h : famOK W u tl cp hd bd = true) (p : ℕ) (hp : 0 < p) (c A : ℂ)
    (h0 : Equal E 1 (blkNet W u cp hd p) c (headNet u hd)) (G : Net) (s g : ℕ → ℂ)
    (hs : ∀ k, s (k + p) = A * s k) (hg : ∀ k, g (k + p) = A * c * g k)
    (base : ∀ r < p, Equal E (s r) (famNet W u tl cp hd bd r) (g r) G) :
    ∀ k, Equal E (s k) (famNet W u tl cp hd bd k) (g k) G := by
  refine Equal.periodic E p hp (famNet W u tl cp hd bd) (fun _ => G) s g (A * c) (fun _ => id)
    (fun _ => Function.injective_id) (fun _ _ => rfl) base (fun k => ?_) (fun k => ?_)
  · rw [mapL_id]
    refine (fam_step E hD h p c h0 k).rescale one_ne_zero ?_
    rw [hs]; ring
  · rw [mapL_id, hg]
    exact ⟨List.Perm.refl _, fun _ _ => by ring⟩

end step

end ArchytasZX
