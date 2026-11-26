/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamDraperSpec
import ArchytasZX.FamDraperEq
import ArchytasZX.FamFcaSpecNet

/-!
# The Draper adder spec's expansion as `engR`

`lab1 n`: the label of each port of `D1.famExpand n` in `DraperEng.engR n`, read from the template
node and copy (`orgOf`). For `n > 0`: `nodesR`, its nodes under `lab1 n` are a permutation of
`engR n`'s; `wires1`, its wires are the `z`–`x` wire and the four chains; `intR`, `hwR`, `portsR`,
`dimW`; `evalR0`, its entry over `engE d n` is that of `engR n`.
-/

namespace ArchytasZX

namespace FamDraper

open List DraperEng
open FcaEng (tri dI)

/-- The `engR` label of the leg `(out, k)` of copy `j` of the template node `x`. -/
def labY (n x j : ℕ) (out : Bool) (k : ℕ) : ℕ :=
  match x, out, k with
  | 0, true, _ => rl n 0
  | 4, true, _ => rl n 2
  | 1, true, _ => bv n 0 j
  | 1, false, 0 => bw n 0 j
  | 1, false, _ => 2 * n + 2 * j
  | 5, true, _ => bv n 1 j
  | 5, false, 0 => bw n 1 j
  | 5, false, _ => 2 * n + 2 * j + 1
  | 2, true, 0 => sv n 0 j
  | 2, true, _ => 2 * j
  | 2, false, _ => sv n 0 (j + 1)
  | 6, true, 0 => sv n 1 j
  | 6, true, _ => 2 * j + 1
  | 6, false, _ => sv n 1 (j + 1)
  | 3, false, _ => rl n 1
  | 7, false, _ => rl n 3
  | 8, true, 0 => rl n 5
  | 8, true, _ => rl n 6
  | 8, false, _ => rl n 4
  | 9, true, _ => rl n 8
  | 9, false, 0 => rl n 7
  | 9, false, _ => rl n 6
  | _, _, _ => 0

/-- The `engR` label of each port of `D1.famExpand n`. -/
def lab1 (n : ℕ) (p : Port) : ℕ := labY n (D1.orgOf n p.node).1 (D1.orgOf n p.node).2 p.out p.pos

theorem lab1_at {n y : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (hy : e.1.id = y) (out : Bool) (k : ℕ) :
    lab1 n ⟨y, out, k⟩ = labY n e.2.1 e.2.2 out k := by
  simp only [lab1, orgOf1_eq he hy]

theorem lab1_eq {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (out : Bool) (k : ℕ) :
    lab1 n ⟨e.1.id, out, k⟩ = labY n e.2.1 e.2.2 out k := lab1_at he rfl out k

theorem side1 (n x j : ℕ) (out : Bool) (ds : List ℕ) : (D1.side n x j out ds).length = ds.length := by
  simp [FamDiagram.side, FamDiagram.cnt_plain n (show D1.pairs = [] from rfl), FamDiagram.growC_one]

theorem node1_lab {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) :
    (D1.node n e.1).node (lab1 n) =
      ⟨e.1.kind, (range e.1.outs.length).map (labY n e.2.1 e.2.2 true),
        (range e.1.ins.length).map (labY n e.2.1 e.2.2 false),
        FamDiagram.ixAt n D1.dim e.2.1 e.2.2 e.1.dim, FamDiagram.ixAt n D1.ph e.2.1 e.2.2 e.1.ph⟩ := by
  simp only [FamDiagram.node, orgOf1_eq he rfl, DNode.node, side1]
  congr 1
  · exact List.map_congr_left fun k _ => lab1_eq he true k
  · exact List.map_congr_left fun k _ => lab1_eq he false k

theorem nodes1_fam (n : ℕ) : (D1.famExpand n).nodes = (Z1 n).map fun e => D1.node n e.1 := by
  simp only [FamDiagram.famExpand, show D1.pairs = [] from rfl, List.zipIdx_nil, List.flatMap_nil,
    List.append_nil]
  rw [← zip1, X1]
  have hl : (E2n n).nodes.length = (D1.org n).length := by rw [org1]; simp [E2n]
  rw [show (fun e : DNode × ℕ × ℕ => D1.node n e.1) = D1.node n ∘ Prod.fst from rfl, ← List.map_map,
    List.map_fst_zip (le_of_eq hl)]

/-- The `engR n` nodes, `n > 0`, in `D1`'s template order. -/
def engRo (n : ℕ) : List Node :=
  [hN n 0] ++ (range n).map (bN n 0) ++ (range n).map (sN n 0) ++ [eN n 0, hN n 1] ++
    (range n).map (bN n 1) ++ (range n).map (sN n 1) ++ [eN n 1, zN n, xN n]

theorem nodesR_eq {n : ℕ} (hn : 0 < n) :
    ((D1.famExpand n).nodes.map (·.node (lab1 n))) = engRo n := by
  rw [nodes1_fam, List.map_map]
  have hc : ∀ e ∈ Z1 n, ((fun x : DNode => x.node (lab1 n)) ∘ fun e : DNode × ℕ × ℕ => D1.node n e.1) e =
      (⟨e.1.kind, (range e.1.outs.length).map (labY n e.2.1 e.2.2 true),
        (range e.1.ins.length).map (labY n e.2.1 e.2.2 false),
        FamDiagram.ixAt n D1.dim e.2.1 e.2.2 e.1.dim, FamDiagram.ixAt n D1.ph e.2.1 e.2.2 e.1.ph⟩ : Node) :=
    fun e he => node1_lab he
  rw [List.map_congr_left hc]
  have hn' : n ≠ 0 := by omega
  have hb : bN n = fun c t => ⟨.B, [bv n c t], [bw n c t, 2 * n + 2 * t + c], dI n t, 0⟩ := rfl
  have hs : sN n = fun c t => ⟨.S, [sv n c t, 2 * t + c], [sv n c (t + 1)], dI n t, 0⟩ := rfl
  simp only [engRo, hb, hs]
  simp [Z1, hN, eN, zN, xN, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, labY,
    FamDiagram.ixAt, D1, hn', dI, List.range_succ, Function.comp_def, List.lookup, rl]

theorem flatMap_pair_perm {β : Type} (l : List ℕ) (f g : ℕ → β) :
    (l.flatMap fun t => [f t, g t]).Perm (l.map f ++ l.map g) := by
  have e : (l.flatMap fun t => [f t, g t]) = l.flatMap fun t => [f t] ++ [g t] := rfl
  rw [e, ← FamFca.flatMap_single f l, ← FamFca.flatMap_single g l]
  exact (List.flatMap_append_perm _ _ _).symm

theorem nodesR {n : ℕ} (hn : 0 < n) :
    ((D1.famExpand n).nodes.map (·.node (lab1 n))).Perm (engR n).nodes := by
  rw [nodesR_eq hn]
  unfold engR engRo
  rcases Nat.lt_or_ge n 2 with h | h
  · obtain rfl : n = 1 := by omega
    simp
  · have h0 : n ≠ 0 := by omega
    have h1 : n ≠ 1 := by omega
    simp only [h0, h1, ite_false]
    rw [List.perm_iff_count]
    intro a
    have c1 := ((flatMap_pair_perm (range n) (bN n 0) (bN n 1)).count_eq a)
    have c2 := ((flatMap_pair_perm (range n) (sN n 0) (sN n 1)).count_eq a)
    simp only [List.count_append, List.count_cons, List.count_nil] at c1 c2 ⊢
    omega

/-! ## The wires -/

theorem mem_Z1_0 {n : ℕ} : (m0, (0, 0)) ∈ Z1 n := by simp [Z1]
theorem mem_Z1_3 {n : ℕ} : (m3, (3, 0)) ∈ Z1 n := by simp [Z1]
theorem mem_Z1_4 {n : ℕ} : (m4, (4, 0)) ∈ Z1 n := by simp [Z1]
theorem mem_Z1_7 {n : ℕ} : (m7, (7, 0)) ∈ Z1 n := by simp [Z1]
theorem mem_Z1_8 {n : ℕ} : (m8, (8, 0)) ∈ Z1 n := by simp [Z1]
theorem mem_Z1_9 {n : ℕ} : (m9, (9, 0)) ∈ Z1 n := by simp [Z1]

theorem mem_Z1_1 {n j : ℕ} (hj : j < n) : (({ m1 with id := 10 + 2 * j } : DNode), (1, j)) ∈ Z1 n :=
  List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
    (List.mem_append_left _ (List.mem_append_right _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩))))))

theorem mem_Z1_2 {n j : ℕ} (hj : j < n) :
    (({ m2 with id := 10 + 2 * n + 2 * j } : DNode), (2, j)) ∈ Z1 n :=
  List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
    (List.mem_append_right _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩)))))

theorem mem_Z1_5 {n j : ℕ} (hj : j < n) : (({ m5 with id := 11 + 2 * j } : DNode), (5, j)) ∈ Z1 n :=
  List.mem_append_left _ (List.mem_append_left _
    (List.mem_append_right _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩)))

theorem mem_Z1_6 {n j : ℕ} (hj : j < n) :
    (({ m6 with id := 11 + 2 * n + 2 * j } : DNode), (6, j)) ∈ Z1 n :=
  List.mem_append_left _ (List.mem_append_right _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩))

/-- Copy `j` of the binder of chain `c`. -/
def bid (c j : ℕ) : ℕ := 10 + c + 2 * j

/-- Copy `j` of the splitter of chain `c`. -/
def sid (n c j : ℕ) : ℕ := 10 + 2 * n + c + 2 * j

theorem lab1_b {n c j : ℕ} (hc : c < 2) (hj : j < n) (out : Bool) (k : ℕ) :
    lab1 n ⟨bid c j, out, k⟩ = labY n (1 + 4 * c) j out k := by
  interval_cases c
  · exact lab1_at (mem_Z1_1 hj) (by simp [bid]) out k
  · exact lab1_at (mem_Z1_5 hj) (by simp [bid]) out k

theorem lab1_s {n c j : ℕ} (hc : c < 2) (hj : j < n) (out : Bool) (k : ℕ) :
    lab1 n ⟨sid n c j, out, k⟩ = labY n (2 + 4 * c) j out k := by
  interval_cases c
  · exact lab1_at (mem_Z1_2 hj) (by simp [sid]) out k
  · exact lab1_at (mem_Z1_6 hj) (by simp [sid]; omega) out k

theorem lab1_f {n x : ℕ} (hx : x = 0 ∨ x = 3 ∨ x = 4 ∨ x = 7 ∨ x = 8 ∨ x = 9) (out : Bool) (k : ℕ) :
    lab1 n ⟨x, out, k⟩ = labY n x 0 out k := by
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl
  · exact lab1_at mem_Z1_0 rfl out k
  · exact lab1_at mem_Z1_3 rfl out k
  · exact lab1_at mem_Z1_4 rfl out k
  · exact lab1_at mem_Z1_7 rfl out k
  · exact lab1_at mem_Z1_8 rfl out k
  · exact lab1_at mem_Z1_9 rfl out k

theorem cp1_b {n c j : ℕ} (hc : c < 2) (hj : j < n) : D1.cp n (1 + 4 * c) j = bid c j := by
  interval_cases c
  · exact (cp1_eq (mem_Z1_1 hj)).trans (by simp [bid])
  · exact (cp1_eq (mem_Z1_5 hj)).trans (by simp [bid])

theorem cp1_s {n c j : ℕ} (hc : c < 2) (hj : j < n) : D1.cp n (2 + 4 * c) j = sid n c j := by
  interval_cases c
  · exact (cp1_eq (mem_Z1_2 hj)).trans (by simp [sid])
  · exact (cp1_eq (mem_Z1_6 hj)).trans (by simp [sid]; omega)

theorem at1 (n j : ℕ) (q : Port) : D1.at_ n j q = ⟨D1.cp n q.node j, q.out, q.pos⟩ := by
  simp [FamDiagram.at_, FamDiagram.cnt_plain n (show D1.pairs = [] from rfl), FamDiagram.posC_one]

theorem mv1 (n : ℕ) (q : Port) : D1.mv n q = q := FamDiagram.mv_plain n rfl q

/-- The binder chain `c`'s wires, `n > 0`. -/
def WB (n c : ℕ) : List (Port × Port) :=
  (⟨4 * c, true, 0⟩, ⟨bid c 0, false, 0⟩) ::
    (range (n - 1)).map (fun j => (⟨bid c j, true, 0⟩, ⟨bid c (j + 1), false, 0⟩)) ++
    [(⟨bid c (n - 1), true, 0⟩, ⟨8 + c, false, 0⟩)]

/-- The splitter chain `c`'s wires, `n > 0`. -/
def WS (n c : ℕ) : List (Port × Port) :=
  (⟨3 + 4 * c, false, 0⟩, ⟨sid n c 0, true, 0⟩) ::
    (range (n - 1)).map (fun j => (⟨sid n c j, false, 0⟩, ⟨sid n c (j + 1), true, 0⟩)) ++
    [(⟨sid n c (n - 1), false, 0⟩, ⟨8 + c, true, 0⟩)]

/-- The `z`–`x` wire. -/
def WZ : List (Port × Port) := [(⟨8, true, 1⟩, ⟨9, false, 1⟩)]

theorem wires1 {n : ℕ} (hn : 0 < n) :
    (D1.famExpand n).wires = WZ ++ (WB n 0 ++ (WB n 1 ++ (WS n 0 ++ WS n 1))) := by
  simp only [FamDiagram.famExpand, X1, show (E2n n).wires = WZ from rfl, WZ, List.map_cons, List.map_nil,
    mv1, show D1.pairs = [] from rfl, List.zipIdx_nil, List.flatMap_nil, List.append_nil]
  have hL : D1.links = [⟨⟨1, true, 0⟩, ⟨1, false, 0⟩, ⟨8, false, 0⟩, ⟨0, true, 0⟩⟩,
      ⟨⟨5, true, 0⟩, ⟨5, false, 0⟩, ⟨9, false, 0⟩, ⟨4, true, 0⟩⟩,
      ⟨⟨2, false, 0⟩, ⟨2, true, 0⟩, ⟨8, true, 0⟩, ⟨3, false, 0⟩⟩,
      ⟨⟨6, false, 0⟩, ⟨6, true, 0⟩, ⟨9, true, 0⟩, ⟨7, false, 0⟩⟩] := rfl
  simp only [hL, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  simp only [FamDiagram.linkWires, show n ≠ 0 by omega, ite_false, mv1, at1]
  have b0 : ∀ j < n, D1.cp n 1 j = bid 0 j := fun j hj => cp1_b (c := 0) (by omega) hj
  have b1 : ∀ j < n, D1.cp n 5 j = bid 1 j := fun j hj => cp1_b (c := 1) (by omega) hj
  have s0 : ∀ j < n, D1.cp n 2 j = sid n 0 j := fun j hj => cp1_s (c := 0) (by omega) hj
  have s1 : ∀ j < n, D1.cp n 6 j = sid n 1 j := fun j hj => cp1_s (c := 1) (by omega) hj
  simp only [WB, WS, List.cons_append, List.singleton_append]
  rw [b0 0 hn, b0 (n - 1) (by omega), b1 0 hn, b1 (n - 1) (by omega), s0 0 hn, s0 (n - 1) (by omega),
    s1 0 hn, s1 (n - 1) (by omega)]
  have e : ∀ (f : ℕ → ℕ) (g : ℕ → ℕ) (o o' : Bool), (∀ j < n, f j = g j) →
      (range (n - 1)).map (fun j => ((⟨f j, o, 0⟩ : Port), (⟨f (j + 1), o', 0⟩ : Port))) =
      (range (n - 1)).map (fun j => ((⟨g j, o, 0⟩ : Port), (⟨g (j + 1), o', 0⟩ : Port))) :=
    fun f g o o' h => List.map_congr_left fun j hj => by
      have := List.mem_range.1 hj; rw [h j (by omega), h (j + 1) (by omega)]
  rw [e _ _ _ _ b0, e _ _ _ _ b1, e _ _ _ _ s0, e _ _ _ _ s1]

/-! ## The wire labels -/

/-- The roles of the wire labels, `n > 0`. -/
def roles (n : ℕ) : List ℕ :=
  [6] ++ ([0] ++ (range (n - 1)).map (fun j => 9 + 2 * j) ++ [4]) ++
    ([2] ++ (range (n - 1)).map (fun j => 9 + 2 * j + 1) ++ [7]) ++
    ([1] ++ (range (n - 1)).map (fun j => 9 + 2 * (n - 1) + 2 * j) ++ [5]) ++
    ([3] ++ (range (n - 1)).map (fun j => 9 + 2 * (n - 1) + 2 * j + 1) ++ [8])

theorem wlabR {n : ℕ} (hn : 0 < n) :
    (D1.famExpand n).wires.map (fun w => lab1 n w.1) = (roles n).map (rl n) := by
  rw [wires1 hn]
  have hn' : n ≠ 0 := by omega
  simp only [WZ, WB, WS, roles, List.map_append, List.map_cons, List.map_nil, List.map_map,
    Function.comp_def, List.cons_append, List.nil_append, List.append_assoc]
  rw [lab1_f (x := 8) (by omega), lab1_f (x := 0) (by omega), lab1_f (x := 4) (by omega),
    lab1_f (x := 3) (by omega), lab1_f (x := 7) (by omega),
    lab1_b (c := 0) (by omega) (by omega), lab1_b (c := 1) (by omega) (by omega),
    lab1_s (c := 0) (by omega) (by omega), lab1_s (c := 1) (by omega) (by omega)]
  have hb : ∀ c < 2, ((range (n - 1)).map fun j => lab1 n ⟨bid c j, true, 0⟩) =
      (range (n - 1)).map fun j => rl n (9 + 2 * j + c) := fun c hc =>
    List.map_congr_left fun j hj => by
      have := List.mem_range.1 hj
      rw [lab1_b hc (by omega)]
      interval_cases c <;> simp [labY, bv, show j + 1 ≠ n by omega]
  have hs : ∀ c < 2, ((range (n - 1)).map fun j => lab1 n ⟨sid n c j, false, 0⟩) =
      (range (n - 1)).map fun j => rl n (9 + 2 * (n - 1) + 2 * j + c) := fun c hc =>
    List.map_congr_left fun j hj => by
      have := List.mem_range.1 hj
      rw [lab1_s hc (by omega)]
      interval_cases c <;> simp [labY, sv, show j + 1 ≠ n by omega]
  rw [hb 0 (by omega), hb 1 (by omega), hs 0 (by omega), hs 1 (by omega)]
  simp [labY, bv, sv, show n - 1 + 1 = n by omega, hn']

theorem roles_perm {n : ℕ} (hn : 0 < n) : (roles n).Perm (range (4 * n + 5)) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have hR : range (4 * (m + 1) + 5) = range 9 ++ (range (2 * m)).map (9 + ·) ++
      (range (2 * m)).map (9 + 2 * m + ·) := by
    rw [show 4 * (m + 1) + 5 = 9 + 2 * m + 2 * m by omega, List.range_add, List.range_add]
  simp only [roles, Nat.add_sub_cancel]
  have p1 := (flatMap_pair_perm (range m) (fun j => 9 + 2 * j) (fun j => 9 + 2 * j + 1))
  have p2 := (flatMap_pair_perm (range m) (fun j => 9 + 2 * m + 2 * j) (fun j => 9 + 2 * m + 2 * j + 1))
  rw [FamDraper.flat2_range] at p1 p2
  rw [hR, List.perm_iff_count]
  intro a
  have c1 := p1.count_eq a
  have c2 := p2.count_eq a
  simp only [List.count_append, List.count_cons, List.count_nil,
    show range 9 = [0, 1, 2, 3, 4, 5, 6, 7, 8] from rfl] at c1 c2 ⊢
  omega

theorem pofs_perm (n : ℕ) : ((range (4 * n + 5)).map (pofs n)).Perm (range (4 * n + 5)) := by
  have hnd : ((range (4 * n + 5)).map (pofs n)).Nodup :=
    List.nodup_range.map_on fun a ha b hb e => by
      rw [← rofs_pofs (List.mem_range.1 ha), ← rofs_pofs (List.mem_range.1 hb), e]
  refine (hnd.subperm fun x hx => ?_).perm_of_length_le (by simp)
  obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hx
  exact List.mem_range.2 (pofs_lt (List.mem_range.1 hk))

theorem intR {n : ℕ} (hn : 0 < n) :
    ((D1.famExpand n).wires.map fun w => lab1 n w.1).Perm (engR n).int := by
  rw [wlabR hn]
  have h := ((roles_perm hn).map (pofs n)).trans (pofs_perm n)
  rw [show (roles n).map (rl n) = ((roles n).map (pofs n)).map (rb n + ·) by
    simp [List.map_map, Function.comp_def, rl]]
  exact h.map _

theorem hwR {n : ℕ} (hn : 0 < n) : ∀ w ∈ (D1.famExpand n).wires, lab1 n w.1 = lab1 n w.2 := by
  have hn' : n ≠ 0 := by omega
  intro w hw
  rw [wires1 hn] at hw
  simp only [WZ, WB, WS, List.mem_append, List.mem_cons, List.mem_map, List.mem_range,
    List.not_mem_nil, or_false] at hw
  rcases hw with rfl | ((rfl | ⟨j, hj, rfl⟩) | rfl) | ((rfl | ⟨j, hj, rfl⟩) | rfl) |
    ((rfl | ⟨j, hj, rfl⟩) | rfl) | ((rfl | ⟨j, hj, rfl⟩) | rfl)
  · rw [lab1_f (by omega), lab1_f (by omega)]; rfl
  · rw [lab1_f (by omega), lab1_b (by omega) hn]; simp [labY, bw]
  · rw [lab1_b (by omega) (by omega), lab1_b (by omega) (by omega)]; simp [labY, bw]
  · rw [lab1_b (by omega) (by omega), lab1_f (by omega)]; simp [labY, bv, show n - 1 + 1 = n by omega]
  · rw [lab1_f (by omega), lab1_b (by omega) hn]; simp [labY, bw]
  · rw [lab1_b (by omega) (by omega), lab1_b (by omega) (by omega)]; simp [labY, bw]
  · rw [lab1_b (by omega) (by omega), lab1_f (by omega)]; simp [labY, bv, show n - 1 + 1 = n by omega]
  · rw [lab1_f (by omega), lab1_s (by omega) hn]; simp [labY, sv]
  · rw [lab1_s (by omega) (by omega), lab1_s (by omega) (by omega)]; simp [labY]
  · rw [lab1_s (by omega) (by omega), lab1_f (by omega)]; simp [labY, sv, show n - 1 + 1 = n by omega, hn']
  · rw [lab1_f (by omega), lab1_s (by omega) hn]; simp [labY, sv]
  · rw [lab1_s (by omega) (by omega), lab1_s (by omega) (by omega)]; simp [labY]
  · rw [lab1_s (by omega) (by omega), lab1_f (by omega)]; simp [labY, sv, show n - 1 + 1 = n by omega, hn']

/-! ## Every leg on the boundary or a wire -/

theorem bndR (n : ℕ) : (D1.famExpand n).bnd = BOn n ++ BIn n := by
  have hm : D1.mv n = id := funext (mv1 n)
  simp only [Diagram.bnd, FamDiagram.famExpand, X1, hm, List.map_id]
  rfl

theorem mem_Z1c {n : ℕ} {a : DNode × ℕ × ℕ} (h : a ∈ Z1 n) :
    a = (m0, (0, 0)) ∨ (∃ j < n, a = ({ m1 with id := 10 + 2 * j }, (1, j))) ∨
      (∃ j < n, a = ({ m2 with id := 10 + 2 * n + 2 * j }, (2, j))) ∨ a = (m3, (3, 0)) ∨
      a = (m4, (4, 0)) ∨ (∃ j < n, a = ({ m5 with id := 11 + 2 * j }, (5, j))) ∨
      (∃ j < n, a = ({ m6 with id := 11 + 2 * n + 2 * j }, (6, j))) ∨ a = (m7, (7, 0)) ∨
      a = (m8, (8, 0)) ∨ a = (m9, (9, 0)) := by
  simp only [Z1, List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.mem_map,
    List.mem_range] at h
  rcases h with ((((((rfl | ⟨j, hj, rfl⟩) | ⟨j, hj, rfl⟩) | rfl | rfl) | ⟨j, hj, rfl⟩) | ⟨j, hj, rfl⟩) |
    rfl | rfl | rfl)
  · exact Or.inl rfl
  · exact Or.inr (Or.inl ⟨j, hj, rfl⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨j, hj, rfl⟩))
  · exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨j, hj, rfl⟩)))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨j, hj, rfl⟩))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl))))))))

theorem okW {n : ℕ} (hn : 0 < n) {w : Port × Port}
    (hw : w ∈ WZ ∨ w ∈ WB n 0 ∨ w ∈ WB n 1 ∨ w ∈ WS n 0 ∨ w ∈ WS n 1) :
    (D1.famExpand n).okPort w.1 = true ∧ (D1.famExpand n).okPort w.2 = true := by
  have hm : w ∈ (D1.famExpand n).wires := by
    rw [wires1 hn]; simp only [List.mem_append]; tauto
  exact ⟨Diagram.okPort_of_mem _ _ (Or.inr ⟨w, hm, Or.inl rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨w, hm, Or.inr rfl⟩)⟩

theorem okBR {n : ℕ} {p : Port} (hp : p ∈ BOn n ++ BIn n) : (D1.famExpand n).okPort p = true :=
  Diagram.okPort_of_mem _ _ (Or.inl (by rw [bndR]; exact hp))

theorem wbh (n c : ℕ) : ((⟨4 * c, true, 0⟩ : Port), (⟨bid c 0, false, 0⟩ : Port)) ∈ WB n c := by simp [WB]
theorem wbm {n c j : ℕ} (hj : j + 1 < n) :
    ((⟨bid c j, true, 0⟩ : Port), (⟨bid c (j + 1), false, 0⟩ : Port)) ∈ WB n c :=
  List.mem_cons_of_mem _ (List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 (by omega), rfl⟩))
theorem wbl (n c : ℕ) : ((⟨bid c (n - 1), true, 0⟩ : Port), (⟨8 + c, false, 0⟩ : Port)) ∈ WB n c := by
  simp [WB]
theorem wsh (n c : ℕ) : ((⟨3 + 4 * c, false, 0⟩ : Port), (⟨sid n c 0, true, 0⟩ : Port)) ∈ WS n c := by
  simp [WS]
theorem wsm {n c j : ℕ} (hj : j + 1 < n) :
    ((⟨sid n c j, false, 0⟩ : Port), (⟨sid n c (j + 1), true, 0⟩ : Port)) ∈ WS n c :=
  List.mem_cons_of_mem _ (List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 (by omega), rfl⟩))
theorem wsl (n c : ℕ) : ((⟨sid n c (n - 1), false, 0⟩ : Port), (⟨8 + c, true, 0⟩ : Port)) ∈ WS n c := by
  simp [WS]

theorem bid_eq (j : ℕ) : 10 + 2 * j = bid 0 j ∧ 11 + 2 * j = bid 1 j := by
  unfold bid; constructor <;> omega
theorem sid_eq (n j : ℕ) : 10 + 2 * n + 2 * j = sid n 0 j ∧ 11 + 2 * n + 2 * j = sid n 1 j := by
  unfold sid; constructor <;> omega

/-- A binder copy's legs: the chain on `in 0` and `out 0`, the boundary on `in 1`. -/
theorem okBin {n c j : ℕ} (hn : 0 < n) (hc : c < 2) (hj : j < n) :
    (D1.famExpand n).okPort ⟨bid c j, true, 0⟩ = true ∧ (D1.famExpand n).okPort ⟨bid c j, false, 0⟩ = true := by
  have hw : ∀ w, w ∈ WB n c → w ∈ WZ ∨ w ∈ WB n 0 ∨ w ∈ WB n 1 ∨ w ∈ WS n 0 ∨ w ∈ WS n 1 := fun w h => by
    interval_cases c <;> tauto
  refine ⟨?_, ?_⟩
  · by_cases h : j + 1 < n
    · exact (okW hn (hw _ (wbm h))).1
    · obtain rfl : j = n - 1 := by omega
      exact (okW hn (hw _ (wbl n c))).1
  · rcases j with _ | i
    · exact (okW hn (hw _ (wbh n c))).2
    · exact (okW hn (hw _ (wbm (j := i) (by omega)))).2

theorem okSpl {n c j : ℕ} (hn : 0 < n) (hc : c < 2) (hj : j < n) :
    (D1.famExpand n).okPort ⟨sid n c j, true, 0⟩ = true ∧
      (D1.famExpand n).okPort ⟨sid n c j, false, 0⟩ = true := by
  have hw : ∀ w, w ∈ WS n c → w ∈ WZ ∨ w ∈ WB n 0 ∨ w ∈ WB n 1 ∨ w ∈ WS n 0 ∨ w ∈ WS n 1 := fun w h => by
    interval_cases c <;> tauto
  refine ⟨?_, ?_⟩
  · rcases j with _ | i
    · exact (okW hn (hw _ (wsh n c))).2
    · exact (okW hn (hw _ (wsm (j := i) (by omega)))).2
  · by_cases h : j + 1 < n
    · exact (okW hn (hw _ (wsm h))).1
    · obtain rfl : j = n - 1 := by omega
      exact (okW hn (hw _ (wsl n c))).1

theorem memBO {n j : ℕ} (hj : j < n) : (⟨10 + 2 * n + 2 * j, true, 1⟩ : Port) ∈ BOn n ++ BIn n ∧
    (⟨11 + 2 * n + 2 * j, true, 1⟩ : Port) ∈ BOn n ++ BIn n :=
  ⟨List.mem_append_left _ (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩),
    List.mem_append_left _ (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩)⟩

theorem memBI {n j : ℕ} (hj : j < n) : (⟨10 + 2 * j, false, 1⟩ : Port) ∈ BOn n ++ BIn n ∧
    (⟨11 + 2 * j, false, 1⟩ : Port) ∈ BOn n ++ BIn n :=
  ⟨List.mem_append_right _ (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩),
    List.mem_append_right _ (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩)⟩

theorem portsR {n : ℕ} (hn : 0 < n) : (D1.famExpand n).portsOK := by
  intro y hy
  rw [nodes1_fam] at hy
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hy
  simp only [FamDiagram.node, side1]
  have wz := okW hn (w := (⟨8, true, 1⟩, ⟨9, false, 1⟩)) (Or.inl (by simp [WZ]))
  rcases mem_Z1c he with rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, rfl⟩ | rfl | rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, rfl⟩ | rfl |
    rfl | rfl <;> refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;>
    simp [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9] at hk
  -- head a
  · subst hk; exact (okW hn (w := _) (Or.inr (Or.inl (wbh n 0)))).1
  -- binder a
  · subst hk; rw [(bid_eq j).1]; exact (okBin hn (by omega) hj).1
  · rcases k with _ | k
    · rw [(bid_eq j).1]; exact (okBin hn (by omega) hj).2
    · obtain rfl : k = 0 := by omega
      exact okBR (memBI hj).1
  -- splitter a
  · rcases k with _ | k
    · rw [(sid_eq n j).1]; exact (okSpl hn (by omega) hj).1
    · obtain rfl : k = 0 := by omega
      exact okBR (memBO hj).1
  · subst hk; rw [(sid_eq n j).1]; exact (okSpl hn (by omega) hj).2
  -- end a
  · subst hk; exact (okW hn (w := _) (Or.inr (Or.inr (Or.inr (Or.inl (wsh n 0)))))).1
  -- head b
  · subst hk; exact (okW hn (w := _) (Or.inr (Or.inr (Or.inl (wbh n 1))))).1
  -- binder b
  · subst hk; rw [(bid_eq j).2]; exact (okBin hn (by omega) hj).1
  · rcases k with _ | k
    · rw [(bid_eq j).2]; exact (okBin hn (by omega) hj).2
    · obtain rfl : k = 0 := by omega
      exact okBR (memBI hj).2
  -- splitter b
  · rcases k with _ | k
    · rw [(sid_eq n j).2]; exact (okSpl hn (by omega) hj).1
    · obtain rfl : k = 0 := by omega
      exact okBR (memBO hj).2
  · subst hk; rw [(sid_eq n j).2]; exact (okSpl hn (by omega) hj).2
  -- end b
  · subst hk; exact (okW hn (w := _) (Or.inr (Or.inr (Or.inr (Or.inr (wsh n 1)))))).1
  -- copy spider
  · rcases k with _ | k
    · exact (okW hn (w := _) (Or.inr (Or.inr (Or.inr (Or.inl (wsl n 0)))))).2
    · obtain rfl : k = 0 := by omega
      exact wz.1
  · subst hk; exact (okW hn (w := _) (Or.inr (Or.inl (wbl n 0)))).2
  -- sum spider
  · subst hk; exact (okW hn (w := _) (Or.inr (Or.inr (Or.inr (Or.inr (wsl n 1)))))).2
  · rcases k with _ | k
    · exact (okW hn (w := _) (Or.inr (Or.inr (Or.inl (wbl n 1))))).2
    · obtain rfl : k = 0 := by omega
      exact wz.2

/-! ## Leg dimensions -/

theorem find1 {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) :
    (D1.famExpand n).nodes.find? (fun y => y.id = e.1.id) = some (D1.node n e.1) := by
  rw [nodes1_fam]
  refine FamFca.find?_unique (List.mem_map.2 ⟨e, he, rfl⟩) (by simp [FamDiagram.node]) fun y hy hp => ?_
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hy
  simp only [FamDiagram.node] at hp
  rw [Z1_inj ha he (of_decide_eq_true hp)]

theorem portDim1 {n y : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (hy : e.1.id = y) (out : Bool) (k : ℕ)
    (hk : k < (if out then e.1.outs else e.1.ins).length) :
    (D1.famExpand n).portDim ⟨y, out, k⟩ = FamDiagram.ixAt n D1.legs ⟨e.2.1, out, k⟩ e.2.2 0 := by
  subst hy
  have h0 : ∀ v ∈ e.1.outs ++ e.1.ins, v = 0 := by
    have := (mem_Z1 he).2.2.2
    rw [this]
    have hx := (mem_Z1 he).1
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with h | h | h | h | h | h | h | h | h | h <;> simp [h, D1]
  unfold Diagram.portDim
  rw [find1 he]
  simp only [FamDiagram.node, orgOf1_eq he rfl, FamDiagram.side,
    FamDiagram.cnt_plain n (show D1.pairs = [] from rfl), FamDiagram.growC_one]
  cases out
  · simp only [Bool.false_eq_true, ite_false] at hk ⊢
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp only [List.getElem_map, List.getElem_range]
    rw [List.getD_eq_getElem _ _ hk, h0 _ (List.mem_append_right _ (List.getElem_mem hk))]
  · simp only [ite_true] at hk ⊢
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp only [List.getElem_map, List.getElem_range]
    rw [List.getD_eq_getElem _ _ hk, h0 _ (List.mem_append_left _ (List.getElem_mem hk))]

theorem lg0 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨0, true, 0⟩ j 0 = n := rfl
theorem lg4 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨4, true, 0⟩ j 0 = n := rfl
theorem lg3 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨3, false, 0⟩ j 0 = n := rfl
theorem lg7 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨7, false, 0⟩ j 0 = n := rfl
theorem lg1 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨1, true, 0⟩ j 0 = if j + 1 = 0 then n else j + 1 - 1 := rfl
theorem lg5 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨5, true, 0⟩ j 0 = if j + 1 = 0 then n else j + 1 - 1 := rfl
theorem lg2 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨2, false, 0⟩ j 0 = if j + 1 = 0 then n else j + 1 - 1 :=
  rfl
theorem lg6 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨6, false, 0⟩ j 0 = if j + 1 = 0 then n else j + 1 - 1 :=
  rfl
theorem lg8 (n j : ℕ) : FamDiagram.ixAt n D1.legs ⟨8, true, 1⟩ j 0 = if n = 0 then n else n - 1 := rfl

theorem dimW {n : ℕ} (hn : 0 < n) (d : ℕ) : ∀ w ∈ (D1.famExpand n).wires,
    labDim (engE d n) (lab1 n w.1) = (engE d n).dimv ((D1.famExpand n).portDim w.1) := by
  have hn' : n ≠ 0 := by omega
  have key : ∀ k, k < 4 * n + 5 → ∀ p : Port, lab1 n p = rl n k →
      (D1.famExpand n).portDim p = dI n (edim n k) →
      labDim (engE d n) (lab1 n p) = (engE d n).dimv ((D1.famExpand n).portDim p) := by
    intro k hk p hl hp
    rw [hl, hp, labDim_rl d hk]
    exact (FcaEng.engD_dI d (edim_le hk)).symm
  intro w hw
  rw [wires1 hn] at hw
  simp only [WZ, WB, WS, List.mem_append, List.mem_cons, List.mem_map, List.mem_range,
    List.not_mem_nil, or_false] at hw
  rcases hw with rfl | ((rfl | ⟨j, hj, rfl⟩) | rfl) | ((rfl | ⟨j, hj, rfl⟩) | rfl) |
    ((rfl | ⟨j, hj, rfl⟩) | rfl) | ((rfl | ⟨j, hj, rfl⟩) | rfl)
  · refine key 6 (by omega) _ (by rw [lab1_f (by omega)]; rfl) ?_
    dsimp only
    rw [portDim1 mem_Z1_8 (y := 8) rfl true 1 (by simp [m8])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 0 (by omega) _ (by rw [lab1_f (by omega)]; rfl) ?_
    dsimp only
    rw [portDim1 mem_Z1_0 (y := 4 * 0) rfl true 0 (by simp [m0])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key (9 + 2 * j) (by omega) _ (by
      rw [lab1_b (by omega) (by omega)]; simp [labY, bv, show j + 1 ≠ n by omega]) ?_
    dsimp only
    rw [portDim1 (mem_Z1_1 (j := j) (by omega)) (bid_eq j).1 true 0 (by simp [m1])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 4 (by omega) _ (by
      rw [lab1_b (by omega) (by omega)]; simp [labY, bv, show n - 1 + 1 = n by omega]) ?_
    dsimp only
    rw [portDim1 (mem_Z1_1 (j := n - 1) (by omega)) (bid_eq (n - 1)).1 true 0 (by simp [m1])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 2 (by omega) _ (by rw [lab1_f (by omega)]; rfl) ?_
    dsimp only
    rw [portDim1 mem_Z1_4 (y := 4 * 1) rfl true 0 (by simp [m4])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key (9 + 2 * j + 1) (by omega) _ (by
      rw [lab1_b (by omega) (by omega)]; simp [labY, bv, show j + 1 ≠ n by omega]) ?_
    dsimp only
    rw [portDim1 (mem_Z1_5 (j := j) (by omega)) (bid_eq j).2 true 0 (by simp [m5])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 7 (by omega) _ (by
      rw [lab1_b (by omega) (by omega)]; simp [labY, bv, show n - 1 + 1 = n by omega]) ?_
    dsimp only
    rw [portDim1 (mem_Z1_5 (j := n - 1) (by omega)) (bid_eq (n - 1)).2 true 0 (by simp [m5])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 1 (by omega) _ (by rw [lab1_f (by omega)]; rfl) ?_
    dsimp only
    rw [portDim1 mem_Z1_3 (y := 3 + 4 * 0) rfl false 0 (by simp [m3])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key (9 + 2 * (n - 1) + 2 * j) (by omega) _ (by
      rw [lab1_s (by omega) (by omega)]
      simp [labY, sv, show j + 1 ≠ n by omega]) ?_
    dsimp only
    rw [portDim1 (mem_Z1_2 (j := j) (by omega)) (sid_eq n j).1 false 0 (by simp [m2])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 5 (by omega) _ (by
      rw [lab1_s (by omega) (by omega)]; simp [labY, sv, show n - 1 + 1 = n by omega, hn']) ?_
    dsimp only
    rw [portDim1 (mem_Z1_2 (j := n - 1) (by omega)) (sid_eq n (n - 1)).1 false 0 (by simp [m2])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 3 (by omega) _ (by rw [lab1_f (by omega)]; rfl) ?_
    dsimp only
    rw [portDim1 mem_Z1_7 (y := 3 + 4 * 1) rfl false 0 (by simp [m7])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key (9 + 2 * (n - 1) + 2 * j + 1) (by omega) _ (by
      rw [lab1_s (by omega) (by omega)]
      simp [labY, sv, show j + 1 ≠ n by omega]) ?_
    dsimp only
    rw [portDim1 (mem_Z1_6 (j := j) (by omega)) (sid_eq n j).2 false 0 (by simp [m6])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all
  · refine key 8 (by omega) _ (by
      rw [lab1_s (by omega) (by omega)]; simp [labY, sv, show n - 1 + 1 = n by omega, hn']) ?_
    dsimp only
    rw [portDim1 (mem_Z1_6 (j := n - 1) (by omega)) (sid_eq n (n - 1)).2 false 0 (by simp [m6])]
    simp only [lg0, lg4, lg1, lg5, lg2, lg6, lg3, lg7, lg8, edim, dI]
    split_ifs <;> first | omega | (simp_all; omega) | simp_all

/-! ## The entry -/

theorem bnd_lab1 (n : ℕ) : (D1.famExpand n).bnd.map (lab1 n) = range (4 * n) := by
  rw [bndR, List.map_append, BOn, BIn, List.map_flatMap, List.map_flatMap]
  have h1 : ((range n).flatMap fun j =>
        [(⟨10 + 2 * n + 2 * j, true, 1⟩ : Port), ⟨11 + 2 * n + 2 * j, true, 1⟩].map (lab1 n)) =
      (range n).flatMap fun j => [0 + 2 * j, 0 + 2 * j + 1] := List.flatMap_congr fun j hj => by
    have hj := List.mem_range.1 hj
    simp only [List.map_cons, List.map_nil]
    rw [lab1_at (mem_Z1_2 hj) rfl, lab1_at (mem_Z1_6 hj) rfl]
    simp [labY]
  have h2 : ((range n).flatMap fun j =>
        [(⟨10 + 2 * j, false, 1⟩ : Port), ⟨11 + 2 * j, false, 1⟩].map (lab1 n)) =
      (range n).flatMap fun j => [2 * n + 2 * j, 2 * n + 2 * j + 1] := List.flatMap_congr fun j hj => by
    have hj := List.mem_range.1 hj
    simp only [List.map_cons, List.map_nil]
    rw [lab1_at (mem_Z1_1 hj) rfl, lab1_at (mem_Z1_5 hj) rfl]
    simp [labY]
  rw [h1, h2, flat2_range, flat2_range, show 4 * n = 2 * n + 2 * n by ring, List.range_add]
  simp

theorem bnd_lenR (n : ℕ) : (D1.famExpand n).bnd.length = 4 * n := by
  have := congrArg List.length (bnd_lab1 n); simpa using this

theorem bdims1 (n : ℕ) : (D1.famExpand n).bdims = List.replicate (4 * n) 0 := by
  refine List.eq_replicate_iff.2 ⟨by simp [Diagram.bdims, bnd_lenR], fun v hv => ?_⟩
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
  rw [bndR, List.mem_append] at hp
  rcases hp with hp | hp <;> obtain ⟨j, hj, hp⟩ := List.mem_flatMap.1 hp <;>
    have hj := List.mem_range.1 hj <;>
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp <;> rcases hp with rfl | rfl
  · exact (portDim1 (mem_Z1_2 hj) rfl true 1 (by simp [m2])).trans rfl
  · exact (portDim1 (mem_Z1_6 hj) rfl true 1 (by simp [m6])).trans rfl
  · exact (portDim1 (mem_Z1_1 hj) rfl false 1 (by simp [m1])).trans rfl
  · exact (portDim1 (mem_Z1_5 hj) rfl false 1 (by simp [m5])).trans rfl

/-- The expanded spec's entry is `engR n`'s, `n > 0`. -/
theorem evalR0 (d : ℕ) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ) (hσ : (D1.famExpand n).InRange (engE d n) σ) :
    (D1.famExpand n).eval (engE d n) σ = ArchytasZX.eval (engE d n) (engR n) σ := by
  have hbnd : ∀ i (hi : i < (D1.famExpand n).bnd.length), lab1 n (D1.famExpand n).bnd[i] = i := by
    intro i hi
    have := congrArg (·[i]?) (bnd_lab1 n)
    simp only [List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some] at this
    rw [List.getElem?_range (by rw [bnd_lenR] at hi; exact hi)] at this
    exact Option.some.inj this
  have hnd : ((D1.famExpand n).wires.map fun w => lab1 n w.1).Nodup :=
    (intR hn).nodup_iff.2 (engR_int_nodup n)
  have hmem : ∀ w ∈ (D1.famExpand n).wires, rb n ≤ lab1 n w.1 ∧ lab1 n w.1 < rb n + (4 * n + 5) :=
    fun w hw => mem_engR_int.1 ((intR hn).subset (List.mem_map_of_mem (f := fun w : Port × Port => lab1 n w.1) hw))
  have hdim : ∀ j (hj : j < (D1.famExpand n).wires.length),
      labDim (engE d n) (lab1 n (D1.famExpand n).wires[j].1) =
        (engE d n).dimv ((D1.famExpand n).wireDim j) := by
    intro j hj
    rw [dimW hn d _ (List.getElem_mem hj)]
    simp only [Diagram.wireDim, List.getElem?_eq_getElem hj, Option.map_some, Option.getD_some]
  have hbd : ∀ i < (D1.famExpand n).bnd.length,
      labDim (engE d n) i = (engE d n).dimv ((D1.famExpand n).bdims.getD i 0) := by
    intro i hi
    rw [bnd_lenR] at hi
    rw [bdims1, FamFca.getD_zero (fun v hv => List.eq_of_mem_replicate hv),
      labDim_low d hn (by unfold rb; omega)]
    exact (engD_zero d hn).symm
  rw [Diagram.eval_netBy _ _ (lab1 n) hbnd (hwR hn) (portsR hn) hnd
    (fun w hw => by rw [bnd_lenR]; have := hmem w hw; unfold rb at this; omega) hdim hbd σ hσ]
  exact eval_perm_net _ _ _ (nodesR hn) (intR hn) σ

end FamDraper

end ArchytasZX
