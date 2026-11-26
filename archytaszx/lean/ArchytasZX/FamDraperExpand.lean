/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaExpand

/-!
# The boxed Draper adder's expansion

`D0`: the engine's boxed `draper_adder` as a `FamDiagram`, one node box `b0` over the thirteen
template nodes, the pair boxes `(2.out 1, 0.in 1)`, `(8.out 1, 7.in 1)` and `(10.out 1, 14.in 1)`.

`expandK0`: its expansion in key order at `n` copies is `copyNodes b0 n`, the boundary copy by
copy (`copyNodes0_outs`, `copyNodes0_ins`). `cp0 j x = 15 + 13 j + rk x` is copy `j` of the
template node `x`. `sort_eq`: an insertion sort by keys read through `lexLe`.
-/

namespace ArchytasZX

namespace FamDraper

open List

/-- The engine's boxed `draper_adder`. -/
def D0 : FamDiagram :=
  { base :=
      { nodes := [⟨0, .Z, 0, 0, [0], [0, 0]⟩, ⟨1, .F, 0, 0, [0], [0]⟩, ⟨2, .Z, 0, 0, [0, 0], [0]⟩,
          ⟨4, .Z, 0, 0, [0, 0], [0]⟩, ⟨5, .Z, 0, 0, [0], [0, 0]⟩, ⟨6, .F, 0, 0, [0], [0]⟩,
          ⟨7, .Z, 0, 0, [0], [0, 0]⟩, ⟨8, .Z, 0, 0, [0, 0], [0]⟩, ⟨10, .Z, 0, 0, [0, 0], [0]⟩,
          ⟨11, .F, 0, 0, [0], [0]⟩, ⟨12, .F, 0, 0, [0], [0]⟩, ⟨13, .F, 0, 0, [0], [0]⟩,
          ⟨14, .Z, 0, 0, [0], [0, 0]⟩],
        wires := [(⟨0, true, 0⟩, ⟨1, false, 0⟩), (⟨1, true, 0⟩, ⟨2, false, 0⟩),
          (⟨2, true, 0⟩, ⟨5, false, 0⟩), (⟨4, true, 0⟩, ⟨7, false, 0⟩),
          (⟨4, true, 1⟩, ⟨6, false, 0⟩), (⟨6, true, 0⟩, ⟨5, false, 1⟩),
          (⟨5, true, 0⟩, ⟨8, false, 0⟩), (⟨8, true, 0⟩, ⟨10, false, 0⟩),
          (⟨10, true, 0⟩, ⟨11, false, 0⟩), (⟨11, true, 0⟩, ⟨12, false, 0⟩),
          (⟨12, true, 0⟩, ⟨13, false, 0⟩), (⟨13, true, 0⟩, ⟨14, false, 0⟩)],
        outs := [⟨7, true, 0⟩, ⟨14, true, 0⟩], ins := [⟨4, false, 0⟩, ⟨0, false, 0⟩],
        boxes := [⟨0, 0, none, [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14], []⟩] },
    links := [],
    pairs := [⟨1, 0, ⟨2, true, 1⟩, ⟨0, false, 1⟩, ⟨3, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩,
              ⟨2, 0, ⟨8, true, 1⟩, ⟨7, false, 1⟩, ⟨9, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩,
              ⟨3, 0, ⟨10, true, 1⟩, ⟨14, false, 1⟩, ⟨15, .ZP 0 0 (-1), 0, 0, [0], [0]⟩⟩],
    ph := [],
    dim := [(3, fun _ c => c), (9, fun _ c => c), (15, fun _ c => c)],
    legs := [] }

/-- The node box. -/
def b0 : DBox := ⟨0, 0, none, [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14], []⟩

/-- The template ids. -/
def ids0 : List ℕ := [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]

/-- The template node `x`'s rank in `b0`. -/
def rk (x : ℕ) : ℕ := ids0.idxOf x

/-- Copy `j` of the template node `x`. -/
def cp0 (j x : ℕ) : ℕ := 15 + 13 * j + rk x

theorem boxes0 : D0.base.boxes = [b0] := rfl

theorem expand0 (n : ℕ) : D0.base.expand (fun _ => n) = D0.base.copyNodes b0 n :=
  Diagram.expand_one _ _ b0 boxes0 rfl rfl

theorem fresh0 : D0.base.fresh = 15 := by decide

theorem sort0 : b0.nodes.mergeSort (· ≤ ·) = ids0 :=
  List.mergeSort_eq_self (r := (· ≤ ·)) (by decide)

theorem cpId0 (j x : ℕ) : D0.base.cpId b0 j x = cp0 j x := by
  simp only [Diagram.cpId, fresh0, sort0, cp0, rk, show b0.nodes.length = 13 from rfl]
  omega

theorem mem_b0 (x : DNode) (hx : x ∈ D0.base.nodes) : x.id ∈ b0.nodes := by
  simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    decide

theorem copyNodes0_nodes (n : ℕ) : (D0.base.copyNodes b0 n).nodes =
    D0.base.nodes.flatMap fun x => (range n).map fun j => { x with id := cp0 j x.id } := by
  simp only [Diagram.copyNodes]
  refine List.flatMap_congr fun x hx => ?_
  simp [mem_b0 x hx, cpId0]

theorem nimg0 (p : Port) (hp : p.node ∈ b0.nodes) (n : ℕ) :
    D0.base.nimg b0 n p = (range n).map fun j => ⟨cp0 j p.node, p.out, p.pos⟩ := by
  simp [Diagram.nimg, Diagram.inS, hp, cpId0]

theorem copyNodes0_wires (n : ℕ) : (D0.base.copyNodes b0 n).wires =
    D0.base.wires.flatMap fun w => (range n).map fun j =>
      ((⟨cp0 j w.1.node, w.1.out, w.1.pos⟩ : Port), (⟨cp0 j w.2.node, w.2.out, w.2.pos⟩ : Port)) := by
  simp only [Diagram.copyNodes]
  refine List.flatMap_congr fun w hw => ?_
  have h : w.1.node ∈ b0.nodes ∧ w.2.node ∈ b0.nodes := by
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hw
    rcases hw with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  rw [nimg0 _ h.1, nimg0 _ h.2, List.zip_map, FamFca.zip_range_self, List.map_map]
  rfl

theorem copyNodes0_outs (n : ℕ) : (D0.base.copyNodes b0 n).outs =
    (range n).flatMap fun j => [⟨cp0 j 7, true, 0⟩, ⟨cp0 j 14, true, 0⟩] := by
  have hO : D0.base.outs = [⟨7, true, 0⟩, ⟨14, true, 0⟩] := rfl
  have hi : Diagram.inS b0 ⟨7, true, 0⟩ = true := by decide
  have hi' : Diagram.inS b0 ⟨14, true, 0⟩ = true := by decide
  simp only [Diagram.copyNodes, Diagram.spliceN, hO, List.filter_cons, hi, hi', ite_true,
    List.filter_nil, List.findIdx_cons, List.take_zero, List.flatMap_nil, List.nil_append,
    List.drop_zero, Bool.not_true, Bool.false_eq_true, ite_false, cpId0, reduceCtorEq,
    List.map_cons, List.map_nil, List.append_nil]

theorem copyNodes0_ins (n : ℕ) : (D0.base.copyNodes b0 n).ins =
    (range n).flatMap fun j => [⟨cp0 j 4, false, 0⟩, ⟨cp0 j 0, false, 0⟩] := by
  have hO : D0.base.ins = [⟨4, false, 0⟩, ⟨0, false, 0⟩] := rfl
  have hi : Diagram.inS b0 ⟨4, false, 0⟩ = true := by decide
  have hi' : Diagram.inS b0 ⟨0, false, 0⟩ = true := by decide
  simp only [Diagram.copyNodes, Diagram.spliceN, hO, List.filter_cons, hi, hi', ite_true,
    List.filter_nil, List.findIdx_cons, List.take_zero, List.flatMap_nil, List.nil_append,
    List.drop_zero, Bool.not_true, Bool.false_eq_true, ite_false, cpId0, reduceCtorEq,
    List.map_cons, List.map_nil, List.append_nil]

/-! ## Sorting by keys -/

theorem lexLe_refl : ∀ a : List ℕ, Diagram.lexLe a a = true
  | [] => rfl
  | a :: as => by simp [Diagram.lexLe, lexLe_refl as]

theorem lexLe_total : ∀ a b : List ℕ, Diagram.lexLe a b = true ∨ Diagram.lexLe b a = true
  | [], _ => Or.inl rfl
  | _ :: _, [] => Or.inr rfl
  | a :: as, b :: bs => by
    simp only [Diagram.lexLe, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true]
    rcases lt_trichotomy a b with h | rfl | h
    · exact Or.inl (Or.inl h)
    · rcases lexLe_total as bs with h | h
      · exact Or.inl (Or.inr ⟨rfl, h⟩)
      · exact Or.inr (Or.inr ⟨rfl, h⟩)
    · exact Or.inr (Or.inl h)

theorem lexLe_trans : ∀ a b c : List ℕ, Diagram.lexLe a b = true → Diagram.lexLe b c = true →
    Diagram.lexLe a c = true
  | [], _, _, _, _ => rfl
  | _ :: _, [], _, h, _ => by simp [Diagram.lexLe] at h
  | _ :: _, _ :: _, [], _, h => by simp [Diagram.lexLe] at h
  | a :: as, b :: bs, c :: cs, h1, h2 => by
    simp only [Diagram.lexLe, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true] at h1 h2 ⊢
    rcases h1 with h1 | ⟨rfl, h1⟩ <;> rcases h2 with h2 | ⟨rfl, h2⟩
    · exact Or.inl (by omega)
    · exact Or.inl h1
    · exact Or.inl h2
    · exact Or.inr ⟨rfl, lexLe_trans as bs cs h1 h2⟩

theorem lexLe_antisymm : ∀ a b : List ℕ, Diagram.lexLe a b = true → Diagram.lexLe b a = true → a = b
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by simp [Diagram.lexLe] at h
  | _ :: _, [], h, _ => by simp [Diagram.lexLe] at h
  | a :: as, b :: bs, h1, h2 => by
    simp only [Diagram.lexLe, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true] at h1 h2
    rcases h1 with h1 | ⟨rfl, h1⟩
    · rcases h2 with h2 | ⟨h2, -⟩ <;> omega
    · rcases h2 with h2 | ⟨-, h2⟩
      · omega
      · rw [lexLe_antisymm as bs h1 h2]

/-- Insertion sort by `lexLe` of keys is the key-sorted permutation, keys distinct. -/
theorem sort_eq {α : Type} (k : α → List ℕ) (l l' : List α) (hp : l.Perm l')
    (hs : l'.Pairwise fun a b => Diagram.lexLe (k a) (k b) = true)
    (hk : ∀ a ∈ l, ∀ b ∈ l, k a = k b → a = b) :
    l.insertionSort (fun a b => Diagram.lexLe (k a) (k b) = true) = l' := by
  have : Std.Total (fun a b : α => Diagram.lexLe (k a) (k b) = true) :=
    ⟨fun a b => lexLe_total (k a) (k b)⟩
  have : IsTrans α (fun a b : α => Diagram.lexLe (k a) (k b) = true) :=
    ⟨fun a b c => lexLe_trans (k a) (k b) (k c)⟩
  have hl := List.perm_insertionSort (fun a b : α => Diagram.lexLe (k a) (k b) = true) l
  refine (hl.trans hp).eq_of_pairwise (fun a b ha hb h1 h2 => ?_)
    (List.pairwise_insertionSort _ l) hs
  exact hk a (hl.mem_iff.1 ha) b (hp.mem_iff.2 hb) (lexLe_antisymm _ _ h1 h2)

/-! ## The boundary in key order -/

theorem keySig0 : D0.base.keySig = [([], [0]), ([0], []), ([], [0]), ([2], [])] := by decide

theorem bnd0 : D0.base.bnd = [⟨7, true, 0⟩, ⟨14, true, 0⟩, ⟨4, false, 0⟩, ⟨0, false, 0⟩] := rfl

theorem SDom0 (n : ℕ) : D0.base.SDom (fun _ => n) =
    (range 4).flatMap fun i => (range n).map fun t => (i, [t]) := by
  unfold Diagram.SDom
  have hl : D0.base.bnd.length = 4 := rfl
  rw [hl]
  refine List.flatMap_congr fun i hi => ?_
  have hc : D0.base.legChain (D0.base.bnd.getD i Port.dflt) = [0] := by
    simp only [List.mem_range] at hi
    interval_cases i <;> decide
  rw [hc, FamFca.cidx0, List.map_map]
  rfl

/-- The slot copy at place `s` in key order. -/
def sl (n s : ℕ) : ℕ × List ℕ :=
  if s < 2 * n then (s % 2, [s / 2]) else (2 + s % 2, [(s - 2 * n) / 2])

/-- The key of a slot copy. -/
def key (x : ℕ × List ℕ) : List ℕ :=
  Diagram.keyOf (D0.base.keySig.getD x.1 ([], [])) x.1 x.2

theorem key_sl (n s : ℕ) : key (sl n s) =
    if s < 2 * n then (if s % 2 = 0 then [0, s / 2] else [0, s / 2, 1])
    else (if s % 2 = 0 then [2, (s - 2 * n) / 2] else [2, (s - 2 * n) / 2, 3]) := by
  unfold key sl
  rw [keySig0]
  have h := Nat.mod_two_eq_zero_or_one s
  split_ifs <;> rcases h with h | h <;> simp_all [Diagram.keyOf]

theorem sl_sorted (n : ℕ) : ((range (4 * n)).map (sl n)).Pairwise fun a b =>
    Diagram.lexLe (key a) (key b) = true := by
  rw [List.pairwise_map]
  refine (List.pairwise_lt_range (n := 4 * n)).imp_of_mem fun {a b} _ _ hab => ?_
  rw [key_sl, key_sl]
  have ha := Nat.mod_two_eq_zero_or_one a
  have hb := Nat.mod_two_eq_zero_or_one b
  split_ifs <;> simp only [Diagram.lexLe, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true,
    true_and, and_true, Bool.false_eq_true, and_false, or_false, lt_self_iff_false, false_or] <;>
    omega

theorem mem_SDom0 {n : ℕ} {x : ℕ × List ℕ} : x ∈ D0.base.SDom (fun _ => n) ↔
    ∃ i < 4, ∃ t < n, x = (i, [t]) := by
  rw [SDom0]
  simp only [List.mem_flatMap, List.mem_range, List.mem_map]
  constructor
  · rintro ⟨i, hi, t, ht, rfl⟩; exact ⟨i, hi, t, ht, rfl⟩
  · rintro ⟨i, hi, t, ht, rfl⟩; exact ⟨i, hi, t, ht, rfl⟩

theorem sl_perm (n : ℕ) : (D0.base.SDom (fun _ => n)).Perm ((range (4 * n)).map (sl n)) := by
  have hnd : ((range (4 * n)).map (sl n)).Nodup := by
    refine List.nodup_range.map_on fun a ha b hb e => ?_
    simp only [List.mem_range] at ha hb
    unfold sl at e
    have := Nat.mod_two_eq_zero_or_one a
    have := Nat.mod_two_eq_zero_or_one b
    split_ifs at e <;> simp only [Prod.mk.injEq, List.cons.injEq, and_true] at e <;> omega
  refine (List.perm_ext_iff_of_nodup (Diagram.SDom_nodup _) hnd).2 fun x => ?_
  rw [mem_SDom0]
  simp only [List.mem_map, List.mem_range]
  constructor
  · rintro ⟨i, hi, t, ht, rfl⟩
    refine ⟨if i < 2 then 2 * t + i else 2 * n + 2 * t + (i - 2), by split_ifs <;> omega, ?_⟩
    unfold sl
    interval_cases i <;> split_ifs <;> simp only [Prod.mk.injEq, List.cons.injEq, and_true] <;>
      omega
  · rintro ⟨s, hs, rfl⟩
    unfold sl
    split_ifs
    · exact ⟨s % 2, by omega, s / 2, by omega, rfl⟩
    · exact ⟨2 + s % 2, by omega, (s - 2 * n) / 2, by omega, rfl⟩

theorem key_inj {n : ℕ} : ∀ a ∈ D0.base.SDom (fun _ => n), ∀ b ∈ D0.base.SDom (fun _ => n),
    key a = key b → a = b := by
  intro a ha b hb e
  obtain ⟨i, hi, t, -, rfl⟩ := mem_SDom0.1 ha
  obtain ⟨i', hi', t', -, rfl⟩ := mem_SDom0.1 hb
  unfold key at e
  rw [keySig0] at e
  interval_cases i <;> interval_cases i' <;> simp_all [Diagram.keyOf]

theorem SL0 (n : ℕ) : D0.base.SL (fun _ => n) = (range (4 * n)).map (sl n) := by
  unfold Diagram.SL
  exact sort_eq key _ _ (sl_perm n) (sl_sorted n) key_inj

theorem pick0 : D0.base.pick = some b0 := by decide

/-- The boundary of the copies, copy by copy. -/
def BO (n : ℕ) : List Port := (range n).flatMap fun j => [⟨cp0 j 7, true, 0⟩, ⟨cp0 j 14, true, 0⟩]

/-- The inputs of the copies, copy by copy. -/
def BI (n : ℕ) : List Port := (range n).flatMap fun j => [⟨cp0 j 4, false, 0⟩, ⟨cp0 j 0, false, 0⟩]

theorem bnd_copyNodes0 (n : ℕ) : (D0.base.copyNodes b0 n).bnd = BO n ++ BI n := by
  rw [Diagram.bnd, copyNodes0_outs, copyNodes0_ins]; rfl

theorem len_flat2 {β : Type} (n : ℕ) (f g : ℕ → β) :
    ((range n).flatMap fun j => [f j, g j]).length = 2 * n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [List.range_succ, List.flatMap_append, List.length_append, ih]; simp; ring

theorem idx_flat2 (f g : ℕ → Port) (hf : ∀ a b, f a = f b → a = b) (hg : ∀ a b, g a = g b → a = b)
    (hfg : ∀ a b, f a ≠ g b) : ∀ {n t : ℕ}, t < n →
      ((range n).flatMap fun j => [f j, g j]).idxOf (f t) = 2 * t ∧
      ((range n).flatMap fun j => [f j, g j]).idxOf (g t) = 2 * t + 1
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, t, h => by
    rw [List.range_succ, List.flatMap_append]
    have hnf : ∀ x, x ∈ ((range n).flatMap fun j => [f j, g j]) → x = f t ∨ x = g t → t < n := by
      intro x hx e
      obtain ⟨a, ha, hx⟩ := List.mem_flatMap.1 hx
      have ha := List.mem_range.1 ha
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases e with rfl | rfl <;> rcases hx with hx | hx
      · rw [hf _ _ hx]; exact ha
      · exact absurd hx (hfg _ _)
      · exact absurd hx.symm (hfg _ _)
      · rw [hg _ _ hx]; exact ha
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · have ih := idx_flat2 f g hf hg hfg h
      refine ⟨?_, ?_⟩
      · rw [List.idxOf_append_of_mem (List.mem_flatMap.2 ⟨t, List.mem_range.2 h, by simp⟩), ih.1]
      · rw [List.idxOf_append_of_mem (List.mem_flatMap.2 ⟨t, List.mem_range.2 h, by simp⟩), ih.2]
    · have h1 : f t ∉ ((range t).flatMap fun j => [f j, g j]) := fun hm =>
        absurd (hnf _ hm (Or.inl rfl)) (lt_irrefl _)
      have h2 : g t ∉ ((range t).flatMap fun j => [f j, g j]) := fun hm =>
        absurd (hnf _ hm (Or.inr rfl)) (lt_irrefl _)
      rw [List.idxOf_append_of_notMem h1, List.idxOf_append_of_notMem h2, len_flat2]
      refine ⟨by simp, ?_⟩
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      rw [List.idxOf_cons_ne _ (hfg t t)]
      simp

theorem cp0_inj {j j' x : ℕ} (h : cp0 j x = cp0 j' x) : j = j' := by
  simp only [cp0] at h; omega

theorem rk_lt {x : ℕ} (hx : x ∈ ids0) : rk x < 13 := by
  simp only [ids0, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem rk_inj {x y : ℕ} (hx : x ∈ ids0) (hy : y ∈ ids0) (h : rk x = rk y) : x = y := by
  have := hy
  exact (List.idxOf_inj hx).1 h

theorem cp0_inj2 {j j' x x' : ℕ} (hx : x ∈ ids0) (hx' : x' ∈ ids0) (h : cp0 j x = cp0 j' x') :
    j = j' ∧ x = x' := by
  have := rk_lt hx; have := rk_lt hx'
  simp only [cp0] at h
  have hj : j = j' := by omega
  exact ⟨hj, rk_inj hx hx' (by subst hj; omega)⟩

theorem idxBO {n t : ℕ} (ht : t < n) : (BO n).idxOf ⟨cp0 t 7, true, 0⟩ = 2 * t ∧
    (BO n).idxOf ⟨cp0 t 14, true, 0⟩ = 2 * t + 1 :=
  idx_flat2 (fun j => ⟨cp0 j 7, true, 0⟩) (fun j => ⟨cp0 j 14, true, 0⟩)
    (fun a b e => cp0_inj (Port.mk.inj e).1) (fun a b e => cp0_inj (Port.mk.inj e).1)
    (fun a b e => absurd (cp0_inj2 (by decide) (by decide) (Port.mk.inj e).1).2 (by decide)) ht

theorem idxBI {n t : ℕ} (ht : t < n) : (BI n).idxOf ⟨cp0 t 4, false, 0⟩ = 2 * t ∧
    (BI n).idxOf ⟨cp0 t 0, false, 0⟩ = 2 * t + 1 :=
  idx_flat2 (fun j => ⟨cp0 j 4, false, 0⟩) (fun j => ⟨cp0 j 0, false, 0⟩)
    (fun a b e => cp0_inj (Port.mk.inj e).1) (fun a b e => cp0_inj (Port.mk.inj e).1)
    (fun a b e => absurd (cp0_inj2 (by decide) (by decide) (Port.mk.inj e).1).2 (by decide)) ht

theorem notin_BO {n : ℕ} (p : Port) (hp : p.out = false) : p ∉ BO n := fun hm => by
  obtain ⟨a, -, hm⟩ := List.mem_flatMap.1 hm
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl <;> simp at hp

theorem phi0 (n t : ℕ) (ht : t < n) (p : Port) (hp : p.node ∈ b0.nodes) :
    φS (D0.base.nimg b0 n) (D0.base.gN b0) p [t] = (⟨cp0 t p.node, p.out, p.pos⟩, []) := by
  have hin : Diagram.inS b0 p = true := by simpa [Diagram.inS] using hp
  simp only [φS, Diagram.gN, hin, Bool.true_or, ite_true, nimg0 p hp]
  rw [List.getD_eq_getElem _ _ (by simpa using ht)]
  simp

/-- The place of the slot copy `(i, [t])` in the boundary of the copies. -/
def bpos (n i t : ℕ) : ℕ := if i < 2 then 2 * t + i else 2 * n + 2 * t + (i - 2)

theorem bidx0 (n i t : ℕ) (hi : i < 4) (ht : t < n) :
    D0.base.bidx (fun _ => n) (i, [t]) = (bpos n i t, []) := by
  unfold Diagram.bidx
  have hB := Diagram.copyNodes_boxes_one D0.base b0 boxes0 n
  simp only [boxes0, List.length_singleton, Diagram.bidxFuel, pick0, Diagram.step,
    show b0.ports = [] from rfl, ite_true, Diagram.expandNode_none _ b0 rfl]
  rw [FamFca.pick_nil _ hB]
  simp only [Diagram.stepIx, Diagram.gather_none _ b0 rfl, ite_self, show b0.ports = [] from rfl,
    ite_true, Diagram.ixMap, bnd0]
  rw [bnd_copyNodes0]
  have hl := len_flat2 n (fun j => (⟨cp0 j 7, true, 0⟩ : Port)) (fun j => ⟨cp0 j 14, true, 0⟩)
  have hg : ∀ i < 4, ([⟨7, true, 0⟩, ⟨14, true, 0⟩, ⟨4, false, 0⟩, ⟨0, false, 0⟩] : List Port).getD
      (([⟨7, true, 0⟩, ⟨14, true, 0⟩, ⟨4, false, 0⟩, ⟨0, false, 0⟩] : List Port).idxOf
        (([⟨7, true, 0⟩, ⟨14, true, 0⟩, ⟨4, false, 0⟩, ⟨0, false, 0⟩] : List Port).getD i Port.dflt))
      Port.dflt = ([⟨7, true, 0⟩, ⟨14, true, 0⟩, ⟨4, false, 0⟩, ⟨0, false, 0⟩] : List Port).getD i
        Port.dflt := by decide
  rw [hg i hi]
  have hBO : (BO n).length = 2 * n := len_flat2 _ _ _
  have mO : ∀ x, (x = 7 ∨ x = 14) → (⟨cp0 t x, true, 0⟩ : Port) ∈ BO n := fun x hx =>
    List.mem_flatMap.2 ⟨t, List.mem_range.2 ht, by rcases hx with rfl | rfl <;> simp⟩
  interval_cases i <;> simp only [List.getD_cons_succ, List.getD_cons_zero] <;>
    rw [phi0 n t ht _ (by decide)] <;> dsimp only
  · rw [List.idxOf_append_of_mem (mO 7 (Or.inl rfl)), (idxBO ht).1]; simp [bpos]
  · rw [List.idxOf_append_of_mem (mO 14 (Or.inr rfl)), (idxBO ht).2]; simp [bpos]
  · rw [List.idxOf_append_of_notMem (notin_BO _ rfl), (idxBI ht).1, hBO]; simp [bpos]
  · rw [List.idxOf_append_of_notMem (notin_BO _ rfl), (idxBI ht).2, hBO]; simp [bpos]
    omega

theorem bidx_sl {n s : ℕ} (hs : s < 4 * n) : (D0.base.bidx (fun _ => n) (sl n s)).1 = s := by
  unfold sl
  split_ifs with h
  · rw [bidx0 n _ _ (by omega) (by omega)]; simp only [bpos]; split_ifs <;> omega
  · rw [bidx0 n _ _ (by omega) (by omega)]; simp only [bpos]; split_ifs <;> omega

/-- The expansion in key order is `copyNodes b0 n`. -/
theorem expandK0 (n : ℕ) : D0.base.expandK (fun _ => n) = D0.base.copyNodes b0 n := by
  unfold Diagram.expandK
  rw [expand0, SL0]
  set E := D0.base.copyNodes b0 n
  have hlen : E.bnd.length = 4 * n := by
    rw [bnd_copyNodes0, List.length_append, BO, BI, len_flat2, len_flat2]; ring
  have hL : ((range (4 * n)).map (sl n)).map
      (fun x => E.bnd.getD (D0.base.bidx (fun _ => n) x).1 Port.dflt) = E.outs ++ E.ins := by
    rw [List.map_map]
    have : ((range (4 * n)).map fun s => E.bnd.getD s Port.dflt) = E.bnd := by
      rw [← hlen]; exact Diagram.range_getD _
    rw [← Diagram.bnd]
    refine (List.map_congr_left fun s hs => ?_).trans this
    simp only [Function.comp, bidx_sl (List.mem_range.1 hs)]
  dsimp only
  rw [hL]
  simp

/-! ## Template node and copy -/

theorem X0 (n : ℕ) : D0.X n = D0.base.copyNodes b0 n := expandK0 n

theorem ids0_eq : D0.base.nodes.map (·.id) = ids0 := rfl

theorem mem_ids0 {x : DNode} (hx : x ∈ D0.base.nodes) : x.id ∈ ids0 :=
  ids0_eq ▸ List.mem_map_of_mem hx

theorem node_of_id {x x' : DNode} (hx : x ∈ D0.base.nodes) (hx' : x' ∈ D0.base.nodes)
    (h : x.id = x'.id) : x = x' := by
  have hnd : (D0.base.nodes.map (·.id)).Nodup := by decide
  exact List.inj_on_of_nodup_map hnd hx hx' h

theorem zip0 (n : ℕ) : (D0.X n).nodes.zip (D0.org n) =
    D0.base.nodes.flatMap fun x => (range n).map fun j => ({ x with id := cp0 j x.id }, (x.id, j)) := by
  rw [X0, copyNodes0_nodes]
  have ho : D0.org n = D0.base.nodes.flatMap fun x => (range n).map (x.id, ·) := by
    unfold FamDiagram.org
    refine List.flatMap_congr fun x hx => ?_
    have : D0.base.boxes.any (x.id ∈ ·.nodes) = true := by
      simp [boxes0, mem_b0 x hx]
    simp [this]
  rw [ho, FamFca.zip_flatMap _ _ _ (fun _ _ => by simp)]
  refine List.flatMap_congr fun x _ => ?_
  rw [List.zip_map, FamFca.zip_range_self, List.map_map]
  rfl

theorem mem_zip0 {n : ℕ} {a : DNode × ℕ × ℕ} (h : a ∈ (D0.X n).nodes.zip (D0.org n)) :
    ∃ x ∈ D0.base.nodes, ∃ j < n, a = ({ x with id := cp0 j x.id }, (x.id, j)) := by
  rw [zip0] at h
  obtain ⟨x, hx, h⟩ := List.mem_flatMap.1 h
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 h
  exact ⟨x, hx, j, List.mem_range.1 hj, rfl⟩

theorem orgOf0 {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) :
    D0.orgOf n (cp0 j x.id) = (x.id, j) := by
  unfold FamDiagram.orgOf
  rw [FamFca.find?_unique (e := ({ x with id := cp0 j x.id }, (x.id, j)))]
  · rfl
  · rw [zip0]; exact List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩⟩
  · simp
  · intro a ha hp
    obtain ⟨x', hx', j', -, rfl⟩ := mem_zip0 ha
    simp only [decide_eq_true_eq] at hp
    obtain ⟨rfl, h⟩ := cp0_inj2 (mem_ids0 hx') (mem_ids0 hx) hp
    rw [node_of_id hx' hx h]

theorem cp_eq0 {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) :
    D0.cp n x.id j = cp0 j x.id := by
  unfold FamDiagram.cp
  rw [FamFca.find?_unique (e := ({ x with id := cp0 j x.id }, (x.id, j)))]
  · rfl
  · rw [zip0]; exact List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩⟩
  · simp
  · intro a ha hp
    obtain ⟨x', hx', j', -, rfl⟩ := mem_zip0 ha
    simp only [decide_eq_true_eq, Prod.mk.injEq] at hp
    obtain ⟨h, rfl⟩ := hp
    rw [node_of_id hx' hx h]

end FamDraper

end ArchytasZX
