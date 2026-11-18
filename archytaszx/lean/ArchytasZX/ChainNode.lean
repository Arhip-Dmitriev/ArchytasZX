/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainLink

/-!
# A chain box against a node box: the carried wire

`DL` is a chain box whose copies hold a phaseless Z(1 -> 1) on a wire of their own (spider `0`)
and one on a wire carried from copy to copy (spider `1`, link `1.out0 -> 1.in0`), the carried
wire's ends on the boundary behind the identities `2` and `3`. `DR` is a node box over the same
own-wire spider beside one phaseless Z(1 -> 1) head on the carried wire. `cyclic_every`: for
every `k`, at any wire offset `o ≥ 2 k + 2`, the network of `DL.expandChain cL k` equals that of
`DR.expand (ksOf [k])`, for every environment with every label of one dimension and phase row `0`
zero (`hD`, `hφ`).

The expansions are written in closed form (`expandL`, `expandR`) and labelled (`netL_closed`,
`netR_closed`): copy `j`'s own spider from slot `k + 2 + j` to slot `j`, the carried wire from
slot `k + 1` to slot `k`. The carried wire's spiders are identities (`val_z11`, `chain_z_i`);
the tail identity is dropped once per step (`Equal.dropIdentity`, `chain_drop`, `chain_collapse`)
and the head meets the goal's head (`chain_netR`).
-/

namespace ArchytasZX

namespace ChainNode

open Finset Diagram

open Finset

/-- A phaseless Z spider with one output and one input is the identity. -/
theorem val_z11 (E : Env) (σ : ℕ → ℕ) (o i d : ℕ) (hφ : ∀ j, E.phase 0 j = 0) :
    val E σ ⟨.Z, [o], [i], d, 0⟩ = val E σ ⟨.I, [o], [i], d, 0⟩ := by
  simp only [val, List.map_cons, List.map_nil, List.cons_append, List.nil_append, tZ, tI, hφ,
    ex_zero]
  by_cases h : σ o = σ i ∧ σ o < E.dimv d
  · have h2 : σ i < E.dimv d := h.1 ▸ h.2
    rw [Finset.sum_eq_single (σ o)]
    · simp [h.1, h2]
    · intro j _ hj; simp [Ne.symm hj]
    · intro hn; exact absurd (Finset.mem_range.2 h.2) hn
  · rw [if_neg h]
    refine Finset.sum_eq_zero fun j hj => ?_
    rw [if_neg]
    rintro hv
    have h1 := hv (σ o) (by simp)
    have h2 := hv (σ i) (by simp)
    exact h ⟨h1.trans h2.symm, h1 ▸ Finset.mem_range.1 hj⟩

/-- Two networks with the same summed and boundary labels whose node products agree. -/
theorem Equal.of_prod (E : Env) (N M : Net) (hi : N.int = M.int) (hb : N.bnd = M.bnd)
    (h : ∀ σ, prodVal E N.nodes σ = prodVal E M.nodes σ) : Equal E 1 N 1 M := by
  refine ⟨by rw [hb], fun σ _ => ?_⟩
  simp only [one_mul, eval, hi]
  exact sumOver_congr _ _ _ _ _ fun τ _ _ => h τ

variable (o k : ℕ)

/-- The copies' own spiders: copy `j` from slot `k + 2 + j` to slot `j`. -/
def ws : List Node := (List.range k).map fun j => ⟨.Z, [j], [k + 2 + j], 0, 0⟩

/-- Step `j` of the carried wire, of kind `c`, from `o + j` to `o + j + 1`. -/
def stp (c : Kind) (j : ℕ) : Node := ⟨c, [o + j + 1], [o + j], 0, 0⟩

/-- The head of the carried wire, from slot `k + 1` to `o`. -/
def hd : Node := ⟨.I, [o], [k + 1], 0, 0⟩

/-- The carried wire with `m` steps of kind `c` between the head and the tail, which goes from
`o + m` to slot `k`; the copies' own spiders beside it. -/
def chainN (c : Kind) (m : ℕ) : Net :=
  ⟨ws k ++ [hd o k] ++ (List.range m).map (stp o c) ++ [⟨.I, [k], [o + m], 0, 0⟩],
    (List.range (m + 1)).map (o + ·), List.range (2 * k + 2)⟩

/-- The goal's network: the head from slot `k + 1` to slot `k`, then the copies. -/
def netR : Net := ⟨⟨.Z, [k], [k + 1], 0, 0⟩ :: ws k, [], List.range (2 * k + 2)⟩

theorem chain_z_i (E : Env) (hφ : ∀ j, E.phase 0 j = 0) (m : ℕ) :
    Equal E 1 (chainN o k .Z m) 1 (chainN o k .I m) := by
  have hz := fun σ o i d => val_z11 E σ o i d hφ
  refine Equal.of_prod E _ _ rfl rfl fun σ => ?_
  simp only [chainN, prodVal, List.map_append, List.map_map, Function.comp_def, stp, hz]

theorem chain_drop (E : Env) (hD : ∀ x, labDim E x = E.dimv 0) (ho : 2 * k + 2 ≤ o) (m : ℕ) :
    Equal E 1 (chainN o k .I (m + 1)) 1 (chainN o k .I m) := by
  have h := Equal.dropIdentity E (ws k ++ [hd o k] ++ (List.range (m + 1)).map (stp o .I))
    ((List.range (m + 1)).map (o + ·)) (List.range (2 * k + 2)) (o + (m + 1)) k 0
    (by rw [hD]) (hD _) (by omega) (by simp; omega) (by simp; omega)
  have hw : (ws k).map (Node.rename (swapTo (o + (m + 1)) k)) = ws k := by
    refine (List.map_congr_left fun n hn => ?_).trans (List.map_id _)
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hn
    have := List.mem_range.1 hj
    simp only [Node.rename, swapTo, List.map_cons, List.map_nil, id]
    congr 2 <;> split_ifs <;> omega
  have hc : ((List.range m).map (stp o .I)).map (Node.rename (swapTo (o + (m + 1)) k)) =
      (List.range m).map (stp o .I) := by
    refine (List.map_congr_left fun n hn => ?_).trans (List.map_id _)
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hn
    have := List.mem_range.1 hj
    simp only [stp, Node.rename, swapTo, List.map_cons, List.map_nil, id]
    congr 2 <;> split_ifs <;> omega
  have hh : Node.rename (swapTo (o + (m + 1)) k) (hd o k) = hd o k := by
    simp only [hd, Node.rename, swapTo, List.map_cons, List.map_nil]
    congr 2 <;> split_ifs <;> omega
  have hs : Node.rename (swapTo (o + (m + 1)) k) (stp o .I m) = ⟨.I, [k], [o + m], 0, 0⟩ := by
    simp only [stp, Node.rename, swapTo, List.map_cons, List.map_nil]
    congr 2 <;> split_ifs <;> omega
  have e : (ws k ++ [hd o k] ++ (List.range (m + 1)).map (stp o .I)).map
      (Node.rename (swapTo (o + (m + 1)) k)) = (chainN o k .I m).nodes := by
    rw [List.range_succ, List.map_append, List.map_append, List.map_append, List.map_append, hw,
      hc, List.map_singleton, List.map_singleton, List.map_singleton, hh, hs]
    simp only [chainN, List.append_assoc]
  have e1 : chainN o k .I (m + 1) = ⟨ws k ++ [hd o k] ++ (List.range (m + 1)).map (stp o .I) ++
      [⟨.I, [k], [o + (m + 1)], 0, 0⟩], (List.range (m + 1)).map (o + ·) ++ [o + (m + 1)],
      List.range (2 * k + 2)⟩ := by
    simp only [chainN, List.range_succ, List.map_append, List.map_singleton]
  rw [e1]
  rw [e] at h
  exact h

/-- The carried wire's identities dropped one by one: the steps, then the tail. -/
theorem chain_collapse (E : Env) (hD : ∀ x, labDim E x = E.dimv 0) (ho : 2 * k + 2 ≤ o) :
    ∀ m, Equal E 1 (chainN o k .I m) 1 (chainN o k .I 0)
  | 0 => Equal.refl _ _ _
  | m + 1 => (chain_drop o k E hD ho m).trans (chain_collapse E hD ho m)

theorem chain_netR (E : Env) (hD : ∀ x, labDim E x = E.dimv 0) (hφ : ∀ j, E.phase 0 j = 0)
    (ho : 2 * k + 2 ≤ o) : Equal E 1 (chainN o k .I 0) 1 (netR k) := by
  have h := Equal.dropIdentity E (ws k ++ [hd o k]) [] (List.range (2 * k + 2)) o k 0
    (by rw [hD]) (hD _) (by omega) (by simp; omega) (by simp)
  have hw : (ws k).map (Node.rename (swapTo o k)) = ws k := by
    refine (List.map_congr_left fun n hn => ?_).trans (List.map_id _)
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hn
    have := List.mem_range.1 hj
    simp only [Node.rename, swapTo, List.map_cons, List.map_nil, id]
    congr 2 <;> split_ifs <;> omega
  have e1 : chainN o k .I 0 = ⟨ws k ++ [hd o k] ++ [⟨.I, [k], [o], 0, 0⟩], [] ++ [o],
      List.range (2 * k + 2)⟩ := by
    simp [chainN]
  rw [e1]
  refine h.trans ?_
  rw [List.map_append, hw, List.map_singleton]
  refine Equal.of_prod E _ _ rfl rfl fun σ => ?_
  have hz := val_z11 E σ k (k + 1) 0 hφ
  simp only [netR, hd, Node.rename, swapTo, prodVal, List.map_append, List.map_cons,
    List.map_nil, List.prod_append, List.prod_cons, List.prod_nil, if_true,
    show k + 1 ≠ o by omega, if_false, mul_one, hz]
  ring

/-- The carried wire's spiders and identities against the goal's single head. -/
theorem chain_equal_goal (E : Env) (hD : ∀ x, labDim E x = E.dimv 0)
    (hφ : ∀ j, E.phase 0 j = 0) (ho : 2 * k + 2 ≤ o) :
    Equal E 1 (chainN o k .Z k) 1 (netR k) :=
  ((chain_z_i o k E hφ k).trans (chain_collapse o k E hD ho k)).trans (chain_netR o k E hD hφ ho)

open Diagram

/-- The goal: the head spider `0` and the node box `0` over the spider `1`. -/
def DR : Diagram :=
  { nodes := [⟨0, .Z, 0, 0, [0], [0]⟩, ⟨1, .Z, 0, 0, [0], [0]⟩],
    wires := [],
    outs := [⟨1, true, 0⟩, ⟨0, true, 0⟩],
    ins := [⟨0, false, 0⟩, ⟨1, false, 0⟩],
    boxes := [⟨0, 0, none, [1], []⟩] }

/-- The goal at `k` copies, copy `j` the spider `2 + j`. -/
def closedR (k : ℕ) : Diagram :=
  { nodes := ⟨0, .Z, 0, 0, [0], [0]⟩ :: (List.range k).map (fun j => ⟨2 + j, .Z, 0, 0, [0], [0]⟩),
    wires := [],
    outs := (List.range k).map (fun j => ⟨2 + j, true, 0⟩) ++ [⟨0, true, 0⟩],
    ins := ⟨0, false, 0⟩ :: (List.range k).map (fun j => ⟨2 + j, false, 0⟩),
    boxes := [] }

theorem expandR (k : ℕ) : DR.expand (ksOf [k]) = closedR k := by
  have h1 : DR.expand (ksOf [k]) =
      (DR.expandNode ⟨0, 0, none, [1], []⟩ k).expandFuel (ksOf [k]) 1 := rfl
  have hg : DR.gather ⟨0, 0, none, [1], []⟩ = DR := rfl
  have hfo : List.findIdx (inS ⟨0, 0, none, [1], []⟩) DR.outs = 0 := by decide
  have hfi : List.findIdx (inS ⟨0, 0, none, [1], []⟩) DR.ins = 1 := by decide
  have hno : DR.nimg ⟨0, 0, none, [1], []⟩ k ⟨0, true, 0⟩ = [⟨0, true, 0⟩] := by
    simp [nimg, inS, growImg, crossP, shiftPos, DR]
  have hni : DR.nimg ⟨0, 0, none, [1], []⟩ k ⟨0, false, 0⟩ = [⟨0, false, 0⟩] := by
    simp [nimg, inS, growImg, crossP, shiftPos, DR]
  have hc : ∀ j, DR.cpId ⟨0, 0, none, [1], []⟩ j 1 = 2 + j := fun j => by
    simp [cpId, fresh, DR]
  have hm : ∀ (f : ℕ → Port), (List.range k).flatMap (fun j => [f j]) = (List.range k).map f :=
    fun f => by induction (List.range k) with
      | nil => rfl
      | cons j l ih => simp [ih]
  have h2 : DR.expandNode ⟨0, 0, none, [1], []⟩ k = closedR k := by
    simp only [expandNode, hg, ite_self, copyNodes, spliceN]
    simp only [DR] at hfo hfi hno hni hc ⊢
    simp [hfo, hfi, hno, hni, hc, hm, inS, crossP, growSide, closedR]
  rw [h1, h2]
  rfl

/-- Boundary slot `i` of `closedR k`. -/
def bR (k i : ℕ) : Port :=
  if i < k then ⟨2 + i, true, 0⟩ else if i = k then ⟨0, true, 0⟩
  else if i = k + 1 then ⟨0, false, 0⟩ else ⟨2 + (i - (k + 2)), false, 0⟩

theorem bnd_closedR (k : ℕ) : (closedR k).bnd = (List.range (2 * k + 2)).map (bR k) := by
  rw [show 2 * k + 2 = k + 2 + k by omega, List.range_add,
    show List.range (k + 2) = List.range k ++ [k, k + 1] by
      rw [List.range_succ, List.range_succ]; simp]
  simp only [List.map_append, List.map_map, bnd, closedR, show List.range 2 = [0, 1] from rfl,
    List.map_cons, List.map_nil, List.append_assoc, List.singleton_append, List.cons_append]
  congr 1
  · exact List.map_congr_left fun j hj => by simp [bR, List.mem_range.1 hj]
  · simp only [Function.comp_def, bR]
    simp only [lt_irrefl, ↓reduceIte, ite_true, show ¬ k + 1 < k by omega,
      show k + 1 ≠ k by omega, List.nil_append, List.cons.injEq, true_and]
    exact List.map_congr_left fun j _ => by
      simp only [show ¬ k + 2 + j < k by omega, show k + 2 + j ≠ k by omega,
        show k + 2 + j ≠ k + 1 by omega, show k + 2 + j - (k + 2) = j by omega, ↓reduceIte]

theorem bR_inj (k i j : ℕ) (hi : i < 2 * k + 2) (hj : j < 2 * k + 2) (h : bR k i = bR k j) :
    i = j := by
  unfold bR at h
  split_ifs at h <;> simp only [Port.mk.injEq, and_true, Bool.true_eq_false, Bool.false_eq_true,
    and_false] at h <;> omega

theorem locateR (k i : ℕ) (hi : i < 2 * k + 2) : (closedR k).locate (bR k i) = some (.inl i) := by
  unfold locate
  rw [(List.idxOf?_eq_some_iff).2 ⟨by simpa [bnd_closedR] using hi, by simp [bnd_closedR],
    fun j hj h => by
      simp only [bnd_closedR, List.getElem_map, List.getElem_range] at h
      exact absurd (bR_inj k j i (by omega) hi h) (by omega)⟩]

theorem labelR (k i o : ℕ) (hi : i < 2 * k + 2) : (closedR k).label o (bR k i) = i := by
  simp [label, locateR k i hi]

theorem labR_h_out (k o : ℕ) : (closedR k).label o ⟨0, true, 0⟩ = k := by
  have := labelR k k o (by omega)
  simpa [bR] using this

theorem labR_h_in (k o : ℕ) : (closedR k).label o ⟨0, false, 0⟩ = k + 1 := by
  have := labelR k (k + 1) o (by omega)
  simpa [bR] using this

theorem labR_out (k o j : ℕ) (hj : j < k) : (closedR k).label o ⟨2 + j, true, 0⟩ = j := by
  have := labelR k j o (by omega)
  simpa [bR, hj] using this

theorem labR_in (k o j : ℕ) (hj : j < k) : (closedR k).label o ⟨2 + j, false, 0⟩ = k + 2 + j := by
  have := labelR k (k + 2 + j) o (by omega)
  simp only [bR, show ¬ k + 2 + j < k by omega, show k + 2 + j ≠ k by omega,
    show k + 2 + j ≠ k + 1 by omega, show k + 2 + j - (k + 2) = j by omega, ↓reduceIte] at this
  exact this

/-- The goal's network at `k` copies. -/
theorem netR_closed (k o : ℕ) : (DR.expand (ksOf [k])).toNet o =
    ⟨⟨.Z, [k], [k + 1], 0, 0⟩ :: (List.range k).map (fun j => ⟨.Z, [j], [k + 2 + j], 0, 0⟩), [],
      List.range (2 * k + 2)⟩ := by
  rw [expandR]
  have hb : (closedR k).bnd.length = 2 * k + 2 := by simp [bnd_closedR]
  have hw : (closedR k).wires.length = 0 := rfl
  have hn : (closedR k).nodes = ⟨0, .Z, 0, 0, [0], [0]⟩ ::
      (List.range k).map (fun j => ⟨2 + j, .Z, 0, 0, [0], [0]⟩) := rfl
  simp only [toNet, hb, hw, hn, List.range_zero, List.map_nil, List.map_cons, List.map_map]
  congr 2
  · simp [DNode.node, labR_h_out, labR_h_in]
  · exact List.map_congr_left fun j hj => by
      simp [DNode.node, labR_out k o j (List.mem_range.1 hj),
        labR_in k o j (List.mem_range.1 hj)]

open Diagram

/-- The start: per copy the spider `0` on its own wire and the spider `1` on the carried wire,
`1`'s output feeding the next copy's input; identities `2` and `3` hold the chain's ends on the
boundary. -/
def DL : Diagram :=
  { nodes := [⟨0, .Z, 0, 0, [0], [0]⟩, ⟨1, .Z, 0, 0, [0], [0]⟩, ⟨2, .I, 0, 0, [0], [0]⟩,
      ⟨3, .I, 0, 0, [0], [0]⟩],
    wires := [(⟨2, true, 0⟩, ⟨1, false, 0⟩), (⟨1, true, 0⟩, ⟨3, false, 0⟩)],
    outs := [⟨0, true, 0⟩, ⟨3, true, 0⟩],
    ins := [⟨2, false, 0⟩, ⟨0, false, 0⟩],
    boxes := [⟨0, 0, none, [0, 1], []⟩] }

/-- The chain box of `DL`. -/
def cL : DChain := ⟨⟨0, 0, none, [0, 1], []⟩, [(⟨1, true, 0⟩, ⟨1, false, 0⟩)]⟩

/-- Wire `i` of the start at `k` copies: the head into copy `0`, copy `i - 1` into copy `i`,
copy `k - 1` into the tail. -/
def wL (k i : ℕ) : Port × Port :=
  (if i = 0 then ⟨2, true, 0⟩ else ⟨5 + 2 * (i - 1), true, 0⟩,
    if i = k then ⟨3, false, 0⟩ else ⟨5 + 2 * i, false, 0⟩)

/-- The start at `k` copies, copy `j` the spiders `4 + 2 j` and `5 + 2 j`. -/
def closedL (k : ℕ) : Diagram :=
  { nodes := (List.range k).map (fun j => ⟨4 + 2 * j, .Z, 0, 0, [0], [0]⟩) ++
      (List.range k).map (fun j => ⟨5 + 2 * j, .Z, 0, 0, [0], [0]⟩) ++
      [⟨2, .I, 0, 0, [0], [0]⟩, ⟨3, .I, 0, 0, [0], [0]⟩],
    wires := (List.range (k + 1)).map (wL k),
    outs := (List.range k).map (fun j => ⟨4 + 2 * j, true, 0⟩) ++ [⟨3, true, 0⟩],
    ins := ⟨2, false, 0⟩ :: (List.range k).map (fun j => ⟨4 + 2 * j, false, 0⟩),
    boxes := [] }

/-- `DL` without its link wires. -/
def DLf : Diagram := { DL with wires := [] }

theorem DLf_eq : ({ DL with wires := DL.wires.filter fun w => !onLink cL w } : Diagram) = DLf := by
  decide

theorem copyL (k : ℕ) : DLf.copyNodes cL.box k =
    { closedL k with wires := [] } := by
  have hfo : List.findIdx (inS cL.box) DLf.outs = 0 := by decide
  have hfi : List.findIdx (inS cL.box) DLf.ins = 1 := by decide
  have hno : DLf.nimg cL.box k ⟨3, true, 0⟩ = [⟨3, true, 0⟩] := by
    simp [nimg, inS, growImg, crossP, shiftPos, DLf, DL, cL]
  have hni : DLf.nimg cL.box k ⟨2, false, 0⟩ = [⟨2, false, 0⟩] := by
    simp [nimg, inS, growImg, crossP, shiftPos, DLf, DL, cL]
  have hc0 : ∀ j, DLf.cpId cL.box j 0 = 4 + 2 * j := fun j => by
    simp [cpId, fresh, DLf, DL, cL]; omega
  have hc1 : ∀ j, DLf.cpId cL.box j 1 = 5 + 2 * j := fun j => by
    simp [cpId, fresh, DLf, DL, cL]; omega
  have hm : ∀ (f : ℕ → Port), (List.range k).flatMap (fun j => [f j]) = (List.range k).map f :=
    fun f => by induction (List.range k) with
      | nil => rfl
      | cons j l ih => simp [ih]
  simp only [copyNodes, spliceN]
  simp only [DLf, DL, cL] at hfo hfi hno hni hc0 hc1 ⊢
  simp [hfo, hfi, hno, hni, hc0, hc1, hm, inS, crossP, growSide, closedL]

theorem partner1 : DL.partner ⟨1, true, 0⟩ = some ⟨3, false, 0⟩ := by decide

theorem partner2 : DL.partner ⟨1, false, 0⟩ = some ⟨2, true, 0⟩ := by decide

theorem cp1 (j : ℕ) (b : Bool) (q : ℕ) :
    DL.cpPort ⟨0, 0, none, [0, 1], []⟩ j ⟨1, b, q⟩ = ⟨5 + 2 * j, b, q⟩ := by
  simp [cpPort, cpId, fresh, DL]; omega

theorem links_closed (k : ℕ) : cL.links.flatMap (DL.linkWires cL k) = (List.range (k + 1)).map (wL k) := by
  simp only [cL, List.flatMap_cons, List.flatMap_nil, List.append_nil, linkWires, partner1, partner2]
  cases k with
  | zero => simp [wL]
  | succ k =>
    simp only [show k + 1 ≠ 0 by omega, ↓reduceIte, Nat.add_sub_cancel, cp1]
    rw [List.range_succ_eq_map, List.range_succ]
    simp only [List.map_cons, List.map_append, List.map_map, List.map_singleton, wL, ↓reduceIte,
      show k + 1 ≠ 0 by omega, List.cons_append, List.cons.injEq, true_and]
    refine ⟨by simp, ?_⟩
    congr 1
    · exact List.map_congr_left fun j hj => by
        have := List.mem_range.1 hj
        simp [wL, cp1, show j + 1 ≠ k + 1 by omega]
        omega

theorem expandL (k : ℕ) : DL.expandChain cL k = closedL k := by
  simp only [expandChain, DLf_eq, copyL, links_closed]
  simp [closedL]


/-- Boundary slot `i` of `closedL k`. -/
def bL (k i : ℕ) : Port :=
  if i < k then ⟨4 + 2 * i, true, 0⟩ else if i = k then ⟨3, true, 0⟩
  else if i = k + 1 then ⟨2, false, 0⟩ else ⟨4 + 2 * (i - (k + 2)), false, 0⟩

theorem bnd_closedL (k : ℕ) : (closedL k).bnd = (List.range (2 * k + 2)).map (bL k) := by
  rw [show 2 * k + 2 = k + 2 + k by omega, List.range_add,
    show List.range (k + 2) = List.range k ++ [k, k + 1] by
      rw [List.range_succ, List.range_succ]; simp]
  simp only [List.map_append, List.map_map, bnd, closedL, List.map_cons, List.map_nil,
    List.append_assoc, List.cons_append]
  congr 1
  · exact List.map_congr_left fun j hj => by simp [bL, List.mem_range.1 hj]
  · simp only [Function.comp_def, bL]
    simp only [lt_irrefl, ↓reduceIte, show ¬ k + 1 < k by omega,
      show k + 1 ≠ k by omega, List.nil_append, List.cons.injEq, true_and]
    exact List.map_congr_left fun j _ => by
      simp only [show ¬ k + 2 + j < k by omega, show k + 2 + j ≠ k by omega,
        show k + 2 + j ≠ k + 1 by omega, show k + 2 + j - (k + 2) = j by omega, ↓reduceIte]

theorem bL_inj (k i j : ℕ) (hi : i < 2 * k + 2) (hj : j < 2 * k + 2) (h : bL k i = bL k j) :
    i = j := by
  unfold bL at h
  split_ifs at h <;> simp only [Port.mk.injEq, and_true, Bool.true_eq_false, Bool.false_eq_true,
    and_false] at h <;> omega

theorem locateL (k i : ℕ) (hi : i < 2 * k + 2) : (closedL k).locate (bL k i) = some (.inl i) := by
  unfold locate
  rw [(List.idxOf?_eq_some_iff).2 ⟨by simpa [bnd_closedL] using hi, by simp [bnd_closedL],
    fun j hj h => by
      simp only [bnd_closedL, List.getElem_map, List.getElem_range] at h
      exact absurd (bL_inj k j i (by omega) hi h) (by omega)⟩]

theorem labelL (k i o : ℕ) (hi : i < 2 * k + 2) : (closedL k).label o (bL k i) = i := by
  simp [label, locateL k i hi]

theorem notL (k : ℕ) (p : Port) (h : ∀ i < 2 * k + 2, bL k i ≠ p) : p ∉ (closedL k).bnd := by
  simp only [bnd_closedL, List.mem_map, List.mem_range, not_exists, not_and]
  exact h

theorem locateL_wire (k i : ℕ) (p : Port) (hi : i < k + 1) (hp : (wL k i).1 = p ∨ (wL k i).2 = p)
    (hb : p ∉ (closedL k).bnd) (hprev : ∀ j < i, ¬((wL k j).1 = p ∨ (wL k j).2 = p)) :
    (closedL k).locate p = some (.inr i) := by
  unfold locate
  rw [(List.idxOf?_eq_none_iff).2 hb]
  simp only [Option.map_eq_some_iff]
  refine ⟨i, (List.findIdx?_eq_some_iff_getElem).2 ⟨by simpa [closedL] using hi, ?_, ?_⟩, rfl⟩
  · simpa [closedL] using hp
  · intro j hj
    simpa [closedL] using hprev j hj

theorem labL_w_out (k o j : ℕ) (hj : j < k) : (closedL k).label o ⟨4 + 2 * j, true, 0⟩ = j := by
  have := labelL k j o (by omega)
  simpa [bL, hj] using this

theorem labL_w_in (k o j : ℕ) (hj : j < k) :
    (closedL k).label o ⟨4 + 2 * j, false, 0⟩ = k + 2 + j := by
  have := labelL k (k + 2 + j) o (by omega)
  simp only [bL, show ¬ k + 2 + j < k by omega, show k + 2 + j ≠ k by omega,
    show k + 2 + j ≠ k + 1 by omega, show k + 2 + j - (k + 2) = j by omega, ↓reduceIte] at this
  exact this

theorem labL_t_out (k o : ℕ) : (closedL k).label o ⟨3, true, 0⟩ = k := by
  have := labelL k k o (by omega)
  simpa [bL] using this

theorem labL_h_in (k o : ℕ) : (closedL k).label o ⟨2, false, 0⟩ = k + 1 := by
  have := labelL k (k + 1) o (by omega)
  simpa [bL] using this

theorem notL_wire (k : ℕ) (x : ℕ) (b : Bool) (hx : x = 2 ∨ x = 3 ∨ x % 2 = 1)
    (hb : x = 2 → b = true) (hc : x = 3 → b = false) : (⟨x, b, 0⟩ : Port) ∉ (closedL k).bnd :=
  notL k _ fun i _ => by
    unfold bL
    split_ifs <;> simp only [ne_eq, Port.mk.injEq, not_and] <;> intros <;> subst_vars <;> simp_all <;>
      omega

theorem labL_h_out (k o : ℕ) : (closedL k).label o ⟨2, true, 0⟩ = o := by
  have := locateL_wire k 0 ⟨2, true, 0⟩ (by omega) (by simp [wL])
    (notL_wire k 2 true (by omega) (fun _ => rfl) (by omega)) (fun j hj => by omega)
  simp [label, this]

theorem labL_c_in (k o j : ℕ) (hj : j < k) : (closedL k).label o ⟨5 + 2 * j, false, 0⟩ = o + j := by
  have := locateL_wire k j ⟨5 + 2 * j, false, 0⟩ (by omega) (by simp [wL, show j ≠ k by omega])
    (notL_wire k _ false (by omega) (by omega) (fun _ => rfl))
    (fun i hi => by unfold wL; split_ifs <;> simp <;> omega)
  simp [label, this]

theorem labL_c_out (k o j : ℕ) (hj : j < k) :
    (closedL k).label o ⟨5 + 2 * j, true, 0⟩ = o + j + 1 := by
  have := locateL_wire k (j + 1) ⟨5 + 2 * j, true, 0⟩ (by omega) (by simp [wL])
    (notL_wire k _ true (by omega) (by omega) (by omega))
    (fun i hi => by unfold wL; split_ifs <;> simp <;> omega)
  simp [label, this]; omega

theorem labL_t_in (k o : ℕ) : (closedL k).label o ⟨3, false, 0⟩ = o + k := by
  have := locateL_wire k k ⟨3, false, 0⟩ (by omega) (by simp [wL])
    (notL_wire k 3 false (by omega) (by omega) (fun _ => rfl))
    (fun i hi => by unfold wL; split_ifs <;> simp <;> omega)
  simp [label, this]

/-- The start's network at `k` copies: the copies' own spiders, the carried wire's spiders, the
head and the tail. -/
theorem netL_closed (k o : ℕ) : (DL.expandChain cL k).toNet o =
    ⟨(List.range k).map (fun j => ⟨.Z, [j], [k + 2 + j], 0, 0⟩) ++
      (List.range k).map (fun j => ⟨.Z, [o + j + 1], [o + j], 0, 0⟩) ++
      [⟨.I, [o], [k + 1], 0, 0⟩, ⟨.I, [k], [o + k], 0, 0⟩],
      (List.range (k + 1)).map (o + ·), List.range (2 * k + 2)⟩ := by
  rw [expandL]
  have hb : (closedL k).bnd.length = 2 * k + 2 := by simp [bnd_closedL]
  have hw : (closedL k).wires.length = k + 1 := by simp [closedL]
  have hn : (closedL k).nodes = (List.range k).map (fun j => ⟨4 + 2 * j, .Z, 0, 0, [0], [0]⟩) ++
      (List.range k).map (fun j => ⟨5 + 2 * j, .Z, 0, 0, [0], [0]⟩) ++
      [⟨2, .I, 0, 0, [0], [0]⟩, ⟨3, .I, 0, 0, [0], [0]⟩] := rfl
  simp only [toNet, hb, hw, hn, List.map_append, List.map_cons, List.map_nil, List.map_map]
  congr 3
  · exact List.map_congr_left fun j hj => by
      simp [DNode.node, labL_w_out k o j (List.mem_range.1 hj), labL_w_in k o j (List.mem_range.1 hj)]
  · exact List.map_congr_left fun j hj => by
      simp [DNode.node, labL_c_out k o j (List.mem_range.1 hj), labL_c_in k o j (List.mem_range.1 hj)]
  all_goals simp [DNode.node, labL_h_out, labL_h_in, labL_t_out, labL_t_in]

/-- The start's and the goal's networks at every copy count. -/
theorem cyclic_every (E : Env) (hD : ∀ x, labDim E x = E.dimv 0) (hφ : ∀ j, E.phase 0 j = 0)
    (k o : ℕ) (ho : 2 * k + 2 ≤ o) :
    Equal E 1 ((DL.expandChain cL k).toNet o) 1 ((DR.expand (ksOf [k])).toNet o) := by
  rw [netL_closed, netR_closed]
  refine (chain_equal_goal o k E hD hφ ho).reorderL E ?_ (List.Perm.refl _) (List.Perm.refl _)
  simp only [chainN, ws, hd, List.append_assoc, List.singleton_append, List.cons_append,
    List.nil_append]
  exact List.Perm.append_left _ List.perm_middle.symm

end ChainNode

end ArchytasZX
