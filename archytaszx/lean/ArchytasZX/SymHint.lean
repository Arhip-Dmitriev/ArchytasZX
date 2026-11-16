/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymChunk

/-!
# Symbolic normal forms with hinted pivot passes

`trNodesG` is `trNodesB` with any value-preserving normaliser per node; `eval_trG` and
`equal_symG` prove `Equal` through it. `pivPassH c` pivots the first `c` deltas on every factor
and each later delta only on the first `c` deltas and the phases (`pivTail`); `canonTmH`,
`normTmH`, `nfGoH` are the normaliser with one such `c` per canonicalisation and the
eliminations given by sum, rule and delta (`elimStepH`), read from a hint list; `nfGoH_den`
shows it keeps the value.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Any normaliser per node -/

/-- `trNodesB` with the expression after node `i` normalised by `nf i`. -/
def trNodesG (nf : ℕ → List STm → List STm) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ) :
    List ℕ → ℕ → List Node → Option (List STm)
  | _, _, [] => some [⟨CMon.ofQ 1, [], [], []⟩]
  | pre, i, n :: ns =>
    match trNode dt pt rl (OFF + i) n, trNodesG nf dt pt rl labs int (pre ++ (n.outs ++ n.ins))
        (i + 1) ns with
    | some a, some b =>
      if exMulOK a b then some (nf i (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)))
      else none
    | _, _ => none

theorem trNodesG_ok (E : Env) (ρ : Val) (hρ : ρ.OK) (nf : ℕ → List STm → List STm)
    (hnf : ∀ i e ν, exDen ρ ν (nf i e) = exDen ρ ν e)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ)
    (hD : ∀ l ∈ int, labDim E l = (labDM dt labs l).den ρ.dv) (ns : List Node)
    (hn : ∀ n ∈ ns, NodeOK E ρ dt pt rl n ∧
      (bsKind n.kind = true → E.dimv n.ph = (dt n.ph).den ρ.dv) ∧ (∀ l ∈ n.outs ++ n.ins, l < OFF) ∧
      ∀ p ∈ (n.outs ++ n.ins).zip (legDMs dt n), labDim E p.1 = p.2.den ρ.dv)
    (pre : List ℕ) (i : ℕ) (e : List STm)
    (h : trNodesG nf dt pt rl labs int pre i ns = some e) (σ : ℕ → ℕ)
    (hσ : ∀ l ∈ labelsOf ns, l ∉ firstOf int pre ns → σ l < labDim E l) :
    sumOver (labDim E) (firstOf int pre ns) (prodVal E ns) σ = exDen ρ (fun x => (σ x : ℤ)) e := by
  induction ns generalizing pre i e σ with
  | nil =>
    simp only [trNodesG, Option.some.injEq] at h
    subst h
    simp [firstOf, labelsOf, sumOver, prodVal, exDen_cons, exDen_nil, den_zero, ofQ_den, bodyOf]
  | cons n ns ih =>
    simp only [trNodesG] at h
    split at h
    · next a b ha hb =>
      split_ifs at h with hok
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨hnode, hph, hlt, hdl⟩ := hn n List.mem_cons_self
      have hperm := firstOf_perm int pre n ns
      have hmem : ∀ l, l ∈ firstOf int pre (n :: ns) →
          l ∈ firstAt int pre n ∨ l ∈ firstOf int (pre ++ (n.outs ++ n.ins)) ns := by
        intro l hl; rw [← List.mem_append]; exact hperm.subset hl
      have hS : ∀ x ∈ n.outs ++ n.ins, x ∉ firstOf int (pre ++ (n.outs ++ n.ins)) ns := by
        intro x hx hx'
        have h1 := (List.mem_filter.1 hx').2
        have h2 : (pre ++ (n.outs ++ n.ins)).contains x = true := by simp [hx]
        rw [h2] at h1; simp at h1
      rw [sumOver_perm _ hperm, sumOver_append, hnf, exDen_sums]
      unfold labSums
      refine sumOver_sumsV ρ _ _ _ (fun l hl => hD l (List.mem_filter.1 hl).1) _ _ σ
        fun τ hτ hr => ?_
      have hlab : ∀ l ∈ labelsOf (n :: ns), l ∉ firstOf int pre (n :: ns) →
          l ∉ firstAt int pre n → τ l < labDim E l := by
        intro l hl hn' hF
        rw [hτ l hF]; exact hσ l hl hn'
      have hprod : ∀ τ', prodVal E (n :: ns) τ' = val E τ' n * prodVal E ns τ' := fun τ' => by
        simp [prodVal]
      rw [sumOver_congr _ _ _ (fun τ' => val E τ' n * prodVal E ns τ') τ (fun τ' _ _ => hprod τ'),
        sumOver_const_mul _ _ (n.outs ++ n.ins) _ _ (val_reads E n) hS τ]
      rw [ih (fun m hm => hn m (List.mem_cons_of_mem _ hm)) _ _ _ hb τ (fun l hl hl' => by
        by_cases hF : l ∈ firstAt int pre n
        · exact hr l hF
        · refine hlab l (by simp [labelsOf] at hl ⊢; tauto) (fun h' => ?_) hF
          rcases hmem l h' with h1 | h1
          · exact hF h1
          · exact hl' h1)]
      rw [exDen_mul ρ hρ a b hok, ← val_trG E ρ hρ dt pt rl (OFF + i) n a hnode hph τ _
        (fun h' => by have := hlt _ h'; omega) (fun p hp => ⟨?_, rfl⟩) ha]
      have hl : p.1 ∈ n.outs ++ n.ins := (List.of_mem_zip hp).1
      rw [← hdl p hp]
      by_cases hF : p.1 ∈ firstAt int pre n
      · exact hr _ hF
      · refine hlab _ (by simp [labelsOf]; tauto) (fun h' => ?_) hF
        rcases hmem _ h' with h1 | h1
        · exact hF h1
        · exact hS _ hl h1
    · simp at h

/-- The network's entry over its boundary labels through `trNodesG`. -/
def trNetG (nf : ℕ → List STm → List STm) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (N : Net) : Option (List STm) :=
  (trNodesG nf dt pt rl labs N.int [] 0 N.nodes).map
    (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)))


theorem eval_trG (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (nf : ℕ → List STm → List STm) (hnf : ∀ i e ν, exDen ρ ν (nf i e) = exDen ρ ν e)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N : Net) (e : List STm) (hc : netChk fl dt pt rl labs IX N = true)
    (ht : trNetG nf dt pt rl labs N = some e) (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = exDen ρ (fun x => (σ x : ℤ)) e := by
  unfold trNetG at ht
  cases h0 : trNodesG nf dt pt rl labs N.int [] 0 N.nodes with
  | none => rw [h0] at ht; simp at ht
  | some e0 =>
    rw [h0] at ht
    simp only [Option.map_some, Option.some.injEq] at ht
    subst ht
    rw [exDen_sums]
    obtain ⟨hc, hci⟩ := netChk_ok E ρ fl hfl dt pt rl labs IX hE hP hL N hc
    set rest := N.int.filter fun l => !(labelsOf N.nodes).contains l
    have hfirst : firstOf N.int [] N.nodes = N.int.filter fun l => (labelsOf N.nodes).contains l := by
      unfold firstOf; exact List.filter_congr fun l _ => by simp
    have hperm : N.int.Perm (rest ++ firstOf N.int [] N.nodes) := by
      rw [hfirst]
      exact (List.filter_append_perm _ N.int).symm.trans List.perm_append_comm
    unfold eval labSums
    rw [sumOver_perm _ hperm, sumOver_append]
    refine sumOver_sumsV ρ _ _ _ (fun l hl => hci l (List.mem_filter.1 hl).1) _ _ σ
      fun τ hτ hr => ?_
    refine trNodesG_ok E ρ hρ nf hnf dt pt rl labs N.int hci N.nodes
      (fun n hn => ⟨(hc n hn).1, (hc n hn).2.1, fun l hl => ((hc n hn).2.2.1 l hl).1,
        (hc n hn).2.2.2⟩) [] 0 e0 h0 τ fun l hl hl' => ?_
    obtain ⟨n, hnm, hln⟩ := List.mem_flatMap.1 hl
    have hbi := ((hc n hnm).2.2.1 l hln).2
    have hni : l ∉ N.int := fun hi => hl' (by
      rw [hfirst]; exact List.mem_filter.2 ⟨hi, by simpa using hl⟩)
    have hnr : l ∉ rest := fun h' => hni (List.mem_filter.1 h').1
    rcases hbi with hb | hi
    · rw [hτ l hnr]; exact hσ l hb
    · exact absurd hi hni

theorem equal_symG (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (nf0 nf1 : ℕ → List STm → List STm) (hnf0 : ∀ i e ν, exDen ρ ν (nf0 i e) = exDen ρ ν e)
    (hnf1 : ∀ i e ν, exDen ρ ν (nf1 i e) = exDen ρ ν e) (e0 e1 : List STm)
    (h0 : trNetG nf0 dt pt rl labs N0 = some e0) (h1 : trNetG nf1 dt pt rl labs N1 = some e1)
    (h : (N0.bnd.isPerm N1.bnd && netChk fl dt pt rl labs IX N0 && netChk fl dt pt rl labs IX N1 &&
      exConst S0 && exConst S1 && (exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
      (nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty)) = true) :
    Equal E s0 N0 s1 N1 := by
  simp only [Bool.and_eq_true, List.isEmpty_iff] at h
  obtain ⟨⟨⟨⟨⟨hperm, hc0⟩, hc1⟩, hk0⟩, hk1⟩, ⟨hm0, hm1⟩, hnf⟩ := h
  have hp : N0.bnd.Perm N1.bnd := List.isPerm_iff.1 hperm
  refine ⟨hp, fun σ hσ => ?_⟩
  set ν : ℕ → ℤ := fun x => (σ x : ℤ)
  have hz := nfEx_zero ρ hρ fl hfl _ hnf ν
  rw [exDen_append, exDen_mul ρ hρ _ _ hm0, exDen_mul ρ hρ _ _ hm1, exDen_neg,
    ← eval_trG E ρ hρ fl hfl nf0 hnf0 dt pt rl labs IX hE hP hL N0 e0 hc0 h0 σ hσ,
    ← eval_trG E ρ hρ fl hfl nf1 hnf1 dt pt rl labs IX hE hP hL N1 e1 hc1 h1 σ (hσ.perm hp),
    exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
    ← hs1] at hz
  linear_combination hz

/-- `equal_symG` with the check's parts as separate hypotheses. -/
theorem equal_symS (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (nf0 nf1 : ℕ → List STm → List STm) (hnf0 : ∀ i e ν, exDen ρ ν (nf0 i e) = exDen ρ ν e)
    (hnf1 : ∀ i e ν, exDen ρ ν (nf1 i e) = exDen ρ ν e) (e0 e1 : List STm)
    (h0 : trNetG nf0 dt pt rl labs N0 = some e0) (h1 : trNetG nf1 dt pt rl labs N1 = some e1)
    (hp : N0.bnd.isPerm N1.bnd = true) (hc0 : netChk fl dt pt rl labs IX N0 = true)
    (hc1 : netChk fl dt pt rl labs IX N1 = true) (hk : (exConst S0 && exConst S1) = true)
    (hm : (exMulOK S0 e0 && exMulOK (exNeg S1) e1) = true)
    (hz : (nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty = true) :
    Equal E s0 N0 s1 N1 := by
  refine equal_symG E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 N1 S0 S1 s0 s1 hs0 hs1 nf0 nf1
    hnf0 hnf1 e0 e1 h0 h1 ?_
  simp only [Bool.and_eq_true] at hk hm ⊢
  exact ⟨⟨⟨⟨⟨hp, hc0⟩, hc1⟩, hk.1⟩, hk.2⟩, hm, hz⟩

/-! ## Blocks through `trNodesG` -/

/-- One node's step of `trNodesG` on the tail's expression `b`. -/
def trStepG (nf : ℕ → List STm → List STm) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node)
    (b : List STm) : Option (List STm) :=
  match trNode dt pt rl (OFF + i) n with
  | some a =>
    if exMulOK a b then some (nf i (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)))
    else none
  | none => none

theorem trNodesG_cons' (nf : ℕ → List STm → List STm) (dt : ℕ → DMon)
    (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (int pre : List ℕ) (i : ℕ) (n : Node) (ns : List Node) :
    trNodesG nf dt pt rl labs int pre i (n :: ns) =
      match trNodesG nf dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns with
      | some b => trStepG nf dt pt rl labs int pre i n b
      | none => none := by
  conv_lhs => rw [trNodesG]
  unfold trStepG
  cases trNodesG nf dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns <;>
    cases trNode dt pt rl (OFF + i) n <;> rfl

/-- `trNodesG` on the nodes `xs` with the tail's expression `b0`, each tail forced. -/
def trBlockG (nf : ℕ → List STm → List STm) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ) (b0 : List STm) :
    List ℕ → ℕ → List Node → Option (List STm)
  | _, _, [] => some b0
  | pre, i, n :: ns =>
    match trBlockG nf dt pt rl labs int b0 (pre ++ (n.outs ++ n.ins)) (i + 1) ns with
    | some b => fL fTm b fun b => trStepG nf dt pt rl labs int pre i n b
    | none => none

theorem trNodesG_block (nf : ℕ → List STm → List STm) (dt : ℕ → DMon)
    (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ)
    (b : List STm) : ∀ (xs ys : List Node) (pre : List ℕ) (i : ℕ),
    trNodesG nf dt pt rl labs int (blockPre pre xs) (i + xs.length) ys = some b →
    trNodesG nf dt pt rl labs int pre i (xs ++ ys) = trBlockG nf dt pt rl labs int b pre i xs := by
  intro xs
  induction xs with
  | nil => intro ys pre i h; simpa [blockPre, trBlockG] using h
  | cons n xs ih =>
    intro ys pre i h
    have h' : trNodesG nf dt pt rl labs int (blockPre (pre ++ (n.outs ++ n.ins)) xs)
        (i + 1 + xs.length) ys = some b := by
      have e : i + (n :: xs).length = i + 1 + xs.length := by simp [List.length_cons]; omega
      rw [← e]
      exact h
    rw [List.cons_append, trNodesG_cons', ih ys _ _ h']
    simp only [trBlockG]
    cases trBlockG nf dt pt rl labs int b (pre ++ (n.outs ++ n.ins)) (i + 1) xs with
    | none => rfl
    | some c => simp [fL_eq _ (fun a k => fTm_eq a k)]

theorem trNetG_of (nf : ℕ → List STm → List STm) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (N : Net) (ns : List Node) (b : List STm)
    (hn : N.nodes = ns) (h : trNodesG nf dt pt rl labs N.int [] 0 ns = some b) :
    trNetG nf dt pt rl labs N =
      some (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)) b) := by
  unfold trNetG
  rw [hn, h]
  subst hn
  rfl

/-! ## Hinted pivot passes -/

/-- Each delta of the third list pivoting only the first list and the phases. -/
def pivTail : List Fac → List Fac → List Fac → List Fac → List Fac × List Fac
  | C, U, [], ps => (C ++ U, ps)
  | C, U, f :: B, ps =>
    match stepOf f with
    | some σ => pivTail (C.map σ) (U ++ [f]) B (ps.map σ)
    | none => pivTail C (U ++ [f]) B ps

theorem part_body (dv : ℕ → ℕ) (C U B ps : List Fac) (f : Fac) (σ : Fac → Fac)
    (hσ : StepOK dv f σ) (ν : ℕ → ℤ) :
    bodyOf dv (C.map σ ++ (U ++ f :: B)) (ps.map σ) ν = bodyOf dv (C ++ (U ++ f :: B)) ps ν := by
  simp only [bodyOf, List.map_append, List.map_cons, List.prod_append, List.prod_cons,
    List.map_map]
  rcases delV_cases dv ν f with h0 | h1
  · rw [h0]; ring
  · have hv := hσ ν h1
    have e1 : ∀ l : List Fac, (l.map (delV dv ν ∘ σ)).prod = (l.map (delV dv ν)).prod :=
      fun l => congrArg List.prod (List.map_congr_left fun g _ => (hv g).1)
    have e2 : ∀ l : List Fac, (l.map (phV dv ν ∘ σ)).prod = (l.map (phV dv ν)).prod :=
      fun l => congrArg List.prod (List.map_congr_left fun g _ => (hv g).2)
    rw [e1, e2]

theorem pivTail_ok (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) : ∀ (B C U ps : List Fac) (ν : ℕ → ℤ),
    bodyOf dv (pivTail C U B ps).1 (pivTail C U B ps).2 ν = bodyOf dv (C ++ (U ++ B)) ps ν := by
  intro B
  induction B with
  | nil => intro C U ps ν; simp [pivTail]
  | cons f B ih =>
    intro C U ps ν
    simp only [pivTail]
    split
    · next σ hq =>
      rw [ih, List.append_assoc, List.singleton_append]
      exact part_body dv C U B ps f σ (stepOf_ok dv hdv f σ hq) ν
    · rw [ih, List.append_assoc, List.singleton_append]

/-- `pivPass` on the first `c` deltas, `pivTail` on the rest. -/
def pivPassH (c : ℕ) (ds ps : List Fac) : List Fac × List Fac :=
  pivTail ((pivPass c [] ds ps).1.take c) [] ((pivPass c [] ds ps).1.drop c) (pivPass c [] ds ps).2

theorem pivPassH_ok (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (c : ℕ) (ds ps : List Fac) (ν : ℕ → ℤ) :
    bodyOf dv (pivPassH c ds ps).1 (pivPassH c ds ps).2 ν = bodyOf dv ds ps ν := by
  unfold pivPassH
  rw [pivTail_ok dv hdv, List.nil_append, List.take_append_drop, pivPass_ok dv hdv,
    List.nil_append]

/-- `canonTm'` with the pivot pass `pf`. -/
def canonTmP (pf : List Fac → List Fac → List Fac × List Fac) (fl : List ℕ) (t : STm) :
    Option STm :=
  let pp := pf t.dels t.phs
  match normDels fl pp.1 with
  | none => none
  | some ds =>
    let np := normPhs (phMerge pp.2)
    some ⟨CMon.mul t.c np.1, t.sums, ds.foldr delIns [], np.2.foldr facIns []⟩

theorem canonTmP_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (pf : List Fac → List Fac → List Fac × List Fac)
    (hpf : ∀ ds ps ν, bodyOf ρ.dv (pf ds ps).1 (pf ds ps).2 ν = bodyOf ρ.dv ds ps ν) (t : STm) :
    TmRes ρ t (canonTmP pf fl t) := by
  have hpp : ∀ ν, body ρ.dv t ν = bodyOf ρ.dv (pf t.dels t.phs).1
      (pf t.dels t.phs).2 ν := fun ν => by
    rw [hpf]; rfl
  unfold canonTmP
  simp only
  split
  · next h =>
    have hn : ∀ ν, ((pf t.dels t.phs).1.map (delV ρ.dv ν)).prod = 0 :=
      fun ν => by have := normDels_ok ρ fl hfl (pf t.dels t.phs).1 ν
                  rw [h] at this; exact this
    intro ν
    unfold STm.den
    rw [sumsV_congr _ _ _ (fun _ => 0) (fun ν' => by rw [hpp ν']; simp [bodyOf, hn ν']),
      sumsV_zero, mul_zero]
  · next ds h =>
    have hn : ∀ ν, (ds.map (delV ρ.dv ν)).prod =
        ((pf t.dels t.phs).1.map (delV ρ.dv ν)).prod :=
      fun ν => by have := normDels_ok ρ fl hfl (pf t.dels t.phs).1 ν
                  rw [h] at this; exact this
    intro ν
    unfold STm.den
    simp only [CMon.den_mul ρ hρ]
    have key : ∀ ν', bodyOf ρ.dv (ds.foldr delIns [])
        ((normPhs (phMerge (pf t.dels t.phs).2)).2.foldr facIns []) ν' *
        (normPhs (phMerge (pf t.dels t.phs).2)).1.den ρ = body ρ.dv t ν' := by
      intro ν'
      rw [hpp ν']
      simp only [bodyOf, delSort_ok, facSort_ok, hn ν']
      rw [← phMerge_ok ρ.dv ν' (pf t.dels t.phs).2,
        normPhs_ok ρ hρ (phMerge (pf t.dels t.phs).2) ν']
      ring
    rw [← sumsV_congr _ _ _ _ key, sumsV_congr _ _ _ _ (fun _ => mul_comm _ _),
      sumsV_mul_left]
    simp only [body_eq]
    ring


/-- `canonTm` with the hinted pivot pass `pivPassH c`. -/
def canonTmH (c : ℕ) (fl : List ℕ) (t : STm) : Option STm :=
  canonTmP (pivPassH c) fl (renList t 0 (bvs t))

theorem canonTmH_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (c : ℕ) (t : STm) :
    TmRes ρ t (canonTmH c fl t) := by
  have h := canonTmP_ok ρ hρ fl hfl (pivPassH c) (pivPassH_ok ρ.dv hρ.dpos c)
    (renList t 0 (bvs t))
  unfold canonTmH
  revert h
  cases canonTmP (pivPassH c) fl (renList t 0 (bvs t)) with
  | none => intro h ν; rw [← renList_ok ρ t 0 (bvs t) ν]; exact h ν
  | some t' => intro h ν; rw [h ν, renList_ok]

/-- The elimination of a variable by rule `r`: 0 unused, 1 the delta at `j`, 2 characters,
3 the gcd of the delta at `j`. -/
def elimVarH (x : ℕ) (D : DMon) (ds ps : List Fac) (r j : ℕ) : Option Elim :=
  match r with
  | 0 => elimUnused x D ds ps
  | 1 =>
    match ds.drop j with
    | f :: post => elimDelta1 x D (ds.take j) f post ps
    | [] => none
  | 2 => elimChar x D ds ps
  | _ =>
    match ds.drop j with
    | f :: post => elimGcd1 x D (ds.take j) f post ps
    | [] => none

theorem elimVarH_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (ds ps : List Fac) (r j : ℕ)
    (e : Elim) (h : elimVarH x D ds ps r j = some e) : ElimOK ρ x D ds ps e := by
  rcases r with _ | _ | _ | r
  · exact elimUnused_ok ρ x D ds ps e h
  · simp only [elimVarH] at h
    split at h
    · next f post hd =>
      have := elimDelta1_ok ρ hρ x D _ f post ps e h
      rw [← hd, List.take_append_drop] at this
      exact this
    · simp at h
  · exact elimChar_ok ρ hρ x D ds ps e h
  · simp only [elimVarH] at h
    split at h
    · next f post hd =>
      have := elimGcd1_ok ρ hρ x D _ f post ps e h
      rw [← hd, List.take_append_drop] at this
      exact this
    · simp at h

/-- The elimination of the sum at `i` by `elimVarH r j`. -/
def elimStepH (t : STm) (i r j : ℕ) : Option STm :=
  match t.sums.drop i with
  | (x, D) :: post =>
    if ((t.sums.take i ++ post).map (·.1)).contains x then none
    else
      match elimVarH x D t.dels t.phs r j with
      | some e => some ⟨CMon.mul t.c e.k, t.sums.take i ++ post, e.ds, e.ps⟩
      | none => none
  | [] => none

theorem elimStepH_ok (ρ : Val) (hρ : ρ.OK) (t t' : STm) (i r j : ℕ)
    (h : elimStepH t i r j = some t') (ν : ℕ → ℤ) : t'.den ρ ν = t.den ρ ν := by
  unfold elimStepH at h
  split at h
  · next x D post hd =>
    split_ifs at h with hx
    split at h
    · next e hev =>
      simp only [Option.some.injEq] at h
      subst h
      have hok := elimVarH_ok ρ hρ x D t.dels t.phs r j e hev
      have hs : t.sums = t.sums.take i ++ (x, D) :: post := by rw [← hd, List.take_append_drop]
      set A := t.sums.take i with hA
      unfold STm.den
      rw [hs, sumsV_pull ρ.dv A post x D (by simpa using hx)]
      simp only [body_eq] at *
      rw [sumsV_congr _ _ _ _ hok, sumsV_mul_left, CMon.den_mul ρ hρ]
      ring
    · simp at h
  · simp at h

/-- The hinted eliminations in order, each result forced; the term at the first that fails. -/
def elimAllH : List (ℕ × ℕ × ℕ) → STm → STm
  | [], t => t
  | (i, r, j) :: hs, t =>
    match elimStepH t i r j with
    | some t' => fTm t' (elimAllH hs)
    | none => t

theorem elimAllH_ok (ρ : Val) (hρ : ρ.OK) : ∀ (hs : List (ℕ × ℕ × ℕ)) (t : STm) (ν : ℕ → ℤ),
    (elimAllH hs t).den ρ ν = t.den ρ ν := by
  intro hs
  induction hs with
  | nil => intro t ν; rfl
  | cons h hs ih =>
    obtain ⟨i, r, j⟩ := h
    intro t ν
    simp only [elimAllH]
    cases he : elimStepH t i r j with
    | none => rfl
    | some t' => dsimp only; rw [fTm_eq, ih, elimStepH_ok ρ hρ t t' i r j he]

/-- A round's hints: the pivot hint and the eliminations. -/
abbrev RHint := ℕ × List (ℕ × ℕ × ℕ)

/-- Hinted eliminations and canonicalisation, one round per hint, each result forced. -/
def normTmH (fl : List ℕ) : List RHint → STm → Option STm
  | [], t => some t
  | (c, es) :: cs, t =>
    match canonTmH c fl (fTm t (elimAllH es)) with
    | none => none
    | some t' => fTm t' (normTmH fl cs)

theorem normTmH_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) :
    ∀ (cs : List RHint) (t : STm), TmRes ρ t (normTmH fl cs t) := by
  intro cs
  induction cs with
  | nil => intro t ν; rfl
  | cons h cs ih =>
    obtain ⟨c, es⟩ := h
    intro t
    have hc := canonTmH_ok ρ hρ fl hfl c (elimAllH es t)
    have he := elimAllH_ok ρ hρ es t
    simp only [normTmH, fTm_eq]
    revert hc
    cases canonTmH c fl (elimAllH es t) with
    | none => intro hc ν; rw [← he ν, hc ν]
    | some t' =>
      intro hc
      simp only [fTm_eq]
      have := ih t'
      revert this
      cases normTmH fl cs t' with
      | none => intro h' ν; rw [← he ν, ← hc ν, h' ν]
      | some t'' => intro h' ν; rw [h' ν, hc ν, he ν]

/-- The terms normalised with their hint lists, in order, and summed into `acc`. -/
def nfGoH (fl : List ℕ) : List (List RHint) → List STm → List STm → List STm
  | _, acc, [] => acc
  | hs, acc, t :: ts =>
    match fTm t (normTmH fl (hs.headD [])) with
    | none => nfGoH fl hs.tail acc ts
    | some t' => fL fTm (tmAdd t' acc) fun acc => nfGoH fl hs.tail acc ts

theorem nfGoH_den (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (ν : ℕ → ℤ) :
    ∀ (e : List STm) (hs : List (List RHint)) (acc : List STm),
    exDen ρ ν (nfGoH fl hs acc e) = exDen ρ ν acc + exDen ρ ν e := by
  intro e
  induction e with
  | nil => intro hs acc; simp [nfGoH, exDen]
  | cons t ts ih =>
    intro hs acc
    have hn := normTmH_ok ρ hρ fl hfl (hs.headD []) t
    simp only [nfGoH, fTm_eq]
    revert hn
    cases normTmH fl (hs.headD []) t with
    | none =>
      intro hn
      dsimp only
      rw [ih]
      simp only [exDen, List.map_cons, List.sum_cons]
      rw [hn ν]; ring
    | some t' =>
      intro hn
      dsimp only
      rw [fL_eq _ (fun a k => fTm_eq a k), ih, tmAdd_ok]
      simp only [exDen, List.map_cons, List.sum_cons]
      rw [hn ν]; ring

/-- The hinted normaliser of node `i`, its hints `H i`. -/
def nfH (fl : List ℕ) (H : ℕ → List (List RHint)) (i : ℕ) (e : List STm) : List STm :=
  nfGoH fl (H i) [] e

theorem nfH_den (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (H : ℕ → List (List RHint)) (i : ℕ) (e : List STm) (ν : ℕ → ℤ) :
    exDen ρ ν (nfH fl H i e) = exDen ρ ν e := by
  unfold nfH
  rw [nfGoH_den ρ hρ fl hfl ν e (H i) []]
  simp [exDen]

end ArchytasZX.Sym
