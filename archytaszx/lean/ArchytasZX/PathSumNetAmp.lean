/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumNetW
import ArchytasZX.PathSumAmp

/-!
# Networks with amplitude atoms

A weighted Z spider whose coefficient at `v` is `C` times `ex` of its phase at `v` times the
product of the coefficients `A i` listed at `v` is the unweighted spider's fragment scaled by `C`
times atoms on its anchor digits (`za_frag`); `atabOK` checks the listed indices against the
atoms at every value. `fvalA` reads a fragment's atoms inside its sum (`fvalA_of` for atoms off
its path, `fvalA_joinAll` for fragments with `DisjA`), and `xa_frag` gives a weighted X spider (a
legless weighted Z spider through `val_zw_legless`) with atoms on its fresh digits. `NetPSDA` is
`NetPSD` with atoms per node, and `NetPSDA.eval_eq` gives a network's value as `valA` with the
summed digits and the fragments' paths held.
-/

namespace ArchytasZX
namespace PSem

open Complex Finset

theorem val_zw_at (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .ZW) (C : ℂ) (W : ℕ → ℂ)
    (hc : ∀ v, E.coef n.ph v = C * ex (E.phase n.ph v) * W v) (l0 : ℕ)
    (hl : l0 ∈ n.outs ++ n.ins) :
    ArchytasZX.val E σ n = C * W (σ l0) * ArchytasZX.val E σ (n.withKind .Z) := by
  have hm : σ l0 ∈ n.outs.map σ ++ n.ins.map σ := by
    rw [← List.map_append]; exact List.mem_map.2 ⟨l0, hl, rfl⟩
  unfold ArchytasZX.val Node.withKind
  rw [hk]
  simp only [tZw, tZ, Finset.mul_sum, hc]
  apply Finset.sum_congr rfl
  intro k _
  by_cases H : ∀ v ∈ n.outs.map σ ++ n.ins.map σ, v = k
  · rw [if_pos H, if_pos H, H _ hm]; ring
  · rw [if_neg H, if_neg H, mul_zero]

/-- The coefficient indices of the atoms whose polynomial is 1 at `τ`. -/
def atomsAt (as : Atoms) (τ : ℕ → ZMod 2) : List ℕ := (as.filter fun a => fz τ a.2 != 0).map (·.1)

theorem aval_atomsAt (A : ℕ → ℂ) (τ : ℕ → ZMod 2) :
    ∀ as : Atoms, aval A as τ = ((atomsAt as τ).map A).prod
  | [] => rfl
  | a :: as => by
    rw [aval_cons, aval_atomsAt A τ as]
    unfold atomsAt
    by_cases h : fz τ a.2 = 0
    · simp [h]
    · simp [h]

theorem aval_agree (A : ℕ → ℂ) (τ τ' : ℕ → ZMod 2) :
    ∀ as : Atoms, (∀ x ∈ avars as, τ x = τ' x) → aval A as τ = aval A as τ'
  | [] => fun _ => rfl
  | a :: as => fun h => by
    have h1 : ∀ m ∈ a.2, ∀ x ∈ m, τ x = τ' x := fun m hm x hx =>
      h x (List.mem_flatMap.2 ⟨a, List.mem_cons_self, List.mem_flatMap.2 ⟨m, hm, hx⟩⟩)
    have h2 : ∀ x ∈ avars as, τ x = τ' x := fun x hx => h x (by
      simp only [avars, List.flatMap_cons, List.mem_append] at hx ⊢; exact Or.inr hx)
    rw [aval_cons, aval_cons, fz_agree τ τ' a.2 h1, aval_agree A τ τ' as h2]

/-- Whether the atoms read only `ds` and list the indices `tbl[v]` at the digits of every
`v < 2^|ds|` on `ds`. -/
def atabOK (ds : List ℕ) (as : Atoms) (tbl : List (List ℕ)) : Bool :=
  decide (∀ x ∈ avars as, x ∈ ds) && (List.range (2 ^ ds.length)).all fun v =>
    decide (atomsAt as (setBits ds v fun _ => 0) = tbl.getD v [])

/-- A weighted Z spider whose coefficients are `C`, its phase and the coefficients listed in
`tbl`: the unweighted spider's fragment scaled by `C`, times atoms on its anchor digits. -/
theorem za_frag (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hk : n.kind = .ZW) (k : ℕ) (hd : E.dimv n.dim = 2 ^ k)
    (hlabs : ∀ l ∈ n.outs ++ n.ins, l ∈ labs) (hlen : lensOK dig k (n.outs ++ n.ins) = true)
    (as : List ℕ) (has : as.length = k) (hasl : ∀ a ∈ as, a ∈ n.outs ++ n.ins)
    (l0 : ℕ) (hl0 : l0 ∈ n.outs ++ n.ins) (hnd : (anchorD dig as).Nodup)
    (M : ℕ) (f : Frag) (C : ℂ) (A : ℕ → ℂ) (tbl : List (List ℕ)) (ats : Atoms)
    (hc : ∀ v, E.coef n.ph v = C * ex (E.phase n.ph v) * ((tbl.getD v []).map A).prod)
    (hat : atabOK (anchorD dig as) ats tbl = true)
    (hf : ArchytasZX.val E σ (n.withKind .Z) = fval M f (dbits own σ)) :
    ArchytasZX.val E σ n = fval M (f.scale C) (dbits own σ) * aval A ats (dbits own σ) := by
  set ls := n.outs ++ n.ins with hls
  have hlen' : ∀ l ∈ ls, (dig l).length = k := by
    intro l hl
    simpa [lensOK] using (List.all_eq_true.1 hlen) l hl
  have hσk : ∀ l ∈ ls, σ l < 2 ^ k := fun l hl => (hlen' l hl) ▸ hσ l (hlabs l hl)
  rw [val_zw_at E σ n hk C _ hc l0 hl0, fval_scale, ← hf]
  have hv : ArchytasZX.val E σ (n.withKind .Z) = tZ (2 ^ k) (E.phase n.ph) (σ l0 :: ls.map σ) := by
    unfold ArchytasZX.val Node.withKind
    simp only [hd, List.map_append]
    apply tZ_congr
    intro x
    simp only [List.mem_cons, ← List.map_append]
    constructor
    · exact Or.inr
    · rintro (rfl | hx)
      · exact List.mem_map.2 ⟨l0, hl0, rfl⟩
      · exact hx
  rw [hv, tZ_val _ _ _ _ (hσk l0 hl0)]
  split_ifs with hc'
  · have hal : (anchorD dig as).length = k := by simp [anchorD, has]
    have hset : ∀ x ∈ anchorD dig as, dbits own σ x = setBits (anchorD dig as) (σ l0)
        (fun _ => 0) x := by
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
      have hj' : j < k := hal ▸ hj
      have := setBits_idx (anchorD dig as) (σ l0) (fun _ => 0) hnd j hj
      simp only [List.get_eq_getElem] at this
      rw [this]
      have ha : as[j]'(has ▸ hj') ∈ ls := hasl _ (List.getElem_mem (has ▸ hj'))
      have hlj : j < (dig (as[j]'(has ▸ hj'))).length := by rw [hlen' _ ha]; exact hj'
      have hanc : (anchorD dig as)[j] = (dig (as[j]'(has ▸ hj')))[j]'hlj := by
        simp only [anchorD, List.getElem_mapIdx, List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem hlj, Option.getD_some]
      rw [hanc, dbits_get hown σ (hlabs _ ha) hlj, hc' _ (List.mem_map.2 ⟨_, ha, rfl⟩)]
    simp only [atabOK, Bool.and_eq_true, decide_eq_true_eq] at hat
    obtain ⟨hread, htab⟩ := hat
    have hag : aval A ats (dbits own σ) =
        aval A ats (setBits (anchorD dig as) (σ l0) fun _ => 0) :=
      aval_agree A _ _ ats fun x hx => hset x (hread x hx)
    have hv0 : σ l0 < 2 ^ (anchorD dig as).length := by rw [hal]; exact hσk l0 hl0
    have := List.all_eq_true.1 htab (σ l0) (List.mem_range.2 hv0)
    simp only [decide_eq_true_eq] at this
    rw [hag, aval_atomsAt, this]
    ring
  · simp

/-- An entry equal to a fragment's value, times no atoms. -/
theorem atoms_nil {x : ℂ} (A : ℕ → ℂ) (M : ℕ) (f : Frag) (τ : ℕ → ZMod 2) (h : x = fval M f τ) :
    x = fval M f τ * aval A [] τ := by
  rw [h, aval_nil, mul_one]

/-! ## Fragments with atoms -/

/-- A fragment's value with the atoms read inside its sum. -/
noncomputable def fvalA (M : ℕ) (A : ℕ → ℂ) (f : Frag) (as : Atoms) (τ : ℕ → ZMod 2) : ℂ :=
  bsum f.path (fun ρ => aval A as ρ * summand M f.c ⟨f.path, f.phase, f.cons, []⟩ ρ) τ

/-- A fragment's value times atoms that read no path variable of it. -/
theorem fvalA_of {x : ℂ} (M : ℕ) (A : ℕ → ℂ) (f : Frag) (as : Atoms) (τ : ℕ → ZMod 2)
    (hr : (f.path.all fun y => !decide (y ∈ avars as)) = true)
    (h : x = fval M f τ * aval A as τ) : x = fvalA M A f as τ := by
  have hr' : ∀ y ∈ f.path, y ∉ avars as := by
    intro y hy
    simpa using List.all_eq_true.1 hr y hy
  rw [h]
  unfold fvalA fval val
  dsimp only
  rw [bsum_mul_left _ _ f.path (fun τ' y hy b => aval_update_not A τ' y b as (hr' y hy)) τ]
  ring

theorem summand_scale (M : ℕ) (C c : ℂ) (s : Data) (τ : ℕ → ZMod 2) :
    summand M (C * c) s τ = C * summand M c s τ := by
  unfold summand
  split_ifs <;> ring

/-- `Disj` with no path variable of the first fragment read by the second's atoms. -/
def DisjA (a b : Frag × Atoms) : Prop :=
  Disj a.1 b.1 ∧ ∀ y ∈ a.1.path, y ∉ avars b.2

instance (a b : Frag × Atoms) : Decidable (DisjA a b) := by
  unfold DisjA
  infer_instance

theorem fvalA_join (M : ℕ) (A : ℕ → ℂ) (a b : Frag × Atoms) (hab : DisjA a b)
    (hba : DisjA b a) (τ : ℕ → ZMod 2) :
    fvalA M A (a.1.join b.1) (a.2 ++ b.2) τ = fvalA M A a.1 a.2 τ * fvalA M A b.1 b.2 τ := by
  set ga : (ℕ → ZMod 2) → ℂ := fun ρ =>
    aval A a.2 ρ * summand M a.1.c ⟨a.1.path, a.1.phase, a.1.cons, []⟩ ρ with hga_def
  set gb : (ℕ → ZMod 2) → ℂ := fun ρ =>
    aval A b.2 ρ * summand M b.1.c ⟨b.1.path, b.1.phase, b.1.cons, []⟩ ρ with hgb_def
  have hsplit : ∀ ρ, aval A (a.2 ++ b.2) ρ * summand M (a.1.join b.1).c
      ⟨(a.1.join b.1).path, (a.1.join b.1).phase, (a.1.join b.1).cons, []⟩ ρ = ga ρ * gb ρ := by
    intro ρ
    rw [aval_append, summand_join]
    simp only [hga_def, hgb_def]
    ring
  have hga : ∀ τ', ∀ y ∈ b.1.path, ∀ w, ga (Function.update τ' y w) = ga τ' := by
    intro τ' y hy w
    obtain ⟨_, h2, h3⟩ := hba.1 y hy
    simp only [hga_def]
    rw [aval_update_not A τ' y w a.2 (hba.2 y hy), summand_frag_inv M a.1 y w h2 h3 τ']
  have hgb : ∀ τ', ∀ y ∈ a.1.path, ∀ w,
      bsum b.1.path gb (Function.update τ' y w) = bsum b.1.path gb τ' := by
    intro τ' y hy w
    obtain ⟨h1, h2, h3⟩ := hab.1 y hy
    refine bsum_inv _ y w (fun ρ => ?_) b.1.path τ' h1
    simp only [hgb_def]
    rw [aval_update_not A ρ y w b.2 (hab.2 y hy), summand_frag_inv M b.1 y w h2 h3 ρ]
  unfold fvalA
  show bsum (a.1.path ++ b.1.path) _ τ = _
  rw [bsum_congr _ _ hsplit, bsum_append,
    bsum_congr _ _ (fun τ' => bsum_mul_left ga gb b.1.path hga τ'),
    bsum_congr _ _ (fun τ' => mul_comm (ga τ') (bsum b.1.path gb τ')),
    bsum_mul_left _ _ a.1.path hgb τ]
  ring

theorem fvalA_unit (M : ℕ) (A : ℕ → ℂ) (τ : ℕ → ZMod 2) : fvalA M A Frag.unit [] τ = 1 := by
  simp [fvalA, Frag.unit, bsum, summand, pn, ex_zero, aval]

theorem mem_avars_flatten {y : ℕ} : ∀ L : List Atoms, y ∈ avars L.flatten → ∃ as ∈ L, y ∈ avars as
  | [], h => by simp [avars] at h
  | as :: L, h => by
    simp only [List.flatten_cons, avars, List.flatMap_append, List.mem_append] at h
    rcases h with h | h
    · exact ⟨as, List.mem_cons_self, h⟩
    · obtain ⟨bs, hbs, hy⟩ := mem_avars_flatten L h
      exact ⟨bs, List.mem_cons_of_mem _ hbs, hy⟩

theorem mem_joinAll_path {y : ℕ} : ∀ fs : List Frag, y ∈ (joinAll fs).path → ∃ f ∈ fs, y ∈ f.path
  | [], h => by simp [joinAll, Frag.unit] at h
  | f :: fs, h => by
    simp only [joinAll, Frag.join, List.mem_append] at h
    rcases h with h | h
    · exact ⟨f, List.mem_cons_self, h⟩
    · obtain ⟨g, hg, hy⟩ := mem_joinAll_path fs h
      exact ⟨g, List.mem_cons_of_mem _ hg, hy⟩

theorem fvalA_joinAll {α : Type} (M : ℕ) (A : ℕ → ℂ) (τ : ℕ → ZMod 2) :
    ∀ ps : List (α × Frag × Atoms), (ps.map (·.2)).Pairwise (fun a b => DisjA a b ∧ DisjA b a) →
      fvalA M A (joinAll (ps.map (·.2.1))) (ps.map (·.2.2)).flatten τ =
        (ps.map fun p => fvalA M A p.2.1 p.2.2 τ).prod
  | [], _ => fvalA_unit M A τ
  | p :: ps, h => by
    simp only [List.map_cons, List.pairwise_cons] at h
    have hall : ∀ q ∈ ps, DisjA p.2 q.2 ∧ DisjA q.2 p.2 := fun q hq =>
      h.1 q.2 (List.mem_map.2 ⟨q, hq, rfl⟩)
    have h1 : DisjA p.2 (joinAll (ps.map (·.2.1)), (ps.map (·.2.2)).flatten) := by
      refine ⟨disj_joinAll_right p.2.1 _ fun g hg => ?_, fun y hy hy' => ?_⟩
      · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hg
        exact (hall q hq).1.1
      · obtain ⟨as, has, hya⟩ := mem_avars_flatten _ hy'
        obtain ⟨q, hq, rfl⟩ := List.mem_map.1 has
        exact (hall q hq).1.2 y hy hya
    have h2 : DisjA (joinAll (ps.map (·.2.1)), (ps.map (·.2.2)).flatten) p.2 := by
      refine ⟨disj_joinAll_left p.2.1 _ fun g hg => ?_, fun y hy hy' => ?_⟩
      · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hg
        exact (hall q hq).2.1
      · obtain ⟨g, hg, hyg⟩ := mem_joinAll_path _ hy
        obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hg
        exact (hall q hq).2.2 y hyg hy'
    have e1 : joinAll ((p :: ps).map (·.2.1)) = p.2.1.join (joinAll (ps.map (·.2.1))) := rfl
    have e2 : ((p :: ps).map (·.2.2)).flatten = p.2.2 ++ (ps.map (·.2.2)).flatten := rfl
    rw [e1, e2, fvalA_join M A p.2 _ h1 h2 τ, fvalA_joinAll M A τ ps h.2, List.map_cons,
      List.prod_cons]

/-! ## Weighted X and legless weighted Z spiders with atoms -/

theorem val_zw_legless (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .ZW) (ho : n.outs = [])
    (hi : n.ins = []) : ArchytasZX.val E σ n = ArchytasZX.val E σ (n.withKind .XW) := by
  unfold ArchytasZX.val Node.withKind
  rw [hk, ho, hi]
  simp [tZw, tXw]

theorem setBits_agree (ys : List ℕ) (j : ℕ) (τ τ' : ℕ → ZMod 2) (hnd : ys.Nodup) :
    ∀ x ∈ ys, setBits ys j τ x = setBits ys j τ' x := by
  intro x hx
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
  have h1 := setBits_idx ys j τ hnd i hi
  have h2 := setBits_idx ys j τ' hnd i hi
  simp only [List.get_eq_getElem] at h1 h2
  rw [h1, h2]

/-- A weighted X spider whose coefficients are `C`, its phase and the coefficients listed in
`tbl`: the unweighted spider's fragment scaled by `C`, with atoms on its fresh digits `ys`. -/
theorem xa_frag (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hkind : n.kind = .XW) (k : ℕ) (hd : E.dimv n.dim = 2 ^ k)
    (hlabs : ∀ l ∈ n.outs ++ n.ins, l ∈ labs) (hlen : lensOK dig k (n.outs ++ n.ins) = true)
    (ys : List ℕ) (hys : ys.length = k) (hnd : ys.Nodup)
    (hfresh : ys.all (fun y => (List.lookup y own).isNone) = true)
    (M K q : ℕ) (hK : K = 2 ^ k) (hq : M = K * q) (hM : 0 < M) (P : Poly) (ts : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / M)
    (htab : tableOK M ys P ts = true)
    (hPv : P.all (fun t => t.1.all fun x => decide (x ∈ ys)) = true)
    (C : ℂ) (A : ℕ → ℂ) (tbl : List (List ℕ)) (ats : Atoms)
    (hc : ∀ v, E.coef n.ph v = C * ex (E.phase n.ph v) * ((tbl.getD v []).map A).prod)
    (hat : atabOK ys ats tbl = true) :
    ArchytasZX.val E σ n =
      fvalA M A ((xFragD M K k dig P ys n.outs n.ins).scale C) ats (dbits own σ) := by
  subst hK
  have hv : ArchytasZX.val E σ n = tXw (2 ^ k) (E.coef n.ph) (n.outs.map σ) (n.ins.map σ) := by
    unfold ArchytasZX.val
    rw [hkind]
    simp only [hd]
  simp only [atabOK, Bool.and_eq_true, decide_eq_true_eq] at hat
  obtain ⟨hread, htab'⟩ := hat
  rw [hv]
  unfold fvalA xFragD Frag.scale
  dsimp only
  rw [← sum_setBits _ ys (dbits own σ), hys]
  unfold tXw
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' := Finset.mem_range.1 hj
  have ht := x_term_dig σ own labs dig hown hσ (E.phase n.ph) n.outs n.ins k hlabs hlen ys hys hnd
    hfresh M q hq hM P ts hph htab hPv j hj'
  have hag : aval A ats (setBits ys j (dbits own σ)) = aval A ats (setBits ys j fun _ => 0) :=
    aval_agree A _ _ ats fun x hx => setBits_agree ys j _ _ hnd x (hread x hx)
  have hw : aval A ats (setBits ys j fun _ => 0) = ((tbl.getD j []).map A).prod := by
    have := List.all_eq_true.1 htab' j (List.mem_range.2 (by rw [hys]; exact hj'))
    simp only [decide_eq_true_eq] at this
    rw [aval_atomsAt, this]
  rw [hc j, summand_scale, ← ht, hag, hw]
  ring

/-! ## Networks -/

/-- The network's path sum with atoms: the summed labels' digits and the fragments' paths held,
every node's atoms. -/
def netDataA (dig : ℕ → List ℕ) (N : Net) (fs : List Frag) (ats : List Atoms) : AData :=
  ⟨⟨[], (joinAll fs).phase, (joinAll fs).cons, []⟩, N.int.flatMap dig ++ (joinAll fs).path,
    ats.flatten⟩

/-- `NetPSD` with atoms: each node's entry is its fragment's value with its atoms read inside the
fragment's sum. -/
structure NetPSDA (E : Env) (N : Net) (M : ℕ) (A : ℕ → ℂ) (own : Own) (labs : List ℕ)
    (dig : ℕ → List ℕ) (ps : List (Node × Frag × Atoms)) : Prop where
  ownok : ownOK own labs dig = true
  nodes : N.nodes = ps.map Prod.fst
  node : ∀ p ∈ ps, ∀ σ, (∀ l ∈ labs, σ l < 2 ^ (dig l).length) →
    ArchytasZX.val E σ p.1 = fvalA M A p.2.1 p.2.2 (dbits own σ)
  disj : (ps.map (·.2)).Pairwise fun a b => DisjA a b ∧ DisjA b a
  dims : ∀ l ∈ labs, labDim E l = 2 ^ (dig l).length
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd
  sub : ∀ l ∈ N.int ++ N.bnd, l ∈ labs

theorem NetPSDA.eval_eq {E : Env} {N : Net} {M : ℕ} {A : ℕ → ℂ} {own : Own} {labs : List ℕ}
    {dig : ℕ → List ℕ} {ps : List (Node × Frag × Atoms)} (h : NetPSDA E N M A own labs dig ps)
    (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = valA M (joinAll (ps.map (·.2.1))).c A
      (netDataA dig N (ps.map (·.2.1)) (ps.map (·.2.2))) (dbits own (clamp N.bnd σ)) := by
  have he : eval E N σ = eval E N (clamp N.bnd σ) := by
    unfold eval
    apply sumOver_ext (labDim E) N.int (labelsOf N.nodes) _ (prodVal_reads E N.nodes)
    intro x hx hn
    rcases h.cover x hx with hi | hb
    · exact absurd hi hn
    · simp [clamp, hb]
  rw [he]
  unfold eval
  have hint : ∀ l ∈ N.int, l ∈ labs := fun l hl => h.sub l (List.mem_append_left _ hl)
  set G : (ℕ → ZMod 2) → ℂ := fun τ =>
    fvalA M A (joinAll (ps.map (·.2.1))) (ps.map (·.2.2)).flatten τ with hG
  have hcongr : sumOver (labDim E) N.int (prodVal E N.nodes) (clamp N.bnd σ) =
      sumOver (labDim E) N.int (fun σ' => G (dbits own σ')) (clamp N.bnd σ) := by
    apply sumOver_congr
    intro τ hτ hr
    have hrange : ∀ l ∈ labs, τ l < 2 ^ (dig l).length := by
      intro l hl
      by_cases hli : l ∈ N.int
      · have := hr l hli
        rwa [h.dims l hl] at this
      · rw [hτ l hli]
        unfold clamp
        split_ifs with hlb
        · have := hσ l hlb
          rwa [h.dims l hl] at this
        · positivity
    simp only [hG]
    rw [fvalA_joinAll M A _ ps h.disj, prodVal, h.nodes, List.map_map]
    congr 1
    apply List.map_congr_left
    intro p hp
    exact h.node p hp τ hrange
  rw [hcongr, sumOver_dbits (labDim E) h.ownok G N.int hint
    (fun l hl => h.dims l (hint l hl))]
  unfold valA netDataA
  dsimp only
  rw [bsum_append]
  rfl

/-- `equal_psd2` with atoms. -/
theorem equal_psda2 {E : Env} {N₁ N₂ : Net} {M : ℕ} {A : ℕ → ℂ} {own₁ own₂ : Own}
    {labs₁ labs₂ : List ℕ} {dig₁ dig₂ : ℕ → List ℕ} {ps₁ ps₂ : List (Node × Frag × Atoms)}
    (s₁ s₂ : ℂ) (h₁ : NetPSDA E N₁ M A own₁ labs₁ dig₁ ps₁)
    (h₂ : NetPSDA E N₂ M A own₂ labs₂ dig₂ ps₂) (hb : N₁.bnd = N₂.bnd)
    (hv : ∀ σ, s₁ * valA M (joinAll (ps₁.map (·.2.1))).c A
        (netDataA dig₁ N₁ (ps₁.map (·.2.1)) (ps₁.map (·.2.2))) (dbits own₁ (clamp N₁.bnd σ)) =
      s₂ * valA M (joinAll (ps₂.map (·.2.1))).c A
        (netDataA dig₂ N₂ (ps₂.map (·.2.1)) (ps₂.map (·.2.2))) (dbits own₂ (clamp N₂.bnd σ))) :
    Equal E s₁ N₁ s₂ N₂ := by
  refine ⟨hb ▸ List.Perm.refl _, fun σ hσ => ?_⟩
  have hσ₂ : InRange E N₂.bnd σ := hb ▸ hσ
  rw [h₁.eval_eq σ hσ, h₂.eval_eq σ hσ₂]
  exact hv σ

theorem valA_smul (M : ℕ) (a c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2) :
    valA M (a * c) A t σ = a * valA M c A t σ := by
  unfold valA
  rw [← bsum_smul]
  apply bsum_congr
  intro τ
  rw [val_smul]
  ring

/-- A sum with no held or path variable, at two assignments agreeing on `vs`. -/
theorem valA_agree (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (vs : List ℕ) (τ τ' : ℕ → ZMod 2)
    (hh : t.held = []) (hp : t.s.path = []) (hv : varsIn vs t.s = true)
    (ha : (avars t.atoms).all (fun x => decide (x ∈ vs)) = true) (h : ∀ x ∈ vs, τ x = τ' x) :
    valA M c A t τ = valA M c A t τ' := by
  unfold valA
  rw [hh]
  simp only [bsum]
  rw [val_agree M c t.s vs τ τ' hp hv h, aval_agree A τ τ' t.atoms fun x hx =>
    h x (by simpa using List.all_eq_true.1 ha x hx)]

end PSem
end ArchytasZX
