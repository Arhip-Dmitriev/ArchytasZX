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
atoms at every value. `NetPSDA` is `NetPSD` with atoms per node, and `NetPSDA.eval_eq` gives a
network's value as `valA` with the summed digits held and the fragments' paths as the path.
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

/-! ## Networks -/

/-- The network's path sum with atoms: the summed labels' digits held, the fragments' paths as
the path, every node's atoms. -/
def netDataA (dig : ℕ → List ℕ) (N : Net) (fs : List Frag) (ats : List Atoms) : AData :=
  ⟨⟨(joinAll fs).path, (joinAll fs).phase, (joinAll fs).cons, []⟩, N.int.flatMap dig,
    ats.flatten⟩

/-- `NetPSD` with atoms: each node's entry is its fragment's value times its atoms. -/
structure NetPSDA (E : Env) (N : Net) (M : ℕ) (A : ℕ → ℂ) (own : Own) (labs : List ℕ)
    (dig : ℕ → List ℕ) (ps : List (Node × Frag × Atoms)) : Prop where
  ownok : ownOK own labs dig = true
  nodes : N.nodes = ps.map Prod.fst
  node : ∀ p ∈ ps, ∀ σ, (∀ l ∈ labs, σ l < 2 ^ (dig l).length) →
    ArchytasZX.val E σ p.1 = fval M p.2.1 (dbits own σ) * aval A p.2.2 (dbits own σ)
  disj : (ps.map (·.2.1)).Pairwise fun a b => Disj a b ∧ Disj b a
  dims : ∀ l ∈ labs, labDim E l = 2 ^ (dig l).length
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd
  sub : ∀ l ∈ N.int ++ N.bnd, l ∈ labs

theorem aval_flatten (A : ℕ → ℂ) (τ : ℕ → ZMod 2) :
    ∀ L : List Atoms, aval A L.flatten τ = (L.map fun as => aval A as τ).prod
  | [] => rfl
  | as :: L => by
    rw [List.flatten_cons, aval_append, aval_flatten A τ L, List.map_cons, List.prod_cons]

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
  set F := joinAll (ps.map (·.2.1)) with hF
  set G : (ℕ → ZMod 2) → ℂ := fun τ => aval A (ps.map (·.2.2)).flatten τ * fval M F τ with hG
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
    simp only [hG, hF]
    rw [fval_joinAll M _ _ h.disj, aval_flatten, prodVal, h.nodes]
    simp only [List.map_map]
    rw [← List.prod_map_mul]
    congr 1
    apply List.map_congr_left
    intro p hp
    simp only [Function.comp_apply]
    rw [h.node p hp τ hrange]
    ring
  rw [hcongr, sumOver_dbits (labDim E) h.ownok G N.int hint
    (fun l hl => h.dims l (hint l hl))]
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
