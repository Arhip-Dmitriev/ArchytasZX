/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumMixed
import ArchytasZX.PathSumNetAmp
import ArchytasZX.PathSumQuditNet
import ArchytasZX.PathSumQuditDigits

/-!
# Labels of dimension 2, 3 and 6 as digit variables

A mixed label `(l, k, b, t)` is the label `l` of dimension `k` (2, 3 or 6) with the F_2
variable `b` (dimensions 2 and 6) and the Z_3 variable `t` (dimensions 3 and 6); a value `j` on 6
is the pair `(j / 2, j % 2)` of `sum_six`. `mbits` and `mtrits` read the digit variables from
label values, `mOK` checks the labels and variables distinct, and `sumOver_mixed` is the sum over
labels as the sum over their F_2 variables of the sum over their Z_3 variables.
-/

namespace ArchytasZX
namespace MSem

open Complex Finset PSem

/-- A label with its dimension, F_2 variable and Z_3 variable. -/
abbrev MLab := ℕ × ℕ × ℕ × ℕ

/-- The F_2 digit of a value. -/
def mbit (j : ℕ) : ZMod 2 := ((j % 2 : ℕ) : ZMod 2)

/-- The Z_3 digit of a value on dimension `k`. -/
def mtrit (k j : ℕ) : ZMod 3 := if k = 6 then ((j / 2 : ℕ) : ZMod 3) else ((j : ℕ) : ZMod 3)

/-- The F_2 variables of the labels of dimension 2 and 6. -/
def bvars (ms : List MLab) : List ℕ :=
  ms.filterMap fun m => if m.2.1 = 3 then none else some m.2.2.1

/-- The Z_3 variables of the labels of dimension 3 and 6. -/
def tvars (ms : List MLab) : List ℕ :=
  ms.filterMap fun m => if m.2.1 = 2 then none else some m.2.2.2

/-- The F_2 assignment read from label values. -/
def mbits : List MLab → (ℕ → ℕ) → ℕ → ZMod 2
  | [], _ => fun _ => 0
  | m :: ms, σ =>
    if m.2.1 = 3 then mbits ms σ else Function.update (mbits ms σ) m.2.2.1 (mbit (σ m.1))

/-- The Z_3 assignment read from label values. -/
def mtrits : List MLab → (ℕ → ℕ) → ℕ → ZMod 3
  | [], _ => fun _ => 0
  | m :: ms, σ =>
    if m.2.1 = 2 then mtrits ms σ
    else Function.update (mtrits ms σ) m.2.2.2 (mtrit m.2.1 (σ m.1))

/-- Distinct labels, distinct F_2 variables, distinct Z_3 variables, dimensions 2, 3, 6. -/
def mOK (ms : List MLab) : Bool :=
  decide ((ms.map (·.1)).Nodup) && decide ((bvars ms).Nodup) && decide ((tvars ms).Nodup) &&
    ms.all fun m => decide (m.2.1 = 2 ∨ m.2.1 = 3 ∨ m.2.1 = 6)

theorem mbits_agree (σ σ' : ℕ → ℕ) :
    ∀ ms : List MLab, (∀ m ∈ ms, σ m.1 = σ' m.1) → mbits ms σ = mbits ms σ'
  | [], _ => rfl
  | m :: ms, h => by
    have ih := mbits_agree σ σ' ms fun m' hm => h m' (List.mem_cons_of_mem _ hm)
    simp only [mbits, ih, h m List.mem_cons_self]

theorem mtrits_agree (σ σ' : ℕ → ℕ) :
    ∀ ms : List MLab, (∀ m ∈ ms, σ m.1 = σ' m.1) → mtrits ms σ = mtrits ms σ'
  | [], _ => rfl
  | m :: ms, h => by
    have ih := mtrits_agree σ σ' ms fun m' hm => h m' (List.mem_cons_of_mem _ hm)
    simp only [mtrits, ih, h m List.mem_cons_self]

theorem mem_bvars {m : MLab} {ms : List MLab} (hm : m ∈ ms) (hk : m.2.1 ≠ 3) :
    m.2.2.1 ∈ bvars ms :=
  List.mem_filterMap.2 ⟨m, hm, by simp [hk]⟩

theorem mem_tvars {m : MLab} {ms : List MLab} (hm : m ∈ ms) (hk : m.2.1 ≠ 2) :
    m.2.2.2 ∈ tvars ms :=
  List.mem_filterMap.2 ⟨m, hm, by simp [hk]⟩

/-- Setting one label's value updates its F_2 variable. -/
theorem mbits_update (σ : ℕ → ℕ) (m : MLab) (j : ℕ) :
    ∀ ms : List MLab, m ∈ ms → (ms.map (·.1)).Nodup → (bvars ms).Nodup →
      mbits ms (Function.update σ m.1 j) =
        if m.2.1 = 3 then mbits ms σ else Function.update (mbits ms σ) m.2.2.1 (mbit j)
  | [], hm, _, _ => absurd hm List.not_mem_nil
  | m' :: ms, hm, hl, hb => by
    simp only [List.map_cons, List.nodup_cons] at hl
    rcases List.mem_cons.1 hm with rfl | hm'
    · have hout : ∀ m'' ∈ ms, Function.update σ m.1 j m''.1 = σ m''.1 := by
        intro m'' h''
        refine Function.update_of_ne (fun e => hl.1 ?_) _ _
        exact e ▸ List.mem_map.2 ⟨m'', h'', rfl⟩
      simp only [mbits, mbits_agree _ σ ms hout, Function.update_self]
      split_ifs <;> simp
    · have hne : m'.1 ≠ m.1 := fun e => hl.1 (e ▸ List.mem_map.2 ⟨m, hm', rfl⟩)
      have hb' : (bvars ms).Nodup := by
        unfold bvars at hb ⊢
        rw [List.filterMap_cons] at hb
        split_ifs at hb
        · exact hb
        · exact (List.nodup_cons.1 hb).2
      have ih := mbits_update σ m j ms hm' hl.2 hb'
      simp only [mbits, Function.update_of_ne hne, ih]
      by_cases h3 : m.2.1 = 3
      · simp only [h3, ↓reduceIte]
      · simp only [h3, ↓reduceIte]
        by_cases h3' : m'.2.1 = 3
        · simp only [h3', ↓reduceIte]
        · simp only [h3', ↓reduceIte]
          have hv : m'.2.2.1 ≠ m.2.2.1 := by
            intro e
            unfold bvars at hb
            rw [List.filterMap_cons, if_neg h3'] at hb
            exact (List.nodup_cons.1 hb).1 (e ▸ mem_bvars hm' h3)
          exact (Function.update_comm hv _ _ _).symm

/-- Setting one label's value updates its Z_3 variable. -/
theorem mtrits_update (σ : ℕ → ℕ) (m : MLab) (j : ℕ) :
    ∀ ms : List MLab, m ∈ ms → (ms.map (·.1)).Nodup → (tvars ms).Nodup →
      mtrits ms (Function.update σ m.1 j) =
        if m.2.1 = 2 then mtrits ms σ
        else Function.update (mtrits ms σ) m.2.2.2 (mtrit m.2.1 j)
  | [], hm, _, _ => absurd hm List.not_mem_nil
  | m' :: ms, hm, hl, hb => by
    simp only [List.map_cons, List.nodup_cons] at hl
    rcases List.mem_cons.1 hm with rfl | hm'
    · have hout : ∀ m'' ∈ ms, Function.update σ m.1 j m''.1 = σ m''.1 := by
        intro m'' h''
        refine Function.update_of_ne (fun e => hl.1 ?_) _ _
        exact e ▸ List.mem_map.2 ⟨m'', h'', rfl⟩
      simp only [mtrits, mtrits_agree _ σ ms hout, Function.update_self]
      split_ifs <;> simp
    · have hne : m'.1 ≠ m.1 := fun e => hl.1 (e ▸ List.mem_map.2 ⟨m, hm', rfl⟩)
      have hb' : (tvars ms).Nodup := by
        unfold tvars at hb ⊢
        rw [List.filterMap_cons] at hb
        split_ifs at hb
        · exact hb
        · exact (List.nodup_cons.1 hb).2
      have ih := mtrits_update σ m j ms hm' hl.2 hb'
      simp only [mtrits, Function.update_of_ne hne, ih]
      by_cases h2 : m.2.1 = 2
      · simp only [h2, ↓reduceIte]
      · simp only [h2, ↓reduceIte]
        by_cases h2' : m'.2.1 = 2
        · simp only [h2', ↓reduceIte]
        · simp only [h2', ↓reduceIte]
          have hv : m'.2.2.2 ≠ m.2.2.2 := by
            intro e
            unfold tvars at hb
            rw [List.filterMap_cons, if_neg h2'] at hb
            exact (List.nodup_cons.1 hb).1 (e ▸ mem_tvars hm' h2)
          exact (Function.update_comm hv _ _ _).symm

theorem bsum_sum3 (H : ZMod 3 → (ℕ → ZMod 2) → ℂ) :
    ∀ (ys : List ℕ) (τ : ℕ → ZMod 2),
      bsum ys (fun τ' => ∑ s : ZMod 3, H s τ') τ = ∑ s : ZMod 3, bsum ys (H s) τ
  | [], _ => rfl
  | y :: ys, τ => by
    simp only [bsum, bsum_sum3 H ys, ← Finset.sum_add_distrib]

theorem sum_range_three (G : ℕ → ℂ) : ∑ j ∈ range 3, G j = ∑ s : ZMod 3, G s.val := by
  show _ = ∑ s : Fin 3, G (s : ℕ)
  rw [Fin.sum_univ_eq_sum_range]

theorem sum_two (G : ZMod 2 → ℂ) : ∑ f : ZMod 2, G f = G 0 + G 1 := Fin.sum_univ_two G

/-- The sum over the labels `L` of a function of the digit variables: the sum over their F_2
variables of the sum over their Z_3 variables. -/
theorem sumOver_mixed (D : ℕ → ℕ) (ms : List MLab) (hw : mOK ms = true)
    (g : (ℕ → ZMod 2) → (ℕ → ZMod 3) → ℂ) :
    ∀ L : List MLab, (∀ m ∈ L, m ∈ ms) → (∀ m ∈ L, D m.1 = m.2.1) → ∀ σ,
      sumOver D (L.map (·.1)) (fun σ' => g (mbits ms σ') (mtrits ms σ')) σ =
        bsum (bvars L) (fun τ => QSem.qsum (tvars L) (fun ρ => g τ ρ) (mtrits ms σ))
          (mbits ms σ)
  | [], _, _, σ => rfl
  | m :: L, hL, hD, σ => by
    have hw' := hw
    simp only [mOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hw'
    obtain ⟨⟨⟨hl, hb⟩, ht⟩, hk⟩ := hw'
    have hm : m ∈ ms := hL m List.mem_cons_self
    have ih := sumOver_mixed D ms hw g L
      (fun x hx => hL x (List.mem_cons_of_mem _ hx)) (fun x hx => hD x (List.mem_cons_of_mem _ hx))
    have hbu := fun j => mbits_update σ m j ms hm hl hb
    have htu := fun j => mtrits_update σ m j ms hm hl ht
    have hDm := hD m List.mem_cons_self
    have hkm := hk m hm
    obtain ⟨l, k, b, t⟩ := m
    simp only at hbu htu hDm hkm
    simp only [List.map_cons, sumOver, ih, hbu, htu, hDm, bvars, tvars, List.filterMap_cons]
    rcases hkm with rfl | rfl | rfl
    · simp only [show (2 : ℕ) ≠ 3 by decide, ↓reduceIte]
      rw [Finset.sum_range_succ, Finset.sum_range_one]
      simp [bsum, mbit]
    · simp only [show (3 : ℕ) ≠ 2 by decide, ↓reduceIte]
      rw [sum_range_three]
      simp only [QSem.qsum]
      rw [← bsum_sum3]
      apply bsum_congr
      intro τ
      apply Finset.sum_congr rfl
      intro s _
      simp [mtrit, ZMod.natCast_zmod_val]
    · simp only [show (6 : ℕ) ≠ 2 by decide, show (6 : ℕ) ≠ 3 by decide, ↓reduceIte]
      rw [sum_six]
      have hd : ∀ (s : ZMod 3) (f : ZMod 2), mbit (2 * s.val + f.val) = f ∧
          mtrit 6 (2 * s.val + f.val) = s := by
        intro s f
        have hf := ZMod.val_lt f
        constructor
        · unfold mbit
          rw [show (2 * s.val + f.val) % 2 = f.val by omega, ZMod.natCast_zmod_val]
        · unfold mtrit
          rw [if_pos rfl, show (2 * s.val + f.val) / 2 = s.val by omega, ZMod.natCast_zmod_val]
      simp only [hd]
      simp only [bsum, QSem.qsum]
      rw [Finset.sum_comm, sum_two, ← bsum_sum3, ← bsum_sum3]


/-! ## Networks on dimensions 2, 3 and 6 -/

/-- Every label of `ms` below its dimension. -/
def MIn (ms : List MLab) (σ : ℕ → ℕ) : Prop := ∀ m ∈ ms, σ m.1 < m.2.1

/-- A network on dimensions 2, 3, 6: each node's entry is a Z_3 fragment at the Z_3 digits
times an F_2 fragment with atoms at the F_2 digits with the indicator entries `es` set. -/
structure NetPSM (E : Env) (N : Net) (M : ℕ) (A : ℕ → ℂ) (ms mi : List MLab) (es : List Ent)
    (ps : List ((Node × QSem.QFrag) × Frag × Atoms)) : Prop where
  ok : mOK ms = true
  nodes : N.nodes = ps.map (·.1.1)
  node : ∀ p ∈ ps, ∀ σ, MIn ms σ → ArchytasZX.val E σ p.1.1 =
    QSem.fval 3 p.1.2 (mtrits ms σ) * fvalA M A p.2.1 p.2.2 (liftA es (mtrits ms σ) (mbits ms σ))
  disj3 : (ps.map (·.1.2)).Pairwise fun a b => QSem.QDisj a b ∧ QSem.QDisj b a
  disj2 : (ps.map (·.2)).Pairwise fun a b => DisjA a b ∧ DisjA b a
  dims : ∀ m ∈ ms, labDim E m.1 = m.2.1
  sub : ∀ m ∈ mi, m ∈ ms
  int : N.int = mi.map (·.1)
  bnd : ∀ l ∈ N.bnd, ∃ m ∈ ms, m.1 = l
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd

theorem NetPSM.eval_eq {E : Env} {N : Net} {M : ℕ} {A : ℕ → ℂ} {ms mi : List MLab}
    {es : List Ent} {ps : List ((Node × QSem.QFrag) × Frag × Atoms)}
    (h : NetPSM E N M A ms mi es ps) (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = bsum (bvars mi) (fun τ => QSem.qsum (tvars mi) (fun ρ =>
      QSem.fval 3 (QSem.qjoinAll (ps.map (·.1.2))) ρ *
        fvalA M A (joinAll (ps.map (·.2.1))) (ps.map (·.2.2)).flatten (liftA es ρ τ))
      (mtrits ms (clamp N.bnd σ))) (mbits ms (clamp N.bnd σ)) := by
  have he : eval E N σ = eval E N (clamp N.bnd σ) := by
    unfold eval
    apply sumOver_ext (labDim E) N.int (labelsOf N.nodes) _ (prodVal_reads E N.nodes)
    intro x hx hn
    rcases h.cover x hx with hi | hb
    · exact absurd hi hn
    · simp [clamp, hb]
  rw [he]
  unfold eval
  have hok := h.ok
  simp only [mOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hok
  set g : (ℕ → ZMod 2) → (ℕ → ZMod 3) → ℂ := fun τ ρ =>
    QSem.fval 3 (QSem.qjoinAll (ps.map (·.1.2))) ρ *
      fvalA M A (joinAll (ps.map (·.2.1))) (ps.map (·.2.2)).flatten (liftA es ρ τ) with hg
  have hcongr : sumOver (labDim E) N.int (prodVal E N.nodes) (clamp N.bnd σ) =
      sumOver (labDim E) N.int (fun σ' => g (mbits ms σ') (mtrits ms σ')) (clamp N.bnd σ) := by
    apply sumOver_congr
    intro τ hτ hr
    have hin : MIn ms τ := by
      intro m hm
      by_cases hli : m.1 ∈ N.int
      · have := hr m.1 hli
        rwa [h.dims m hm] at this
      · rw [hτ m.1 hli]
        unfold clamp
        split_ifs with hlb
        · have := hσ m.1 hlb
          rwa [h.dims m hm] at this
        · rcases hok.2 m hm with e | e | e <;> omega
    simp only [hg]
    rw [QSem.fval_joinAll _ _ h.disj3, fvalA_joinAll M A _ ps h.disj2, prodVal, h.nodes,
      List.map_map, List.map_map, ← List.prod_map_mul]
    congr 1
    apply List.map_congr_left
    intro p hp
    exact h.node p hp τ hin
  rw [hcongr, h.int]
  exact sumOver_mixed (labDim E) ms h.ok g mi h.sub (fun m hm => h.dims m (h.sub m hm)) _


theorem bsum_qsum (F : (ℕ → ZMod 2) → (ℕ → ZMod 3) → ℂ) (B : List ℕ) (τ₀ : ℕ → ZMod 2) :
    ∀ (ys : List ℕ) (ρ : ℕ → ZMod 3),
      bsum B (fun τ => QSem.qsum ys (F τ) ρ) τ₀ = QSem.qsum ys (fun ρ' => bsum B (fun τ => F τ ρ') τ₀) ρ
  | [], _ => rfl
  | y :: ys, ρ => by
    simp only [QSem.qsum]
    rw [bsum_sum3 (fun s τ => QSem.qsum ys (F τ) (Function.update ρ y s))]
    simp only [bsum_qsum F B τ₀ ys]

theorem liftA_update (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) (b : ℕ) (w : ZMod 2) :
    ∀ es : List Ent, b ∉ es.map (·.1) →
      liftA es ρ (Function.update τ b w) = Function.update (liftA es ρ τ) b w
  | [], _ => rfl
  | e :: es, h => by
    simp only [List.map_cons, List.mem_cons, not_or] at h
    rw [liftA, liftA, liftA_update ρ τ b w es h.2]
    exact Function.update_comm h.1 _ _ _

theorem bsum_liftA (G : (ℕ → ZMod 2) → ℂ) (ρ : ℕ → ZMod 3) (es : List Ent) :
    ∀ (B : List ℕ) (τ : ℕ → ZMod 2), (∀ b ∈ B, b ∉ es.map (·.1)) →
      bsum B (fun τ' => G (liftA es ρ τ')) τ = bsum B G (liftA es ρ τ)
  | [], _, _ => rfl
  | b :: B, τ, h => by
    have hB : ∀ b' ∈ B, b' ∉ es.map (·.1) := fun b' hb' => h b' (List.mem_cons_of_mem _ hb')
    simp only [bsum, bsum_liftA G ρ es B _ hB, liftA_update ρ τ b _ es (h b List.mem_cons_self)]

theorem fvalA_valA (M : ℕ) (A : ℕ → ℂ) (F : Frag) (ats : Atoms) (τ : ℕ → ZMod 2) :
    fvalA M A F ats τ = valA M F.c A ⟨⟨[], F.phase, F.cons, []⟩, F.path, ats⟩ τ := rfl

/-- The sum of `NetPSM.eval_eq` as a mixed path sum: the summed F_2 variables and the F_2
fragments' paths held, the summed Z_3 variables and the Z_3 fragments' paths as path. -/
theorem sum_mval (M : ℕ) (A : ℕ → ℂ) (J : QSem.QFrag) (F : Frag) (ats : Atoms) (es : List Ent)
    (B T : List ℕ) (hB : ∀ b ∈ B, b ∉ es.map (·.1)) (hJ : ∀ e ∈ es, e.2.1 ∉ J.path)
    (τ₀ : ℕ → ZMod 2) (ρ₀ : ℕ → ZMod 3) :
    bsum B (fun τ => QSem.qsum T (fun ρ => QSem.fval 3 J ρ * fvalA M A F ats (liftA es ρ τ)) ρ₀)
        τ₀ =
      J.c * mval M F.c A ⟨⟨[], F.phase, F.cons, []⟩, B ++ F.path, ats⟩
        ⟨T ++ J.path, J.phase, J.cons, []⟩ es τ₀ ρ₀ := by
  set q : AData := ⟨⟨[], F.phase, F.cons, []⟩, B ++ F.path, ats⟩ with hq
  set t : QSem.Data := ⟨T ++ J.path, J.phase, J.cons, []⟩ with ht
  rw [bsum_qsum]
  have hin : ∀ ρ : ℕ → ZMod 3, bsum B (fun τ => QSem.fval 3 J ρ * fvalA M A F ats (liftA es ρ τ))
      τ₀ = QSem.fval 3 J ρ * valA M F.c A q (liftA es ρ τ₀) := by
    intro ρ
    rw [bsum_smul, bsum_liftA (fvalA M A F ats) ρ es B τ₀ hB]
    congr 1
    simp only [hq, valA, bsum_append]
    rfl
  simp only [hin]
  have hV : ∀ ρ, ∀ y ∈ J.path, ∀ w, valA M F.c A q (liftA es (Function.update ρ y w) τ₀) =
      valA M F.c A q (liftA es ρ τ₀) := by
    intro ρ y hy w
    rw [liftA_rho _ ρ τ₀ es fun e he => Function.update_of_ne (fun h => hJ e he (by rw [h]; exact hy)) _ _]
  have hf : ∀ ρ, QSem.fval 3 J ρ * valA M F.c A q (liftA es ρ τ₀) =
      QSem.qsum J.path (fun ρ' => J.c * (wt3 t ρ' * valA M F.c A q (liftA es ρ' τ₀))) ρ := by
    intro ρ
    rw [mul_comm, QSem.fval, QSem.val,
      ← QSem.qsum_mul_left (fun ρ' => valA M F.c A q (liftA es ρ' τ₀)) _ J.path (hV) ρ]
    apply QSem.qsum_congr
    intro ρ'
    simp only [QSem.summand, wt3, ht, List.append_nil]
    split_ifs <;> ring
  simp only [hf]
  rw [← QSem.qsum_append, qsum_const_mul]
  rfl


/-- `NetPSM.eval_eq` as `J.c` times a mixed path sum. -/
theorem NetPSM.eval_mval {E : Env} {N : Net} {M : ℕ} {A : ℕ → ℂ} {ms mi : List MLab}
    {es : List Ent} {ps : List ((Node × QSem.QFrag) × Frag × Atoms)}
    (h : NetPSM E N M A ms mi es ps) (hB : ∀ b ∈ bvars mi, b ∉ es.map (·.1))
    (hJ : ∀ e ∈ es, e.2.1 ∉ (QSem.qjoinAll (ps.map (·.1.2))).path)
    (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = (QSem.qjoinAll (ps.map (·.1.2))).c *
      mval M (joinAll (ps.map (·.2.1))).c A
        ⟨⟨[], (joinAll (ps.map (·.2.1))).phase, (joinAll (ps.map (·.2.1))).cons, []⟩,
          bvars mi ++ (joinAll (ps.map (·.2.1))).path, (ps.map (·.2.2)).flatten⟩
        ⟨tvars mi ++ (QSem.qjoinAll (ps.map (·.1.2))).path, (QSem.qjoinAll (ps.map (·.1.2))).phase,
          (QSem.qjoinAll (ps.map (·.1.2))).cons, []⟩
        es (mbits ms (clamp N.bnd σ)) (mtrits ms (clamp N.bnd σ)) := by
  rw [h.eval_eq σ hσ]
  exact sum_mval M A _ _ _ es _ _ hB hJ _ _

theorem mbits_get (σ : ℕ → ℕ) {m : MLab} (hk : m.2.1 ≠ 3) :
    ∀ ms : List MLab, m ∈ ms → (bvars ms).Nodup → mbits ms σ m.2.2.1 = mbit (σ m.1)
  | [], hm, _ => absurd hm List.not_mem_nil
  | m' :: ms, hm, hb => by
    unfold bvars at hb
    rw [List.filterMap_cons] at hb
    rcases List.mem_cons.1 hm with rfl | hm'
    · simp only [mbits, hk, ↓reduceIte, Function.update_self]
    · have hb' : (bvars ms).Nodup := by
        split_ifs at hb
        · exact hb
        · exact (List.nodup_cons.1 hb).2
      simp only [mbits]
      split_ifs with h3
      · exact mbits_get σ hk ms hm' hb'
      · rw [if_neg h3] at hb
        have hne : m.2.2.1 ≠ m'.2.2.1 := fun e =>
          (List.nodup_cons.1 hb).1 (e ▸ mem_bvars hm' hk)
        rw [Function.update_of_ne hne, mbits_get σ hk ms hm' hb']

theorem mtrits_get (σ : ℕ → ℕ) {m : MLab} (hk : m.2.1 ≠ 2) :
    ∀ ms : List MLab, m ∈ ms → (tvars ms).Nodup → mtrits ms σ m.2.2.2 = mtrit m.2.1 (σ m.1)
  | [], hm, _ => absurd hm List.not_mem_nil
  | m' :: ms, hm, hb => by
    unfold tvars at hb
    rw [List.filterMap_cons] at hb
    rcases List.mem_cons.1 hm with rfl | hm'
    · simp only [mtrits, hk, ↓reduceIte, Function.update_self]
    · have hb' : (tvars ms).Nodup := by
        split_ifs at hb
        · exact hb
        · exact (List.nodup_cons.1 hb).2
      simp only [mtrits]
      split_ifs with h2
      · exact mtrits_get σ hk ms hm' hb'
      · rw [if_neg h2] at hb
        have hne : m.2.2.2 ≠ m'.2.2.2 := fun e =>
          (List.nodup_cons.1 hb).1 (e ▸ mem_tvars hm' hk)
        rw [Function.update_of_ne hne, mtrits_get σ hk ms hm' hb']

theorem liftA_out (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) (x : ℕ) :
    ∀ es : List Ent, x ∉ es.map (·.1) → liftA es ρ τ x = τ x
  | [], _ => rfl
  | e :: es, h => by
    simp only [List.map_cons, List.mem_cons, not_or] at h
    rw [liftA, Function.update_of_ne h.1, liftA_out ρ τ x es h.2]

theorem bsum_agree (g : (ℕ → ZMod 2) → ℂ) (S : List ℕ)
    (hg : ∀ τ τ', (∀ x ∈ S, τ x = τ' x) → g τ = g τ') :
    ∀ (ys : List ℕ) (τ τ' : ℕ → ZMod 2), (∀ x ∈ S, x ∉ ys → τ x = τ' x) →
      bsum ys g τ = bsum ys g τ'
  | [], τ, τ', h => hg τ τ' fun x hx => h x hx List.not_mem_nil
  | y :: ys, τ, τ', h => by
    have hu : ∀ w, ∀ x ∈ S, x ∉ ys →
        Function.update τ y w x = Function.update τ' y w x := by
      intro w x hx hn
      by_cases hxy : x = y
      · subst hxy; simp
      · rw [Function.update_of_ne hxy, Function.update_of_ne hxy]
        exact h x hx (by simp [hxy, hn])
    simp only [bsum]
    rw [bsum_agree g S hg ys _ _ (hu 0), bsum_agree g S hg ys _ _ (hu 1)]

/-- An F_2 fragment with atoms at assignments agreeing on the variables `vs` off its path. -/
theorem fvalA_agree (M : ℕ) (A : ℕ → ℂ) (f : Frag) (ats : Atoms) (vs : List ℕ)
    (hv : varsIn vs ⟨[], f.phase, f.cons, []⟩ = true)
    (ha : (avars ats).all (fun x => decide (x ∈ vs)) = true) (τ τ' : ℕ → ZMod 2)
    (h : ∀ x ∈ vs, x ∉ f.path → τ x = τ' x) : fvalA M A f ats τ = fvalA M A f ats τ' := by
  unfold fvalA
  apply bsum_agree _ vs _ f.path τ τ' h
  intro ρ ρ' hr
  have h1 := aval_agree A ρ ρ' ats fun x hx => hr x (by simpa using List.all_eq_true.1 ha x hx)
  have h2 := val_agree M f.c ⟨[], f.phase, f.cons, []⟩ vs ρ ρ' rfl hv hr
  have e : ∀ τ'', summand M f.c ⟨f.path, f.phase, f.cons, []⟩ τ'' =
      PSem.val M f.c ⟨[], f.phase, f.cons, []⟩ τ'' := fun _ => rfl
  rw [h1, e, e, h2]

/-- A node on dimension 3 whose labels are their own Z_3 variables, from its Z_3 fragment. -/
theorem node_q3 (E : Env) (M : ℕ) (A : ℕ → ℂ) (ms : List MLab) (hw : mOK ms = true)
    (n : Node) (f : QSem.QFrag)
    (hn : ∀ σ' : ℕ → ℕ, (∀ l, σ' l < 3) → ArchytasZX.val E σ' n = QSem.fval 3 f (QSem.zm 3 σ'))
    (hlab : n.labels.all (fun l => decide ((l, 3, l, l) ∈ ms)) = true)
    (hf : (QSem.fvarsQ f).all (fun x => decide (x ∈ n.labels)) = true)
    (σ : ℕ → ℕ) (hσ : MIn ms σ) (τ : ℕ → ZMod 2) :
    ArchytasZX.val E σ n = QSem.fval 3 f (mtrits ms σ) * fvalA M A Frag.unit [] τ := by
  rw [fvalA_unit, mul_one]
  have hl : ∀ l ∈ n.labels, (l, 3, l, l) ∈ ms := fun l h => by
    simpa using List.all_eq_true.1 hlab l h
  set σ' : ℕ → ℕ := fun l => if l ∈ n.labels then σ l else 0 with hσ'
  have hr : ArchytasZX.val E σ n = ArchytasZX.val E σ' n :=
    val_reads E n σ σ' fun x hx => by simp [hσ', hx]
  have hlt : ∀ l, σ' l < 3 := by
    intro l
    simp only [hσ']
    split_ifs with h
    · exact hσ _ (hl l h)
    · omega
  rw [hr, hn σ' hlt]
  apply QSem.fval_agree
  intro x hx
  have hx' : x ∈ n.labels := by simpa using List.all_eq_true.1 hf x hx
  have ht : (tvars ms).Nodup := by
    simp only [mOK, Bool.and_eq_true, decide_eq_true_eq] at hw
    exact hw.1.2
  have := mtrits_get σ (m := (x, 3, x, x)) (by simp) ms (hl x hx') ht
  simp only at this
  rw [this]
  simp [QSem.zm, hσ', hx', mtrit]

/-- A node on dimension 2 whose labels are their own F_2 variables, from its F_2 fragment with
atoms over the labels `labs` (`dig l = [l]`). -/
theorem node_q2 (E : Env) (M : ℕ) (A : ℕ → ℂ) (ms : List MLab) (hw : mOK ms = true)
    (es : List Ent) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (n : Node) (f : Frag) (ats : Atoms)
    (hn : ∀ σ' : ℕ → ℕ, (∀ l ∈ labs, σ' l < 2 ^ (dig l).length) →
      ArchytasZX.val E σ' n = fvalA M A f ats (dbits own σ'))
    (hlab : labs.all (fun l => decide ((l, 2, l, l) ∈ ms) && decide (dig l = [l]) &&
      decide (l ∉ es.map (·.1))) = true)
    (vs : List ℕ) (hv : varsIn vs ⟨[], f.phase, f.cons, []⟩ = true)
    (ha : (avars ats).all (fun x => decide (x ∈ vs)) = true)
    (hvs : vs.all (fun x => decide (x ∈ f.path) || decide (x ∈ labs)) = true)
    (σ : ℕ → ℕ) (hσ : MIn ms σ) (ρ : ℕ → ZMod 3) :
    ArchytasZX.val E σ n = QSem.fval 3 QSem.QFrag.unit ρ *
      fvalA M A f ats (liftA es ρ (mbits ms σ)) := by
  rw [QSem.fval_unit, one_mul]
  have hl : ∀ l ∈ labs, (l, 2, l, l) ∈ ms ∧ dig l = [l] ∧ l ∉ es.map (·.1) := fun l h => by
    simpa [and_assoc] using List.all_eq_true.1 hlab l h
  have hlt : ∀ l ∈ labs, σ l < 2 ^ (dig l).length := by
    intro l h
    rw [(hl l h).2.1]
    exact hσ _ (hl l h).1
  rw [hn σ hlt]
  apply fvalA_agree M A f ats vs hv ha
  intro x hx hp
  have hxl : x ∈ labs := by
    have := List.all_eq_true.1 hvs x hx
    simpa [hp] using this
  obtain ⟨hm, hd, he⟩ := hl x hxl
  have hb : (bvars ms).Nodup := by
    simp only [mOK, Bool.and_eq_true, decide_eq_true_eq] at hw
    exact hw.1.1.2
  rw [liftA_out ρ _ x es he]
  have h1 := dbits_get hown σ hxl (i := 0) (by rw [hd]; simp)
  simp only [hd, List.getElem_cons_zero] at h1
  have h2 := mbits_get σ (m := (x, 2, x, x)) (by simp) ms hm hb
  simp only at h2
  rw [h1, h2]
  simp [mbit, digit]


/-! ## Binders and splitters (3, 2) on 6 -/

theorem tB_six (x y z : ℕ) (hx : x < 6) (hy : y < 3) (hz : z < 2) :
    tB 3 2 x y z = (if mtrit 6 x - mtrit 3 y = 0 then (1 : ℂ) else 0) *
      (if mbit x + mbit z = 0 then 1 else 0) := by
  unfold tB mtrit mbit
  interval_cases x <;> interval_cases y <;> interval_cases z <;> simp <;> decide

/-- The Z_3 fragment `[k_3 = a]` of a binder or splitter. -/
def bFrag3 (k a : ℕ) : QSem.QFrag := ⟨[], [], [QSem.eqQ 3 k a], 1⟩

/-- The F_2 fragment `[k_2 = b]` of a binder or splitter. -/
def bFrag2 (k b : ℕ) : Frag := ⟨[], [], [[[k], [b]]], 1⟩

theorem bFrag_val (M : ℕ) (A : ℕ → ℂ) (k a b : ℕ) (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) :
    QSem.fval 3 (bFrag3 k a) ρ * fvalA M A (bFrag2 k b) [] τ =
      (if ρ k - ρ a = 0 then (1 : ℂ) else 0) * (if τ k + τ b = 0 then 1 else 0) := by
  simp [QSem.fval, QSem.val, QSem.qsum, QSem.summand, bFrag3, QSem.qv_eqQ, fvalA, bFrag2, bsum,
    aval, summand, fz, mz, pn, ex_zero, QSem.qv_nil, QSem.ψ_zero]

/-- A (3, 2) binder or splitter on labels `k` (6), `a` (3), `b` (2) that are their own digit
variables. -/
theorem node_b (E : Env) (M : ℕ) (A : ℕ → ℂ) (ms : List MLab) (hw : mOK ms = true)
    (es : List Ent) (n : Node) (k a b : ℕ)
    (hv : ∀ σ : ℕ → ℕ, ArchytasZX.val E σ n = tB 3 2 (σ k) (σ a) (σ b))
    (hm : (k, 6, k, k) ∈ ms ∧ (a, 3, a, a) ∈ ms ∧ (b, 2, b, b) ∈ ms)
    (he : k ∉ es.map (·.1) ∧ b ∉ es.map (·.1)) (σ : ℕ → ℕ) (hσ : MIn ms σ) :
    ArchytasZX.val E σ n = QSem.fval 3 (bFrag3 k a) (mtrits ms σ) *
      fvalA M A (bFrag2 k b) [] (liftA es (mtrits ms σ) (mbits ms σ)) := by
  have hw' := hw
  simp only [mOK, Bool.and_eq_true, decide_eq_true_eq] at hw'
  obtain ⟨⟨⟨_, hb⟩, ht⟩, _⟩ := hw'
  rw [bFrag_val, liftA_out _ _ k es he.1, liftA_out _ _ b es he.2, hv σ]
  have t1 := mtrits_get σ (m := (k, 6, k, k)) (by simp) ms hm.1 ht
  have t2 := mtrits_get σ (m := (a, 3, a, a)) (by simp) ms hm.2.1 ht
  have b1 := mbits_get σ (m := (k, 6, k, k)) (by simp) ms hm.1 hb
  have b2 := mbits_get σ (m := (b, 2, b, b)) (by simp) ms hm.2.2 hb
  simp only at t1 t2 b1 b2
  rw [t1, t2, b1, b2]
  exact tB_six _ _ _ (hσ _ hm.1) (hσ _ hm.2.1) (hσ _ hm.2.2)

theorem val_binder (E : Env) (σ : ℕ → ℕ) (n : Node) (k a b : ℕ) (hk : n.kind = .B)
    (ho : n.outs = [k]) (hi : n.ins = [a, b]) (hd : E.dimv n.dim = 3) (hp : E.dimv n.ph = 2) :
    ArchytasZX.val E σ n = tB 3 2 (σ k) (σ a) (σ b) := by
  unfold ArchytasZX.val
  rw [hk, ho, hi]
  simp [hd, hp]

theorem val_splitter (E : Env) (σ : ℕ → ℕ) (n : Node) (k a b : ℕ) (hk : n.kind = .S)
    (ho : n.outs = [a, b]) (hi : n.ins = [k]) (hd : E.dimv n.dim = 3) (hp : E.dimv n.ph = 2) :
    ArchytasZX.val E σ n = tB 3 2 (σ k) (σ a) (σ b) := by
  unfold ArchytasZX.val
  rw [hk, ho, hi]
  simp [hd, hp]


/-! ## Z spiders on 6 -/

/-- The index `f + 2 [s = 1] + 4 [s = 2]` of the value `x = 2 s + f` on the variables
`(f, u₁, u₂)`. -/
def sixIdx (x : ℕ) : ℕ :=
  x % 2 + 2 * (if x / 2 = 1 then 1 else 0) + 4 * (if x / 2 = 2 then 1 else 0)

/-- Whether `ts8` at `sixIdx x` is `ts` at `x` for every `x < 6`. -/
def sixOK (ts8 ts : List ℕ) : Bool :=
  (List.range 6).all fun x => decide (ts8.getD (sixIdx x) 0 = ts.getD x 0)

/-- The Z_3 fragment of a Z spider on 6: each leg's Z_3 digit equal to the anchor's. -/
def z6Frag3 (l0 : ℕ) (ls : List ℕ) : QSem.QFrag := ⟨[], [], ls.map (QSem.eqQ 3 l0), 1⟩

/-- The F_2 fragment of a Z spider on 6: each leg's F_2 digit equal to the anchor's, the phase
`P` over the anchor's F_2 digit and its indicators. -/
def z6Frag2 (l0 : ℕ) (ls : List ℕ) (P : Poly) : Frag :=
  ⟨[], P, ls.map (fun l => [[l0], [l]]), 1⟩

theorem liftA_get (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) :
    ∀ es : List Ent, (es.map (·.1)).Nodup → ∀ e ∈ es, liftA es ρ τ e.1 = entVal ρ e
  | [], _, e, he => absurd he List.not_mem_nil
  | e' :: es, hn, e, he => by
    simp only [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 he with rfl | he'
    · simp [liftA]
    · have hne : e.1 ≠ e'.1 := fun h => hn.1 (h ▸ List.mem_map.2 ⟨e, he', rfl⟩)
      rw [liftA, Function.update_of_ne hne, liftA_get ρ τ es hn.2 e he']

theorem six_eq (x y : ℕ) (hx : x < 6) (hy : y < 6) :
    (mtrit 6 x - mtrit 6 y = 0 ∧ mbit x + mbit y = 0) ↔ y = x := by
  unfold mtrit mbit
  interval_cases x <;> interval_cases y <;> simp <;> decide

theorem six_bits (x : ℕ) (hx : x < 6) :
    ((digit 2 0 (sixIdx x) : ℕ) : ZMod 2) = mbit x ∧
      ((digit 2 1 (sixIdx x) : ℕ) : ZMod 2) = ind 1 (mtrit 6 x) ∧
      ((digit 2 2 (sixIdx x) : ℕ) : ZMod 2) = ind 2 (mtrit 6 x) := by
  unfold sixIdx mbit mtrit ind digit
  interval_cases x <;> simp <;> decide

theorem sixIdx_lt (x : ℕ) : sixIdx x < 2 ^ [0, 0, 0].length := by
  unfold sixIdx
  have := Nat.mod_lt x (show 2 > 0 by norm_num)
  split_ifs <;> simp <;> omega

/-- A Z spider on 6 whose legs are their own digit variables, with the indicator entries
`(u₁, l0, 1, 0, 1)` and `(u₂, l0, 1, 0, 2)` of its anchor `l0`. -/
theorem node_z6 (E : Env) (M : ℕ) (A : ℕ → ℂ) (ms : List MLab) (hw : mOK ms = true)
    (es : List Ent) (n : Node) (hk : n.kind = .Z) (hd : E.dimv n.dim = 6) (l0 u₁ u₂ : ℕ)
    (hl0 : l0 ∈ n.outs ++ n.ins) (hM : 0 < M) (P : Poly) (ts ts8 : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / M)
    (htab : tableOK M [l0, u₁, u₂] P ts8 = true) (hsix : sixOK ts8 ts = true)
    (hPv : P.all (fun t => t.1.all fun x => decide (x ∈ [l0, u₁, u₂])) = true)
    (hnd : [l0, u₁, u₂].Nodup)
    (hlab : (n.outs ++ n.ins).all
      (fun l => decide ((l, 6, l, l) ∈ ms) && decide (l ∉ es.map (·.1))) = true)
    (hes : (es.map (·.1)).Nodup) (he₁ : (u₁, l0, 1, 0, 1) ∈ es) (he₂ : (u₂, l0, 1, 0, 2) ∈ es)
    (σ : ℕ → ℕ) (hσ : MIn ms σ) :
    ArchytasZX.val E σ n = QSem.fval 3 (z6Frag3 l0 (n.outs ++ n.ins)) (mtrits ms σ) *
      fvalA M A (z6Frag2 l0 (n.outs ++ n.ins) P) [] (liftA es (mtrits ms σ) (mbits ms σ)) := by
  set ls := n.outs ++ n.ins with hls
  set ρ := mtrits ms σ with hρdef
  set τ := liftA es ρ (mbits ms σ) with hτdef
  have hw' := hw
  simp only [mOK, Bool.and_eq_true, decide_eq_true_eq] at hw'
  obtain ⟨⟨⟨_, hb⟩, ht⟩, _⟩ := hw'
  have hl : ∀ l ∈ ls, (l, 6, l, l) ∈ ms ∧ l ∉ es.map (·.1) := fun l h => by
    simpa using List.all_eq_true.1 hlab l h
  have hρ : ∀ l ∈ ls, ρ l = mtrit 6 (σ l) := fun l h => by
    have := mtrits_get σ (m := (l, 6, l, l)) (by simp) ms (hl l h).1 ht
    simpa using this
  have hτ : ∀ l ∈ ls, τ l = mbit (σ l) := fun l h => by
    rw [hτdef, liftA_out _ _ l es (hl l h).2]
    have := mbits_get σ (m := (l, 6, l, l)) (by simp) ms (hl l h).1 hb
    simpa using this
  have hlt : ∀ l ∈ ls, σ l < 6 := fun l h => hσ _ (hl l h).1
  have hv : ArchytasZX.val E σ n = tZ 6 (E.phase n.ph) (σ l0 :: ls.map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd, List.map_append]
    apply tZ_congr
    intro x
    simp only [List.mem_cons, ← List.map_append]
    constructor
    · exact Or.inr
    · rintro (rfl | hx)
      · exact List.mem_map.2 ⟨l0, hl0, rfl⟩
      · exact hx
  rw [hv, tZ_val _ _ _ _ (hlt l0 hl0)]
  have hcond : ((∀ q ∈ ls.map (QSem.eqQ 3 l0) ++ [], QSem.qv ρ q = 0) ∧
      (∀ q ∈ ls.map (fun l => [[l0], [l]]) ++ [], fz τ q = 0)) ↔
      ∀ w ∈ ls.map σ, w = σ l0 := by
    simp only [List.append_nil, List.forall_mem_map, QSem.qv_eqQ, fz, mz, List.map_cons,
      List.map_nil, List.prod_cons, List.prod_nil, mul_one, List.sum_cons, List.sum_nil,
      add_zero]
    constructor
    · rintro ⟨h3, h2⟩ l h
      have := (six_eq (σ l0) (σ l) (hlt l0 hl0) (hlt l h)).1
        ⟨by rw [← hρ l0 hl0, ← hρ l h]; exact h3 l h, by rw [← hτ l0 hl0, ← hτ l h]; exact h2 l h⟩
      exact this
    · intro H
      refine ⟨fun l h => ?_, fun l h => ?_⟩
      · rw [hρ l0 hl0, hρ l h]
        exact ((six_eq (σ l0) (σ l) (hlt l0 hl0) (hlt l h)).2 (H l h)).1
      · rw [hτ l0 hl0, hτ l h]
        exact ((six_eq (σ l0) (σ l) (hlt l0 hl0) (hlt l h)).2 (H l h)).2
  have hfv : QSem.fval 3 (z6Frag3 l0 ls) ρ * fvalA M A (z6Frag2 l0 ls P) [] τ =
      if (∀ q ∈ ls.map (QSem.eqQ 3 l0) ++ [], QSem.qv ρ q = 0) ∧
        (∀ q ∈ ls.map (fun l => [[l0], [l]]) ++ [], fz τ q = 0) then
        ex ((pn τ P : ℝ) / M) else 0 := by
    have h3v : QSem.fval 3 (z6Frag3 l0 ls) ρ =
        if (∀ q ∈ ls.map (QSem.eqQ 3 l0) ++ [], QSem.qv ρ q = 0) then 1 else 0 := by
      simp only [QSem.fval, QSem.val, QSem.qsum, QSem.summand, z6Frag3, QSem.qv_nil,
        QSem.ψ_zero, mul_one]
    have h2v : fvalA M A (z6Frag2 l0 ls P) [] τ =
        if (∀ q ∈ ls.map (fun l => [[l0], [l]]) ++ [], fz τ q = 0) then
          ex ((pn τ P : ℝ) / M) else 0 := by
      simp only [fvalA, z6Frag2, bsum, aval, summand, one_mul, List.map_nil, List.prod_nil]
    rw [h3v, h2v]
    by_cases h1 : ∀ a ∈ ls, QSem.qv ρ (QSem.eqQ 3 l0 a) = 0 <;>
      by_cases h2 : ∀ a ∈ ls, fz τ [[l0], [a]] = 0 <;>
      simp [h1, h2] <;>
      rw [if_pos h2, if_pos h1, if_pos ⟨h1, h2⟩]
  rw [hfv, if_congr hcond rfl rfl]
  split_ifs with hc
  swap
  · rfl
  set x := σ l0 with hx
  have hx6 : x < 6 := hlt l0 hl0
  obtain ⟨d0, d1, d2⟩ := six_bits x hx6
  have hu₁ : τ u₁ = ind 1 (mtrit 6 x) := by
    rw [hτdef, liftA_get ρ _ es hes _ he₁]
    simp [entVal, hρ l0 hl0, ← hx]
  have hu₂ : τ u₂ = ind 2 (mtrit 6 x) := by
    rw [hτdef, liftA_get ρ _ es hes _ he₂]
    simp [entVal, hρ l0 hl0, ← hx]
  have s0 := setBits_idx [l0, u₁, u₂] (sixIdx x) (fun _ => 0) hnd 0 (by simp)
  have s1 := setBits_idx [l0, u₁, u₂] (sixIdx x) (fun _ => 0) hnd 1 (by simp)
  have s2 := setBits_idx [l0, u₁, u₂] (sixIdx x) (fun _ => 0) hnd 2 (by simp)
  simp only [List.get_eq_getElem, List.getElem_cons_zero, List.getElem_cons_succ] at s0 s1 s2
  have hpn : pn τ P = pn (setBits [l0, u₁, u₂] (sixIdx x) fun _ => 0) P := by
    apply pn_congr
    intro t ht' y hy
    have hy' : y ∈ [l0, u₁, u₂] := by
      have := List.all_eq_true.1 hPv t ht'
      simpa using List.all_eq_true.1 this y hy
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hy'
    rcases hy' with rfl | rfl | rfl
    · rw [s0, d0, hτ _ hl0]
    · rw [s1, d1, hu₁]
    · rw [s2, d2, hu₂]
  rw [hpn, hph]
  have hv8 : sixIdx x < 2 ^ [l0, u₁, u₂].length := sixIdx_lt x
  have := List.all_eq_true.1 htab (sixIdx x) (List.mem_range.2 hv8)
  simp only [decide_eq_true_eq] at this
  have h6 := List.all_eq_true.1 hsix x (List.mem_range.2 hx6)
  simp only [decide_eq_true_eq] at h6
  rw [h6] at this
  obtain ⟨q, hq⟩ := Int.dvd_of_emod_eq_zero this
  have e : (((ts.getD x 0 : ℕ) : ℝ)) = (((ts.getD x 0 : ℕ) : ℤ) : ℝ) := by
    push_cast; rfl
  rw [e]
  first
    | exact ex_peq hM ⟨q, by linarith⟩
    | exact ex_peq hM ⟨-q, by linarith⟩
    | exact (ex_peq hM ⟨q, by linarith⟩).symm
    | exact (ex_peq hM ⟨-q, by linarith⟩).symm

/-! ## Steps for the Z_3 replay -/

/-- A mixed path sum with its Z_3 path permuted. -/
theorem mval_perm (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : AData) (t : QSem.Data)
    (es : List Ent) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3) (p' : List ℕ) (h : t.path.Perm p') :
    mval M c A q t es τ ρ = mval M c A q ⟨p', t.phase, t.cons, t.rows⟩ es τ ρ := by
  unfold mval
  exact QSem.qsum_perm _ h ρ

/-- Two entries with one reading take one value. -/
theorem liftA_dup (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) (es : List Ent)
    (hes : (es.map (·.1)).Nodup) (e e' : Ent) (he : e ∈ es) (he' : e' ∈ es)
    (h : e.2 = e'.2) : liftA es ρ τ e'.1 = liftA es ρ τ e.1 := by
  rw [liftA_get ρ τ es hes e he, liftA_get ρ τ es hes e' he']
  unfold entVal
  rw [h]

/-- `z` replaced by `B` in a qubit path sum with atoms. -/
def substA (M z : ℕ) (B : F2) (q : AData) : AData :=
  ⟨⟨q.s.path, substP M z B q.s.phase, q.s.cons.map (substF z B), q.s.rows.map (substF z B)⟩,
    q.held, q.atoms.map fun a => (a.1, substF z B a.2)⟩

theorem bsum_update_dep (G : (ℕ → ZMod 2) → ℂ) (z : ℕ) (B : F2) :
    ∀ (ys : List ℕ) (τ : ℕ → ZMod 2), z ∉ ys → (∀ m ∈ B, ∀ v ∈ m, v ∉ ys) →
      bsum ys (fun ρ => G (Function.update ρ z (fz ρ B))) τ =
        bsum ys G (Function.update τ z (fz τ B))
  | [], _, _, _ => rfl
  | y :: ys, τ, hz, hB => by
    have hy : y ≠ z := fun e => hz (e ▸ List.mem_cons_self)
    have hz' : z ∉ ys := fun e => hz (List.mem_cons_of_mem _ e)
    have hB' : ∀ m ∈ B, ∀ v ∈ m, v ∉ ys := fun m hm v hv e =>
      hB m hm v hv (List.mem_cons_of_mem _ e)
    have hBy : ∀ m ∈ B, y ∉ m := fun m hm e => hB m hm y e List.mem_cons_self
    simp only [bsum]
    rw [bsum_update_dep G z B ys _ hz' hB', bsum_update_dep G z B ys _ hz' hB',
      fz_update_not τ y 0 B hBy, fz_update_not τ y 1 B hBy, Function.update_comm hy,
      Function.update_comm hy]

/-- A free variable `z` of a qubit path sum with atoms set to `B`: the sum with `z` replaced. -/
theorem valA_subst (M : ℕ) (hM : 2 ≤ M) (c : ℂ) (A : ℕ → ℂ) (z : ℕ) (B : F2) (q : AData)
    (hz : z ∉ q.held ++ q.s.path) (hB : ∀ m ∈ B, ∀ v ∈ m, v ∉ q.held ++ q.s.path)
    (hpa : ∀ y ∈ q.s.path, y ∉ avars q.atoms)
    (hpa' : ∀ y ∈ q.s.path, y ∉ avars (substA M z B q).atoms) (τ : ℕ → ZMod 2) :
    valA M c A q (Function.update τ z (fz τ B)) = valA M c A (substA M z B q) τ := by
  rw [valA_flat M c A q _ hpa, valA_flat M c A _ τ hpa']
  show _ = bsum (q.held ++ q.s.path) _ τ
  rw [← bsum_update_dep _ z B _ τ hz hB]
  apply bsum_congr
  intro ρ
  congr 1
  · unfold aval substA
    simp only [List.map_map]
    congr 1
    apply List.map_congr_left
    intro a _
    simp only [Function.comp, fz_substF]
  · unfold summand substA
    simp only [← List.map_append, List.forall_mem_map, fz_substF]
    split_ifs
    · congr 1
      exact (ex_peq (by omega) (pn_substP ρ M z hM B q.s.phase)).symm
    · rfl

end MSem
end ArchytasZX
