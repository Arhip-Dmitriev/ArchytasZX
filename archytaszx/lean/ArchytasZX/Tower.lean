/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Positional

/-!
# Towers of copies

`tensor_towerC`: a generator copied along the chain `C` (outermost first) is the product over the
copy indices `t` of `C` of the generator at each leg's block `pos C t`, in base `d` to the leg's
own remaining chain.
-/

namespace ArchytasZX

/-- Every leg value at its block `p`, in base `d` to the leg's chain. -/
def blkC (mult : ℕ → ℕ) (d p : ℕ) (cs : List (List ℕ)) (vs : List ℕ) : List ℕ :=
  List.zipWith (fun c v => digit (d ^ size mult c) p v) cs vs

theorem tower_outCsC (b : Kind) (C : List ℕ) (n : ℕ) :
    (C.foldr Kind.Bun b).outCs n = (b.outCs n).map (C ++ ·) := by
  induction C with
  | nil => simp
  | cons m C ih => simp [Kind.outCs, ih, List.map_map, Function.comp_def]

theorem tower_inCsC (b : Kind) (C : List ℕ) (n : ℕ) :
    (C.foldr Kind.Bun b).inCs n = (b.inCs n).map (C ++ ·) := by
  induction C with
  | nil => simp
  | cons m C ih => simp [Kind.inCs, ih, List.map_map, Function.comp_def]

theorem outCs_length_ge (b : Kind) (n : ℕ) : n ≤ (b.outCs n).length := by
  induction b generalizing n with
  | XB co ci => simp [Kind.outCs]; omega
  | WB cs => simp [Kind.outCs]; omega
  | Bun m b ih => simpa [Kind.outCs] using ih n
  | _ => simp [Kind.outCs]

theorem inCs_length_ge (b : Kind) (n : ℕ) : n ≤ (b.inCs n).length := by
  induction b generalizing n with
  | ZB cs => simp [Kind.inCs]; omega
  | XB co ci => simp [Kind.inCs]; omega
  | Bun m b ih => simpa [Kind.inCs] using ih n
  | _ => simp [Kind.inCs]

theorem zipWith_zipWith_same {α β γ δ : Type*} (f : α → γ → δ) (g : α → β → γ) (cs : List α)
    (vs : List β) : List.zipWith f cs (List.zipWith g cs vs) = List.zipWith (fun c v => f c (g c v)) cs vs := by
  induction cs generalizing vs with
  | nil => simp
  | cons c cs ih => cases vs with
    | nil => simp
    | cons v vs => simp [ih]

theorem zipWith_snd {α β : Type*} (cs : List α) (vs : List β) (h : vs.length ≤ cs.length) :
    List.zipWith (fun _ v => v) cs vs = vs := by
  induction cs generalizing vs with
  | nil => cases vs with
    | nil => rfl
    | cons v vs => simp at h
  | cons c cs ih => cases vs with
    | nil => simp
    | cons v vs => simp only [List.zipWith_cons_cons]; rw [ih vs (by simpa using h)]

theorem zipWith_congr' {α β γ : Type*} (f g : α → β → γ) (cs : List α) (vs : List β)
    (h : ∀ p ∈ cs.zip vs, f p.1 p.2 = g p.1 p.2) : List.zipWith f cs vs = List.zipWith g cs vs := by
  induction cs generalizing vs with
  | nil => simp
  | cons c cs ih => cases vs with
    | nil => simp
    | cons v vs =>
      simp only [List.zipWith_cons_cons, List.zip_cons_cons, List.mem_cons, forall_eq_or_imp] at h ⊢
      rw [h.1, ih vs h.2]

theorem prod_flatMap'' {α : Type*} (l : List α) (f : α → List ℂ) :
    (l.flatMap f).prod = (l.map fun a => (f a).prod).prod := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.flatMap_cons, List.prod_append, ih]

theorem length_zipWith_ge {α β γ : Type*} (f : α → β → γ) (cs : List α) (vs : List β)
    (h : vs.length ≤ cs.length) : (List.zipWith f cs vs).length = vs.length := by
  simp [List.length_zipWith]; omega

/-- A generator copied along `C`, at leg values below `d` to the power of their chains' sizes. -/
theorem tensor_towerC (E : Env) (b : Kind) (d ph : ℕ) (hd : 0 < d) (C : List ℕ) (os is : List ℕ)
    (hos : ∀ p ∈ (b.outCs os.length).zip os, p.2 < d ^ size E.mult (C ++ p.1))
    (his : ∀ p ∈ (b.inCs is.length).zip is, p.2 < d ^ size E.mult (C ++ p.1)) :
    tensor E (C.foldr Kind.Bun b) d ph os is = ((cidx E.mult C).map fun t =>
      tensor E b d ph (blkC E.mult d (pos E.mult C t) (b.outCs os.length) os)
        (blkC E.mult d (pos E.mult C t) (b.inCs is.length) is)).prod := by
  induction C generalizing os is with
  | nil =>
    simp only [List.foldr_nil, cidx, pos, List.map_cons, List.map_nil, List.prod_cons,
      List.prod_nil, mul_one, blkC]
    congr 1
    · rw [zipWith_congr' _ (fun _ v => v) _ _ fun p hp => by
        simp only [digit, pow_zero, Nat.div_one]; exact Nat.mod_eq_of_lt (by simpa using hos p hp),
        zipWith_snd _ _ (outCs_length_ge b _)]
    · rw [zipWith_congr' _ (fun _ v => v) _ _ fun p hp => by
        simp only [digit, pow_zero, Nat.div_one]; exact Nat.mod_eq_of_lt (by simpa using his p hp),
        zipWith_snd _ _ (inCs_length_ge b _)]
  | cons m C ih =>
    have hco : ∀ i, copyOuts E (C.foldr Kind.Bun b) d i os =
        List.zipWith (fun c v => digit (d ^ size E.mult (C ++ c)) i v) (b.outCs os.length) os :=
      fun i => by simp only [copyOuts, copyLegs, tower_outCsC, List.zipWith_map_left]; rfl
    have hci : ∀ i, copyIns E (C.foldr Kind.Bun b) d i is =
        List.zipWith (fun c v => digit (d ^ size E.mult (C ++ c)) i v) (b.inCs is.length) is :=
      fun i => by simp only [copyIns, copyLegs, tower_inCsC, List.zipWith_map_left]; rfl
    simp only [List.foldr_cons]
    show ((List.range (E.mult m)).map fun i => tensor E (C.foldr Kind.Bun b) d ph
      (copyOuts E (C.foldr Kind.Bun b) d i os) (copyIns E (C.foldr Kind.Bun b) d i is)).prod = _
    simp only [hco, hci, cidx, List.map_flatMap, prod_flatMap'', List.map_map]
    congr 1
    refine List.map_congr_left fun i hi => ?_
    have hlo := length_zipWith_ge (fun c v => digit (d ^ size E.mult (C ++ c)) i v) _ os
      (outCs_length_ge b _)
    have hli := length_zipWith_ge (fun c v => digit (d ^ size E.mult (C ++ c)) i v) _ is
      (inCs_length_ge b _)
    have hpos : 0 < d ^ size E.mult (C ++ []) := pow_pos hd _
    rw [ih _ _ (fun p hp => ?_) (fun p hp => ?_)]
    · congr 1
      refine List.map_congr_left fun t ht => ?_
      simp only [Function.comp, blkC, hlo, hli, zipWith_zipWith_same]
      have hpt := pos_lt E.mult C t ht
      have key : ∀ c v, digit (d ^ size E.mult c) (pos E.mult C t)
          (digit (d ^ size E.mult (C ++ c)) i v) = digit (d ^ size E.mult c) (pos E.mult (m :: C) (i :: t)) v :=
        fun c v => by
          rw [show d ^ size E.mult (C ++ c) = (d ^ size E.mult c) ^ size E.mult C by
            rw [← pow_mul, size_append, Nat.mul_comm], digit_digit _ _ _ _ _ (pow_pos hd _) hpt]
          simp [pos]
      simp only [key]
    · rw [hlo] at hp
      obtain ⟨j, hj, e⟩ := List.mem_iff_getElem.1 hp
      rw [← e]
      simp only [List.getElem_zip, List.getElem_zipWith]
      exact digit_lt _ _ _ (pow_pos hd _)
    · rw [hli] at hp
      obtain ⟨j, hj, e⟩ := List.mem_iff_getElem.1 hp
      rw [← e]
      simp only [List.getElem_zip, List.getElem_zipWith]
      exact digit_lt _ _ _ (pow_pos hd _)

theorem digsC_pos (mult : ℕ → ℕ) (d : ℕ) (hd : 0 < d) (c : List ℕ) (w : ℕ) :
    digsC mult d c w = if c = [] then [w] else (cidx mult c).map fun s => digit d (pos mult c s) w := by
  induction c generalizing w with
  | nil => simp [digsC]
  | cons m c ih =>
    simp only [digsC, digs, List.flatMap_map, reduceCtorEq, ↓reduceIte, cidx, List.map_flatMap,
      List.map_map]
    refine List.flatMap_congr fun i hi => ?_
    rw [ih]
    split_ifs with hc
    · subst hc
      simp [pos, size, digit, cidx]
    · refine List.map_congr_left fun s hs => ?_
      simp only [Function.comp, pos]
      rw [show (c.map mult).prod = size mult c from rfl,
        digit_digit d _ _ _ _ hd (pos_lt mult c s hs)]

end ArchytasZX
