/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.BoxFree
import ArchytasZX.Bundle

/-!
# Positional digits

`pos ks c t` is the place of the copy index `t` in `cidx ks c`; a value below `d ^ size ks c`
is its digits `digit d (pos ks c t) v`. `sumOver_digits` turns a sum over the digit labels of a
bundle into one sum over the bundle's value.
-/

namespace ArchytasZX

open Finset

/-- The number of copy indices along `c`. -/
def size (ks : ℕ → ℕ) (c : List ℕ) : ℕ := (c.map ks).prod

/-- The place of `t` in `cidx ks c`. -/
def pos (ks : ℕ → ℕ) : List ℕ → List ℕ → ℕ
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: c, i :: t => i * size ks c + pos ks c t

theorem size_cons (ks : ℕ → ℕ) (m : ℕ) (c : List ℕ) : size ks (m :: c) = ks m * size ks c := by
  simp [size]

theorem size_append (ks : ℕ → ℕ) (a b : List ℕ) : size ks (a ++ b) = size ks a * size ks b := by
  simp [size, List.prod_append]

theorem cidx_pos (ks : ℕ → ℕ) (c : List ℕ) : (cidx ks c).map (pos ks c) = List.range (size ks c) := by
  induction c with
  | nil => simp [cidx, pos, size]
  | cons m c ih =>
    simp only [cidx, List.map_flatMap, List.map_map, size_cons]
    have : ∀ i, List.map (pos ks (m :: c) ∘ (i :: ·)) (cidx ks c) =
        (List.range (size ks c)).map (fun j => i * size ks c + j) := fun i => by
      rw [← ih, List.map_map]; rfl
    simp only [this]
    have hb := range_blocks (fun _ => size ks c) (ks m)
    have hw : ∀ i, wsum (fun _ => size ks c) i = i * size ks c := fun i => by
      simp [wsum, List.sum_replicate, List.map_const']
    simp only [hw] at hb
    rw [hb, Nat.mul_comm]

theorem pos_lt (ks : ℕ → ℕ) (c t : List ℕ) (ht : t ∈ cidx ks c) : pos ks c t < size ks c := by
  have : pos ks c t ∈ List.range (size ks c) := by
    rw [← cidx_pos]; exact List.mem_map.2 ⟨t, ht, rfl⟩
  exact List.mem_range.1 this

theorem pos_inj (ks : ℕ → ℕ) (c t u : List ℕ) (ht : t ∈ cidx ks c) (hu : u ∈ cidx ks c)
    (h : pos ks c t = pos ks c u) : t = u := by
  have hnd : ((cidx ks c).map (pos ks c)).Nodup := by rw [cidx_pos]; exact List.nodup_range
  exact List.inj_on_of_nodup_map hnd ht hu h

theorem pos_append (ks : ℕ → ℕ) (a b t s : List ℕ) (ht : t ∈ cidx ks a) :
    pos ks (a ++ b) (t ++ s) = pos ks a t * size ks b + pos ks b s := by
  induction a generalizing t with
  | nil => cases t with
    | nil => simp [pos, size]
    | cons i t => simp [cidx] at ht
  | cons m a ih =>
    cases t with
    | nil => simp [cidx] at ht
    | cons i t =>
      simp only [cidx, List.mem_flatMap, List.mem_map, List.mem_range] at ht
      obtain ⟨i', -, t', ht', e⟩ := ht
      obtain ⟨rfl, rfl⟩ := List.cons.inj e
      simp only [List.cons_append, pos, ih t' ht', size_append]
      ring

theorem cidx_length (ks : ℕ → ℕ) (c : List ℕ) : (cidx ks c).length = size ks c := by
  have := congrArg List.length (cidx_pos ks c)
  simpa using this

theorem getElem_cidx_pos (ks : ℕ → ℕ) (c : List ℕ) (i : ℕ) (hi : i < (cidx ks c).length) :
    pos ks c (cidx ks c)[i] = i := by
  have := congrArg (·[i]?) (cidx_pos ks c)
  simp only [List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some] at this
  rw [List.getElem?_range (by rw [← cidx_length]; exact hi)] at this
  exact Option.some.inj this

/-- `digit` in base `B ^ k` then base `B`: the inner place counts `k` per outer place. -/
theorem digit_digit (B k i j v : ℕ) (hB : 0 < B) (hj : j < k) :
    digit B j (digit (B ^ k) i v) = digit B (i * k + j) v := by
  have key : ∀ x, x / B ^ j % B = x % B ^ k / B ^ j % B := fun x => by
    have hk : B ^ k = B ^ j * (B * B ^ (k - j - 1)) := by
      rw [← pow_succ', ← pow_add]; congr 1; omega
    conv_lhs => rw [← Nat.mod_add_div x (B ^ k)]
    rw [hk, show x % (B ^ j * (B * B ^ (k - j - 1))) + B ^ j * (B * B ^ (k - j - 1)) *
        (x / (B ^ j * (B * B ^ (k - j - 1)))) = x % (B ^ j * (B * B ^ (k - j - 1))) +
        (B * (B ^ (k - j - 1) * (x / (B ^ j * (B * B ^ (k - j - 1)))))) * B ^ j by ring,
      Nat.add_mul_div_right _ _ (pow_pos hB j), Nat.add_mul_mod_self_left]
  unfold digit
  rw [← key, Nat.div_div_eq_div_mul, ← pow_mul, ← pow_add, Nat.mul_comm k i]

/-- The labels `ls` set to the base-`d` digits of `v`, label `ls[i]` to digit `i`. -/
def setDigits (ls : List ℕ) (d v : ℕ) (σ : ℕ → ℕ) (x : ℕ) : ℕ :=
  if x ∈ ls then digit d (ls.idxOf x) v else σ x

theorem digit_zero_add (d a y : ℕ) (ha : a < d) : digit d 0 (a + d * y) = a := by
  simp [digit, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ha]

theorem digit_succ_add (d a y i : ℕ) (hd : 0 < d) (ha : a < d) :
    digit d (i + 1) (a + d * y) = digit d i y := by
  unfold digit
  rw [pow_succ', ← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ hd, Nat.div_eq_of_lt ha, zero_add]

/-- A sum over `n` distinct labels of dimension `d` is one sum over values below `d ^ n`. -/
theorem sumOver_digits (Dm : ℕ → ℕ) (d : ℕ) (hd : 0 < d) (ls : List ℕ) (hnd : ls.Nodup)
    (hD : ∀ l ∈ ls, Dm l = d) (g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver Dm ls g σ = ∑ v ∈ range (d ^ ls.length), g (setDigits ls d v σ) := by
  induction ls generalizing σ with
  | nil =>
    simp only [sumOver, List.length_nil, pow_zero, Finset.range_one, Finset.sum_singleton]
    congr 1
  | cons l ls ih =>
    obtain ⟨hl, hnd'⟩ := List.nodup_cons.1 hnd
    simp only [sumOver]
    rw [hD l List.mem_cons_self, List.length_cons, pow_succ', sum_range_mul]
    refine Finset.sum_congr rfl fun a ha => ?_
    rw [ih hnd' (fun x hx => hD x (List.mem_cons_of_mem _ hx))]
    refine Finset.sum_congr rfl fun y _ => ?_
    congr 1
    funext x
    have ha' := Finset.mem_range.1 ha
    unfold setDigits
    by_cases hxl : x = l
    · subst hxl
      simp [hl, List.idxOf_cons_self, digit_zero_add d a y ha']
    · by_cases hx : x ∈ ls
      · have : (l :: ls).idxOf x = ls.idxOf x + 1 := by
          simp [List.idxOf_cons, Ne.symm hxl, beq_iff_eq]
        simp only [List.mem_cons, hxl, hx, or_true, ↓reduceIte, this, digit_succ_add d a y _ hd ha']
      · simp [hxl, hx, Function.update_of_ne hxl]

end ArchytasZX
