/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFamFca

/-!
# The engine's Fourier constant adder networks

The networks the engine exports for the flat `fourier_constant_adder` at `n` copies and its
binder / shift / splitter spec, with their environment.

* `engL n`: outputs `0 … n - 1`, inputs `n … 2n - 1`, copy `j`'s chain at `2n + 8j`, the pair
  legs from `10n`, the pair phases as `ZP` nodes.
* `engR n`: the spec from `2n² + 8n`.
* `engE d n`: dimension `i < n` is `d ^ (i + 1)`, dimension `n` is `1`; phase `p ∈ [1, n]` is
  `k ↦ 2k / d ^ (n - p + 1)`, phase `n + 1` is `k ↦ -2k / d ^ n`.
* `engL_lit`, `engR_lit`: equal to the engine's literal exports at `n = 1 … 5`.
-/

namespace ArchytasZX

namespace FcaEng

open List

/-- `j (j - 1) / 2`. -/
def tri (j : ℕ) : ℕ := j * (j - 1) / 2

/-- Copy `j`'s chain base. -/
def cb (n j : ℕ) : ℕ := 2 * n + 8 * j

/-- Position of the pair `(m, j)`, `m < j`, ordered by `m` then `j`. -/
def i2 (n m j : ℕ) : ℕ := m * (n - 1) - tri m + (j - m - 1)

/-- The `g0` leg of copy `j` to copy `m < j`. -/
def a0 (n j m : ℕ) : ℕ := 10 * n + tri j + m

/-- The `g2` leg of copy `m` to copy `j > m`. -/
def a2 (n m j : ℕ) : ℕ := 10 * n + tri n + i2 n m j

/-- The `g5` leg of copy `m` to copy `j > m`. -/
def a5 (n m j : ℕ) : ℕ := 10 * n + 2 * tri n + i2 n m j

/-- The `g9` leg of copy `j` to copy `m < j`. -/
def a9 (n j m : ℕ) : ℕ := 10 * n + 3 * tri n + tri j + m

/-- A node of dimension and phase index `0`. -/
def nd (k : Kind) (os is : List ℕ) : Node := ⟨k, os, is, 0, 0⟩

/-- Copy `j`'s chain nodes `g1, g4, g6, g7, g8`. -/
def chainN (n j : ℕ) : List Node :=
  [nd .F [cb n j + 1] [cb n j], ⟨.Z, [cb n j + 3], [cb n j + 2], 0, j + 1⟩,
   nd .F [cb n j + 5] [cb n j + 4], nd .F [cb n j + 6] [cb n j + 5],
   nd .F [cb n j + 7] [cb n j + 6]]

/-- The copies `j > m`. -/
def later (n m : ℕ) : List ℕ := (range (n - m - 1)).map (m + 1 + ·)

/-- The engine's boxed side at `n` copies. -/
def engL (n : ℕ) : Net where
  nodes :=
    (range n).flatMap (chainN n) ++
    (range n).map (fun j => nd .Z [cb n j] ((n + j) :: (range j).map (a0 n j))) ++
    (range n).map (fun j => nd .Z ((cb n j + 2) :: (later n j).map (a2 n j)) [cb n j + 1]) ++
    (range n).map (fun j => nd .Z ((cb n j + 4) :: (later n j).map (a5 n j)) [cb n j + 3]) ++
    (range n).map (fun j => nd .Z [j] ((cb n j + 7) :: (range j).map (a9 n j))) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 1, [a0 n j m], [a2 n m j], j - m, 0⟩) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 (-1), [a9 n j m], [a5 n m j], j - m, 0⟩)
  int := (range (8 * n + 4 * tri n)).map (2 * n + ·)
  bnd := range (2 * n)

/-- Start of the spec's labels. -/
def rb (n : ℕ) : ℕ := 2 * n * n + 8 * n

/-- The dimension index of `d ^ t`. -/
def dI (n t : ℕ) : ℕ := if t = 0 then n else t - 1

/-- The output of binder `t`. -/
def bout (n t : ℕ) : ℕ := if t + 1 = n then rb n + 1 else rb n + 4 + t

/-- The input of binder `t`. -/
def bin (n t : ℕ) : ℕ := if t = 0 then rb n else rb n + 3 + t

/-- The splitter value of the top `t` digits. -/
def sv (n t : ℕ) : ℕ := if t = 0 then rb n + 3 else if t = n then rb n + 2 else rb n + n + 2 + t

/-- The binder chain. -/
def bN (n : ℕ) : List Node := (range n).map fun t => ⟨.B, [bout n t], [bin n t, n + t], dI n t, 0⟩

/-- The splitter chain. -/
def sN (n : ℕ) : List Node := (range n).map fun t => ⟨.S, [sv n t, t], [sv n (t + 1)], dI n t, 0⟩

/-- The binder head. -/
def hN (n : ℕ) : Node := ⟨.Z, [rb n], [], n, 0⟩

/-- The shift by `2` mod `d ^ n`. -/
def xN (n : ℕ) : Node := ⟨.X, [rb n + 2], [rb n + 1], n - 1, n + 1⟩

/-- The splitter end. -/
def eN (n : ℕ) : Node := ⟨.Z, [], [rb n + 3], n, 0⟩

/-- The engine's spec at `n` digits. -/
def engR (n : ℕ) : Net where
  nodes := if n = 1 then [hN n] ++ bN n ++ [xN n] ++ sN n ++ [eN n]
    else [hN n, xN n, eN n] ++ bN n ++ sN n
  int := (range (2 * n + 2)).map (rb n + ·)
  bnd := range (2 * n)

/-- The exponent of `d` in the dimension of the spec's label `rb n + k`. -/
def edim (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = 3 then 0 else if k ≤ 2 then n else if k < n + 3 then k - 3 else k - n - 2

/-- The spec's label dimensions. -/
def engLabs (n : ℕ) : List (ℕ × ℕ) := (range (2 * n + 2)).map fun k => (rb n + k, dI n (edim n k))

/-- The engine's dimensions. -/
def engD (d n i : ℕ) : ℕ := if i < n then d ^ (i + 1) else 1

/-- The engine's phases. -/
noncomputable def engP (d n p k : ℕ) : ℝ :=
  if p = 0 then 0
  else if p ≤ n then 2 * k / (d : ℝ) ^ (n - p + 1)
  else -(2 * k) / (d : ℝ) ^ n

/-- The engine's environment at `n`. -/
noncomputable def engE (d n : ℕ) : Env := { dimv := engD d n, phase := engP d n, labs := engLabs n }

/-! ## The spec in closed form -/

theorem engD_dI (d : ℕ) {n t : ℕ} (h : t ≤ n) : engD d n (dI n t) = d ^ t := by
  unfold engD dI
  split_ifs with h0 h1 h1
  · omega
  · subst h0; rfl
  · congr 1; omega
  · omega

theorem edim_le {n k : ℕ} (hk : k < 2 * n + 2) : edim n k ≤ n := by
  unfold edim; split_ifs <;> omega

theorem lookup_shift (g : ℕ → ℕ) (N k b : ℕ) (hk : k < N) :
    ((range N).map fun j => (b + j, g j)).lookup (b + k) = some (g k) := by
  refine Fca.lookup_of_mem _ ?_ (mem_map.2 ⟨k, mem_range.2 hk, rfl⟩)
  rw [map_map]
  exact (nodup_range).map fun x y h => by simpa using h

theorem labDim_eng (d : ℕ) {n k : ℕ} (hk : k < 2 * n + 2) :
    labDim (engE d n) (rb n + k) = d ^ edim n k := by
  unfold labDim
  rw [show (engE d n).labs = engLabs n from rfl, engLabs,
    lookup_shift (fun k => dI n (edim n k)) _ _ _ hk]
  exact engD_dI d (edim_le hk)

/-- The value of the first `t` digits from label `c`, the first most significant. -/
def etv (d : ℕ) (σ : ℕ → ℕ) (c : ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => etv d σ c t * d + σ (c + t)

theorem etv_lt {d : ℕ} {σ : ℕ → ℕ} {c n : ℕ} (h : ∀ t < n, σ (c + t) < d) :
    ∀ t ≤ n, etv d σ c t < d ^ t
  | 0, _ => by simp [etv]
  | t + 1, ht => by
    have ih := etv_lt h t (by omega)
    have hdig := h t (by omega)
    simp only [etv, pow_succ]
    nlinarith

/-- The pinned value of the spec's label `rb n + k`. -/
def pv (d : ℕ) (σ : ℕ → ℕ) (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = 3 then 0 else if k = 1 then etv d σ n n else if k = 2 then etv d σ 0 n
  else if k < n + 3 then etv d σ n (k - 3) else etv d σ 0 (k - n - 2)

/-- `σ` with the spec's summed labels at their pinned values. -/
def tauE (d : ℕ) (σ : ℕ → ℕ) (n x : ℕ) : ℕ :=
  if rb n ≤ x ∧ x < rb n + (2 * n + 2) then pv d σ n (x - rb n) else σ x

theorem mem_engR_int {n x : ℕ} : x ∈ (engR n).int ↔ rb n ≤ x ∧ x < rb n + (2 * n + 2) := by
  simp only [engR, mem_map, mem_range]
  constructor
  · rintro ⟨k, hk, rfl⟩; omega
  · intro h; exact ⟨x - rb n, by omega, by omega⟩

theorem engR_int_nodup (n : ℕ) : (engR n).int.Nodup :=
  (nodup_range).map fun x y h => by simpa using h

theorem tauE_rb (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : tauE d σ n (rb n) = 0 := by
  simp [tauE, pv]

theorem tauE_bin (d : ℕ) (σ : ℕ → ℕ) {n t : ℕ} (ht : t < n) :
    tauE d σ n (bin n t) = etv d σ n t := by
  unfold tauE bin pv
  split_ifs <;> first | omega | (subst_vars; rfl) | (congr 1; omega)

theorem tauE_bout (d : ℕ) (σ : ℕ → ℕ) {n t : ℕ} (ht : t < n) :
    tauE d σ n (bout n t) = etv d σ n (t + 1) := by
  unfold tauE bout pv
  split_ifs <;> first | omega | (congr 1; omega)

theorem tauE_sv (d : ℕ) (σ : ℕ → ℕ) {n t : ℕ} (hn : 0 < n) (ht : t ≤ n) :
    tauE d σ n (sv n t) = etv d σ 0 t := by
  unfold tauE sv pv
  split_ifs <;> first | omega | (subst_vars; rfl) | (congr 1; omega)

theorem pv_lt {d : ℕ} {σ : ℕ → ℕ} {n k : ℕ} (hk : k < 2 * n + 2)
    (ha : ∀ t < n, σ (n + t) < d) (hb : ∀ t < n, σ (0 + t) < d) :
    pv d σ n k < d ^ edim n k := by
  unfold pv edim
  split_ifs <;> first
    | simp
    | exact etv_lt ha _ (by omega)
    | exact etv_lt hb _ (by omega)
    | omega

theorem mem_engR {n : ℕ} {x : Node} :
    x ∈ (engR n).nodes ↔ x = hN n ∨ x ∈ bN n ∨ x = xN n ∨ x ∈ sN n ∨ x = eN n := by
  unfold engR
  split_ifs <;> simp only [mem_append, mem_cons, not_mem_nil, or_false] <;> tauto

theorem prodVal_engR (E : Env) (n : ℕ) (τ : ℕ → ℕ) :
    prodVal E (engR n).nodes τ = val E τ (hN n) * ((bN n).map (val E τ)).prod * val E τ (xN n) *
      ((sN n).map (val E τ)).prod * val E τ (eN n) := by
  unfold engR prodVal
  split_ifs <;> simp only [map_append, prod_append, map_cons, map_nil, prod_cons, prod_nil] <;>
    ring

theorem engP_zero (d n : ℕ) : engP d n 0 = fun _ => 0 := by funext k; simp [engP]

theorem engD_zero (d : ℕ) {n : ℕ} (hn : 0 < n) : engD d n 0 = d := by simp [engD, hn]

theorem engR_support (d : ℕ) {n : ℕ} (hn : 0 < n) {τ : ℕ → ℕ}
    (h : prodVal (engE d n) (engR n).nodes τ ≠ 0) :
    τ (rb n) = 0 ∧ (∀ t < n, τ (bout n t) = etv d τ n (t + 1)) ∧
      ∀ t ≤ n, τ (sv n t) = etv d τ 0 t := by
  have hv := Fca.prodVal_ne h
  have h0 : τ (rb n) = 0 := by
    have h1 := hv _ (mem_engR.2 (Or.inl rfl))
    simp only [hN, val, map_cons, map_nil, append_nil] at h1
    rw [show (engE d n).dimv n = 1 by simp [engE, engD], show (engE d n).phase 0 = fun _ => 0
      from engP_zero d n, Fca.tZ_one _ _ (fun _ => rfl)] at h1
    simpa using h1
  have hB : ∀ t < n, τ (bout n t) = τ (bin n t) * d + τ (n + t) := by
    intro t ht
    have h1 := hv _ (mem_engR.2 (Or.inr (Or.inl (mem_map.2 ⟨t, mem_range.2 ht, rfl⟩))))
    simp only [val, map_cons, map_nil] at h1
    have e := Fca.tB_eq h1
    rwa [show (engE d n).dimv 0 = d from engD_zero d hn] at e
  have hS : ∀ t < n, τ (sv n (t + 1)) = τ (sv n t) * d + τ t := by
    intro t ht
    have h1 := hv _ (mem_engR.2 (Or.inr (Or.inr (Or.inr (Or.inl
      (mem_map.2 ⟨t, mem_range.2 ht, rfl⟩))))))
    simp only [val, map_cons, map_nil] at h1
    have e := Fca.tB_eq h1
    rwa [show (engE d n).dimv 0 = d from engD_zero d hn] at e
  have hE : τ (sv n 0) = 0 := by
    have h1 := hv _ (mem_engR.2 (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))
    simp only [eN, val, map_cons, map_nil, nil_append] at h1
    rw [show (engE d n).dimv n = 1 by simp [engE, engD], show (engE d n).phase 0 = fun _ => 0
      from engP_zero d n, Fca.tZ_one _ _ (fun _ => rfl)] at h1
    simpa [sv] using h1
  refine ⟨h0, fun t ht => ?_, fun t ht => ?_⟩
  · induction t with
    | zero => rw [hB 0 ht]; simp [etv, bin, h0]
    | succ t ih =>
      have e : bin n (t + 1) = bout n t := by
        simp only [bin, bout]; rw [if_neg (by omega), if_neg (by omega)]; omega
      rw [hB _ ht, e, ih (by omega)]
      rfl
  · induction t with
    | zero => simpa [etv] using hE
    | succ t ih => rw [hS t (by omega), ih (by omega)]; simp [etv]

theorem engP_shift (d n : ℕ) :
    engP d n (n + 1) = fun k : ℕ => -(2 * (k : ℝ)) / ((d ^ n : ℕ) : ℝ) := by
  funext k; simp [engP]


/-- The spec at `n ≥ 1` digits: the shift by `2` mod `d ^ n` of the input digits. -/
theorem eval_engR (d : ℕ) (hd : 0 < d) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ)
    (ha : ∀ t < n, σ (n + t) < d) (hb : ∀ t < n, σ t < d) :
    eval (engE d n) (engR n) σ =
      if etv d σ 0 n % d ^ n = (etv d σ n n + 2) % d ^ n then 1 else 0 := by
  have hb' : ∀ t < n, σ (0 + t) < d := fun t ht => by simpa using hb t ht
  have hrb : 2 * n ≤ rb n := by unfold rb; nlinarith
  have hoff : ∀ x, x < 2 * n → tauE d σ n x = σ x := fun x hx => by
    unfold tauE; rw [if_neg (by omega)]
  unfold eval
  rw [Fca.sumOver_single (labDim (engE d n)) _ (engR_int_nodup n) _ σ (tauE d σ n) ?h0 ?hr ?hs]
  case h0 =>
    intro x hx
    unfold tauE
    rw [if_neg fun h => hx (mem_engR_int.2 h)]
  case hr =>
    intro l hl
    obtain ⟨h1, h2⟩ := mem_engR_int.1 hl
    obtain ⟨k, rfl⟩ : ∃ k, l = rb n + k := ⟨l - rb n, by omega⟩
    rw [labDim_eng d (by omega)]
    unfold tauE
    rw [if_pos (by omega), show rb n + k - rb n = k by omega]
    exact pv_lt (by omega) ha hb'
  case hs =>
    intro τ hτ _ hne l hl
    obtain ⟨s0, s1, s2⟩ := engR_support d hn hne
    have dig : ∀ t, t < 2 * n → τ t = σ t := fun t ht =>
      hτ t fun h => by have := (mem_engR_int.1 h).1; omega
    have eA : ∀ t ≤ n, etv d τ n t = etv d σ n t := by
      intro t ht
      induction t with
      | zero => rfl
      | succ t ih => simp only [etv, ih (by omega), dig (n + t) (by omega)]
    have eB : ∀ t ≤ n, etv d τ 0 t = etv d σ 0 t := by
      intro t ht
      induction t with
      | zero => rfl
      | succ t ih => simp only [etv, ih (by omega), dig (0 + t) (by omega)]
    obtain ⟨h1, h2⟩ := mem_engR_int.1 hl
    obtain ⟨k, rfl⟩ : ∃ k, l = rb n + k := ⟨l - rb n, by omega⟩
    rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ (4 ≤ k ∧ k < n + 3) ∨ n + 3 ≤ k by omega) with
      rfl | rfl | rfl | rfl | hk | hk
    · rw [Nat.add_zero, s0, tauE_rb]
    · have e : rb n + 1 = bout n (n - 1) := by unfold bout; rw [if_pos (by omega)]
      rw [e, s1 _ (by omega), tauE_bout d σ (by omega), eA _ (by omega)]
    · have e : rb n + 2 = sv n n := by unfold sv; split_ifs <;> omega
      rw [e, s2 _ le_rfl, tauE_sv d σ hn le_rfl, eB _ le_rfl]
    · have e : rb n + 3 = sv n 0 := by unfold sv; split_ifs <;> omega
      rw [e, s2 _ (Nat.zero_le _), tauE_sv d σ hn (Nat.zero_le _), eB _ (Nat.zero_le _)]
    · have e : rb n + k = bout n (k - 4) := by unfold bout; split_ifs <;> omega
      rw [e, s1 _ (by omega), tauE_bout d σ (by omega), eA _ (by omega)]
    · have e : rb n + k = sv n (k - n - 2) := by unfold sv; split_ifs <;> omega
      rw [e, s2 _ (by omega), tauE_sv d σ hn (by omega), eB _ (by omega)]
  have eD1 : (engE d n).dimv n = 1 := by simp [engE, engD]
  have eP0 : (engE d n).phase 0 = fun _ => 0 := engP_zero d n
  have eD0 : (engE d n).dimv 0 = d := engD_zero d hn
  have hH : val (engE d n) (tauE d σ n) (hN n) = 1 := by
    simp only [hN, val, map_cons, map_nil, append_nil]
    rw [eD1, eP0, Fca.tZ_one _ _ (fun _ => rfl), tauE_rb]; simp
  have hE : val (engE d n) (tauE d σ n) (eN n) = 1 := by
    simp only [eN, val, map_cons, map_nil, nil_append]
    rw [eD1, eP0, Fca.tZ_one _ _ (fun _ => rfl)]
    have e : rb n + 3 = sv n 0 := by unfold sv; split_ifs <;> omega
    rw [e, tauE_sv d σ hn (Nat.zero_le _)]; simp [etv]
  have hX : val (engE d n) (tauE d σ n) (xN n) =
      if etv d σ 0 n % d ^ n = (etv d σ n n + 2) % d ^ n then 1 else 0 := by
    simp only [xN, val, map_cons, map_nil]
    have e1 : rb n + 1 = bout n (n - 1) := by unfold bout; rw [if_pos (by omega)]
    have e2 : rb n + 2 = sv n n := by unfold sv; split_ifs <;> omega
    have eX : (engE d n).dimv (n - 1) = d ^ n := by
      have := engD_dI d (le_refl n); rwa [dI, if_neg hn.ne'] at this
    rw [e1, e2, tauE_bout d σ (by omega), tauE_sv d σ hn le_rfl, eX,
      show (engE d n).phase (n + 1) = _ from engP_shift d n, Fca.tX_shift _ (pow_pos hd n),
      show n - 1 + 1 = n by omega]
  have hBs : ((bN n).map (val (engE d n) (tauE d σ n))).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨y, hy, rfl⟩ := mem_map.1 hx
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hy
    have ht := mem_range.1 ht
    simp only [val, map_cons, map_nil]
    rw [show (engE d n).dimv (dI n t) = d ^ t from engD_dI d ht.le, eD0, tauE_bout d σ ht,
      tauE_bin d σ ht, hoff (n + t) (by omega)]
    unfold tB
    rw [if_pos ⟨etv_lt ha t ht.le, ha t ht, rfl⟩]
  have hSs : ((sN n).map (val (engE d n) (tauE d σ n))).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨y, hy, rfl⟩ := mem_map.1 hx
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hy
    have ht := mem_range.1 ht
    simp only [val, map_cons, map_nil]
    rw [show (engE d n).dimv (dI n t) = d ^ t from engD_dI d ht.le, eD0, tauE_sv d σ hn ht.le,
      tauE_sv d σ hn (by omega), hoff t (by omega)]
    unfold tB
    rw [if_pos ⟨etv_lt hb' t ht.le, hb t ht, by simp [etv]⟩]
  rw [prodVal_engR, hH, hBs, hX, hSs, hE]
  ring

/-! ## The adder arithmetic over all copies -/

section arith

open Fca

/-- Copy `i`'s Fourier sum in the form of `Fca.sumC_lift`, without its pair scalar. -/
noncomputable def Gl (d : ℕ) (σ : ℕ → ℕ) (i : ℕ) : ℂ :=
  (d : ℂ)⁻¹ * ∑ x ∈ Finset.range d, (om (d ^ (i + 1)) ^ (dv d σ 0 (i + 1) + 2) *
    (om (d ^ (i + 1)))⁻¹ ^ dv d σ 1 (i + 1)) ^ x

theorem cs_ne (d n : ℕ) (hd : 0 < d) : cs d n ≠ 0 := by
  unfold cs
  rw [lprod_range]
  exact Finset.prod_ne_zero_iff.2 fun i _ => inv_ne_zero (pow_ne_zero _ (by exact_mod_cast hd.ne'))

theorem prod_Gl (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) : ∀ n, (∀ i < n, σ (sl i 0) < d) →
    (∀ i < n, σ (sl i 1) < d) → ∏ i ∈ Finset.range n, Gl d σ i = IndS d σ n
  | 0, _, _ => by simp [IndS, topv]
  | n + 1, ha, hb => by
    rw [Finset.prod_range_succ, prod_Gl d hd σ n (fun i hi => ha i (by omega))
      (fun i hi => hb i (by omega))]
    have h := arith d hd n σ ha hb
    rw [sumC_lift d hd] at h
    apply mul_left_cancel₀ (cs_ne d n hd)
    rw [← h, Gl]
    ring

theorem tP_pos (D a x : ℕ) : tP D 1 [a, x] = om D ^ (a * x) := by
  rw [om_eq_ex]; simp only [tP]; congr 1; push_cast; ring

theorem tP_neg (D o x : ℕ) : tP D (-1) [o, x] = (om D ^ (o * x))⁻¹ := by
  rw [om_eq_ex, ex_inv]; simp only [tP]; congr 1; push_cast; ring

/-- Copy `i`'s summand at the Fourier value `x`, its pairs as pair phases. -/
noncomputable def Ce (d : ℕ) (σ : ℕ → ℕ) (i x : ℕ) : ℂ :=
  fou d x (σ (sl i 0)) * ex (phase d (2 * i + 1) x) * (starRingEnd ℂ) (fou d (σ (sl i 1)) x) *
    ((range i).map fun k => tP (pD d i k) 1 [σ (sl k 0), x] * tP (pD d i k) (-1) [σ (sl k 1), x]).prod

theorem Ce_sum (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) (i : ℕ) :
    ∑ x ∈ Finset.range d, Ce d σ i x = Gl d σ i := by
  rw [Gl, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  have hp : ((range i).map fun k => tP (pD d i k) 1 [σ (sl k 0), x] *
      tP (pD d i k) (-1) [σ (sl k 1), x]).prod =
      om (d ^ (i + 1)) ^ (x * dv d σ 0 i) * (om (d ^ (i + 1)))⁻¹ ^ (x * dv d σ 1 i) := by
    have hf : ∀ k ∈ Finset.range i, tP (pD d i k) 1 [σ (sl k 0), x] *
        tP (pD d i k) (-1) [σ (sl k 1), x] =
        om (d ^ (i + 1)) ^ (x * (σ (sl k 0) * d ^ k)) *
          (om (d ^ (i + 1)))⁻¹ ^ (x * (σ (sl k 1) * d ^ k)) := by
      intro k hk
      have hk := Finset.mem_range.1 hk
      rw [tP_pos, tP_neg, om_lift d hd (n := i) (D := pD d i k) hk.le rfl, ← pow_mul, ← pow_mul,
        ← inv_pow,
        show d ^ k * (σ (sl k 0) * x) = x * (σ (sl k 0) * d ^ k) by ring,
        show d ^ k * (σ (sl k 1) * x) = x * (σ (sl k 1) * d ^ k) by ring]
    rw [lprod_range, Finset.prod_congr rfl hf, Finset.prod_mul_distrib,
      Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum, dv, dv, Finset.mul_sum,
      Finset.mul_sum]
  have hdn : d = d ^ (i - i + 1) := by simp
  have hc : (starRingEnd ℂ) (fou d (σ (sl i 1)) x) =
      (om (d ^ (i + 1)))⁻¹ ^ (x * (σ (sl i 1) * d ^ i)) / rt d := by
    unfold fou
    rw [map_div₀, map_pow, conj_om, Complex.conj_ofReal, om_lift d hd le_rfl hdn, ← inv_pow,
      ← pow_mul, show d ^ i * (σ (sl i 1) * x) = x * (σ (sl i 1) * d ^ i) by ring]
  rw [Ce, hp, hc, fou_lift d hd le_rfl hdn, ex_lift, dv_succ, dv_succ, mul_pow, ← pow_mul,
    ← pow_mul,
    show (dv d σ 0 i + σ (sl i 0) * d ^ i + 2) * x =
      x * (σ (sl i 0) * d ^ i) + 2 * x + x * dv d σ 0 i by ring,
    show (dv d σ 1 i + σ (sl i 1) * d ^ i) * x =
      x * (σ (sl i 1) * d ^ i) + x * dv d σ 1 i by ring,
    pow_add, pow_add, pow_add, ← rt_inv_sq d]
  ring

/-- Every copy's Fourier sum with its pair phases, multiplied: the shift factor. -/
theorem prod_Ce (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) (n : ℕ) (ha : ∀ i < n, σ (sl i 0) < d)
    (hb : ∀ i < n, σ (sl i 1) < d) :
    ∏ i ∈ Finset.range n, ∑ x ∈ Finset.range d, Ce d σ i x = IndS d σ n := by
  simp only [Ce_sum d hd]
  exact prod_Gl d hd σ n ha hb

end arith


/-! ## The engine's literal exports -/

/-- The engine export of the boxed side at n = 1. -/
def litL1 : Net := ⟨([⟨.F, [3], [2], 0, 0⟩, ⟨.Z, [5], [4], 0, 1⟩, ⟨.F, [7], [6], 0, 0⟩, ⟨.F, [8], [7], 0, 0⟩, ⟨.F, [9], [8], 0, 0⟩, ⟨.Z, [2], [1], 0, 0⟩, ⟨.Z, [4], [3], 0, 0⟩, ⟨.Z, [6], [5], 0, 0⟩, ⟨.Z, [0], [9], 0, 0⟩] : List Node), [2, 3, 4, 5, 6, 7, 8, 9], [0, 1]⟩

/-- The engine export of the spec at n = 1. -/
def litR1 : Net := ⟨([⟨.Z, [10], [], 1, 0⟩, ⟨.B, [11], [10, 1], 1, 0⟩, ⟨.X, [12], [11], 0, 2⟩, ⟨.S, [13, 0], [12], 1, 0⟩, ⟨.Z, [], [13], 1, 0⟩] : List Node), [10, 11, 12, 13], [0, 1]⟩

/-- The engine export of the boxed side at n = 2. -/
def litL2 : Net := ⟨([⟨.F, [5], [4], 0, 0⟩, ⟨.Z, [7], [6], 0, 1⟩, ⟨.F, [9], [8], 0, 0⟩, ⟨.F, [10], [9], 0, 0⟩, ⟨.F, [11], [10], 0, 0⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.Z, [15], [14], 0, 2⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.F, [18], [17], 0, 0⟩, ⟨.F, [19], [18], 0, 0⟩, ⟨.Z, [4], [2], 0, 0⟩, ⟨.Z, [12], [3, 20], 0, 0⟩, ⟨.Z, [6, 21], [5], 0, 0⟩, ⟨.Z, [14], [13], 0, 0⟩, ⟨.Z, [8, 22], [7], 0, 0⟩, ⟨.Z, [16], [15], 0, 0⟩, ⟨.Z, [0], [11], 0, 0⟩, ⟨.Z, [1], [19, 23], 0, 0⟩, ⟨.ZP 0 0 (1), [20], [21], 1, 0⟩, ⟨.ZP 0 0 (-1), [23], [22], 1, 0⟩] : List Node), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [0, 1, 2, 3]⟩

/-- The engine export of the spec at n = 2. -/
def litR2 : Net := ⟨([⟨.Z, [24], [], 2, 0⟩, ⟨.X, [26], [25], 1, 3⟩, ⟨.Z, [], [27], 2, 0⟩, ⟨.B, [28], [24, 2], 2, 0⟩, ⟨.B, [25], [28, 3], 0, 0⟩, ⟨.S, [27, 0], [29], 2, 0⟩, ⟨.S, [29, 1], [26], 0, 0⟩] : List Node), [24, 25, 26, 27, 28, 29], [0, 1, 2, 3]⟩

/-- The engine export of the boxed side at n = 3. -/
def litL3 : Net := ⟨([⟨.F, [7], [6], 0, 0⟩, ⟨.Z, [9], [8], 0, 1⟩, ⟨.F, [11], [10], 0, 0⟩, ⟨.F, [12], [11], 0, 0⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.Z, [17], [16], 0, 2⟩, ⟨.F, [19], [18], 0, 0⟩, ⟨.F, [20], [19], 0, 0⟩, ⟨.F, [21], [20], 0, 0⟩, ⟨.F, [23], [22], 0, 0⟩, ⟨.Z, [25], [24], 0, 3⟩, ⟨.F, [27], [26], 0, 0⟩, ⟨.F, [28], [27], 0, 0⟩, ⟨.F, [29], [28], 0, 0⟩, ⟨.Z, [6], [3], 0, 0⟩, ⟨.Z, [14], [4, 30], 0, 0⟩, ⟨.Z, [22], [5, 31, 32], 0, 0⟩, ⟨.Z, [8, 33, 34], [7], 0, 0⟩, ⟨.Z, [16, 35], [15], 0, 0⟩, ⟨.Z, [24], [23], 0, 0⟩, ⟨.Z, [10, 36, 37], [9], 0, 0⟩, ⟨.Z, [18, 38], [17], 0, 0⟩, ⟨.Z, [26], [25], 0, 0⟩, ⟨.Z, [0], [13], 0, 0⟩, ⟨.Z, [1], [21, 39], 0, 0⟩, ⟨.Z, [2], [29, 40, 41], 0, 0⟩, ⟨.ZP 0 0 (1), [30], [33], 1, 0⟩, ⟨.ZP 0 0 (1), [31], [34], 2, 0⟩, ⟨.ZP 0 0 (1), [32], [35], 1, 0⟩, ⟨.ZP 0 0 (-1), [39], [36], 1, 0⟩, ⟨.ZP 0 0 (-1), [40], [37], 2, 0⟩, ⟨.ZP 0 0 (-1), [41], [38], 1, 0⟩] : List Node), [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [0, 1, 2, 3, 4, 5]⟩

/-- The engine export of the spec at n = 3. -/
def litR3 : Net := ⟨([⟨.Z, [42], [], 3, 0⟩, ⟨.X, [44], [43], 2, 4⟩, ⟨.Z, [], [45], 3, 0⟩, ⟨.B, [46], [42, 3], 3, 0⟩, ⟨.B, [47], [46, 4], 0, 0⟩, ⟨.B, [43], [47, 5], 1, 0⟩, ⟨.S, [45, 0], [48], 3, 0⟩, ⟨.S, [48, 1], [49], 0, 0⟩, ⟨.S, [49, 2], [44], 1, 0⟩] : List Node), [42, 43, 44, 45, 46, 47, 48, 49], [0, 1, 2, 3, 4, 5]⟩

/-- The engine export of the boxed side at n = 4. -/
def litL4 : Net := ⟨([⟨.F, [9], [8], 0, 0⟩, ⟨.Z, [11], [10], 0, 1⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.F, [14], [13], 0, 0⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.Z, [19], [18], 0, 2⟩, ⟨.F, [21], [20], 0, 0⟩, ⟨.F, [22], [21], 0, 0⟩, ⟨.F, [23], [22], 0, 0⟩, ⟨.F, [25], [24], 0, 0⟩, ⟨.Z, [27], [26], 0, 3⟩, ⟨.F, [29], [28], 0, 0⟩, ⟨.F, [30], [29], 0, 0⟩, ⟨.F, [31], [30], 0, 0⟩, ⟨.F, [33], [32], 0, 0⟩, ⟨.Z, [35], [34], 0, 4⟩, ⟨.F, [37], [36], 0, 0⟩, ⟨.F, [38], [37], 0, 0⟩, ⟨.F, [39], [38], 0, 0⟩, ⟨.Z, [8], [4], 0, 0⟩, ⟨.Z, [16], [5, 40], 0, 0⟩, ⟨.Z, [24], [6, 41, 42], 0, 0⟩, ⟨.Z, [32], [7, 43, 44, 45], 0, 0⟩, ⟨.Z, [10, 46, 47, 48], [9], 0, 0⟩, ⟨.Z, [18, 49, 50], [17], 0, 0⟩, ⟨.Z, [26, 51], [25], 0, 0⟩, ⟨.Z, [34], [33], 0, 0⟩, ⟨.Z, [12, 52, 53, 54], [11], 0, 0⟩, ⟨.Z, [20, 55, 56], [19], 0, 0⟩, ⟨.Z, [28, 57], [27], 0, 0⟩, ⟨.Z, [36], [35], 0, 0⟩, ⟨.Z, [0], [15], 0, 0⟩, ⟨.Z, [1], [23, 58], 0, 0⟩, ⟨.Z, [2], [31, 59, 60], 0, 0⟩, ⟨.Z, [3], [39, 61, 62, 63], 0, 0⟩, ⟨.ZP 0 0 (1), [40], [46], 1, 0⟩, ⟨.ZP 0 0 (1), [41], [47], 2, 0⟩, ⟨.ZP 0 0 (1), [43], [48], 3, 0⟩, ⟨.ZP 0 0 (1), [42], [49], 1, 0⟩, ⟨.ZP 0 0 (1), [44], [50], 2, 0⟩, ⟨.ZP 0 0 (1), [45], [51], 1, 0⟩, ⟨.ZP 0 0 (-1), [58], [52], 1, 0⟩, ⟨.ZP 0 0 (-1), [59], [53], 2, 0⟩, ⟨.ZP 0 0 (-1), [61], [54], 3, 0⟩, ⟨.ZP 0 0 (-1), [60], [55], 1, 0⟩, ⟨.ZP 0 0 (-1), [62], [56], 2, 0⟩, ⟨.ZP 0 0 (-1), [63], [57], 1, 0⟩] : List Node), [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], [0, 1, 2, 3, 4, 5, 6, 7]⟩

/-- The engine export of the spec at n = 4. -/
def litR4 : Net := ⟨([⟨.Z, [64], [], 4, 0⟩, ⟨.X, [66], [65], 3, 5⟩, ⟨.Z, [], [67], 4, 0⟩, ⟨.B, [68], [64, 4], 4, 0⟩, ⟨.B, [69], [68, 5], 0, 0⟩, ⟨.B, [70], [69, 6], 1, 0⟩, ⟨.B, [65], [70, 7], 2, 0⟩, ⟨.S, [67, 0], [71], 4, 0⟩, ⟨.S, [71, 1], [72], 0, 0⟩, ⟨.S, [72, 2], [73], 1, 0⟩, ⟨.S, [73, 3], [66], 2, 0⟩] : List Node), [64, 65, 66, 67, 68, 69, 70, 71, 72, 73], [0, 1, 2, 3, 4, 5, 6, 7]⟩

/-- The engine export of the boxed side at n = 5. -/
def litL5 : Net := ⟨([⟨.F, [11], [10], 0, 0⟩, ⟨.Z, [13], [12], 0, 1⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.F, [16], [15], 0, 0⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.F, [19], [18], 0, 0⟩, ⟨.Z, [21], [20], 0, 2⟩, ⟨.F, [23], [22], 0, 0⟩, ⟨.F, [24], [23], 0, 0⟩, ⟨.F, [25], [24], 0, 0⟩, ⟨.F, [27], [26], 0, 0⟩, ⟨.Z, [29], [28], 0, 3⟩, ⟨.F, [31], [30], 0, 0⟩, ⟨.F, [32], [31], 0, 0⟩, ⟨.F, [33], [32], 0, 0⟩, ⟨.F, [35], [34], 0, 0⟩, ⟨.Z, [37], [36], 0, 4⟩, ⟨.F, [39], [38], 0, 0⟩, ⟨.F, [40], [39], 0, 0⟩, ⟨.F, [41], [40], 0, 0⟩, ⟨.F, [43], [42], 0, 0⟩, ⟨.Z, [45], [44], 0, 5⟩, ⟨.F, [47], [46], 0, 0⟩, ⟨.F, [48], [47], 0, 0⟩, ⟨.F, [49], [48], 0, 0⟩, ⟨.Z, [10], [5], 0, 0⟩, ⟨.Z, [18], [6, 50], 0, 0⟩, ⟨.Z, [26], [7, 51, 52], 0, 0⟩, ⟨.Z, [34], [8, 53, 54, 55], 0, 0⟩, ⟨.Z, [42], [9, 56, 57, 58, 59], 0, 0⟩, ⟨.Z, [12, 60, 61, 62, 63], [11], 0, 0⟩, ⟨.Z, [20, 64, 65, 66], [19], 0, 0⟩, ⟨.Z, [28, 67, 68], [27], 0, 0⟩, ⟨.Z, [36, 69], [35], 0, 0⟩, ⟨.Z, [44], [43], 0, 0⟩, ⟨.Z, [14, 70, 71, 72, 73], [13], 0, 0⟩, ⟨.Z, [22, 74, 75, 76], [21], 0, 0⟩, ⟨.Z, [30, 77, 78], [29], 0, 0⟩, ⟨.Z, [38, 79], [37], 0, 0⟩, ⟨.Z, [46], [45], 0, 0⟩, ⟨.Z, [0], [17], 0, 0⟩, ⟨.Z, [1], [25, 80], 0, 0⟩, ⟨.Z, [2], [33, 81, 82], 0, 0⟩, ⟨.Z, [3], [41, 83, 84, 85], 0, 0⟩, ⟨.Z, [4], [49, 86, 87, 88, 89], 0, 0⟩, ⟨.ZP 0 0 (1), [50], [60], 1, 0⟩, ⟨.ZP 0 0 (1), [51], [61], 2, 0⟩, ⟨.ZP 0 0 (1), [53], [62], 3, 0⟩, ⟨.ZP 0 0 (1), [56], [63], 4, 0⟩, ⟨.ZP 0 0 (1), [52], [64], 1, 0⟩, ⟨.ZP 0 0 (1), [54], [65], 2, 0⟩, ⟨.ZP 0 0 (1), [57], [66], 3, 0⟩, ⟨.ZP 0 0 (1), [55], [67], 1, 0⟩, ⟨.ZP 0 0 (1), [58], [68], 2, 0⟩, ⟨.ZP 0 0 (1), [59], [69], 1, 0⟩, ⟨.ZP 0 0 (-1), [80], [70], 1, 0⟩, ⟨.ZP 0 0 (-1), [81], [71], 2, 0⟩, ⟨.ZP 0 0 (-1), [83], [72], 3, 0⟩, ⟨.ZP 0 0 (-1), [86], [73], 4, 0⟩, ⟨.ZP 0 0 (-1), [82], [74], 1, 0⟩, ⟨.ZP 0 0 (-1), [84], [75], 2, 0⟩, ⟨.ZP 0 0 (-1), [87], [76], 3, 0⟩, ⟨.ZP 0 0 (-1), [85], [77], 1, 0⟩, ⟨.ZP 0 0 (-1), [88], [78], 2, 0⟩, ⟨.ZP 0 0 (-1), [89], [79], 1, 0⟩] : List Node), [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]⟩

/-- The engine export of the spec at n = 5. -/
def litR5 : Net := ⟨([⟨.Z, [90], [], 5, 0⟩, ⟨.X, [92], [91], 4, 6⟩, ⟨.Z, [], [93], 5, 0⟩, ⟨.B, [94], [90, 5], 5, 0⟩, ⟨.B, [95], [94, 6], 0, 0⟩, ⟨.B, [96], [95, 7], 1, 0⟩, ⟨.B, [97], [96, 8], 2, 0⟩, ⟨.B, [91], [97, 9], 3, 0⟩, ⟨.S, [93, 0], [98], 5, 0⟩, ⟨.S, [98, 1], [99], 0, 0⟩, ⟨.S, [99, 2], [100], 1, 0⟩, ⟨.S, [100, 3], [101], 2, 0⟩, ⟨.S, [101, 4], [92], 3, 0⟩] : List Node), [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]⟩

theorem engL_lit1 : engL 1 = litL1 := by decide +kernel

theorem engR_lit1 : engR 1 = litR1 := by decide +kernel

theorem engL_lit2 : engL 2 = litL2 := by decide +kernel

theorem engR_lit2 : engR 2 = litR2 := by decide +kernel

theorem engL_lit3 : engL 3 = litL3 := by decide +kernel

theorem engR_lit3 : engR 3 = litR3 := by decide +kernel

theorem engL_lit4 : engL 4 = litL4 := by decide +kernel

theorem engR_lit4 : engR 4 = litR4 := by decide +kernel

theorem engL_lit5 : engL 5 = litL5 := by decide +kernel

theorem engR_lit5 : engR 5 = litR5 := by decide +kernel

theorem engLabs_lit1 : (engLabs 1).filter (fun p => p.2 ≠ 0) = [(10, 1), (13, 1)] := by
  decide +kernel

theorem engLabs_lit2 : (engLabs 2).filter (fun p => p.2 ≠ 0) = [(24, 2), (25, 1), (26, 1), (27, 2)] := by
  decide +kernel

theorem engLabs_lit3 : (engLabs 3).filter (fun p => p.2 ≠ 0) = [(42, 3), (43, 2), (44, 2), (45, 3), (47, 1), (49, 1)] := by
  decide +kernel

theorem engLabs_lit4 : (engLabs 4).filter (fun p => p.2 ≠ 0) = [(64, 4), (65, 3), (66, 3), (67, 4), (69, 1), (70, 2), (72, 1), (73, 2)] := by
  decide +kernel

theorem engLabs_lit5 : (engLabs 5).filter (fun p => p.2 ≠ 0) = [(90, 5), (91, 4), (92, 4), (93, 5), (95, 1), (96, 2), (97, 3), (99, 1), (100, 2), (101, 3)] := by
  decide +kernel

end FcaEng

end ArchytasZX
