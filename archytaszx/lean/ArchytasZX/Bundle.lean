/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.WZ

/-!
# Bundles

A family of diagrams indexed by a multiplicity `k` is one network: a bundle label of base
dimension `d` carries a value below `d ^ k`, read as `k` digits. `Spec.at S k` is the
environment at `k`; `Net.split` peels the last copy into fresh ordinary labels `b + H`.
-/

namespace ArchytasZX

open Complex Finset

theorem digit_lt (d i v : ℕ) (hd : 0 < d) : digit d i v < d := Nat.mod_lt _ hd

theorem digit_low (d k i v y : ℕ) (hd : 0 < d) (hi : i < k) (hv : v < d ^ k) :
    digit d i (v + d ^ k * y) = digit d i v := by
  unfold digit
  have hk : d ^ k = d ^ i * d ^ (k - i) := by rw [← pow_add]; congr 1; omega
  rw [hk, mul_assoc, Nat.add_mul_div_left _ _ (pow_pos hd i)]
  have : d ^ (k - i) = d * d ^ (k - i - 1) := by
    rw [← pow_succ']; congr 1; omega
  rw [this, mul_assoc, Nat.add_mul_mod_self_left]

theorem digit_top (d k v y : ℕ) (hd : 0 < d) (hv : v < d ^ k) (hy : y < d) :
    digit d k (v + d ^ k * y) = y := by
  unfold digit
  rw [Nat.add_mul_div_left _ _ (pow_pos hd k), Nat.div_eq_of_lt hv, zero_add,
    Nat.mod_eq_of_lt hy]

theorem digs_succ (d k v y : ℕ) (hd : 0 < d) (hv : v < d ^ k) (hy : y < d) :
    digs d (k + 1) (v + d ^ k * y) = digs d k v ++ [y] := by
  unfold digs
  rw [List.range_succ, List.map_append, List.map_singleton, digit_top d k v y hd hv hy]
  congr 1
  exact List.map_congr_left fun i hi => digit_low d k i v y hd (List.mem_range.1 hi) hv

/-- A sum below `a * b` as a double sum over the low and high parts. -/
theorem sum_range_mul (a b : ℕ) (f : ℕ → ℂ) :
    ∑ v ∈ range (a * b), f v = ∑ v ∈ range a, ∑ y ∈ range b, f (v + a * y) := by
  induction b with
  | zero => simp
  | succ b ih =>
    rw [mul_add, mul_one, Finset.sum_range_add, ih, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [Finset.sum_range_succ, add_comm (a * b) v]

/-! ## Families -/

/-- A family's data: dimension values, each bundle dimension index with the index it bundles
and its multiplicity, phases, and label dimension indices. -/
structure Fam where
  dims : ℕ → ℕ
  bund : List (ℕ × ℕ × ℕ)
  phase : ℕ → ℕ → ℝ
  labs : List (ℕ × ℕ)

/-- The multiplicities a dimension index bundles along, outermost first, within `n` steps. -/
def Fam.chainF (F : Fam) : ℕ → ℕ → List ℕ
  | 0, _ => []
  | n + 1, i =>
    match F.bund.lookup i with
    | some (j, m) => m :: F.chainF n j
    | Option.none => []

/-- The plain dimension index under a bundle index, within `n` steps. -/
def Fam.rootF (F : Fam) : ℕ → ℕ → ℕ
  | 0, i => i
  | n + 1, i =>
    match F.bund.lookup i with
    | some (j, _) => F.rootF n j
    | Option.none => i

/-- The multiplicities a dimension index bundles along, outermost first. -/
def Fam.chain (F : Fam) (i : ℕ) : List ℕ := F.chainF (i + 1) i

/-- The plain dimension index under a dimension index. -/
def Fam.root (F : Fam) (i : ℕ) : ℕ := F.rootF (i + 1) i

/-- Every bundle index bundles a smaller index. -/
def Fam.wf (F : Fam) : Bool := F.bund.all fun p => decide (p.2.1 < p.1)

/-- A dimension index's value under the multiplicities `ks`. -/
def Fam.dimv (F : Fam) (ks : ℕ → ℕ) (i : ℕ) : ℕ :=
  match F.bund.lookup i with
  | Option.none => F.dims i
  | some _ => F.dims (F.root i) ^ ((F.chain i).map ks).prod

/-- The environment at the multiplicities `ks`. -/
noncomputable def Fam.env (F : Fam) (ks : ℕ → ℕ) : Env :=
  { dimv := F.dimv ks, phase := F.phase, labs := F.labs, mult := ks }

theorem lookup_mem {α : Type*} {l : List (ℕ × α)} {x : ℕ} {i : α} (h : l.lookup x = some i) :
    (x, i) ∈ l := by
  induction l with
  | nil => simp at h
  | cons p l ih =>
    obtain ⟨a, b⟩ := p
    by_cases hx : x = a
    · subst hx; simp [List.lookup] at h; subst h; simp
    · have hb : (x == a) = false := by simpa using hx
      have : l.lookup x = some i := by simpa [List.lookup, hb] using h
      exact List.mem_cons_of_mem _ (ih this)

theorem Fam.lt_of_lookup (F : Fam) (hwf : F.wf = true) {i j m : ℕ}
    (h : F.bund.lookup i = some (j, m)) : j < i := by
  have := List.all_eq_true.1 hwf _ (lookup_mem h)
  simpa using this

theorem Fam.chainF_stable (F : Fam) (hwf : F.wf = true) :
    ∀ i n, i + 1 ≤ n → F.chainF n i = F.chain i ∧ F.rootF n i = F.root i := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    intro n hn
    obtain ⟨n, rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
    simp only [Fam.chain, Fam.root, Fam.chainF, Fam.rootF]
    cases h : F.bund.lookup i with
    | none => exact ⟨rfl, rfl⟩
    | some p =>
      obtain ⟨j, m⟩ := p
      have hj := F.lt_of_lookup hwf h
      obtain ⟨a1, b1⟩ := ih j hj n (by omega)
      obtain ⟨a2, b2⟩ := ih j hj i (by omega)
      exact ⟨by simp [a1, a2], by simp [b1, b2]⟩

theorem Fam.chain_some (F : Fam) (hwf : F.wf = true) {i j m : ℕ}
    (h : F.bund.lookup i = some (j, m)) : F.chain i = m :: F.chain j ∧ F.root i = F.root j := by
  have hj := F.lt_of_lookup hwf h
  obtain ⟨a, b⟩ := F.chainF_stable hwf j i (by omega)
  exact ⟨by simp [Fam.chain, Fam.chainF, h, a], by simp [Fam.root, Fam.rootF, h, b]⟩

theorem Fam.chain_none (F : Fam) {i : ℕ} (h : F.bund.lookup i = Option.none) :
    F.chain i = [] ∧ F.root i = i := by
  simp [Fam.chain, Fam.root, Fam.chainF, Fam.rootF, h]

theorem Fam.dimv_eq (F : Fam) (ks : ℕ → ℕ) (i : ℕ) :
    F.dimv ks i = F.dims (F.root i) ^ ((F.chain i).map ks).prod := by
  unfold Fam.dimv
  split
  · rename_i h; rw [(F.chain_none h).1, (F.chain_none h).2]; simp
  · rfl

theorem Fam.dimv_congr (F : Fam) (ks ks' : ℕ → ℕ) (i : ℕ)
    (h : ∀ x ∈ F.chain i, ks x = ks' x) : F.dimv ks i = F.dimv ks' i := by
  simp only [Fam.dimv_eq]
  congr 2
  exact List.map_congr_left h

theorem Fam.dimv_pos (F : Fam) (ks : ℕ → ℕ) (i : ℕ) (h : 0 < F.dims (F.root i)) :
    0 < F.dimv ks i := by rw [Fam.dimv_eq]; exact pow_pos h _

theorem Fam.env_dimv_pos (F : Fam) (ks : ℕ → ℕ) (i : ℕ) (h : 0 < F.dims (F.root i)) :
    0 < (F.env ks).dimv i := F.dimv_pos ks i h

theorem Fam.dimv_of (F : Fam) (ks : ℕ → ℕ) (i dz : ℕ) (ms : List ℕ) (hc : F.chain i = ms)
    (hr : F.root i = dz) (hd : F.chain dz = [] ∧ F.root dz = dz) :
    F.dimv ks i = F.dimv ks dz ^ (ms.map ks).prod := by
  simp [Fam.dimv_eq, hc, hr, hd.1, hd.2]

theorem Fam.env_dimv_of (F : Fam) (ks : ℕ → ℕ) (i dz : ℕ) (ms : List ℕ) (hc : F.chain i = ms)
    (hr : F.root i = dz) (hd : F.chain dz = [] ∧ F.root dz = dz) :
    (F.env ks).dimv i = (F.env ks).dimv dz ^ (ms.map (F.env ks).mult).prod :=
  F.dimv_of ks i dz ms hc hr hd

/-- A family along its multiplicity `m`, the others held at `ks`. -/
structure Spec where
  F : Fam
  m : ℕ
  ks : ℕ → ℕ

/-- The environment at `m = k`. -/
noncomputable def Spec.at (S : Spec) (k : ℕ) : Env := S.F.env (Function.update S.ks S.m k)

/-- Whether the dimension index `i` bundles along `m` outermost. -/
def Spec.cur (S : Spec) (i : ℕ) : Bool :=
  match S.F.bund.lookup i with
  | some (_, m) => m == S.m
  | Option.none => false

/-- Whether `m` bundles the dimension index `i` at most outermost. -/
def Spec.good (S : Spec) (i : ℕ) : Bool := decide (S.m ∉ (S.F.chain i).tail)

/-- Whether a label is a bundle along `m`. -/
def Spec.isB (S : Spec) (l : ℕ) : Bool := S.cur (labIdx S.F.labs l)

/-- The dimension index a bundle label bundles. -/
def Spec.baseIdx (S : Spec) (l : ℕ) : ℕ :=
  match S.F.bund.lookup (labIdx S.F.labs l) with
  | some (j, _) => j
  | Option.none => 0

/-- The value of the dimension a bundle label bundles. -/
def Spec.base (S : Spec) (l : ℕ) : ℕ := S.F.dimv S.ks (S.baseIdx l)

theorem Spec.cur_some (S : Spec) (i : ℕ) (h : S.cur i = true) :
    ∃ j, S.F.bund.lookup i = some (j, S.m) := by
  simp only [Spec.cur] at h
  split at h
  · rename_i j m hl; simp only [beq_iff_eq] at h; subst h; exact ⟨j, hl⟩
  · simp at h

theorem Spec.not_mem_of_good (S : Spec) (hwf : S.F.wf = true) (i : ℕ) (hc : S.cur i = false)
    (hg : S.good i = true) : S.m ∉ S.F.chain i := by
  simp only [Spec.good, decide_eq_true_eq] at hg
  simp only [Spec.cur] at hc
  cases h : S.F.bund.lookup i with
  | none => rw [(S.F.chain_none h).1]; simp
  | some p =>
    obtain ⟨j, m'⟩ := p
    rw [h] at hc
    simp only [beq_eq_false_iff_ne, ne_eq] at hc
    rw [(S.F.chain_some hwf h).1] at hg ⊢
    simp only [List.tail_cons] at hg
    simp only [List.mem_cons, not_or]
    exact ⟨Ne.symm hc, hg⟩

theorem Spec.dimv_fixed (S : Spec) (hwf : S.F.wf = true) (k : ℕ) (i : ℕ)
    (h : S.m ∉ S.F.chain i) : (S.at k).dimv i = S.F.dimv S.ks i := by
  simp only [Spec.at, Fam.env]
  exact S.F.dimv_congr _ _ _ fun x hx => Function.update_of_ne (by rintro rfl; exact h hx) _ _

theorem Spec.chain_cur (S : Spec) (hwf : S.F.wf = true) (i j : ℕ)
    (hl : S.F.bund.lookup i = some (j, S.m)) (hg : S.good i = true) :
    S.F.chain i = S.m :: S.F.chain j ∧ S.F.root i = S.F.root j ∧ S.m ∉ S.F.chain j := by
  obtain ⟨a, b⟩ := S.F.chain_some hwf hl
  refine ⟨a, b, ?_⟩
  simp only [Spec.good, a, List.tail_cons, decide_eq_true_eq] at hg
  exact hg

theorem Spec.dimv_cur (S : Spec) (hwf : S.F.wf = true) (k i j : ℕ)
    (hl : S.F.bund.lookup i = some (j, S.m)) (hg : S.good i = true) :
    (S.at k).dimv i = S.F.dimv S.ks j ^ k := by
  obtain ⟨a, b, c⟩ := S.chain_cur hwf i j hl hg
  have hj := S.dimv_fixed hwf k j c
  simp only [Spec.at, Fam.env] at hj ⊢
  rw [← hj]
  simp only [Fam.dimv_eq, a, b, List.map_cons, Function.update_self, List.prod_cons, ← pow_mul,
    mul_comm]

theorem Spec.mult_ne (S : Spec) (k k' j : ℕ) (h : j ≠ S.m) : (S.at k).mult j = (S.at k').mult j := by
  simp [Spec.at, Fam.env, Function.update_of_ne h]

theorem Spec.mult_self (S : Spec) (k : ℕ) : (S.at k).mult S.m = k := by
  simp [Spec.at, Fam.env]

/-- The last copy along `m` of the bundled legs `tl` (chains `cs`) peeled into `b + H`. -/
def peel (m H : ℕ) (tl : List ℕ) (cs : List (List ℕ)) : List ℕ × List (List ℕ) :=
  let ps := (List.zip tl cs).filter fun p => p.2.head? == some m
  (ps.map (·.1 + H), ps.map (·.2.tail))

/-- The legs of `ls` bundled by the last `cs.length`. -/
def tailOf (ls : List ℕ) (cs : List (List ℕ)) : List ℕ := ls.drop (ls.length - cs.length)

/-- The last copy along `m` of a node peeled into the labels `b + H`. -/
def Node.split (m H : ℕ) (n : Node) : List Node :=
  match n.kind with
  | .ZB cs =>
    let p := peel m H (tailOf n.ins cs) cs
    [{ n with kind := .ZB (cs ++ p.2), ins := n.ins ++ p.1 }]
  | .XB co ci =>
    let p := peel m H (tailOf n.outs co) co
    let q := peel m H (tailOf n.ins ci) ci
    [{ n with kind := .XB (co ++ p.2) (ci ++ q.2), outs := n.outs ++ p.1, ins := n.ins ++ q.1 }]
  | .WB cs =>
    let p := peel m H (tailOf n.outs cs) cs
    [{ n with kind := .WB (cs ++ p.2), outs := n.outs ++ p.1 }]
  | .Bun m' b => if m' = m then [n, ⟨b, n.outs.map (· + H), n.ins.map (· + H), n.dim, n.ph⟩] else [n]
  | _ => [n]

/-- A label list with each bundle followed by its peeled label. -/
def Spec.expand (S : Spec) (H : ℕ) (ls : List ℕ) : List ℕ :=
  ls.flatMap fun l => if S.isB l then [l, l + H] else [l]

/-- The network with its last copy along `m` peeled into the labels `b + H`. -/
def Net.split (S : Spec) (H : ℕ) (N : Net) : Net :=
  ⟨N.nodes.flatMap (Node.split S.m H), S.expand H N.int, S.expand H N.bnd⟩

/-- The bundles and peeled labels of `τ` put back together. -/
def Spec.combine (S : Spec) (H k : ℕ) (τ : ℕ → ℕ) (x : ℕ) : ℕ :=
  if S.isB x then τ x + S.base x ^ k * τ (x + H) else τ x

/-- The multiplicities a kind reads. -/
def Kind.ms : Kind → List ℕ
  | .ZB cs => cs.flatten
  | .XB co ci => co.flatten ++ ci.flatten
  | .WB cs => cs.flatten
  | .Bun m b => m :: b.ms
  | _ => []

/-- The generators a `Bun` node may copy. -/
def copyKind : Kind → Bool
  | .B | .S => false
  | .Bun _ b => copyKind b
  | _ => true

/-- Each leg's label has the chain `c` over `dim`. -/
def Spec.legsOK (S : Spec) (dim : ℕ) (ls : List ℕ) (cs : List (List ℕ)) : Bool :=
  ls.length == cs.length && (List.zip ls cs).all fun p =>
    S.F.chain (labIdx S.F.labs p.1) == p.2 && S.F.root (labIdx S.F.labs p.1) == dim

/-- Each chain has `m` at most outermost. -/
def Spec.chainsOK (S : Spec) (cs : List (List ℕ)) : Bool := cs.all fun c => decide (S.m ∉ c.tail)

/-- The node's legs are labelled as its kind bundles them, `m` at most outermost. -/
def Spec.nodeOK (S : Spec) (n : Node) : Bool :=
  (S.F.bund.lookup n.dim).isNone &&
  match n.kind with
  | .B | .S => (S.F.bund.lookup n.ph).isNone &&
      (n.outs ++ n.ins).all fun l => (S.F.bund.lookup (labIdx S.F.labs l)).isNone
  | .Bun m b => copyKind b && decide (S.m ∉ b.ms) &&
      S.legsOK n.dim n.outs ((Kind.Bun m b).outCs n.outs.length) &&
      S.legsOK n.dim n.ins ((Kind.Bun m b).inCs n.ins.length)
  | k => S.chainsOK (k.outCs n.outs.length) && S.chainsOK (k.inCs n.ins.length) &&
      S.legsOK n.dim n.outs (k.outCs n.outs.length) &&
      S.legsOK n.dim n.ins (k.inCs n.ins.length)

/-- The network can be split along `m` at the offset `H`. -/
def Spec.splitOK (S : Spec) (H : ℕ) (N : Net) : Bool :=
  decide (0 < H) && S.F.wf && (S.F.bund.lookup 0).isNone &&
  (labelsOf N.nodes).all (fun l => decide (l ∈ N.int ∨ l ∈ N.bnd)) &&
  (labelsOf N.nodes ++ N.int ++ N.bnd).all (fun l => S.good (labIdx S.F.labs l)) &&
  (labelsOf N.nodes ++ N.int ++ N.bnd).all (fun l => decide (l < H)) &&
  decide N.int.Nodup &&
  (labelsOf N.nodes ++ N.int ++ N.bnd).all (fun l => !S.isB l ||
    (labIdx S.F.labs (l + H) == S.baseIdx l)) &&
  N.nodes.all S.nodeOK

/-! ## Splitting sums -/

/-- The data the splitting lemmas read off `splitOK`. -/
structure SplitData (S : Spec) (H : ℕ) : Prop where
  pos : 0 < H
  wf : S.F.wf = true

theorem Spec.labDim_at (S : Spec) (k x : ℕ) :
    labDim (S.at k) x = (S.at k).dimv (labIdx S.F.labs x) := rfl

theorem Spec.labDim_bundle (S : Spec) (H k x : ℕ) (D : SplitData S H)
    (hg : S.good (labIdx S.F.labs x) = true) (hx : S.isB x = true) :
    labDim (S.at k) x = S.base x ^ k := by
  rw [Spec.labDim_at]
  obtain ⟨j, hl⟩ := S.cur_some _ hx
  rw [S.dimv_cur D.wf k _ j hl hg, Spec.base, Spec.baseIdx, hl]

theorem Spec.labDim_plain (S : Spec) (H k k' x : ℕ) (D : SplitData S H)
    (hg : S.good (labIdx S.F.labs x) = true) (hx : S.isB x = false) :
    labDim (S.at k) x = labDim (S.at k') x := by
  rw [Spec.labDim_at, Spec.labDim_at]
  have := S.not_mem_of_good D.wf _ hx hg
  rw [S.dimv_fixed D.wf k _ this, S.dimv_fixed D.wf k' _ this]

theorem Spec.labDim_hat (S : Spec) (H k x : ℕ) (D : SplitData S H)
    (hg : S.good (labIdx S.F.labs x) = true) (hx : S.isB x = true)
    (hh : labIdx S.F.labs (x + H) = S.baseIdx x) :
    labDim (S.at k) (x + H) = S.base x := by
  rw [Spec.labDim_at, hh]
  obtain ⟨j, hl⟩ := S.cur_some _ hx
  obtain ⟨-, -, c⟩ := S.chain_cur D.wf _ j hl hg
  have hj : S.baseIdx x = j := by simp [Spec.baseIdx, hl]
  rw [hj, S.dimv_fixed D.wf k j c, Spec.base, hj]

theorem Spec.base_pos (S : Spec) (hb : ∀ p ∈ S.F.bund, p.2.2 = S.m → 0 < S.F.dimv S.ks p.2.1)
    (x : ℕ) (hx : S.isB x = true) : 0 < S.base x := by
  obtain ⟨j, hl⟩ := S.cur_some _ hx
  have := hb _ (lookup_mem hl) rfl
  simpa [Spec.base, Spec.baseIdx, hl] using this

theorem Spec.combine_upd (S : Spec) (H k : ℕ) (D : SplitData S H) (τ : ℕ → ℕ) (l j : ℕ)
    (hl : l < H) (hb : S.isB l = false) (x : ℕ) (hx : x < H) :
    S.combine H k (Function.update τ l j) x = Function.update (S.combine H k τ) l j x := by
  by_cases hxl : x = l
  · subst hxl; simp [Spec.combine, hb]
  · rw [Function.update_of_ne hxl]
    simp only [Spec.combine]
    split_ifs
    · rw [Function.update_of_ne hxl, Function.update_of_ne (by omega)]
    · rw [Function.update_of_ne hxl]

theorem Spec.combine_upd₂ (S : Spec) (H k : ℕ) (D : SplitData S H) (τ : ℕ → ℕ) (l v y : ℕ)
    (hl : l < H) (hb : S.isB l = true) (x : ℕ) (hx : x < H) :
    S.combine H k (Function.update (Function.update τ l v) (l + H) y) x =
      Function.update (S.combine H k τ) l (v + S.base l ^ k * y) x := by
  by_cases hxl : x = l
  · subst hxl
    simp only [Spec.combine, hb, ite_true, Function.update_self]
    rw [Function.update_of_ne (by omega), Function.update_self]
  · rw [Function.update_of_ne hxl]
    simp only [Spec.combine]
    rw [Function.update_of_ne (by omega), Function.update_of_ne hxl]
    split_ifs
    · rw [Function.update_of_ne (by omega), Function.update_of_ne (by omega)]
    · rfl

theorem Spec.expand_plain (S : Spec) (H l : ℕ) (rest : List ℕ) (hb : S.isB l = false) :
    S.expand H (l :: rest) = l :: S.expand H rest := by
  simp [Spec.expand, hb]

theorem Spec.expand_bund (S : Spec) (H l : ℕ) (rest : List ℕ) (hb : S.isB l = true) :
    S.expand H (l :: rest) = l :: (l + H) :: S.expand H rest := by
  simp [Spec.expand, hb]

theorem sumOver_expand (S : Spec) (H k : ℕ) (D : SplitData S H) (S₀ : List ℕ)
    (h₀ : ∀ x ∈ S₀, x < H) (f : (ℕ → ℕ) → ℂ) (hf : Reads S₀ f) (ls : List ℕ)
    (hls : ∀ l ∈ ls, l < H) (hgs : ∀ l ∈ ls, S.good (labIdx S.F.labs l) = true)
    (hh : ∀ l ∈ ls, S.isB l = true → labIdx S.F.labs (l + H) = S.baseIdx l) (τ : ℕ → ℕ) :
    sumOver (labDim (S.at (k + 1))) ls f (S.combine H k τ) =
      sumOver (labDim (S.at k)) (S.expand H ls) (fun ρ => f (S.combine H k ρ)) τ := by
  induction ls generalizing τ with
  | nil => rfl
  | cons l rest ih =>
    have hl := hls l List.mem_cons_self
    have hrest := fun x hx => hls x (List.mem_cons_of_mem _ hx)
    have hhr := fun x hx => hh x (List.mem_cons_of_mem _ hx)
    have hgr := fun x hx => hgs x (List.mem_cons_of_mem _ hx)
    have hg := hgs l List.mem_cons_self
    have hfr : ∀ σ σ' : ℕ → ℕ, (∀ x < H, σ x = σ' x) →
        sumOver (labDim (S.at (k + 1))) rest f σ = sumOver (labDim (S.at (k + 1))) rest f σ' :=
      fun σ σ' h => sumOver_ext _ rest S₀ f hf σ σ' fun x hx _ => h x (h₀ x hx)
    cases hb : S.isB l with
    | false =>
      rw [S.expand_plain H l rest hb]
      simp only [sumOver]
      rw [Spec.labDim_plain S H (k + 1) k l D hg hb]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [← ih hrest hgr hhr]
      exact hfr _ _ fun x hx => (S.combine_upd H k D τ l j hl hb x hx).symm
    | true =>
      rw [S.expand_bund H l rest hb]
      simp only [sumOver]
      rw [S.labDim_bundle H (k + 1) l D hg hb, S.labDim_bundle H k l D hg hb,
        S.labDim_hat H k l D hg hb (hh l List.mem_cons_self hb), pow_succ, sum_range_mul]
      refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun y _ => ?_
      rw [← ih hrest hgr hhr]
      exact hfr _ _ fun x hx => (S.combine_upd₂ H k D τ l v y hl hb x hx).symm

theorem Spec.combine_bund (S : Spec) (H k : ℕ) (τ : ℕ → ℕ) (l : ℕ) (h : S.isB l = true) :
    S.combine H k τ l = τ l + S.base l ^ k * τ (l + H) := by
  simp [Spec.combine, h]

theorem Spec.digit_combine (S : Spec) (H k : ℕ) (τ : ℕ → ℕ) (l : ℕ) (hb : S.isB l = true)
    (hd : 0 < S.base l) (r1 : τ l < S.base l ^ k) (r2 : τ (l + H) < S.base l) :
    (∀ i < k, digit (S.base l) i (S.combine H k τ l) = digit (S.base l) i (τ l)) ∧
      digit (S.base l) k (S.combine H k τ l) = τ (l + H) := by
  rw [S.combine_bund H k τ l hb]
  exact ⟨fun i hi => digit_low _ _ _ _ _ hd hi r1, digit_top _ _ _ _ hd r1 r2⟩

/-! ## Splitting nodes -/

theorem digsC_congr (mult mult' : ℕ → ℕ) (d : ℕ) (c : List ℕ) (h : ∀ x ∈ c, mult x = mult' x)
    (v : ℕ) : digsC mult d c v = digsC mult' d c v := by
  induction c generalizing v with
  | nil => rfl
  | cons m c ih =>
    simp only [digsC]
    have hc : ∀ x ∈ c, mult x = mult' x := fun x hx => h x (List.mem_cons_of_mem _ hx)
    rw [h m List.mem_cons_self, List.map_congr_left hc]
    exact List.flatMap_congr fun v _ => ih hc v

theorem tailVals_congr (mult mult' : ℕ → ℕ) (d : ℕ) (vs : List ℕ) (cs : List (List ℕ))
    (h : ∀ c ∈ cs, ∀ x ∈ c, mult x = mult' x) : tailVals mult d vs cs = tailVals mult' d vs cs := by
  induction vs generalizing cs with
  | nil => simp [tailVals]
  | cons v vs ih =>
    cases cs with
    | nil => simp [tailVals]
    | cons c cs =>
      simp only [tailVals, List.zipWith_cons_cons, List.flatten_cons] at ih ⊢
      rw [digsC_congr _ _ d c (h c List.mem_cons_self) v,
        ih cs fun c' hc' => h c' (List.mem_cons_of_mem _ hc')]

theorem bundled_congr (mult mult' : ℕ → ℕ) (d : ℕ) (vs : List ℕ) (cs : List (List ℕ))
    (h : ∀ c ∈ cs, ∀ x ∈ c, mult x = mult' x) : bundled mult d vs cs = bundled mult' d vs cs := by
  simp only [bundled, tailVals_congr _ _ d _ cs h]

theorem copyLegs_congr (mult mult' : ℕ → ℕ) (d i : ℕ) (cs : List (List ℕ)) (vs : List ℕ)
    (h : ∀ c ∈ cs, ∀ x ∈ c, mult x = mult' x) : copyLegs mult d i cs vs = copyLegs mult' d i cs vs := by
  induction cs generalizing vs with
  | nil => simp [copyLegs]
  | cons c cs ih =>
    cases vs with
    | nil => simp [copyLegs]
    | cons v vs =>
      simp only [copyLegs, List.zipWith_cons_cons] at ih ⊢
      rw [List.map_congr_left (h c List.mem_cons_self),
        ih vs fun c' hc' => h c' (List.mem_cons_of_mem _ hc')]

theorem bundled_app (mult : ℕ → ℕ) (d : ℕ) (hd tl : List ℕ) (cs : List (List ℕ))
    (h : tl.length = cs.length) : bundled mult d (hd ++ tl) cs = hd ++ tailVals mult d tl cs := by
  simp only [bundled, List.length_append, h, Nat.add_sub_cancel, List.take_left',
    List.drop_left']

theorem tailVals_append (mult : ℕ → ℕ) (d : ℕ) (a b : List ℕ) (ca cb : List (List ℕ))
    (h : a.length = ca.length) :
    tailVals mult d (a ++ b) (ca ++ cb) = tailVals mult d a ca ++ tailVals mult d b cb := by
  simp only [tailVals, List.zipWith_append h, List.flatten_append]

theorem Kind.outCs_ms (b : Kind) (n : ℕ) : ∀ c ∈ b.outCs n, ∀ x ∈ c, x ∈ b.ms := by
  induction b with
  | XB co ci =>
    intro c hc x hx
    simp only [Kind.outCs, List.mem_append, List.mem_replicate] at hc
    rcases hc with ⟨-, rfl⟩ | hc
    · simp at hx
    · exact List.mem_append_left _ (List.mem_flatten.2 ⟨c, hc, hx⟩)
  | WB cs =>
    intro c hc x hx
    simp only [Kind.outCs, List.mem_append, List.mem_replicate] at hc
    rcases hc with ⟨-, rfl⟩ | hc
    · simp at hx
    · exact List.mem_flatten.2 ⟨c, hc, hx⟩
  | Bun m b ih =>
    intro c hc x hx
    simp only [Kind.outCs, List.mem_map] at hc
    obtain ⟨c', hc', rfl⟩ := hc
    simp only [Kind.ms, List.mem_cons] at hx ⊢
    rcases hx with rfl | hx
    · exact Or.inl rfl
    · exact Or.inr (ih c' hc' x hx)
  | _ =>
    intro c hc x hx
    simp only [Kind.outCs, List.mem_replicate] at hc
    obtain ⟨-, rfl⟩ := hc
    simp at hx

theorem Kind.inCs_ms (b : Kind) (n : ℕ) : ∀ c ∈ b.inCs n, ∀ x ∈ c, x ∈ b.ms := by
  induction b with
  | ZB cs =>
    intro c hc x hx
    simp only [Kind.inCs, List.mem_append, List.mem_replicate] at hc
    rcases hc with ⟨-, rfl⟩ | hc
    · simp at hx
    · exact List.mem_flatten.2 ⟨c, hc, hx⟩
  | XB co ci =>
    intro c hc x hx
    simp only [Kind.inCs, List.mem_append, List.mem_replicate] at hc
    rcases hc with ⟨-, rfl⟩ | hc
    · simp at hx
    · exact List.mem_append_right _ (List.mem_flatten.2 ⟨c, hc, hx⟩)
  | Bun m b ih =>
    intro c hc x hx
    simp only [Kind.inCs, List.mem_map] at hc
    obtain ⟨c', hc', rfl⟩ := hc
    simp only [Kind.ms, List.mem_cons] at hx ⊢
    rcases hx with rfl | hx
    · exact Or.inl rfl
    · exact Or.inr (ih c' hc' x hx)
  | _ =>
    intro c hc x hx
    simp only [Kind.inCs, List.mem_replicate] at hc
    obtain ⟨-, rfl⟩ := hc
    simp at hx

theorem tensor_env (E E' : Env) (hp : E.phase = E'.phase) (b : Kind)
    (hm : ∀ j ∈ b.ms, E.mult j = E'.mult j) (d ph : ℕ) (os is : List ℕ) :
    tensor E b d ph os is = tensor E' b d ph os is := by
  induction b generalizing os is with
  | ZB cs =>
    have h : ∀ c ∈ cs, ∀ x ∈ c, E.mult x = E'.mult x :=
      fun c hc x hx => hm x (List.mem_flatten.2 ⟨c, hc, hx⟩)
    simp only [tensor, hp, bundled_congr _ _ d is cs h]
  | XB co ci =>
    have h1 : ∀ c ∈ co, ∀ x ∈ c, E.mult x = E'.mult x :=
      fun c hc x hx => hm x (List.mem_append_left _ (List.mem_flatten.2 ⟨c, hc, hx⟩))
    have h2 : ∀ c ∈ ci, ∀ x ∈ c, E.mult x = E'.mult x :=
      fun c hc x hx => hm x (List.mem_append_right _ (List.mem_flatten.2 ⟨c, hc, hx⟩))
    simp only [tensor, hp, bundled_congr _ _ d os co h1, bundled_congr _ _ d is ci h2]
  | WB cs =>
    have h : ∀ c ∈ cs, ∀ x ∈ c, E.mult x = E'.mult x :=
      fun c hc x hx => hm x (List.mem_flatten.2 ⟨c, hc, hx⟩)
    rcases is with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> simp only [tensor, bundled_congr _ _ d os cs h]
  | Bun m b ih =>
    have hb : ∀ j ∈ b.ms, E.mult j = E'.mult j :=
      fun j hj => hm j (List.mem_cons_of_mem _ hj)
    simp only [tensor, hm m List.mem_cons_self]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [copyOuts, copyIns]
    rw [copyLegs_congr _ _ d i _ os fun c hc x hx => hb x (b.outCs_ms _ c hc x hx),
      copyLegs_congr _ _ d i _ is fun c hc x hx => hb x (b.inCs_ms _ c hc x hx)]
    exact ih hb _ _
  | Z => simp only [tensor, hp]
  | X => simp only [tensor, hp]
  | _ => rcases os with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> rfl

theorem val_tensor (E : Env) (σ : ℕ → ℕ) (b : Kind) (hb : b ≠ .B ∧ b ≠ .S)
    (os is : List ℕ) (dim ph : ℕ) :
    val E σ ⟨b, os, is, dim, ph⟩ = tensor E b (E.dimv dim) ph (os.map σ) (is.map σ) := by
  simp only [val]
  generalize os.map σ = A; generalize is.map σ = B
  cases b <;> simp at hb <;>
    rcases A with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;> rcases B with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;> rfl

theorem copyKind_ne (b : Kind) (h : copyKind b = true) : b ≠ .B ∧ b ≠ .S := by
  cases b <;> simp_all [copyKind]

/-- A node whose value ignores `m`. -/
theorem val_env (E E' : Env) (σ : ℕ → ℕ) (n : Node) (hd : E.dimv n.dim = E'.dimv n.dim)
    (hph : n.kind = .B ∨ n.kind = .S → E.dimv n.ph = E'.dimv n.ph) (hp : E.phase = E'.phase)
    (hm : ∀ j ∈ n.kind.ms, E.mult j = E'.mult j) : val E σ n = val E' σ n := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  by_cases hk : kind ≠ .B ∧ kind ≠ .S
  · rw [val_tensor E σ kind hk, val_tensor E' σ kind hk, hd]
    exact tensor_env E E' hp kind hm _ _ _ _
  · simp only [not_and_or, not_not] at hk
    simp only at hd hph
    rcases hk with rfl | rfl <;> simp only [val, hd, hph (by simp)] <;>
      generalize os.map σ = A <;> generalize is.map σ = B <;>
      rcases A with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;> rcases B with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;> rfl

theorem Spec.legsOK_append (S : Spec) (dim : ℕ) (a b : List ℕ) (ca cb : List (List ℕ))
    (h : a.length = ca.length) :
    S.legsOK dim (a ++ b) (ca ++ cb) = (S.legsOK dim a ca && S.legsOK dim b cb) := by
  simp only [Spec.legsOK, List.length_append, h, List.zip_append h, List.all_append]
  cases ha : ((List.zip a ca).all _) <;> cases hb : ((List.zip b cb).all _) <;> simp [ha, hb]

theorem Spec.legsOK_cons (S : Spec) (dim l : ℕ) (ls : List ℕ) (c : List ℕ) (cs : List (List ℕ)) :
    S.legsOK dim (l :: ls) (c :: cs) = ((S.F.chain (labIdx S.F.labs l) == c &&
      S.F.root (labIdx S.F.labs l) == dim) && S.legsOK dim ls cs) := by
  simp only [Spec.legsOK, List.length_cons, List.zip_cons_cons, List.all_cons]
  cases h1 : (S.F.chain (labIdx S.F.labs l) == c && S.F.root (labIdx S.F.labs l) == dim) <;>
    simp [h1, Nat.succ_inj]

theorem Spec.isB_of_chain (S : Spec) (hwf : S.F.wf = true) (l : ℕ) (c : List ℕ)
    (h : S.F.chain (labIdx S.F.labs l) = S.m :: c) :
    S.isB l = true ∧ ∃ j, S.F.bund.lookup (labIdx S.F.labs l) = some (j, S.m) ∧
      S.F.chain j = c ∧ S.baseIdx l = j := by
  cases hl : S.F.bund.lookup (labIdx S.F.labs l) with
  | none => rw [(S.F.chain_none hl).1] at h; simp at h
  | some p =>
    obtain ⟨j, m'⟩ := p
    rw [(S.F.chain_some hwf hl).1] at h
    simp only [List.cons.injEq] at h
    obtain ⟨rfl, hc⟩ := h
    exact ⟨by simp [Spec.isB, Spec.cur, hl], j, rfl, hc, by simp [Spec.baseIdx, hl]⟩

theorem Spec.not_isB_of_chain (S : Spec) (hwf : S.F.wf = true) (l : ℕ)
    (h : S.m ∉ S.F.chain (labIdx S.F.labs l)) : S.isB l = false := by
  cases hl : S.F.bund.lookup (labIdx S.F.labs l) with
  | none => simp [Spec.isB, Spec.cur, hl]
  | some p =>
    obtain ⟨j, m'⟩ := p
    rw [(S.F.chain_some hwf hl).1] at h
    simp only [List.mem_cons, not_or] at h
    simp [Spec.isB, Spec.cur, hl, Ne.symm h.1]

theorem Spec.base_chain (S : Spec) (hwf : S.F.wf = true) (l dim : ℕ) (c : List ℕ)
    (hc : S.F.chain (labIdx S.F.labs l) = S.m :: c) (hr : S.F.root (labIdx S.F.labs l) = dim) :
    S.base l = S.F.dims dim ^ (c.map S.ks).prod := by
  obtain ⟨-, j, hl, hj, hb⟩ := S.isB_of_chain hwf l c hc
  have := (S.F.chain_some hwf hl).2
  simp only [Spec.base, hb, Fam.dimv_eq, hj, ← this, hr]

theorem digs_combine (B k v y : ℕ) (hB : 0 < B) (hv : v < B ^ k) (hy : y < B) :
    digs B (k + 1) (v + B ^ k * y) = digs B k v ++ [y] := digs_succ B k v y hB hv hy

/-- The bundled legs at `k + 1`: their digits at `k`, and the peeled ones. -/
theorem Spec.tail_split (S : Spec) (H k : ℕ) (D : SplitData S H) (τ : ℕ → ℕ) (dim : ℕ)
    (tl : List ℕ) (cs : List (List ℕ)) (hok : S.legsOK dim tl cs = true)
    (hch : S.chainsOK cs = true)
    (hr : ∀ l ∈ tl, S.isB l = true → τ l < S.base l ^ k ∧ τ (l + H) < S.base l)
    (hd' : ∀ l ∈ tl, S.isB l = true → 0 < S.base l) :
    (tailVals (S.at (k + 1)).mult (S.F.dims dim) (tl.map (S.combine H k τ)) cs).Perm
      (tailVals (S.at k).mult (S.F.dims dim) (tl.map τ) cs ++
        tailVals (S.at k).mult (S.F.dims dim) ((peel S.m H tl cs).1.map τ)
          (peel S.m H tl cs).2) := by
  induction tl generalizing cs with
  | nil => simp [tailVals, peel]
  | cons b tl ih =>
    cases cs with
    | nil => simp [Spec.legsOK] at hok
    | cons c cs =>
      rw [S.legsOK_cons] at hok
      simp only [Bool.and_eq_true, beq_iff_eq] at hok
      obtain ⟨⟨hcb, hrb⟩, hok⟩ := hok
      simp only [Spec.chainsOK, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hch
      obtain ⟨hct, hch⟩ := hch
      have ih' := ih cs hok (by simpa [Spec.chainsOK] using hch)
        (fun l hl => hr l (List.mem_cons_of_mem _ hl))
        (fun l hl => hd' l (List.mem_cons_of_mem _ hl))
      have hmc : ∀ x ∈ c.tail, (S.at (k + 1)).mult x = (S.at k).mult x :=
        fun x hx => S.mult_ne _ _ _ (by rintro rfl; exact hct hx)
      by_cases hh : c.head? = some S.m
      · obtain ⟨c', rfl⟩ : ∃ c', c = S.m :: c' := by
          cases c with
          | nil => simp at hh
          | cons x c' => simp at hh; exact ⟨c', by rw [hh]⟩
        simp only [List.tail_cons] at hct hmc
        obtain ⟨hb, -⟩ := S.isB_of_chain D.wf b c' hcb
        obtain ⟨r1, r2⟩ := hr b List.mem_cons_self hb
        have hd := hd' b List.mem_cons_self hb
        have hbase := S.base_chain D.wf b dim c' hcb hrb
        have hB : S.F.dims dim ^ (c'.map (S.at (k + 1)).mult).prod = S.base b := by
          rw [hbase, List.map_congr_left hmc]
          simp only [Spec.at, Fam.env]
          congr 2
          exact List.map_congr_left fun x hx =>
            Function.update_of_ne (by rintro rfl; exact hct hx) _ _
        have hB0 : S.F.dims dim ^ (c'.map (S.at k).mult).prod = S.base b := by
          rw [← List.map_congr_left hmc, hB]
        simp only [List.map_cons, tailVals, List.zipWith_cons_cons, List.flatten_cons, peel,
          List.zip_cons_cons, List.filter_cons, List.head?_cons, beq_self_eq_true, ite_true,
          List.tail_cons] at ih' ⊢
        rw [show digsC (S.at (k + 1)).mult (S.F.dims dim) (S.m :: c') (S.combine H k τ b) =
            digsC (S.at k).mult (S.F.dims dim) (S.m :: c') (τ b) ++
              digsC (S.at k).mult (S.F.dims dim) c' (τ (b + H)) by
          simp only [digsC, hB, hB0, S.mult_self, S.combine_bund H k τ b hb]
          rw [digs_succ _ _ _ _ hd r1 r2, List.flatMap_append, List.flatMap_singleton,
            digsC_congr _ _ _ c' hmc]
          congr 1
          exact List.flatMap_congr fun v _ => digsC_congr _ _ _ c' hmc v]
        refine ((List.Perm.append_left _ ih').trans ?_)
        simp only [List.append_assoc]
        refine List.Perm.append_left _ ?_
        rw [← List.append_assoc, ← List.append_assoc]
        exact List.Perm.append_right _ List.perm_append_comm
      · have hm : S.m ∉ c := by
          intro h
          cases c with
          | nil => simp at h
          | cons x c' =>
            simp only [List.mem_cons] at h
            rcases h with rfl | h
            · exact hh rfl
            · exact hct h
        have hnb := S.not_isB_of_chain D.wf b (by rw [hcb]; exact hm)
        have hcomb : S.combine H k τ b = τ b := by simp [Spec.combine, hnb]
        have hmc' : ∀ x ∈ c, (S.at (k + 1)).mult x = (S.at k).mult x :=
          fun x hx => S.mult_ne _ _ _ (by rintro rfl; exact hm hx)
        have hf : (c.head? == some S.m) = false := by simpa using hh
        simp only [List.map_cons, tailVals, List.zipWith_cons_cons, List.flatten_cons, peel,
          List.zip_cons_cons, List.filter_cons, hf, Bool.false_eq_true, ite_false] at ih' ⊢
        rw [hcomb, digsC_congr _ _ _ c hmc', List.append_assoc]
        exact List.Perm.append_left _ ih'

theorem Spec.legsOK_split (S : Spec) (dim : ℕ) (ls : List ℕ) (cs : List (List ℕ))
    (h : S.legsOK dim ls (List.replicate (ls.length - cs.length) [] ++ cs) = true) :
    cs.length ≤ ls.length ∧
      S.legsOK dim (ls.take (ls.length - cs.length)) (List.replicate (ls.length - cs.length) [])
        = true ∧ S.legsOK dim (tailOf ls cs) cs = true := by
  have hlen : ls.length = ls.length - cs.length + cs.length := by
    have := h; simp only [Spec.legsOK, Bool.and_eq_true, beq_iff_eq, List.length_append,
      List.length_replicate] at this
    exact this.1
  refine ⟨by omega, ?_⟩
  set n := ls.length - cs.length with hn
  have h' : S.legsOK dim (ls.take n ++ tailOf ls cs) (List.replicate n [] ++ cs) = true := by
    rw [show ls.take n ++ tailOf ls cs = ls from List.take_append_drop n ls]; exact h
  rw [S.legsOK_append _ _ _ _ _ (by simp [n])] at h'
  simpa using h'

theorem Spec.plain_of_legsOK (S : Spec) (dim : ℕ) (ls : List ℕ) (n : ℕ)
    (h : S.legsOK dim ls (List.replicate n []) = true) :
    ∀ l ∈ ls, S.F.chain (labIdx S.F.labs l) = [] := by
  intro l hl
  simp only [Spec.legsOK, Bool.and_eq_true, beq_iff_eq, List.length_replicate,
    List.all_eq_true] at h
  obtain ⟨hn, h⟩ := h
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hl
  have := h (ls[i], []) (by
    rw [List.mem_iff_getElem]
    exact ⟨i, by simp [List.length_zip, hn]; omega, by simp⟩)
  simpa using this.1

theorem Spec.combine_plain_chain (S : Spec) (H k : ℕ) (τ : ℕ → ℕ) (ls : List ℕ)
    (h : ∀ l ∈ ls, S.F.chain (labIdx S.F.labs l) = []) (hwf : S.F.wf = true) :
    ls.map (S.combine H k τ) = ls.map τ := by
  refine List.map_congr_left fun l hl => ?_
  have := S.not_isB_of_chain hwf l (by rw [h l hl]; simp)
  simp [Spec.combine, this]

theorem peel_len (m H : ℕ) (tl : List ℕ) (cs : List (List ℕ)) :
    (peel m H tl cs).1.length = (peel m H tl cs).2.length := by
  simp [peel]

/-- The value of a spider's legs at `k + 1` against its split node's at `k`. -/
theorem Spec.legs_split (S : Spec) (H k : ℕ) (D : SplitData S H) (τ : ℕ → ℕ) (dim : ℕ)
    (ls : List ℕ) (cs : List (List ℕ))
    (hok : S.legsOK dim ls (List.replicate (ls.length - cs.length) [] ++ cs) = true)
    (hch : S.chainsOK cs = true)
    (hr : ∀ l ∈ ls, S.isB l = true → τ l < S.base l ^ k ∧ τ (l + H) < S.base l)
    (hd' : ∀ l ∈ ls, S.isB l = true → 0 < S.base l) :
    (bundled (S.at (k + 1)).mult (S.F.dims dim) (ls.map (S.combine H k τ)) cs).Perm
      (bundled (S.at k).mult (S.F.dims dim) ((ls ++ (peel S.m H (tailOf ls cs) cs).1).map τ)
        (cs ++ (peel S.m H (tailOf ls cs) cs).2)) := by
  obtain ⟨hle, hhd, htl⟩ := S.legsOK_split dim ls cs hok
  have e : ls = ls.take (ls.length - cs.length) ++ tailOf ls cs := (List.take_append_drop _ _).symm
  have hlt : (tailOf ls cs).length = cs.length := by simp [tailOf]; omega
  set hd := ls.take (ls.length - cs.length)
  set tl := tailOf ls cs
  have hp := peel_len S.m H tl cs
  rw [e]
  simp only [List.map_append, List.append_assoc]
  rw [bundled_app _ _ _ _ _ (by simp [hlt]),
    bundled_app _ _ _ _ _ (by simp [hlt, hp]),
    tailVals_append _ _ _ _ _ _ (by simp [hlt]),
    S.combine_plain_chain H k τ hd (S.plain_of_legsOK dim hd _ hhd) D.wf]
  refine List.Perm.append_left _ ?_
  exact S.tail_split H k D τ dim tl cs htl hch (fun l hl => hr l (by rw [e]; simp [hl]))
    (fun l hl => hd' l (by rw [e]; simp [hl]))

/-- Digits of copied legs at `k + 1`: those at `k` below `k`, the peeled values at `k`. -/
theorem Spec.copy_split (S : Spec) (H k : ℕ) (D : SplitData S H) (τ : ℕ → ℕ) (dim : ℕ)
    (ls : List ℕ) (cs : List (List ℕ)) (hok : S.legsOK dim ls (cs.map (S.m :: ·)) = true)
    (hm : ∀ c ∈ cs, S.m ∉ c)
    (hr : ∀ l ∈ ls, S.isB l = true → τ l < S.base l ^ k ∧ τ (l + H) < S.base l)
    (hd' : ∀ l ∈ ls, S.isB l = true → 0 < S.base l) :
    (∀ i < k, copyLegs (S.at (k + 1)).mult (S.F.dims dim) i cs (ls.map (S.combine H k τ)) =
      copyLegs (S.at k).mult (S.F.dims dim) i cs (ls.map τ)) ∧
    copyLegs (S.at (k + 1)).mult (S.F.dims dim) k cs (ls.map (S.combine H k τ)) =
      (ls.map (· + H)).map τ := by
  induction ls generalizing cs with
  | nil => cases cs <;> simp [copyLegs, Spec.legsOK] at hok ⊢
  | cons l ls ih =>
    cases cs with
    | nil => simp [Spec.legsOK] at hok
    | cons c cs =>
      simp only [List.map_cons] at hok
      rw [S.legsOK_cons] at hok
      simp only [Bool.and_eq_true, beq_iff_eq] at hok
      obtain ⟨⟨hcl, hrl⟩, hok⟩ := hok
      obtain ⟨ihl, ihk⟩ := ih cs hok (fun c' hc' => hm c' (List.mem_cons_of_mem _ hc'))
        (fun x hx => hr x (List.mem_cons_of_mem _ hx))
        (fun x hx => hd' x (List.mem_cons_of_mem _ hx))
      have hmc := hm c List.mem_cons_self
      obtain ⟨hb, -⟩ := S.isB_of_chain D.wf l c hcl
      obtain ⟨r1, r2⟩ := hr l List.mem_cons_self hb
      have hd := hd' l List.mem_cons_self hb
      have hbase := S.base_chain D.wf l dim c hcl hrl
      have e1 : ∀ k', S.F.dims dim ^ (c.map (S.at k').mult).prod = S.base l := by
        intro k'
        rw [hbase]
        simp only [Spec.at, Fam.env]
        congr 2
        exact List.map_congr_left fun x hx =>
          Function.update_of_ne (by rintro rfl; exact hmc hx) _ _
      obtain ⟨dl, dt⟩ := S.digit_combine H k τ l hb hd r1 r2
      refine ⟨fun i hi => ?_, ?_⟩
      · simp only [copyLegs, List.map_cons, List.zipWith_cons_cons] at ihl ⊢
        rw [e1, e1, dl i hi, ihl i hi]
      · simp only [copyLegs, List.map_cons, List.zipWith_cons_cons] at ihk ⊢
        rw [e1, dt, ihk]


theorem Spec.dimv_plain (S : Spec) (k : ℕ) (i : ℕ) (h : (S.F.bund.lookup i).isNone = true) :
    (S.at k).dimv i = S.F.dims i := by
  rw [Option.isNone_iff_eq_none] at h
  obtain ⟨a, b⟩ := S.F.chain_none h
  simp [Spec.at, Fam.env, Fam.dimv_eq, a, b]

theorem Spec.chainsOK_append (S : Spec) (a b : List (List ℕ)) :
    S.chainsOK (a ++ b) = (S.chainsOK a && S.chainsOK b) := by
  simp [Spec.chainsOK, List.all_append]

theorem Spec.chainsOK_replicate (S : Spec) (n : ℕ) : S.chainsOK (List.replicate n []) = true := by
  simp [Spec.chainsOK, List.all_eq_true]

/-- A node none of whose legs bundles along `m` and whose value ignores `m`. -/
theorem Spec.val_still (S : Spec) (H k : ℕ) (D : SplitData S H) (n : Node) (τ : ℕ → ℕ)
    (hl : ∀ l ∈ n.outs ++ n.ins, S.isB l = false)
    (hd : (S.F.bund.lookup n.dim).isNone = true)
    (hph : n.kind = .B ∨ n.kind = .S → (S.F.bund.lookup n.ph).isNone = true)
    (hm : S.m ∉ n.kind.ms) :
    val (S.at (k + 1)) (S.combine H k τ) n = val (S.at k) τ n := by
  have h := val_reads (S.at (k + 1)) n (S.combine H k τ) τ fun x hx => by
    simp [Spec.combine, hl x hx]
  simp only at h
  rw [h]
  refine val_env _ _ τ n (by rw [S.dimv_plain _ _ hd, S.dimv_plain _ _ hd])
    (fun h => by rw [S.dimv_plain _ _ (hph h), S.dimv_plain _ _ (hph h)]) rfl
    fun j hj => S.mult_ne _ _ _ (by rintro rfl; exact hm hj)

theorem Spec.legs_plain (S : Spec) (hwf : S.F.wf = true) (dim : ℕ) (ls : List ℕ) (n : ℕ)
    (h : S.legsOK dim ls (List.replicate n []) = true) : ∀ l ∈ ls, S.isB l = false :=
  fun l hl => S.not_isB_of_chain hwf l (by rw [S.plain_of_legsOK dim ls n h l hl]; simp)

theorem val_split (S : Spec) (H k : ℕ) (D : SplitData S H) (n : Node) (hn : S.nodeOK n = true)
    (τ : ℕ → ℕ)
    (hd' : ∀ l ∈ n.outs ++ n.ins, S.isB l = true → 0 < S.base l)
    (hr : ∀ l ∈ n.outs ++ n.ins, S.isB l = true → τ l < S.base l ^ k ∧ τ (l + H) < S.base l) :
    val (S.at (k + 1)) (S.combine H k τ) n = ((Node.split S.m H n).map (val (S.at k) τ)).prod := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only [Spec.nodeOK, Bool.and_eq_true] at hn
  obtain ⟨hdim, hn⟩ := hn
  have hdk : ∀ k', (S.at k').dimv dim = S.F.dims dim := fun k' => S.dimv_plain _ _ hdim
  have hro : ∀ l ∈ os, S.isB l = true → τ l < S.base l ^ k ∧ τ (l + H) < S.base l :=
    fun l hl => hr l (List.mem_append_left _ hl)
  have hri : ∀ l ∈ is, S.isB l = true → τ l < S.base l ^ k ∧ τ (l + H) < S.base l :=
    fun l hl => hr l (List.mem_append_right _ hl)
  have hdo : ∀ l ∈ os, S.isB l = true → 0 < S.base l := fun l hl => hd' l (List.mem_append_left _ hl)
  have hdi : ∀ l ∈ is, S.isB l = true → 0 < S.base l := fun l hl => hd' l (List.mem_append_right _ hl)
  have plain : kind.ms = [] → (∀ l ∈ os ++ is, S.isB l = false) →
      (kind = .B ∨ kind = .S → (S.F.bund.lookup ph).isNone = true) →
      Node.split S.m H ⟨kind, os, is, dim, ph⟩ = [⟨kind, os, is, dim, ph⟩] →
      val (S.at (k + 1)) (S.combine H k τ) ⟨kind, os, is, dim, ph⟩ =
        ((Node.split S.m H ⟨kind, os, is, dim, ph⟩).map (val (S.at k) τ)).prod := by
    intro hms hl hph hs
    rw [hs, List.map_singleton, List.prod_singleton]
    exact S.val_still H k D _ τ hl hdim hph (by simp [hms])
  cases kind with
  | ZB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, hli⟩ := hn
    simp only [Node.split, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      val, hdk]
    rw [S.combine_plain_chain H k τ os (S.plain_of_legsOK dim os _ hlo) D.wf]
    exact tZ_perm _ _ _ _ (List.Perm.append_left _
      (S.legs_split H k D τ dim is cs hli hch hri hdi))
  | XB co ci =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨⟨hco, hci⟩, hlo⟩, hli⟩ := hn
    simp only [Node.split, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      val, hdk]
    exact tX_perm _ _ _ _ _ _ (S.legs_split H k D τ dim os co hlo hco hro hdo)
      (S.legs_split H k D τ dim is ci hli hci hri hdi)
  | WB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, hli⟩ := hn
    have hpi := S.legs_plain D.wf dim is _ hli
    simp only [Node.split, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      val, hdk]
    rw [show is.map (S.combine H k τ) = is.map τ from
      List.map_congr_left fun l hl => by simp [Spec.combine, hpi l hl]]
    have hp := S.legs_split H k D τ dim os cs hlo hch.1 hro hdo
    generalize is.map τ = B
    rcases B with _ | ⟨w, _ | ⟨_, _⟩⟩
    · rfl
    · exact tW_perm _ _ _ _ hp
    · rfl
  | Bun m b =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hn
    obtain ⟨⟨⟨hck, hmb⟩, hlo⟩, hli⟩ := hn
    by_cases hm : m = S.m
    · subst hm
      have hmo : ∀ c ∈ b.outCs os.length, S.m ∉ c :=
        fun c hc h => hmb (b.outCs_ms _ c hc _ h)
      have hmi : ∀ c ∈ b.inCs is.length, S.m ∉ c :=
        fun c hc h => hmb (b.inCs_ms _ c hc _ h)
      obtain ⟨ol, ok⟩ := S.copy_split H k D τ dim os _ hlo hmo hro hdo
      obtain ⟨il, ik⟩ := S.copy_split H k D τ dim is _ hli hmi hri hdi
      have he : tensor (S.at (k + 1)) b (S.F.dims dim) ph =
          tensor (S.at k) b (S.F.dims dim) ph := by
        funext x y
        exact tensor_env (S.at (k + 1)) (S.at k) rfl b
          (fun j hj => S.mult_ne _ _ _ (by rintro rfl; exact hmb hj)) _ _ _ _
      simp only [Node.split, ite_true, List.map_cons, List.map_nil, List.prod_cons,
        List.prod_nil, mul_one]
      rw [val_tensor (S.at k) τ b (copyKind_ne b hck)]
      simp only [val, tensor, hdk, S.mult_self, copyOuts, copyIns, List.length_map]
      rw [List.range_succ, List.map_append, List.prod_append, List.map_singleton,
        List.prod_singleton, ok, ik, he, List.map_map, List.map_map]
      congr 1
      congr 1
      refine List.map_congr_left fun i hi' => ?_
      have hi := List.mem_range.1 hi'
      rw [ol i hi, il i hi]
    · have hl : ∀ l ∈ os ++ is, S.isB l = false := by
        intro l hl
        rcases List.mem_append.1 hl with hl | hl
        · exact S.not_isB_of_chain D.wf l (by
            simp only [Spec.legsOK, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at hlo
            obtain ⟨hlen, hz⟩ := hlo
            obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hl
            have := hz (os[i], ((Kind.Bun m b).outCs os.length)[i]'(by
              simp [Kind.outCs] at hlen ⊢; omega)) (by
              rw [List.mem_iff_getElem]; exact ⟨i, by simp [List.length_zip]; simp [Kind.outCs] at hlen ⊢; omega, by simp⟩)
            simp only [beq_iff_eq, Bool.and_eq_true] at this
            rw [this.1]
            simp only [Kind.outCs, List.getElem_map, List.mem_cons, not_or]
            exact ⟨Ne.symm hm, fun h => hmb (b.outCs_ms _ _ (List.getElem_mem _) _ h)⟩)
        · exact S.not_isB_of_chain D.wf l (by
            simp only [Spec.legsOK, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at hli
            obtain ⟨hlen, hz⟩ := hli
            obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hl
            have := hz (is[i], ((Kind.Bun m b).inCs is.length)[i]'(by
              simp [Kind.inCs] at hlen ⊢; omega)) (by
              rw [List.mem_iff_getElem]; exact ⟨i, by simp [List.length_zip]; simp [Kind.inCs] at hlen ⊢; omega, by simp⟩)
            simp only [beq_iff_eq, Bool.and_eq_true] at this
            rw [this.1]
            simp only [Kind.inCs, List.getElem_map, List.mem_cons, not_or]
            exact ⟨Ne.symm hm, fun h => hmb (b.inCs_ms _ _ (List.getElem_mem _) _ h)⟩)
      simp only [Node.split, hm, ite_false, List.map_cons, List.map_nil, List.prod_cons,
        List.prod_nil, mul_one]
      exact S.val_still H k D _ τ hl hdim (by simp) (by
        simp only [Kind.ms, List.mem_cons, not_or]; exact ⟨Ne.symm hm, hmb⟩)
  | B | S =>
    simp only [Bool.and_eq_true, List.all_eq_true] at hn
    refine plain rfl (fun l hl => ?_) (fun _ => hn.1) rfl
    have := hn.2 l hl
    cases h : S.F.bund.lookup (labIdx S.F.labs l) with
    | none => simp [Spec.isB, Spec.cur, h]
    | some p => rw [h] at this; simp at this
  | _ =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_replicate, Bool.true_and, Bool.and_eq_true] at hn
    refine plain rfl (fun l hl => ?_) (by simp) rfl
    rcases List.mem_append.1 hl with hl | hl
    · exact S.legs_plain D.wf dim os _ hn.1 l hl
    · exact S.legs_plain D.wf dim is _ hn.2 l hl


/-! ## Splitting networks -/

/-- What `splitOK` guarantees. -/
structure SplitFacts (S : Spec) (H : ℕ) (N : Net) : Prop where
  data : SplitData S H
  closed : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd
  lt : ∀ l ∈ labelsOf N.nodes ++ N.int ++ N.bnd, l < H
  hat : ∀ l ∈ labelsOf N.nodes ++ N.int ++ N.bnd, S.isB l = true →
    labIdx S.F.labs (l + H) = S.baseIdx l
  good : ∀ l ∈ labelsOf N.nodes ++ N.int ++ N.bnd, S.good (labIdx S.F.labs l) = true
  nodes : ∀ n ∈ N.nodes, S.nodeOK n = true

theorem Spec.splitOK_facts (S : Spec) (H : ℕ) (N : Net) (h : S.splitOK H N = true) :
    SplitFacts S H N := by
  simp only [Spec.splitOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hH, hwf⟩, -⟩, hcl⟩, hgood⟩, hlt⟩, -⟩, hhat⟩, hnodes⟩ := h
  exact ⟨⟨hH, hwf⟩, fun l hl => by simpa using hcl l hl, fun l hl => by simpa using hlt l hl,
    fun l hl hb => by
      have := hhat l hl; simp only [Bool.or_eq_true, Bool.not_eq_true', beq_iff_eq] at this
      exact this.resolve_left (by simp [hb]), hgood, hnodes⟩

theorem prodVal_flatMap (E : Env) (ns : List Node) (f : Node → List Node) (ρ : ℕ → ℕ) :
    prodVal E (ns.flatMap f) ρ = (ns.map fun n => prodVal E (f n) ρ).prod := by
  induction ns with
  | nil => rfl
  | cons n ns ih =>
    rw [List.flatMap_cons, prodVal_append, ih, List.map_cons, List.prod_cons]


theorem Spec.mem_expand (S : Spec) (H x : ℕ) (ls : List ℕ) :
    x ∈ S.expand H ls ↔ x ∈ ls ∨ ∃ m ∈ ls, S.isB m = true ∧ x = m + H := by
  simp only [Spec.expand, List.mem_flatMap]
  constructor
  · rintro ⟨m, hm, hx⟩
    split_ifs at hx with hb
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · exact Or.inl hm
      · exact Or.inr ⟨m, hm, hb, rfl⟩
    · simp only [List.mem_singleton] at hx; subst hx; exact Or.inl hm
  · rintro (hx | ⟨m, hm, hb, rfl⟩)
    · refine ⟨x, hx, ?_⟩; split_ifs <;> simp
    · exact ⟨m, hm, by simp [hb]⟩

theorem eval_split (S : Spec) (H k : ℕ) (N : Net) (F : SplitFacts S H N)
    (hb : ∀ p ∈ S.F.bund, p.2.2 = S.m → 0 < S.F.dimv S.ks p.2.1) (τ : ℕ → ℕ)
    (hr : InRange (S.at k) (S.expand H N.bnd) τ) :
    eval (S.at (k + 1)) N (S.combine H k τ) = eval (S.at k) (N.split S H) τ := by
  have hltN : ∀ x ∈ labelsOf N.nodes, x < H := fun x hx => F.lt x (by simp [hx])
  have hltI : ∀ x ∈ N.int, x < H := fun x hx => F.lt x (by simp [hx])
  have hltB : ∀ x ∈ N.bnd, x < H := fun x hx => F.lt x (by simp [hx])
  simp only [eval, Net.split]
  rw [sumOver_expand S H k F.data (labelsOf N.nodes) hltN _ (prodVal_reads _ _) N.int hltI
    (fun l hl => F.good l (by simp [hl])) (fun l hl hb => F.hat l (by simp [hl]) hb) τ]
  refine sumOver_congr _ _ _ _ _ fun ρ hρ hrng => ?_
  rw [prodVal_flatMap]
  unfold prodVal
  congr 1
  refine List.map_congr_left fun n hn => ?_
  have hleg : ∀ l ∈ n.outs ++ n.ins, l ∈ labelsOf N.nodes := fun l hl =>
    List.mem_flatMap.2 ⟨n, hn, hl⟩
  refine val_split S H k F.data n (F.nodes n hn) ρ (fun l hl hbl => S.base_pos hb l hbl)
    (fun l hl hbl => ?_)
  have hlH := hltN l (hleg l hl)
  rcases F.closed l (hleg l hl) with hi | hbd
  · have h1 := hrng l ((S.mem_expand H l N.int).2 (Or.inl hi))
    have h2 := hrng (l + H) ((S.mem_expand H _ N.int).2 (Or.inr ⟨l, hi, hbl, rfl⟩))
    rw [S.labDim_bundle H k l F.data (F.good l (by simp [hleg l hl])) hbl] at h1
    rw [S.labDim_hat H k l F.data (F.good l (by simp [hleg l hl])) hbl
      (F.hat l (by simp [hleg l hl]) hbl)] at h2
    exact ⟨h1, h2⟩
  · by_cases hi : l ∈ N.int
    · have h1 := hrng l ((S.mem_expand H l N.int).2 (Or.inl hi))
      have h2 := hrng (l + H) ((S.mem_expand H _ N.int).2 (Or.inr ⟨l, hi, hbl, rfl⟩))
      rw [S.labDim_bundle H k l F.data (F.good l (by simp [hleg l hl])) hbl] at h1
      rw [S.labDim_hat H k l F.data (F.good l (by simp [hleg l hl])) hbl
      (F.hat l (by simp [hleg l hl]) hbl)] at h2
      exact ⟨h1, h2⟩
    · have n1 : l ∉ S.expand H N.int := by
        rw [S.mem_expand]; rintro (h | ⟨m, -, -, hm⟩)
        · exact hi h
        · omega
      have n2 : l + H ∉ S.expand H N.int := by
        rw [S.mem_expand]; rintro (h | ⟨m, hm, -, he⟩)
        · have := hltI _ h; omega
        · have : m = l := by omega
          exact hi (this ▸ hm)
      rw [hρ l n1, hρ (l + H) n2]
      have h1 := hr l ((S.mem_expand H l N.bnd).2 (Or.inl hbd))
      have h2 := hr (l + H) ((S.mem_expand H _ N.bnd).2 (Or.inr ⟨l, hbd, hbl, rfl⟩))
      rw [S.labDim_bundle H k l F.data (F.good l (by simp [hleg l hl])) hbl] at h1
      rw [S.labDim_hat H k l F.data (F.good l (by simp [hleg l hl])) hbl
      (F.hat l (by simp [hleg l hl]) hbl)] at h2
      exact ⟨h1, h2⟩


/-- An assignment at `k + 1` cut into bundles below `d ^ k` and their peeled digits. -/
def Spec.unsplit (S : Spec) (H k : ℕ) (σ : ℕ → ℕ) (x : ℕ) : ℕ :=
  if x < H then (if S.isB x then σ x % S.base x ^ k else σ x)
  else if S.isB (x - H) then σ (x - H) / S.base (x - H) ^ k else σ x

theorem Spec.combine_unsplit (S : Spec) (H k : ℕ) (σ : ℕ → ℕ) (x : ℕ) (hx : x < H) :
    S.combine H k (S.unsplit H k σ) x = σ x := by
  by_cases hb : S.isB x = true
  · have h1 : S.unsplit H k σ x = σ x % S.base x ^ k := by simp [Spec.unsplit, hx, hb]
    have h2 : S.unsplit H k σ (x + H) = σ x / S.base x ^ k := by
      simp [Spec.unsplit, show ¬ x + H < H by omega, hb]
    simp only [Spec.combine, hb, ite_true, h1, h2]
    exact Nat.mod_add_div _ _
  · simp [Spec.combine, Spec.unsplit, hx, hb]

theorem expand_filter (S : Spec) (H : ℕ) (ls : List ℕ) (h : ∀ l ∈ ls, l < H) :
    (S.expand H ls).filter (fun x => decide (x < H)) = ls := by
  induction ls with
  | nil => rfl
  | cons l ls ih =>
    have hl := h l List.mem_cons_self
    have ih' := ih fun x hx => h x (List.mem_cons_of_mem _ hx)
    cases hb : S.isB l
    · rw [S.expand_plain H l ls hb, List.filter_cons, if_pos (by simpa using hl), ih']
    · rw [S.expand_bund H l ls hb, List.filter_cons, if_pos (by simpa using hl),
        List.filter_cons, if_neg (by simp), ih']

theorem Equal.of_split (S : Spec) (H k : ℕ) (N M : Net) (hN : S.splitOK H N = true)
    (hM : S.splitOK H M = true) (hb : ∀ p ∈ S.F.bund, p.2.2 = S.m → 0 < S.F.dimv S.ks p.2.1)
    {s s' : ℂ}
    (h : Equal (S.at k) s (N.split S H) s' (M.split S H)) : Equal (S.at (k + 1)) s N s' M := by
  have FN := S.splitOK_facts H N hN
  have FM := S.splitOK_facts H M hM
  have hbN : ∀ l ∈ N.bnd, l < H := fun l hl => FN.lt l (by simp [hl])
  have hbM : ∀ l ∈ M.bnd, l < H := fun l hl => FM.lt l (by simp [hl])
  have hbnd : N.bnd.Perm M.bnd := by
    have := h.1.filter (fun x => decide (x < H))
    simpa [Net.split, expand_filter S H _ hbN, expand_filter S H _ hbM] using this
  have hBpos := S.base_pos hb
  refine ⟨hbnd, fun σ hσ => ?_⟩
  set τ := S.unsplit H k σ
  have hτ : InRange (S.at k) (S.expand H N.bnd) τ := by
    intro x hx
    rcases (S.mem_expand H x N.bnd).1 hx with hx | ⟨m, hm, hbm, rfl⟩
    · have hxH := hbN x hx
      have := hσ x hx
      simp only [τ, Spec.unsplit, hxH, ite_true]
      cases hbx : S.isB x
      · rw [S.labDim_plain H k (k + 1) x FN.data (FN.good x (by simp [hx])) hbx]
        simpa [hbx] using this
      · rw [S.labDim_bundle H k x FN.data (FN.good x (by simp [hx])) hbx]
        simp only [ite_true]
        exact Nat.mod_lt _ (pow_pos (hBpos x hbx) _)
    · have hmH := hbM m (hbnd.subset hm)
      have := hσ m hm
      rw [S.labDim_bundle H (k + 1) m FN.data (FN.good m (by simp [hm])) hbm, pow_succ] at this
      rw [S.labDim_hat H k m FN.data (FN.good m (by simp [hm])) hbm (FN.hat m (by simp [hm]) hbm)]
      simp only [τ, Spec.unsplit, if_neg (show ¬ m + H < H by omega), Nat.add_sub_cancel, hbm,
        ite_true]
      exact (Nat.div_lt_iff_lt_mul (pow_pos (hBpos m hbm) _)).2 (by rw [mul_comm]; exact this)
  have hev : ∀ (X : Net), SplitFacts S H X → InRange (S.at k) (S.expand H X.bnd) τ →
      eval (S.at (k + 1)) X σ = eval (S.at k) (X.split S H) τ := by
    intro X FX hX
    rw [← eval_split S H k X FX hb τ hX]
    refine sumOver_ext _ _ (labelsOf X.nodes) _ (prodVal_reads _ _) _ _ fun x hx _ => ?_
    exact (S.combine_unsplit H k σ x (FX.lt x (by simp [hx]))).symm
  rw [hev N FN hτ, hev M FM (hτ.perm (by simpa [Net.split] using h.1))]
  exact h.2 τ hτ

/-! ## The empty family -/

/-- The legs of `ls` without the bundled ones (chains `cs`) that bundle along `m`. -/
def dropM (m : ℕ) (ls : List ℕ) (cs : List (List ℕ)) : List ℕ × List (List ℕ) :=
  let ps := (List.zip (tailOf ls cs) cs).filter fun p => p.2.head? != some m
  (ls.take (ls.length - cs.length) ++ ps.map (·.1), ps.map (·.2))

/-- A Z spider bundling along `cs`. -/
def zk : List (List ℕ) → Kind
  | [] => .Z
  | cs => .ZB cs

/-- An X spider bundling along `co`, `ci`. -/
def xk : List (List ℕ) → List (List ℕ) → Kind
  | [], [] => .X
  | co, ci => .XB co ci

/-- A W node bundling along `cs`. -/
def wk : List (List ℕ) → Kind
  | [] => .W
  | cs => .WB cs

/-- A node at `m = 0`, its bundles along `m` removed; none for a copied generator. -/
def Node.zero (m : ℕ) (n : Node) : Option Node :=
  match n.kind with
  | .ZB cs =>
    let p := dropM m n.ins cs
    some { n with kind := zk p.2, ins := p.1 }
  | .XB co ci =>
    let p := dropM m n.outs co
    let q := dropM m n.ins ci
    some { n with kind := xk p.2 q.2, outs := p.1, ins := q.1 }
  | .WB cs =>
    let p := dropM m n.outs cs
    some { n with kind := wk p.2, outs := p.1 }
  | .Bun m' _ => if m' = m then Option.none else some n
  | _ => some n

/-- The network at `m = 0` without its bundles along `m`. -/
def Net.zero (S : Spec) (N : Net) : Net :=
  ⟨N.nodes.filterMap (Node.zero S.m), N.int.filter (fun l => !S.isB l),
    N.bnd.filter (fun l => !S.isB l)⟩

theorem bundled_nil (mult : ℕ → ℕ) (d : ℕ) (vs : List ℕ) : bundled mult d vs [] = vs := by
  simp [bundled, tailVals]

/-- Bundles along `m` at `m = 0` carry no digits. -/
theorem bundled_zero (mult : ℕ → ℕ) (m : ℕ) (hm : mult m = 0) (d : ℕ) (σ : ℕ → ℕ)
    (ls : List ℕ) (cs : List (List ℕ)) (hl : cs.length ≤ ls.length) :
    bundled mult d (ls.map σ) cs = bundled mult d ((dropM m ls cs).1.map σ) (dropM m ls cs).2 := by
  have e : ls = ls.take (ls.length - cs.length) ++ tailOf ls cs := (List.take_append_drop _ _).symm
  have hlt : (tailOf ls cs).length = cs.length := by simp [tailOf]; omega
  simp only [dropM]
  conv_lhs => rw [e]
  rw [List.map_append, bundled_app _ _ _ _ _ (by simp [hlt]), List.map_append,
    bundled_app _ _ _ _ _ (by simp)]
  congr 1
  generalize tailOf ls cs = tl at hlt
  clear e hl
  induction tl generalizing cs with
  | nil => cases cs <;> simp_all [tailVals]
  | cons b tl ih =>
    cases cs with
    | nil => simp at hlt
    | cons c cs =>
      simp only [List.length_cons, Nat.add_right_cancel_iff] at hlt
      have ih' := ih cs hlt
      by_cases hh : c.head? = some m
      · obtain ⟨c', rfl⟩ : ∃ c', c = m :: c' := by
          cases c with
          | nil => simp at hh
          | cons x c' => simp at hh; exact ⟨c', by rw [hh]⟩
        simp only [List.map_cons, tailVals, List.zipWith_cons_cons, List.flatten_cons,
          List.zip_cons_cons, List.filter_cons, List.head?_cons, bne_self_eq_false,
          Bool.false_eq_true, ite_false] at ih' ⊢
        simp [digsC, hm, digs, ih']
      · have hf : (c.head? != some m) = true := by simpa using hh
        simp only [List.map_cons, tailVals, List.zipWith_cons_cons, List.flatten_cons,
          List.zip_cons_cons, List.filter_cons, hf, ite_true] at ih' ⊢
        rw [ih']

theorem val_zk (E : Env) (σ : ℕ → ℕ) (cs : List (List ℕ)) (os is : List ℕ) (dim ph : ℕ) :
    val E σ ⟨zk cs, os, is, dim, ph⟩ =
      tZ (E.dimv dim) (E.phase ph) (os.map σ ++ bundled E.mult (E.dimv dim) (is.map σ) cs) := by
  cases cs with
  | nil => simp [zk, val, bundled_nil]
  | cons c cs => simp [zk, val]

theorem val_xk (E : Env) (σ : ℕ → ℕ) (co ci : List (List ℕ)) (os is : List ℕ) (dim ph : ℕ) :
    val E σ ⟨xk co ci, os, is, dim, ph⟩ = tX (E.dimv dim) (E.phase ph)
      (bundled E.mult (E.dimv dim) (os.map σ) co) (bundled E.mult (E.dimv dim) (is.map σ) ci) := by
  rcases co with _ | ⟨c, co⟩ <;> rcases ci with _ | ⟨c', ci⟩ <;> simp [xk, val, bundled_nil]

theorem val_wk (E : Env) (σ : ℕ → ℕ) (cs : List (List ℕ)) (os : List ℕ) (w dim ph : ℕ) :
    val E σ ⟨wk cs, os, [w], dim, ph⟩ =
      tW (E.dimv dim) (bundled E.mult (E.dimv dim) (os.map σ) cs) (σ w) := by
  cases cs with
  | nil => simp [wk, val, bundled_nil]
  | cons c cs => simp [wk, val]

theorem Spec.legsOK_len (S : Spec) (dim : ℕ) (ls : List ℕ) (cs : List (List ℕ))
    (h : S.legsOK dim ls (List.replicate (ls.length - cs.length) [] ++ cs) = true) :
    cs.length ≤ ls.length := (S.legsOK_split dim ls cs h).1

theorem val_zero (S : Spec) (n : Node) (hn : S.nodeOK n = true) (σ : ℕ → ℕ) :
    val (S.at 0) σ n = ((Node.zero S.m n).map (val (S.at 0) σ)).getD 1 := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only [Spec.nodeOK, Bool.and_eq_true] at hn
  obtain ⟨-, hn⟩ := hn
  have hm := S.mult_self 0
  cases kind with
  | ZB cs =>
    simp only [Kind.outCs, Kind.inCs, Bool.and_eq_true] at hn
    obtain ⟨-, hli⟩ := hn
    simp only [Node.zero, Option.map_some, Option.getD_some, val_zk]
    simp only [val]
    rw [bundled_zero _ S.m hm _ σ is cs (S.legsOK_len dim is cs hli)]
  | XB co ci =>
    simp only [Kind.outCs, Kind.inCs, Bool.and_eq_true] at hn
    obtain ⟨⟨-, hlo⟩, hli⟩ := hn
    simp only [Node.zero, Option.map_some, Option.getD_some, val_xk]
    simp only [val]
    rw [bundled_zero _ S.m hm _ σ os co (S.legsOK_len dim os co hlo),
      bundled_zero _ S.m hm _ σ is ci (S.legsOK_len dim is ci hli)]
  | WB cs =>
    simp only [Kind.outCs, Kind.inCs, Bool.and_eq_true] at hn
    obtain ⟨⟨-, hlo⟩, -⟩ := hn
    simp only [Node.zero, Option.map_some, Option.getD_some]
    rcases is with _ | ⟨w, _ | ⟨w', is⟩⟩
    · cases h : (dropM S.m os cs).2 <;> simp [val, wk, h]
    · rw [val_wk]
      simp only [val, List.map_cons, List.map_nil]
      rw [bundled_zero _ S.m hm _ σ os cs (S.legsOK_len dim os cs hlo)]
    · cases h : (dropM S.m os cs).2 <;> simp [val, wk, h]
  | Bun m b =>
    by_cases hmb : m = S.m
    · subst hmb
      simp [Node.zero, val, tensor, hm]
    · simp [Node.zero, hmb]
  | _ => simp [Node.zero]

theorem Spec.chain_of_legsOK (S : Spec) (dim : ℕ) (ls : List ℕ) (cs : List (List ℕ))
    (h : S.legsOK dim ls cs = true) (l : ℕ) (c : List ℕ) (hz : (l, c) ∈ List.zip ls cs) :
    S.F.chain (labIdx S.F.labs l) = c := by
  simp only [Spec.legsOK, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  exact (h.2 _ hz).1

theorem Spec.legsOK_not_isB (S : Spec) (hwf : S.F.wf = true) (dim : ℕ) (ls : List ℕ)
    (cs : List (List ℕ)) (h : S.legsOK dim ls cs = true) (hm : ∀ c ∈ cs, S.m ∉ c) :
    ∀ l ∈ ls, S.isB l = false := by
  induction ls generalizing cs with
  | nil => simp
  | cons l ls ih =>
    cases cs with
    | nil => simp [Spec.legsOK] at h
    | cons c cs =>
      rw [S.legsOK_cons] at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact S.not_isB_of_chain hwf _ (by rw [h.1.1]; exact hm c List.mem_cons_self)
      · exact ih cs h.2 (fun c' hc' => hm c' (List.mem_cons_of_mem _ hc')) x hx

/-- A spider's legs kept at `m = 0` are not bundles along `m`. -/
theorem Spec.dropM_not_isB (S : Spec) (hwf : S.F.wf = true) (dim : ℕ) (ls : List ℕ)
    (cs : List (List ℕ)) (h : S.legsOK dim ls (List.replicate (ls.length - cs.length) [] ++ cs) = true)
    (hch : S.chainsOK cs = true) : ∀ l ∈ (dropM S.m ls cs).1, S.isB l = false := by
  obtain ⟨-, hhd, htl⟩ := S.legsOK_split dim ls cs h
  intro l hl
  simp only [dropM, List.mem_append, List.mem_map, List.mem_filter, bne_iff_ne, ne_eq] at hl
  rcases hl with h1 | ⟨⟨a, c⟩, ⟨hz, hh⟩, rfl⟩
  · exact S.not_isB_of_chain hwf l (by rw [S.plain_of_legsOK dim _ _ hhd l h1]; simp)
  · have hct : S.m ∉ c.tail := by
      simp only [Spec.chainsOK, List.all_eq_true, decide_eq_true_eq] at hch
      exact hch c (List.of_mem_zip hz).2
    have hmc : S.m ∉ c := by
      intro hmem
      cases c with
      | nil => simp at hmem
      | cons x c' =>
        simp only [List.mem_cons] at hmem
        rcases hmem with rfl | hmem
        · exact hh rfl
        · exact hct hmem
    exact S.not_isB_of_chain hwf a (by rw [S.chain_of_legsOK dim _ cs htl a c hz]; exact hmc)

/-- The labels a node keeps at `m = 0`. -/
theorem Node.zero_labels (S : Spec) (hwf : S.F.wf = true) (n m : Node) (hn : S.nodeOK n = true)
    (h : Node.zero S.m n = some m) (l : ℕ) (hl : l ∈ m.outs ++ m.ins) : S.isB l = false := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only [Spec.nodeOK, Bool.and_eq_true] at hn
  obtain ⟨-, hn⟩ := hn
  cases kind with
  | ZB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, hli⟩ := hn
    simp only [Node.zero, Option.some.injEq] at h; subst h
    rcases List.mem_append.1 hl with hl | hl
    · exact S.legs_plain hwf dim os _ hlo l hl
    · exact S.dropM_not_isB hwf dim is cs hli hch l hl
  | XB co ci =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨⟨hco, hci⟩, hlo⟩, hli⟩ := hn
    simp only [Node.zero, Option.some.injEq] at h; subst h
    rcases List.mem_append.1 hl with hl | hl
    · exact S.dropM_not_isB hwf dim os co hlo hco l hl
    · exact S.dropM_not_isB hwf dim is ci hli hci l hl
  | WB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, hli⟩ := hn
    simp only [Node.zero, Option.some.injEq] at h; subst h
    rcases List.mem_append.1 hl with hl | hl
    · exact S.dropM_not_isB hwf dim os cs hlo hch.1 l hl
    · exact S.legs_plain hwf dim is _ hli l hl
  | Bun m' b =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hn
    obtain ⟨⟨⟨-, hmb⟩, hlo⟩, hli⟩ := hn
    by_cases hm : m' = S.m
    · subst hm; simp [Node.zero] at h
    · simp only [Node.zero, hm, ite_false, Option.some.injEq] at h; subst h
      have hco : ∀ c ∈ (Kind.Bun m' b).outCs os.length, S.m ∉ c := by
        intro c hc hmem
        simp only [Kind.outCs, List.mem_map] at hc
        obtain ⟨c', hc', rfl⟩ := hc
        simp only [List.mem_cons] at hmem
        rcases hmem with h' | h'
        · exact hm h'.symm
        · exact hmb (b.outCs_ms _ c' hc' _ h')
      have hci : ∀ c ∈ (Kind.Bun m' b).inCs is.length, S.m ∉ c := by
        intro c hc hmem
        simp only [Kind.inCs, List.mem_map] at hc
        obtain ⟨c', hc', rfl⟩ := hc
        simp only [List.mem_cons] at hmem
        rcases hmem with h' | h'
        · exact hm h'.symm
        · exact hmb (b.inCs_ms _ c' hc' _ h')
      rcases List.mem_append.1 hl with hl | hl
      · exact S.legsOK_not_isB hwf dim os _ hlo hco l hl
      · exact S.legsOK_not_isB hwf dim is _ hli hci l hl
  | B | S =>
    simp only [Node.zero, Option.some.injEq] at h; subst h
    simp only [Bool.and_eq_true, List.all_eq_true] at hn
    have := hn.2 l hl
    cases hb : S.F.bund.lookup (labIdx S.F.labs l) with
    | none => simp [Spec.isB, Spec.cur, hb]
    | some p => rw [hb] at this; simp at this
  | _ =>
    simp only [Node.zero, Option.some.injEq] at h; subst h
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_replicate, Bool.true_and, Bool.and_eq_true] at hn
    rcases List.mem_append.1 hl with hl | hl
    · exact S.legs_plain hwf dim _ _ hn.1 l hl
    · exact S.legs_plain hwf dim _ _ hn.2 l hl

theorem sumOver_drop (D : ℕ → ℕ) (p : ℕ → Bool) (S₀ : List ℕ) (f : (ℕ → ℕ) → ℂ)
    (hf : Reads S₀ f) (ls : List ℕ) (hD : ∀ l ∈ ls, p l = false → D l = 1)
    (hS : ∀ l ∈ ls, p l = false → l ∉ S₀) (σ : ℕ → ℕ) :
    sumOver D ls f σ = sumOver D (ls.filter p) f σ := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l rest ih =>
    have ihr := fun σ => ih (fun x hx => hD x (List.mem_cons_of_mem _ hx))
      (fun x hx => hS x (List.mem_cons_of_mem _ hx)) σ
    cases hp : p l
    · rw [List.filter_cons, if_neg (by simp [hp])]
      simp only [sumOver, hD l List.mem_cons_self hp, Finset.sum_range_one]
      rw [ihr]
      refine sumOver_ext _ _ S₀ f hf _ _ fun x hx _ => ?_
      have : x ≠ l := fun h => hS l List.mem_cons_self hp (h ▸ hx)
      exact Function.update_of_ne this _ _
    · rw [List.filter_cons, if_pos (by simp [hp])]
      simp only [sumOver]
      exact Finset.sum_congr rfl fun j _ => ihr _

theorem eval_zero (S : Spec) (H : ℕ) (N : Net) (hN : S.splitOK H N = true) (σ : ℕ → ℕ) :
    eval (S.at 0) N σ = eval (S.at 0) (N.zero S) σ := by
  have FN := S.splitOK_facts H N hN
  have hprod : prodVal (S.at 0) N.nodes = prodVal (S.at 0) (N.nodes.filterMap (Node.zero S.m)) := by
    funext ρ
    have : ∀ ns : List Node, (∀ n ∈ ns, S.nodeOK n = true) →
        prodVal (S.at 0) ns ρ = prodVal (S.at 0) (ns.filterMap (Node.zero S.m)) ρ := by
      intro ns hns
      induction ns with
      | nil => rfl
      | cons n ns ih =>
        have hz := val_zero S n (hns n List.mem_cons_self) ρ
        have ih' := ih fun x hx => hns x (List.mem_cons_of_mem _ hx)
        simp only [prodVal, List.map_cons, List.prod_cons] at ih' ⊢
        rw [List.filterMap_cons]
        cases h : Node.zero S.m n with
        | none => rw [h] at hz; simp only [Option.map_none, Option.getD_none] at hz
                  rw [hz, one_mul, ih']
        | some m => rw [h] at hz; simp only [Option.map_some, Option.getD_some] at hz
                    simp only [List.map_cons, List.prod_cons]; rw [hz, ih']
    exact this N.nodes FN.nodes
  simp only [eval, Net.zero]
  rw [hprod]
  refine sumOver_drop _ (fun l => !S.isB l) (labelsOf (N.nodes.filterMap (Node.zero S.m))) _
    (prodVal_reads _ _) N.int (fun l hli hl => ?_) (fun l _ hl hm => ?_) σ
  · simp only [Bool.not_eq_false'] at hl
    rw [S.labDim_bundle H 0 l FN.data (FN.good l (by simp [hli])) hl, pow_zero]
  · simp only [Bool.not_eq_false'] at hl
    obtain ⟨m, hm', hlm⟩ := List.mem_flatMap.1 hm
    obtain ⟨n, hn, hnm⟩ := List.mem_filterMap.1 hm'
    have := Node.zero_labels S FN.data.wf n m (FN.nodes n hn) hnm l hlm
    rw [hl] at this; simp at this

theorem Equal.of_zero (S : Spec) (H : ℕ) (N M : Net) (hN : S.splitOK H N = true)
    (hM : S.splitOK H M = true) (hbnd : N.bnd.Perm M.bnd) {s s' : ℂ}
    (h : Equal (S.at 0) s (N.zero S) s' (M.zero S)) : Equal (S.at 0) s N s' M := by
  refine ⟨hbnd, fun σ hσ => ?_⟩
  rw [eval_zero S H N hN, eval_zero S H M hM]
  exact h.2 σ fun l hl => hσ l (List.mem_filter.1 hl).1


/-! ## Relabelling and plugging -/

/-- Every label of a network through `ren π`. -/
def Net.rename (π : List (ℕ × ℕ)) (N : Net) : Net :=
  ⟨N.nodes.map (Node.rename (ren π)), N.int.map (ren π), N.bnd.map (ren π)⟩

/-- `π` is a bijection keeping the dimension of every label of `N`. -/
def renameOK (labs : List (ℕ × ℕ)) (π : List (ℕ × ℕ)) (N : Net) : Bool :=
  decide (π.map Prod.fst).Nodup && ((π.map Prod.fst).map (ren π)).isPerm (π.map Prod.fst) &&
    (labelsOf N.nodes ++ N.int ++ N.bnd).all fun x => labIdx labs (ren π x) == labIdx labs x

theorem Equal.rename (E : Env) (π : List (ℕ × ℕ)) (A B : Net) (hA : renameOK E.labs π A = true)
    (hB : renameOK E.labs π B = true) {s s' : ℂ} (h : Equal E s A s' B) :
    Equal E s (A.rename π) s' (B.rename π) := by
  simp only [renameOK, Bool.and_eq_true, decide_eq_true_eq, List.isPerm_iff, List.all_eq_true,
    beq_iff_eq] at hA hB
  have hf := ren_injective π hA.1.1 hA.1.2
  have hev : ∀ (X : Net), (∀ x ∈ labelsOf X.nodes ++ X.int ++ X.bnd,
      labIdx E.labs (ren π x) = labIdx E.labs x) → ∀ σ,
      eval E (X.rename π) σ = eval E X (σ ∘ ren π) := by
    intro X hX σ
    simp only [eval, Net.rename]
    rw [show prodVal E (X.nodes.map (Node.rename (ren π))) =
      fun τ => prodVal E X.nodes (τ ∘ ren π) from funext fun τ => prodVal_rename E _ τ _]
    rw [sumOver_map _ _ hf X.int fun x hx => by
      simp only [labDim]; simp only [labIdx] at hX; rw [hX x (by simp [hx])]]
  refine ⟨by simpa [Net.rename] using h.1.map (ren π), fun σ hσ => ?_⟩
  rw [hev A hA.2, hev B hB.2]
  refine h.2 _ fun l hl => ?_
  have := hσ (ren π l) (by simp [Net.rename]; exact ⟨l, hl, rfl⟩)
  simp only [labDim, Function.comp] at this ⊢
  simp only [labIdx] at hA
  rwa [hA.2 l (by simp [hl])] at this

/-- An equation between whole networks as a local equation on their boundary. -/
theorem Equal.toLocal (E : Env) {s s' : ℂ} (A B : Net) (h : Equal E s A s' B) :
    LocalEq E A.nodes A.int B.nodes B.int A.bnd s s' := fun τ hτ => h.2 τ hτ


/-- A local equation with any scalars, in context. -/
theorem plug (E : Env) (C L R : List Node) (iC iL iR bnd iface : List ℕ) (kL kR : ℂ)
    (hloc : LocalEq E L iL R iR iface kL kR)
    (hL : ∀ x ∈ labelsOf C, x ∉ iL) (hR : ∀ x ∈ labelsOf C, x ∉ iR)
    (hface : ∀ x ∈ iface, x ∈ iC ∨ x ∈ bnd) :
    Equal E kL ⟨C ++ L, iC ++ iL, bnd⟩ kR ⟨C ++ R, iC ++ iR, bnd⟩ :=
  ⟨List.Perm.refl _, fun σ hσ => frame E C L R iC iL iR bnd iface kL kR hloc hL hR hface σ hσ⟩


/-! ## Isomorphism up to the leg symmetry of every generator -/

/-- The bundled legs of `a`, `b` (chains `ca`, `cb`) agree up to order with their chains. -/
def tailEquiv (a b : List ℕ) (ca cb : List (List ℕ)) : Bool :=
  decide (ca.length ≤ a.length) && decide (cb.length ≤ b.length) &&
    (List.zip (tailOf a ca) ca).isPerm (List.zip (tailOf b cb) cb)

/-- The plain legs of `a` (chains `ca` on its last legs). -/
def headOf (a : List ℕ) (ca : List (List ℕ)) : List ℕ := a.take (a.length - ca.length)

/-- The legs `ls` with the bundled ones of empty chain made plain, and the other chains. -/
def normTail (ls : List ℕ) (cs : List (List ℕ)) : List ℕ × List (List ℕ) :=
  let ps := List.zip (tailOf ls cs) cs
  (headOf ls cs ++ (ps.filter fun p => p.2.isEmpty).map (·.1) ++
      (ps.filter fun p => !p.2.isEmpty).map (·.1),
    (ps.filter fun p => !p.2.isEmpty).map (·.2))

/-- A node with its bundled legs of empty chain made plain. -/
def Node.norm (n : Node) : Node :=
  match n.kind with
  | .ZB cs => let p := normTail n.ins cs; { n with kind := zk p.2, ins := p.1 }
  | .XB co ci =>
    let p := normTail n.outs co
    let q := normTail n.ins ci
    { n with kind := xk p.2 q.2, outs := p.1, ins := q.1 }
  | .WB cs => let p := normTail n.outs cs; { n with kind := wk p.2, outs := p.1 }
  | _ => n

/-- `nodeEquivB` before normalising. -/
def nodeEquivB₀ (n m : Node) : Bool :=
  nodeEquiv n m || (n.dim == m.dim && n.ph == m.ph &&
    match n.kind, m.kind with
    | .ZB cs, .ZB cs' => tailEquiv n.ins m.ins cs cs' &&
        (n.outs ++ headOf n.ins cs).isPerm (m.outs ++ headOf m.ins cs')
    | .XB co ci, .XB co' ci' => tailEquiv n.outs m.outs co co' && tailEquiv n.ins m.ins ci ci' &&
        (headOf n.outs co).isPerm (headOf m.outs co') && (headOf n.ins ci).isPerm (headOf m.ins ci')
    | .WB cs, .WB cs' => tailEquiv n.outs m.outs cs cs' &&
        (headOf n.outs cs).isPerm (headOf m.outs cs') && n.ins == m.ins
    | .W, .W => n.outs.isPerm m.outs && n.ins == m.ins
    | .Bun k .Z, .Bun k' .Z => k == k' && (n.outs ++ n.ins).isPerm (m.outs ++ m.ins)
    | .Bun k .X, .Bun k' .X => k == k' && n.outs.isPerm m.outs && n.ins.isPerm m.ins
    | _, _ => false)

/-- `nodeEquivB₀` on the normalised nodes. -/
def nodeEquivB (n m : Node) : Bool := nodeEquivB₀ n.norm m.norm

theorem tailVals_zip (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (tl : List ℕ) (cs : List (List ℕ)) :
    tailVals mult d (tl.map σ) cs = ((List.zip tl cs).map fun p => digsC mult d p.2 (σ p.1)).flatten := by
  induction tl generalizing cs with
  | nil => simp [tailVals]
  | cons b tl ih =>
    cases cs with
    | nil => simp [tailVals]
    | cons c cs =>
      simp only [tailVals, List.map_cons, List.zipWith_cons_cons, List.flatten_cons,
        List.zip_cons_cons] at ih ⊢
      rw [ih]

theorem bundled_map (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (a : List ℕ) (ca : List (List ℕ)) :
    bundled mult d (a.map σ) ca = (headOf a ca).map σ ++
      ((List.zip (tailOf a ca) ca).map fun p => digsC mult d p.2 (σ p.1)).flatten := by
  simp only [bundled, headOf, tailOf, List.length_map, List.map_take, List.map_drop]
  rw [← List.map_drop, tailVals_zip]

theorem bundled_perm (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (a b : List ℕ) (ca cb : List (List ℕ))
    (h : tailEquiv a b ca cb = true) :
    ((List.zip (tailOf a ca) ca).map fun p => digsC mult d p.2 (σ p.1)).flatten.Perm
      ((List.zip (tailOf b cb) cb).map fun p => digsC mult d p.2 (σ p.1)).flatten := by
  simp only [tailEquiv, Bool.and_eq_true, decide_eq_true_eq, List.isPerm_iff] at h
  exact (h.2.map _).flatten

theorem copyLegs_rep (mult : ℕ → ℕ) (d i : ℕ) (vs : List ℕ) :
    copyLegs mult d i (List.replicate vs.length []) vs = vs.map (digit d i) := by
  induction vs with
  | nil => rfl
  | cons v vs ih => simp [copyLegs, List.replicate_succ] at ih ⊢; exact ih

theorem copyLegs_rep' (mult : ℕ → ℕ) (d i : ℕ) (σ : ℕ → ℕ) (l : List ℕ) :
    copyLegs mult d i (List.replicate l.length []) (l.map σ) = (l.map σ).map (digit d i) := by
  rw [← List.length_map σ]; exact copyLegs_rep mult d i _

theorem val_nodeEquivB₀ (E : Env) (σ : ℕ → ℕ) (n m : Node) (h : nodeEquivB₀ n m = true) :
    val E σ n = val E σ m := by
  simp only [nodeEquivB₀, Bool.or_eq_true] at h
  rcases h with h | h
  · exact val_nodeEquiv E σ n m h
  obtain ⟨kn, on, inn, dn, pn⟩ := n
  obtain ⟨km, om, im, dm, pm⟩ := m
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨rfl, rfl⟩, h⟩ := h
  cases kn <;> cases km <;> try (simp at h; done)
  case ZB.ZB cs cs' =>
    simp only [Bool.and_eq_true, List.isPerm_iff] at h
    simp only [val, bundled_map]
    refine tZ_perm _ _ _ _ ?_
    rw [← List.append_assoc, ← List.append_assoc, ← List.map_append, ← List.map_append]
    exact List.Perm.append (h.2.map σ) (bundled_perm _ _ σ _ _ _ _ h.1)
  case XB.XB co ci co' ci' =>
    simp only [Bool.and_eq_true, List.isPerm_iff] at h
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
    simp only [val, bundled_map]
    exact tX_perm _ _ _ _ _ _ (List.Perm.append (h3.map σ) (bundled_perm _ _ σ _ _ _ _ h1))
      (List.Perm.append (h4.map σ) (bundled_perm _ _ σ _ _ _ _ h2))
  case WB.WB cs cs' =>
    simp only [Bool.and_eq_true, List.isPerm_iff, beq_iff_eq] at h
    obtain ⟨⟨h1, h2⟩, rfl⟩ := h
    rcases inn with _ | ⟨w, _ | ⟨w', rest⟩⟩
    · simp [val]
    · simp only [val, List.map_cons, List.map_nil, bundled_map]
      exact tW_perm _ _ _ _ (List.Perm.append (h2.map σ) (bundled_perm _ _ σ _ _ _ _ h1))
    · simp [val]
  case W.W =>
    simp only [Bool.and_eq_true, beq_iff_eq, List.isPerm_iff] at h
    obtain ⟨h1, rfl⟩ := h
    rcases inn with _ | ⟨w, _ | ⟨w', rest⟩⟩
    · simp [val]
    · simp only [val, List.map_cons, List.map_nil]
      exact tW_perm _ _ _ _ (h1.map σ)
    · simp [val]
  case Bun.Bun k b k' b' =>
    cases b <;> cases b' <;> try (simp at h; done)
    · simp only [Bool.and_eq_true, beq_iff_eq, List.isPerm_iff] at h
      obtain ⟨rfl, h⟩ := h
      simp only [val, tensor, copyOuts, copyIns, Kind.outCs, Kind.inCs, List.length_map,
        copyLegs_rep']
      congr 1
      refine List.map_congr_left fun i _ => ?_
      refine tZ_perm _ _ _ _ ?_
      simpa only [List.map_append] using (h.map σ).map (digit (E.dimv dn) i)
    · simp only [Bool.and_eq_true, beq_iff_eq, List.isPerm_iff] at h
      obtain ⟨⟨rfl, h1⟩, h2⟩ := h
      simp only [val, tensor, copyOuts, copyIns, Kind.outCs, Kind.inCs, List.length_map,
        copyLegs_rep']
      congr 1
      refine List.map_congr_left fun i _ => ?_
      exact tX_perm _ _ _ _ _ _ ((h1.map σ).map _) ((h2.map σ).map _)

theorem zip_fst_snd {α β : Type*} (l : List (α × β)) :
    List.zip (l.map (·.1)) (l.map (·.2)) = l := by
  induction l with
  | nil => rfl
  | cons p l ih => simp [ih]

theorem headOf_tailOf (ls : List ℕ) (cs : List (List ℕ)) : headOf ls cs ++ tailOf ls cs = ls :=
  List.take_append_drop _ _

theorem bundled_norm (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (ls : List ℕ) (cs : List (List ℕ)) :
    (bundled mult d (ls.map σ) cs).Perm
      (bundled mult d ((normTail ls cs).1.map σ) (normTail ls cs).2) := by
  set ps := List.zip (tailOf ls cs) cs
  set A := ps.filter fun p => p.2.isEmpty
  set B := ps.filter fun p => !p.2.isEmpty
  have e1 : (normTail ls cs).1 = headOf ls cs ++ A.map (·.1) ++ B.map (·.1) := rfl
  have e2 : (normTail ls cs).2 = B.map (·.2) := rfl
  rw [e1, e2]
  have hlen : (headOf ls cs ++ A.map (·.1) ++ B.map (·.1)).length - (B.map (·.2)).length =
      (headOf ls cs ++ A.map (·.1)).length := by
    simp only [List.length_append, List.length_map]; omega
  have hh : headOf (headOf ls cs ++ A.map (·.1) ++ B.map (·.1)) (B.map (·.2)) =
      headOf ls cs ++ A.map (·.1) := by
    rw [show ∀ X Y, headOf X Y = X.take (X.length - Y.length) from fun _ _ => rfl, hlen]
    exact List.take_left' rfl
  have ht : tailOf (headOf ls cs ++ A.map (·.1) ++ B.map (·.1)) (B.map (·.2)) = B.map (·.1) := by
    rw [show ∀ X (Y : List (List ℕ)), tailOf X Y = X.drop (X.length - Y.length) from fun _ _ => rfl,
      hlen]
    exact List.drop_left' rfl
  rw [bundled_map, bundled_map, hh, ht, zip_fst_snd]
  have hA : ∀ l : List (ℕ × List ℕ), (∀ p ∈ l, p.2.isEmpty = true) →
      (l.map fun p => digsC mult d p.2 (σ p.1)).flatten = (l.map (·.1)).map σ := by
    intro l hl
    induction l with
    | nil => rfl
    | cons p l ih =>
      have hp := hl p List.mem_cons_self
      obtain ⟨a, c⟩ := p
      simp only [List.isEmpty_iff] at hp
      subst hp
      simp only [List.map_cons, List.flatten_cons, digsC, List.singleton_append]
      rw [ih fun q hq => hl q (List.mem_cons_of_mem _ hq)]
  have hAe : (A.map fun p => digsC mult d p.2 (σ p.1)).flatten = (A.map (·.1)).map σ :=
    hA A fun p hp => (List.mem_filter.1 hp).2
  have hperm : (ps.map fun p => digsC mult d p.2 (σ p.1)).flatten.Perm
      ((A.map fun p => digsC mult d p.2 (σ p.1)).flatten ++
        (B.map fun p => digsC mult d p.2 (σ p.1)).flatten) := by
    rw [← List.flatten_append, ← List.map_append]
    exact (((List.filter_append_perm _ ps).symm).map _).flatten
  rw [List.map_append, List.append_assoc, ← hAe]
  exact List.Perm.append_left _ hperm

theorem val_norm (E : Env) (σ : ℕ → ℕ) (n : Node) : val E σ n.norm = val E σ n := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  cases kind with
  | ZB cs =>
    simp only [Node.norm, val_zk]
    simp only [val]
    exact tZ_perm _ _ _ _ (List.Perm.append_left _ (bundled_norm _ _ σ is cs).symm)
  | XB co ci =>
    simp only [Node.norm, val_xk]
    simp only [val]
    exact tX_perm _ _ _ _ _ _ (bundled_norm _ _ σ os co).symm (bundled_norm _ _ σ is ci).symm
  | WB cs =>
    simp only [Node.norm]
    rcases is with _ | ⟨w, _ | ⟨w', is⟩⟩
    · cases h : (normTail os cs).2 <;> simp [val, wk, h]
    · rw [val_wk]
      simp only [val, List.map_cons, List.map_nil]
      exact tW_perm _ _ _ _ (bundled_norm _ _ σ os cs).symm
    · cases h : (normTail os cs).2 <;> simp [val, wk, h]
  | _ => rfl

theorem val_nodeEquivB (E : Env) (σ : ℕ → ℕ) (n m : Node) (h : nodeEquivB n m = true) :
    val E σ n = val E σ m := by
  rw [← val_norm E σ n, ← val_norm E σ m]
  exact val_nodeEquivB₀ E σ _ _ h

/-- `isoOK` with `nodeEquivB`. -/
def isoOKB (labs : List (ℕ × ℕ)) (A B : Net) (π : List (ℕ × ℕ)) (p : List ℕ) : Bool :=
  let f := ren π
  let K := π.map Prod.fst
  let A' := A.nodes.map (Node.rename f)
  let P := p.map fun j => A'.getD j Node.none
  decide K.Nodup && (K.map f).isPerm K &&
    (labelsOf A.nodes).all (fun x => decide (x ∈ A.int) || !decide (x ∈ K)) &&
    A.int.all (fun x => labIdx labs (f x) == labIdx labs x) &&
    A.bnd.isPerm B.bnd && B.int.isPerm (A.int.map f) && P.isPerm A' &&
    P.length == B.nodes.length && (List.zip P B.nodes).all fun q => nodeEquivB q.1 q.2

theorem prodVal_equivB (E : Env) (σ : ℕ → ℕ) (P Q : List Node) (hl : P.length = Q.length)
    (h : (List.zip P Q).all (fun q => nodeEquivB q.1 q.2) = true) :
    prodVal E P σ = prodVal E Q σ := by
  induction P generalizing Q with
  | nil => cases Q <;> simp_all
  | cons n P ih =>
    cases Q with
    | nil => simp at hl
    | cons m Q =>
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at h
      simp only [prodVal, List.map_cons, List.prod_cons] at ih ⊢
      rw [val_nodeEquivB E σ n m h.1, ih Q (by simpa using hl) h.2]

theorem isoB (E : Env) (s : ℂ) (A B : Net) (π : List (ℕ × ℕ)) (p : List ℕ)
    (h : isoOKB E.labs A B π p = true) : Equal E s A s B := by
  simp only [isoOKB, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.isPerm_iff,
    List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hk, hp⟩, hfree⟩, hdim⟩, hbnd⟩, hint⟩, hP⟩, hlen⟩, hnodes⟩ := h
  refine ⟨hbnd, fun σ _ => ?_⟩
  congr 1
  set f := ren π
  have hf := ren_injective π hk hp
  have hB : prodVal E B.nodes = prodVal E (A.nodes.map (Node.rename f)) := by
    funext τ
    rw [← prodVal_equivB E τ _ _ hlen (by simpa using hnodes)]
    exact (hP.map (val E τ)).prod_eq
  simp only [eval]
  rw [sumOver_perm _ hint, hB]
  have hr : prodVal E (A.nodes.map (Node.rename f)) =
      fun τ => prodVal E A.nodes (τ ∘ f) := funext fun τ => prodVal_rename E f τ A.nodes
  rw [hr, sumOver_map _ f hf A.int (fun x hx => by
    have := hdim x hx; simp only [labIdx] at this; simp only [labDim, this])]
  refine (sumOver_ext _ _ (labelsOf A.nodes) _ (prodVal_reads E A.nodes) _ _ fun x hx hn => ?_)
  rcases hfree x hx with h' | h'
  · exact absurd h' hn
  · have : f x = x := ren_of_not_mem π x h'
    simp [Function.comp, this]

end ArchytasZX

namespace ArchytasZX

/-! ## Multiplicities as a list -/

/-- The multiplicities `l`, zero past its end. -/
def ksOf (l : List ℕ) (i : ℕ) : ℕ := l.getD i 0

theorem update_ksOf (l : List ℕ) (m v : ℕ) (h : m < l.length) :
    Function.update (ksOf l) m v = ksOf (l.set m v) := by
  funext i
  by_cases hi : i = m
  · subst hi; simp [ksOf, List.getD_eq_getElem?_getD, h]
  · rw [Function.update_of_ne hi]
    simp [ksOf, List.getD_eq_getElem?_getD, List.getElem?_set_ne (Ne.symm hi)]

theorem Spec.at_ksOf (F : Fam) (m : ℕ) (l : List ℕ) (k : ℕ) (h : m < l.length) :
    Spec.at ⟨F, m, ksOf l⟩ k = F.env (ksOf (l.set m k)) := by
  simp only [Spec.at, update_ksOf l m k h]

/-- `Equal.of_split` with the multiplicities written as a list. -/
theorem Equal.of_split_at (F : Fam) (m H k : ℕ) (l : List ℕ) (hm : m < l.length) (N M : Net)
    (hN : Spec.splitOK ⟨F, m, ksOf l⟩ H N = true) (hM : Spec.splitOK ⟨F, m, ksOf l⟩ H M = true)
    (hb : ∀ p ∈ F.bund, p.2.2 = m → 0 < F.dimv (ksOf l) p.2.1) {s s' : ℂ}
    (h : Equal (F.env (ksOf (l.set m k))) s (N.split ⟨F, m, ksOf l⟩ H) s'
      (M.split ⟨F, m, ksOf l⟩ H)) :
    Equal (F.env (ksOf (l.set m (k + 1)))) s N s' M := by
  rw [← Spec.at_ksOf F m l _ hm] at h ⊢
  exact Equal.of_split _ H k N M hN hM hb h

/-- `Equal.of_zero` with the multiplicities written as a list. -/
theorem Equal.of_zero_at (F : Fam) (m H : ℕ) (l : List ℕ) (hm : m < l.length) (N M : Net)
    (hN : Spec.splitOK ⟨F, m, ksOf l⟩ H N = true) (hM : Spec.splitOK ⟨F, m, ksOf l⟩ H M = true)
    (hbnd : N.bnd.Perm M.bnd) {s s' : ℂ}
    (h : Equal (F.env (ksOf (l.set m 0))) s (N.zero ⟨F, m, ksOf l⟩) s'
      (M.zero ⟨F, m, ksOf l⟩)) :
    Equal (F.env (ksOf (l.set m 0))) s N s' M := by
  rw [← Spec.at_ksOf F m l _ hm] at h ⊢
  exact Equal.of_zero _ H N M hN hM hbnd h

end ArchytasZX

namespace ArchytasZX

/-- A local equation with its right side replaced node for node by equivalent nodes. -/
theorem LocalEq.equivR {E : Env} {L R R' : List Node} {iL iR iface : List ℕ} {kL kR : ℂ}
    (h : LocalEq E L iL R iR iface kL kR) (hl : R.length = R'.length)
    (he : (List.zip R R').all (fun q => nodeEquivB q.1 q.2) = true) :
    LocalEq E L iL R' iR iface kL kR := by
  intro τ hτ
  rw [h τ hτ]
  congr 2
  funext ρ
  exact prodVal_equivB E ρ R R' hl he

/-- A local equation with its left side replaced node for node by equivalent nodes. -/
theorem LocalEq.equivL {E : Env} {L L' R : List Node} {iL iR iface : List ℕ} {kL kR : ℂ}
    (h : LocalEq E L iL R iR iface kL kR) (hl : L'.length = L.length)
    (he : (List.zip L' L).all (fun q => nodeEquivB q.1 q.2) = true) :
    LocalEq E L' iL R iR iface kL kR := by
  intro τ hτ
  rw [← h τ hτ]
  congr 2
  funext ρ
  exact prodVal_equivB E ρ L' L hl he

end ArchytasZX
