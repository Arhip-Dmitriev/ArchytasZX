/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PairNet

/-!
# The pair box's network evaluated

`eval_pairNet`: when the template nodes holding the pair ports are Z spiders tied by a chain of
Z spiders to the slots `sL` and `sR` (`pinOK`), the network of the expansion at `K` copies is,
at every boundary value, the product of the template's network at each copy's slots times, per
pair `j < k`, the pair node's `F` entry at copy `k`'s slot `sR` and copy `j`'s slot `sL`.

The pieces: a Z spider is nonzero only where its legs agree (`tZ_const`), so on the support of
the copies every pair leg reads its pinned slot (`chain_val`) and the pair nodes give a constant
factor; summing out a label read once, by a Z spider, drops that leg (`sum_dropZ`); copies on
disjoint labels give a product (`sumOver_blocks`); each copy is the template renamed by `shJ`.
-/

namespace ArchytasZX

/-- `n` without its legs at the labels `gs`. -/
def Node.dropL (gs : List ℕ) (n : Node) : Node :=
  { n with outs := n.outs.filter (fun x => decide (x ∉ gs)),
           ins := n.ins.filter (fun x => decide (x ∉ gs)) }

theorem prod_flatMap' {α : Type*} (l : List α) (f : α → List ℂ) :
    (l.flatMap f).prod = (l.map fun a => (f a).prod).prod := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.flatMap_cons, List.prod_append, ih]

theorem mem_dropL (gs : List ℕ) (n : Node) (x : ℕ) :
    x ∈ (n.dropL gs).labels ↔ x ∈ n.labels ∧ x ∉ gs := by
  simp only [Node.dropL, Node.labels, List.mem_append, List.mem_filter, decide_eq_true_eq]
  tauto

theorem dropL_of_not (gs : List ℕ) (n : Node) (h : ∀ x ∈ n.labels, x ∉ gs) : n.dropL gs = n := by
  obtain ⟨k, os, is, d, ph⟩ := n
  simp only [Node.dropL, Node.mk.injEq, true_and, and_true]
  constructor
  · exact List.filter_eq_self.2 fun x hx => by
      simpa using h x (List.mem_append_left _ hx)
  · exact List.filter_eq_self.2 fun x hx => by
      simpa using h x (List.mem_append_right _ hx)

theorem dropL_nil (n : Node) : n.dropL [] = n := dropL_of_not [] n (by simp)

theorem dropL_cons (g : ℕ) (gs : List ℕ) (n : Node) :
    (n.dropL [g]).dropL gs = n.dropL (g :: gs) := by
  obtain ⟨k, os, is, d, ph⟩ := n
  simp only [Node.dropL, List.filter_filter, Node.mk.injEq, true_and, and_true]
  constructor <;> (congr 1; funext x; rw [Bool.eq_iff_iff]; simp; tauto)

theorem val_Z (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .Z) :
    val E σ n = tZ (E.dimv n.dim) (E.phase n.ph) (n.labels.map σ) := by
  obtain ⟨k, os, is, d, ph⟩ := n
  simp only at hk
  subst hk
  simp [val, Node.labels, List.map_append]

theorem tZ_const {d : ℕ} {φ : ℕ → ℝ} {vs : List ℕ} (h : tZ d φ vs ≠ 0) :
    ∃ c, ∀ v ∈ vs, v = c := by
  by_contra hc
  push_neg at hc
  apply h
  unfold tZ
  refine Finset.sum_eq_zero fun k _ => ?_
  obtain ⟨v, hv, hne⟩ := hc k
  rw [if_neg (fun h' => hne (h' v hv))]

theorem Z_eq (E : Env) (τ : ℕ → ℕ) (n : Node) (hk : n.kind = .Z) (h : val E τ n ≠ 0) {a b : ℕ}
    (ha : a ∈ n.labels) (hb : b ∈ n.labels) : τ a = τ b := by
  rw [val_Z E τ n hk] at h
  obtain ⟨c, hc⟩ := tZ_const h
  rw [hc _ (List.mem_map.2 ⟨a, ha, rfl⟩), hc _ (List.mem_map.2 ⟨b, hb, rfl⟩)]

/-- Summing a label of a Z spider over its dimension drops that leg. -/
theorem sum_Z (E : Env) (τ : ℕ → ℕ) (n : Node) (hk : n.kind = .Z) (g : ℕ) (hg : g ∈ n.labels)
    (Dg : ℕ) (hD : Dg = E.dimv n.dim) :
    ∑ v ∈ Finset.range Dg, val E (Function.update τ g v) n = val E τ (n.dropL [g]) := by
  have hk' : (n.dropL [g]).kind = .Z := hk
  rw [val_Z E τ _ hk']
  simp_rw [val_Z E _ n hk]
  show ∑ v ∈ Finset.range Dg, tZ (E.dimv n.dim) (E.phase n.ph) (n.labels.map (Function.update τ g v)) =
    tZ (E.dimv n.dim) (E.phase n.ph) ((n.dropL [g]).labels.map τ)
  unfold tZ
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c hc => ?_
  have key : ∀ v, (∀ u ∈ n.labels.map (Function.update τ g v), u = c) ↔
      v = c ∧ ∀ u ∈ (n.dropL [g]).labels.map τ, u = c := by
    intro v
    constructor
    · intro h
      refine ⟨by simpa using h v (List.mem_map.2 ⟨g, hg, by simp⟩), fun u hu => ?_⟩
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hu
      have hx' := (mem_dropL [g] n x).1 hx
      have hxg : x ≠ g := by simpa using hx'.2
      exact h _ (List.mem_map.2 ⟨x, hx'.1, by simp [Function.update_of_ne hxg]⟩)
    · rintro ⟨rfl, h⟩ u hu
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hu
      by_cases hxg : x = g
      · subst hxg; simp
      · rw [Function.update_of_ne hxg]
        exact h _ (List.mem_map.2 ⟨x, (mem_dropL [g] n x).2 ⟨hx, by simpa using hxg⟩, rfl⟩)
  simp_rw [key, ite_and]
  rw [Finset.sum_ite_eq']
  rw [if_pos (by rw [hD]; exact hc)]

theorem sumOne_dropZ (E : Env) (g : ℕ) : ∀ (ns : List Node) (τ : ℕ → ℕ),
    (∀ n ∈ ns, g ∈ n.labels → n.kind = .Z ∧ E.dimv n.dim = labDim E g) →
    ns.countP (fun n => decide (g ∈ n.labels)) = 1 →
    sumOver (labDim E) [g] (prodVal E ns) τ = prodVal E (ns.map (Node.dropL [g])) τ
  | [], τ, _, h => by simp at h
  | n :: ns, τ, hZ, h => by
    show ∑ v ∈ Finset.range (labDim E g), prodVal E (n :: ns) (Function.update τ g v) = _
    simp only [prodVal, List.map_cons, List.prod_cons]
    by_cases hg : g ∈ n.labels
    · have h0 : ns.countP (fun n => decide (g ∈ n.labels)) = 0 := by
        simpa [List.countP_cons, hg] using h
      have hns : ∀ m ∈ ns, g ∉ m.labels := by
        intro m hm hgm
        rw [List.countP_eq_zero] at h0
        exact h0 m hm (by simpa using hgm)
      have hr : ∀ v, (ns.map (val E (Function.update τ g v))).prod = (ns.map (val E τ)).prod :=
        fun v => prodVal_reads E ns _ _ fun x hx => by
          obtain ⟨m, hm, hxm⟩ := List.mem_flatMap.1 hx
          exact Function.update_of_ne (fun e => hns m hm (by rw [← e]; exact hxm)) _ _
      simp_rw [hr]
      rw [← Finset.sum_mul, sum_Z E τ n (hZ n List.mem_cons_self hg).1 g hg _
        (hZ n List.mem_cons_self hg).2.symm]
      congr 1
      rw [List.map_map]
      exact congrArg List.prod (List.map_congr_left fun m hm => by
        simp only [Function.comp]
        rw [dropL_of_not [g] m (fun x hx hx' => hns m hm (by simp at hx'; exact hx' ▸ hx))])
    · have h1 : ns.countP (fun n => decide (g ∈ n.labels)) = 1 := by
        simpa [List.countP_cons, hg] using h
      have hv : ∀ v, val E (Function.update τ g v) n = val E τ n := fun v =>
        val_reads E n _ _ fun x hx => Function.update_of_ne (fun e => hg (by rw [← e]; exact hx)) _ _
      simp_rw [hv]
      rw [← Finset.mul_sum, dropL_of_not [g] n (fun x hx hx' => hg (by simp at hx'; exact hx' ▸ hx))]
      congr 1
      have ih := sumOne_dropZ E g ns τ (fun m hm => hZ m (List.mem_cons_of_mem _ hm)) h1
      simpa [sumOver, prodVal] using ih

/-- Summing labels each read once, by a Z spider of their dimension, drops their legs. -/
theorem sum_dropZ (E : Env) : ∀ (gs : List ℕ) (ns : List Node) (τ : ℕ → ℕ), gs.Nodup →
    (∀ g ∈ gs, (∀ n ∈ ns, g ∈ n.labels → n.kind = .Z ∧ E.dimv n.dim = labDim E g) ∧
      ns.countP (fun n => decide (g ∈ n.labels)) = 1) →
    sumOver (labDim E) gs (prodVal E ns) τ = prodVal E (ns.map (Node.dropL gs)) τ
  | [], ns, τ, _, _ => by
    show prodVal E ns τ = prodVal E (ns.map (Node.dropL [])) τ
    rw [List.map_congr_left (fun n _ => dropL_nil n)]; simp
  | g :: gs, ns, τ, hnd, h => by
    rw [sumOver_perm _ (List.perm_append_singleton g gs).symm, sumOver_append]
    rw [sumOver_congr _ _ _ (prodVal E (ns.map (Node.dropL [g]))) τ
      (fun ρ _ _ => sumOne_dropZ E g ns ρ (h g List.mem_cons_self).1 (h g List.mem_cons_self).2)]
    have hg : ∀ g' ∈ gs, g' ≠ g := fun g' hg' e => (List.nodup_cons.1 hnd).1 (e ▸ hg')
    rw [sum_dropZ E gs (ns.map (Node.dropL [g])) τ (List.nodup_cons.1 hnd).2 fun g' hg' => ⟨?_, ?_⟩]
    · rw [List.map_map]
      exact congrArg (fun l => prodVal E l τ) (List.map_congr_left fun n _ => dropL_cons g gs n)
    · intro n hn hgn
      obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hn
      have := (mem_dropL [g] m g').1 hgn
      exact (h g' (List.mem_cons_of_mem _ hg')).1 m hm this.1
    · rw [List.countP_map]
      refine ((List.countP_congr fun m _ => ?_).trans (h g' (List.mem_cons_of_mem _ hg')).2)
      simp only [Function.comp, decide_eq_true_eq, mem_dropL, List.mem_singleton, hg g' hg',
        not_false_eq_true, and_true]

/-- Blocks of nodes on disjoint summed labels, none reading another's: the sum is the product of
the blocks' sums. -/
theorem sumOver_blocks (E : Env) : ∀ (bs : List (List ℕ × List Node)) (τ : ℕ → ℕ),
    bs.Pairwise (fun a b => (∀ x ∈ labelsOf a.2, x ∉ b.1) ∧ ∀ x ∈ labelsOf b.2, x ∉ a.1) →
    sumOver (labDim E) (bs.flatMap (·.1)) (prodVal E (bs.flatMap (·.2))) τ =
      (bs.map fun b => sumOver (labDim E) b.1 (prodVal E b.2) τ).prod
  | [], τ, _ => by simp [sumOver, prodVal]
  | a :: bs, τ, h => by
    rw [List.pairwise_cons] at h
    simp only [List.flatMap_cons, List.map_cons, List.prod_cons]
    rw [sumOver_append]
    have hp : prodVal E (a.2 ++ bs.flatMap (·.2)) =
        fun ρ => prodVal E a.2 ρ * prodVal E (bs.flatMap (·.2)) ρ := funext (prodVal_append E _ _)
    rw [hp]
    have h1 : ∀ x ∈ labelsOf a.2, x ∉ bs.flatMap (·.1) := fun x hx hm => by
      obtain ⟨c, hc, hxc⟩ := List.mem_flatMap.1 hm
      exact (h.1 c hc).1 x hx hxc
    set C := (bs.map fun b => sumOver (labDim E) b.1 (prodVal E b.2) τ).prod
    rw [sumOver_congr _ _ _ (fun ρ => C * prodVal E a.2 ρ) τ (fun ρ hρ _ => ?_), ← mul_sumOver,
      mul_comm]
    rw [sumOver_const_mul _ _ _ _ _ (prodVal_reads E a.2) h1, sumOver_blocks E bs ρ h.2, mul_comm]
    congr 1
    refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    exact sumOver_ext _ _ _ _ (prodVal_reads E c.2) _ _ fun x hx _ => hρ x ((h.1 c hc).2 x hx)

/-- A network's nodes renamed by an injective map, summed over its renamed summed labels, is the
network at the boundary read through the map. -/
theorem sumOver_rename (E : Env) (hE : ∀ x y, labDim E x = labDim E y) (f : ℕ → ℕ)
    (hf : Function.Injective f) (N : Net) (τ : ℕ → ℕ) :
    sumOver (labDim E) (N.int.map f) (prodVal E (N.nodes.map (Node.rename f))) τ = eval E N (τ ∘ f) := by
  rw [show prodVal E (N.nodes.map (Node.rename f)) = fun ρ => prodVal E N.nodes (ρ ∘ f) from
    funext fun ρ => prodVal_rename E f ρ N.nodes]
  exact sumOver_map _ f hf _ (fun x _ => hE _ _) _ τ

/-- A chain certificate: each entry `(i, x)` a Z node `i` holding `x`, consecutive entries
sharing the earlier label, the last label the slot. -/
def chainOK (ns : List Node) : List (ℕ × ℕ) → ℕ → Bool
  | [], _ => false
  | [(i, x)], s => decide (x = s) && (match ns[i]? with
      | some n => decide (n.kind = .Z) && decide (x ∈ n.labels)
      | none => false)
  | (i, x) :: (i', x') :: r, s =>
      (match ns[i]? with
        | some n => decide (n.kind = .Z) && decide (x ∈ n.labels)
        | none => false) &&
      (match ns[i']? with
        | some n => decide (n.kind = .Z) && decide (x ∈ n.labels)
        | none => false) && chainOK ns ((i', x') :: r) s

theorem hasZ_spec {ns : List Node} {i x : ℕ}
    (h : (match ns[i]? with
      | some n => decide (n.kind = .Z) && decide (x ∈ n.labels)
      | none => false) = true) :
    ∃ hi : i < ns.length, ns[i].kind = .Z ∧ x ∈ ns[i].labels := by
  rcases e : ns[i]? with _ | n
  · rw [e] at h; simp at h
  · rw [e] at h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨hi, he⟩ := List.getElem?_eq_some_iff.1 e
    exact ⟨hi, he ▸ h.1, he ▸ h.2⟩

/-- Along a chain certificate on `ns`, in a node list `cs` holding `ns`'s labels moved by `f`,
where every node is nonzero, every label of the first node reads the slot. -/
theorem chain_val (E : Env) (ns cs : List Node) (hlen : cs.length = ns.length) (f : ℕ → ℕ)
    (hlab : ∀ i (hi : i < ns.length) x, x ∈ ns[i].labels → f x ∈ (cs[i]'(hlen ▸ hi)).labels)
    (hkind : ∀ i (hi : i < ns.length), (cs[i]'(hlen ▸ hi)).kind = ns[i].kind)
    (τ : ℕ → ℕ) (hnz : ∀ n ∈ cs, val E τ n ≠ 0) :
    ∀ (cert : List (ℕ × ℕ)) (s i x : ℕ), chainOK ns cert s = true → cert.head? = some (i, x) →
      ∃ hi : i < ns.length, ∀ y ∈ (cs[i]'(hlen ▸ hi)).labels, τ y = τ (f s)
  | [], _, _, _, h, _ => by simp [chainOK] at h
  | [(i, x)], s, i', x', h, hh => by
    simp only [List.head?_cons, Option.some.injEq, Prod.mk.injEq] at hh
    obtain ⟨rfl, rfl⟩ := hh
    simp only [chainOK, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨rfl, h2⟩ := h
    obtain ⟨hi, hz, hx⟩ := hasZ_spec h2
    refine ⟨hi, fun y hy => ?_⟩
    exact Z_eq E τ _ (by rw [hkind i hi]; exact hz) (hnz _ (List.getElem_mem _)) hy (hlab i hi _ hx)
  | (i, x) :: (i₂, x₂) :: r, s, i', x', h, hh => by
    simp only [List.head?_cons, Option.some.injEq, Prod.mk.injEq] at hh
    obtain ⟨rfl, rfl⟩ := hh
    simp only [chainOK, Bool.and_eq_true] at h
    obtain ⟨⟨h1, h2⟩, h3⟩ := h
    obtain ⟨hi, hz, hx⟩ := hasZ_spec h1
    obtain ⟨hi₂, hz₂, hx₂⟩ := hasZ_spec h2
    obtain ⟨hi₂', ih⟩ := chain_val E ns cs hlen f hlab hkind τ hnz ((i₂, x₂) :: r) s i₂ x₂ h3 rfl
    refine ⟨hi, fun y hy => ?_⟩
    rw [Z_eq E τ _ (by rw [hkind i hi]; exact hz) (hnz _ (List.getElem_mem _)) hy (hlab i hi _ hx)]
    exact ih _ (hlab i₂ hi₂ _ hx₂)

namespace Diagram

variable {D : Diagram} {p : DPair} {b : DBox}

/-- The template node of index `i` holds the port `q`, and a chain certificate ties it to the
slot `s`. -/
def pinOK (D : Diagram) (p : DPair) (q : Port) (cert : List (ℕ × ℕ)) (s : ℕ) : Bool :=
  decide (s < D.outs.length) &&
  (match cert.head? with
    | some (i, _) => decide ((D.nodes[i]?.map (·.id)) = some q.node)
    | none => false) &&
  chainOK (D.tmpl p).nodes cert s

theorem label_lt (s : PairSpec D p) (q : Port) (h : D.okPort q = true) :
    D.label D.outs.length q < D.outs.length + D.wires.length := by
  have hb : D.bnd.length = D.outs.length := by simp [bnd, s.ins]
  rcases hloc : D.locate q with _ | (i | j)
  · simp [okPort, hloc] at h
  · simp only [okPort, hloc, decide_eq_true_eq] at h
    simp only [label, hloc]; omega
  · simp only [okPort, hloc, decide_eq_true_eq] at h
    simp only [label, hloc]; omega

theorem tNode_lt (s : PairSpec D p) {n : DNode} (hn : n ∈ D.nodes) :
    ∀ x ∈ (D.tNode p D.outs.length n).labels, x < D.outs.length + D.wires.length := by
  intro x hx
  simp only [tNode, Node.labels, List.mem_append, List.mem_map, List.mem_filter,
    List.mem_range] at hx
  rcases hx with ⟨k, ⟨hk, hkp⟩, rfl⟩ | ⟨k, ⟨hk, hkp⟩, rfl⟩
  · exact label_lt s _ (s.legsO n hn k hk hkp)
  · exact label_lt s _ (s.legsI n hn k hk hkp)

theorem shJ_inj (K m L j : ℕ) (hj : j < K) : Function.Injective (shJ K m L j) := by
  intro x y h
  have hjm : (j + 1) * m ≤ K * m := Nat.mul_le_mul_right _ hj
  rw [Nat.succ_mul] at hjm
  unfold shJ at h
  split_ifs at h <;> omega

/-- Copy `j`'s pair legs. -/
def pGam (K j : ℕ) : List ℕ :=
  (later K j).map (gL K D.outs.length D.wires.length j) ++
    (List.range j).map (fun i => gR K D.outs.length D.wires.length i j)

theorem exL_sub (K j x : ℕ) (out : Bool) : ∀ y ∈ D.exL p K j x out, y ∈ D.pGam K j := by
  intro y hy
  simp only [exL, List.mem_append] at hy
  simp only [pGam, List.mem_append]
  rcases hy with hy | hy
  · split_ifs at hy
    · exact Or.inl hy
    · simp at hy
  · split_ifs at hy
    · exact Or.inr hy
    · simp at hy

theorem shJ_lt_pG (K m L j x : ℕ) (hj : j < K) (hx : x < m + L) : shJ K m L j x < pG K m L := by
  have hjm : (j + 1) * m ≤ K * m := Nat.mul_le_mul_right _ hj
  have hjL : (j + 1) * L ≤ K * L := Nat.mul_le_mul_right _ hj
  rw [Nat.succ_mul] at hjm hjL
  unfold shJ pG
  split_ifs <;> omega

theorem pGam_ge (K j y : ℕ) (hy : y ∈ D.pGam K j) : pG K D.outs.length D.wires.length ≤ y := by
  simp only [pGam, List.mem_append, List.mem_map] at hy
  rcases hy with ⟨k, -, rfl⟩ | ⟨i, -, rfl⟩
  · simp [gL]
  · simp [gR]; omega

theorem mem_pNode (K j : ℕ) (n : DNode) (y : ℕ) :
    y ∈ (D.pNode p K j n).labels ↔
      y ∈ (D.tNode p D.outs.length n).labels.map (shJ K D.outs.length D.wires.length j) ∨
      y ∈ D.exL p K j n.id true ∨ y ∈ D.exL p K j n.id false := by
  simp only [pNode, Node.labels, List.map_append, List.mem_append]
  tauto

theorem drop_pNode (s : PairSpec D p) (K j : ℕ) (hj : j < K) {n : DNode} (hn : n ∈ D.nodes) :
    (D.pNode p K j n).dropL (D.pGam K j) =
      (D.tNode p D.outs.length n).rename (shJ K D.outs.length D.wires.length j) := by
  have hlow : ∀ x ∈ (D.tNode p D.outs.length n).labels,
      shJ K D.outs.length D.wires.length j x ∉ D.pGam K j := fun x hx hm =>
    absurd (pGam_ge K j _ hm) (not_le.2 (shJ_lt_pG _ _ _ _ _ hj (tNode_lt s hn x hx)))
  simp only [pNode, Node.dropL, Node.rename, List.filter_append]
  congr 1
  · rw [List.filter_eq_self.2 fun y hy => ?_, List.filter_eq_nil_iff.2 fun y hy => ?_,
      List.append_nil]
    · simpa using exL_sub K j n.id true y hy
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
      simpa using hlow x (List.mem_append_left _ hx)
  · rw [List.filter_eq_self.2 fun y hy => ?_, List.filter_eq_nil_iff.2 fun y hy => ?_,
      List.append_nil]
    · simpa using exL_sub K j n.id false y hy
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
      simpa using hlow x (List.mem_append_right _ hx)

theorem count_id (l : List DNode) (hnd : (l.map (·.id)).Nodup) (x : ℕ) (hx : ∃ n ∈ l, n.id = x) :
    l.countP (fun n => decide (n.id = x)) = 1 := by
  induction l with
  | nil => simp at hx
  | cons a l ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    rw [List.countP_cons]
    by_cases ha : a.id = x
    · subst ha
      have : l.countP (fun n => decide (n.id = a.id)) = 0 :=
        List.countP_eq_zero.2 fun n hn e => hnd.1 ⟨n, hn, by simpa using e⟩
      simp [this]
    · obtain ⟨n, hn, e⟩ := hx
      rcases List.mem_cons.1 hn with rfl | hn'
      · exact absurd e ha
      · simp [ha, ih hnd.2 ⟨n, hn', e⟩]


theorem gL_ne_gR (K m L a b c e : ℕ) : gL K m L a b ≠ gR K m L c e := by
  unfold gL gR; omega

theorem gL_inj (K m L j k k' : ℕ) (h : gL K m L j k = gL K m L j k') : k = k' := by
  unfold gL at h
  have : Nat.pair j k = Nat.pair j k' := by omega
  exact (Nat.pair_eq_pair.1 this).2

theorem gR_inj (K m L j i i' : ℕ) (h : gR K m L i j = gR K m L i' j) : i = i' := by
  unfold gR at h
  have : Nat.pair i j = Nat.pair i' j := by omega
  exact (Nat.pair_eq_pair.1 this).1

theorem side_iff (q : Port) (x : ℕ) :
    ((x = q.node ∧ true = q.out) ∨ (x = q.node ∧ false = q.out)) ↔ x = q.node := by
  cases q.out <;> simp

theorem gL_mem (s : PairSpec D p) (K j k : ℕ) (hj : j < K) (hk : k ∈ later K j) {n : DNode}
    (hn : n ∈ D.nodes) :
    gL K D.outs.length D.wires.length j k ∈ (D.pNode p K j n).labels ↔ n.id = p.left.node := by
  have t1 : gL K D.outs.length D.wires.length j k ∉
      (D.tNode p D.outs.length n).labels.map (shJ K D.outs.length D.wires.length j) := fun hm => by
    obtain ⟨x, hx, e⟩ := List.mem_map.1 hm
    have := shJ_lt_pG _ _ _ _ _ hj (tNode_lt s hn x hx)
    rw [e] at this; unfold gL at this; omega
  have hl : ∀ out, gL K D.outs.length D.wires.length j k ∈ D.exL p K j n.id out ↔
      (n.id = p.left.node ∧ out = p.left.out) := by
    intro out
    have a1 : gL K D.outs.length D.wires.length j k ∈
        (later K j).map (gL K D.outs.length D.wires.length j) := List.mem_map.2 ⟨k, hk, rfl⟩
    have a2 : gL K D.outs.length D.wires.length j k ∉
        (List.range j).map (fun i => gR K D.outs.length D.wires.length i j) := fun hm => by
      obtain ⟨i, -, e⟩ := List.mem_map.1 hm
      exact gL_ne_gR _ _ _ _ _ _ _ e.symm
    simp only [exL, List.mem_append]
    split_ifs with h1 h2 h2 <;> simp [a1, a2, h1]
  rw [mem_pNode, hl, hl]
  simp only [t1, false_or]
  exact side_iff p.left n.id

theorem gR_mem (s : PairSpec D p) (K i j : ℕ) (hj : j < K) (hi : i < j) {n : DNode}
    (hn : n ∈ D.nodes) :
    gR K D.outs.length D.wires.length i j ∈ (D.pNode p K j n).labels ↔ n.id = p.right.node := by
  have t1 : gR K D.outs.length D.wires.length i j ∉
      (D.tNode p D.outs.length n).labels.map (shJ K D.outs.length D.wires.length j) := fun hm => by
    obtain ⟨x, hx, e⟩ := List.mem_map.1 hm
    have := shJ_lt_pG _ _ _ _ _ hj (tNode_lt s hn x hx)
    rw [e] at this; unfold gR at this; omega
  have hl : ∀ out, gR K D.outs.length D.wires.length i j ∈ D.exL p K j n.id out ↔
      (n.id = p.right.node ∧ out = p.right.out) := by
    intro out
    have a1 : gR K D.outs.length D.wires.length i j ∈
        (List.range j).map (fun i => gR K D.outs.length D.wires.length i j) :=
      List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩
    have a2 : gR K D.outs.length D.wires.length i j ∉
        (later K j).map (gL K D.outs.length D.wires.length j) := fun hm => by
      obtain ⟨k, -, e⟩ := List.mem_map.1 hm
      exact gL_ne_gR _ _ _ _ _ _ _ e
    simp only [exL, List.mem_append]
    split_ifs with h1 h2 h2 <;> simp [a1, a2, h2]
  rw [mem_pNode, hl, hl]
  simp only [t1, false_or]
  exact side_iff p.right n.id

/-- Every label of copy `j` is a slot below `K m` or one of copy `j`'s summed labels. -/
theorem lab_cases (s : PairSpec D p) (K j : ℕ) (hj : j < K) {n : DNode} (hn : n ∈ D.nodes)
    (y : ℕ) (hy : y ∈ (D.pNode p K j n).labels) :
    y < K * D.outs.length ∨ y ∈ D.pInt K j := by
  rcases (mem_pNode K j n y).1 hy with hy | hy | hy
  · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
    have hxl := tNode_lt s hn x hx
    have hjm : (j + 1) * D.outs.length ≤ K * D.outs.length := Nat.mul_le_mul_right _ hj
    rw [Nat.succ_mul] at hjm
    by_cases hxm : x < D.outs.length
    · left; simp only [shJ, hxm, ite_true]; omega
    · right
      simp only [pInt, List.mem_append, List.mem_map, List.mem_range]
      refine Or.inl (Or.inl ⟨x - D.outs.length, by omega, ?_⟩)
      simp only [shJ, hxm, ite_false]
  all_goals
    right
    have := exL_sub K j n.id _ y hy
    simp only [pGam, List.mem_append] at this
    simp only [pInt, List.mem_append]
    tauto

theorem pin_at (s : PairSpec D p) (E : Env) (K j : ℕ) (hj : j < K) (q : Port)
    (cert : List (ℕ × ℕ)) (sq : ℕ) (hc : D.pinOK p q cert sq = true) (τ : ℕ → ℕ)
    (hnz : ∀ n ∈ D.nodes.map (D.pNode p K j), val E τ n ≠ 0) :
    ∃ n ∈ D.nodes, n.id = q.node ∧ n.kind = .Z ∧
      ∀ y ∈ (D.pNode p K j n).labels, τ y = τ (j * D.outs.length + sq) := by
  simp only [pinOK, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨hs, hh⟩, hch⟩ := hc
  rcases e : cert.head? with _ | ⟨i, x⟩
  · rw [e] at hh; simp at hh
  · rw [e] at hh
    simp only [decide_eq_true_eq] at hh
    have hlen : (D.nodes.map (D.pNode p K j)).length = (D.tmpl p).nodes.length := by simp [tmpl]
    obtain ⟨hi, hv⟩ := chain_val E (D.tmpl p).nodes (D.nodes.map (D.pNode p K j)) hlen
      (shJ K D.outs.length D.wires.length j)
      (fun i hi x hx => by
        simp only [tmpl, List.getElem_map] at hx ⊢
        exact (mem_pNode K j _ _).2 (Or.inl (List.mem_map.2 ⟨x, hx, rfl⟩)))
      (fun i hi => by simp [tmpl, pNode, tNode])
      τ hnz cert sq i x hch e
    have hi' : i < D.nodes.length := by simpa [tmpl] using hi
    obtain ⟨hi2, hz, -⟩ := hasZ_spec (ns := (D.tmpl p).nodes) (i := i) (x := x) (by
      have := hch
      cases cert with
      | nil => simp at e
      | cons a r =>
        simp only [List.head?_cons, Option.some.injEq] at e
        subst e
        cases r with
        | nil => simp only [chainOK, Bool.and_eq_true] at this; exact this.2
        | cons a' r => simp only [chainOK, Bool.and_eq_true] at this; exact this.1.1)
    refine ⟨D.nodes[i], List.getElem_mem hi', ?_, ?_, fun y hy => ?_⟩
    · simpa [List.getElem?_eq_getElem hi'] using hh
    · simpa [tmpl, tNode] using hz
    · have := hv y (by simpa using hy)
      rw [this]
      simp [shJ, hs]

theorem uniq_id (s : PairSpec D p) {n n' : DNode} (hn : n ∈ D.nodes) (hn' : n' ∈ D.nodes)
    (h : n.id = n'.id) : n = n' :=
  List.inj_on_of_nodup_map s.nodup hn hn' h

/-- The pair box's network at `K` copies: the template at each copy's slots times the pair
nodes' entries at their pinned slots. -/
theorem eval_pairNet (h : D.PairOK p b = true) (E : Env) (hE : ∀ x, labDim E x = E.dimv 0)
    (K : ℕ) (cl cr : List (ℕ × ℕ)) (sL sR : ℕ) (hcl : D.pinOK p p.left cl sL = true)
    (hcr : D.pinOK p p.right cr sR = true) (σ : ℕ → ℕ) :
    ArchytasZX.eval E (D.pairNet p K) σ =
      ((List.range K).map fun j =>
        ArchytasZX.eval E (D.tmpl p) (σ ∘ shJ K D.outs.length D.wires.length j)).prod *
      ((List.range K).map fun j => ((later K j).map fun k =>
        fou (E.dimv 0) (σ (k * D.outs.length + sR)) (σ (j * D.outs.length + sL))).prod).prod := by
  have s := PairOK.spec h
  have hE' : ∀ x y, labDim E x = labDim E y := fun x y => by rw [hE x, hE y]
  set m := D.outs.length
  set L := D.wires.length
  set C : ℕ → List Node := fun j => D.nodes.map (D.pNode p K j)
  set Cs := (List.range K).flatMap C
  set PF := ((List.range K).map fun j => ((later K j).map fun k =>
    fou (E.dimv 0) (σ (k * m + sR)) (σ (j * m + sL))).prod).prod
  have hint : ∀ x ∈ (List.range K).flatMap (D.pInt K), K * m ≤ x := fun x hx => by
    obtain ⟨j, hj, hx⟩ := List.mem_flatMap.1 hx
    exact (mem_pInt K j x (List.mem_range.1 hj) hx).1
  have hslot : ∀ τ : ℕ → ℕ, (∀ x ∉ (List.range K).flatMap (D.pInt K), τ x = σ x) →
      ∀ x < K * m, τ x = σ x := fun τ hτ x hx => hτ x fun hm => by have := hint x hm; omega
  have hpt : ∀ τ : ℕ → ℕ, (∀ x ∉ (List.range K).flatMap (D.pInt K), τ x = σ x) →
      prodVal E (D.pairNet p K).nodes τ = prodVal E Cs τ * PF := by
    intro τ hτ
    show prodVal E (Cs ++ _) τ = _
    rw [prodVal_append]
    by_cases h0 : prodVal E Cs τ = 0
    · simp [h0]
    congr 1
    have hnz : ∀ j < K, ∀ n ∈ C j, val E τ n ≠ 0 := fun j hj n hn h' =>
      h0 (List.prod_eq_zero (List.mem_map.2 ⟨n, List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, hn⟩, h'⟩))
    have pl : ∀ j < K, ∀ k ∈ later K j, τ (gL K m L j k) = σ (j * m + sL) := by
      intro j hj k hk
      obtain ⟨n, hn, hid, -, hv⟩ := pin_at s E K j hj p.left cl sL hcl τ (hnz j hj)
      rw [hv _ ((gL_mem s K j k hj hk hn).2 hid)]
      have hsl : sL < m := by simp only [pinOK, Bool.and_eq_true, decide_eq_true_eq] at hcl; exact hcl.1.1
      have hjm : (j + 1) * m ≤ K * m := Nat.mul_le_mul_right _ hj
      rw [Nat.succ_mul] at hjm
      exact hslot τ hτ _ (lt_of_lt_of_le (Nat.add_lt_add_left (by assumption) _) hjm)
    have pr : ∀ k < K, ∀ j < k, τ (gR K m L j k) = σ (k * m + sR) := by
      intro k hk j hjk
      obtain ⟨n, hn, hid, -, hv⟩ := pin_at s E K k hk p.right cr sR hcr τ (hnz k hk)
      rw [hv _ ((gR_mem s K j k hk hjk hn).2 hid)]
      have hsr : sR < m := by simp only [pinOK, Bool.and_eq_true, decide_eq_true_eq] at hcr; exact hcr.1.1
      have hkm : (k + 1) * m ≤ K * m := Nat.mul_le_mul_right _ hk
      rw [Nat.succ_mul] at hkm
      exact hslot τ hτ _ (lt_of_lt_of_le (Nat.add_lt_add_left (by assumption) _) hkm)
    simp only [prodVal, List.map_flatMap, prod_flatMap', List.map_map]
    refine congrArg List.prod (List.map_congr_left fun j hj => congrArg List.prod
      (List.map_congr_left fun k hk => ?_))
    have hj' := List.mem_range.1 hj
    have hk' : j + 1 ≤ k ∧ k < j + 1 + (K - 1 - j) := by simpa [later] using hk
    simp only [Function.comp, pFNode, val, s.fk, s.fd, List.map_cons, List.map_nil]
    exact congrArg₂ (fou (E.dimv 0)) (pr k (by omega) j (by omega)) (pl j hj' k hk)
  unfold ArchytasZX.eval
  show sumOver (labDim E) ((List.range K).flatMap (D.pInt K)) (prodVal E (D.pairNet p K).nodes) σ = _
  rw [sumOver_congr _ _ _ (fun τ => prodVal E Cs τ * PF) σ fun τ hτ _ => hpt τ hτ]
  rw [show (fun τ => prodVal E Cs τ * PF) = fun τ => PF * prodVal E Cs τ from
    funext fun _ => mul_comm _ _, ← mul_sumOver, mul_comm]
  congr 1
  -- the copies' sum
  have hbs := sumOver_blocks E ((List.range K).map fun j => (D.pInt K j, C j)) σ (by
    rw [List.pairwise_map]
    refine List.nodup_range.pairwise_of_forall_ne fun i hi j hj hij => ?_
    have hi' := List.mem_range.1 hi
    have hj' := List.mem_range.1 hj
    have side : ∀ a b, a < K → b < K → a ≠ b → ∀ x ∈ labelsOf (C a), x ∉ D.pInt K b :=
      fun a b ha hb hab x hx hm => by
        obtain ⟨n', hn', hx'⟩ := List.mem_flatMap.1 hx
        obtain ⟨n, hn, rfl⟩ := List.mem_map.1 hn'
        rcases lab_cases s K a ha hn x hx' with hlt | hin
        · have := (mem_pInt K b x hb hm).1; omega
        · exact hab ((mem_pInt K a x ha hin).2.symm.trans (mem_pInt K b x hb hm).2)
    exact ⟨side i j hi' hj' hij, side j i hj' hi' (Ne.symm hij)⟩)
  simp only [List.flatMap_map] at hbs
  show sumOver (labDim E) ((List.range K).flatMap (D.pInt K)) (prodVal E Cs) σ = _
  rw [hbs, List.map_map]
  refine congrArg List.prod (List.map_congr_left fun j hj => ?_)
  have hj' := List.mem_range.1 hj
  simp only [Function.comp]
  have hsplit : D.pInt K j = (List.range L).map (fun i => K * m + j * L + i) ++ D.pGam K j := by
    simp only [pInt, pGam, List.append_assoc]; rfl
  have hnd : (D.pGam K j).Nodup := by
    have := pInt_nodup (D := D) K j hj'
    rw [hsplit] at this
    exact this.of_append_right
  rw [hsplit, sumOver_append]
  rw [sumOver_congr _ _ _ (prodVal E ((C j).map (Node.dropL (D.pGam K j)))) σ
    (fun ρ _ _ => sum_dropZ E _ _ ρ hnd fun g hg => ?_)]
  · have hC : (C j).map (Node.dropL (D.pGam K j)) =
        (D.tmpl p).nodes.map (Node.rename (shJ K m L j)) := by
      simp only [C, tmpl, List.map_map]
      exact List.map_congr_left fun n hn => drop_pNode s K j hj' hn
    have hL : (List.range L).map (fun i => K * m + j * L + i) = (D.tmpl p).int.map (shJ K m L j) := by
      simp only [tmpl, List.map_map]
      exact List.map_congr_left fun i _ => (shJ_wire K m L j i).symm
    rw [hC, hL, sumOver_rename E hE' _ (shJ_inj K m L j hj')]; rfl
  · -- the pair legs: each read once, by a Z spider of its dimension
    have key : ∃ q : Port, (∃ cert sq, D.pinOK p q cert sq = true) ∧
        ∀ n ∈ D.nodes, (g ∈ (D.pNode p K j n).labels ↔ n.id = q.node) := by
      simp only [pGam, List.mem_append, List.mem_map, List.mem_range] at hg
      rcases hg with ⟨k, hk, rfl⟩ | ⟨i, hi, rfl⟩
      · exact ⟨p.left, ⟨cl, sL, hcl⟩, fun n hn => gL_mem s K j k hj' hk hn⟩
      · exact ⟨p.right, ⟨cr, sR, hcr⟩, fun n hn => gR_mem s K i j hj' hi hn⟩
    obtain ⟨q, ⟨cert, sq, hc⟩, hq⟩ := key
    -- the node of `q` exists and is a Z spider
    simp only [pinOK, Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨-, hh⟩, hch⟩ := hc
    rcases e : cert.head? with _ | ⟨i0, x0⟩
    · rw [e] at hh; simp at hh
    rw [e] at hh
    simp only [decide_eq_true_eq] at hh
    have hi0 : i0 < D.nodes.length := by
      by_contra hc; rw [List.getElem?_eq_none (by omega)] at hh; simp at hh
    have hid : D.nodes[i0].id = q.node := by simpa [List.getElem?_eq_getElem hi0] using hh
    have hz : D.nodes[i0].kind = .Z := by
      obtain ⟨hi2, hz, -⟩ := hasZ_spec (ns := (D.tmpl p).nodes) (i := i0) (x := x0) (by
        cases cert with
        | nil => simp at e
        | cons a r =>
          simp only [List.head?_cons, Option.some.injEq] at e
          subst e
          cases r with
          | nil => simp only [chainOK, Bool.and_eq_true] at hch; exact hch.2
          | cons a' r => simp only [chainOK, Bool.and_eq_true] at hch; exact hch.1.1)
      simpa [tmpl, tNode] using hz
    refine ⟨fun n hn hgn => ?_, ?_⟩
    · obtain ⟨n0, hn0, rfl⟩ := List.mem_map.1 hn
      have e0 : n0 = D.nodes[i0] :=
        uniq_id s hn0 (List.getElem_mem hi0) (((hq n0 hn0).1 hgn).trans hid.symm)
      subst e0
      refine ⟨by simpa [pNode, tNode] using hz, ?_⟩
      simp only [pNode, tNode]
      rw [s.dims _ (List.getElem_mem hi0), hE]
    · rw [List.countP_map]
      rw [List.countP_congr (q := fun n => decide (n.id = q.node)) fun n hn => by
        simpa using hq n hn]
      exact count_id D.nodes s.nodup q.node ⟨_, List.getElem_mem hi0, hid⟩

end Diagram

end ArchytasZX
