/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.OrderK
import ArchytasZX.PairBox

/-!
# Boxed diagrams whose copies differ

A `FamDiagram` is a boxed `Diagram` at one copy count `n` (every box at `n` copies) with chain
links, pair boxes and indices read at `n` and a copy coordinate `c`: `c = j` on copy `j` of a
boxed node, `c = k - j` on the node of the pair `(j, k)`, `c = 0` outside the boxes.

`famExpand F n` is `F.base.expandK` at `n` copies with
* copy `j` of the node `x` (`cp`): its phase index, dimension index and leg dimension indices
  read by `ph`, `dim`, `legs` at `(n, j)`, and each pair box's `left` leg grown to `n - 1 - j`
  legs and `right` leg to `j` legs in place (`cnt`, `growC`, `posC`);
* a link `l`: copy `j`'s `l1` wired to copy `j + 1`'s `l2`, `e2` to copy `0`'s `l2`, copy
  `n - 1`'s `l1` to `e1`, and `e2` to `e1` at `n = 0`;
* the pair `(j, k)`, `j < k`, of each pair box: a copy of its node, input wired to copy `j`'s
  grown `left` leg `k - j - 1`, output to copy `k`'s grown `right` leg `j`.

`famExpand_plain`: with no links, pairs or indices it is `F.base.expandK`.
-/

namespace ArchytasZX

/-- A chain link: copy `j`'s `l1` meets copy `j + 1`'s `l2`, the last copy's `l1` meets `e1`,
the first copy's `l2` meets `e2`, and at no copies `e2` meets `e1`. -/
structure FLink where
  l1 : Port
  l2 : Port
  e1 : Port
  e2 : Port
  deriving DecidableEq, Repr

/-- An index at the copy count `n` and the copy coordinate `c`. -/
abbrev Ix := ℕ → ℕ → ℕ

/-- A boxed diagram with chain links, pair boxes, and phase, dimension and leg dimension indices
keyed by template node or leg. -/
structure FamDiagram where
  base : Diagram
  links : List FLink := []
  pairs : List DPair := []
  ph : List (ℕ × Ix) := []
  dim : List (ℕ × Ix) := []
  legs : List (Port × Ix) := []

namespace FamDiagram

variable (F : FamDiagram) (n : ℕ)

/-- The index keyed `x` in `l` at `(n, c)`, else `d`. -/
def ixAt {α : Type} [BEq α] (l : List (α × Ix)) (x : α) (c d : ℕ) : ℕ :=
  match l.lookup x with
  | some f => f n c
  | none => d

/-- The plain expansion. -/
def X : Diagram := F.base.expandK fun _ => n

/-- Each node of `X` in order: its template node and copy. -/
def org : List (ℕ × ℕ) := F.base.nodes.flatMap fun x =>
  if F.base.boxes.any (x.id ∈ ·.nodes) then (List.range n).map (x.id, ·) else [(x.id, 0)]

/-- The template node and copy of the node `y` of `X`. -/
def orgOf (y : ℕ) : ℕ × ℕ :=
  ((((F.X n).nodes.zip (F.org n)).find? (·.1.id = y)).map (·.2)).getD (y, 0)

/-- The id of copy `j` of the template node `x` in `X`. -/
def cp (x j : ℕ) : ℕ :=
  ((((F.X n).nodes.zip (F.org n)).find? (·.2 = (x, j))).map (·.1.id)).getD 0

/-- The number of legs the template leg `q` becomes in copy `j`. -/
def cnt (j : ℕ) (q : Port) : ℕ :=
  if F.pairs.any (·.left = q) then n - 1 - j else if F.pairs.any (·.right = q) then j else 1

/-- Copy `j`'s first leg for the template leg `q`. -/
def at_ (j : ℕ) (q : Port) : Port := ⟨F.cp n q.node j, q.out, Diagram.posC (F.cnt n j) q⟩

/-- A port of `X` at its place once the pair legs grow. -/
def mv (q : Port) : Port :=
  let o := F.orgOf n q.node
  ⟨q.node, q.out, Diagram.posC (F.cnt n o.2) ⟨o.1, q.out, q.pos⟩⟩

/-- A side `ds` of copy `j` of the template node `x`: leg indices read, pair legs grown. -/
def side (x j : ℕ) (out : Bool) (ds : List ℕ) : List ℕ :=
  Diagram.growC (F.cnt n j) x out ((List.range ds.length).map fun i => ixAt n F.legs ⟨x, out, i⟩ j (ds.getD i 0))

/-- A node of `X` with its copy's indices and grown legs. -/
def node (y : DNode) : DNode :=
  let o := F.orgOf n y.id
  { y with ph := ixAt n F.ph o.1 o.2 y.ph, dim := ixAt n F.dim o.1 o.2 y.dim,
           outs := F.side n o.1 o.2 true y.outs, ins := F.side n o.1 o.2 false y.ins }

/-- The wires of the link `l`. -/
def linkWires (l : FLink) : List (Port × Port) :=
  if n = 0 then [(F.mv n l.e2, F.mv n l.e1)]
  else (F.mv n l.e2, F.at_ n 0 l.l2) ::
    ((List.range (n - 1)).map fun j => (F.at_ n j l.l1, F.at_ n (j + 1) l.l2)) ++
    [(F.at_ n (n - 1) l.l1, F.mv n l.e1)]

/-- The pairs `(j, k)`, `j < k < n`. -/
def jk : List (ℕ × ℕ) := (List.range n).flatMap fun j => (Diagram.later n j).map (j, ·)

/-- The id of the pair `(j, k)` of the `a`-th pair box. -/
def pid (a j k : ℕ) : ℕ := (F.X n).fresh + Nat.pair a (Nat.pair j k)

/-- The nodes of the `a`-th pair box `p`. -/
def pairNodes (a : ℕ) (p : DPair) : List DNode := (jk n).map fun q =>
  { p.node with id := F.pid n a q.1 q.2, ph := ixAt n F.ph p.node.id (q.2 - q.1) p.node.ph,
                dim := ixAt n F.dim p.node.id (q.2 - q.1) p.node.dim }

/-- The wires of the `a`-th pair box `p`. -/
def pairWires (a : ℕ) (p : DPair) : List (Port × Port) := (jk n).flatMap fun q =>
  let l := F.at_ n q.1 p.left
  let r := F.at_ n q.2 p.right
  [(⟨l.node, l.out, l.pos + (q.2 - q.1 - 1)⟩, ⟨F.pid n a q.1 q.2, false, 0⟩),
   (⟨F.pid n a q.1 q.2, true, 0⟩, ⟨r.node, r.out, r.pos + q.1⟩)]

/-- The diagram at `n` copies. -/
def famExpand : Diagram :=
  let X := F.X n
  { nodes := X.nodes.map (F.node n) ++ F.pairs.zipIdx.flatMap (fun q => F.pairNodes n q.2 q.1),
    wires := X.wires.map (fun w => (F.mv n w.1, F.mv n w.2)) ++ F.links.flatMap (F.linkWires n) ++
      F.pairs.zipIdx.flatMap (fun q => F.pairWires n q.2 q.1),
    outs := X.outs.map (F.mv n), ins := X.ins.map (F.mv n), boxes := X.boxes }

/-! ## Without links, pairs or indices -/

theorem posC_one (q : Port) : Diagram.posC (fun _ => 1) q = q.pos := by
  simp [Diagram.posC]

theorem map_getD (ds : List ℕ) : (List.range ds.length).map (fun i => ds.getD i 0) = ds := by
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp [List.getElem?_eq_getElem h₂]

theorem flatMap_one (l : List ℕ) (f : ℕ → ℕ) : l.flatMap (fun i => [f i]) = l.map f := by
  induction l <;> simp_all

theorem growC_one (x : ℕ) (out : Bool) (ds : List ℕ) : Diagram.growC (fun _ => 1) x out ds = ds := by
  simp only [Diagram.growC, List.replicate_one, flatMap_one, map_getD]

variable {F}

theorem cnt_plain (hp : F.pairs = []) (j : ℕ) : F.cnt n j = fun _ => 1 := by
  funext q; simp [cnt, hp]

theorem mv_plain (hp : F.pairs = []) (q : Port) : F.mv n q = q := by
  simp [mv, cnt_plain n hp, posC_one]

theorem node_plain (hp : F.pairs = []) (hph : F.ph = []) (hd : F.dim = []) (hl : F.legs = [])
    (y : DNode) : F.node n y = y := by
  have hs : ∀ x j out (ds : List ℕ), F.side n x j out ds = ds := fun x j out ds => by
    simp only [side, cnt_plain n hp, growC_one, hl, ixAt, List.lookup_nil, map_getD]
  simp [node, hs, ixAt, hph, hd]

/-- With no links, pairs or indices, `famExpand` is the expansion in key order. -/
theorem famExpand_plain (hk : F.links = []) (hp : F.pairs = []) (hph : F.ph = []) (hd : F.dim = [])
    (hl : F.legs = []) : F.famExpand n = F.base.expandK fun _ => n := by
  have hm : F.mv n = id := funext (mv_plain n hp)
  have hn : F.node n = id := funext (node_plain n hp hph hd hl)
  simp only [famExpand, hk, hp, List.zipIdx_nil, List.flatMap_nil, List.append_nil, hm, hn, id,
    List.map_id]
  simp only [Prod.mk.eta, List.map_id']
  rfl

end FamDiagram

end ArchytasZX
