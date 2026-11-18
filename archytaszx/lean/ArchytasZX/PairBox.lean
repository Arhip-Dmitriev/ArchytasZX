/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainLabel
import ArchytasZX.Relabel

/-!
# Pair boxes

A pair box over a node box holds one node, instantiated once per pair of copies `j < k` of its
parent, its input wired to copy `j`'s port `left` and its output to copy `k`'s port `right`
(`DPair`); a `PDiagram` is a `Diagram` with its pair boxes, none by default.
`Diagram.expandPair D p K` is the expansion of a diagram whose only box is a node box holding
every node, with the pair box `p` over it, at `K` copies, copy by copy: copy `j` of each node
(id `pcp j x`) and wire, `left` grown in place to one leg per later copy and `right` to one per
earlier copy (`pcnt`, `growC`, `posC`), the boundary copy by copy, and the node of the pair
`(j, k)` (id `pfid j k`). `PairOK` (decided by the kernel) states the form the bridge reads:
every node in the box, no inputs, the pair ports dangling and last on their sides, every other
leg on the boundary or a wire, the slots and wire labels of `D.label m` in order.

`tmpl D p` is the template's network without the pair ports' legs, slots `0 .. m - 1` and wire
`i` at `m + i`. `plab` labels the expansion: copy `j` reads the template's labels through
`shJ` (slot `x` at `j m + x`, wire label `m + i` at `K m + j L + i`), the pair `(j, k)`'s left
leg `gL j k` and right leg `gR j k`. `pairNet` is the expansion's network in that labelling:
copy `j` of each template node with copy `j`'s pair legs (`exL`) appended, then the pair nodes.
-/

namespace ArchytasZX

/-- A pair box: id, parent node box, the parent's ports `left` and `right`, and the node one copy
holds, its input at copy `j`'s `left` and its output at copy `k`'s `right`. -/
structure DPair where
  id : ℕ
  parent : ℕ
  left : Port
  right : Port
  node : DNode
  deriving DecidableEq, Repr

/-- A diagram with its pair boxes. -/
structure PDiagram extends Diagram where
  pairs : List DPair := []
  deriving DecidableEq, Repr

namespace Diagram

/-- Copy `j` of the node `x`. -/
def pcp (j x : ℕ) : ℕ := 2 * Nat.pair j x

/-- The node of the pair `(j, k)`. -/
def pfid (j k : ℕ) : ℕ := 2 * Nat.pair j k + 1

/-- The number of legs the template leg `q` becomes in copy `j` of `K`. -/
def pcnt (p : DPair) (K j : ℕ) (q : Port) : ℕ :=
  if q = p.left then K - 1 - j else if q = p.right then j else 1

/-- The side `ds` of the node `x` with each leg `q` grown to `c q` legs in place. -/
def growC (c : Port → ℕ) (x : ℕ) (out : Bool) (ds : List ℕ) : List ℕ :=
  (List.range ds.length).flatMap fun i => List.replicate (c ⟨x, out, i⟩) (ds.getD i 0)

/-- The position of the first leg the leg `q` becomes. -/
def posC (c : Port → ℕ) (q : Port) : ℕ :=
  ((List.range q.pos).map fun i => c ⟨q.node, q.out, i⟩).sum

/-- Copy `j` of the template port `q`. -/
def pimg (p : DPair) (K j : ℕ) (q : Port) : Port := ⟨pcp j q.node, q.out, posC (pcnt p K j) q⟩

/-- Copy `j` of the template node `n`, its pair legs grown. -/
def pcpNode (p : DPair) (K j : ℕ) (n : DNode) : DNode :=
  { n with id := pcp j n.id, outs := growC (pcnt p K j) n.id true n.outs,
           ins := growC (pcnt p K j) n.id false n.ins }

/-- The wire from copy `j`'s left port to the pair `(j, k)`. -/
def plw (p : DPair) (K j k : ℕ) : Port × Port :=
  (⟨pcp j p.left.node, p.left.out, posC (pcnt p K j) p.left + (k - j - 1)⟩, ⟨pfid j k, false, 0⟩)

/-- The wire from the pair `(i, j)` to copy `j`'s right port. -/
def prw (p : DPair) (K i j : ℕ) : Port × Port :=
  (⟨pfid i j, true, 0⟩, ⟨pcp j p.right.node, p.right.out, posC (pcnt p K j) p.right + i⟩)

/-- The later copies `j + 1 .. K - 1`. -/
def later (K j : ℕ) : List ℕ := List.range' (j + 1) (K - 1 - j)

/-- The pair box `p` over the node box holding every node of `D`, at `K` copies. -/
def expandPair (D : Diagram) (p : DPair) (K : ℕ) : Diagram where
  nodes := (List.range K).flatMap (fun j => D.nodes.map (pcpNode p K j)) ++
    (List.range K).flatMap (fun j => (later K j).map fun k => { p.node with id := pfid j k })
  wires := (List.range K).flatMap fun j =>
    D.wires.map (fun w => (pimg p K j w.1, pimg p K j w.2)) ++ (later K j).map (plw p K j) ++
      (List.range j).map (fun i => prw p K i j)
  outs := (List.range K).flatMap fun j => D.outs.map (pimg p K j)
  ins := (List.range K).flatMap fun j => D.ins.map (pimg p K j)

end Diagram

/-- The expansion of a diagram with one pair box at `K` copies of its node box. -/
def PDiagram.expandP (P : PDiagram) (K : ℕ) : Diagram :=
  match P.pairs with
  | [p] => P.toDiagram.expandPair p K
  | _ => P.toDiagram

namespace Diagram

variable (D : Diagram) (p : DPair)

/-- A template port off the pair ports. -/
def keepP (q : Port) : Bool := q ≠ p.left && q ≠ p.right

/-- The number of legs on the side `out` of the node `x`. -/
def sideLen (x : ℕ) (out : Bool) : ℕ :=
  match D.nodes.find? (·.id = x) with
  | some n => if out then n.outs.length else n.ins.length
  | none => 0

/-- The pair port `q` is the last leg of its side, or `p.left` just before `p.right`. -/
def lastP (q : Port) : Bool :=
  decide (q.pos + 1 = D.sideLen q.node q.out) ||
    (decide (q = p.left) && decide (p.right = ⟨q.node, q.out, q.pos + 1⟩) &&
      decide (q.pos + 2 = D.sideLen q.node q.out))

/-- The template node `n` as a network node without its pair legs, labelled by `D.label m`. -/
def tNode (m : ℕ) (n : DNode) : Node :=
  ⟨n.kind, ((List.range n.outs.length).filter fun k => keepP p ⟨n.id, true, k⟩).map
      fun k => D.label m ⟨n.id, true, k⟩,
    ((List.range n.ins.length).filter fun k => keepP p ⟨n.id, false, k⟩).map
      fun k => D.label m ⟨n.id, false, k⟩, n.dim, n.ph⟩

/-- The template's network: the nodes without their pair legs, wire `i` at `m + i`, the slots
`0 .. m - 1`. -/
def tmpl : Net :=
  ⟨D.nodes.map (D.tNode p D.outs.length), (List.range D.wires.length).map (D.outs.length + ·),
    List.range D.outs.length⟩

/-- The form the pair bridge reads: one node box `b` holding every node (ids distinct), no
inputs, the pair ports on nodes of the box, dangling and last on their sides (`left` before
`right` on a shared side), every other leg on the boundary or a wire, the slots and the wire
labels of `D.label m` in order, the pair node an `F` of one input and one output, and every
dimension index `0`. -/
def PairOK (b : DBox) : Bool :=
  let m := D.outs.length
  decide (D.boxes = [b]) && decide (b.parent = none) && decide (b.ports = []) &&
  decide (p.parent = b.id) && D.nodes.all (fun n => decide (n.id ∈ b.nodes)) &&
  decide ((D.nodes.map (·.id)).Nodup) && decide (D.ins = []) &&
  decide (p.left ≠ p.right) &&
  decide (p.left.pos < D.sideLen p.left.node p.left.out) &&
  decide (p.right.pos < D.sideLen p.right.node p.right.out) &&
  D.lastP p p.left && D.lastP p p.right &&
  !(D.bnd.contains p.left) && !(D.bnd.contains p.right) &&
  D.wires.all (fun w => decide (w.1 ≠ p.left) && decide (w.1 ≠ p.right) &&
    decide (w.2 ≠ p.left) && decide (w.2 ≠ p.right) &&
    decide (w.1.pos < D.sideLen w.1.node w.1.out) && decide (w.2.pos < D.sideLen w.2.node w.2.out)) &&
  D.outs.all (fun q => decide (q.pos < D.sideLen q.node q.out)) &&
  D.nodes.all (fun n =>
    (List.range n.outs.length).all (fun k => !keepP p ⟨n.id, true, k⟩ || D.okPort ⟨n.id, true, k⟩) &&
    (List.range n.ins.length).all (fun k => !keepP p ⟨n.id, false, k⟩ || D.okPort ⟨n.id, false, k⟩)) &&
  (List.range m).all (fun i => decide (D.label m (D.outs.getD i Port.dflt) = i)) &&
  (List.range D.wires.length).all (fun i =>
    decide (D.label m (D.wires.getD i (Port.dflt, Port.dflt)).1 = m + i) &&
    decide (D.label m (D.wires.getD i (Port.dflt, Port.dflt)).2 = m + i)) &&
  decide (p.node.kind = .F) && decide (p.node.outs = [0]) && decide (p.node.ins = [0]) &&
  decide (p.node.dim = 0) &&
  D.nodes.all (fun n => decide (n.dim = 0))

/-- Slot `x` of copy `j` at `j m + x`, a template wire label `m + i` at `K m + j L + i`. -/
def shJ (K m L j x : ℕ) : ℕ := if x < m then j * m + x else K * m + j * L + (x - m)

/-- The first label above the copies' wire labels. -/
def pG (K m L : ℕ) : ℕ := K * m + K * L

/-- The left leg of the pair `(j, k)`. -/
def gL (K m L j k : ℕ) : ℕ := pG K m L + 2 * Nat.pair j k

/-- The right leg of the pair `(j, k)`. -/
def gR (K m L j k : ℕ) : ℕ := pG K m L + 2 * Nat.pair j k + 1

/-- The label of copy `j`'s leg at `pos` on the side `out` of the template node `x`. -/
def clab (K j x : ℕ) (out : Bool) (pos : ℕ) : ℕ :=
  let m := D.outs.length
  let L := D.wires.length
  if x = p.left.node ∧ out = p.left.out ∧ p.left.pos ≤ pos then
    (if pos - p.left.pos < K - 1 - j then gL K m L j (j + 1 + (pos - p.left.pos))
     else gR K m L (pos - p.left.pos - (K - 1 - j)) j)
  else if x = p.right.node ∧ out = p.right.out ∧ p.right.pos ≤ pos then
    gR K m L (pos - p.right.pos) j
  else shJ K m L j (D.label m ⟨x, out, pos⟩)

/-- The expansion's labelling: a copy's leg through `clab`, a pair node's input at `gL` and
output at `gR`. -/
def plab (K : ℕ) (q : Port) : ℕ :=
  if q.node % 2 = 0 then D.clab p K (q.node / 2).unpair.1 (q.node / 2).unpair.2 q.out q.pos
  else if q.out then gR K D.outs.length D.wires.length (q.node / 2).unpair.1 (q.node / 2).unpair.2
  else gL K D.outs.length D.wires.length (q.node / 2).unpair.1 (q.node / 2).unpair.2

/-- Copy `j`'s pair legs on the side `out` of the template node `x`: one left leg per later
copy, then one right leg per earlier copy. -/
def exL (K j x : ℕ) (out : Bool) : List ℕ :=
  (if x = p.left.node ∧ out = p.left.out then
    (later K j).map (gL K D.outs.length D.wires.length j) else []) ++
  (if x = p.right.node ∧ out = p.right.out then
    (List.range j).map (fun i => gR K D.outs.length D.wires.length i j) else [])

/-- Copy `j` of the template node `n` in the expansion's labelling. -/
def pNode (K j : ℕ) (n : DNode) : Node :=
  let t := D.tNode p D.outs.length n
  let sh := shJ K D.outs.length D.wires.length j
  ⟨t.kind, t.outs.map sh ++ D.exL p K j n.id true, t.ins.map sh ++ D.exL p K j n.id false,
    t.dim, t.ph⟩

/-- The node of the pair `(j, k)` in the expansion's labelling. -/
def pFNode (K j k : ℕ) : Node :=
  ⟨p.node.kind, [gR K D.outs.length D.wires.length j k], [gL K D.outs.length D.wires.length j k],
    p.node.dim, p.node.ph⟩

/-- Copy `j`'s summed labels: its wire labels, its left legs, its right legs. -/
def pInt (K j : ℕ) : List ℕ :=
  (List.range D.wires.length).map (fun i => K * D.outs.length + j * D.wires.length + i) ++
    (later K j).map (gL K D.outs.length D.wires.length j) ++
    (List.range j).map (fun i => gR K D.outs.length D.wires.length i j)

/-- The expansion's network at `K` copies in the labelling `plab`. -/
def pairNet (K : ℕ) : Net :=
  ⟨(List.range K).flatMap (fun j => D.nodes.map (D.pNode p K j)) ++
      (List.range K).flatMap (fun j => (later K j).map (D.pFNode p K j)),
    (List.range K).flatMap (D.pInt K), List.range (K * D.outs.length)⟩

end Diagram

end ArchytasZX
