/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Expand

/-!
# Boxed diagrams as bundled networks

`Diagram.toNetB bund off` labels a boxed diagram as one network for the whole family: a node in
node boxes is a `Bun` tower along their multiplicities, outermost first; a leg whose wire reaches
deeper boxes, or whose boundary port is in port boxes, is a bundled tail (`ZB`, `XB`, `WB`); a
label's dimension index bundles its plain index along its chain through the table `bund`.
-/

namespace ArchytasZX

namespace Diagram

variable (D : Diagram)

/-- The boxes satisfying `p`, from the one whose parent is `start` down through its children,
with `fuel` steps. -/
def walk (p : DBox → Bool) : ℕ → Option ℕ → List DBox
  | 0, _ => []
  | fuel + 1, start =>
    match D.boxes.find? fun b => p b && b.parent == start with
    | none => []
    | some b => b :: walk p fuel (some b.id)

/-- The node boxes holding the node `x`, outermost first. -/
def nodeBoxes (x : ℕ) : List DBox :=
  D.walk (fun b => decide (x ∈ b.nodes)) D.boxes.length none

/-- The port boxes holding the port `q`, outermost first. -/
def portBoxes (q : Port) : List DBox :=
  D.walk (fun b => decide (q ∈ b.ports)) D.boxes.length ((D.nodeBoxes q.node).getLast?.map (·.id))

/-- The multiplicity indices of the node boxes holding `x`, outermost first. -/
def nodeChain (x : ℕ) : List ℕ := (D.nodeBoxes x).map (·.mult)

/-- The other end of the wire at `q`. -/
def partner (q : Port) : Option Port :=
  (D.wires.find? fun w => w.1 = q ∨ w.2 = q).map fun w => if w.1 = q then w.2 else w.1

/-- The multiplicities the leg at `q` bundles beyond its node's own copies. -/
def tail (q : Port) : List ℕ :=
  match D.partner q with
  | some r =>
    let a := D.nodeChain q.node
    let b := D.nodeChain r.node
    if a.length < b.length then b.drop a.length else []
  | none => (D.portBoxes q).map (·.mult)

/-- A wire's chain: the node chain of its deeper end. -/
def wireChain (w : Port × Port) : List ℕ :=
  let a := D.nodeChain w.1.node
  let b := D.nodeChain w.2.node
  if a.length < b.length then b else a

/-- A boundary port's chain: its node's chain, then its port boxes'. -/
def bndChain (q : Port) : List ℕ := D.nodeChain q.node ++ (D.portBoxes q).map (·.mult)

end Diagram

/-- The dimension index bundling `base` along `m` in the table `bund`. -/
def bundIdx (bund : List (ℕ × ℕ × ℕ)) (base m : ℕ) : ℕ :=
  ((bund.find? fun p => p.2 = (base, m)).map (·.1)).getD 0

/-- The dimension index bundling `base` along the chain `c`, outermost first. -/
def dimFor (bund : List (ℕ × ℕ × ℕ)) (base : ℕ) (c : List ℕ) : ℕ :=
  c.foldr (fun m i => bundIdx bund i m) base

/-- Lexicographic order on chains, a proper prefix first. -/
def chainLt : List ℕ → List ℕ → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => a < b || (a = b && chainLt as bs)

/-- Lexicographic order on (chain, side, position). -/
def tailLe (p q : List ℕ × ℕ × ℕ) : Bool :=
  chainLt p.1 q.1 || (p.1 = q.1 && (p.2.1 < q.2.1 || (p.2.1 = q.2.1 && p.2.2 ≤ q.2.2)))

/-- A node's kind and legs with the legs of nonempty tail moved last: the tails of a Z sorted
together, an X's per side, a W's outputs. -/
def bundleNode (n : Node) (tO tI : List (ℕ × List ℕ)) : Node :=
  let keepO := ((List.range n.outs.length).filter fun j => (tO.lookup j).isNone).map
    fun j => n.outs.getD j 0
  let keepI := ((List.range n.ins.length).filter fun j => (tI.lookup j).isNone).map
    fun j => n.ins.getD j 0
  let so := (tO.map fun p => (p.2, 0, p.1)).insertionSort (fun a b => tailLe a b)
  let si := (tI.map fun p => (p.2, 1, p.1)).insertionSort (fun a b => tailLe a b)
  if tO = [] ∧ tI = [] then n
  else match n.kind with
  | .Z =>
    let both := (so ++ si).insertionSort (fun a b => tailLe a b)
    { n with kind := .ZB (both.map (·.1)), outs := keepO,
             ins := keepI ++ both.map fun t => if t.2.1 = 0 then n.outs.getD t.2.2 0
               else n.ins.getD t.2.2 0 }
  | .X =>
    { n with kind := .XB (so.map (·.1)) (si.map (·.1)),
             outs := keepO ++ so.map (fun t => n.outs.getD t.2.2 0),
             ins := keepI ++ si.map (fun t => n.ins.getD t.2.2 0) }
  | .W =>
    { n with kind := .WB (so.map (·.1)), outs := keepO ++ so.map (fun t => n.outs.getD t.2.2 0) }
  | _ => n

namespace Diagram

variable (D : Diagram)

/-- The node `n` labelled at `off`, its tails bundled, as a `Bun` tower along its node boxes. -/
def bnode (off : ℕ) (n : DNode) : Node :=
  let tl := fun (out : Bool) (len : ℕ) => ((List.range len).map fun j =>
    (j, D.tail ⟨n.id, out, j⟩)).filter fun p => p.2 ≠ []
  let b := bundleNode (n.node (D.label off)) (tl true n.outs.length) (tl false n.ins.length)
  { b with kind := (D.nodeChain n.id).foldr Kind.Bun b.kind }

/-- The family's network: wire `j` labelled `off + j`. -/
def toNetB (off : ℕ) : Net :=
  ⟨D.nodes.map (D.bnode off), (List.range D.wires.length).map (off + ·),
    List.range D.bnd.length⟩

/-- Each label's dimension index through the table `bund`: a boundary slot's chain, a wire's
chain over its plain index. -/
def labsB (bund : List (ℕ × ℕ × ℕ)) (off : ℕ) : List (ℕ × ℕ) :=
  (D.bnd.zipIdx.map fun (q, i) => (i, dimFor bund (D.portDim q) (D.bndChain q))) ++
  (D.wires.zipIdx.map fun (w, j) => (off + j, dimFor bund (D.portDim w.1) (D.wireChain w)))

end Diagram

end ArchytasZX
