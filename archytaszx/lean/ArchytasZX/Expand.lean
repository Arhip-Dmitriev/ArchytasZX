/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Diagram

/-!
# Expansion of bang boxes

`Diagram.expand ks` instantiates every bang box of a diagram, the box `b` at multiplicity
`ks b.mult`, one leaf box at a time (least id first). A port box grows its boundary legs to `k`
legs each, in place; a node box copies its nodes and internal wires `k` times, a wire crossing
into it fans out into `k` legs of the outside node, and its boundary ports become one block, copy
by copy, at the position of the first of them. Every port maps to the list of ports it becomes,
and every wire to the zip of its two ends' lists.
-/

namespace ArchytasZX

namespace Diagram

/-- A port's place in the engine's order: node, side (inputs first), position. -/
def portKey (p : Port) : ℕ × ℕ × ℕ := (p.node, if p.out then 1 else 0, p.pos)

/-- Lexicographic order on port keys. -/
def keyLe (p q : Port) : Bool :=
  let a := portKey p
  let b := portKey q
  a.1 < b.1 || (a.1 = b.1 && (a.2.1 < b.2.1 || (a.2.1 = b.2.1 && a.2.2 ≤ b.2.2)))

/-- A node id above every node of the diagram. -/
def fresh (D : Diagram) : ℕ := (D.nodes.map (·.id)).foldr max 0 + 1

/-- `refs` with its ports in `old` replaced by one block at the first of them: every copy's image
of those ports, copy by copy. -/
def splice (refs old : List Port) (maps : List (List (ℕ × ℕ))) : List Port :=
  let here := refs.filter (· ∈ old)
  if here = [] then refs
  else
    let at_ := refs.findIdx (· ∈ old)
    let block := maps.flatMap fun mp => here.map fun r => ⟨(mp.lookup r.node).getD r.node, r.out, r.pos⟩
    (refs.take at_).filter (· ∉ old) ++ block ++ (refs.drop at_).filter (· ∉ old)

/-- The boundary ports on the nodes `scope`, in port order. -/
def scopeBnd (D : Diagram) (scope : List ℕ) : List Port :=
  ((D.ins ++ D.outs).filter (·.node ∈ scope)).mergeSort keyLe

/-- Each node-box ancestor of `b`'s boundary ports gathered into one block at its first. -/
def gather (D : Diagram) (b : DBox) : Diagram :=
  go D b.parent D.boxes.length
where
  go (D : Diagram) (anc : Option ℕ) : ℕ → Diagram
  | 0 => D
  | fuel + 1 =>
    match anc.bind fun a => D.boxes.find? (·.id = a) with
    | none => D
    | some outer =>
      if outer.nodes = [] then go D outer.parent fuel
      else
        let ident := outer.nodes.map fun x => (x, x)
        let cr := D.scopeBnd outer.nodes
        go { D with ins := splice D.ins cr [ident], outs := splice D.outs cr [ident] }
          outer.parent fuel

/-- Where the leg at `p` lands once every leg `g` holds grows to `k` legs, in place: the legs
before it on its side that grow count `k` each. -/
def shiftPos (g : Port → Bool) (k : ℕ) (p : Port) : ℕ :=
  let c := ((List.range p.pos).filter fun i => g ⟨p.node, p.out, i⟩).length
  p.pos - c + k * c

/-- The side `ds` of the node `x` with every leg `g` holds grown to `k` legs. -/
def growSide (g : Port → Bool) (k x : ℕ) (out : Bool) (ds : List ℕ) : List ℕ :=
  ds.zipIdx.flatMap fun (d, i) => if g ⟨x, out, i⟩ then List.replicate k d else [d]

/-- The legs the leg at `p` becomes once every leg `g` holds grows to `k` legs, in place. -/
def growImg (g : Port → Bool) (k : ℕ) (p : Port) : List Port :=
  if g p then (List.range k).map fun j => ⟨p.node, p.out, shiftPos g k p + j⟩
  else [⟨p.node, p.out, shiftPos g k p⟩]

/-- Copy `j` of the node `x` of the node box `b`: fresh ids, copy by copy, scope in id order. -/
def cpId (D : Diagram) (b : DBox) (j x : ℕ) : ℕ :=
  D.fresh + j * b.nodes.length + (b.nodes.mergeSort (· ≤ ·)).idxOf x

/-- A port on a node of the node box `b`. -/
def inS (b : DBox) (p : Port) : Bool := decide (p.node ∈ b.nodes)

/-- A port outside the node box `b` wired to a port inside it. -/
def crossP (D : Diagram) (b : DBox) (p : Port) : Bool :=
  !inS b p && D.wires.any fun w => (w.1 = p && inS b w.2) || (w.2 = p && inS b w.1)

/-- The ports `p` becomes once the node box `b` is copied `k` times. -/
def nimg (D : Diagram) (b : DBox) (k : ℕ) (p : Port) : List Port :=
  if inS b p then (List.range k).map fun j => ⟨D.cpId b j p.node, p.out, p.pos⟩
  else growImg (D.crossP b) k p

/-- The boundary `refs` with its ports in `b` one block at the first of them, copy by copy. -/
def spliceN (D : Diagram) (b : DBox) (k : ℕ) (refs : List Port) : List Port :=
  let here := refs.filter (inS b)
  if here = [] then refs.flatMap (D.nimg b k)
  else
    let at_ := refs.findIdx (inS b)
    (refs.take at_).flatMap (D.nimg b k) ++
      (List.range k).flatMap (fun j => here.map fun p => (⟨D.cpId b j p.node, p.out, p.pos⟩ : Port)) ++
      ((refs.drop at_).filter (!inS b ·)).flatMap (D.nimg b k)

/-- The nodes of the node box `b` and the wires inside it copied `k` times, a wire crossing in
fanned out over the outside leg grown to `k` legs. -/
def copyNodes (D : Diagram) (b : DBox) (k : ℕ) : Diagram :=
  let copies := (List.range k).flatMap fun j => (b.nodes.mergeSort (· ≤ ·)).map (D.cpId b j)
  { nodes := D.nodes.flatMap fun n =>
      if n.id ∈ b.nodes then (List.range k).map fun j => { n with id := D.cpId b j n.id }
      else [{ n with outs := growSide (D.crossP b) k n.id true n.outs,
                     ins := growSide (D.crossP b) k n.id false n.ins }],
    wires := D.wires.flatMap fun w => (D.nimg b k w.1).zip (D.nimg b k w.2),
    outs := D.spliceN b k D.outs, ins := D.spliceN b k D.ins,
    boxes := ((D.boxes.filter (·.id ≠ b.id)).map fun x =>
      { x with nodes := x.nodes.filter (· ∉ b.nodes) ++
                 (if x.nodes.any (· ∈ b.nodes) then copies else []),
               ports := x.ports.flatMap (D.nimg b k) }).filter fun x => x.nodes ≠ [] ∨ x.ports ≠ [] }

/-- The node box `b` instantiated at `k`. -/
def expandNode (D : Diagram) (b : DBox) (k : ℕ) : Diagram :=
  (if k = 0 then D.gather b else D).copyNodes b k

/-- Each port of the port box `b` grown to `k` legs in place. -/
def growPorts (D : Diagram) (b : DBox) (k : ℕ) : Diagram :=
  let g := fun (p : Port) => decide (p ∈ b.ports)
  let img := growImg g k
  { nodes := D.nodes.map fun n =>
      { n with outs := growSide g k n.id true n.outs, ins := growSide g k n.id false n.ins },
    wires := D.wires.flatMap fun w => (img w.1).zip (img w.2),
    outs := D.outs.flatMap img, ins := D.ins.flatMap img,
    boxes := ((D.boxes.filter (·.id ≠ b.id)).map fun x => { x with ports := x.ports.flatMap img }).filter
      fun x => x.nodes ≠ [] ∨ x.ports ≠ [] }

/-- The port box `b` instantiated at `k`: each of its ports grown to `k` legs in place. -/
def expandPort (D : Diagram) (b : DBox) (k : ℕ) : Diagram :=
  (if k = 0 then D.gather b else D).growPorts b k

/-- Every box instantiated, the box `b` at `ks b.mult`, the least-id leaf box first, with `fuel`
steps. -/
def expandFuel (ks : ℕ → ℕ) : ℕ → Diagram → Diagram
  | 0, D => D
  | fuel + 1, D =>
    let leaves := D.boxes.filter fun b => !D.boxes.any (·.parent = some b.id)
    match leaves.foldr (fun b acc => match acc with
        | none => some b
        | some c => if b.id < c.id then some b else some c) none with
    | none => D
    | some b =>
      let k := ks b.mult
      expandFuel ks fuel (if b.ports = [] then D.expandNode b k else D.expandPort b k)

/-- The box-free diagram with every box `b` instantiated at `ks b.mult`. -/
def expand (ks : ℕ → ℕ) (D : Diagram) : Diagram := expandFuel ks (D.boxes.length + 1) D

end Diagram

end ArchytasZX
