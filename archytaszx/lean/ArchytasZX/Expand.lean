/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Diagram

/-!
# Expansion of bang boxes

`Diagram.expand ks` instantiates every bang box of a diagram, the box `b` at multiplicity
`ks b.mult`, one leaf box at a time (least id first). A port box grows its boundary legs to `k`
legs each; a node box copies its nodes and internal wires `k` times, a wire crossing into it fans
out into `k` legs of the outside node, and its boundary ports become one block, copy by copy, at
the position of the first of them.
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

/-- Where the port `old` lands once the leg `(out, idx)` of `node` has grown to `k` legs on the
node `new`. -/
def remapGrown (old : Port) (node new : ℕ) (out : Bool) (idx k : ℕ) : List Port :=
  if old.node ≠ node then [old]
  else if old.out ≠ out ∨ old.pos < idx then [⟨new, old.out, old.pos⟩]
  else if old.pos = idx then (List.range k).map fun i => ⟨new, out, idx + i⟩
  else [⟨new, old.out, old.pos + k - 1⟩]

/-- A node id above every node of the diagram. -/
def fresh (D : Diagram) : ℕ := (D.nodes.map (·.id)).foldr max 0 + 1

/-- The node of `ref` replaced by a fresh node whose leg at `ref` is `k` legs, and its id. -/
def regrow (D : Diagram) (ref : Port) (k : ℕ) : Diagram × ℕ :=
  match D.nodes.find? (·.id = ref.node) with
  | none => (D, ref.node)
  | some n =>
    let side := if ref.out then n.outs else n.ins
    let grown := side.take ref.pos ++ List.replicate k (side.getD ref.pos 0) ++ side.drop (ref.pos + 1)
    let new := D.fresh
    let n' : DNode := if ref.out then { n with id := new, outs := grown }
      else { n with id := new, ins := grown }
    let rm := fun p => remapGrown p ref.node new ref.out ref.pos k
    let first := fun p => (rm p).headD p
    ({ nodes := D.nodes.filter (·.id ≠ ref.node) ++ [n'],
       wires := D.wires.map fun w => (first w.1, first w.2),
       outs := D.outs.flatMap rm, ins := D.ins.flatMap rm,
       boxes := D.boxes.map fun b =>
         { b with nodes := b.nodes.map (fun x => if x = ref.node then new else x),
                  ports := b.ports.flatMap rm } }, new)

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

/-- The ports `pending` grown to `k` legs each, in order, with `fuel` steps. -/
def growAll (k : ℕ) : ℕ → Diagram → List Port → Diagram
  | 0, D, _ => D
  | _, D, [] => D
  | fuel + 1, D, ref :: rest =>
    let (D', new) := D.regrow ref k
    growAll k fuel D' (rest.flatMap fun p => remapGrown p ref.node new ref.out ref.pos k)

/-- Each crossing `(inside, outside)` rewired as one outside leg per copy in `maps`, with `fuel`
steps. -/
def fanOut (maps : List (List (ℕ × ℕ))) : ℕ → Diagram → List (Port × Port) → Diagram
  | 0, D, _ => D
  | _, D, [] => D
  | fuel + 1, D, (inside, outside) :: rest =>
    let D₁ := { D with wires := D.wires.filter fun w => !((w.1 = inside ∧ w.2 = outside) ∨
      (w.1 = outside ∧ w.2 = inside)) }
    let k := maps.length
    let (D₂, new) := D₁.regrow outside k
    let D₃ := { D₂ with wires := D₂.wires ++ (maps.zipIdx.map fun (mp, j) =>
      (⟨(mp.lookup inside.node).getD inside.node, inside.out, inside.pos⟩,
        ⟨new, outside.out, outside.pos + j⟩)) }
    fanOut maps fuel D₃ (rest.flatMap fun (i, o) =>
      (remapGrown o outside.node new outside.out outside.pos k).map fun o' => (i, o'))

/-- The copies `maps` of the nodes `scope` and of the wires inside it. -/
def addCopies (D : Diagram) (scope : List ℕ) (maps : List (List (ℕ × ℕ))) : Diagram :=
  let look := fun (mp : List (ℕ × ℕ)) (x : ℕ) => (mp.lookup x).getD x
  let internal := D.wires.filter fun w => w.1.node ∈ scope ∧ w.2.node ∈ scope
  { D with
    nodes := D.nodes ++ maps.flatMap fun mp => (D.nodes.filter (·.id ∈ scope)).map fun n =>
      { n with id := look mp n.id },
    wires := D.wires ++ maps.flatMap fun mp => internal.map fun w =>
      (⟨look mp w.1.node, w.1.out, w.1.pos⟩, ⟨look mp w.2.node, w.2.out, w.2.pos⟩) }

/-- The node box `b` instantiated at `k`. -/
def expandNode (D : Diagram) (b : DBox) (k : ℕ) : Diagram :=
  let D := if k = 0 then D.gather b else D
  let D := { D with boxes := D.boxes.filter (·.id ≠ b.id) }
  let scope := b.nodes
  let sorted := scope.mergeSort (· ≤ ·)
  let base := D.fresh
  let maps : List (List (ℕ × ℕ)) :=
    if k = 1 then [scope.map fun x => (x, x)]
    else (List.range k).map fun j => sorted.zipIdx.map fun (x, i) => (x, base + j * sorted.length + i)
  let crossings := ((D.wires.filter fun w => (w.1.node ∈ scope) ≠ (w.2.node ∈ scope)).map fun w =>
    if w.1.node ∈ scope then w else (w.2, w.1)).mergeSort fun p q => keyLe p.2 q.2
  let D := if k = 1 then D else D.addCopies scope maps
  let D := if k = 1 then D else fanOut maps (crossings.length + 1) D crossings
  let cr := D.scopeBnd scope
  let D := { D with ins := splice D.ins cr maps, outs := splice D.outs cr maps }
  let copies := maps.flatMap fun mp => mp.map (·.2)
  let D := { D with boxes := (D.boxes.map fun x =>
      if x.nodes.any (· ∈ scope) then
        let rest := x.nodes.filter (· ∉ scope)
        { x with nodes := rest ++ copies.filter (· ∉ rest) }
      else x).filter fun x => x.nodes ≠ [] ∨ x.ports ≠ [] }
  if k = 1 then D
  else { D with nodes := D.nodes.filter (·.id ∉ scope),
                wires := D.wires.filter fun w => w.1.node ∉ scope ∧ w.2.node ∉ scope }

/-- The port box `b` instantiated at `k`. -/
def expandPort (D : Diagram) (b : DBox) (k : ℕ) : Diagram :=
  let D := if k = 0 then D.gather b else D
  let D := { D with boxes := D.boxes.filter (·.id ≠ b.id) }
  let pending := b.ports.mergeSort keyLe
  growAll k (pending.length + 1) D pending

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

/-- The families `s₁ • D₁` and `s₂ • D₂` agree at every assignment `σ` of their multiplicity
symbols: the boxes of index `i` instantiated at `M σ i`, at the environment `E σ`. -/
def DEqualF (E : (ℕ → ℕ) → Env) (M : (ℕ → ℕ) → ℕ → ℕ) (s₁ : (ℕ → ℕ) → ℂ) (D₁ : Diagram)
    (s₂ : (ℕ → ℕ) → ℂ) (D₂ : Diagram) : Prop :=
  ∀ σ, DEqual (E σ) (s₁ σ) (D₁.expand (M σ)) (s₂ σ) (D₂.expand (M σ))

end ArchytasZX
