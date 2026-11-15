/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymF2

/-!
# Checkpoints for `runQ`

`qchain n IM R ks` runs `runQ IM` over the steps `R` with compiled code, states the value after
each block of steps (the blocks' sizes the list `ks`) as a literal definition `n_s<i>` of raw
numerals, proves each block by `decide` in its own declaration `n_h<i>`, and composes the
blocks into `n : runQ IM R st0 = some n_fin`.
`qscal` proves a scalar `a / b * √2^K` equal to its stated form.
-/

namespace ArchytasZX
namespace SymF2

section Chain

open Lean Meta Elab Command

/-- A list literal of element type `ty`. -/
def qListE (ty : Expr) (xs : List Expr) : Expr :=
  xs.foldr (fun x acc => mkApp3 (mkConst ``List.cons [Level.zero]) ty x acc)
    (mkApp (mkConst ``List.nil [Level.zero]) ty)

/-- An integer literal. -/
def qIntE : ℤ → Expr
  | .ofNat n => mkApp (mkConst ``Int.ofNat) (mkRawNatLit n)
  | .negSucc n => mkApp (mkConst ``Int.negSucc) (mkRawNatLit n)

/-- A state literal of raw numerals. -/
def qStE (s : St) : Expr :=
  let nat := mkConst ``Nat
  let pairT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) nat nat
  let pair := fun (p : ℕ × ℕ) =>
    mkApp4 (mkConst ``Prod.mk [Level.zero, Level.zero]) nat nat (mkRawNatLit p.1) (mkRawNatLit p.2)
  let qT := mkConst ``QTm
  let oT := mkApp (mkConst ``Option [Level.zero]) qT
  let t := match s.2 with
    | none => mkApp (mkConst ``Option.none [Level.zero]) qT
    | some t => mkApp2 (mkConst ``Option.some [Level.zero]) qT
        (mkApp4 (mkConst ``QTm.mk) (qIntE t.q) (qIntE t.k)
          (qListE nat (t.dels.map mkRawNatLit)) (qListE pairT (t.phs.map pair)))
  mkApp4 (mkConst ``Prod.mk [Level.zero, Level.zero]) nat oT (mkRawNatLit s.1) t

/-- The states after each block of `runQ IM` over `R` from `st0`, the blocks' sizes `ks`. -/
unsafe def qStatesImpl (IM : ℕ) (ks : List ℕ) (R : Expr) : MetaM (Array St) := do
  let stepT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) (mkConst ``Node)
    (mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat))
  let l ← evalExpr (List (Node × List ℕ)) (mkApp (mkConst ``List [Level.zero]) stepT) R
  let mut out := #[]
  let mut cur := st0
  let mut rest := l
  for k in ks do
    match runQ IM (rest.take k) cur with
    | some s => cur := s
    | none => throwError "qchain: a block of steps fails"
    out := out.push cur
    rest := rest.drop k
  unless rest.isEmpty do throwError "qchain: the blocks do not cover the steps"
  return out

@[implemented_by qStatesImpl]
def qStates (_IM : ℕ) (_ks : List ℕ) (_R : Expr) : MetaM (Array St) := pure #[]

/-- A number's value. -/
unsafe def qNatImpl (e : Expr) : MetaM ℕ := evalExpr ℕ (mkConst ``Nat) e

@[implemented_by qNatImpl]
def qNat (_e : Expr) : MetaM ℕ := pure 0

/-- A list of numbers' value. -/
unsafe def qNatsImpl (e : Expr) : MetaM (List ℕ) :=
  evalExpr (List ℕ) (mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat)) e

@[implemented_by qNatsImpl]
def qNats (_e : Expr) : MetaM (List ℕ) := pure []

/-- `type` added as a theorem with proof `value`, checked by the kernel at once. -/
def qAddThm (n : Name) (type value : Expr) : MetaM Unit :=
  withOptions (fun o => o.setBool `Elab.async false) do
    addDecl (.thmDecl { name := n, levelParams := [], type, value })

/-- `n : runQ IM R st0 = some n_fin`, through a literal state `n_s<i>` after each block of
steps, the blocks' sizes the list `ks`, each block a theorem `n_h<i>` proved by `decide`. -/
elab "qchain " n:ident im:ident r:ident ks:ident : command =>
  liftTermElabM do
    let imE := mkConst (← resolveGlobalConstNoOverload im)
    let rE := mkConst (← resolveGlobalConstNoOverload r)
    let base := (← getCurrNamespace) ++ n.getId
    let name := fun (s : String) => base.appendAfter s
    let IM ← qNat imE
    let sizes ← qNats (mkConst (← resolveGlobalConstNoOverload ks))
    let vals ← qStates IM sizes rE
    let stT := mkConst ``St
    let someE := fun e => mkApp2 (mkConst ``Option.some [Level.zero]) stT e
    let runE := fun R s => mkApp3 (mkConst ``runQ) imE R s
    let stepT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) (mkConst ``Node)
      (mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat))
    let dropE := fun (i : ℕ) => mkApp3 (mkConst ``List.drop [Level.zero]) stepT (mkNatLit i) rE
    let takeE := fun (i : ℕ) (R : Expr) => mkApp3 (mkConst ``List.take [Level.zero]) stepT (mkNatLit i) R
    let mut prev := mkConst ``st0
    let mut pf ← mkEqRefl (runE rE prev)
    let mut i := 0
    for j in [0:vals.size] do
      let k := sizes.getD j 0
      let sn := name s!"_s{j + 1}"
      addDecl (.defnDecl (mkDefinitionValEx sn [] stT (qStE vals[j]!) .abbrev .safe [sn]))
      let cur := mkConst sn
      let hT ← mkEq (runE (takeE k (dropE i)) prev) (someE cur)
      let hn := name s!"_h{j}"
      qAddThm hn hT (← mkDecideProof hT)
      let seg := mkAppN (mkConst ``runQ_seg) #[imE, rE, mkNatLit i, mkNatLit k, mkNatLit (i + k),
        prev, cur, ← mkEqRefl (mkNatLit (i + k)), mkConst hn]
      pf ← mkEqTrans pf seg
      prev := cur
      i := i + k
    let lenT ← mkAppM ``LE.le #[← mkAppM ``List.length #[rE], mkNatLit i]
    let endP := mkAppN (mkConst ``runQ_end) #[imE, rE, mkNatLit i, prev, ← mkDecideProof lenT]
    pf ← mkEqTrans pf endP
    let fin := name "_fin"
    addDecl (.defnDecl (mkDefinitionValEx fin [] stT prev .abbrev .safe [fin]))
    qAddThm base (← mkEq (runE rE (mkConst ``st0)) (someE (mkConst fin))) pf

end Chain

/-- A scalar `a / b * √2^K` equals its stated form. -/
macro "qscal" : tactic =>
  `(tactic| (norm_num [zpow_one, zpow_zero, rt_sq, rt_sq'] <;> (try field_simp) <;>
    (try norm_num [rt_sq, rt_sq']) <;> (try ring_nf)))

end SymF2
end ArchytasZX
