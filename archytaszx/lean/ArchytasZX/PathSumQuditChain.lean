/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumChain
import ArchytasZX.PathSumQudit

/-!
# Qudit step lists checked through checkpoints

`run_cut` splits a qudit step list after `k` steps at a stated intermediate value. The command
`qrun_chain` computes the values after each block of at most `k` steps with compiled code,
states each as a literal definition, proves each block by `decide` in its own declaration
(`PSem.chain`), and composes the blocks into `<n>_ok : QSem.runOK p sts s = true` and
`<n>_eq : QSem.run p sts s = <n>`.
-/

namespace ArchytasZX
namespace QSem

theorem runOK_append (p : ℕ) : ∀ (a b : List Step) (s : Data),
    runOK p (a ++ b) s = (runOK p a s && runOK p b (run p a s))
  | [], b, s => by simp [runOK, run]
  | st :: a, b, s => by
    simp only [List.cons_append, runOK, run, runOK_append p a b, Bool.and_assoc]

theorem run_append (p : ℕ) : ∀ (a b : List Step) (s : Data),
    run p (a ++ b) s = run p b (run p a s)
  | [], _, _ => rfl
  | _ :: a, b, _ => run_append p a b _

theorem run_cut (p : ℕ) (sts : List Step) (k : ℕ) (s s' : Data)
    (he : (runOK p (sts.take k) s && decide (run p (sts.take k) s = s')) = true) :
    runOK p sts s = runOK p (sts.drop k) s' ∧ run p sts s = run p (sts.drop k) s' := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at he
  obtain ⟨h, e⟩ := he
  constructor
  · calc runOK p sts s = runOK p (sts.take k ++ sts.drop k) s := by rw [List.take_append_drop]
      _ = runOK p (sts.drop k) s' := by rw [runOK_append, h, e, Bool.true_and]
  · calc run p sts s = run p (sts.take k ++ sts.drop k) s := by rw [List.take_append_drop]
      _ = run p (sts.drop k) s' := by rw [run_append, e]

open Lean Meta Elab Command

/-- `P` as a literal of raw numerals. -/
def polyE (P : Poly) : Expr :=
  let nat := mkConst ``Nat
  let pairT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) nat nat
  let monoT := mkApp (mkConst ``List [Level.zero]) pairT
  let pair := fun (a b : ℕ) =>
    mkApp4 (mkConst ``Prod.mk [Level.zero, Level.zero]) nat nat (mkRawNatLit a) (mkRawNatLit b)
  let mono := fun (m : Mono) => PSem.listE pairT (m.map fun t => pair t.1 t.2)
  let termT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) monoT nat
  PSem.listE termT (P.map fun t =>
    mkApp4 (mkConst ``Prod.mk [Level.zero, Level.zero]) monoT nat (mono t.1) (mkRawNatLit t.2))

/-- `d` as a literal of raw numerals. -/
def qdataE (d : Data) : Expr :=
  let polyT := mkApp (mkConst ``List [Level.zero])
    (mkApp2 (mkConst ``Prod [Level.zero, Level.zero])
      (mkApp (mkConst ``List [Level.zero])
        (mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) (mkConst ``Nat) (mkConst ``Nat)))
      (mkConst ``Nat))
  mkApp4 (mkConst ``Data.mk) (PSem.natsE d.path) (polyE d.phase)
    (PSem.listE polyT (d.cons.map polyE)) (PSem.listE polyT (d.rows.map polyE))

/-- The value a `QSem.Data.mk` term of numerals and lists denotes. -/
def decQData (e : Expr) : MetaM Data := do
  unless e.isAppOfArity ``Data.mk 4 do throwError "decQData: not a Data{indentExpr e}"
  let nat := PSem.decNat
  let poly := PSem.decList (PSem.decPair (PSem.decList (PSem.decPair nat nat)) nat)
  return ⟨← PSem.decList nat (e.getArg! 0), ← poly (e.getArg! 1),
    ← PSem.decList poly (e.getArg! 2), ← PSem.decList poly (e.getArg! 3)⟩

/-- The number of numerals in `d`. -/
def Data.size (d : Data) : ℕ :=
  d.path.length + ((d.phase :: d.cons ++ d.rows).map fun P =>
    (P.map fun t => 2 * t.1.length + 1).sum).sum

/-- The blocks of `run p` from `s0` (see `PSem.blocks`), values as literals. -/
unsafe def qdataStatesImpl (p k w : ℕ) (s0 sts : Expr) : MetaM (Array (Expr × ℕ)) := do
  let s ← decQData s0
  let l ← evalExpr (List Step) (mkApp (mkConst ``List [Level.zero]) (mkConst ``Step)) sts
  return (PSem.blocks (fun st d => run p [st] d) Data.size k w s l).map fun (d, n) =>
    (qdataE d, n)

@[implemented_by qdataStatesImpl]
def qdataStates (_p _k _w : ℕ) (_s0 _sts : Expr) : MetaM (Array (Expr × ℕ)) := pure #[]

/-- `n_ok : QSem.runOK p sts s = true` and `n_eq : QSem.run p sts s = n`, through checkpoints
after blocks of at most `k` steps. -/
elab "qrun_chain " n:ident p:term:max sts:term:max s:term:max k:num : command =>
  liftTermElabM do
    let pE ← Term.elabTerm p (some (mkConst ``Nat))
    let stsE ← Term.elabTerm sts (some (mkApp (mkConst ``List [Level.zero]) (mkConst ``Step)))
    let sE ← Term.elabTerm s (some (mkConst ``Data))
    Term.synthesizeSyntheticMVarsNoPostponing
    let kind : PSem.ChainKind := ⟨``Step, ``Data, ``runOK, ``run, ``run_cut⟩
    PSem.chain kind qdataStates ((← getCurrNamespace) ++ n.getId) (← instantiateMVars pE)
      (← instantiateMVars stsE) (← instantiateMVars sE) k.getNat (2 ^ 62)

end QSem
end ArchytasZX
