/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumSupport

/-!
# Step lists checked through checkpoints

`run_cut` and `aruns_cut` split a step list after `k` steps at a stated intermediate value.
The commands `run_chain` and `aruns_chain` compute the values after every `k` steps with
compiled code, state each as a literal definition, prove each block of `k` steps by `decide` in
its own declaration, and compose the blocks into `<n>_ok : runOK M sts s = true` and
`<n>_eq : run M sts s = <n>` (and the same for `arunsOK` and `aruns`).
-/

namespace ArchytasZX
namespace PSem

theorem runOK_append (M : ℕ) : ∀ (a b : List Step) (s : Data),
    runOK M (a ++ b) s = (runOK M a s && runOK M b (run M a s))
  | [], b, s => by simp [runOK, run]
  | st :: a, b, s => by
    simp only [List.cons_append, runOK, run, runOK_append M a b, Bool.and_assoc]

theorem run_append (M : ℕ) : ∀ (a b : List Step) (s : Data),
    run M (a ++ b) s = run M b (run M a s)
  | [], _, _ => rfl
  | _ :: a, b, _ => run_append M a b _

theorem run_cut (M : ℕ) (sts : List Step) (k : ℕ) (s s' : Data)
    (he : (runOK M (sts.take k) s && decide (run M (sts.take k) s = s')) = true) :
    runOK M sts s = runOK M (sts.drop k) s' ∧ run M sts s = run M (sts.drop k) s' := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at he
  obtain ⟨h, e⟩ := he
  constructor
  · calc runOK M sts s = runOK M (sts.take k ++ sts.drop k) s := by rw [List.take_append_drop]
      _ = runOK M (sts.drop k) s' := by rw [runOK_append, h, e, Bool.true_and]
  · calc run M sts s = run M (sts.take k ++ sts.drop k) s := by rw [List.take_append_drop]
      _ = run M (sts.drop k) s' := by rw [run_append, e]

deriving instance DecidableEq for AData

theorem arunsOK_append (M : ℕ) : ∀ (a b : List AStep) (t : AData),
    arunsOK M (a ++ b) t = (arunsOK M a t && arunsOK M b (aruns M a t))
  | [], b, t => by simp [arunsOK, aruns]
  | st :: a, b, t => by
    simp only [List.cons_append, arunsOK, aruns, arunsOK_append M a b, Bool.and_assoc]

theorem aruns_append (M : ℕ) : ∀ (a b : List AStep) (t : AData),
    aruns M (a ++ b) t = aruns M b (aruns M a t)
  | [], _, _ => rfl
  | _ :: a, b, _ => aruns_append M a b _

theorem aruns_cut (M : ℕ) (sts : List AStep) (k : ℕ) (t t' : AData)
    (he : (arunsOK M (sts.take k) t && decide (aruns M (sts.take k) t = t')) = true) :
    arunsOK M sts t = arunsOK M (sts.drop k) t' ∧ aruns M sts t = aruns M (sts.drop k) t' := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at he
  obtain ⟨h, e⟩ := he
  constructor
  · calc arunsOK M sts t
        = arunsOK M (sts.take k ++ sts.drop k) t := by rw [List.take_append_drop]
      _ = arunsOK M (sts.drop k) t' := by rw [arunsOK_append, h, e, Bool.true_and]
  · calc aruns M sts t = aruns M (sts.take k ++ sts.drop k) t := by rw [List.take_append_drop]
      _ = aruns M (sts.drop k) t' := by rw [aruns_append, e]


open Lean Meta Elab Command

/-- A list literal of element type `ty`. -/
def listE (ty : Expr) (xs : List Expr) : Expr :=
  xs.foldr (fun x acc => mkApp3 (mkConst ``List.cons [Level.zero]) ty x acc)
    (mkApp (mkConst ``List.nil [Level.zero]) ty)

/-- A list of raw numerals. -/
def natsE (xs : List ℕ) : Expr := listE (mkConst ``Nat) (xs.map mkRawNatLit)

/-- A list of lists of raw numerals. -/
def f2E (q : F2) : Expr :=
  listE (mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat)) (q.map natsE)

/-- `d` as a literal of raw numerals. -/
def dataE (d : Data) : Expr :=
  let monoT := mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat)
  let termT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) monoT (mkConst ``Nat)
  let term := fun (t : Mono × ℕ) =>
    mkApp4 (mkConst ``Prod.mk [Level.zero, Level.zero]) monoT (mkConst ``Nat) (natsE t.1)
      (mkRawNatLit t.2)
  let f2T := mkApp (mkConst ``List [Level.zero]) monoT
  mkApp4 (mkConst ``Data.mk) (natsE d.path) (listE termT (d.phase.map term))
    (listE f2T (d.cons.map f2E)) (listE f2T (d.rows.map f2E))

/-- `t` as a literal of raw numerals. -/
def adataE (t : AData) : Expr :=
  let monoT := mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat)
  let f2T := mkApp (mkConst ``List [Level.zero]) monoT
  let atomT := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) (mkConst ``Nat) f2T
  let atom := fun (a : ℕ × F2) =>
    mkApp4 (mkConst ``Prod.mk [Level.zero, Level.zero]) (mkConst ``Nat) f2T (mkRawNatLit a.1)
      (f2E a.2)
  mkApp3 (mkConst ``AData.mk) (dataE t.s) (natsE t.held) (listE atomT (t.atoms.map atom))

/-- `e` with every constructor argument in kernel weak head normal form, recursively. -/
partial def knorm (e : Expr) : MetaM Expr := do
  let env ← getEnv
  let w ← match Kernel.whnf env {} e with
    | .ok w => pure w
    | .error _ => throwError "knorm: the kernel cannot reduce{indentExpr e}"
  if w.isLit then return w
  let .const c _ := w.getAppFn | throwError "knorm: not a constructor{indentExpr w}"
  let some (.ctorInfo info) := env.find? c | throwError "knorm: not a constructor{indentExpr w}"
  let args := w.getAppArgs
  if c == ``Nat.succ then
    let a ← knorm args[0]!
    if let some n := a.rawNatLit? then return mkRawNatLit (n + 1)
    return mkApp w.getAppFn a
  let mut out := args
  for i in [info.numParams:args.size] do
    out := out.set! i (← knorm args[i]!)
  return mkAppN w.getAppFn out

/-- The number a numeral, `Nat.zero` or `Nat.succ` term denotes. -/
partial def decNat (e : Expr) : MetaM ℕ := do
  if let some n := e.rawNatLit? then return n
  if let some n := e.nat? then return n
  if e.isConstOf ``Nat.zero then return 0
  if e.isAppOfArity ``Nat.succ 1 then return (← decNat e.appArg!) + 1
  throwError "decNat: not a numeral{indentExpr e}"

/-- The list a `List.cons`/`List.nil` term denotes, its elements read by `f`. -/
def decList {α : Type} (f : Expr → MetaM α) (e : Expr) : MetaM (List α) := do
  let mut out := #[]
  let mut e := e
  while e.isAppOfArity ``List.cons 3 do
    out := out.push (← f (e.getArg! 1))
    e := e.getArg! 2
  unless e.isAppOfArity ``List.nil 1 do throwError "decList: not a list{indentExpr e}"
  return out.toList

/-- The pair a `Prod.mk` term denotes, its components read by `f` and `g`. -/
def decPair {α β : Type} (f : Expr → MetaM α) (g : Expr → MetaM β) (e : Expr) :
    MetaM (α × β) := do
  unless e.isAppOfArity ``Prod.mk 4 do throwError "decPair: not a pair{indentExpr e}"
  return (← f (e.getArg! 2), ← g (e.getArg! 3))

/-- The value a `Data.mk` term of numerals and lists denotes. -/
def decData (e : Expr) : MetaM Data := do
  unless e.isAppOfArity ``Data.mk 4 do throwError "decData: not a Data{indentExpr e}"
  let f2 := decList (decList decNat)
  return ⟨← decList decNat (e.getArg! 0), ← decList (decPair (decList decNat) decNat) (e.getArg! 1),
    ← decList f2 (e.getArg! 2), ← decList f2 (e.getArg! 3)⟩

/-- The value an `AData.mk` term of numerals and lists denotes. -/
def decAData (e : Expr) : MetaM AData := do
  unless e.isAppOfArity ``AData.mk 3 do throwError "decAData: not an AData{indentExpr e}"
  return ⟨← decData (e.getArg! 0), ← decList decNat (e.getArg! 1),
    ← decList (decPair decNat (decList (decList decNat))) (e.getArg! 2)⟩

/-- The values after each block of `k` steps of `run M` from `s0`. -/
unsafe def dataStatesImpl (M k : ℕ) (s0 sts : Expr) : MetaM (Array Expr) := do
  let s ← decData s0
  let l ← evalExpr (List Step) (mkApp (mkConst ``List [Level.zero]) (mkConst ``Step)) sts
  let mut out := #[]
  let mut cur := s
  let mut rest := l
  while !rest.isEmpty do
    cur := run M (rest.take k) cur
    out := out.push (dataE cur)
    rest := rest.drop k
  return out

@[implemented_by dataStatesImpl]
def dataStates (_M _k : ℕ) (_s0 _sts : Expr) : MetaM (Array Expr) := pure #[]

/-- The values after each block of `k` steps of `aruns M` from `t0`. -/
unsafe def adataStatesImpl (M k : ℕ) (t0 sts : Expr) : MetaM (Array Expr) := do
  let t ← decAData t0
  let l ← evalExpr (List AStep) (mkApp (mkConst ``List [Level.zero]) (mkConst ``AStep)) sts
  let mut out := #[]
  let mut cur := t
  let mut rest := l
  while !rest.isEmpty do
    cur := aruns M (rest.take k) cur
    out := out.push (adataE cur)
    rest := rest.drop k
  return out

@[implemented_by adataStatesImpl]
def adataStates (_M _k : ℕ) (_t0 _sts : Expr) : MetaM (Array Expr) := pure #[]

/-- The names of a step list's type, value type, checker, runner and cut lemma. -/
structure ChainKind where
  step : Name
  ty : Name
  ok : Name
  run : Name
  cut : Name

/-- `value` added as a definition of type `type`. -/
def addLit (n : Name) (type value : Expr) : MetaM Unit :=
  addDecl (.defnDecl (mkDefinitionValEx n [] type value .abbrev .safe [n]))

/-- `type` added as a theorem proved by `decide`, checked by the kernel at once. -/
def addDecided (n : Name) (type : Expr) : MetaM Unit := do
  let proof ← mkDecideProof type
  withOptions (fun o => o.setBool `Elab.async false) do
    addDecl (.thmDecl { name := n, levelParams := [], type, value := proof })

/-- `type` added as a theorem with proof `value`. -/
def addThm (n : Name) (type value : Expr) : MetaM Unit :=
  withOptions (fun o => o.setBool `Elab.async false) do
    addDecl (.thmDecl { name := n, levelParams := [], type, value })

/-- The checkpoint declarations of `sts` from `start` in blocks of `k`, under the prefix `n`. -/
def chain (kind : ChainKind) (states : ℕ → ℕ → Expr → Expr → MetaM (Array Expr)) (n : Name)
    (ME stsE startE : Expr) (k : ℕ) : MetaM Unit := do
  let some M ← evalNat ME | throwError "chain: {ME} is not a numeral"
  let name := fun (s : String) => n.appendAfter s
  let ty := mkConst kind.ty
  let s0 ← knorm startE
  addLit (name "_s0") ty s0
  let e0T ← mkEq startE (mkConst (name "_s0"))
  addDecided (name "_e0") e0T
  let vals ← states M k s0 stsE
  let kE := mkNatLit k
  let stepTy := mkConst kind.step
  let take := fun R => mkAppN (mkConst ``List.take [Level.zero]) #[stepTy, kE, R]
  let drop := fun R => mkAppN (mkConst ``List.drop [Level.zero]) #[stepTy, kE, R]
  let okOf := fun R s => mkApp3 (mkConst kind.ok) ME R s
  let runOf := fun R s => mkApp3 (mkConst kind.run) ME R s
  let mut R := stsE
  let mut okC ← mkEqRefl (okOf stsE (mkConst (name "_s0")))
  let mut runC ← mkEqRefl (runOf stsE (mkConst (name "_s0")))
  for i in [0:vals.size] do
    let si := mkConst (name s!"_s{i}")
    let si1 := mkConst (name s!"_s{i + 1}")
    addLit (name s!"_s{i + 1}") ty vals[i]!
    let both := mkApp2 (mkConst ``and) (okOf (take R) si)
      (← mkDecide (← mkEq (runOf (take R) si) si1))
    addDecided (name s!"_h{i}") (← mkEq both (mkConst ``Bool.true))
    let cut := mkAppN (mkConst kind.cut) #[ME, R, kE, si, si1, mkConst (name s!"_h{i}")]
    okC ← mkEqTrans okC (← mkAppM ``And.left #[cut])
    runC ← mkEqTrans runC (← mkAppM ``And.right #[cut])
    R := drop R
  let sn := mkConst (name s!"_s{vals.size}")
  addDecided (name "_hend") (← mkEq (okOf R sn) (mkConst ``Bool.true))
  addDecided (name "_qend") (← mkEq (runOf R sn) sn)
  addLit n ty (mkConst (name s!"_s{vals.size}"))
  let e0 := mkConst (name "_e0")
  let okT ← mkEq (okOf stsE startE) (mkConst ``Bool.true)
  let okP ← mkEqTrans (← mkCongrArg (mkApp2 (mkConst kind.ok) ME stsE) e0)
    (← mkEqTrans okC (mkConst (name "_hend")))
  addThm (name "_ok") okT okP
  let runT ← mkEq (runOf stsE startE) (mkConst n)
  let runP ← mkEqTrans (← mkCongrArg (mkApp2 (mkConst kind.run) ME stsE) e0)
    (← mkEqTrans runC (mkConst (name "_qend")))
  addThm (name "_eq") runT runP

/-- `n_ok : runOK M sts s = true` and `n_eq : run M sts s = n`, through checkpoints every `k`
steps. -/
elab "run_chain " n:ident M:term:max sts:term:max s:term:max k:num : command =>
  liftTermElabM do
    let ME ← Term.elabTerm M (some (mkConst ``Nat))
    let stsE ← Term.elabTerm sts (some (mkApp (mkConst ``List [Level.zero]) (mkConst ``Step)))
    let sE ← Term.elabTerm s (some (mkConst ``Data))
    Term.synthesizeSyntheticMVarsNoPostponing
    let kind : ChainKind := ⟨``Step, ``Data, ``runOK, ``run, ``run_cut⟩
    chain kind dataStates ((← getCurrNamespace) ++ n.getId) (← instantiateMVars ME)
      (← instantiateMVars stsE) (← instantiateMVars sE) k.getNat

/-- `n_ok : arunsOK M sts t = true` and `n_eq : aruns M sts t = n`, through checkpoints every
`k` steps. -/
elab "aruns_chain " n:ident M:term:max sts:term:max t:term:max k:num : command =>
  liftTermElabM do
    let ME ← Term.elabTerm M (some (mkConst ``Nat))
    let stsE ← Term.elabTerm sts (some (mkApp (mkConst ``List [Level.zero]) (mkConst ``AStep)))
    let tE ← Term.elabTerm t (some (mkConst ``AData))
    Term.synthesizeSyntheticMVarsNoPostponing
    let kind : ChainKind := ⟨``AStep, ``AData, ``arunsOK, ``aruns, ``aruns_cut⟩
    chain kind adataStates ((← getCurrNamespace) ++ n.getId) (← instantiateMVars ME)
      (← instantiateMVars stsE) (← instantiateMVars tE) k.getNat

end PSem
end ArchytasZX
