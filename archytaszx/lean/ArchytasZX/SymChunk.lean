/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymNet

/-!
# Symbolic normal forms checked node by node

`trStepB` is one node's step of `trNodesB` on the tail's expression. `trNodesB_cons` reduces
`trNodesB` on a node list to that step, so an export states each intermediate expression and
checks each step with a `decide` of its own. `symCheck_of` gives `symCheck` from both
networks' stated expressions.
-/

namespace ArchytasZX.Sym

/-- One node's step of `trNodesB` on the tail's expression `b`. -/
def trStepB (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node) (b : List STm) :
    Option (List STm) :=
  match trNode dt pt rl (OFF + i) n with
  | some a =>
    if exMulOK a b then some (nfEx fl (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)))
    else none
  | none => none

theorem trNodesB_nil (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) :
    trNodesB fl dt pt rl labs int pre i [] = some [⟨CMon.ofQ 1, [], [], []⟩] := rfl

theorem trNodesB_cons (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node)
    (ns : List Node) (b : List STm)
    (h : trNodesB fl dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns = some b) :
    trNodesB fl dt pt rl labs int pre i (n :: ns) = trStepB fl dt pt rl labs int pre i n b := by
  unfold trNodesB trStepB
  rw [h]
  cases trNode dt pt rl (OFF + i) n <;> rfl

theorem trNet_of (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (N : Net) (ns : List Node) (b : List STm)
    (hn : N.nodes = ns) (h : trNodesB fl dt pt rl labs N.int [] 0 ns = some b) :
    trNet fl dt pt rl labs N =
      some (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)) b) := by
  unfold trNet
  rw [hn, h]
  subst hn
  rfl

theorem symCheck_of (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (IX : List ℕ) (N0 N1 : Net)
    (S0 S1 e0 e1 : List STm) (h0 : trNet fl dt pt rl labs N0 = some e0)
    (h1 : trNet fl dt pt rl labs N1 = some e1)
    (h : (N0.bnd.isPerm N1.bnd && netChk fl dt pt rl labs IX N0 && netChk fl dt pt rl labs IX N1 &&
      exConst S0 && exConst S1 && (exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
      (nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty)) = true) :
    symCheck fl dt pt rl labs IX N0 N1 S0 S1 = true := by
  unfold symCheck checkWith
  rw [h0, h1]
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h ⊢
  obtain ⟨⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, f⟩ := h
  exact ⟨⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, Or.inl f⟩

/-! ## Strict evaluation

`fN`, `fZ`, `fQ`, `fL`, `fDM`, `fCM`, `fFac`, `fTm` pass a value to a continuation after
matching on every constructor in it, so the kernel hands the continuation literals instead of
the unevaluated terms that built them. Each is the identity (`fTm_eq`). -/

/-- `k n` after matching on `n`. -/
def fN {β : Sort _} (n : ℕ) (k : ℕ → β) : β :=
  match n with
  | 0 => k 0
  | m + 1 => k (m + 1)

/-- `k z` after matching on `z`. -/
def fZ {β : Sort _} (z : ℤ) (k : ℤ → β) : β :=
  match z with
  | .ofNat m => fN m fun m => k (.ofNat m)
  | .negSucc m => fN m fun m => k (.negSucc m)

/-- `k q` after matching on its numerator and denominator. -/
def fQ {β : Sort _} (q : ℚ) (k : ℚ → β) : β := fZ q.num fun a => fN q.den fun b => k (mkRat a b)

/-- `k l` after forcing every element by `f`. -/
def fL {α : Type} {β : Sort _} (f : α → (α → β) → β) : List α → (List α → β) → β
  | [], k => k []
  | a :: l, k => f a fun a => fL f l fun l => k (a :: l)

/-- A pair forced by `f` and `g`. -/
def fP {α γ : Type} {β : Sort _} (f : α → (α → β) → β) (g : γ → (γ → β) → β) (p : α × γ)
    (k : α × γ → β) : β :=
  match p with
  | (a, c) => f a fun a => g c fun c => k (a, c)

/-- A dimension monomial forced. -/
def fDM {β : Sort _} (m : DMon) (k : DMon → β) : β :=
  match m with
  | ⟨c, e⟩ => fN c fun c => fL fN e fun e => k ⟨c, e⟩

/-- A factor forced. -/
def fFac {β : Sort _} (f : Fac) (k : Fac → β) : β := fP fDM (fL (fP (fL fN) fZ)) f k

/-- A coefficient monomial forced. -/
def fCM {β : Sort _} (c : CMon) (k : CMon → β) : β :=
  match c with
  | ⟨q, r, sq, rc, am, gc⟩ =>
    fQ q fun q => fQ r fun r => fL fZ sq fun sq => fL fN rc fun rc => fL fZ am fun am =>
      fL (fP fZ fDM) gc fun gc => k ⟨q, r, sq, rc, am, gc⟩

/-- A term forced. -/
def fTm {β : Sort _} (t : STm) (k : STm → β) : β :=
  match t with
  | ⟨c, sums, dels, phs⟩ =>
    fCM c fun c => fL (fP fN fDM) sums fun sums => fL fFac dels fun dels =>
      fL fFac phs fun phs => k ⟨c, sums, dels, phs⟩

theorem fN_eq {β : Sort _} (n : ℕ) (k : ℕ → β) : fN n k = k n := by cases n <;> rfl

theorem fZ_eq {β : Sort _} (z : ℤ) (k : ℤ → β) : fZ z k = k z := by
  cases z <;> simp [fZ, fN_eq]

theorem fQ_eq {β : Sort _} (q : ℚ) (k : ℚ → β) : fQ q k = k q := by
  simp [fQ, fZ_eq, fN_eq, Rat.mkRat_self]

theorem fL_eq {α : Type} {β : Sort _} (f : α → (α → β) → β)
    (hf : ∀ (a : α) (k : α → β), f a k = k a) (l : List α) (k : List α → β) : fL f l k = k l := by
  induction l generalizing k with
  | nil => rfl
  | cons a l ih => simp [fL, hf, ih]

theorem fP_eq {α γ : Type} {β : Sort _} (f : α → (α → β) → β) (g : γ → (γ → β) → β)
    (hf : ∀ (a : α) (k : α → β), f a k = k a) (hg : ∀ (c : γ) (k : γ → β), g c k = k c)
    (p : α × γ) (k : α × γ → β) : fP f g p k = k p := by
  obtain ⟨a, c⟩ := p
  simp [fP, hf, hg]

theorem fDM_eq {β : Sort _} (m : DMon) (k : DMon → β) : fDM m k = k m := by
  obtain ⟨c, e⟩ := m
  simp [fDM, fN_eq, fL_eq _ (fun a k => fN_eq a k)]

theorem fFac_eq {β : Sort _} (f : Fac) (k : Fac → β) : fFac f k = k f :=
  fP_eq _ _ (fun a k => fDM_eq a k)
    (fun c k => fL_eq _ (fun p k => fP_eq _ _ (fun a k => fL_eq _ (fun a k => fN_eq a k) a k)
      (fun z k => fZ_eq z k) p k) c k) f k

theorem fCM_eq {β : Sort _} (c : CMon) (k : CMon → β) : fCM c k = k c := by
  obtain ⟨q, r, sq, rc, am, gc⟩ := c
  simp only [fCM, fQ_eq, fL_eq _ (fun a k => fZ_eq a k), fL_eq _ (fun a k => fN_eq a k),
    fL_eq _ (fun p k => fP_eq _ _ (fun a k => fZ_eq a k) (fun a k => fDM_eq a k) p k)]

theorem fTm_eq {β : Sort _} (t : STm) (k : STm → β) : fTm t k = k t := by
  obtain ⟨c, sums, dels, phs⟩ := t
  simp only [fTm, fCM_eq, fL_eq _ (fun a k => fFac_eq a k),
    fL_eq _ (fun p k => fP_eq _ _ (fun a k => fN_eq a k) (fun a k => fDM_eq a k) p k)]

/-- `elimAll`, the term forced after each elimination. -/
def elimAllS : ℕ → STm → STm
  | 0, t => t
  | n + 1, t => match elimStep t with
    | some t' => fTm t' (elimAllS n)
    | none => t

theorem elimAllS_eq : ∀ (n : ℕ) (t : STm), elimAllS n t = elimAll n t := by
  intro n
  induction n with
  | zero => intro t; rfl
  | succ n ih =>
    intro t
    simp only [elimAllS, elimAll]
    cases elimStep t with
    | none => rfl
    | some t' => simp [fTm_eq, ih]

/-- `normTm`, the term forced after each round. -/
def normTmS (fl : List ℕ) : ℕ → STm → Option STm
  | 0, t => some t
  | r + 1, t =>
    match canonTm fl (fTm t fun t => elimAllS (t.sums.length + 1) t) with
    | none => none
    | some t' => fTm t' (normTmS fl r)

theorem normTmS_eq (fl : List ℕ) : ∀ (r : ℕ) (t : STm), normTmS fl r t = normTm fl r t := by
  intro r
  induction r with
  | zero => intro t; rfl
  | succ r ih =>
    intro t
    simp only [normTmS, normTm, fTm_eq, elimAllS_eq]
    cases canonTm fl (elimAll (t.sums.length + 1) t) with
    | none => rfl
    | some t' => simp [fTm_eq, ih]

/-- `nfEx` from the expression `acc`, each normalised term and each sum forced. -/
def nfGo (fl : List ℕ) : List STm → List STm → List STm
  | acc, [] => acc
  | acc, t :: ts =>
    match fTm t (normTmS fl ROUNDS) with
    | none => nfGo fl acc ts
    | some t' => fL fTm (tmAdd t' acc) fun acc => nfGo fl acc ts

theorem nfGo_eq (fl : List ℕ) : ∀ (e acc : List STm),
    nfGo fl acc e = (e.filterMap (normTm fl ROUNDS)).foldl (fun acc t => tmAdd t acc) acc := by
  intro e
  induction e with
  | nil => intro acc; rfl
  | cons t ts ih =>
    intro acc
    simp only [nfGo, fTm_eq, normTmS_eq, List.filterMap_cons]
    cases normTm fl ROUNDS t with
    | none => simp [ih]
    | some t' => simp [fL_eq _ (fun a k => fTm_eq a k), ih]

/-- `trStepB` with the strict normaliser. -/
def trStepS (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node) (b : List STm) :
    Option (List STm) :=
  match trNode dt pt rl (OFF + i) n with
  | some a =>
    if exMulOK a b then some (nfGo fl [] (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)))
    else none
  | none => none

theorem trStepS_eq (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node)
    (b : List STm) :
    trStepS fl dt pt rl labs int pre i n b = trStepB fl dt pt rl labs int pre i n b := by
  simp only [trStepS, trStepB, nfGo_eq, nfEx]

/-! ## Blocks of nodes -/

/-- The labels before a node list, after `xs`. -/
def blockPre : List ℕ → List Node → List ℕ
  | pre, [] => pre
  | pre, n :: ns => blockPre (pre ++ (n.outs ++ n.ins)) ns

/-- `trNodesB` on the nodes `xs` with the tail's expression `b0`, each step strict. -/
def trBlockS (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int : List ℕ) (b0 : List STm) : List ℕ → ℕ → List Node →
    Option (List STm)
  | _, _, [] => some b0
  | pre, i, n :: ns =>
    match trBlockS fl dt pt rl labs int b0 (pre ++ (n.outs ++ n.ins)) (i + 1) ns with
    | some b => fL fTm b fun b => trStepS fl dt pt rl labs int pre i n b
    | none => none

theorem trNodesB_cons' (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node)
    (ns : List Node) :
    trNodesB fl dt pt rl labs int pre i (n :: ns) =
      match trNodesB fl dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns with
      | some b => trStepB fl dt pt rl labs int pre i n b
      | none => none := by
  cases h : trNodesB fl dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns with
  | none =>
    unfold trNodesB
    rw [h]
    cases trNode dt pt rl (OFF + i) n <;> rfl
  | some b => exact trNodesB_cons fl dt pt rl labs int pre i n ns b h

theorem trNodesB_block (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ) (b : List STm) :
    ∀ (xs ys : List Node) (pre : List ℕ) (i : ℕ),
    trNodesB fl dt pt rl labs int (blockPre pre xs) (i + xs.length) ys = some b →
    trNodesB fl dt pt rl labs int pre i (xs ++ ys) = trBlockS fl dt pt rl labs int b pre i xs := by
  intro xs
  induction xs with
  | nil => intro ys pre i h; simpa [blockPre, trBlockS] using h
  | cons n xs ih =>
    intro ys pre i h
    have h' : trNodesB fl dt pt rl labs int (blockPre (pre ++ (n.outs ++ n.ins)) xs)
        (i + 1 + xs.length) ys = some b := by
      have e : i + (n :: xs).length = i + 1 + xs.length := by simp [List.length_cons]; omega
      rw [← e]
      exact h
    rw [List.cons_append, trNodesB_cons', ih ys _ _ h']
    simp only [trBlockS]
    cases trBlockS fl dt pt rl labs int b (pre ++ (n.outs ++ n.ins)) (i + 1) xs with
    | none => rfl
    | some c => simp [fL_eq _ (fun a k => fTm_eq a k), trStepS_eq]

end ArchytasZX.Sym
