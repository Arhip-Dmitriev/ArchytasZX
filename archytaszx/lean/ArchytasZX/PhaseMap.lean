/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bundle

/-!
# Phase rows read through a map

`Equal.of_phRows`: an equation over one environment gives the equation over another one with
the same dimensions, labels and multiplicities, each network's phase indices read through its
own map, when the two environments agree through those maps on every phase row the networks
use.
-/

namespace ArchytasZX

/-- `x` through the pairs `ps`, itself when it has no pair. -/
def phMap (ps : List (ℕ × ℕ)) (x : ℕ) : ℕ := (ps.lookup x).getD x

/-- A node's phase index through `g`; a B or S node keeps its own. -/
def Node.mapPh (g : ℕ → ℕ) (n : Node) : Node :=
  match n.kind with
  | .B | .S => n
  | _ => { n with ph := g n.ph }

/-- Every node's phase index through `g`. -/
def Net.mapPh (g : ℕ → ℕ) (N : Net) : Net := ⟨N.nodes.map (Node.mapPh g), N.int, N.bnd⟩

/-- Every node other than B and S reads a phase row in `qs`. -/
def phIn (qs : List ℕ) (N : Net) : Bool :=
  N.nodes.all fun n => n.kind == .B || n.kind == .S || qs.contains n.ph

theorem tensor_mapPh (E E' : Env) (g : ℕ → ℕ) (hm : E'.mult = E.mult) (b : Kind) (d ph : ℕ)
    (hp : E'.phase (g ph) = E.phase ph) (hc : E'.coef (g ph) = E.coef ph) (os is : List ℕ) :
    tensor E' b d (g ph) os is = tensor E b d ph os is := by
  induction b generalizing os is with
  | Bun m b ih => simp only [tensor, copyOuts, copyIns, hm, ih]
  | ZB cs => simp only [tensor, hp, hm]
  | XB co ci => simp only [tensor, hp, hm]
  | WB cs => rcases is with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> simp only [tensor, hm]
  | Z => simp only [tensor, hp]
  | X => simp only [tensor, hp]
  | ZW => simp only [tensor, hc]
  | XW => simp only [tensor, hc]
  | _ => rcases os with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> rfl

theorem val_mapPh (E E' : Env) (g : ℕ → ℕ) (hd : E'.dimv = E.dimv) (hm : E'.mult = E.mult)
    (σ : ℕ → ℕ) (n : Node)
    (hp : n.kind ≠ .B ∧ n.kind ≠ .S →
      E'.phase (g n.ph) = E.phase n.ph ∧ E'.coef (g n.ph) = E.coef n.ph) :
    val E' σ (n.mapPh g) = val E σ n := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  by_cases hk : kind ≠ .B ∧ kind ≠ .S
  · have e : Node.mapPh g ⟨kind, os, is, dim, ph⟩ = ⟨kind, os, is, dim, g ph⟩ := by
      cases kind <;> simp_all [Node.mapPh]
    obtain ⟨h1, h2⟩ := hp hk
    rw [e, val_tensor _ σ kind hk, val_tensor E σ kind hk, hd]
    exact tensor_mapPh E E' g hm kind _ ph h1 h2 _ _
  · have hbs : kind = .B ∨ kind = .S := by
      by_contra h'; exact hk ⟨fun e => h' (Or.inl e), fun e => h' (Or.inr e)⟩
    rcases hbs with rfl | rfl <;>
      rcases os with _ | ⟨a, _ | ⟨b, _ | _⟩⟩ <;> rcases is with _ | ⟨c, _ | ⟨e, _ | _⟩⟩ <;>
      simp [val, Node.mapPh, hd]

theorem eval_mapPh (E E' : Env) (g : ℕ → ℕ) (qs : List ℕ) (hd : E'.dimv = E.dimv)
    (hl : E'.labs = E.labs) (hm : E'.mult = E.mult)
    (hp : ∀ q ∈ qs, E'.phase (g q) = E.phase q) (hc : ∀ q ∈ qs, E'.coef (g q) = E.coef q)
    (N : Net) (hN : phIn qs N = true) (σ : ℕ → ℕ) : eval E' (N.mapPh g) σ = eval E N σ := by
  have hD : labDim E' = labDim E := by funext l; simp only [labDim, hd, hl]
  simp only [eval, Net.mapPh, hD]
  refine sumOver_congr _ _ _ _ _ fun τ _ _ => ?_
  simp only [prodVal, List.map_map]
  congr 1
  refine List.map_congr_left fun n hn => ?_
  refine val_mapPh E E' g hd hm τ n fun hk => ?_
  have h := List.all_eq_true.1 hN n hn
  have hq : n.ph ∈ qs := by simpa [hk.1, hk.2] using h
  exact ⟨hp _ hq, hc _ hq⟩

/-- Two families with the same dimensions and bundles have the same dimension values. -/
theorem Fam.env_dimv_congr (F F' : Fam) (ks : ℕ → ℕ) (hd : F.dims = F'.dims)
    (hb : F.bund = F'.bund) : (F'.env ks).dimv = (F.env ks).dimv := by
  have hc : ∀ n i, F'.chainF n i = F.chainF n i := by
    intro n
    induction n with
    | zero => intro i; rfl
    | succ n ih => intro i; simp only [Fam.chainF, hb, ih]
  have hr : ∀ n i, F'.rootF n i = F.rootF n i := by
    intro n
    induction n with
    | zero => intro i; rfl
    | succ n ih => intro i; simp only [Fam.rootF, hb, ih]
  funext i
  simp only [Fam.env, Fam.dimv, Fam.root, Fam.chain, hb, hd, hc, hr]

/-- An equation carried to `E'` through the phase maps `gN` and `gM` of its two sides. -/
theorem Equal.of_phRows (E E' : Env) (gN gM : ℕ → ℕ) (qN qM : List ℕ)
    (hd : E'.dimv = E.dimv) (hl : E'.labs = E.labs) (hm : E'.mult = E.mult)
    (hpN : ∀ q ∈ qN, E'.phase (gN q) = E.phase q) (hcN : ∀ q ∈ qN, E'.coef (gN q) = E.coef q)
    (hpM : ∀ q ∈ qM, E'.phase (gM q) = E.phase q) (hcM : ∀ q ∈ qM, E'.coef (gM q) = E.coef q)
    (N M N' M' : Net) (hqN : phIn qN N = true) (hqM : phIn qM M = true)
    (hN : N.mapPh gN = N') (hM : M.mapPh gM = M') {s s' : ℂ}
    (h : Equal E s N s' M) : Equal E' s N' s' M' := by
  subst hN hM
  have hD : labDim E' = labDim E := by funext l; simp only [labDim, hd, hl]
  refine ⟨h.1, fun σ hσ => ?_⟩
  have hσ' : InRange E N.bnd σ := fun l hl' => by
    have := hσ l hl'
    rwa [hD] at this
  rw [eval_mapPh E E' gN qN hd hl hm hpN hcN N hqN, eval_mapPh E E' gM qM hd hl hm hpM hcM M hqM]
  exact h.2 σ hσ'

end ArchytasZX
