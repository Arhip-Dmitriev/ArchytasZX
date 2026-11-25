/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaSpecNet

/-!
# The Fourier constant adder family at the diagram level

`famFca`: for every `n` and every `d > 0`, the engine's boxed `fourier_constant_adder` and its
spec, each expanded at `n` copies (`D0.famExpand n`, `D1.famExpand n`), are the same tensor over
`engE d n`: `evalL0`, `FcaEng.engFam` and `evalR0` for `n > 0`, both entries `1` at `n = 0`.
-/

namespace ArchytasZX

namespace FamFca

open List FcaEng

theorem fe0_0 : D0.famExpand 0 = ⟨[], [], [], [], []⟩ := by decide +kernel

theorem fe1_0 : D1.famExpand 0 =
    ⟨[⟨0, .Z, 0, 0, [0], []⟩, ⟨2, .X, 0, 1, [0], [0]⟩, ⟨4, .Z, 0, 0, [], [0]⟩],
      [(⟨0, true, 0⟩, ⟨2, false, 0⟩), (⟨4, false, 0⟩, ⟨2, true, 0⟩)], [], [], []⟩ := by
  decide +kernel

theorem bdims0 (n : ℕ) : (D0.famExpand n).bdims = List.replicate (2 * n) 0 := by
  refine List.eq_replicate_iff.2 ⟨by simp [Diagram.bdims, bnd_len0], fun v hv => ?_⟩
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 hv
  exact portDim0 n p

theorem bdims1 (n : ℕ) : (D1.famExpand n).bdims = List.replicate (2 * n) 0 := by
  refine List.eq_replicate_iff.2 ⟨by simp [Diagram.bdims, bnd_lenR], fun v hv => ?_⟩
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
  rw [bndR, List.mem_append] at hp
  rcases hp with hp | hp <;> obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hp <;>
    have hj := List.mem_range.1 hj
  · exact (portDim1 (mem_Z1_3 hj) true 1 (by simp [n3])).trans (legs1_31 _ _ _)
  · exact (portDim1 (mem_Z1_1 hj) false 1 (by simp [n1])).trans (legs1_11 _ _ _)

theorem eval_fe0_0 (d : ℕ) (σ : ℕ → ℕ) : (D0.famExpand 0).eval (engE d 0) σ = 1 := by
  rw [fe0_0]; simp [Diagram.eval, sumOver]

theorem eval_fe1_0 (d : ℕ) (σ : ℕ → ℕ) : (D1.famExpand 0).eval (engE d 0) σ = 1 := by
  rw [fe1_0]
  simp [Diagram.eval, Diagram.wireDim, Diagram.portDim, engE, engD, DNode.node, val,
    tZ, tX, engP, fou, Diagram.value, Diagram.locate, Diagram.bnd]
  rw [show range 2 = [0, 1] from rfl]
  simp [sumOver, List.findIdx?, List.findIdx?.go, ex]

/-- The expanded boxed side and spec are the same tensor for every `n` and `d > 0`. -/
theorem famFca (d : ℕ) (hd : 0 < d) (n : ℕ) :
    DEqual (engE d n) 1 (D0.famExpand n) 1 (D1.famExpand n) := by
  refine ⟨by rw [bdims0, bdims1], fun σ hσ => ?_⟩
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [eval_fe0_0, eval_fe1_0]
  have hσ1 : (D1.famExpand n).InRange (engE d n) σ := by
    intro i hi
    have := hσ i (by rw [bnd_len0]; rw [bnd_lenR] at hi; exact hi)
    rwa [bdims0, ← bdims1] at this
  have hL : InRange (engE d n) (engL n).bnd σ := by
    intro l hl
    have hl' : l < 2 * n := by simpa [engL] using hl
    have := hσ l (by rw [bnd_len0]; exact hl')
    rw [bdims0, getD_zero (fun v hv => List.eq_of_mem_replicate hv)] at this
    · rw [labDim_low d hn (by rw [rb_eq]; omega)]
      rwa [show (engE d n).dimv 0 = d from engD_zero d hn] at this
  rw [evalL0 d n σ hσ, evalR0 d hn σ hσ1]
  simpa using (engFam d hd n).2 σ hL

end FamFca

end ArchytasZX
