// Copyright 2026 Arkhip A. Dmitriev
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Headless Quantomatic driver: java -cp quanto.jar quanto.bench.QuantoBench PAYLOAD. The payload
// names the theory, the shared rule file and the two graphs. Each graph is rewritten to a
// fixpoint with the size-reducing rules of the rule file (lhs larger than rhs), first match
// first; equal when each normal form matches the other and their vertex and edge counts agree,
// unknown otherwise. Prints one JSON line.

package quanto.bench

import java.io.File

import quanto.data._
import quanto.rewrite.{Matcher, Rewriter}
import quanto.util.json._

object QuantoBench {
  private def size(g: Graph): Int = g.verts.size + g.edges.size

  private def normalise(g0: Graph, rules: List[Rule], maxSteps: Int): (Graph, Int) = {
    var g = g0
    var steps = 0
    var changed = true
    while (changed && steps < maxSteps) {
      changed = false
      val it = rules.iterator
      while (!changed && it.hasNext) {
        val rule = it.next()
        Matcher.findMatches(rule.lhs, g).headOption.foreach { m =>
          g = Rewriter.rewrite(m, rule.rhs, rule.description)._1
          steps += 1
          changed = true
        }
      }
    }
    (g, steps)
  }

  private def same(a: Graph, b: Graph): Boolean =
    a.verts.size == b.verts.size && a.edges.size == b.edges.size &&
      Matcher.findMatches(a, b).nonEmpty && Matcher.findMatches(b, a).nonEmpty

  def main(args: Array[String]): Unit = {
    val p = Json.parse(new File(args(0)))
    val theory = Theory.fromFile((p / "theory").stringValue)
    val ruleJson = Json.parse(new File((p / "rules").stringValue)) / "rules"
    val rules = ruleJson.mapValue.toList.sortBy(_._1).map { case (name, j) =>
      Rule.fromJson(j, theory, Some(RuleDesc(name)))
    }.filter(r => size(r.lhs) > size(r.rhs))
    val maxSteps = (p / "max_steps").intValue
    val c0 = java.lang.management.ManagementFactory.getThreadMXBean.getCurrentThreadCpuTime
    val w0 = System.nanoTime()
    val lhs = Graph.fromJson(p / "lhs", theory)
    val rhs = Graph.fromJson(p / "rhs", theory)
    val (nl, sl) = normalise(lhs, rules, maxSteps)
    val (nr, sr) = normalise(rhs, rules, maxSteps)
    val verdict = if (same(nl, nr)) "equal" else "unknown"
    val cpu = (java.lang.management.ManagementFactory.getThreadMXBean.getCurrentThreadCpuTime - c0) / 1e9
    val wall = (System.nanoTime() - w0) / 1e9
    val out = JsonObject(
      "verdict" -> JsonString(verdict),
      "steps" -> JsonInt(sl + sr),
      "rules" -> JsonInt(rules.size),
      "vertices_before" -> JsonInt(lhs.verts.size),
      "vertices_after" -> JsonInt(nl.verts.size),
      "edges_after" -> JsonInt(nl.edges.size),
      "cpu" -> JsonDouble(cpu),
      "wall" -> JsonDouble(wall))
    println(out.toString)
  }
}
