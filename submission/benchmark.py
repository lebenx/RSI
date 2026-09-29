"""Generate the reproducible SharedPremiseBench JSONL benchmark.

The benchmark is synthetic by design, but its public task view and hidden
lineage are separated. Every imagined rollout carries a ground-truth parent
premise, source ID and sample ID. The generator is deterministic and writes
train/dev/test splits, statistics and a small SVG lineage example.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from collections import Counter
from pathlib import Path

FAMILIES = (
    "hidden_state_dependency",
    "permission_tool_availability",
    "object_property_uncertainty",
    "resource_constraint",
    "planning_failure",
)
DIFFICULTIES = ("easy", "medium", "hard")

def stable_int(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

def split_for(seed):
    x = stable_int(f"split-{seed}") % 100
    return "train" if x < 70 else ("dev" if x < 85 else "test")

def family_spec(family, difficulty, rng):
    n = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    if family == "hidden_state_dependency":
        key = rng.choice(["door_locked", "bridge_intact", "switch_on"])
        value = bool(rng.getrandbits(1))
        public = f"The {key.replace('_',' ')} is not directly observed. You may inspect it or attempt the route."
        actions = ["inspect", "take_safe_route", "take_short_route", "commit_route"]
        return key, value, public, actions
    if family == "permission_tool_availability":
        key = rng.choice(["write_permission", "admin_token", "tool_available"])
        value = bool(rng.getrandbits(1))
        public = f"A required permission or tool ({key.replace('_',' ')}) may be unavailable. Verification has a cost."
        actions = ["verify_permission", "use_tool", "manual_fallback", "abort"]
        return key, value, public, actions
    if family == "object_property_uncertainty":
        key = rng.choice(["fragile", "edible", "conductive"])
        value = bool(rng.getrandbits(1))
        public = f"An object's property ({key}) is hidden. Handling it without checking may cause damage."
        actions = ["inspect_object", "handle_gently", "handle_fast", "discard"]
        return key, value, public, actions
    if family == "resource_constraint":
        key = rng.choice(["battery", "water", "budget"])
        value = bool(rng.getrandbits(1))
        public = f"The remaining {key} is partially observed and may be insufficient for a long plan."
        actions = ["measure_resource", "conserve", "spend_now", "switch_plan"]
        return key, value, public, actions
    key = rng.choice(["account_access", "safe_path", "machine_ready"])
    value = bool(rng.getrandbits(1))
    public = f"Several plans have different surface actions but share an unverified premise ({key.replace('_',' ')})."
    actions = ["plan_a", "plan_b", "plan_c", "verify_premise"]
    return key, value, public, actions

def action_return(family, hidden, action, rng, difficulty):
    # Values are environment outcomes, not planner labels. A fixed table makes
    # the benchmark easy to audit and independent of any language model.
    if family == "hidden_state_dependency":
        v = {"inspect": 0.1, "take_safe_route": 1.0, "take_short_route": 1.8 if hidden else -2.0, "commit_route": .4}[action]
    elif family == "permission_tool_availability":
        v = {"verify_permission": -.2, "use_tool": 2.0 if hidden else -2.5, "manual_fallback": .8, "abort": -.4}[action]
    elif family == "object_property_uncertainty":
        v = {"inspect_object": -.1, "handle_gently": 1.2, "handle_fast": 1.8 if not hidden else -2.0, "discard": -.5}[action]
    elif family == "resource_constraint":
        v = {"measure_resource": -.1, "conserve": .8, "spend_now": 1.5 if hidden else -1.4, "switch_plan": .4}[action]
    else:
        v = {"plan_a": 1.6 if hidden else -1.8, "plan_b": 1.2 if hidden else -1.2, "plan_c": .9 if hidden else -.8, "verify_premise": -.15}[action]
    noise = rng.gauss(0, {"easy": .08, "medium": .18, "hard": .35}[difficulty])
    return round(v + noise, 5)

def make_task(seed, family, difficulty, rollouts_per_action=4):
    rng = random.Random(seed)
    key, hidden, public, actions = family_spec(family, difficulty, rng)
    task_id = f"spb-{family[:3]}-{difficulty[:1]}-{seed:06d}"
    prior = round(rng.uniform(.15, .85), 4)
    evidence_id = f"evidence-{task_id}"
    premise_id = f"premise-{task_id}-{key}"
    task = {
        "task_id": task_id, "seed": seed, "family": family,
        "difficulty": difficulty, "split": split_for(seed),
        "public_prompt": public, "actions": actions,
        "root_premises": [{"premise_id": premise_id, "name": key, "domain": "hidden", "prior": prior}],
        "evidence": [{"evidence_id": evidence_id, "content": public, "source_type": "environment", "visible": True}],
        "hidden_state": {"premise_id": premise_id, "value": hidden},
        "provenance_graph": {"nodes": [], "edges": []}, "rollout_lineage": [],
        "public_rollouts": [], "optimal_action": None,
    }
    scores = {}
    for hypothesis_value in (False, True):
      for action_idx, action in enumerate(actions):
        scores[(hypothesis_value, action)] = []
        for j in range(rollouts_per_action):
            sample_id = f"sample-{task_id}-h{int(hypothesis_value)}-a{action_idx}-j{j}"
            # Two generator sources emit descendants of the same premise. This
            # lets source grouping, premise grouping and learned recovery be
            # evaluated separately rather than collapsing into one oracle ID.
            source_id = f"source-{task_id}-branch-{j % 2}"
            semantic_group = f"{premise_id}-h{int(hypothesis_value)}-action-{action_idx}"
            val = action_return(family, hypothesis_value, action, rng, difficulty)
            text_templates = [
                f"Under the shared {key}={str(hypothesis_value).lower()} premise, action {action} is simulated to return {val:+.5f}.",
                f"Conditioned on {key}={str(hypothesis_value).lower()}, a rollout for {action} predicts value {val:+.5f}.",
            ]
            text = text_templates[j % len(text_templates)]
            rollout = {"sample_id": sample_id, "source_id": source_id,
                       "premise_ids": [premise_id], "evidence_ids": [evidence_id],
                       "action": action, "action_index": action_idx,
                       "hypothesis_value": hypothesis_value,
                       "draw_id": j, "sampler_type": "independent_condition",
                       "semantic_group": semantic_group, "value": val,
                       "text": text}
            task["rollout_lineage"].append(rollout)
            task["public_rollouts"].append({k: rollout[k] for k in ("sample_id","action","action_index","hypothesis_value","text")})
            task["provenance_graph"]["nodes"].extend([
                {"id": sample_id, "type": "rollout"},
                {"id": source_id, "type": "source"},
                {"id": premise_id, "type": "premise"},
                {"id": evidence_id, "type": "evidence"},
            ])
            task["provenance_graph"]["edges"].extend([
                {"parent": evidence_id, "child": premise_id, "relation": "supports"},
                {"parent": premise_id, "child": sample_id, "relation": "conditions", "hypothesis_value": hypothesis_value},
                {"parent": source_id, "child": sample_id, "relation": "generated"},
            ])
            scores[(hypothesis_value, action)].append(val)
    # Deduplicate graph nodes while retaining every edge/lineage row.
    task["provenance_graph"]["nodes"] = list({n["id"]: n for n in task["provenance_graph"]["nodes"]}.values())
    means = {a: sum(v) / len(v) for a, v in scores.items() if a[0] == bool(hidden)}
    task["optimal_action"] = max(means, key=lambda pair: means[pair])[1]
    return task

def write_benchmark(output, instances=1000, seed=20260928, rollouts_per_action=4):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for family in FAMILIES:
        for i in range(instances):
            difficulty = DIFFICULTIES[(i + stable_int(family)) % len(DIFFICULTIES)]
            rows.append(make_task(seed + len(rows) * 17, family, difficulty, rollouts_per_action))
    files = {split: (output / f"{split}.jsonl").open("w") for split in ("train", "dev", "test")}
    public_files = {split: (output / f"{split}_public.jsonl").open("w") for split in files}
    counts = Counter(); difficulty_counts = Counter(); family_counts = Counter()
    for task in rows:
        split = task["split"]; counts[split] += 1; difficulty_counts[(split, task["difficulty"])] += 1; family_counts[(split, task["family"])] += 1
        files[split].write(json.dumps(task, ensure_ascii=False) + "\n")
        public = {k: task[k] for k in ("task_id","family","difficulty","split","public_prompt","actions","evidence","public_rollouts")}
        public_files[split].write(json.dumps(public, ensure_ascii=False) + "\n")
    for f in [*files.values(), *public_files.values()]: f.close()
    stats = {"instances": len(rows), "family_counts": dict(Counter(t["family"] for t in rows)),
             "difficulty_counts": dict(Counter(t["difficulty"] for t in rows)), "split_counts": dict(counts),
             "family_by_split": {f"{s}/{fam}": n for (s, fam), n in family_counts.items()},
             "difficulty_by_split": {f"{s}/{d}": n for (s, d), n in difficulty_counts.items()},
             "rollouts": len(rows) * 2 * 4 * rollouts_per_action, "lineage_edges": sum(len(t["provenance_graph"]["edges"]) for t in rows),
             "generator_seed": seed}
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    # Lightweight SVG example, avoiding a plotting dependency for the benchmark.
    ex = rows[0]; nodes = ex["provenance_graph"]["nodes"][:8]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260">', '<style>text{font:13px sans-serif}.box{fill:#eef;stroke:#446}</style>']
    for i, node in enumerate(nodes):
        x = 20 + (i % 4) * 220; y = 25 + (i // 4) * 100
        svg += [f'<rect class="box" x="{x}" y="{y}" width="190" height="50"/>', f'<text x="{x+8}" y="{y+28}">{node["type"]}: {node["id"][-18:]}</text>']
    svg.append('</svg>'); (output / "lineage_example.svg").write_text("\n".join(svg))
    return stats

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", default="results_submission/benchmark"); p.add_argument("--instances-per-family", type=int, default=1000); p.add_argument("--seed", type=int, default=20260928); p.add_argument("--rollouts-per-action", type=int, default=4)
    a = p.parse_args(); print(json.dumps(write_benchmark(a.output, a.instances_per_family, a.seed, a.rollouts_per_action), indent=2))
