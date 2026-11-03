"""Legacy name-only priorities; no values, profiles, or diagnosis lookup."""
from typing import Tuple

def ablation_enabled(name):
    return False  # Runtime ablations are explicit configuration, never ambient env.

def _test_priority(name: str) -> Tuple[float, str]:
    if ablation_enabled("flat_priority"):
        return (2.0, "flat-priority ablation: all result-blinded tools have equal priority")

    lower = name.lower()
    if any(token in lower for token in ["vital", "temperature", "pressure", "heart rate"]):
        return (1.2, "vitals are useful context but usually low specificity")
    high_yield = {
        "biopsy": (5.0, "tissue/pathology can separate mimics"),
        "microscopic": (5.0, "microscopy can provide discriminative morphology"),
        "antibod": (4.6, "serology can confirm immune/infectious hypotheses"),
        "pcr": (4.6, "molecular test can confirm infection/genetic causes"),
        "culture": (4.2, "culture can separate infectious from noninfectious causes"),
        "electromyography": (4.2, "physiologic test can separate neuromuscular causes"),
        "repetitive": (4.2, "physiologic test can separate neuromuscular causes"),
        "mri": (4.1, "cross-sectional imaging can localize structural disease"),
        "ct ": (4.0, "cross-sectional imaging can localize structural disease"),
        " ct": (4.0, "cross-sectional imaging can localize structural disease"),
        "ct scan": (4.0, "cross-sectional imaging can localize structural disease"),
        "ultrasound": (3.8, "imaging can narrow anatomic causes"),
        "x-ray": (3.2, "initial imaging can rule out gross structural pathology"),
        "x ray": (3.2, "initial imaging can rule out gross structural pathology"),
        "urinalysis": (3.4, "urine findings can separate renal/infectious causes"),
        "stool": (3.4, "stool testing can separate infectious gastrointestinal causes"),
        "blood": (2.6, "blood testing gives broad but less specific screening evidence"),
    }
    for token, result in high_yield.items():
        if token in lower:
            return result
    if "examination" in lower:
        return (2.4, "physical finding can refine the differential")
    return (2.0, "available structured finding may add discriminative evidence")

def _menu_priority(name: str) -> Tuple[int, str]:
    lower = name.lower()
    high_value = [
        "biopsy", "microscopic", "antibod", "torch", "rubella", "barium",
        "mri", "ct scan", "x-ray", "x ray", "cytospin", "hansel", "ldh",
        "lactate", "electromyography", "repetitive", "culture", "stool",
        "urinalysis", "mammography",
    ]
    low_value = ["vital signs", "temperature", "blood pressure", "heart rate", "respiratory rate"]
    if any(token in lower for token in high_value):
        return (0, lower)
    if any(token in lower for token in low_value):
        return (3, lower)
    if "examination" in lower:
        return (1, lower)
    return (2, lower)

def plan(actions, requested, limit=12, flat=False):
    rows=[a for a in actions if a["action_id"] not in set(requested)]
    rows.sort(key=lambda a: (-(2.0 if flat else _test_priority(a["name"])[0]), _menu_priority(a["name"])))
    lines = []
    for a in rows[:limit]:
        score, reason = (2.0, 'flat priority') if flat else _test_priority(a['name'])
        lines.append(f"- {a['action_id']} | {a['name']} | priority={score:.1f} | rationale={reason}")
    return '\n'.join(lines) or 'No unrequested structured tests remain.'
