from dataclasses import dataclass
from typing import Dict, List
from .presets import STYLE_PRESETS

@dataclass
class AgentStep:
    name: str
    output: Dict

def planner(goal: str) -> Dict:
    g = (goal or "").strip()
    if not g:
        g = "A futuristic city skyline at sunset, rain reflections, neon lights"

    style = "Cinematic"
    gl = g.lower()
    if "anime" in gl:
        style = "Anime"
    if "portrait" in gl or "photo" in gl:
        style = "Photoreal"
    if "fantasy" in gl or "dragon" in gl:
        style = "Fantasy Art"

    return {"goal": g, "style": style}

def prompt_engineer(plan: Dict) -> Dict:
    preset = STYLE_PRESETS[plan["style"]]
    prompt = f"{plan['goal']}, {preset['suffix']}"
    negative_prompt = f"{preset['negative']}, low quality, worst quality, jpeg artifacts, watermark, text, logo"
    return {"prompt": prompt, "negative_prompt": negative_prompt}

def critic(prompt: str, negative_prompt: str) -> Dict:
    tips = []
    if "lighting" not in prompt.lower():
        tips.append("Add lighting: cinematic lighting / soft rim light / volumetric light.")
    if "composition" not in prompt.lower():
        tips.append("Add composition: centered composition / rule of thirds / wide angle.")
    if "highly detailed" not in prompt.lower():
        tips.append("Add detail: highly detailed, sharp focus, texture-rich.")
    if "watermark" not in negative_prompt.lower():
        tips.append("Add watermark/text/logo to negative prompt.")

    return {
        "critique": " ".join(tips) if tips else "Looks strong. No major issues.",
        "recommendations": tips
    }

def refiner(prompt: str, negative_prompt: str, critique: Dict) -> Dict:
    refined_prompt = prompt
    if critique["recommendations"]:
        refined_prompt += ", " + ", ".join([
            "cinematic lighting",
            "volumetric light",
            "rule of thirds",
            "sharp focus",
            "highly detailed"
        ])
    refined_negative = negative_prompt + ", deformed, bad anatomy, blurry, oversaturated"
    return {"final_prompt": refined_prompt, "final_negative_prompt": refined_negative}

def run_agent_loop(goal: str) -> Dict:
    steps: List[AgentStep] = []

    plan = planner(goal)
    steps.append(AgentStep("Planner", plan))

    engineered = prompt_engineer(plan)
    steps.append(AgentStep("Prompt Engineer", engineered))

    crit = critic(engineered["prompt"], engineered["negative_prompt"])
    steps.append(AgentStep("Critic", crit))

    final = refiner(engineered["prompt"], engineered["negative_prompt"], crit)
    steps.append(AgentStep("Refiner", final))

    return {
        "goal": plan["goal"],
        "style": plan["style"],
        "prompt": engineered["prompt"],
        "negative_prompt": engineered["negative_prompt"],
        "critique": crit,
        "final_prompt": final["final_prompt"],
        "final_negative_prompt": final["final_negative_prompt"],
        "steps": [{"name": s.name, "output": s.output} for s in steps]
    }
