"""
Snakebite Emergency Chatbot Module
Rule-based intent classification with structured medical response templates.
"""

import re
from typing import Optional
from pydantic import BaseModel


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # reserved for future session tracking


class ChatResponse(BaseModel):
    intent: str
    confidence: str          # "high" | "medium" | "low"
    response_type: str
    title: str
    content: dict
    disclaimer: str


# ── Medical knowledge base (controlled, safe templates) ──────────────────────

MEDICAL_KB = {

    "first_aid": {
        "title": "🚨 Snakebite First Aid Instructions",
        "dos": [
            "Stay calm and keep the victim as still as possible",
            "Remove jewellery, watches, and tight clothing near the bite",
            "Immobilise the bitten limb with a splint or sling",
            "Keep the bitten area at or below heart level",
            "Transport the victim to the nearest hospital immediately",
            "Note the time of bite and try to remember the snake's appearance",
            "Call emergency services (112 / local emergency number) right away",
        ],
        "donts": [
            "Do NOT cut the wound or attempt to suck out venom",
            "Do NOT apply a tourniquet or tight bandage",
            "Do NOT apply ice, heat, or any chemicals to the wound",
            "Do NOT give alcohol, aspirin, or NSAIDs",
            "Do NOT allow the victim to walk if avoidable",
            "Do NOT waste time trying to catch or kill the snake",
            "Do NOT apply electric shock therapy",
        ],
        "emergency_warning_signs": [
            "Difficulty breathing or swallowing",
            "Severe swelling spreading rapidly",
            "Blurred vision or drooping eyelids",
            "Muscle weakness or paralysis",
            "Uncontrolled bleeding from the wound or elsewhere",
            "Loss of consciousness or seizures",
            "Severe chest or abdominal pain",
        ],
    },

    "neurotoxic_symptoms": {
        "title": "🧠 Neurotoxic Venom — Symptoms & Signs",
        "venom_type": "Neurotoxic",
        "snake_examples": ["Cobra", "Krait", "Mamba", "Coral Snake", "Sea Snake"],
        "early_symptoms": [
            "Drooping eyelids (ptosis) within 1–4 hours",
            "Double or blurred vision",
            "Difficulty speaking or swallowing (dysarthria / dysphagia)",
            "Excessive drooling or dry mouth",
            "Weakness of facial muscles",
        ],
        "progressive_symptoms": [
            "Descending muscle paralysis from head to limbs",
            "Respiratory muscle weakness → breathing difficulty",
            "Generalised body weakness",
            "Nausea and vomiting",
        ],
        "critical_signs": [
            "Respiratory failure (requires immediate ventilator support)",
            "Complete flaccid paralysis",
            "Cardiovascular collapse",
        ],
        "treatment_note": "Anti-venom is the definitive treatment. Seek hospital care immediately.",
    },

    "hemotoxic_symptoms": {
        "title": "🩸 Hemotoxic / Cytotoxic Venom — Symptoms & Signs",
        "venom_type": "Hemotoxic / Cytotoxic",
        "snake_examples": ["Viper", "Pit Viper", "Rattlesnake", "Russell's Viper", "Saw-scaled Viper"],
        "early_symptoms": [
            "Severe local pain and swelling at bite site",
            "Blistering and bruising around the wound",
            "Oozing blood from the bite wound",
            "Nausea, vomiting, and abdominal pain",
        ],
        "progressive_symptoms": [
            "Widespread bruising (ecchymosis)",
            "Bleeding from gums, nose, or IV sites",
            "Urine turning dark brown / blood-tinged",
            "Severe tissue damage (necrosis) around the bite",
        ],
        "critical_signs": [
            "Kidney failure (oliguria / anuria)",
            "Uncontrolled haemorrhage",
            "Disseminated Intravascular Coagulation (DIC)",
            "Cardiovascular shock",
        ],
        "treatment_note": "Polyvalent anti-venom + supportive care. Surgical debridement may be needed.",
    },

    "cytotoxic_symptoms": {
        "title": "⚠️ Cytotoxic Venom — Local Tissue Destruction",
        "venom_type": "Cytotoxic",
        "snake_examples": ["Puff Adder", "Gaboon Viper", "Some Cobras (spitting)"],
        "early_symptoms": [
            "Intense burning pain at the bite site",
            "Rapid swelling and redness",
            "Formation of fluid-filled blisters",
            "Skin discolouration (dark purple / black)",
        ],
        "progressive_symptoms": [
            "Deep tissue necrosis (tissue death)",
            "Compartment syndrome",
            "Secondary infection risk",
        ],
        "critical_signs": [
            "Limb-threatening necrosis requiring amputation",
            "Systemic infection / sepsis",
        ],
        "treatment_note": "Anti-venom + wound care + possible surgical intervention.",
    },

    "dangerous_snake": {
        "title": "🐍 Is This Snake Dangerous?",
        "assessment_guide": {
            "HIGH_DANGER — Seek immediate hospital care": [
                "Triangular / arrow-shaped head",
                "Visible fangs (front-fanged or rear-fanged)",
                "Hood that spreads when threatened (cobra)",
                "Rattle on the tail (rattlesnake)",
                "Bright warning colours (coral snake pattern: red-yellow-black)",
                "Very thick body relative to length",
                "Slit / elliptical pupils",
            ],
            "LOWER_RISK — but still seek medical advice": [
                "Rounded head with no visible fangs",
                "Slender, smooth-scaled body",
                "Round pupils",
                "No distinctive markings",
            ],
        },
        "important_note": (
            "⚠️ IMPORTANT: Snake identification from appearance alone is unreliable. "
            "ALL snakebites should be treated as potentially dangerous until medically assessed. "
            "Do NOT attempt to handle the snake."
        ),
    },

    "anti_venom": {
        "title": "💉 Anti-Venom Information",
        "key_facts": [
            "Anti-venom is the ONLY proven treatment for severe envenomation",
            "It must be administered in a hospital setting with resuscitation facilities",
            "Polyvalent anti-venom covers multiple species; monovalent is species-specific",
            "Best administered within 4–6 hours of the bite, but still effective later",
            "Allergic reactions (anaphylaxis) are possible — hospital monitoring is essential",
        ],
        "where_available": [
            "Government and district hospitals (usually stocked)",
            "Snake rescue centres / herpetology institutes",
            "Toxicology departments of medical colleges",
            "Poison control centres can guide you to nearest stock",
        ],
        "emergency_contacts": {
            "India": "National Poison Information Centre: 1800-116-117",
            "USA": "Poison Control: 1-800-222-1222",
            "UK": "NHS 111",
            "Australia": "Poisons Information: 13 11 26",
            "Global": "Call local emergency services (112 / 911 / 999)",
        },
    },

    "general_info": {
        "title": "ℹ️ Snakebite — General Information",
        "global_burden": [
            "~2.7 million envenomations occur worldwide each year (WHO)",
            "~81,000–138,000 deaths annually; 400,000+ survivors left with disabilities",
            "Snakebite is a neglected tropical disease classified by WHO",
            "Most deaths occur in South Asia, Sub-Saharan Africa, and Latin America",
        ],
        "prevention_tips": [
            "Wear boots and long trousers in snake-prone areas",
            "Use a torch when walking at night",
            "Do not put hands in rock crevices or under logs without checking",
            "Keep grass short around homes",
            "Do not handle or provoke snakes",
            "Store food securely to avoid attracting rodents (snake prey)",
        ],
    },
}

DISCLAIMER = (
    "⚠️ MEDICAL DISCLAIMER: This information is for emergency guidance only and does NOT "
    "replace professional medical advice. Always call emergency services and get to a hospital "
    "immediately after any snakebite."
)

# ── Intent patterns ───────────────────────────────────────────────────────────

INTENT_PATTERNS = [
    {
        "intent": "first_aid",
        "patterns": [
            r"\b(first aid|what (should|do) i do|help|bitten|bite|emergency|treat|treatment|steps)\b",
            r"\b(after (a |the )?snakebite|snakebite (help|treatment|steps))\b",
        ],
        "confidence": "high",
    },
    {
        "intent": "neurotoxic_symptoms",
        "patterns": [
            r"\bneurotox(ic|in)\b",
            r"\b(cobra|krait|mamba|coral snake|sea snake)\b",
            r"\b(paralys(is|ed)|drooping (eyelid|eye)|blurred vision|can.t breathe|breathing difficulty)\b",
            r"\bnerve (damage|venom|poison)\b",
        ],
        "confidence": "high",
    },
    {
        "intent": "hemotoxic_symptoms",
        "patterns": [
            r"\bhemotox(ic|in)\b",
            r"\b(viper|pit viper|rattlesnake|russell.s viper)\b",
            r"\b(bleed(ing)?|blood|bruising|clot(ting)?|haemorrhage|hemorrhage)\b",
            r"\bblood (venom|poison)\b",
        ],
        "confidence": "high",
    },
    {
        "intent": "cytotoxic_symptoms",
        "patterns": [
            r"\bcytotox(ic|in)\b",
            r"\bnecrosis\b",
            r"\b(tissue (damage|death|destruction)|flesh (eating|rot))\b",
            r"\b(puff adder|gaboon)\b",
        ],
        "confidence": "high",
    },
    {
        "intent": "dangerous_snake",
        "patterns": [
            r"\b(dangerous|venomous|poisonous|deadly|harmful|safe)\b",
            r"\bis (this|the) snake\b",
            r"\bhow (dangerous|deadly|harmful|venomous|poisonous)\b",
            r"\b(identify|identification|what (kind|type|species) of snake)\b",
        ],
        "confidence": "high",
    },
    {
        "intent": "anti_venom",
        "patterns": [
            r"\banti.?venom\b",
            r"\bantidote\b",
            r"\b(where (to get|find|buy)|how to get).*(medicine|treatment|anti)\b",
            r"\bserotherapy\b",
        ],
        "confidence": "high",
    },
    {
        "intent": "general_info",
        "patterns": [
            r"\b(what is|tell me about|information|info|general|overview)\b.*snake",
            r"\bsnakebite (statistics|facts|data|burden|deaths)\b",
            r"\b(prevent|prevention|avoid) (snakebite|snake)\b",
        ],
        "confidence": "medium",
    },
    {
        "intent": "symptoms",  # generic symptoms fallback
        "patterns": [
            r"\bsymptom(s)?\b",
            r"\bsign(s)? of\b",
            r"\beffect(s)? of (venom|bite)\b",
        ],
        "confidence": "medium",
    },
]


# ── Intent classifier ─────────────────────────────────────────────────────────

def classify_intent(message: str) -> tuple[str, str]:
    """Return (intent, confidence) by matching message against patterns."""
    text = message.lower().strip()

    for rule in INTENT_PATTERNS:
        for pattern in rule["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                return rule["intent"], rule["confidence"]

    return "first_aid", "low"   # safe default


def build_response(intent: str, confidence: str) -> ChatResponse:
    """Build a structured ChatResponse from the classified intent."""

    # Generic symptoms query → guide user to specific types
    if intent == "symptoms":
        return ChatResponse(
            intent=intent,
            confidence=confidence,
            response_type="symptom_guide",
            title="🐍 Snakebite Symptoms — Choose Venom Type",
            content={
                "guidance": "Symptoms vary by venom type. Please ask about:",
                "venom_types": [
                    {"type": "Neurotoxic", "example_query": "What are symptoms of neurotoxic venom?",
                     "snakes": "Cobra, Krait, Mamba"},
                    {"type": "Hemotoxic / Cytotoxic", "example_query": "What are symptoms of hemotoxic venom?",
                     "snakes": "Viper, Rattlesnake, Russell's Viper"},
                    {"type": "Cytotoxic (local)", "example_query": "What are symptoms of cytotoxic venom?",
                     "snakes": "Puff Adder, Gaboon Viper"},
                ],
                "immediate_action": "If bitten — go to the hospital NOW regardless of symptoms.",
            },
            disclaimer=DISCLAIMER,
        )

    kb_entry = MEDICAL_KB.get(intent, MEDICAL_KB["first_aid"])
    response_type_map = {
        "first_aid": "first_aid_guide",
        "neurotoxic_symptoms": "symptom_info",
        "hemotoxic_symptoms": "symptom_info",
        "cytotoxic_symptoms": "symptom_info",
        "dangerous_snake": "safety_assessment",
        "anti_venom": "treatment_info",
        "general_info": "general_information",
    }

    return ChatResponse(
        intent=intent,
        confidence=confidence,
        response_type=response_type_map.get(intent, "general_information"),
        title=kb_entry["title"],
        content=kb_entry,
        disclaimer=DISCLAIMER,
    )


# ── Public interface ──────────────────────────────────────────────────────────

def process_chat(request: ChatRequest) -> ChatResponse:
    """Main entry point called by the FastAPI route."""
    intent, confidence = classify_intent(request.message)
    return build_response(intent, confidence)
