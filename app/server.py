"""
FastAPI web server for microstructure image retrieval and identification.

Endpoints:
    POST /api/identify        - Upload image, get phase identification + diagnostic report
    POST /api/search          - Upload image, get similar microstructures
    POST /api/search-text     - Text query (CLIP only), get matching microstructures
    GET  /api/image/{index}   - Serve a micrograph thumbnail by index
    GET  /api/image/{idx}/full - Serve full-resolution image
    GET  /api/health          - Health check
    GET  /api/stats           - Index statistics
    GET  /api/phases          - Phase knowledge base
    GET  /api/phase-guide     - Phase encyclopedia with example counts
    GET  /                    - Web UI

Usage:
    uvicorn app.server:app --reload --port 8000
"""
import io
import sys
import json
import base64
import tempfile
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# LABEL NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
# Maps raw dataset labels (from Kaggle folders, UHCS metadata, etc.) to
# canonical phase keys used in PHASE_INFO. This is the critical bridge
# between messy real-world data and the educational layer.
#
# Rules:
#   - Exact matches first, then substring/fuzzy matching
#   - Compound labels like "pearlite+widmanstatten" get mapped to dominant phase
#   - Morphological descriptors (e.g. "grain_structure") map to closest phase
#   - "unknown" stays "unknown"

LABEL_NORMALIZATION = {
    # ── Direct phase names (with common variations) ──
    "pearlite": "pearlite",
    "perlite": "pearlite",
    "pearlitic": "pearlite",

    "bainite": "bainite",
    "bainitic": "bainite",
    "bainite_ma": "bainite_ma",
    "bainite ma": "bainite_ma",
    "bainite with ma": "bainite_ma",
    "bainite with m/a": "bainite_ma",
    "bainite with m/a islands": "bainite_ma",

    "martensite": "martensite",
    "martensitic": "martensite",
    "tempered martensite": "tempered_martensite",
    "tempered_martensite": "tempered_martensite",
    "lath martensite": "martensite",

    "ferrite": "ferrite",
    "ferritic": "ferrite",
    "proeutectoid ferrite": "ferrite",
    "polygonal ferrite": "ferrite",
    "acicular ferrite": "acicular_ferrite",
    "acicular_ferrite": "acicular_ferrite",

    "austenite": "austenite",
    "austenitic": "austenite",
    "retained austenite": "austenite",
    "retained_austenite": "austenite",

    "cementite": "cementite",
    "fe3c": "cementite",
    "iron carbide": "cementite",

    "spheroidite": "spheroidite",
    "spheroidized": "spheroidite",
    "spheroidised": "spheroidite",

    "widmanstatten": "widmanstatten",
    "widmanstätten": "widmanstatten",
    "widmanstatten ferrite": "widmanstatten",

    # ── Compound / mixed labels ──
    "pearlite+widmanstatten": "pearlite_widmanstatten",
    "pearlite + widmanstatten": "pearlite_widmanstatten",
    "pearlite+spheroidite": "pearlite_spheroidite",
    "pearlite + spheroidite": "pearlite_spheroidite",
    "ferrite+pearlite": "ferrite_pearlite",
    "ferrite + pearlite": "ferrite_pearlite",
    "ferrite-pearlite": "ferrite_pearlite",

    # ── UHCS-specific labels ──
    "network": "network_cementite",
    "network cementite": "network_cementite",
    "proeutectoid": "network_cementite",

    # ── Morphological descriptors (Kaggle) ──
    "grain structure": "ferrite",  # Equiaxed grain images → most likely ferrite
    "grain_structure": "ferrite",
    "grain boundaries": "ferrite",
    "equiaxed": "ferrite",

    # ── Duplex / multi-phase ──
    "bi-modal/duplex": "duplex",
    "bi-modal": "duplex",
    "bimodal": "duplex",
    "duplex": "duplex",
    "dual phase": "duplex",
    "dual_phase": "duplex",
    "dp steel": "duplex",

    # ── Other / catch-all ──
    "unknown": "unknown",
    "unlabeled": "unknown",
    "other": "unknown",
}


def _normalize_label(raw_label: str) -> str:
    """
    Normalize a raw dataset label to a canonical phase key.

    Tries exact match first, then lowercased match, then substring matching.
    Returns 'unknown' if no mapping found.
    """
    if not raw_label or raw_label.strip() == "":
        return "unknown"

    cleaned = raw_label.strip().lower()

    # Exact match
    if cleaned in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[cleaned]

    # Try with underscores replaced by spaces and vice versa
    alt = cleaned.replace("_", " ")
    if alt in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[alt]
    alt = cleaned.replace(" ", "_")
    if alt in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[alt]

    # Substring matching: check if any known phase name is IN the label
    phase_keywords = [
        ("pearlite", "pearlite"),
        ("bainite", "bainite"),
        ("martensite", "martensite"),
        ("ferrite", "ferrite"),
        ("austenite", "austenite"),
        ("cementite", "cementite"),
        ("spheroidite", "spheroidite"),
        ("spheroidize", "spheroidite"),
        ("widmanstatten", "widmanstatten"),
        ("widmanstätten", "widmanstatten"),
        ("duplex", "duplex"),
        ("dual", "duplex"),
    ]
    for keyword, canonical in phase_keywords:
        if keyword in cleaned:
            return canonical

    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
# Comprehensive metallurgical reference for each phase. This is the
# educational backbone of the application — it should stand on its own
# as a useful reference regardless of what's in the database.

PHASE_INFO = {
    "pearlite": {
        "name": "Pearlite",
        "category": "Eutectoid Constituent",
        "summary": "The classic 'fingerprint' of steel — alternating layers of soft ferrite and hard cementite.",
        "description": (
            "Pearlite is a lamellar mixture of ferrite (α-Fe) and cementite (Fe₃C) that forms "
            "during the eutectoid decomposition of austenite at ~727°C. The alternating plates of "
            "soft ferrite and hard cementite create a characteristic fingerprint-like pattern. "
            "Pearlite derives its name from its pearl-like luster under the optical microscope. "
            "It is one of the most common microstructural constituents in plain carbon steels."
        ),
        "key_features": [
            "Alternating light/dark lamellar plates (ferrite + cementite)",
            "Fingerprint or wood-grain pattern at medium–high magnification",
            "Colony structure — lamellae are parallel within each colony but change orientation between colonies",
            "Interlamellar spacing decreases with faster cooling (fine pearlite is harder)",
            "Dark-etching nodules at low magnification"
        ],
        "optical_appearance": (
            "At low magnification, pearlite colonies appear as dark patches or nodules against a lighter "
            "ferrite background. At higher magnification (500×+), the alternating lamellae of ferrite "
            "(light) and cementite (dark) become resolvable. Fine pearlite may appear uniformly dark "
            "even at moderate magnification because the lamellae are below the optical resolution limit."
        ),
        "sem_appearance": (
            "SEM reveals sharp contrast between ferrite (dark, lower atomic number) and cementite "
            "(bright, higher average atomic number) lamellae. 3D topographic relief is visible due to "
            "differential etching. Lamellar spacing can be directly measured, typically ranging from "
            "0.1 μm (fine pearlite) to 1+ μm (coarse pearlite)."
        ),
        "confused_with": [
            "Bainite — both show aligned internal features, but bainite is acicular (needle-like) rather than lamellar",
            "Tempered martensite — at low magnification, tempered martensite can look uniformly dark like pearlite"
        ],
        "typical_hardness": "200–350 HV",
        "formation": "Slow cooling through the eutectoid temperature (~727°C). Faster cooling produces finer pearlite.",
        "engineering_significance": (
            "Pearlite provides a good balance of strength and ductility. Fine pearlite (rapid cooling) "
            "is harder and stronger than coarse pearlite (slow cooling). Used extensively in rail steels, "
            "wire ropes, and spring steels."
        ),
        "composition_range": "Near-eutectoid steels (~0.76 wt% C), but also present in hypo- and hyper-eutectoid steels.",
        "related_phases": ["ferrite", "cementite", "spheroidite"],
    },

    "bainite": {
        "name": "Bainite",
        "category": "Intermediate Transformation Product",
        "summary": "Needle-like ferrite with fine carbides — harder than pearlite, tougher than martensite.",
        "description": (
            "Bainite is an acicular (needle-like) microstructure consisting of ferrite laths or plates "
            "with fine dispersed carbide particles. It forms at intermediate cooling rates — faster than "
            "pearlite but slower than martensite — typically between 250–550°C. Named after Edgar Bain, "
            "bainite has two main variants: upper bainite (coarser, forms 400–550°C) and lower bainite "
            "(finer, forms 250–400°C). Lower bainite is often preferred because it combines high strength "
            "with good toughness."
        ),
        "key_features": [
            "Acicular or lath-like ferrite plates growing in sheaf-like colonies",
            "Fine carbide particles dispersed within or between laths",
            "No lamellar pattern — distinguishes it from pearlite",
            "Upper bainite: carbides between ferrite laths (lower toughness)",
            "Lower bainite: carbides within ferrite laths at ~60° angle (better toughness)"
        ],
        "optical_appearance": (
            "Dark etching acicular features with a feathery or sheaf-like morphology. Upper bainite shows "
            "coarser laths that may be individually resolvable. Lower bainite appears finer and more densely "
            "packed, sometimes resembling tempered martensite. At low magnification, bainite regions appear "
            "as dark, textured patches."
        ),
        "sem_appearance": (
            "Lath/plate morphology is clearly resolved in SEM. Carbide particles are visible at lath "
            "boundaries (upper bainite) or within laths at characteristic angles (lower bainite). "
            "The sheaf structure — multiple parallel laths sharing a common growth direction — is distinctive."
        ),
        "confused_with": [
            "Pearlite — similar dark etching at low magnification, but pearlite has lamellae not needles",
            "Martensite — both are acicular, but martensite has no visible carbides in as-quenched condition"
        ],
        "typical_hardness": "300–450 HV",
        "formation": "Intermediate cooling rates (250–550°C) or isothermal holding in the bainite temperature range.",
        "engineering_significance": (
            "Bainitic steels offer excellent combinations of strength and toughness. Lower bainite is "
            "particularly valued in structural applications, pipeline steels, and armor plates. "
            "Austempering heat treatment specifically produces bainite."
        ),
        "composition_range": "Low to medium carbon steels (0.1–0.5 wt% C), often with Cr, Mo, Mn additions.",
        "related_phases": ["bainite_ma", "martensite", "pearlite"],
    },

    "bainite_ma": {
        "name": "Bainite with M/A Islands",
        "category": "Multi-Phase Constituent",
        "summary": "Bainitic ferrite with bright martensite-austenite islands — common in modern pipeline steels.",
        "description": (
            "A bainitic ferrite matrix containing martensite-austenite (M/A) constituent islands. "
            "During cooling, some austenite regions become carbon-enriched and stabilized, remaining "
            "untransformed as the surrounding matrix transforms to bainitic ferrite. Upon final cooling "
            "to room temperature, these austenite islands partially transform to martensite, creating "
            "the characteristic M/A islands. The fraction, size, and distribution of M/A islands "
            "significantly affect mechanical properties."
        ),
        "key_features": [
            "Bainitic ferrite lath matrix (dark background after etching)",
            "Bright M/A islands — retained austenite + fresh martensite",
            "Islands appear as blocky or elongated bright features",
            "M/A fraction typically 2–15%, varying with composition and cooling rate",
            "Requires specific etching (e.g., LePera or Klemm) to clearly reveal M/A"
        ],
        "optical_appearance": (
            "Acicular ferrite matrix with small bright islands revealed by color etching. "
            "LePera's reagent colors M/A white/light against a brown/tan bainitic ferrite matrix. "
            "Without specialized etching, M/A islands may be difficult to distinguish."
        ),
        "sem_appearance": (
            "Lath ferrite matrix with topographically distinct M/A islands. In secondary electron "
            "imaging, islands may appear raised due to differential etching. In backscatter mode, "
            "M/A islands show slightly different contrast due to composition/structure differences."
        ),
        "confused_with": [
            "Plain bainite — without proper etching, M/A islands are easily missed",
            "Acicular ferrite — similar matrix morphology but different nucleation mechanism"
        ],
        "typical_hardness": "280–400 HV",
        "formation": "Controlled cooling of low-carbon microalloyed steels. M/A forms when carbon-enriched austenite is stabilized.",
        "engineering_significance": (
            "Critical microstructure in modern pipeline steels (X70, X80, X100) and HSLA steels. "
            "M/A islands can be beneficial (strengthening) or detrimental (embrittlement in HAZ of welds) "
            "depending on their size, distribution, and carbon content."
        ),
        "composition_range": "Low-carbon (0.03–0.10 wt% C) microalloyed steels with Nb, V, Ti, Mo.",
        "related_phases": ["bainite", "martensite", "austenite"],
    },

    "martensite": {
        "name": "Martensite",
        "category": "Diffusionless Transformation Product",
        "summary": "The hardest steel phase — formed by rapid quenching, with a characteristic needle/lath structure.",
        "description": (
            "Martensite is a supersaturated solid solution of carbon in body-centered tetragonal (BCT) "
            "iron, formed by rapid quenching of austenite. Unlike pearlite and bainite, martensite forms "
            "by a diffusionless, shear transformation — the atoms move cooperatively without diffusion, "
            "trapping carbon in the lattice. This trapped carbon creates extreme lattice strain, making "
            "martensite the hardest and most brittle of the common steel microstructures. "
            "Almost all hardened steel tools rely on martensitic transformation."
        ),
        "key_features": [
            "Lath morphology in low-carbon steel (<0.6% C): parallel laths organized into packets and blocks",
            "Plate or lens morphology in high-carbon steel (>0.6% C): individual needles with midribs",
            "No visible carbides in the as-quenched condition",
            "Very high dislocation density (~10¹⁵ m⁻²)",
            "Dark, featureless etching at low magnification"
        ],
        "optical_appearance": (
            "As-quenched martensite etches dark and may appear as a featureless dark mass at low magnification. "
            "At higher magnification, lath structure (low C) or plate/needle structure (high C) becomes visible. "
            "Nital etching reveals lath boundaries. The prior austenite grain boundaries are often visible as "
            "the laths are organized within them."
        ),
        "sem_appearance": (
            "Fine lath structure is clearly visible in SEM. Individual laths are typically 0.2–2 μm wide, "
            "arranged in parallel packets. High-carbon plate martensite shows individual plates with smooth "
            "habit planes. No carbides are visible in the as-quenched condition (unlike bainite)."
        ),
        "confused_with": [
            "Lower bainite — similar lath structure, but bainite contains visible carbides within laths",
            "Heavily deformed ferrite — can appear dark and featureless, but lacks the internal lath structure"
        ],
        "typical_hardness": "400–700+ HV (increases with carbon content)",
        "formation": "Rapid quenching (water, oil, or air depending on alloy) from the austenite phase field.",
        "engineering_significance": (
            "The basis of most steel hardening. As-quenched martensite is too brittle for most applications, "
            "so it is usually tempered (reheated to 150–650°C) to reduce brittleness while retaining most "
            "of the hardness. Critical in cutting tools, bearings, springs, and automotive components."
        ),
        "composition_range": "All carbon and alloy steels. Hardenability depends on carbon content and alloying.",
        "related_phases": ["tempered_martensite", "bainite", "austenite"],
    },

    "tempered_martensite": {
        "name": "Tempered Martensite",
        "category": "Heat-Treated Phase",
        "summary": "Martensite reheated to reduce brittleness — fine carbides precipitate within the lath structure.",
        "description": (
            "Tempered martensite forms when as-quenched martensite is reheated (tempered) to allow "
            "carbon to diffuse out of the supersaturated BCT lattice and form fine carbide precipitates. "
            "This relieves internal stresses and significantly improves toughness while retaining much "
            "of the hardness. The degree of tempering (temperature and time) controls the trade-off "
            "between hardness and toughness."
        ),
        "key_features": [
            "Retains the lath/packet structure of as-quenched martensite",
            "Fine carbide particles precipitated along lath boundaries and within laths",
            "Dark etching but with more internal texture than as-quenched martensite",
            "Carbide size increases with tempering temperature/time",
            "Prior austenite grain boundaries often clearly visible"
        ],
        "optical_appearance": (
            "Dark etching overall, but with a finer, more resolved texture than as-quenched martensite. "
            "At high magnification, the lath boundaries and fine carbide particles become visible. "
            "Heavily tempered martensite can appear lighter and begin to resemble bainite."
        ),
        "sem_appearance": (
            "Lath boundaries are clearer than in as-quenched martensite. Fine carbide particles "
            "(typically cementite or transition carbides) are visible within and between laths. "
            "Higher tempering temperatures produce coarser, more easily resolved carbides."
        ),
        "confused_with": [
            "As-quenched martensite — tempered version shows fine carbides and slightly lighter etching",
            "Lower bainite — very similar appearance; both have carbides within ferrite laths"
        ],
        "typical_hardness": "300–600 HV (depends on tempering temperature)",
        "formation": "Quench from austenite to form martensite, then reheat to 150–650°C for tempering.",
        "engineering_significance": (
            "The most common microstructure in hardened and tempered steel components. Used in virtually "
            "all structural and tool applications where high strength with reasonable toughness is needed."
        ),
        "composition_range": "All hardenable steels. Typically 0.2–1.0 wt% C with various alloy additions.",
        "related_phases": ["martensite", "bainite", "spheroidite"],
    },

    "ferrite": {
        "name": "Ferrite (α-Iron)",
        "category": "Equilibrium Phase",
        "summary": "The softest steel phase — clean, equiaxed grains that are the building block of most steels.",
        "description": (
            "Ferrite is body-centered cubic (BCC) iron with very low carbon solubility (max ~0.02% at 727°C). "
            "It is the softest and most ductile phase in steel. In hypoeutectoid steels (< 0.76% C), "
            "proeutectoid ferrite forms first during cooling, appearing as equiaxed polygonal grains "
            "that nucleate along prior austenite grain boundaries. The remaining austenite then transforms "
            "to pearlite at the eutectoid temperature. Ferrite is the dominant phase in low-carbon and "
            "mild steels."
        ),
        "key_features": [
            "Equiaxed polygonal grains with clean, featureless interiors",
            "Clearly visible grain boundaries after nital or picral etching",
            "Light/bright etching — the lightest phase in most etched steel micrographs",
            "May appear as a proeutectoid network along prior austenite grain boundaries",
            "Grain size is a key strengthening mechanism (Hall-Petch relationship)"
        ],
        "optical_appearance": (
            "Bright, clean equiaxed grains with dark grain boundaries revealed by etching. Grain interiors "
            "are featureless. In hypoeutectoid steels, ferrite appears as bright regions surrounding darker "
            "pearlite colonies. Grain size ranges from ~5 μm (fine-grained HSLA steels) to 100+ μm "
            "(coarse-grained, slowly cooled)."
        ),
        "sem_appearance": (
            "Smooth, featureless grain surfaces. Grain boundaries appear as grooves (preferential etching) "
            "or ridges depending on etchant and conditions. Very little internal contrast or topography "
            "within grains."
        ),
        "confused_with": [
            "Austenite — also equiaxed grains, but austenite is FCC and shows annealing twins",
            "Recrystallized grains in other phases — equiaxed morphology can look similar"
        ],
        "typical_hardness": "80–150 HV",
        "formation": "Stable below 912°C in pure iron. Forms during slow cooling from the austenite phase field.",
        "engineering_significance": (
            "Ferrite provides ductility and formability. Grain refinement of ferrite (via controlled rolling "
            "or microalloying with Nb, V, Ti) is the primary strengthening mechanism in HSLA steels, "
            "providing strength without sacrificing weldability."
        ),
        "composition_range": "Dominant in low-carbon steels (< 0.25 wt% C). Present in all hypoeutectoid steels.",
        "related_phases": ["pearlite", "austenite", "acicular_ferrite"],
    },

    "acicular_ferrite": {
        "name": "Acicular Ferrite",
        "category": "Transformation Product",
        "summary": "Fine, interlocking needle-like ferrite nucleated on inclusions — excellent toughness in weld metals.",
        "description": (
            "Acicular ferrite consists of fine, interlocking, randomly oriented ferrite needles or laths "
            "that nucleate intragranularly on non-metallic inclusions (oxides, nitrides, sulfides). Unlike "
            "Widmanstätten ferrite which grows from grain boundaries, acicular ferrite nucleates throughout "
            "the grain interior, creating a chaotic, interlocking microstructure that forces crack deflection "
            "and provides excellent toughness."
        ),
        "key_features": [
            "Fine needle-like ferrite laths oriented in multiple random directions",
            "Interlocking, chaotic arrangement (no parallel alignment like Widmanstätten)",
            "Nucleates on inclusions within prior austenite grains, not at grain boundaries",
            "Typical lath width 1–3 μm, length 5–15 μm",
            "Often found in weld metals and controlled-rolled steels"
        ],
        "optical_appearance": (
            "Fine, randomly oriented needle-like features that create an interlocking basket-weave pattern. "
            "Lighter than bainite or martensite after etching. The chaotic orientation is the key visual "
            "distinction from Widmanstätten ferrite (which grows in parallel side-plates from grain boundaries)."
        ),
        "sem_appearance": (
            "Individual laths resolvable with random orientations. Inclusions may be visible at nucleation "
            "sites within laths. High-angle boundaries between laths contribute to toughness."
        ),
        "confused_with": [
            "Bainite — similar acicular morphology, but bainite has carbides and grows in sheaves",
            "Widmanstätten ferrite — both are needle-like, but Widmanstätten grows from grain boundaries in parallel sets"
        ],
        "typical_hardness": "200–280 HV",
        "formation": "Nucleates on non-metallic inclusions during continuous cooling. Favored by specific inclusion engineering.",
        "engineering_significance": (
            "The ideal microstructure for weld metal toughness. Inclusion engineering (controlling Ti, O, Al, S) "
            "is used to promote acicular ferrite nucleation in weld deposits."
        ),
        "composition_range": "Low-carbon steels, especially weld metals with controlled inclusion populations.",
        "related_phases": ["ferrite", "widmanstatten", "bainite"],
    },

    "spheroidite": {
        "name": "Spheroidite",
        "category": "Annealed Microstructure",
        "summary": "Rounded carbide particles in a ferrite matrix — the softest microstructure for a given carbon content.",
        "description": (
            "Spheroidite consists of spheroidal (globular) cementite particles dispersed in a continuous "
            "ferrite matrix. It forms when pearlite or bainite is annealed for extended periods (hours), "
            "causing the lamellar or acicular carbides to break up, coarsen, and minimize their surface "
            "energy by becoming spherical (Ostwald ripening). Spheroidite is the equilibrium microstructure "
            "and has the lowest hardness of any microstructure for a given carbon content."
        ),
        "key_features": [
            "Rounded, spherical carbide particles (cementite) uniformly dispersed in ferrite",
            "No lamellar, acicular, or aligned features",
            "Particle size typically 0.5–5 μm, depends on annealing time/temperature",
            "Soft and highly ductile — excellent formability",
            "Dark spherical particles on a light ferrite background"
        ],
        "optical_appearance": (
            "Dispersed dark spherical particles in a light ferrite matrix. Easily distinguished from "
            "lamellar pearlite by the absence of any aligned structures. At low magnification, may appear "
            "as a uniformly gray field. At higher magnification, individual carbide particles are clearly "
            "resolved as discrete dark dots."
        ),
        "sem_appearance": (
            "Discrete spherical particles clearly resolved on a smooth ferrite matrix. Particle size "
            "typically 0.5–5 μm. Larger particles may show faceting at very high magnification."
        ),
        "confused_with": [
            "Tempered martensite — also contains fine carbide particles, but retains the lath structure of martensite",
            "Overtempered bainite — can develop spheroidal carbides but retains some acicular ferrite morphology"
        ],
        "typical_hardness": "150–250 HV",
        "formation": "Prolonged subcritical annealing of pearlite, typically 15–25 hours at ~700°C.",
        "engineering_significance": (
            "Spheroidite is the softest, most ductile condition for steel of any given carbon content. "
            "Used when maximum formability is needed before subsequent forming or machining operations. "
            "Spheroidize annealing is common for medium- and high-carbon steels."
        ),
        "composition_range": "Any carbon steel can be spheroidized. Most common in medium-high carbon (0.4–1.0% C).",
        "related_phases": ["pearlite", "cementite", "ferrite"],
    },

    "austenite": {
        "name": "Austenite (γ-Iron / Retained)",
        "category": "High-Temperature / Stabilized Phase",
        "summary": "The FCC parent phase of steel — stable at high temperature, can be retained at room temperature by alloying.",
        "description": (
            "Austenite is the face-centered cubic (FCC) phase of iron, normally stable above 727°C "
            "in carbon steels. It can exist at room temperature as 'retained austenite' when stabilized "
            "by alloying elements (Mn, Ni, C enrichment) or by rapid cooling that prevents complete "
            "transformation. In TRIP (Transformation-Induced Plasticity) steels, retained austenite "
            "provides both strength and ductility by transforming to martensite during deformation. "
            "In fully austenitic stainless steels and Hadfield manganese steels, austenite is the "
            "primary stable phase."
        ),
        "key_features": [
            "Equiaxed grains when fully austenitic (austenitic stainless steels)",
            "Annealing twins — parallel straight lines within grains (FCC signature feature)",
            "As retained austenite in carbon steels: thin films between martensite laths or blocky islands",
            "Non-magnetic (unlike ferrite and martensite)",
            "Higher carbon solubility than ferrite (up to 2.11% C)"
        ],
        "optical_appearance": (
            "In austenitic steels: light etching equiaxed grains with characteristic annealing twins "
            "visible as parallel lines within grains. In multiphase steels, retained austenite appears "
            "as small bright islands or thin films between other phases."
        ),
        "sem_appearance": (
            "Smooth grain surfaces with twin boundaries clearly visible. In multiphase steels, EBSD "
            "(electron backscatter diffraction) is often needed to reliably identify retained austenite "
            "because morphology alone is not sufficient."
        ),
        "confused_with": [
            "Ferrite — also equiaxed, but ferrite is BCC and does not show annealing twins",
            "M/A islands — contain retained austenite as a component"
        ],
        "typical_hardness": "150–300 HV (depends on alloy composition)",
        "formation": "Stable above 727°C; retained at room temperature by alloying or rapid cooling.",
        "engineering_significance": (
            "Retained austenite is critical in TRIP steels (ductility via transformation), Q&P steels, "
            "and cryogenic applications. Austenitic stainless steels (304, 316) rely on stable austenite "
            "for corrosion resistance and non-magnetic properties."
        ),
        "composition_range": "Retained in C-enriched regions of multiphase steels. Stable in high-Mn, high-Ni alloys.",
        "related_phases": ["ferrite", "martensite", "bainite_ma"],
    },

    "cementite": {
        "name": "Cementite (Fe₃C)",
        "category": "Carbide Phase",
        "summary": "Iron carbide — the hard, brittle phase that provides strengthening in pearlite and other microstructures.",
        "description": (
            "Cementite (Fe₃C) is the primary carbide phase in carbon steels, containing 6.67 wt% carbon. "
            "It is hard (~1000 HV) and brittle. Cementite appears in many morphologies depending on "
            "heat treatment: as lamellae in pearlite, as a network along prior austenite grain boundaries "
            "(proeutectoid cementite in hypereutectoid steels), as spheroids in spheroidite, or as "
            "Widmanstätten plates. The network morphology is particularly detrimental to mechanical "
            "properties because it provides a continuous crack path."
        ),
        "key_features": [
            "Very hard and brittle — provides strengthening but limits ductility",
            "Appears bright/white in SEM (higher average atomic number than ferrite)",
            "Multiple morphologies: lamellar, network, spheroidal, Widmanstätten",
            "Network form along grain boundaries is detrimental to properties",
            "Contains 6.67 wt% carbon in stoichiometric Fe₃C"
        ],
        "optical_appearance": (
            "Typically bright/white after nital etching. Network cementite appears as continuous bright "
            "films along grain boundaries. In pearlite, cementite lamellae are the dark-etching component "
            "at standard optical resolution (due to shadowing) but appear bright in SEM."
        ),
        "sem_appearance": (
            "Bright in backscatter mode due to higher average atomic number than ferrite. Various "
            "morphologies are clearly resolved. Network cementite shows as continuous thin films; "
            "spheroidized cementite as discrete round particles."
        ),
        "confused_with": [
            "Other carbides in alloy steels (Cr₇C₃, Mo₂C, VC) — require EDS/EBSD for positive identification"
        ],
        "typical_hardness": "800–1000 HV",
        "formation": "Precipitates from austenite during cooling. Morphology depends on temperature, cooling rate, and composition.",
        "engineering_significance": (
            "The essential strengthening carbide in plain carbon steels. Controlled morphology is key: "
            "lamellar (in pearlite) is good for wear, spheroidized is good for formability, "
            "network cementite along grain boundaries is always detrimental and must be avoided."
        ),
        "composition_range": "Present in all steels with carbon. Dominant in hypereutectoid steels (> 0.76% C).",
        "related_phases": ["pearlite", "spheroidite", "network_cementite"],
    },

    "network_cementite": {
        "name": "Network Cementite",
        "category": "Carbide Morphology (Detrimental)",
        "summary": "Continuous cementite films along grain boundaries — creates crack paths and severely embrittles steel.",
        "description": (
            "Network cementite (or proeutectoid cementite) forms as a continuous film along prior "
            "austenite grain boundaries in hypereutectoid steels (> 0.76% C) during slow cooling. "
            "This continuous network provides an easy crack propagation path, making the steel "
            "extremely brittle regardless of the matrix microstructure. Avoiding network cementite "
            "through faster cooling or spheroidize annealing is critical in high-carbon steels."
        ),
        "key_features": [
            "Continuous or near-continuous bright films along prior austenite grain boundaries",
            "Outlines the prior austenite grain structure clearly",
            "Creates a 3D interconnected network (brittle skeleton)",
            "Most common in slowly cooled hypereutectoid steels",
            "Must be eliminated by heat treatment before use"
        ],
        "optical_appearance": (
            "Bright/white continuous lines outlining prior austenite grain boundaries against a "
            "darker pearlite matrix. The network pattern is distinctive and immediately recognizable."
        ),
        "sem_appearance": (
            "Bright continuous films at grain boundaries. Film thickness is typically 0.1–1 μm. "
            "Backscatter imaging clearly resolves the cementite network."
        ),
        "confused_with": [
            "Proeutectoid ferrite network — but ferrite network is softer and in hypoeutectoid steels",
            "Grain boundary segregation — elemental segregation can outline boundaries but without a second phase"
        ],
        "typical_hardness": "800–1000 HV (the cementite itself)",
        "formation": "Slow cooling of hypereutectoid steels (> 0.76% C) through the Acm temperature.",
        "engineering_significance": (
            "Always detrimental. Must be eliminated by normalizing (air cooling to break up the network) "
            "or spheroidize annealing before the steel can be used. A classic failure mode in improperly "
            "heat-treated high-carbon steels."
        ),
        "composition_range": "Hypereutectoid steels (> 0.76 wt% C).",
        "related_phases": ["cementite", "pearlite", "spheroidite"],
    },

    "widmanstatten": {
        "name": "Widmanstätten Ferrite",
        "category": "Transformation Product",
        "summary": "Long, parallel ferrite plates growing from grain boundaries — forms at moderate undercoolings.",
        "description": (
            "Widmanstätten ferrite consists of plate-like or needle-like ferrite (or, less commonly, "
            "cementite) growing along specific crystallographic planes of the parent austenite. The plates "
            "grow as 'side-plates' from grain boundaries into the grain interior, creating a characteristic "
            "herringbone or basket-weave pattern. Widmanstätten ferrite forms at moderate undercoolings — "
            "faster than equiaxed ferrite but slower than bainite — and is favored by large prior "
            "austenite grain sizes."
        ),
        "key_features": [
            "Long, parallel-sided plates or needles emanating from grain boundaries",
            "Characteristic herringbone or basket-weave pattern",
            "Plates grow along specific crystallographic habit planes",
            "Favored by large austenite grain size and moderate cooling rates",
            "Pearlite or other transformation products fill the space between plates"
        ],
        "optical_appearance": (
            "Long bright ferrite plates growing from prior austenite grain boundaries into grain interiors. "
            "Dark pearlite or other transformation products between plates. The side-plate geometry — plates "
            "angled relative to the grain boundary — is the key identifier."
        ),
        "sem_appearance": (
            "Plate morphology clearly resolved with sharp, planar boundaries. Carbides may decorate plate "
            "boundaries. The crystallographic nature of the plates (flat, faceted interfaces) distinguishes "
            "them from irregular allotriomorphic ferrite."
        ),
        "confused_with": [
            "Acicular ferrite — similar needle morphology but nucleates intragranularly, not from grain boundaries",
            "Bainite laths — similar acicular shape but bainite forms at lower temperatures with carbides"
        ],
        "typical_hardness": "150–250 HV",
        "formation": "Moderate cooling rates from austenite. Favored at large grain sizes (> 100 μm).",
        "engineering_significance": (
            "Generally considered undesirable because it reduces toughness compared to equiaxed ferrite. "
            "Avoided by grain refinement and controlled cooling. Sometimes observed in weld heat-affected "
            "zones where grain coarsening has occurred."
        ),
        "composition_range": "Low to medium carbon steels (0.1–0.4 wt% C) with large prior austenite grains.",
        "related_phases": ["ferrite", "acicular_ferrite", "bainite"],
    },

    "duplex": {
        "name": "Duplex / Dual-Phase",
        "category": "Engineered Multi-Phase",
        "summary": "Two or more distinct phases in designed proportions — the basis of advanced high-strength steels.",
        "description": (
            "Duplex or dual-phase microstructures contain two or more clearly distinct phases in "
            "significant proportions, engineered for specific mechanical property combinations. The most "
            "common example is ferrite-martensite dual-phase (DP) steel, where islands of hard martensite "
            "in a soft ferrite matrix provide high strength with good ductility. Other duplex systems "
            "include ferrite-austenite (duplex stainless steels) and ferrite-bainite combinations."
        ),
        "key_features": [
            "Two clearly distinct microstructural constituents with different etching response",
            "Designed phase fractions (e.g., 70% ferrite + 30% martensite in DP600)",
            "Each phase contributes different mechanical properties",
            "Continuous yielding behavior (no yield point in DP steels)",
            "Phase boundaries provide sites for strain localization"
        ],
        "optical_appearance": (
            "Distinct regions with different contrast/color after etching. In DP steel, ferrite is light "
            "and martensite is dark. Phase distribution depends on prior austenite grain size and "
            "intercritical annealing conditions."
        ),
        "sem_appearance": (
            "Phases distinguishable by topographic or compositional contrast. EDS/EBSD can confirm "
            "composition and crystal structure differences between phases."
        ),
        "confused_with": [
            "Incomplete transformation products — partial transformation during quenching can look duplex",
            "Segregation bands — chemical banding can create apparent phase differences"
        ],
        "typical_hardness": "Varies widely — depends on constituent phases and fractions",
        "formation": "Controlled thermal processing (intercritical annealing, controlled cooling) to develop target phase fractions.",
        "engineering_significance": (
            "The foundation of Advanced High-Strength Steels (AHSS). DP steels are used extensively in "
            "automotive body structures for crash safety. TRIP, Q&P, and 3rd generation AHSS all rely "
            "on engineered multi-phase microstructures."
        ),
        "composition_range": "Low-carbon steels (0.05–0.2% C) with Mn, Si, Cr for DP steels. Wide range for other duplex types.",
        "related_phases": ["ferrite", "martensite", "austenite", "bainite"],
    },

    "pearlite_widmanstatten": {
        "name": "Pearlite + Widmanstätten",
        "category": "Mixed Microstructure",
        "summary": "Combination of pearlite colonies and Widmanstätten ferrite side-plates — common in moderately cooled steels.",
        "description": (
            "A mixed microstructure where Widmanstätten ferrite side-plates form first during cooling "
            "(at moderate undercooling), and the remaining austenite subsequently transforms to pearlite "
            "at the eutectoid temperature. This combination is common in medium-carbon steels cooled at "
            "moderate rates, especially with large prior austenite grain sizes."
        ),
        "key_features": [
            "Widmanstätten ferrite plates emanating from grain boundaries",
            "Pearlite colonies filling the space between ferrite plates",
            "Both components clearly identifiable at appropriate magnification",
            "Indicates moderate cooling rate and/or large prior austenite grain size"
        ],
        "optical_appearance": (
            "Long bright ferrite plates with dark pearlite regions between them. Both the lamellar "
            "pearlite and the plate-like Widmanstätten ferrite are distinguishable at higher magnification."
        ),
        "sem_appearance": "Combination of features from both phases — ferrite plates and lamellar pearlite regions.",
        "confused_with": [
            "Pure pearlite — at low magnification, the ferrite plates may not be obvious",
            "Bainite + pearlite — bainite plates can resemble Widmanstätten ferrite"
        ],
        "typical_hardness": "180–300 HV",
        "formation": "Moderate cooling of medium-carbon steels with large prior austenite grains.",
        "engineering_significance": "Generally indicates suboptimal heat treatment. Grain refinement eliminates the Widmanstätten component.",
        "composition_range": "Medium-carbon steels (0.2–0.5% C) with large grain size.",
        "related_phases": ["pearlite", "widmanstatten", "ferrite"],
    },

    "pearlite_spheroidite": {
        "name": "Pearlite + Spheroidite (Partially Spheroidized)",
        "category": "Transitional Microstructure",
        "summary": "Pearlite in the process of spheroidizing — lamellae breaking up into spheroidal carbides.",
        "description": (
            "A partially spheroidized microstructure where some pearlite colonies retain their lamellar "
            "structure while others have begun breaking up into spheroidal cementite particles. This is "
            "a transitional state during the spheroidize annealing process."
        ),
        "key_features": [
            "Mix of lamellar pearlite regions and spheroidized carbide regions",
            "Broken-up lamellae visible — segments of former lamellae becoming rounded",
            "Indicates partial completion of spheroidize annealing",
            "Hardness intermediate between pearlite and spheroidite"
        ],
        "optical_appearance": "Regions of lamellar pearlite adjacent to regions with discrete rounded carbide particles.",
        "sem_appearance": "Transition clearly visible — lamellar segments breaking apart and rounding at tips.",
        "confused_with": [
            "Fully spheroidized — but some lamellar remnants remain",
            "Divorced pearlite — forms directly from austenite with some spheroidal carbides"
        ],
        "typical_hardness": "180–280 HV",
        "formation": "Intermediate stage during subcritical annealing of pearlite.",
        "engineering_significance": "Indicates annealing is incomplete. Longer time or higher temperature needed for full spheroidization.",
        "composition_range": "Medium to high carbon steels undergoing spheroidize anneal.",
        "related_phases": ["pearlite", "spheroidite", "cementite"],
    },

    "ferrite_pearlite": {
        "name": "Ferrite + Pearlite",
        "category": "Common Two-Phase Microstructure",
        "summary": "The most common steel microstructure — equiaxed ferrite grains with pearlite colonies.",
        "description": (
            "Ferrite-pearlite is the most commonly observed microstructure in plain carbon and low-alloy "
            "steels. During slow cooling, proeutectoid ferrite nucleates at austenite grain boundaries and "
            "grows into the grains. The remaining austenite transforms to pearlite at the eutectoid "
            "temperature. The relative amounts of ferrite and pearlite depend on carbon content: "
            "more carbon → more pearlite."
        ),
        "key_features": [
            "Bright equiaxed ferrite grains + dark pearlite colonies",
            "Ferrite:pearlite ratio reflects carbon content",
            "Well-defined grain boundaries in the ferrite",
            "The 'textbook' microstructure for slowly cooled hypoeutectoid steels"
        ],
        "optical_appearance": (
            "Classic two-phase appearance: bright ferrite grains and dark pearlite colonies. "
            "The ratio is approximately predictable by the lever rule from the iron-carbon phase diagram."
        ),
        "sem_appearance": "Ferrite grains are smooth; pearlite colonies show internal lamellar structure.",
        "confused_with": [
            "Ferrite + bainite — dark bainite regions can resemble pearlite at low magnification"
        ],
        "typical_hardness": "120–250 HV (depends on ferrite:pearlite ratio)",
        "formation": "Slow cooling of hypoeutectoid steels (< 0.76% C) from the austenite region.",
        "engineering_significance": (
            "The default microstructure for structural steels, rebar, plate steels, and many other "
            "applications. Properties are controlled mainly by grain size and pearlite fraction."
        ),
        "composition_range": "All hypoeutectoid steels (0.05–0.76 wt% C).",
        "related_phases": ["ferrite", "pearlite"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-SHOT PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════
# Multiple text prompts per phase for ensemble averaging.
# These describe what each phase looks like so CLIP can match an uploaded
# image's visual appearance to the right phase.

PHASE_PROMPTS = {
    "pearlite": [
        "a metallographic micrograph showing pearlite with alternating lamellar plates of ferrite and cementite",
        "a microscope image of pearlite microstructure with fingerprint-like lamellar pattern in steel",
        "pearlite colonies showing parallel alternating light and dark lamellae in a polished and etched steel sample",
        "a micrograph of eutectoid steel showing dark pearlite regions with visible lamellar structure",
    ],
    "bainite": [
        "a metallographic micrograph showing bainite with acicular needle-like ferrite and dispersed carbides",
        "a microscope image of bainitic microstructure with lath-like ferrite plates in steel",
        "bainite sheaf structure showing feathery acicular features in an etched steel sample",
        "a micrograph showing upper or lower bainite with carbide particles between ferrite laths",
    ],
    "bainite_ma": [
        "a metallographic micrograph showing bainite with bright martensite-austenite islands in steel",
        "SEM image of bainitic ferrite matrix containing bright M/A constituent islands",
        "a micrograph of low-carbon steel showing bainite laths with blocky martensite-austenite features",
        "bainitic microstructure with retained austenite islands visible as bright features",
    ],
    "martensite": [
        "a metallographic micrograph showing martensite with lath or needle-like morphology in quenched steel",
        "a microscope image of martensitic microstructure appearing dark with fine lath structure",
        "martensite in steel showing high-contrast dark etching acicular or lath features",
        "a micrograph of quenched steel showing dense martensitic lath packets and blocks",
    ],
    "ferrite": [
        "a metallographic micrograph showing ferrite with clean equiaxed polygonal grains and visible grain boundaries",
        "a microscope image of ferritic microstructure with bright featureless equiaxed grains in steel",
        "ferrite grains in a polished and etched steel sample showing clear grain boundaries on light background",
        "a micrograph of low-carbon steel with bright polygonal ferrite grains",
    ],
    "spheroidite": [
        "a metallographic micrograph showing spheroidite with rounded globular carbide particles dispersed in a ferrite matrix",
        "a microscope image of spheroidized cementite particles uniformly distributed in steel",
        "spheroidite microstructure showing dark spherical particles scattered in a light matrix",
        "a micrograph of annealed steel with spheroidal carbide particles replacing lamellar pearlite",
    ],
    "austenite": [
        "a metallographic micrograph showing retained austenite with equiaxed grains and annealing twins",
        "a microscope image of austenitic microstructure with twin boundaries visible inside grains",
        "retained austenite appearing as bright islands or equiaxed grains with parallel twin lines",
        "a micrograph of austenitic steel showing equiaxed grains with characteristic annealing twins",
    ],
    "cementite": [
        "a metallographic micrograph showing cementite network along grain boundaries in steel",
        "a microscope image of iron carbide cementite appearing as bright white phase at grain boundaries",
        "cementite in steel appearing as bright continuous films or particles along prior austenite grain boundaries",
        "a micrograph showing proeutectoid cementite network surrounding other microstructural phases",
    ],
    "widmanstatten": [
        "a metallographic micrograph showing Widmanstatten ferrite plates growing from grain boundaries",
        "a microscope image of Widmanstatten structure with long parallel-sided plates in a herringbone pattern",
        "Widmanstatten side-plates of ferrite emanating from prior austenite grain boundaries in steel",
        "a micrograph showing characteristic Widmanstatten basket-weave pattern of elongated ferrite plates",
    ],
    "duplex": [
        "a metallographic micrograph showing dual-phase microstructure with two distinct constituents",
        "a microscope image of duplex steel with two clearly different phases visible",
        "bi-modal microstructure showing regions of different contrast and morphology in steel",
        "a micrograph of advanced high-strength steel with distinct light and dark phase regions",
    ],
}


sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.embed import get_embedder
from src.index import MicrostructureIndex

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="μStruct · Microstructure Identifier",
    description="Upload a micrograph to identify phases, learn about microstructures, and find similar matches.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models on startup ───────────────────────────────────────────────────
embedder = None
index = None


@app.on_event("startup")
async def load_models():
    global embedder, index

    try:
        embedder = get_embedder()
        print(f"Loaded {config.EMBEDDING_MODEL} embedder")
    except Exception as e:
        print(f"Warning: Could not load embedder: {e}")

    try:
        index = MicrostructureIndex.load()
        print(f"Loaded index with {index.size} vectors")

        # Log label distribution and normalization coverage
        raw_labels = Counter(m.get("label", "unknown") for m in index.metadata)
        print("Label distribution in index:")
        for lbl, count in raw_labels.most_common():
            normalized = _normalize_label(lbl)
            marker = "✓" if normalized in PHASE_INFO else "⚠"
            print(f"  {marker} '{lbl}' → '{normalized}' ({count} images)")

    except FileNotFoundError:
        print("Warning: No index found. Run `python scripts/build_index.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFICATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _zero_shot_classify(image_embedding, embedder_instance) -> dict[str, float]:
    """
    Run CLIP zero-shot classification against all known phases.
    Returns dict of phase_key -> probability.
    """
    if not hasattr(embedder_instance, "classify_zero_shot"):
        return {}

    return embedder_instance.classify_zero_shot(image_embedding, PHASE_PROMPTS)


def _voting_classify(results: list[dict], top_k: int = 20) -> dict[str, float]:
    """
    Weighted voting from similarity search results.

    Each result votes for its NORMALIZED label, weighted by similarity score.
    Results closer to the query get more vote weight.
    """
    if not results:
        return {}

    votes = {}
    total_weight = 0.0

    for r in results[:top_k]:
        raw_label = r.get("label", "unknown")
        label = _normalize_label(raw_label)
        if label == "unknown":
            continue

        # For cosine similarity, distance IS the similarity (higher = better)
        # For L2, we invert: weight = 1 / (1 + distance)
        if config.SIMILARITY_METRIC == "cosine":
            weight = max(0.0, float(r.get("distance", 0)))
        else:
            weight = 1.0 / (1.0 + float(r.get("distance", 1)))

        votes[label] = votes.get(label, 0.0) + weight
        total_weight += weight

    if total_weight == 0:
        return {}

    # Normalize to probabilities
    return {k: v / total_weight for k, v in votes.items()}


def _ensemble_classify(
    zero_shot_probs: dict[str, float],
    voting_probs: dict[str, float],
    zs_weight: float = 0.4,
    vote_weight: float = 0.6,
) -> dict[str, float]:
    """
    Combine zero-shot and voting probabilities into a single prediction.

    Voting gets higher weight because it uses actual labeled data from
    the database. Zero-shot provides a useful regularizing signal,
    especially when the database has class imbalance.
    """
    all_phases = set(list(zero_shot_probs.keys()) + list(voting_probs.keys()))

    combined = {}
    for phase in all_phases:
        zs = zero_shot_probs.get(phase, 0.0)
        vt = voting_probs.get(phase, 0.0)
        combined[phase] = zs * zs_weight + vt * vote_weight

    # Re-normalize
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}

    # Sort by probability descending
    return dict(sorted(combined.items(), key=lambda x: -x[1]))


def _build_diagnostic_report(
    ensemble_probs: dict[str, float],
    zero_shot_probs: dict[str, float],
    voting_probs: dict[str, float],
    results: list[dict],
) -> dict:
    """
    Build the full diagnostic report from identification results.

    ALWAYS returns educational content, even for low confidence or unknown labels.
    The report includes:
    - Primary identification with confidence
    - Runner-up identification(s)
    - Phase knowledge (description, features, formation, etc.)
    - Method agreement analysis
    - Reference images from the library matching the identified phase
    - "How to tell the difference" when phases are close
    """
    if not ensemble_probs:
        return {
            "identified": False,
            "message": "Could not identify — no labeled data available.",
            "educational_fallback": _get_top_phase_summaries(3),
        }

    sorted_phases = list(ensemble_probs.items())
    primary_key, primary_conf = sorted_phases[0]
    primary_info = PHASE_INFO.get(primary_key, {})

    # If primary phase isn't in our knowledge base, still provide a useful report
    if not primary_info:
        # Try to find the best matching known phase from the ensemble
        for phase_key, _ in sorted_phases:
            if phase_key in PHASE_INFO:
                primary_key = phase_key
                primary_conf = ensemble_probs[phase_key]
                primary_info = PHASE_INFO[phase_key]
                break

    # If still no match, use the zero-shot top pick as a fallback
    if not primary_info and zero_shot_probs:
        zs_top = max(zero_shot_probs, key=zero_shot_probs.get)
        if zs_top in PHASE_INFO:
            primary_key = zs_top
            primary_conf = zero_shot_probs[zs_top]
            primary_info = PHASE_INFO[zs_top]

    # Last resort: provide a generic report with top candidates
    if not primary_info:
        return {
            "identified": False,
            "message": (
                "The uploaded image does not closely match any phase in our knowledge base. "
                "This may be a less common microstructure, a non-steel material, or an image "
                "that is not a micrograph. Here are the closest candidates:"
            ),
            "educational_fallback": _get_top_phase_summaries(3),
            "probabilities": {k: round(v, 4) for k, v in ensemble_probs.items() if v > 0.005},
        }

    # Runner-up (if confidence is split)
    runner_up = None
    if len(sorted_phases) > 1:
        ru_key, ru_conf = sorted_phases[1]
        ru_normalized = ru_key if ru_key in PHASE_INFO else _find_closest_known_phase(ru_key)
        ru_info = PHASE_INFO.get(ru_normalized, PHASE_INFO.get(ru_key, {}))
        if ru_conf > 0.10 and ru_info:  # Only show if meaningful
            runner_up = {
                "phase_key": ru_normalized or ru_key,
                "phase_name": ru_info.get("name", ru_key),
                "confidence": round(ru_conf, 3),
                "description": ru_info.get("description", ""),
            }

    # Method agreement
    zs_top = max(zero_shot_probs, key=zero_shot_probs.get) if zero_shot_probs else None
    vt_top = max(voting_probs, key=voting_probs.get) if voting_probs else None
    methods_agree = zs_top == vt_top if (zs_top and vt_top) else None

    agreement_note = ""
    if methods_agree is True:
        agreement_note = f"Both CLIP zero-shot analysis and database matching agree: this is {primary_info.get('name', primary_key)}."
    elif methods_agree is False:
        zs_name = PHASE_INFO.get(zs_top, {}).get("name", zs_top) if zs_top else "unknown"
        vt_name = PHASE_INFO.get(vt_top, {}).get("name", vt_top) if vt_top else "unknown"
        agreement_note = (
            f"Note: CLIP visual analysis suggests {zs_name}, while database "
            f"matching leans toward {vt_name}. This can happen when phases share "
            f"visual texture patterns. Consider the key distinguishing features below."
        )

    # Collect reference images using normalized labels
    reference_images = []
    other_images = []
    for r in results:
        r_label = _normalize_label(r.get("label", "unknown"))
        if r_label == primary_key and len(reference_images) < 6:
            reference_images.append({
                "index": r["index"],
                "filename": r.get("filename", ""),
                "distance": r.get("distance", 0),
                "source": r.get("source", ""),
                "imaging": r.get("metadata", {}).get("imaging", ""),
            })
        elif r_label != "unknown" and len(other_images) < 4:
            other_images.append({
                "index": r["index"],
                "filename": r.get("filename", ""),
                "label": r_label,
                "label_name": PHASE_INFO.get(r_label, {}).get("name", r_label),
                "distance": r.get("distance", 0),
                "source": r.get("source", ""),
            })

    # "How to tell the difference" section
    differentiation = []
    if runner_up and primary_info.get("confused_with"):
        ru_key_check = runner_up["phase_key"]
        for confused_item in primary_info.get("confused_with", []):
            if ru_key_check in confused_item.lower() or runner_up["phase_name"].lower() in confused_item.lower():
                ru_info_full = PHASE_INFO.get(ru_key_check, {})
                differentiation.append({
                    "phases": f"{primary_info.get('name', primary_key)} vs {runner_up['phase_name']}",
                    "note": confused_item,
                    "primary_features": primary_info.get("key_features", [])[:3],
                    "runner_up_features": ru_info_full.get("key_features", [])[:3],
                })
                break

    # Confidence level label
    if primary_conf > 0.75:
        confidence_level = "high"
    elif primary_conf > 0.45:
        confidence_level = "moderate"
    elif primary_conf > 0.25:
        confidence_level = "low"
    else:
        confidence_level = "uncertain"

    return {
        "identified": True,
        "confidence_level": confidence_level,

        # Primary identification — always populated from PHASE_INFO
        "primary": {
            "phase_key": primary_key,
            "phase_name": primary_info.get("name", primary_key),
            "category": primary_info.get("category", ""),
            "confidence": round(primary_conf, 3),
            "summary": primary_info.get("summary", ""),
            "description": primary_info.get("description", ""),
            "key_features": primary_info.get("key_features", []),
            "optical_appearance": primary_info.get("optical_appearance", ""),
            "sem_appearance": primary_info.get("sem_appearance", ""),
            "typical_hardness": primary_info.get("typical_hardness", ""),
            "formation": primary_info.get("formation", ""),
            "confused_with": primary_info.get("confused_with", []),
            "engineering_significance": primary_info.get("engineering_significance", ""),
            "composition_range": primary_info.get("composition_range", ""),
            "related_phases": primary_info.get("related_phases", []),
        },

        # Runner-up
        "runner_up": runner_up,

        # Full probability breakdown (using display names)
        "probabilities": {
            k: round(v, 4) for k, v in ensemble_probs.items() if v > 0.005
        },

        # Individual method scores (for transparency)
        "method_detail": {
            "zero_shot": {k: round(v, 4) for k, v in zero_shot_probs.items() if v > 0.01} if zero_shot_probs else {},
            "voting": {k: round(v, 4) for k, v in voting_probs.items() if v > 0.01} if voting_probs else {},
            "methods_agree": methods_agree,
            "agreement_note": agreement_note,
        },

        # Reference images from library
        "reference_images": reference_images,
        "other_matches": other_images,

        # How to tell the difference
        "differentiation": differentiation,
    }


def _get_top_phase_summaries(n: int = 3) -> list[dict]:
    """Get brief summaries of the most common phases as educational fallback."""
    common_phases = ["ferrite", "pearlite", "martensite", "bainite", "spheroidite"]
    summaries = []
    for key in common_phases[:n]:
        info = PHASE_INFO.get(key, {})
        summaries.append({
            "phase_key": key,
            "name": info.get("name", key),
            "summary": info.get("summary", ""),
            "key_features": info.get("key_features", [])[:3],
        })
    return summaries


def _find_closest_known_phase(label: str) -> str | None:
    """Try to find the closest known phase for an unmapped label."""
    normalized = _normalize_label(label)
    if normalized in PHASE_INFO:
        return normalized
    return None


def _load_index_manifest() -> dict | None:
    """Return reproducibility metadata for the current index, if available."""
    if not config.INDEX_MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(config.INDEX_MANIFEST_PATH.read_text())
    except Exception as e:
        return {"error": f"Could not read index manifest: {type(e).__name__}: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    has_text_search = hasattr(embedder, "embed_text") if embedder else False
    has_zero_shot = hasattr(embedder, "classify_zero_shot") if embedder else False
    manifest = _load_index_manifest()
    manifest_model = manifest.get("model") if manifest else None
    return {
        "status": "ok",
        "model": config.EMBEDDING_MODEL,
        "index_loaded": index is not None,
        "index_size": index.size if index else 0,
        "index_manifest": manifest,
        "index_model_matches_runtime": (
            manifest_model is None or manifest_model == config.EMBEDDING_MODEL
        ),
        "text_search": has_text_search,
        "zero_shot": has_zero_shot,
    }


@app.get("/api/stats")
async def stats():
    if not index:
        raise HTTPException(503, "Index not loaded")

    sources = {}
    labels = {}
    normalized_labels = {}
    modalities = {}
    for meta in index.metadata:
        src = meta.get("source", "unknown")
        lbl = meta.get("label", "unknown")
        sources[src] = sources.get(src, 0) + 1
        labels[lbl] = labels.get(lbl, 0) + 1
        norm_lbl = _normalize_label(lbl)
        normalized_labels[norm_lbl] = normalized_labels.get(norm_lbl, 0) + 1
        imaging = meta.get("metadata", {}).get("imaging", "unknown")
        modalities[imaging] = modalities.get(imaging, 0) + 1

    has_text_search = hasattr(embedder, "embed_text") if embedder else False
    has_zero_shot = hasattr(embedder, "classify_zero_shot") if embedder else False

    return {
        "index_size": index.size,
        "embedding_dim": index.embedding_dim,
        "metric": index.metric,
        "model": config.EMBEDDING_MODEL,
        "text_search": has_text_search,
        "zero_shot": has_zero_shot,
        "sources": sources,
        "labels": labels,
        "normalized_labels": normalized_labels,
        "modalities": modalities,
    }


@app.get("/api/phases")
async def get_phases():
    """Return the phase knowledge base for UI tooltips and info cards."""
    return PHASE_INFO


@app.get("/api/phase-guide")
async def get_phase_guide():
    """
    Phase encyclopedia endpoint.
    Returns all phases with their full knowledge base entries plus
    example counts and sample image indices from the database.
    """
    guide = {}

    # Count images per normalized label in the index
    label_counts = Counter()
    label_examples = {}  # phase_key -> list of (index, metadata) for sample images

    if index:
        for i, meta in enumerate(index.metadata):
            raw_label = meta.get("label", "unknown")
            norm = _normalize_label(raw_label)
            label_counts[norm] += 1

            if norm not in label_examples:
                label_examples[norm] = []
            if len(label_examples[norm]) < 6:
                label_examples[norm].append({
                    "index": i,
                    "filename": meta.get("filename", ""),
                    "source": meta.get("source", ""),
                    "imaging": meta.get("metadata", {}).get("imaging", ""),
                })

    for phase_key, info in PHASE_INFO.items():
        guide[phase_key] = {
            **info,
            "image_count": label_counts.get(phase_key, 0),
            "example_images": label_examples.get(phase_key, []),
            "has_examples": phase_key in label_examples,
        }

    return guide


# ── Identify Endpoint ─────────────────────────────────────────────────────────

@app.post("/api/identify")
async def identify_microstructure(
    file: UploadFile = File(...),
    top_k: int = Query(default=20, ge=5, le=50),
):
    """
    Upload a micrograph and get a full diagnostic identification report.

    Combines two identification methods:
    1. CLIP zero-shot: compares image against text descriptions of each phase
    2. Similarity voting: searches database, counts phase labels weighted by similarity

    Returns a diagnostic report with phase identification, confidence levels,
    educational content, reference images, and differentiation guidance.
    """
    if not index or not embedder:
        raise HTTPException(503, "Model or index not loaded")

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(400, "Invalid image file")

    try:
        # Generate embedding (directly from bytes — no temp file needed)
        if hasattr(embedder, "embed_image_bytes"):
            query_embedding = embedder.embed_image_bytes(contents)
        else:
            # Fallback: write temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                query_embedding = embedder.embed_single(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # ── Method 1: CLIP Zero-Shot Classification ──
        zero_shot_probs = _zero_shot_classify(query_embedding, embedder)

        # ── Method 2: Similarity Search + Voting ──
        search_results = index.search(query_embedding, top_k=top_k)

        # Filter out exact self-match
        if config.SIMILARITY_METRIC == "cosine":
            search_results = [r for r in search_results if r["distance"] < 0.999]
        else:
            search_results = [r for r in search_results if r["distance"] > 0.001]

        voting_probs = _voting_classify(search_results, top_k=top_k)

        # ── Ensemble ──
        ensemble_probs = _ensemble_classify(zero_shot_probs, voting_probs)

        # ── Build Report ──
        report = _build_diagnostic_report(
            ensemble_probs, zero_shot_probs, voting_probs, search_results
        )

        # Encode query image as thumbnail for display
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.thumbnail((400, 400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        query_thumb = base64.b64encode(buf.getvalue()).decode()

        return {
            "query": file.filename,
            "query_thumbnail": query_thumb,
            "report": report,
            "search_results": search_results[:8],
            "metric": config.SIMILARITY_METRIC,
        }

    except Exception as e:
        raise HTTPException(500, f"Identification failed: {str(e)}")


# ── Search Endpoints ──────────────────────────────────────────────────────────

def _analyze_results(results: list[dict], query_type: str = "image") -> dict:
    """Generate a results analysis summary."""
    if not results:
        return {"summary": "No results found.", "warnings": [], "phase_breakdown": {}}

    phase_counts = Counter()
    source_counts = Counter()
    modality_counts = Counter()

    for r in results:
        raw_lbl = r.get("label", "unknown")
        lbl = _normalize_label(raw_lbl)
        phase_counts[lbl] += 1
        source_counts[r.get("source", "unknown")] += 1
        imaging = r.get("metadata", {}).get("imaging", "unknown")
        modality_counts[imaging] += 1

    total = len(results)
    dominant_phase = phase_counts.most_common(1)[0] if phase_counts else ("unknown", 0)
    dominant_name, dominant_count = dominant_phase

    phase_breakdown = {}
    for phase, count in phase_counts.most_common():
        pct = round(100 * count / total)
        phase_info = PHASE_INFO.get(phase, {})
        phase_breakdown[phase] = {
            "count": count,
            "percentage": pct,
            "description": phase_info.get("description", ""),
            "name": phase_info.get("name", phase),
        }

    warnings = []

    non_unknown_phases = {p for p in phase_counts if p != "unknown"}
    if len(non_unknown_phases) > 1:
        phase_list = ", ".join(sorted(non_unknown_phases))
        warnings.append({
            "type": "mixed_phases",
            "message": f"Results span multiple phases ({phase_list}). Visual similarity may not indicate phase match — CLIP matches texture patterns, not metallurgical phases.",
            "severity": "warning"
        })

    if len(modality_counts) > 1 and "unknown" not in modality_counts:
        mod_list = ", ".join(sorted(modality_counts.keys()))
        warnings.append({
            "type": "mixed_modality",
            "message": f"Results include multiple imaging modalities ({mod_list}). Consider filtering by modality for more comparable matches.",
            "severity": "info"
        })

    if dominant_count == total:
        disp_name = PHASE_INFO.get(dominant_name, {}).get("name", dominant_name)
        summary = f"All {total} results are labeled as {disp_name}."
    else:
        parts = []
        for phase, info in phase_breakdown.items():
            name = info["name"]
            parts.append(f"{name} ({info['count']}/{total})")
        summary = f"Results include: {', '.join(parts)}."

    if modality_counts:
        mod_parts = [f"{m}: {c}" for m, c in modality_counts.most_common()]
        summary += f" Imaging: {', '.join(mod_parts)}."

    return {
        "summary": summary,
        "warnings": warnings,
        "phase_breakdown": phase_breakdown,
        "source_breakdown": dict(source_counts),
        "modality_breakdown": dict(modality_counts),
        "dominant_phase": dominant_name,
        "is_mixed": len(non_unknown_phases) > 1,
    }


def _filter_results(results: list[dict],
                     modality_filter: Optional[list[str]] = None,
                     source_filter: Optional[list[str]] = None,
                     label_filter: Optional[list[str]] = None) -> list[dict]:
    """Apply optional post-search filters to results."""
    filtered = results

    if modality_filter:
        modality_set = {m.lower() for m in modality_filter}
        filtered = [
            r for r in filtered
            if r.get("metadata", {}).get("imaging", "unknown").lower() in modality_set
        ]

    if source_filter:
        source_set = {s.lower() for s in source_filter}
        filtered = [
            r for r in filtered
            if r.get("source", "unknown").lower() in source_set
        ]

    if label_filter:
        label_set = {l.lower() for l in label_filter}
        filtered = [
            r for r in filtered
            if _normalize_label(r.get("label", "unknown")).lower() in label_set
        ]

    for i, r in enumerate(filtered):
        r["rank"] = i + 1

    return filtered


@app.post("/api/search")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=8, ge=1, le=50),
    modality: Optional[str] = Query(default=None, description="Comma-separated modality filter: optical,SEM"),
    source: Optional[str] = Query(default=None, description="Comma-separated source filter: aachen,kaggle,uhcs"),
    label: Optional[str] = Query(default=None, description="Comma-separated label filter"),
):
    """Upload a micrograph image and get the most similar microstructures."""
    if not index or not embedder:
        raise HTTPException(503, "Model or index not loaded")

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(400, "Invalid image file")

    modality_filter = [m.strip() for m in modality.split(",")] if modality else None
    source_filter = [s.strip() for s in source.split(",")] if source else None
    label_filter = [l.strip() for l in label.split(",")] if label else None
    has_filters = any([modality_filter, source_filter, label_filter])

    fetch_k = top_k * 5 if has_filters else top_k + 1

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        query_embedding = embedder.embed_single(tmp_path)
        results = index.search(query_embedding, top_k=fetch_k)

        if config.SIMILARITY_METRIC == "cosine":
            results = [r for r in results if r["distance"] < 0.999]
        else:
            results = [r for r in results if r["distance"] > 0.001]

        if has_filters:
            results = _filter_results(results, modality_filter, source_filter, label_filter)

        results = results[:top_k]

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.thumbnail((300, 300))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        query_thumb = base64.b64encode(buf.getvalue()).decode()

        analysis = _analyze_results(results, query_type="image")

        return {
            "query": file.filename,
            "query_thumbnail": query_thumb,
            "query_type": "image",
            "top_k": top_k,
            "metric": config.SIMILARITY_METRIC,
            "results": results,
            "analysis": analysis,
            "filters_applied": has_filters,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 8
    modality: Optional[str] = None
    source: Optional[str] = None
    label: Optional[str] = None


@app.post("/api/search-text")
async def search_by_text(request: TextSearchRequest):
    """Search microstructures by text description (CLIP only)."""
    if not index or not embedder:
        raise HTTPException(503, "Model or index not loaded")

    if not hasattr(embedder, "embed_text"):
        raise HTTPException(
            400,
            "Text search requires CLIP model. Rebuild index with: "
            "python scripts/build_index.py --model clip"
        )

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(400, "Empty query")

    top_k = max(1, min(request.top_k, 50))

    modality_filter = [m.strip() for m in request.modality.split(",")] if request.modality else None
    source_filter = [s.strip() for s in request.source.split(",")] if request.source else None
    label_filter = [l.strip() for l in request.label.split(",")] if request.label else None
    has_filters = any([modality_filter, source_filter, label_filter])

    fetch_k = top_k * 5 if has_filters else top_k

    try:
        query_embedding = embedder.embed_text(query_text)
        results = index.search(query_embedding, top_k=fetch_k)

        if has_filters:
            results = _filter_results(results, modality_filter, source_filter, label_filter)

        results = results[:top_k]

        analysis = _analyze_results(results, query_type="text")

        return {
            "query": query_text,
            "query_type": "text",
            "top_k": top_k,
            "metric": config.SIMILARITY_METRIC,
            "results": results,
            "analysis": analysis,
            "filters_applied": has_filters,
        }
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.get("/api/image/{idx}")
async def get_image(idx: int, size: int = Query(default=300, ge=50, le=1200)):
    """Serve a micrograph thumbnail by its index in the database."""
    if not index:
        raise HTTPException(503, "Index not loaded")

    if idx < 0 or idx >= len(index.metadata):
        raise HTTPException(404, f"Image index {idx} not found")

    meta = index.metadata[idx]

    # Try pre-baked thumbnail first
    if meta.get("thumbnail_b64"):
        return {
            "index": idx,
            "thumbnail": meta["thumbnail_b64"],
            "metadata": {k: v for k, v in meta.items() if k != "thumbnail_b64"},
        }

    # Fall back to loading from disk
    image_path = Path(meta["path"])
    if not image_path.exists():
        raise HTTPException(404, f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "index": idx,
        "thumbnail": b64,
        "metadata": meta,
    }


@app.get("/api/image/{idx}/full")
async def get_full_image(idx: int):
    """Serve the full-resolution micrograph image."""
    if not index:
        raise HTTPException(503, "Index not loaded")

    if idx < 0 or idx >= len(index.metadata):
        raise HTTPException(404, f"Image index {idx} not found")

    meta = index.metadata[idx]
    image_path = Path(meta["path"])

    if not image_path.exists():
        raise HTTPException(404, f"Image file not found: {image_path}")

    return FileResponse(image_path)


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    ui_path = Path(__file__).parent / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>UI not found</h1><p>Place index.html in the app/ directory.</p>")