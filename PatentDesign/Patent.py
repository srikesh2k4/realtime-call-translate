
# 1. Semantic Embedding Generation (Incremental embedding trajectory)


def embed(text: str) -> np.ndarray:
    """
    Converts an interim speech transcript into a semantic vector.
    This enables tracking of intent evolution over time.
    """
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype=np.float32)




# 2. Semantic Drift Calculation (Measures intent change)


def cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes semantic drift between two consecutive intent states.
    Lower values indicate semantic convergence.
    """
    return 1.0 - float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )

# 3. Semantic Intent Stabilization Logic (Primary patented mechanism)

def semantic_stable(embeds: list[np.ndarray]) -> bool:
    """
    Determines whether speaker intent has stabilized.
    Translation is deferred until semantic drift converges.
    """
    if len(embeds) < MIN_EMBEDDINGS:
        return False

    drifts = [
        cosine_drift(embeds[i - 1], embeds[i])
        for i in range(1, len(embeds))
    ]

    return float(np.mean(drifts[-3:])) < DRIFT_THRESHOLD


# 4. Linguistic Completeness Gate (Ensures intent is expressively complete)

def linguistically_complete(text: str) -> bool:
    """
    Confirms that the stabilized intent forms a complete linguistic unit.
    Prevents premature translation of partial thoughts.
    """
    if len(text.split()) < 4:
        return False

    return any(p in text for p in [".", "?", "!", "।"])


#5. Multi-Factor Intent Commit Controller (Core system claim)

def intent_commit_controller(
    texts: list[str],
    embeddings: list[np.ndarray],
    last_voice_time: float,
    current_time: float
) -> bool:
    """
    Determines whether translation should be committed.
    Uses multiple independent confirmation signals.
    """

    if not semantic_stable(embeddings):
        return False

    if not linguistically_complete(texts[-1]):
        return False

    if current_time - last_voice_time < MIN_SILENCE_SEC:
        return False

    return True

#6. Intent-Aligned Translation Execution (Translation only after stabilization)**

def commit_translation(
    texts: list[str],
    source_lang: str,
    target_langs: tuple[str, ...]
) -> list[dict]:
    """
    Executes translation only after intent stabilization.
    Output corresponds to finalized speaker intent.
    """

    final_text = " ".join(texts)
    outputs = []

    for tgt in target_langs:
        if tgt == source_lang:
            continue

        translated = strict_translate(final_text, source_lang, tgt)
        outputs.append({
            "sourceText": final_text,
            "translatedText": translated,
            "sourceLang": source_lang,
            "targetLang": tgt
        })

    return outputs

#7. Session-Scoped Semantic Isolation Reset (Prevents intent leakage)

def reset_session_state(state: dict):
    """
    Clears semantic buffers after translation commitment.
    Ensures each utterance is semantically isolated.
    """
    state["texts"].clear()
    state["embeddings"].clear()

#8. Minimal Patent-Core Flow (All together)

def patent_core_pipeline(
    state: dict,
    new_text: str,
    current_time: float
):
    """
    Minimal executable representation of the patented invention.
    """

    state["texts"].append(new_text)
    state["embeddings"].append(embed(new_text))

    if intent_commit_controller(
        texts=state["texts"],
        embeddings=state["embeddings"],
        last_voice_time=state["last_voice"],
        current_time=current_time
    ):
        outputs = commit_translation(
            texts=state["texts"],
            source_lang=state["lang"],
            target_langs=("en", "hi")
        )
        reset_session_state(state)
        return outputs

    return []

