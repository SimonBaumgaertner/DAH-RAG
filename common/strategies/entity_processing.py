from collections import defaultdict
from typing import List, Tuple, Set

from common.data_classes.knowledge_triplets import ExtractedKnowledgeTriplet, Entity



def build_aliases(
    triplets: List[ExtractedKnowledgeTriplet],
    all_entities: List[Tuple[str, str]],
) -> List[Entity]:
    """
    Build a unique Entity list with aliases using two stages, **after first pruning**
    `all_entities` to those that matter for the given triplets.

    Pruning (first step):
      - Keep an entity if its name appears in any triplet subject/object.
      - Additionally, if a one-word Person name is mentioned and maps *unambiguously*
        to a two-word Person name present in `all_entities`, also keep that two-word name.

    Stage 1 (rule-based, Person-only):
      - Map unambiguous one-word Person names to their two-word Person names and add
        the one-word forms as aliases (canonical is the two-word).

    Stage 2 (triplet-driven):
      - For alias-like triplets (alias/aka/also known as/same as/synonym/nickname),
        add aliases or merge entities as appropriate.

    Notes:
      - Triplets are never modified here.
      - We only create canonical Entities from `all_entities`. Triplets can add *string*
        aliases but won't create brand-new canonical Entities.
    """

    # --------------------------
    # Helpers / normalizers
    # --------------------------
    def norm(s: str) -> str:
        return s.strip().lower()

    def word_count(s: str) -> int:
        return len(s.split())

    def is_alias_rel(rel: str) -> bool:
        r = norm(rel)
        if "alias" in r:
            return True
        if "also known as" in r:
            return True
        return r in {"aka", "a.k.a.", "a.k.a", "same as", "synonym", "nickname"}

    # --------------------------
    # PRUNE FIRST: which names matter?
    # --------------------------
    # Mentions present in triplets (by surface form)
    mentioned: Set[str] = set()
    for tr in triplets:
        mentioned.add(norm(tr.subject))
        mentioned.add(norm(tr.object))

    # Precompute 2-word Person candidates: token -> set(full names)
    token_to_two_word: Dict[str, Set[str]] = defaultdict(set)
    one_word_person_names: Set[str] = set()
    for name, etype in all_entities:
        if etype == "Person":
            if word_count(name) == 2:
                t1, t2 = [norm(tok) for tok in name.split()]
                token_to_two_word[t1].add(name)
                token_to_two_word[t2].add(name)
            elif word_count(name) == 1:
                one_word_person_names.add(norm(name))

    # Compute names to keep:
    # 1) any entity name directly mentioned
    keep_names_lower: Set[str] = set(n for n in mentioned)

    # 2) unambiguous two-word Person target for a mentioned one-word Person name
    for token in one_word_person_names:
        if token in mentioned:
            candidates = token_to_two_word.get(token, set())
            if len(candidates) == 1:
                (only_full_name,) = tuple(candidates)
                keep_names_lower.add(norm(only_full_name))

    # --------------------------
    # Pre-index the raw inputs (after pruning)
    # --------------------------
    seen_pairs: Set[Tuple[str, str]] = set()
    entities_input: List[Tuple[str, str]] = []
    for name, etype in all_entities:
        if norm(name) not in keep_names_lower:
            continue  # pruned up front
        key = (norm(name), etype)
        if key not in seen_pairs:
            seen_pairs.add(key)
            entities_input.append((name, etype))

    # Early exit if nothing remains
    if not entities_input:
        return []

    # name_lower -> set(types) for canonical type selection
    name2types: Dict[str, Set[str]] = defaultdict(set)
    for name, etype in entities_input:
        name2types[norm(name)].add(etype)

    # --------------------------
    # Stage 1: Person-only, rule-based
    # --------------------------
    person_one_word: Set[str] = set()
    person_two_word_verbatim: Set[str] = set()
    for name, etype in entities_input:
        if etype == "Person":
            wc = word_count(name)
            if wc == 1:
                person_one_word.add(norm(name))
            elif wc == 2:
                person_two_word_verbatim.add(name)

    token_to_two_word_stage1: Dict[str, Set[str]] = defaultdict(set)
    for full in person_two_word_verbatim:
        tokens = [norm(tok) for tok in full.split()]
        for t in tokens:
            token_to_two_word_stage1[t].add(full)

    canonical_target_for_name: Dict[str, str] = {}
    for name, _ in entities_input:
        canonical_target_for_name[norm(name)] = name

    for token in person_one_word:
        candidates = token_to_two_word_stage1.get(token, set())
        if len(candidates) == 1:
            (only_full_name,) = tuple(candidates)
            canonical_target_for_name[token] = only_full_name

    # --------------------------
    # Create Entities from stage 1 results
    # --------------------------
    entities_by_key: Dict[str, Entity] = {}
    alias_index: Dict[str, str] = {}  # lower(name or alias) -> canonical_key

    def pick_canonical_type(canonical_name: str, fallback_type: str) -> str:
        types = name2types.get(norm(canonical_name))
        if types:
            if "Person" in types:
                return "Person"
            return next(iter(types))
        return fallback_type

    for name, etype in entities_input:
        canonical_name = canonical_target_for_name.get(norm(name), name)
        ckey = norm(canonical_name)
        if ckey not in entities_by_key:
            entities_by_key[ckey] = Entity(
                name=canonical_name,
                type=pick_canonical_type(canonical_name, etype),
                aliases=[],
            )
            alias_index[ckey] = ckey

        if norm(name) != ckey:
            ent = entities_by_key[ckey]
            if name not in ent.aliases and name != ent.name:
                ent.aliases.append(name)
            alias_index[norm(name)] = ckey

        alias_index[norm(canonical_name)] = ckey

    # --------------------------
    # Stage 2: Triplet-driven alias handling
    # --------------------------
    def resolve_key(name: str) -> str:
        return alias_index.get(norm(name), "")

    def add_alias_to_key(ckey: str, alias_str: str) -> None:
        if not alias_str:
            return
        alias_l = norm(alias_str)
        if not ckey or alias_l in alias_index:
            return
        ent = entities_by_key.get(ckey)
        if not ent:
            return
        if alias_str != ent.name and alias_str not in ent.aliases:
            ent.aliases.append(alias_str)
        alias_index[alias_l] = ckey

    def merge_entities(ckey_a: str, ckey_b: str) -> str:
        if ckey_a == ckey_b:
            return ckey_a
        ent_a = entities_by_key[ckey_a]
        ent_b = entities_by_key[ckey_b]

        def is_two_word_person(e: Entity) -> bool:
            return e.type == "Person" and word_count(e.name) == 2

        choose_a = False
        choose_b = False

        if is_two_word_person(ent_a) and not is_two_word_person(ent_b):
            choose_a = True
        elif is_two_word_person(ent_b) and not is_two_word_person(ent_a):
            choose_b = True
        elif ent_a.type == "Person" and ent_b.type != "Person":
            choose_a = True
        elif ent_b.type == "Person" and ent_a.type != "Person":
            choose_b = True
        else:
            w_a, w_b = word_count(ent_a.name), word_count(ent_b.name)
            if w_a > w_b:
                choose_a = True
            elif w_b > w_a:
                choose_b = True
            else:
                if len(ent_a.name) > len(ent_b.name):
                    choose_a = True
                elif len(ent_b.name) > len(ent_a.name):
                    choose_b = True
                else:
                    choose_a = True

        primary_key, secondary_key = (ckey_a, ckey_b) if choose_a else (ckey_b, ckey_a)
        primary = entities_by_key[primary_key]
        secondary = entities_by_key[secondary_key]

        for candidate in [secondary.name, *secondary.aliases]:
            if candidate != primary.name and candidate not in primary.aliases:
                primary.aliases.append(candidate)
            alias_index[norm(candidate)] = primary_key

        del entities_by_key[secondary_key]
        return primary_key

    for tr in triplets:
        if not is_alias_rel(tr.relationship):
            continue

        s_key = resolve_key(tr.subject)
        o_key = resolve_key(tr.object)

        if s_key and o_key:
            if s_key != o_key:
                primary_key = merge_entities(s_key, o_key)
                alias_index[norm(tr.subject)] = primary_key
                alias_index[norm(tr.object)] = primary_key
        elif s_key and not o_key:
            add_alias_to_key(s_key, tr.object)
        elif o_key and not s_key:
            add_alias_to_key(o_key, tr.subject)
        else:
            continue

    # Deduplicate alias lists (preserve order) and drop any alias equal to the canonical
    for ent in entities_by_key.values():
        seen_aliases: Set[str] = set()
        unique_aliases: List[str] = []
        ent_name_l = norm(ent.name)
        for a in ent.aliases:
            al = norm(a)
            if al == ent_name_l:
                continue
            if al not in seen_aliases:
                seen_aliases.add(al)
                unique_aliases.append(a)
        ent.aliases = unique_aliases

    # Stable order
    return sorted(entities_by_key.values(), key=lambda e: (e.type, e.name.lower()))


def resolve_aliases(
    triplets: List[ExtractedKnowledgeTriplet],
    entities: List[Entity],
) -> List[ExtractedKnowledgeTriplet]:
    """
    Replace subject/object mentions with their canonical entity names using `entities`.
    Drop any triplet whose relationship is an alias-like relation *and* resolves to the
    same canonical on both sides (i.e., it's redundant after resolution).
    """

    def norm(s: str) -> str:
        return s.strip().lower()

    def is_alias_rel(rel: str) -> bool:
        r = norm(rel)
        if "alias" in r:
            return True
        if "also known as" in r:
            return True
        return r in {"aka", "a.k.a.", "a.k.a", "same as", "synonym", "nickname"}

    # Build alias -> canonical mapping (case-insensitive), preserving canonical casing.
    alias_to_canonical: Dict[str, str] = {}
    for ent in entities:
        alias_to_canonical[norm(ent.name)] = ent.name
        for a in ent.aliases:
            alias_to_canonical[norm(a)] = ent.name

    def resolve(name: str) -> str:
        return alias_to_canonical.get(norm(name), name)

    resolved: List[ExtractedKnowledgeTriplet] = []
    for tr in triplets:
        s_new = resolve(tr.subject)
        o_new = resolve(tr.object)

        # Drop alias triplets that collapse to the same canonical name.
        if is_alias_rel(tr.relationship) and norm(s_new) == norm(o_new):
            continue

        if s_new != tr.subject or o_new != tr.object:
            resolved.append(
                ExtractedKnowledgeTriplet(
                    subject=s_new,
                    relationship=tr.relationship,
                    object=o_new,
                    chunk_id=tr.chunk_id,
                    rank=tr.rank,
                )
            )
        else:
            # No change needed; keep original
            resolved.append(tr)

    return resolved

from typing import List, Tuple, Dict

def deduplicate_triplets(triplets: List[ExtractedKnowledgeTriplet]) -> List[ExtractedKnowledgeTriplet]:
    """
    Remove exact duplicate triplets (same subject, relationship, object).
    Keep the first occurrence and increase its `rank` by 1 for every additional duplicate.
    Order of first occurrences is preserved.
    """
    by_key: Dict[Tuple[str, str, str], ExtractedKnowledgeTriplet] = {}
    order: List[Tuple[str, str, str]] = []

    for tr in triplets:
        key = (tr.subject, tr.relationship, tr.object)
        if key in by_key:
            # Per spec: increase by exactly 1 per duplicate
            by_key[key].rank += 1
        else:
            # Keep the first occurrence as the canonical one
            by_key[key] = ExtractedKnowledgeTriplet(
                subject=tr.subject,
                relationship=tr.relationship,
                object=tr.object,
                chunk_id=tr.chunk_id,
                rank=tr.rank,
            )
            order.append(key)

    return [by_key[k] for k in order]
