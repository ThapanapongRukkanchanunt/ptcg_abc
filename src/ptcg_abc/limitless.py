from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from lxml import html

from ptcg_abc.config import (
    LIMITLESS_BASE_URL,
    LIMITLESS_FORMAT,
    LIMITLESS_RAW_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from ptcg_abc.http_client import fetch_text_with_cache
from ptcg_abc.models import Archetype, CardLine, CollectionResult, Decklist, TournamentResult, Variant
from ptcg_abc.normalize import clean_text, deck_fingerprint, normalize_card_name, slugify


DECK_LINK_RE = re.compile(r"^/decks/(\d+)$")
DECKLIST_LINK_RE = re.compile(r"^/decks/list/\d+$")


def _doc(html_text: str) -> html.HtmlElement:
    return html.fromstring(html_text)


def _text(node: html.HtmlElement) -> str:
    return clean_text(node.text_content())


def _absolute_url(path: str) -> str:
    return urljoin(LIMITLESS_BASE_URL, path)


def with_query_params(url: str, **params: str | None) -> str:
    split = urlsplit(url)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    for key, value in params.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = value
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def _parse_int(value: str) -> int | None:
    match = re.search(r"\d[\d,]*", value)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def _parse_placement(value: str) -> int:
    return _parse_int(value) or 999999


def parse_metagame(html_text: str, *, limit: int = 10) -> list[Archetype]:
    doc = _doc(html_text)
    archetypes: list[Archetype] = []
    seen_ids: set[str] = set()

    for row in doc.xpath("//table[contains(@class, 'data-table')]//tr[td]"):
        link = None
        deck_id = None
        for anchor in row.xpath(".//a[@href]"):
            href = anchor.get("href", "")
            match = DECK_LINK_RE.match(href)
            if match:
                link = anchor
                deck_id = match.group(1)
                break
        if link is None or deck_id is None or deck_id in seen_ids:
            continue

        cells = [_text(cell) for cell in row.xpath("./td")]
        rank = _parse_int(cells[0]) if cells else None
        points = None
        share = None
        for cell in reversed(cells):
            if "%" in cell and share is None:
                share = cell
                continue
            parsed = _parse_int(cell)
            if parsed is not None and points is None and cell != str(rank):
                points = parsed
                break

        seen_ids.add(deck_id)
        archetypes.append(
            Archetype(
                rank=rank or len(archetypes) + 1,
                name=_text(link),
                deck_id=deck_id,
                points=points,
                share=share,
                source_url=_absolute_url(f"/decks/{deck_id}"),
            )
        )
        if len(archetypes) >= limit:
            break

    if not archetypes:
        raise RuntimeError("Could not parse any Limitless archetypes from the metagame page.")
    return archetypes


def parse_variants(html_text: str, archetype: Archetype) -> list[Variant]:
    doc = _doc(html_text)
    options = doc.xpath("//select[@id='variant-select']/option")
    variants: list[Variant] = []
    for option in options:
        value = option.get("value")
        name = _text(option)
        if value == "null" or name.casefold() == "all":
            continue
        source_url = with_query_params(
            _absolute_url(f"/decks/{archetype.deck_id}/results"),
            format=LIMITLESS_FORMAT,
        )
        if value:
            source_url = with_query_params(source_url, variant=value)
        variants.append(Variant(name=name, value=value, source_url=source_url))

    if variants:
        return variants
    return [
        Variant(
            name="All",
            value=None,
            source_url=with_query_params(
                _absolute_url(f"/decks/{archetype.deck_id}/results"),
                format=LIMITLESS_FORMAT,
            ),
        )
    ]


def parse_results(html_text: str, *, source_url: str) -> list[TournamentResult]:
    doc = _doc(html_text)
    rows = doc.xpath("//table[contains(@class, 'data-table')]//tr")
    results: list[TournamentResult] = []
    current_event_name = ""
    current_event_date = ""

    for row in rows:
        heading = row.xpath("./th[contains(@class, 'sub-heading')]")
        if heading:
            heading_text = _text(heading[0])
            if " - " in heading_text:
                current_event_date, current_event_name = heading_text.split(" - ", 1)
            else:
                current_event_name = heading_text
                current_event_date = ""
            continue

        cells = row.xpath("./td")
        if len(cells) < 5:
            continue
        link_nodes = row.xpath(".//a[@href]")
        decklist_url = ""
        for link in link_nodes:
            href = link.get("href", "")
            if DECKLIST_LINK_RE.match(href):
                decklist_url = _absolute_url(href)
                break
        if not decklist_url:
            continue

        player_links = [link for link in link_nodes if link.get("href", "").startswith("/players/")]
        player = _text(player_links[0]) if player_links else _text(cells[3])
        placement = _text(cells[1])
        results.append(
            TournamentResult(
                event_name=current_event_name,
                event_date=current_event_date,
                placement=placement,
                placement_rank=_parse_placement(placement),
                player=player,
                decklist_url=decklist_url,
                source_url=source_url,
                page_order=len(results),
            )
        )

    return sorted(results, key=lambda result: (result.placement_rank, result.page_order))


def parse_decklist(
    html_text: str,
    *,
    archetype: Archetype,
    variant: Variant,
    result: TournamentResult,
) -> Decklist:
    doc = _doc(html_text)
    title_nodes = doc.xpath("//div[contains(@class, 'decklist-title')]")
    title = clean_text(title_nodes[0].text or "") if title_nodes else f"{archetype.name} {variant.name}"
    cards: list[CardLine] = []
    for column in doc.xpath("//div[@data-text-decklist]//div[contains(@class, 'decklist-column')]"):
        heading_nodes = column.xpath("./div[contains(@class, 'decklist-column-heading')]")
        section = _text(heading_nodes[0]) if heading_nodes else ""
        section = re.sub(r"\s*\(\d+\)\s*$", "", section)
        for card in column.xpath(".//div[contains(@class, 'decklist-card')]"):
            count_nodes = card.xpath(".//span[contains(@class, 'card-count')]")
            name_nodes = card.xpath(".//span[contains(@class, 'card-name')]")
            if not count_nodes or not name_nodes:
                continue
            count = _parse_int(_text(count_nodes[0]))
            name = _text(name_nodes[0])
            if count and name:
                cards.append(CardLine(count=count, name=name, section=section))

    total = sum(card.count for card in cards)
    return Decklist(
        archetype=archetype,
        variant=variant,
        result=result,
        title=title,
        cards=cards,
        total_cards=total,
        fingerprint=deck_fingerprint(cards),
        source_url=result.decklist_url,
    )


def collect_limitless_decks(
    *,
    snapshot_date: str | None = None,
    top_archetypes: int = 10,
    lists_per_variant: int = 2,
    refresh: bool = False,
    candidate_limit: int = 250,
    delay_seconds: float = 0.2,
    limitless_format: str = LIMITLESS_FORMAT,
) -> CollectionResult:
    snapshot_date = snapshot_date or date.today().isoformat()
    cache_dir = LIMITLESS_RAW_DIR / snapshot_date
    result = CollectionResult()
    seen_fingerprints: set[str] = set()

    metagame_url = with_query_params(f"{LIMITLESS_BASE_URL}/decks", format=limitless_format)
    metagame_html = fetch_text_with_cache(metagame_url, cache_dir, refresh=refresh)
    archetypes = parse_metagame(metagame_html, limit=top_archetypes)

    for archetype in archetypes:
        overview_url = with_query_params(archetype.source_url, format=limitless_format)
        overview_html = fetch_text_with_cache(overview_url, cache_dir, refresh=refresh)
        variants = parse_variants(overview_html, archetype)
        for variant in variants:
            variant = Variant(
                name=variant.name,
                value=variant.value,
                source_url=with_query_params(variant.source_url, format=limitless_format),
            )
            results_html = fetch_text_with_cache(variant.source_url, cache_dir, refresh=refresh)
            candidates = parse_results(results_html, source_url=variant.source_url)
            accepted_for_variant = 0
            for candidate in candidates[:candidate_limit]:
                if accepted_for_variant >= lists_per_variant:
                    break
                time.sleep(delay_seconds)
                try:
                    deck_html = fetch_text_with_cache(candidate.decklist_url, cache_dir, refresh=refresh)
                    deck = parse_decklist(
                        deck_html, archetype=archetype, variant=variant, result=candidate
                    )
                except Exception as exc:
                    result.add_skip(
                        "decklist_parse_error",
                        archetype=archetype.name,
                        variant=variant.name,
                        decklist_url=candidate.decklist_url,
                        error=str(exc),
                    )
                    continue

                if deck.total_cards != 60:
                    result.add_skip(
                        "not_60_cards",
                        archetype=archetype.name,
                        variant=variant.name,
                        decklist_url=candidate.decklist_url,
                        total_cards=deck.total_cards,
                    )
                    continue
                if deck.fingerprint in seen_fingerprints:
                    result.add_skip(
                        "duplicate_60_card_fingerprint",
                        archetype=archetype.name,
                        variant=variant.name,
                        decklist_url=candidate.decklist_url,
                    )
                    continue

                seen_fingerprints.add(deck.fingerprint)
                result.decks.append(deck)
                accepted_for_variant += 1

            if accepted_for_variant < lists_per_variant:
                result.add_skip(
                    "variant_shortfall",
                    archetype=archetype.name,
                    variant=variant.name,
                    accepted=accepted_for_variant,
                    requested=lists_per_variant,
                    candidate_count=len(candidates),
                )

    return result


def write_deck_collection(collection: CollectionResult, *, snapshot_date: str) -> Path:
    output_dir = PROCESSED_DIR / snapshot_date
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "limitless_decks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for deck in collection.decks:
            handle.write(json.dumps(deck.to_dict(), ensure_ascii=False) + "\n")
    manifest = {
        "snapshot_date": snapshot_date,
        "deck_count": len(collection.decks),
        "skip_count": len(collection.skips),
        "skips": collection.skips,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonl_path


def write_missing_report(
    collection: CollectionResult,
    legal_cards: dict[str, str],
    output_path: Path | None = None,
    limitless_format: str = LIMITLESS_FORMAT,
) -> Path:
    output_path = output_path or REPORTS_DIR / "missing_limitless_cards.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usage: dict[str, list[Decklist]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for deck in collection.decks:
        for card in deck.cards:
            normalized = normalize_card_name(card.name)
            display_names.setdefault(normalized, card.name)
            usage[normalized].append(deck)

    missing = sorted(
        normalized for normalized in usage.keys() if normalized not in legal_cards
    )
    lines = [
        "# Missing Limitless Cards",
        "",
        f"Limitless format: {limitless_format}",
        f"Selected Limitless decks: {len(collection.decks)}",
        f"Unique Limitless card names: {len(usage)}",
        f"Missing from Kaggle legal list: {len(missing)}",
        "",
    ]
    if missing:
        lines.extend(
            [
                "| Limitless card | Deck appearances | Example source | Alternative to use |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for normalized in missing:
            decks = usage[normalized]
            example = decks[0]
            label = (
                f"{example.archetype.name} / {example.variant.name} / "
                f"{example.result.player} {example.result.placement}"
            )
            lines.append(
                f"| {display_names[normalized]} | {len(decks)} | "
                f"[{label}]({example.source_url}) |  |"
            )
    else:
        lines.append("No missing Limitless card names found.")

    if collection.skips:
        lines.extend(["", "## Collection Notes", ""])
        for skip in collection.skips[:100]:
            lines.append(f"- {skip}")
        if len(collection.skips) > 100:
            lines.append(f"- ... {len(collection.skips) - 100} more skip records omitted.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def deck_collection_summary(collection: CollectionResult) -> str:
    archetypes = {deck.archetype.name for deck in collection.decks}
    variants = {(deck.archetype.name, deck.variant.name) for deck in collection.decks}
    return (
        f"accepted_decks={len(collection.decks)} "
        f"archetypes={len(archetypes)} variants={len(variants)} skips={len(collection.skips)}"
    )
