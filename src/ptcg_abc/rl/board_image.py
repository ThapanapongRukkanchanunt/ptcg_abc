from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import (
    _card_id,
    _card_name,
    _card_type_name,
    _get,
)
from ptcg_abc.rl.card_art import CardArtCache


CANVAS_SIZE = (1024, 1024)
CARD_ASPECT_RATIO = 660 / 920
ATTACHMENT_TAB_HEIGHT = 16
ATTACHMENT_TAB_WIDTH = 16
BACKGROUND = (183, 255, 96)
LINE = (12, 18, 14)
TEXT = (16, 22, 20)
MUTED = (70, 88, 76)
HIDDEN = (64, 88, 132)

ENERGY_COLORS = {
    1: (94, 178, 96),
    2: (238, 92, 72),
    3: (84, 150, 219),
    4: (248, 216, 72),
    5: (142, 92, 196),
    6: (180, 118, 64),
    7: (156, 168, 178),
    8: (226, 128, 190),
    9: (72, 74, 78),
    10: (210, 214, 216),
}


@dataclass(frozen=True)
class SnapshotImage:
    path: str
    step: int
    player_index: int
    label: str
    select_type: str
    context: str
    selected_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotManifest:
    output_dir: str
    images: list[SnapshotImage]
    battle_result: dict[str, Any]
    our_deck: str
    benchmark_deck: str
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "images": [image.to_dict() for image in self.images],
            "battle_result": self.battle_result,
            "our_deck": self.our_deck,
            "benchmark_deck": self.benchmark_deck,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class CardConditions:
    poisoned: bool = False
    burned: bool = False
    asleep: bool = False
    paralyzed: bool = False
    confused: bool = False


def render_tabletop_snapshot(
    observation: Any,
    *,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None = None,
    selected_indices: Sequence[int] = (),
    title: str = "",
    output_path: Path,
) -> None:
    Image, ImageDraw, ImageFont = _pil()
    image = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    fonts = _fonts(ImageFont)

    current = _get(observation, "current")
    actor = int(_get(current, "yourIndex", 0) or 0)
    players = list(_get(current, "players", []) or [])
    mine = players[actor] if 0 <= actor < len(players) else None
    opponent_index = 1 - actor
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else None

    _draw_board_lines(draw)
    _draw_center(draw)
    _draw_zone_labels(draw, fonts)

    _draw_side(
        image,
        draw,
        mine,
        card_by_id=card_by_id,
        card_art=card_art,
        fonts=fonts,
        bottom=True,
        hand_face_up=True,
    )
    _draw_side(
        image,
        draw,
        opponent,
        card_by_id=card_by_id,
        card_art=card_art,
        fonts=fonts,
        bottom=False,
        hand_face_up=False,
    )
    _draw_stadium(image, draw, current, card_by_id, card_art, fonts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _pil() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to render board snapshots.") from exc
    return Image, ImageDraw, ImageFont


def _fonts(ImageFont: Any) -> dict[str, Any]:
    try:
        return {
            "title": ImageFont.truetype("arial.ttf", 30),
            "large": ImageFont.truetype("arial.ttf", 24),
            "body": ImageFont.truetype("arial.ttf", 18),
            "small": ImageFont.truetype("arial.ttf", 14),
            "tiny": ImageFont.truetype("arial.ttf", 12),
        }
    except OSError:
        default = ImageFont.load_default()
        return {name: default for name in ("title", "large", "body", "small", "tiny")}


def _card_size(width: int) -> tuple[int, int]:
    return width, int(round(width / CARD_ASPECT_RATIO))


def _hidden_hand_layout(count: int) -> tuple[int, int, int]:
    if count <= 0:
        card_w, card_h = _card_size(92)
        return card_w, card_h, 6
    max_width = 820
    gap = 6 if count <= 10 else 4
    if count == 1:
        width = 92
    else:
        width = min(92, max(22, int((max_width - gap * (count - 1)) / count)))
    card_w, card_h = _card_size(width)
    return card_w, card_h, gap


def _card_rect(x: int, y: int, width: int) -> tuple[int, int, int, int]:
    card_w, card_h = _card_size(width)
    return (x, y, x + card_w, y + card_h)


def _card_rect_center(cx: int, cy: int, width: int) -> tuple[int, int, int, int]:
    card_w, card_h = _card_size(width)
    left = cx - card_w // 2
    top = cy - card_h // 2
    return (left, top, left + card_w, top + card_h)


def _draw_board_lines(draw: Any) -> None:
    return


def _draw_center(draw: Any) -> None:
    return


def _draw_zone_labels(draw: Any, fonts: dict[str, Any]) -> None:
    return


def _draw_side(
    canvas: Any,
    draw: Any,
    player: Any,
    *,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    bottom: bool,
    hand_face_up: bool,
) -> None:
    active = _first(list(_get(player, "active", []) or []))
    bench = [card for card in list(_get(player, "bench", []) or []) if card is not None]
    hand = list(_get(player, "hand", []) or [])
    hand_count_value = _get(player, "handCount", None)
    hand_count = int(hand_count_value) if hand_count_value is not None else len(hand)
    discard = list(_get(player, "discard", []) or [])
    prize_count = len(list(_get(player, "prize", []) or []))
    deck_count = int(_get(player, "deckCount", 0) or 0)

    if bottom:
        _draw_card(
            canvas,
            draw,
            _card_rect_center(512, 610, 118),
            active,
            card_by_id,
            card_art,
            fonts,
            face_up=True,
            conditions=_active_conditions(player, active),
            attachment_direction="right",
        )
        _draw_card_row(
            canvas,
            draw,
            bench,
            (225, 742),
            card_by_id,
            card_art,
            fonts,
            face_up=True,
            max_cards=5,
            card_width=96,
            gap=22,
        )
        _draw_hand(
            canvas,
            draw,
            hand,
            (205, 888),
            card_by_id,
            card_art,
            fonts,
            face_up=hand_face_up,
            hidden_count=hand_count,
            card_width=74,
            gap=6,
            max_cards=10,
        )
        _draw_prize_backs(draw, (0, 454), prize_count, fonts)
        _draw_count_badge(draw, (820, 600), "Deck", deck_count, fonts)
        _draw_count_badge(draw, (798, 720), "Discard", len(discard), fonts)
    else:
        _draw_card(
            canvas,
            draw,
            _card_rect_center(512, 404, 118),
            active,
            card_by_id,
            card_art,
            fonts,
            face_up=True,
            conditions=_active_conditions(player, active),
            attachment_direction="right",
        )
        _draw_card_row(
            canvas,
            draw,
            bench,
            (225, 158),
            card_by_id,
            card_art,
            fonts,
            face_up=True,
            max_cards=5,
            card_width=96,
            gap=22,
        )
        _draw_hand(
            canvas,
            draw,
            hand,
            (0, 0),
            card_by_id,
            card_art,
            fonts,
            face_up=False,
            hidden_count=hand_count,
        )
        _draw_prize_backs(draw, (816, 144), prize_count, fonts)
        _draw_count_badge(draw, (52, 158), "Deck", deck_count, fonts)
        _draw_count_badge(draw, (30, 276), "Discard", len(discard), fonts)


def _draw_hand(
    canvas: Any,
    draw: Any,
    hand: list[Any],
    origin: tuple[int, int],
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    *,
    face_up: bool,
    hidden_count: int | None = None,
    card_width: int = 74,
    gap: int = 6,
    max_cards: int = 10,
) -> None:
    x, y = origin
    card_w, card_h = _card_size(card_width)
    shown = hand[:max_cards]
    if not face_up:
        count = max(0, hidden_count if hidden_count is not None else len(hand))
        card_w, card_h, gap = _hidden_hand_layout(count)
        for index in range(count):
            _draw_card_back(
                draw,
                (x + index * (card_w + gap), y, x + index * (card_w + gap) + card_w, y + card_h),
                fonts,
            )
        return
    for index, card in enumerate(shown):
        left = x + index * (card_w + gap)
        _draw_card(
            canvas,
            draw,
            (left, y, left + card_w, y + card_h),
            card,
            card_by_id,
            card_art,
            fonts,
            face_up=True,
        )
    if len(hand) > len(shown):
        draw.text((x + len(shown) * (card_w + gap), y + 45), f"+{len(hand) - len(shown)}", font=fonts["large"], fill=TEXT)


def _draw_card_row(
    canvas: Any,
    draw: Any,
    cards: list[Any],
    origin: tuple[int, int],
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    *,
    face_up: bool,
    max_cards: int,
    card_width: int = 104,
    gap: int = 34,
) -> None:
    x, y = origin
    card_w, card_h = _card_size(card_width)
    for index in range(max_cards):
        left = x + index * (card_w + gap)
        rect = (left, y, left + card_w, y + card_h)
        if index < len(cards):
            _draw_card(
                canvas,
                draw,
                rect,
                cards[index],
                card_by_id,
                card_art,
                fonts,
                face_up=face_up,
            )
        else:
            draw.rounded_rectangle(rect, radius=8, outline=(224, 238, 218), width=3)


def _draw_card(
    canvas: Any,
    draw: Any,
    rect: tuple[int, int, int, int],
    card: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    *,
    face_up: bool,
    conditions: CardConditions | None = None,
    attachment_direction: str = "up",
) -> None:
    if card is None:
        draw.rounded_rectangle(rect, radius=12, outline=LINE, width=4)
        return
    if not face_up:
        _draw_card_back(draw, rect, fonts)
        return

    Image, _, _ = _pil()
    card_id = _card_id(card)
    data = card_by_id.get(card_id) if card_id is not None else None
    conditions = conditions or _card_conditions(card)
    surface, main_rect = _make_card_group_surface(
        card,
        data,
        card_by_id,
        card_art,
        fonts,
        rect,
        conditions,
        attachment_direction=attachment_direction,
    )
    rotation = _condition_rotation(conditions)
    if rotation:
        surface = surface.rotate(rotation, expand=True, resample=_resample_bicubic(Image))

    left, top, right, bottom = rect
    if rotation:
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        paste_left = center_x - surface.width // 2
        paste_top = center_y - surface.height // 2
    else:
        paste_left = left - main_rect[0]
        paste_top = top - main_rect[1]
    canvas.paste(surface, (paste_left, paste_top), surface)


def _make_card_group_surface(
    card: Any,
    data: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    rect: tuple[int, int, int, int],
    conditions: CardConditions,
    *,
    attachment_direction: str,
) -> tuple[Any, tuple[int, int, int, int]]:
    width = max(1, rect[2] - rect[0])
    height = max(1, rect[3] - rect[1])
    attachments = _ordered_attachments(card, card_by_id)
    if attachment_direction == "left":
        return _make_left_attachment_group(
            card,
            data,
            card_by_id,
            card_art,
            fonts,
            rect,
            conditions,
            width,
            height,
            attachments,
        )
    if attachment_direction == "right":
        return _make_right_attachment_group(
            card,
            data,
            card_by_id,
            card_art,
            fonts,
            rect,
            conditions,
            width,
            height,
            attachments,
        )
    return _make_up_attachment_group(
        card,
        data,
        card_by_id,
        card_art,
        fonts,
        rect,
        conditions,
        width,
        height,
        attachments,
    )


def _make_up_attachment_group(
    card: Any,
    data: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    rect: tuple[int, int, int, int],
    conditions: CardConditions,
    width: int,
    height: int,
    attachments: list[Any],
) -> tuple[Any, tuple[int, int, int, int]]:
    Image, ImageDraw, _ = _pil()
    tab_count = min(len(attachments), _max_vertical_attachment_tabs(height))
    main_y = tab_count * ATTACHMENT_TAB_HEIGHT
    group = Image.new("RGBA", (width, height + main_y), (0, 0, 0, 0))
    visible_attachments = attachments[:tab_count]
    for index, attachment in enumerate(visible_attachments):
        attachment_data = card_by_id.get(_card_id(attachment))
        y = index * ATTACHMENT_TAB_HEIGHT
        surface = _make_face_up_card_surface(
            attachment,
            attachment_data,
            card_by_id,
            card_art,
            fonts,
            (0, 0, width, height),
        )
        group.alpha_composite(surface, (0, y))

    main_surface = _make_face_up_card_surface(card, data, card_by_id, card_art, fonts, rect)
    group.alpha_composite(main_surface, (0, main_y))
    draw = ImageDraw.Draw(group)
    main_rect = (0, main_y, width, main_y + height)
    _draw_card_markers(draw, main_rect, card, data, conditions, fonts)
    if len(attachments) > tab_count:
        draw.text(
            (width - 8, main_y - 3),
            f"+{len(attachments) - tab_count}",
            font=fonts["tiny"],
            fill=(255, 255, 255),
            anchor="ra",
            stroke_width=2,
            stroke_fill=(34, 34, 34),
        )
    return group, main_rect


def _make_left_attachment_group(
    card: Any,
    data: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    rect: tuple[int, int, int, int],
    conditions: CardConditions,
    width: int,
    height: int,
    attachments: list[Any],
) -> tuple[Any, tuple[int, int, int, int]]:
    Image, ImageDraw, _ = _pil()
    tab_count = min(len(attachments), _max_horizontal_attachment_tabs(width))
    main_x = tab_count * ATTACHMENT_TAB_WIDTH
    group = Image.new("RGBA", (width + main_x, height), (0, 0, 0, 0))

    visible_attachments = attachments[:tab_count]
    for index, attachment in enumerate(visible_attachments):
        attachment_data = card_by_id.get(_card_id(attachment))
        x = index * ATTACHMENT_TAB_WIDTH
        surface = _make_face_up_card_surface(
            attachment,
            attachment_data,
            card_by_id,
            card_art,
            fonts,
            (0, 0, width, height),
        )
        group.alpha_composite(surface, (x, 0))

    main_surface = _make_face_up_card_surface(card, data, card_by_id, card_art, fonts, rect)
    group.alpha_composite(main_surface, (main_x, 0))
    draw = ImageDraw.Draw(group)
    main_rect = (main_x, 0, main_x + width, height)
    _draw_card_markers(draw, main_rect, card, data, conditions, fonts)
    if len(attachments) > tab_count:
        draw.text(
            (main_x - 3, height - 8),
            f"+{len(attachments) - tab_count}",
            font=fonts["tiny"],
            fill=(255, 255, 255),
            anchor="ra",
            stroke_width=2,
            stroke_fill=(34, 34, 34),
        )
    return group, main_rect


def _make_right_attachment_group(
    card: Any,
    data: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    rect: tuple[int, int, int, int],
    conditions: CardConditions,
    width: int,
    height: int,
    attachments: list[Any],
) -> tuple[Any, tuple[int, int, int, int]]:
    Image, ImageDraw, _ = _pil()
    tab_count = min(len(attachments), _max_horizontal_attachment_tabs(width))
    group = Image.new("RGBA", (width + tab_count * ATTACHMENT_TAB_WIDTH, height), (0, 0, 0, 0))

    visible_attachments = attachments[:tab_count]
    for index, attachment in reversed(list(enumerate(visible_attachments))):
        attachment_data = card_by_id.get(_card_id(attachment))
        x = (index + 1) * ATTACHMENT_TAB_WIDTH
        surface = _make_face_up_card_surface(
            attachment,
            attachment_data,
            card_by_id,
            card_art,
            fonts,
            (0, 0, width, height),
        )
        group.alpha_composite(surface, (x, 0))

    main_surface = _make_face_up_card_surface(card, data, card_by_id, card_art, fonts, rect)
    group.alpha_composite(main_surface, (0, 0))
    draw = ImageDraw.Draw(group)
    main_rect = (0, 0, width, height)
    _draw_card_markers(draw, main_rect, card, data, conditions, fonts)
    if len(attachments) > tab_count:
        draw.text(
            (width + tab_count * ATTACHMENT_TAB_WIDTH - 3, height - 8),
            f"+{len(attachments) - tab_count}",
            font=fonts["tiny"],
            fill=(255, 255, 255),
            anchor="ra",
            stroke_width=2,
            stroke_fill=(34, 34, 34),
        )
    return group, main_rect


def _make_face_up_card_surface(
    card: Any,
    data: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
    rect: tuple[int, int, int, int],
) -> Any:
    Image, ImageDraw, _ = _pil()
    width = max(1, rect[2] - rect[0])
    height = max(1, rect[3] - rect[1])
    surface = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    surface_draw = ImageDraw.Draw(surface)

    card_id = _card_id(card)
    art_path = card_art.get(card_id) if card_art is not None else None
    if art_path is not None:
        try:
            from PIL import ImageOps

            with Image.open(art_path) as art:
                resized = ImageOps.fit(
                    art.convert("RGBA"),
                    (width, height),
                    method=_resample_lanczos(Image),
                )
                surface.alpha_composite(resized)
            surface_draw.rounded_rectangle(
                (0, 0, width - 1, height - 1),
                radius=max(5, min(width, height) // 18),
                outline=(40, 44, 40),
                width=max(2, min(width, height) // 42),
            )
            return surface
        except OSError:
            pass

    _draw_symbolic_card(surface_draw, (0, 0, width, height), card, data, card_by_id, fonts)
    return surface


def _draw_symbolic_card(
    draw: Any,
    rect: tuple[int, int, int, int],
    card: Any,
    data: Any,
    card_by_id: dict[int, Any],
    fonts: dict[str, Any],
) -> None:
    card_id = _card_id(card)
    name = _card_name(card_id, card_by_id) or f"Card {card_id or '?'}"
    color = _card_color(data)
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=8, fill=(240, 244, 238), outline=(76, 82, 76), width=3)
    draw.rounded_rectangle((left + 5, top + 5, right - 5, top + 34), radius=4, fill=color)
    draw.text((left + 9, top + 10), _clip(name, 17), font=fonts["tiny"], fill=(12, 16, 14))

    card_type = _card_type_name(data) if data is not None else ""
    draw.text((left + 9, top + 40), card_type.replace("_", " "), font=fonts["tiny"], fill=MUTED)
    if card_id is not None:
        draw.text((right - 9, top + 40), f"#{card_id}", font=fonts["tiny"], fill=MUTED, anchor="ra")

    hp = int(_get(card, "hp", _get(data, "hp", 0)) or 0)
    max_hp = int(_get(card, "maxHp", _get(data, "hp", hp)) or 0)
    if max_hp > 0:
        ratio = max(0.0, min(1.0, hp / max_hp))
        bar = (left + 9, bottom - 42, right - 9, bottom - 30)
        draw.rounded_rectangle(bar, radius=4, fill=(190, 74, 74))
        draw.rounded_rectangle((bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * ratio), bar[3]), radius=4, fill=(78, 178, 95))
        draw.text((left + 9, bottom - 28), f"HP {hp}/{max_hp}", font=fonts["tiny"], fill=TEXT)

    tags = []
    if bool(_get(data, "megaEx", False)):
        tags.append("MEGA")
    elif bool(_get(data, "ex", False)):
        tags.append("ex")
    if bool(_get(data, "stage2", False)):
        tags.append("S2")
    elif bool(_get(data, "stage1", False)):
        tags.append("S1")
    elif bool(_get(data, "basic", False)):
        tags.append("B")
    if tags:
        draw.text((left + 9, top + 58), " ".join(tags), font=fonts["tiny"], fill=TEXT)

    energies = list(_get(card, "energies", []) or [])
    energy_cards = list(_get(card, "energyCards", []) or [])
    tools = list(_get(card, "tools", []) or [])
    pip_y = bottom - 62
    for index, energy in enumerate((energies or [10] * len(energy_cards))[:8]):
        cx = left + 15 + index * 12
        fill = ENERGY_COLORS.get(int(energy) if str(energy).isdigit() else 10, (218, 218, 218))
        draw.ellipse((cx, pip_y, cx + 9, pip_y + 9), fill=fill, outline=(48, 48, 48))
    if tools:
        draw.text((right - 9, bottom - 62), f"Tool x{len(tools)}", font=fonts["tiny"], fill=TEXT, anchor="ra")


def _draw_card_markers(
    draw: Any,
    rect: tuple[int, int, int, int],
    card: Any,
    data: Any,
    conditions: CardConditions,
    fonts: dict[str, Any],
) -> None:
    hp = int(_get(card, "hp", _get(data, "hp", 0)) or 0)
    max_hp = int(_get(card, "maxHp", _get(data, "hp", hp)) or 0)
    damage = max(0, max_hp - hp) if max_hp > 0 else 0

    markers: list[tuple[str, str, tuple[int, int, int]]] = []
    if damage:
        markers.append(("damage", str(damage), (210, 36, 44)))
    if conditions.burned:
        markers.append(("burn", "B", (240, 126, 28)))
    if conditions.poisoned:
        markers.append(("poison", "P", (126, 54, 176)))
    if not markers:
        return

    left, top, right, bottom = rect
    radius = max(10, min(18, int(min(right - left, bottom - top) * 0.13)))
    gap = max(3, radius // 4)
    cx = right - radius - 5
    start_y = top + radius + 5
    for index, (_, label, fill) in enumerate(markers):
        cy = start_y + index * (radius * 2 + gap)
        if cy + radius > bottom - 4:
            cx -= radius * 2 + gap
            cy = start_y
            start_y += radius * 2 + gap
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=fill,
            outline=(255, 246, 238),
            width=max(2, radius // 5),
        )
        draw.text((cx, cy), label, font=fonts["tiny"], fill=(255, 255, 255), anchor="mm")


def _ordered_attachments(card: Any, card_by_id: dict[int, Any]) -> list[Any]:
    tools = list(_get(card, "tools", []) or [])
    energy_cards = list(_get(card, "energyCards", []) or [])
    special_energy = []
    basic_energy = []
    other_energy = []
    for energy_card in energy_cards:
        card_type = _card_type_name(card_by_id.get(_card_id(energy_card)))
        if card_type == "SPECIAL_ENERGY":
            special_energy.append(energy_card)
        elif card_type == "BASIC_ENERGY":
            basic_energy.append(energy_card)
        else:
            other_energy.append(energy_card)
    return tools + special_energy + other_energy + basic_energy


def _max_vertical_attachment_tabs(card_height: int) -> int:
    return max(0, min(4, (card_height - 32) // ATTACHMENT_TAB_HEIGHT))


def _max_horizontal_attachment_tabs(card_width: int) -> int:
    return max(0, min(4, (card_width - 32) // ATTACHMENT_TAB_WIDTH))


def _active_conditions(player: Any, card: Any) -> CardConditions:
    return CardConditions(
        poisoned=bool(_get(player, "poisoned", False)) or bool(_get(card, "poisoned", False)),
        burned=bool(_get(player, "burned", False)) or bool(_get(card, "burned", False)),
        asleep=bool(_get(player, "asleep", False)) or bool(_get(card, "asleep", False)),
        paralyzed=bool(_get(player, "paralyzed", False)) or bool(_get(card, "paralyzed", False)),
        confused=bool(_get(player, "confused", False)) or bool(_get(card, "confused", False)),
    )


def _card_conditions(card: Any) -> CardConditions:
    return CardConditions(
        poisoned=bool(_get(card, "poisoned", False)),
        burned=bool(_get(card, "burned", False)),
        asleep=bool(_get(card, "asleep", False)),
        paralyzed=bool(_get(card, "paralyzed", False)),
        confused=bool(_get(card, "confused", False)),
    )


def _condition_rotation(conditions: CardConditions) -> int:
    if conditions.confused:
        return 180
    if conditions.paralyzed:
        return -90
    if conditions.asleep:
        return 90
    return 0


def _resample_lanczos(Image: Any) -> Any:
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)


def _resample_bicubic(Image: Any) -> Any:
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3)


def _draw_card_back(draw: Any, rect: tuple[int, int, int, int], fonts: dict[str, Any]) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=8, fill=HIDDEN, outline=(226, 232, 244), width=3)
    draw.ellipse((left + 12, top + 18, right - 12, bottom - 18), outline=(226, 232, 244), width=4)


def _draw_prize_backs(
    draw: Any,
    origin: tuple[int, int],
    count: int,
    fonts: dict[str, Any],
) -> None:
    x, y = origin
    card_w, card_h = _card_size(98)
    gap = 6
    for index in range(min(6, max(0, count))):
        left = x + (index % 2) * (card_w + gap)
        top = y + (index // 2) * (card_h + gap)
        _draw_card_back(draw, (left, top, left + card_w, top + card_h), fonts)


def _draw_count_badge(
    draw: Any,
    origin: tuple[int, int],
    label: str,
    count: int,
    fonts: dict[str, Any],
) -> None:
    x, y = origin
    label_box = (x, y, x + 122, y + 60)
    count_box = (x + 28, y + 54, x + 96, y + 112)
    draw.rounded_rectangle(label_box, radius=14, fill=(250, 250, 244))
    draw.rounded_rectangle(count_box, radius=14, fill=(250, 250, 244))
    draw.text(((label_box[0] + label_box[2]) // 2, y + 31), label, font=fonts["large"], fill=TEXT, anchor="mm")
    draw.text(((count_box[0] + count_box[2]) // 2, y + 84), str(count), font=fonts["large"], fill=TEXT, anchor="mm")


def _draw_stack(
    draw: Any,
    rect: tuple[int, int, int, int],
    label: str,
    count: int,
    fonts: dict[str, Any],
    *,
    face_up: bool,
) -> None:
    if face_up:
        draw.rounded_rectangle(rect, radius=8, fill=(232, 236, 230), outline=(76, 82, 76), width=3)
    else:
        _draw_card_back(draw, rect, fonts)
    draw.text(((rect[0] + rect[2]) // 2, rect[3] + 8), f"{label} {count}", font=fonts["small"], fill=TEXT, anchor="ma")


def _draw_discard(
    canvas: Any,
    draw: Any,
    discard: list[Any],
    rect: tuple[int, int, int, int],
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
) -> None:
    if discard:
        _draw_card(canvas, draw, rect, discard[-1], card_by_id, card_art, fonts, face_up=True)
    else:
        draw.rounded_rectangle(rect, radius=12, outline=LINE, width=4)
        draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), "Empty", font=fonts["tiny"], fill=MUTED, anchor="mm")
    draw.text(((rect[0] + rect[2]) // 2, rect[3] + 8), f"DISCARD {len(discard)}", font=fonts["small"], fill=TEXT, anchor="ma")


def _draw_stadium(
    canvas: Any,
    draw: Any,
    current: Any,
    card_by_id: dict[int, Any],
    card_art: CardArtCache | None,
    fonts: dict[str, Any],
) -> None:
    stadium = _first(list(_get(current, "stadium", []) or []))
    zone = (255, 424, 385, 606)
    if stadium is not None:
        _draw_card(canvas, draw, _card_rect(255, 424, 130), stadium, card_by_id, card_art, fonts, face_up=True)
    else:
        draw.rounded_rectangle(zone, radius=12, outline=LINE, width=4)


def _card_color(card_data: Any) -> tuple[int, int, int]:
    if card_data is None:
        return (212, 214, 216)
    card_type = _card_type_name(card_data)
    if card_type == "POKEMON":
        energy_type = _get(card_data, "energyType")
        try:
            return ENERGY_COLORS.get(int(energy_type), (232, 198, 92))
        except (TypeError, ValueError):
            return (232, 198, 92)
    if "ENERGY" in card_type:
        energy_type = _get(card_data, "energyType")
        try:
            return ENERGY_COLORS.get(int(energy_type), (226, 226, 226))
        except (TypeError, ValueError):
            return (226, 226, 226)
    if card_type == "SUPPORTER":
        return (236, 154, 108)
    if card_type == "STADIUM":
        return (126, 192, 144)
    if card_type in {"ITEM", "TOOL"}:
        return (188, 204, 218)
    return (212, 214, 216)


def _first(values: list[Any]) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


def _clip(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "..."


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "snapshot"


def write_manifest(manifest: SnapshotManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
