"""AFK Journey Guild Management."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import cv2
import numpy as np

from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.exceptions import GameTimeoutError
from adb_auto_player.image_manipulation import Cropping
from adb_auto_player.image_manipulation.color import Color
from adb_auto_player.image_manipulation.io import IO
from adb_auto_player.image_manipulation.scaling import Scaling
from adb_auto_player.models import ConfidenceValue
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point
from adb_auto_player.models.image_manipulation import CropRegions
from adb_auto_player.ocr import OEM, PSM, TesseractBackend, TesseractConfig
from adb_auto_player.template_matching import TemplateMatcher

from ..base import AFKJourneyBase
from ..gui_category import AFKJCategory


@dataclass
class GuildMemberList:
    """Guild Member List."""

    player_name: str | None = None
    power: int | None = None


@dataclass
class ClashfrontsTeams:
    """Clashfronts Teams."""

    is_vanguard: bool
    player_name: str | None = None
    team1_power: str | None = None
    team2_power: str | None = None
    team3_power: str | None = None
    team4_power: str | None = None
    team5_power: str | None = None


@dataclass
class MemberListConfig:
    """Configuration for extracting guild member rows."""

    power_icon_path: str = "guild/power.png"
    match_threshold: float = 0.82
    min_row_gap_px: int = 80
    max_passes: int = 4
    gender_match_threshold: float = 0.80
    gender_templates: dict | None = None

    name_dx_left: int = 20
    name_dy_above: int = 100
    name_width: int = 400
    name_height: int = 60

    power_width: int = 115
    power_height: int = 40
    ocr_backend: TesseractBackend = field(
        default_factory=lambda: TesseractBackend(
            TesseractConfig(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_LINE)
        )
    )

    def __post_init__(self) -> None:
        """Populate gender icon templates when none are provided."""
        if self.gender_templates is None:
            templates: dict[str, dict[str, np.ndarray | int | None]] = {}
            for label, pad in {"male": 8, "female": 10}.items():
                try:
                    img = IO.load_image(Path(f"guild/{label}.png"), grayscale=True)
                except FileNotFoundError:
                    img = None
                templates[label] = {"img": img, "pad": pad}
            self.gender_templates = templates


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    g = Color.to_grayscale(roi_bgr)
    g = Scaling.scale_percent(g, 1.5)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return g


def compute_badge_shift(name_gray: np.ndarray, templates: dict, thr: float) -> int:
    fixed_shift_map = {"male": 36, "female": 48}
    best_label, best_val = None, 0.0

    for label, item in templates.items():
        tpl = item.get("img")
        if tpl is None:
            continue
        res = cv2.matchTemplate(name_gray, tpl, cv2.TM_CCOEFF_NORMED)
        if res.size == 0:
            continue
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= thr and max_val > best_val:
            best_val = max_val
            best_label = label

    return fixed_shift_map.get(best_label, 0)


def safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 0, w - 1)
    y2 = clamp(y2, 0, h - 1)
    if y2 <= y1 or x2 <= x1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


_power_re = re.compile(r"(\d+[Kk]|[0-9]+[Mm]|[0-9]+[Bb])")


def clean_power_text(raw: str) -> str:
    compact = raw.replace(" ", "")
    m = _power_re.search(compact)
    return m.group(1) if m else raw


def parse_power_to_int(text: str) -> int | None:
    if not text:
        return None
    t = text.strip().upper()
    mult = 1
    if t.endswith("K"):
        mult = 1_000
        t = t[:-1]
    elif t.endswith("M"):
        mult = 1_000_000
        t = t[:-1]
    elif t.endswith("B"):
        mult = 1_000_000_000
        t = t[:-1]
    try:
        return int(float(t) * mult)
    except ValueError:
        return None


def suppress_vertical_band(res: np.ndarray, y: int, gap: int) -> None:
    y1 = clamp(y - gap // 2, 0, res.shape[0] - 1)
    y2 = clamp(y + gap // 2, 0, res.shape[0] - 1)
    res[y1 : y2 + 1, :] = -1.0


def match_rows(
    gray: np.ndarray, tpl: np.ndarray, cfg: MemberListConfig
) -> tuple[list[tuple[int, int, float]], int, int]:
    th, tw = tpl.shape[:2]
    search_h = int(gray.shape[0] * 0.85)
    res_full = cv2.matchTemplate(gray[:search_h, :], tpl, cv2.TM_CCOEFF_NORMED)
    work = res_full.copy()

    picks: list[tuple[int, int, float]] = []
    for _ in range(cfg.max_passes):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val < cfg.match_threshold:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        picks.append((x, y, float(max_val)))
        suppress_vertical_band(work, y, cfg.min_row_gap_px)

    picks.sort(key=lambda p: p[1])
    return picks, th, tw


def extract_rois(
    screen_bgr: np.ndarray, px: int, py: int, tw: int, th: int, cfg: MemberListConfig
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    pr_x1 = px + tw
    pr_y1 = py
    pr_x2 = pr_x1 + cfg.power_width
    pr_y2 = pr_y1 + cfg.power_height
    power_roi = safe_crop(screen_bgr, pr_x1, pr_y1, pr_x2, pr_y2)

    nr_x1 = px - cfg.name_dx_left
    nr_y1 = py - cfg.name_dy_above
    nr_x2 = nr_x1 + cfg.name_width
    nr_y2 = nr_y1 + cfg.name_height
    name_roi = safe_crop(screen_bgr, nr_x1, nr_y1, nr_x2, nr_y2)

    return power_roi, name_roi, (nr_x1, nr_y1, nr_x2, nr_y2)


def extract_members_from_image(
    screen_bgr: np.ndarray, cfg: MemberListConfig
) -> list[GuildMemberList]:
    gray = Color.to_grayscale(screen_bgr)

    try:
        tpl = IO.load_image(Path(cfg.power_icon_path), grayscale=True)
    except FileNotFoundError:
        logging.warning("power icon template not found")
        return []

    search_h = int(gray.shape[0] * 0.85)
    matches = TemplateMatcher.find_all_template_matches(
        base_image=gray[:search_h, :],
        template_image=tpl,
        threshold=ConfidenceValue(cfg.match_threshold),
        min_distance=cfg.min_row_gap_px,
        grayscale=False,
    )
    matches.sort(key=lambda m: m.box.top_left.y)
    if cfg.max_passes > 0:
        matches = matches[: cfg.max_passes]

    results: list[GuildMemberList] = []

    for match in matches:
        px = match.box.top_left.x
        py = match.box.top_left.y
        tw = match.box.width
        th = match.box.height
        power_roi, name_roi, name_rect = extract_rois(screen_bgr, px, py, tw, th, cfg)

        if name_roi.size != 0:
            badge_shift = compute_badge_shift(
                Color.to_grayscale(name_roi),
                cfg.gender_templates,
                cfg.gender_match_threshold,
            )
        else:
            badge_shift = 0

        if badge_shift > 0:
            nr_x1, nr_y1, _, nr_y2 = name_rect
            nr_x1 += badge_shift
            nr_x2 = nr_x1 + cfg.name_width
            name_roi = safe_crop(screen_bgr, nr_x1, nr_y1, nr_x2, nr_y2)

        power_gray = preprocess_for_ocr(power_roi) if power_roi.size != 0 else power_roi
        name_gray = preprocess_for_ocr(name_roi) if name_roi.size != 0 else name_roi
        name_text = (
            cfg.ocr_backend.extract_text(name_gray) if name_gray.size != 0 else ""
        )
        raw_power = (
            cfg.ocr_backend.extract_text(power_gray) if power_gray.size != 0 else ""
        )
        clean_power = clean_power_text(raw_power)
        power_val = parse_power_to_int(clean_power)

        results.append(GuildMemberList(player_name=name_text, power=power_val))

    return results


class GuildManagement(AFKJourneyBase):
    """AFK Journey Guild Management."""

    @register_command(
        name="GuildManagement",
        gui=GUIMetadata(
            label="Guild Management",
            category=AFKJCategory.EVENTS_AND_OTHER,
            tooltip="AFK Journey Guild Management",
        ),
    )
    @register_custom_routine_choice(label="Guild Management")
    def manage_guild(self) -> None:
        """Manage Guild."""
        self.start_up()

        if not self._is_on_guild_member_list():
            try:
                self.navigate_to_guild()
                self.navigate_to_guild_members()
            except GameTimeoutError as e:
                logging.error(f"{e} {self.LANG_ERROR}")
                return None

        cfg = MemberListConfig()
        members: list[GuildMemberList] = []

        attempts = 0
        max_attempts = 10
        old_screenshot = Cropping.crop(
            self.get_screenshot(), crop_regions=CropRegions(top="50%")
        )
        members.extend(extract_members_from_image(old_screenshot.image, cfg))

        while True:
            self.swipe_up(sy=1350, ey=378, duration=15)
            self.tap(Point(x=1070, y=960), scale=True)
            sleep(2)

            new_screenshot = Cropping.crop(
                self.get_screenshot(), crop_regions=CropRegions(top="50%")
            )

            if TemplateMatcher.similar_image(
                base_image=new_screenshot.image,
                template_image=old_screenshot.image,
                threshold=ConfidenceValue("95%"),
            ):
                logging.info("Reached the end of the member list.")
                break

            members.extend(extract_members_from_image(new_screenshot.image, cfg))
            old_screenshot = new_screenshot
            attempts += 1
            if attempts >= max_attempts:
                logging.warning(
                    "Reached maximum attempts to scroll through the member list."
                )
                break

        for idx, member in enumerate(members, start=1):
            logging.info(
                "row %s — name: %s | power: %s",
                idx,
                member.player_name,
                member.power,
            )

    def _is_on_guild_member_list(self) -> bool:
        """Check if the current screen is the Guild Management screen."""
        return (
            self.game_find_template_match(
                template="guild/guild_member_list_activeness.png",
            )
            is not None
        )

    def _is_in_clashfronts(self) -> bool:
        """Check if the current screen is in Clashfronts."""
        return (
            self.game_find_template_match(
                template="guild/clashfronts_team_selection.png",
            )
            is not None
        )

    def _is_in_dream_realm_rank(self) -> bool:
        """Check if the current screen is in Dream Realm Rank."""
        return (
            self.game_find_template_match(
                template="guild/dr_rank.png",
            )
            is not None
        )

    def _is_dream_realm_rank_filtered(self) -> bool:
        """Check if the Dream Realm Rank is filtered."""
        return (
            self.game_find_template_match(
                template="guild/dr_member_filter.png",
            )
            is not None
        )
