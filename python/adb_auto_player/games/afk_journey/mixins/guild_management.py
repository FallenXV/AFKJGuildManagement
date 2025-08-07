"""AFK Journey Guild Management."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import cv2
import numpy as np
import pytesseract

from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.exceptions import (
    GameTimeoutError,
)
from adb_auto_player.image_manipulation import (
    Cropping,
)
from adb_auto_player.models import ConfidenceValue
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point
from adb_auto_player.models.image_manipulation import CropRegions
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


# Paths and constants used for OCR of guild member list
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "guild"
_GENDER_TEMPLATES = {
    "male": cv2.imread((_TEMPLATE_DIR / "male.png").as_posix(), cv2.IMREAD_GRAYSCALE),
    "female": cv2.imread((_TEMPLATE_DIR / "female.png").as_posix(), cv2.IMREAD_GRAYSCALE),
}
_MATCH_THRESHOLD = 0.8

# Name box coordinates
_NAME_BOX_X1 = 240
_NAME_BOX_X2 = 600
_NAME_BOX_Y_START = 640
_NAME_BOX_HEIGHT = 50

# Power box coordinates
_POWER_BOX_X1 = 295
_POWER_BOX_X2 = 420
_POWER_BOX_Y_START = 740
_POWER_BOX_HEIGHT = 30

# Common row geometry
_ROW_HEIGHT = 249
_ICON_OFFSET = 45


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

        attempts = 0
        max_attempts = 10
        members: list[GuildMemberList] = []
        old_screenshot = Cropping.crop(
            self.get_screenshot(), crop_regions=CropRegions(top="50%")
        )
        members.extend(self._extract_member_info(old_screenshot.image))
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

            members.extend(self._extract_member_info(new_screenshot.image))
            old_screenshot = new_screenshot
            attempts += 1
            if attempts >= max_attempts:
                logging.warning(
                    "Reached maximum attempts to scroll through the member list."
                )
                break

        for member in members:
            logging.info("Guild Member: %s Power: %s", member.player_name, member.power)

    def _extract_member_info(self, image: np.ndarray) -> list[GuildMemberList]:
        """Extract member names and power from a screenshot."""
        members: list[GuildMemberList] = []
        img_bgr = image
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        for row in range(4):
            ny0 = _NAME_BOX_Y_START + row * _ROW_HEIGHT
            ny1 = ny0 + _NAME_BOX_HEIGHT
            crop_x1, crop_x2 = _NAME_BOX_X1, _NAME_BOX_X2

            name_slice = img_gray[ny0:ny1, _NAME_BOX_X1:_NAME_BOX_X2]
            for tpl in _GENDER_TEMPLATES.values():
                if tpl is None:
                    continue
                res = cv2.matchTemplate(name_slice, tpl, cv2.TM_CCOEFF_NORMED)
                if np.any(res >= _MATCH_THRESHOLD):
                    crop_x1 += _ICON_OFFSET
                    break

            name_roi = img_bgr[ny0:ny1, crop_x1:crop_x2]
            name_text = pytesseract.image_to_string(name_roi, config="--psm 7").strip()

            py0 = _POWER_BOX_Y_START + row * _ROW_HEIGHT
            py1 = py0 + _POWER_BOX_HEIGHT
            power_roi = img_bgr[py0:py1, _POWER_BOX_X1:_POWER_BOX_X2]
            raw_power = pytesseract.image_to_string(power_roi, config="--psm 7").strip()
            match = re.search(r"(\d+)K", raw_power)
            power_val = int(match.group(1)) * 1000 if match else None

            members.append(
                GuildMemberList(
                    player_name=name_text if name_text else None, power=power_val
                )
            )

        return members

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
