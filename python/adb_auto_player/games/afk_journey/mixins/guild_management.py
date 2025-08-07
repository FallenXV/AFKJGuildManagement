"""AFK Journey Guild Management."""

import logging
from dataclasses import dataclass
from time import sleep

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
        old_screenshot = Cropping.crop(
            self.get_screenshot(), crop_regions=CropRegions(top="50%")
        )
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

            old_screenshot = new_screenshot
            attempts += 1
            if attempts >= max_attempts:
                logging.warning(
                    "Reached maximum attempts to scroll through the member list."
                )
                break

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
