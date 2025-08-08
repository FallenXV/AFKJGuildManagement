"""AFK Journey Guild Management."""

import logging
from dataclasses import dataclass
from time import sleep

from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.exceptions import (
    GameTimeoutError,
)
from adb_auto_player.models.decorators import GUIMetadata

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

        while not self.game_find_template_match(
            template="guild/guild_member_list_bottom.png"
        ):
            self.swipe_up(sy=1350, ey=379, duration=15)
            sleep(2)

        # Implement guild management logic here
        # For example, checking guild status, performing actions, etc.

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
