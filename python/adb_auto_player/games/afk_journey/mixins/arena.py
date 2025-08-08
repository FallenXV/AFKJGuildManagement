"""Arena Mixin."""

import logging
from time import sleep

from adb_auto_player.decorators import register_command
from adb_auto_player.exceptions import GameTimeoutError
from adb_auto_player.games.afk_journey.base import AFKJourneyBase
from adb_auto_player.games.afk_journey.gui_category import AFKJCategory
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point
from adb_auto_player.models.image_manipulation import CropRegions


class ArenaMixin(AFKJourneyBase):
    """Arena Mixin."""

    @register_command(
        name="Arena",
        gui=GUIMetadata(
            label="Arena",
            category=AFKJCategory.GAME_MODES,
        ),
    )
    def run_arena(self) -> None:
        """Use Arena attempts."""
        self.start_up(device_streaming=False)

        try:
            self._enter_arena()
        except GameTimeoutError:
            return

        for _ in range(5):
            if self.game_find_template_match("arena/no_attempts.png"):
                logging.debug("Free attempts exhausted before 5 attempts.")
                break

            self._choose_opponent()
            self._battle()

        for _ in range(2):
            if not self._claim_free_attempt():
                break

            self._choose_opponent()
            self._battle()

        logging.info("Arena finished.")

    ############################## Helper Functions ##############################

    def _enter_arena(self) -> None:
        """Enter Arena."""
        logging.info("Entering Arena...")
        self.navigate_to_world()
        self.tap(Point(460, 1830))  # Battle Modes
        try:
            arena_mode = self.wait_for_template(
                "arena/label.png",
                timeout_message="Failed to find Arena.",
                timeout=self.MIN_TIMEOUT,
            )
            self.tap(arena_mode)
            sleep(2)
        except GameTimeoutError as fail:
            logging.error(f"{fail} {self.LANG_ERROR}")
            raise

        logging.debug("Checking for weekly arena notices.")
        all(self._confirm_notices() for _ in range(2))

    def _confirm_notices(self) -> bool:
        """Close out weekly reward and weekly notice popups.

        Returns:
            bool: True if notices were closed, False otherwise.
        """
        try:
            _ = self.wait_for_any_template(
                templates=["arena/weekly_rewards.png", "arena/weekly_notice.png"],
                timeout=self.MIN_TIMEOUT,
                timeout_message="No notices found.",
            )
            self.tap(Point(380, 1890))
            sleep(4)

            return True
        except GameTimeoutError as fail:
            logging.debug(fail)
            pass

        return False

    def _choose_opponent(self) -> None:
        """Choose Arena opponent."""
        try:
            logging.debug("Start arena challenge.")
            btn = self.wait_for_any_template(
                templates=["arena/challenge.png", "arena/continue.png"],
                timeout=self.MIN_TIMEOUT,
                timeout_message="Failed to start Arena runs.",
            )
            sleep(2)
            self.tap(btn)

            logging.debug("Choosing opponent.")
            opponent = self.wait_for_template(
                template="arena/opponent.png",
                crop_regions=CropRegions(right=0.6),  # Target weakest opponent.
                timeout=self.MIN_TIMEOUT,
                timeout_message="Failed to find Arena opponent.",
            )
            self.tap(opponent)
        except GameTimeoutError as fail:
            logging.error(fail)

    def _battle(self) -> None:
        """Battle Arena opponent."""
        try:
            logging.debug("Initiate battle.")
            start = self.wait_for_template(
                template="arena/battle.png",
                timeout=self.MIN_TIMEOUT,
                timeout_message="Failed to start Arena battle.",
            )
            sleep(2)
            self.tap(start)

            logging.debug("Skip battle.")
            skip = self.wait_for_template(
                template="arena/skip.png",
                timeout=self.MIN_TIMEOUT,
                timeout_message="Failed to skip Arena battle.",
            )
            self.tap(skip)

            logging.debug("Battle complete.")
            confirm = self.wait_for_template(
                template="arena/done.png",
                timeout=self.MIN_TIMEOUT,
                timeout_message="Failed to confirm Arena battle completion.",
            )
            sleep(4)
            self.tap(confirm)
            sleep(2)
        except GameTimeoutError as fail:
            logging.error(fail)

    def _claim_free_attempt(self) -> bool:
        """Claim free Arena attempts.

        Returns:
            bool: True if free attempt claimed, False not available.
        """
        try:
            logging.debug("Claiming free attempts.")
            buy = self.wait_for_template(
                template="arena/buy.png",
                timeout=self.MIN_TIMEOUT,
                timeout_message="Failed looking for free attempts.",
            )
            self.tap(buy)
        except GameTimeoutError:
            return True  # Not breaking, but would be interested in why it failed.

        try:
            _ = self.wait_for_template(
                template="arena/buy_free.png",
                timeout=self.MIN_TIMEOUT,
                timeout_message="No more free attempts.",
            )
            logging.debug("Free attempt found.")
        except GameTimeoutError as fail:
            logging.info(fail)
            cancel = self.game_find_template_match("arena/cancel_purchase.png")
            (
                self.tap(cancel)
                if cancel
                else self.tap(Point(550, 1790))  # Cancel fallback
            )

            return False

        logging.debug("Purchasing free attempt.")
        self._click_confirm_on_popup()

        return True
