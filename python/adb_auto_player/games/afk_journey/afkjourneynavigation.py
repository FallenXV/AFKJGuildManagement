import logging
from abc import ABC
from time import sleep

from adb_auto_player.exceptions import (
    AutoPlayerError,
    GameActionFailedError,
    GameNotRunningOrFrozenError,
    GameTimeoutError,
)
from adb_auto_player.models import ConfidenceValue
from adb_auto_player.models.geometry import Point
from adb_auto_player.models.image_manipulation import CropRegions
from adb_auto_player.models.template_matching import TemplateMatchResult

from .popup_handler import AFKJourneyPopupHandler


class AFKJourneyNavigation(AFKJourneyPopupHandler, ABC):
    # Timeouts
    NAVIGATION_TIMEOUT = 10.0

    # Points
    CENTER_POINT = Point(x=1080 // 2, y=1920 // 2)
    RESONATING_HALL_POINT = Point(x=620, y=1830)
    BATTLE_MODES_POINT = Point(x=460, y=1830)
    GUILD_POINT = Point(x=780, y=1830)

    def navigate_to_world(
        self,
    ) -> None:
        """Navigate to world view. Previously default_state.

        This is outside of homestead when your character is on the map.
        With buttons: "Mystical House", "Battle Modes", ... visible.
        """
        templates = [
            "navigation/world.png",
            "popup/quick_purchase.png",
            "navigation/confirm.png",
            "navigation/notice.png",
            "navigation/confirm_text.png",
            "navigation/time_of_day.png",
            "navigation/dotdotdot.png",
            "battle/copy.png",
            "guide/close.png",
            "guide/next.png",
            "login/claim.png",
            "arcane_labyrinth/back_arrow.png",
            "battle/exit_door.png",
            "arcane_labyrinth/select_a_crest.png",
            "navigation/resonating_hall_back.png",
        ]

        max_attempts = 40
        restart_attempts = 20
        attempts = 0

        restart_attempted = False

        while True:
            if not self.is_game_running():
                logging.error("Game not running.")
                self._handle_restart(templates)
            elif attempts >= restart_attempts and not restart_attempted:
                logging.warning("Failed to navigate to default state.")
                self._handle_restart(templates)
                restart_attempted = True
            elif attempts >= max_attempts:
                raise GameNotRunningOrFrozenError(
                    "Failed to navigate to default state."
                )
            attempts += 1

            if self._navigate_to_default_state(templates):
                break

        sleep(2)

    def _navigate_to_default_state(self, templates: list[str]) -> bool:
        result = self.find_any_template(templates)

        if result is None:
            self.press_back_button()
            sleep(3)
            return False

        match result.template:
            case "navigation/time_of_day.png":
                return True
            case "navigation/notice.png":
                # This is the Game Entry Screen
                self.tap(self.CENTER_POINT, scale=True)
                sleep(3)
            case "navigation/confirm.png":
                if not self.handle_popup_messages():
                    self.tap(result)
                    sleep(1)
            case "navigation/dotdotdot.png" | "popup/quick_purchase.png":
                self.press_back_button()
                sleep(1)
            case "arcane_labyrinth/select_a_crest.png":
                self.tap(Point(550, 1460))  # bottom crest
                sleep(1)
                self.tap(result)
                sleep(1)
            case "arcane_labyrinth/back_arrow.png":
                self.tap(result)
                sleep(2)
            case _:
                self.tap(result)
                sleep(1)
        return False

    def _handle_restart(self, templates: list[str]) -> None:
        logging.warning("Trying to restart AFK Journey.")
        self.force_stop_game()
        self.start_game()
        # if your game needs more than 6 minutes to start there is no helping yourself
        max_attempts = 120
        attempts = 0
        while not self.find_any_template(templates) and self.is_game_running():
            if attempts >= max_attempts:
                raise GameNotRunningOrFrozenError(
                    "Failed to navigate to default state."
                )
            attempts += 1
            self.tap(self.CENTER_POINT, scale=True)
            sleep(3)
        sleep(1)

    def navigate_to_resonating_hall(self) -> None:
        def i_am_in_resonating_hall() -> bool:
            try:
                _ = self.wait_for_any_template(
                    templates=[
                        "resonating_hall/artifacts.png",
                        "resonating_hall/collections.png",
                        "resonating_hall/equipment.png",
                    ],
                    timeout=1,
                )
                return True
            except GameTimeoutError:
                return False

        if i_am_in_resonating_hall():
            logging.info("Already in Resonating Hall.")
            return

        logging.info("Navigating to the Resonating Hall.")
        if shortcut := self.game_find_template_match(
            template="navigation/resonating_hall_shortcut",
            crop_regions=CropRegions(top="80%", left="30%", right="30%"),
            threshold=ConfidenceValue("75%"),
        ):
            self.tap(shortcut)
            sleep(3)
            if i_am_in_resonating_hall():
                return

        self.navigate_to_world()
        max_click_count = 3
        click_count = 0

        count = 0
        max_count = 3
        last_error: AutoPlayerError | None = None
        while True:
            count += 1
            if count > max_count:
                if last_error is not None:
                    raise last_error
                raise AutoPlayerError("Failed to navigate to Resonating Hall.")
            try:
                while self._can_see_time_of_day_button():
                    self.tap(self.RESONATING_HALL_POINT, scale=True)
                    sleep(3)
                    click_count += 1
                    if click_count > max_click_count:
                        raise GameActionFailedError(
                            "Failed to navigate to the Resonating Hall."
                        )
                _ = self.wait_for_any_template(
                    templates=[
                        "resonating_hall/artifacts.png",
                        "resonating_hall/collections.png",
                        "resonating_hall/equipment.png",
                    ],
                    timeout=self.NAVIGATION_TIMEOUT,
                )
                break
            except AutoPlayerError as e:
                logging.warning(e)
                last_error = e
        sleep(1)
        return

    def _can_see_time_of_day_button(self) -> bool:
        return (
            self.game_find_template_match(
                "navigation/time_of_day.png",
                crop_regions=CropRegions(left=0.6, bottom=0.6),
            )
            is not None
        )

    def navigate_to_afk_stages_screen(self) -> None:
        logging.info("Navigating to AFK stages screen.")
        self.navigate_to_battle_modes_screen()

        self._tap_till_template_disappears(
            "battle_modes/afk_stage.png", ConfidenceValue("75%")
        )

        self.wait_for_template(
            template="navigation/resonating_hall_label.png",
            crop_regions=CropRegions(left=0.3, right=0.3, top=0.9),
            timeout=self.NAVIGATION_TIMEOUT,
        )
        self.tap(Point(x=550, y=1080), scale=True)  # click rewards popup
        sleep(1)

    def _navigate_to_battle_modes_screen(self) -> None:
        self.tap(self.BATTLE_MODES_POINT, scale=True)
        result = self.wait_for_any_template(
            templates=[
                "battle_modes/afk_stage.png",
                "battle_modes/duras_trials.png",
                "battle_modes/arcane_labyrinth.png",
                "popup/quick_purchase.png",
            ],
            threshold=ConfidenceValue("75%"),
            timeout=self.NAVIGATION_TIMEOUT,
        )

        if result.template == "popup/quick_purchase.png":
            self.press_back_button()
            sleep(1)

        _ = self.wait_for_any_template(
            templates=[
                "battle_modes/afk_stage.png",
                "battle_modes/duras_trials.png",
                "battle_modes/arcane_labyrinth.png",
            ],
            threshold=ConfidenceValue("75%"),
            timeout=self.NAVIGATION_TIMEOUT,
        )

    def navigate_to_battle_modes_screen(self) -> None:
        attempt = 0
        max_attempts = 3
        while True:
            self.navigate_to_world()
            sleep(attempt)
            try:
                self._navigate_to_battle_modes_screen()
            except GameTimeoutError as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise e
                else:
                    continue
            break
        sleep(1)

    def navigate_to_duras_trials_screen(self) -> None:
        logging.info("Navigating to Dura's Trial select")

        def stop_condition() -> bool:
            match = self.game_find_template_match(
                template="duras_trials/featured_heroes.png",
                crop_regions=CropRegions(left=0.7, bottom=0.8),
            )
            return match is not None

        if stop_condition():
            return

        self.navigate_to_battle_modes_screen()
        coords = self._find_on_battle_modes(
            template="battle_modes/duras_trials.png",
            timeout_message="Dura's Trial not found.",
        )
        self.tap(coords)
        sleep(1)

        # popups
        self.tap(self.CENTER_POINT, scale=True)
        self.tap(self.CENTER_POINT, scale=True)
        self.tap(self.CENTER_POINT, scale=True)

        self.wait_for_template(
            template="duras_trials/featured_heroes.png",
            crop_regions=CropRegions(left=0.7, bottom=0.8),
            timeout=self.NAVIGATION_TIMEOUT,
        )
        sleep(1)
        return

    def _find_on_battle_modes(
        self, template: str, timeout_message: str
    ) -> TemplateMatchResult:
        if not self.game_find_template_match(template):
            self.swipe_up(sy=1350, ey=500)

        return self.wait_for_template(
            template=template,
            timeout_message=timeout_message,
            timeout=self.NAVIGATION_TIMEOUT,
        )

    def navigate_to_legend_trials_select_tower(self) -> None:
        """Navigate to Legend Trials select tower screen."""
        logging.info("Navigating to Legend Trials tower selection")
        self.navigate_to_battle_modes_screen()

        coords = self._find_on_battle_modes(
            template="battle_modes/legend_trial.png",
            timeout_message="Could not find Legend Trial Label",
        )

        self.tap(coords)
        self.wait_for_template(
            template="legend_trials/s_header.png",
            crop_regions=CropRegions(right=0.8, bottom=0.8),
            timeout_message="Could not find Season Legend Trial Header",
            timeout=self.NAVIGATION_TIMEOUT,
        )
        sleep(1)

    def navigate_to_arcane_labyrinth(self) -> None:
        # Possibility of getting stuck
        # Back button does not work on Arcane Labyrinth screen
        logging.info("Navigating to Arcane Labyrinth screen")

        def stop_condition() -> bool:
            """Stop condition."""
            match = self.find_any_template(
                templates=[
                    "arcane_labyrinth/select_a_crest.png",
                    "arcane_labyrinth/confirm.png",
                    "arcane_labyrinth/quit.png",
                ],
                crop_regions=CropRegions(top=0.8),
            )

            if match is not None:
                logging.info("Select a Crest screen open")
                return True
            return False

        if stop_condition():
            return

        self.navigate_to_battle_modes_screen()
        coords = self._find_on_battle_modes(
            template="battle_modes/arcane_labyrinth.png",
            timeout_message="Could not find Arcane Labyrinth Label",
        )

        self.tap(coords)
        sleep(3)
        _ = self.wait_for_any_template(
            templates=[
                "arcane_labyrinth/select_a_crest.png",
                "arcane_labyrinth/confirm.png",
                "arcane_labyrinth/quit.png",
                "arcane_labyrinth/enter.png",
                "arcane_labyrinth/heroes_icon.png",
            ],
            threshold=ConfidenceValue("70%"),
            timeout=27,  # I imagine this animation can take really long for some people
            delay=1,
        )
        return

    def navigate_to_guild(self):
        """Navigate to Guild screen."""
        logging.info("Navigating to Guild screen")
        self.navigate_to_default_state()
        attempt = 0
        max_attempts = 3

        self.tap(self.GUILD_POINT, scale=True)
        sleep(4)

        while True:
            if attempt >= max_attempts:
                raise GameActionFailedError("Failed to navigate to Guild screen.")
            attempt += 1

            match = self.game_find_template_match(
                template="guild/guild_rank.png",
                crop_regions=CropRegions(left=0.8, bottom=0.8),
                threshold=ConfidenceValue("75%"),
            )

            if match is not None:
                break

            logging.info(match)

            logging.info("Guild screen not found, trying again.")
            self.tap(self.GUILD_POINT, scale=True)
            sleep(4)

    def navigate_to_guild_members(self):
        """Navigate to Guild Members screen."""
        logging.info("Navigating to Guild Members screen")
        attempt = 0
        max_attempts = 3

        self.tap(Point(x=250, y=150), scale=True)
        sleep(4)

        while True:
            if attempt >= max_attempts:
                raise GameActionFailedError(
                    "Failed to navigate to Guild Members screen."
                )
            attempt += 1

            if self.game_find_template_match(
                template="guild/guild_member_list_activeness.png",
            ):
                break

            self.navigate_to_guild()
            self.tap(Point(x=250, y=150), scale=True)
            sleep(4)
