import asyncio
import json
import logging
import os
import re
import time
from contextlib import suppress
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import mention_html


load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_int_list(value: str, env_name: str) -> List[int]:
    result: List[int] = []

    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue

        try:
            result.append(int(raw))
        except ValueError as e:
            raise RuntimeError(
                f"{env_name} must contain integers separated by commas"
            ) from e

    if not result:
        raise RuntimeError(f"{env_name} environment variable is empty")

    return list(dict.fromkeys(result))


def build_public_user_link(username: Optional[str]) -> Optional[str]:
    if not username:
        return None

    username = username.lstrip("@").strip()
    if not username:
        return None

    return f"https://t.me/{username}?profile"


def build_telegram_user_deeplink(user_id: int) -> Optional[str]:
    if user_id <= 0:
        return None
    return f"tg://user?id={user_id}"


class Intent:
    def __init__(self, name: str, triggers: List[str], response: str) -> None:
        self.name = name
        self.response = response
        self.patterns = [
            re.compile(pattern, re.IGNORECASE | re.UNICODE)
            for pattern in triggers
        ]

    def match(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


def load_intents(path: str) -> Dict[str, "Intent"]:
    scenarios_path = Path(path)
    if not scenarios_path.exists():
        raise RuntimeError(f"Scenarios file not found: {path}")

    with scenarios_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    intents: Dict[str, Intent] = {}
    for item in data.get("intents", []):
        name = item.get("name")
        triggers = item.get("triggers", [])
        response = item.get("response", "")

        if not name:
            continue

        if not isinstance(triggers, list):
            raise RuntimeError(f"Intent '{name}' must contain a list in 'triggers'")

        intents[name] = Intent(name=name, triggers=triggers, response=response)

    if "fallback" not in intents:
        intents["fallback"] = Intent(name="fallback", triggers=[], response="")

    return intents


class JsonStateFile:
    def __init__(self, path: str, default_data: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.default_data = default_data

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return dict(self.default_data)

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return dict(self.default_data)
            return raw
        except Exception:
            logger.exception("Failed to load state from %s", self.path)
            return dict(self.default_data)

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


class IgnoreStore:
    """
    Хранит user_id -> unix_timestamp_until
    и сохраняет состояние в JSON-файл.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._data: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load ignore store from %s", self.path)
            self._data = {}
            return

        if not isinstance(raw, dict):
            logger.warning("Ignore store %s is not a dict, resetting", self.path)
            self._data = {}
            return

        now = time.time()
        cleaned: Dict[str, float] = {}

        for key, value in raw.items():
            try:
                user_id = str(int(key))
                until_ts = float(value)
            except (TypeError, ValueError):
                continue

            if until_ts > now:
                cleaned[user_id] = until_ts

        self._data = cleaned
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _remove_if_expired(self, user_id: int) -> bool:
        key = str(user_id)
        until_ts = self._data.get(key)
        if until_ts is None:
            return False

        if until_ts <= time.time():
            self._data.pop(key, None)
            self._save()
            return False

        return True

    def is_ignored(self, user_id: int) -> bool:
        return self._remove_if_expired(user_id)

    def remaining_seconds(self, user_id: int) -> int:
        key = str(user_id)
        until_ts = self._data.get(key)
        if until_ts is None:
            return 0

        remaining = int(until_ts - time.time())
        if remaining <= 0:
            self._data.pop(key, None)
            self._save()
            return 0

        return remaining

    def set_ignore(self, user_id: int, ttl_seconds: int) -> None:
        ttl_seconds = max(1, int(ttl_seconds))
        self._data[str(user_id)] = time.time() + ttl_seconds
        self._save()

    def prune_expired(self) -> int:
        now = time.time()
        before = len(self._data)

        self._data = {
            key: until_ts
            for key, until_ts in self._data.items()
            if until_ts > now
        }

        removed = before - len(self._data)
        if removed > 0:
            self._save()

        return removed

    def count(self) -> int:
        return len(self._data)


class BotControlState:
    def __init__(self, path: str) -> None:
        self.storage = JsonStateFile(path, default_data={"paused": False})
        self.paused = False
        self._load()

    def _load(self) -> None:
        data = self.storage.load()
        self.paused = bool(data.get("paused", False))

    def save(self) -> None:
        self.storage.save({"paused": self.paused})

    def set_paused(self, value: bool) -> None:
        self.paused = bool(value)
        self.save()


class AdmissionsBot:
    def __init__(
        self,
        intents: Dict[str, Intent],
        fallback_intent: Intent,
        staff_chat_ids: List[int],
        staff_user_ids: List[int],
        ignore_ttl_seconds: int,
        ignore_state_file: str,
        bot_state_file: str,
        cleanup_interval_seconds: int,
    ) -> None:
        self.intents = intents
        self.fallback_intent = fallback_intent

        self.staff_chat_ids = staff_chat_ids
        self.staff_chat_ids_set: Set[int] = set(staff_chat_ids)

        self.staff_user_ids = staff_user_ids
        self.staff_user_ids_set: Set[int] = set(staff_user_ids)

        self.ignore_ttl_seconds = int(ignore_ttl_seconds)
        self.cleanup_interval_seconds = max(60, int(cleanup_interval_seconds))

        self.ignore_store = IgnoreStore(ignore_state_file)
        self.control_state = BotControlState(bot_state_file)

        self.escalation_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def is_staff_chat(self, chat_id: int) -> bool:
        return chat_id in self.staff_chat_ids_set

    def is_staff_user(self, user_id: int) -> bool:
        return user_id in self.staff_user_ids_set

    def is_paused(self) -> bool:
        return self.control_state.paused

    def set_paused(self, value: bool) -> None:
        self.control_state.set_paused(value)

    async def is_authorized_admin_command(self, update: Update) -> bool:
        if update.message is None or update.effective_user is None:
            return False

        user = update.effective_user
        chat = update.effective_chat

        if not self.is_staff_user(user.id):
            return False

        if chat is None or chat.type != "private":
            await update.message.reply_text(
                "Эта команда доступна только сотруднику в личном чате с ботом."
            )
            return False

        return True

    async def cmd_user_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if update.message is None:
            return

        user = update.effective_user
        chat = update.effective_chat

        if user is None:
            return

        if self.is_staff_user(user.id):
            await update.message.reply_text(
                "Для управления ботом используйте:\n"
                "/pause — поставить бота на паузу\n"
                "/resume — снять паузу\n"
                "/status — проверить статус"
            )
            return

        if chat and self.is_staff_chat(chat.id):
            return

        if self.is_paused():
            await update.message.reply_text(
                "Бот временно находится на паузе. Попробуйте написать позже."
            )
            return

        await update.message.reply_text(
            "Здравствуйте! Напишите ваш вопрос, и я постараюсь помочь.\n"
            "Если вопрос сложный, я передам его специалисту приёмной комиссии."
        )

    async def cmd_pause(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self.is_authorized_admin_command(update):
            return

        if update.message is None or update.effective_user is None:
            return

        if self.is_paused():
            await update.message.reply_text("Бот уже находится на паузе.")
            return

        self.set_paused(True)
        await update.message.reply_text(
            "Бот переведён в режим паузы.\n"
            "Обычные пользователи сейчас полностью игнорируются."
        )
        logger.warning("Bot paused by staff user_id=%s", update.effective_user.id)

    async def cmd_resume(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self.is_authorized_admin_command(update):
            return

        if update.message is None or update.effective_user is None:
            return

        if not self.is_paused():
            await update.message.reply_text("Бот уже активен.")
            return

        self.set_paused(False)
        await update.message.reply_text(
            "Бот снова активен.\n"
            "Обработка сообщений возобновлена."
        )
        logger.warning("Bot resumed by staff user_id=%s", update.effective_user.id)

    async def cmd_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self.is_authorized_admin_command(update):
            return

        if update.message is None:
            return

        status_text = "на паузе" if self.is_paused() else "активен"

        await update.message.reply_text(
            f"Статус бота: {status_text}\n"
            f"Пользователей в игноре: {self.ignore_store.count()}\n"
            f"Интервал очистки: {self.cleanup_interval_seconds} сек."
        )

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if update.message is None or update.message.text is None:
            return

        message = update.message
        chat = message.chat
        user = update.effective_user

        if user is None:
            logger.info("Ignoring message without effective_user in chat_id=%s", chat.id)
            return

        if user.is_bot:
            return

        if self.is_staff_user(user.id):
            logger.info(
                "Ignoring regular message from staff user_id=%s in chat_id=%s",
                user.id,
                chat.id,
            )
            return

        if self.is_paused():
            logger.info(
                "Ignoring message because bot is paused: user_id=%s chat_id=%s",
                user.id,
                chat.id,
            )
            return

        if self.is_staff_chat(chat.id):
            logger.info("Ignoring message from staff chat_id=%s", chat.id)
            return

        if self.ignore_store.is_ignored(user.id):
            logger.info(
                "Ignoring message from user_id=%s due to active cooldown (%s sec left)",
                user.id,
                self.ignore_store.remaining_seconds(user.id),
            )
            return

        message_text = message.text.strip()
        if not message_text:
            return

        text_normalized = normalize_text(message_text)

        matched_intent: Optional[Intent] = None
        for name, intent in self.intents.items():
            if name == self.fallback_intent.name:
                continue
            if intent.match(text_normalized):
                matched_intent = intent
                break

        if matched_intent:
            logger.info(
                "Matched intent=%s user_id=%s chat_id=%s text=%s",
                matched_intent.name,
                user.id,
                chat.id,
                message_text,
            )
            await message.reply_text(matched_intent.response)
            return

        await self.handle_complex_question(update, context)

    async def handle_complex_question(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if update.message is None or update.effective_user is None:
            return

        message = update.message
        user = update.effective_user
        chat = message.chat

        if chat.type == "private":
            await message.reply_text(
                "Ваш вопрос передан специалисту приёмной комиссии. "
                "Пожалуйста, ожидайте ответа."
            )
        else:
            await message.reply_text(
                "Ваш вопрос передан специалисту. "
                "Он подключится и поможет."
            )

        self.ignore_store.set_ignore(user.id, self.ignore_ttl_seconds)

        await self.escalation_queue.put(
            {
                "user_id": user.id,
                "user_name": user.full_name or user.username or f"Пользователь {user.id}",
                "username": user.username,
                "chat_id": chat.id,
                "chat_title": getattr(chat, "title", None) or str(chat.id),
                "chat_type": chat.type,
                "text": message.text,
                "message_id": message.message_id,
            }
        )

        logger.info(
            "Escalated user_id=%s chat_id=%s and set cooldown for %s seconds",
            user.id,
            chat.id,
            self.ignore_ttl_seconds,
        )

    async def escalation_worker(self, application: Application) -> None:
        logger.info("Escalation worker started")

        while True:
            item = await self.escalation_queue.get()

            try:
                user_id = item["user_id"]
                user_name = item["user_name"]
                username = item.get("username")
                chat_id = item["chat_id"]
                chat_title = item["chat_title"]
                chat_type = item["chat_type"]
                text = item["text"]

                user_mention = mention_html(user_id, user_name)

                tg_deeplink = build_telegram_user_deeplink(user_id)
                public_link = build_public_user_link(username)

                if tg_deeplink:
                    tg_link_line = (
                        f'Telegram-ссылка: <a href="{escape(tg_deeplink)}">'
                        f"{escape(tg_deeplink)}</a>"
                    )
                else:
                    tg_link_line = "Telegram-ссылка: недоступна"

                if public_link:
                    public_link_line = (
                        f'Публичная ссылка: <a href="{escape(public_link)}">'
                        f"{escape(public_link)}</a>"
                    )
                else:
                    public_link_line = (
                        "Публичная ссылка: недоступна "
                        "(у пользователя нет username)"
                    )

                if chat_type == "private":
                    chat_line = f"Чат: private (<code>{chat_id}</code>)"
                else:
                    chat_line = f"Чат: {escape(str(chat_title))} (<code>{chat_id}</code>)"

                staff_notification = (
                    f"⚠️ <b>Сложный вопрос</b>\n"
                    f"Пользователь: {user_mention}\n"
                    f"{tg_link_line}\n"
                    f"{public_link_line}\n"
                    f"ID пользователя: <code>{user_id}</code>\n"
                    f"{chat_line}\n"
                    f"Сообщение: {escape(text)}"
                )

                for staff_chat_id in self.staff_chat_ids:
                    try:
                        await application.bot.send_message(
                            chat_id=staff_chat_id,
                            text=staff_notification,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                        logger.info(
                            "Escalation sent to staff_chat_id=%s for user_id=%s",
                            staff_chat_id,
                            user_id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to send escalation to staff_chat_id=%s",
                            staff_chat_id,
                        )

            except Exception:
                logger.exception("Error while processing escalation item")
            finally:
                self.escalation_queue.task_done()

    async def cleanup_worker(self) -> None:
        logger.info(
            "Cleanup worker started, interval=%s sec",
            self.cleanup_interval_seconds,
        )

        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                removed = self.ignore_store.prune_expired()

                if removed > 0:
                    logger.info(
                        "Cleanup worker removed %s expired ignored users",
                        removed,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cleanup worker failed")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception(
        "Unhandled error while processing update=%s error=%s",
        update,
        context.error,
    )


async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set")

    staff_chat_id_env = os.getenv("STAFF_CHAT_ID")
    if not staff_chat_id_env:
        raise RuntimeError("STAFF_CHAT_ID environment variable not set")

    staff_user_id_env = os.getenv("STAFF_USER_ID")
    if not staff_user_id_env:
        raise RuntimeError("STAFF_USER_ID environment variable not set")

    scenarios_file = os.getenv("SCENARIOS_FILE", "scenarios.yaml")
    ignore_ttl_seconds = int(os.getenv("IGNORE_TTL_SECONDS", "1800"))
    ignore_state_file = os.getenv("IGNORE_STATE_FILE", "ignored_users.json")
    bot_state_file = os.getenv("BOT_STATE_FILE", "bot_state.json")
    cleanup_interval_seconds = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))

    staff_chat_ids = parse_int_list(staff_chat_id_env, "STAFF_CHAT_ID")
    staff_user_ids = parse_int_list(staff_user_id_env, "STAFF_USER_ID")

    intents = load_intents(scenarios_file)
    fallback_intent = intents["fallback"]

    admissions_bot = AdmissionsBot(
        intents=intents,
        fallback_intent=fallback_intent,
        staff_chat_ids=staff_chat_ids,
        staff_user_ids=staff_user_ids,
        ignore_ttl_seconds=ignore_ttl_seconds,
        ignore_state_file=ignore_state_file,
        bot_state_file=bot_state_file,
        cleanup_interval_seconds=cleanup_interval_seconds,
    )

    application = ApplicationBuilder().token(token).build()

    command_filter = ~filters.UpdateType.EDITED_MESSAGE

    application.add_handler(
        CommandHandler("start", admissions_bot.cmd_user_start, filters=command_filter)
    )
    application.add_handler(
        CommandHandler("pause", admissions_bot.cmd_pause, filters=command_filter)
    )
    application.add_handler(
        CommandHandler("resume", admissions_bot.cmd_resume, filters=command_filter)
    )
    application.add_handler(
        CommandHandler("status", admissions_bot.cmd_status, filters=command_filter)
    )

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, admissions_bot.handle_message)
    )

    application.add_error_handler(on_error)

    logger.info("Starting bot...")

    escalation_task: Optional[asyncio.Task[Any]] = None
    cleanup_task: Optional[asyncio.Task[Any]] = None

    await application.initialize()
    await application.start()

    escalation_task = asyncio.create_task(admissions_bot.escalation_worker(application))
    cleanup_task = asyncio.create_task(admissions_bot.cleanup_worker())

    logger.info("Background workers started")
    await application.updater.start_polling()

    try:
        await asyncio.Event().wait()
    finally:
        logger.info("Stopping bot...")

        await application.updater.stop()

        if escalation_task:
            escalation_task.cancel()
            with suppress(asyncio.CancelledError):
                await escalation_task

        if cleanup_task:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task

        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")