import asyncio
import logging
import os
import re
from contextlib import suppress
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
import yaml
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import mention_html

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_chat_ids(value: str, env_name: str) -> List[int]:
    chat_ids: List[int] = []

    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            chat_ids.append(int(raw))
        except ValueError as e:
            raise RuntimeError(
                f"{env_name} must contain integers separated by commas"
            ) from e

    if not chat_ids:
        raise RuntimeError(f"{env_name} environment variable is empty")

    return list(dict.fromkeys(chat_ids))


def build_public_user_link(username: Optional[str]) -> Optional[str]:
    if not username:
        return None
    username = username.lstrip("@").strip()
    if not username:
        return None
    return f"https://t.me/{username}?profile"


def build_telegram_user_deeplink(user_id: int) -> Optional[str]:
    if not user_id or user_id <= 0:
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
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    intents: Dict[str, Intent] = {}
    for item in data.get("intents", []):
        name = item.get("name")
        triggers = item.get("triggers", [])
        response = item.get("response", "")
        if not name:
            continue
        intents[name] = Intent(name=name, triggers=triggers, response=response)
    return intents


class AdmissionsBot:
    def __init__(
        self,
        intents: Dict[str, "Intent"],
        fallback_intent: "Intent",
        staff_chat_ids: List[int],
        group_chat_id: Optional[int] = None,
    ) -> None:
        self.intents = intents
        self.fallback_intent = fallback_intent
        self.staff_chat_ids = staff_chat_ids
        self.staff_chat_ids_set: Set[int] = set(staff_chat_ids)
        self.group_chat_id = group_chat_id
        self.escalation_queue: asyncio.Queue = asyncio.Queue()

    def is_staff_chat(self, chat_id: int) -> bool:
        return chat_id in self.staff_chat_ids_set

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if update.message is None or update.message.text is None:
            return

        message = update.message
        user = update.effective_user
        chat = message.chat

        if user and user.is_bot:
            return

        if self.is_staff_chat(chat.id):
            logger.info("Ignoring message from staff chat_id=%s", chat.id)
            return

        message_text = message.text.strip()
        if not message_text:
            return

        if message_text.startswith("/"):
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
            logger.info("Matched intent %s: %s", matched_intent.name, message_text)
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
                "Сейчас ваш вопрос передам специалисту. "
                "Он подключится и поможет."
            )

        await self.escalation_queue.put(
            {
                "user_id": user.id,
                "user_name": user.full_name or user.username or f"Пользователь {user.id}",
                "username": user.username,
                "chat_id": chat.id,
                "chat_title": getattr(chat, "title", None) or str(chat.id),
                "chat_username": getattr(chat, "username", None),
                "chat_type": chat.type,
                "text": message.text,
                "message_id": message.message_id,
            }
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
                chat_type = item.get("chat_type")
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
                    chat_line = (
                        f"Чат: {escape(str(chat_title))} (<code>{chat_id}</code>)"
                    )

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
                            "Escalation sent to staff chat_id=%s for user_id=%s",
                            staff_chat_id,
                            user_id,
                        )
                    except Exception:
                        logger.exception(
                            "Ошибка отправки сотруднику chat_id=%s",
                            staff_chat_id,
                        )

            except Exception as e:
                logger.exception("Ошибка подготовки эскалации сотрудникам: %s", e)
            finally:
                self.escalation_queue.task_done()


async def main() -> None:
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN environment variable not set")

    scenarios_file = os.environ.get("SCENARIOS_FILE", "scen_v5.yaml")
    if not Path(scenarios_file).exists():
        raise RuntimeError(f"Scenarios file not found: {scenarios_file}")

    staff_chat_id_env = os.environ.get("STAFF_CHAT_ID")
    if not staff_chat_id_env:
        raise RuntimeError("STAFF_CHAT_ID environment variable not set")
    staff_chat_ids = parse_chat_ids(staff_chat_id_env, "STAFF_CHAT_ID")

    group_chat_id_env = os.environ.get("GROUP_CHAT_ID")
    group_chat_id = int(group_chat_id_env) if group_chat_id_env else None

    intents = load_intents(scenarios_file)
    fallback_intent = intents.get("fallback")
    if not fallback_intent:
        raise RuntimeError("No fallback intent found in scenarios file")

    admissions_bot = AdmissionsBot(
        intents=intents,
        fallback_intent=fallback_intent,
        staff_chat_ids=staff_chat_ids,
        group_chat_id=group_chat_id,
    )

    application = ApplicationBuilder().token(token).build()

    application.bot_data["admissions_bot"] = admissions_bot
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, admissions_bot.handle_message)
    )

    logger.info("Starting bot (manual async lifecycle)...")

    worker_task: Optional[asyncio.Task] = None

    await application.initialize()
    await application.start()

    worker_task = asyncio.create_task(admissions_bot.escalation_worker(application))
    logger.info("Escalation worker task created")

    await application.updater.start_polling()

    try:
        await asyncio.Event().wait()
    finally:
        await application.updater.stop()

        if worker_task:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task

        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")