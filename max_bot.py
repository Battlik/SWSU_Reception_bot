import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import yaml
from dotenv import load_dotenv

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


def escape_markdown(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'([\\`*_\[\]()~>#+\-=|{}.!])', r'\\\1', text)


def build_max_user_deeplink(user_id: int) -> Optional[str]:
    if not user_id or user_id <= 0:
        return None
    return f"max://user/{user_id}"


def build_max_user_markdown(user_id: int, user_name: str) -> str:
    safe_name = escape_markdown(user_name or "Пользователь")
    deeplink = build_max_user_deeplink(user_id)
    if not deeplink:
        return safe_name
    return f"[{safe_name}]({deeplink})"


class Intent:
    def __init__(
        self,
        name: str,
        triggers: List[str],
        response: str,
        priority: float = 0,
    ) -> None:
        self.name = name
        self.response = response
        self.priority = priority
        self.patterns = [
            re.compile(pattern, re.IGNORECASE | re.UNICODE)
            for pattern in triggers
        ]

    def match(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


def load_intents(path: str) -> Dict[str, "Intent"]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    items = sorted(
        data.get("intents", []),
        key=lambda x: float(x.get("priority", 0)),
        reverse=True,
    )

    intents: Dict[str, Intent] = {}
    for item in items:
        name = item.get("name")
        triggers = item.get("triggers", [])
        response = item.get("response", "")
        priority = item.get("priority", 0)

        if not name:
            continue

        intents[name] = Intent(
            name=name,
            triggers=triggers,
            response=response,
            priority=priority,
        )
    return intents


class MaxBotAPI:
    BASE_URL = "https://platform-api.max.ru"

    def __init__(self, token: str) -> None:
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(
            total=70,
            connect=10,
            sock_connect=10,
            sock_read=40,
        )
        self.session = aiohttp.ClientSession(
            headers={"Authorization": self.token},
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def get_me(self) -> dict:
        assert self.session is not None
        async with self.session.get(f"{self.BASE_URL}/me") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_updates(self, marker: Optional[int] = None) -> dict:
        assert self.session is not None

        params = {
            "timeout": 30,
            "limit": 100,
        }
        if marker is not None:
            params["marker"] = marker

        async with self.session.get(f"{self.BASE_URL}/updates", params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def send_message(
        self,
        text: str,
        *,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        format: Optional[str] = None,
    ) -> dict:
        assert self.session is not None

        params = {}
        if user_id is not None:
            params["user_id"] = user_id
        if chat_id is not None:
            params["chat_id"] = chat_id

        payload = {
            "text": text,
            "notify": True,
        }
        if format:
            payload["format"] = format

        async with self.session.post(
            f"{self.BASE_URL}/messages",
            params=params,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


class AdmissionsBot:
    def __init__(
        self,
        api: MaxBotAPI,
        intents: Dict[str, "Intent"],
        fallback_intent: "Intent",
        staff_chat_ids: List[int],
        bot_user_id: Optional[int] = None,
    ) -> None:
        self.api = api
        self.intents = intents
        self.fallback_intent = fallback_intent
        self.staff_chat_ids = staff_chat_ids
        self.staff_chat_ids_set: Set[int] = set(staff_chat_ids)
        self.bot_user_id = bot_user_id
        self.escalation_queue: asyncio.Queue = asyncio.Queue()

    def is_staff_chat(self, chat_id: Optional[int]) -> bool:
        return chat_id in self.staff_chat_ids_set if chat_id is not None else False

    async def handle_update(self, update: Dict[str, Any]) -> None:
        update_type = update.get("update_type")
        logger.info("update_type=%s update=%s", update_type, update)

        if update_type == "bot_started":
            chat_id = update.get("chat_id")
            if chat_id:
                await self.api.send_message(
                    "Здравствуйте! Я виртуальный помощник приёмной комиссии ЮЗГУ. "
                    "Напишите ваш вопрос.",
                    chat_id=chat_id,
                )
            return

        if update_type != "message_created":
            return

        message = update.get("message") or {}
        body = message.get("body") or {}
        recipient = message.get("recipient") or {}
        sender = message.get("sender") or {}

        text = body.get("text")
        if not text:
            return

        text_normalized = normalize_text(text)
        chat_id = recipient.get("chat_id")

        user_id = sender.get("user_id")
        first_name = sender.get("first_name") or ""
        last_name = sender.get("last_name") or ""
        username = sender.get("username")
        is_bot = bool(sender.get("is_bot"))

        if is_bot:
            logger.info("Ignoring bot message from sender=%s", user_id)
            return

        if self.bot_user_id is not None and user_id == self.bot_user_id:
            logger.info("Ignoring self message from bot_user_id=%s", user_id)
            return

        if self.is_staff_chat(chat_id):
            logger.info("Ignoring message from staff chat_id=%s", chat_id)
            return

        full_name = f"{first_name} {last_name}".strip()
        if not full_name:
            full_name = username or f"Пользователь {user_id}"

        if not chat_id:
            logger.warning("Не найден chat_id в update: %s", update)
            return

        if text_normalized == "/start":
            await self.api.send_message(
                "Здравствуйте! Я виртуальный помощник приёмной комиссии ЮЗГУ. "
                "Напишите ваш вопрос.",
                chat_id=chat_id,
            )
            return

        matched_intent: Optional[Intent] = None
        for name, intent in self.intents.items():
            if name == self.fallback_intent.name:
                continue
            if intent.match(text_normalized):
                matched_intent = intent
                break

        if matched_intent:
            logger.info("Matched intent %s: %s", matched_intent.name, text)
            await self.api.send_message(
                matched_intent.response,
                chat_id=chat_id,
            )
            return

        await self.handle_complex_question(
            chat_id=chat_id,
            user_id=user_id or 0,
            user_name=full_name,
            text=text,
        )

    async def handle_complex_question(
        self,
        *,
        chat_id: int,
        user_id: int,
        user_name: str,
        text: str,
    ) -> None:
        await self.api.send_message(
            "Ваш вопрос передан специалисту приёмной комиссии. "
            "Пожалуйста, ожидайте ответа.",
            chat_id=chat_id,
        )

        await self.escalation_queue.put(
            {
                "user_id": user_id,
                "user_name": user_name,
                "chat_id": chat_id,
                "text": text,
            }
        )

    async def escalation_worker(self) -> None:
        logger.info("Escalation worker started")
        while True:
            item = await self.escalation_queue.get()
            try:
                user_id = item["user_id"]
                user_name = item["user_name"]
                chat_id = item["chat_id"]
                text = item["text"]

                user_mention = build_max_user_markdown(user_id, user_name)
                user_link = build_max_user_deeplink(user_id)

                if user_link:
                    safe_link_text = escape_markdown(user_link)
                    link_line = f"Полная ссылка: [{safe_link_text}]({user_link})"
                else:
                    link_line = "Полная ссылка: недоступна"

                staff_notification = (
                    f"⚠️ Сложный вопрос\n"
                    f"Пользователь: {user_mention}\n"
                    f"{link_line}\n"
                    f"ID пользователя: {user_id}\n"
                    f"Чат ID: {chat_id}\n"
                    f"Сообщение: {escape_markdown(text)}"
                )

                for staff_chat_id in self.staff_chat_ids:
                    try:
                        await self.api.send_message(
                            staff_notification,
                            chat_id=staff_chat_id,
                            format="markdown",
                        )
                        logger.info(
                            "Escalation sent to staff chat_id=%s",
                            staff_chat_id,
                        )
                    except Exception:
                        logger.exception(
                            "Ошибка отправки сотруднику chat_id=%s",
                            staff_chat_id,
                        )
            except Exception:
                logger.exception("Ошибка подготовки эскалации сотрудникам")
            finally:
                self.escalation_queue.task_done()

    async def polling_loop(self) -> None:
        marker: Optional[int] = None
        logger.info("MAX polling loop started")

        while True:
            try:
                data = await self.api.get_updates(marker=marker)
                updates = data.get("updates", [])
                marker = data.get("marker", marker)

                for update in updates:
                    try:
                        await self.handle_update(update)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("Ошибка обработки update: %s", update)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Ошибка long polling")
                await asyncio.sleep(2)


async def main() -> None:
    token = os.environ.get("MAX_BOT_TOKEN")
    if not token:
        raise RuntimeError("MAX_BOT_TOKEN environment variable not set")

    scenarios_file = os.environ.get("SCENARIOS_FILE", "scen_v5.yaml")
    if not Path(scenarios_file).exists():
        raise RuntimeError(f"Scenarios file not found: {scenarios_file}")

    staff_chat_id_env = os.environ.get("STAFF_CHAT_ID_MAX")
    if not staff_chat_id_env:
        raise RuntimeError("STAFF_CHAT_ID_MAX environment variable not set")
    staff_chat_ids = parse_chat_ids(staff_chat_id_env, "STAFF_CHAT_ID_MAX")

    intents = load_intents(scenarios_file)
    fallback_intent = intents.get("fallback")
    if not fallback_intent:
        raise RuntimeError("No fallback intent found in scenarios file")

    async with MaxBotAPI(token) as api:
        me = await api.get_me()
        bot_user_id = me.get("user_id")
        logger.info(
            "Авторизация успешна: bot_id=%s username=%s",
            bot_user_id,
            me.get("username"),
        )

        bot = AdmissionsBot(
            api=api,
            intents=intents,
            fallback_intent=fallback_intent,
            staff_chat_ids=staff_chat_ids,
            bot_user_id=bot_user_id,
        )

        worker_task = asyncio.create_task(bot.escalation_worker())
        polling_task = asyncio.create_task(bot.polling_loop())

        try:
            await asyncio.gather(worker_task, polling_task)
        finally:
            for task in (worker_task, polling_task):
                task.cancel()
            await asyncio.gather(worker_task, polling_task, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")