import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import yaml
from dotenv import load_dotenv

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


def escape_markdown(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'([\\`*_$begin:math:display$$end:math:display$()~>#+\-=|{}.!])', r'\\\1', text)


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
    scenarios_path = Path(path)
    if not scenarios_path.exists():
        raise RuntimeError(f"Scenarios file not found: {path}")

    with scenarios_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

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

        if not isinstance(triggers, list):
            raise RuntimeError(f"Intent '{name}' must contain a list in 'triggers'")

        intents[name] = Intent(
            name=name,
            triggers=triggers,
            response=response,
            priority=priority,
        )

    if "fallback" not in intents:
        intents["fallback"] = Intent(
            name="fallback",
            triggers=[],
            response="",
            priority=-1,
        )

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

    def is_ignored(self, user_id: int) -> bool:
        key = str(user_id)
        until_ts = self._data.get(key)
        if until_ts is None:
            return False

        if until_ts <= time.time():
            self._data.pop(key, None)
            self._save()
            return False

        return True

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

    def set_paused(self, value: bool) -> None:
        self.paused = bool(value)
        self.storage.save({"paused": self.paused})


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

        params: Dict[str, Any] = {
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

        params: Dict[str, Any] = {}
        if user_id is not None:
            params["user_id"] = user_id
        if chat_id is not None:
            params["chat_id"] = chat_id

        payload: Dict[str, Any] = {
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
        intents: Dict[str, Intent],
        fallback_intent: Intent,
        staff_ids: List[int],
        bot_user_id: Optional[int],
        ignore_ttl_seconds: int,
        ignore_state_file: str,
        bot_state_file: str,
        cleanup_interval_seconds: int,
    ) -> None:
        self.api = api
        self.intents = intents
        self.fallback_intent = fallback_intent

        # STAFF_CHAT_ID_MAX используем и как список сотрудников,
        # и как список получателей уведомлений в ЛС
        self.staff_ids = staff_ids
        self.staff_ids_set: Set[int] = set(staff_ids)

        self.bot_user_id = bot_user_id

        self.ignore_ttl_seconds = int(ignore_ttl_seconds)
        self.cleanup_interval_seconds = max(60, int(cleanup_interval_seconds))

        self.ignore_store = IgnoreStore(ignore_state_file)
        self.control_state = BotControlState(bot_state_file)

        self.escalation_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def is_staff(self, user_id: Optional[int]) -> bool:
        return user_id in self.staff_ids_set if user_id is not None else False

    def is_paused(self) -> bool:
        return self.control_state.paused

    def set_paused(self, value: bool) -> None:
        self.control_state.set_paused(value)

    async def send_user_start(self, chat_id: int) -> None:
        if self.is_paused():
            await self.api.send_message(
                "Бот временно находится на паузе. Попробуйте написать позже.",
                chat_id=chat_id,
            )
            return

        await self.api.send_message(
            "Здравствуйте! Я виртуальный помощник приёмной комиссии ЮЗГУ. "
            "Напишите ваш вопрос. Если вопрос сложный, я передам его специалисту.",
            chat_id=chat_id,
        )

    async def handle_admin_command(
        self,
        text_normalized: str,
        chat_id: int,
        user_id: int,
    ) -> bool:
        if not self.is_staff(user_id):
            return False

        if text_normalized == "/pause":
            if self.is_paused():
                await self.api.send_message(
                    "Бот уже находится на паузе.",
                    chat_id=chat_id,
                )
            else:
                self.set_paused(True)
                await self.api.send_message(
                    "Бот переведён в режим паузы.\n"
                    "Обычные пользователи сейчас полностью игнорируются.",
                    chat_id=chat_id,
                )
                logger.warning("Bot paused by staff user_id=%s", user_id)
            return True

        if text_normalized == "/resume":
            if not self.is_paused():
                await self.api.send_message(
                    "Бот уже активен.",
                    chat_id=chat_id,
                )
            else:
                self.set_paused(False)
                await self.api.send_message(
                    "Бот снова активен.\n"
                    "Обработка сообщений возобновлена.",
                    chat_id=chat_id,
                )
                logger.warning("Bot resumed by staff user_id=%s", user_id)
            return True

        if text_normalized == "/status":
            status_text = "на паузе" if self.is_paused() else "активен"
            await self.api.send_message(
                f"Статус бота: {status_text}\n"
                f"Пользователей в игноре: {self.ignore_store.count()}\n"
                f"Интервал очистки: {self.cleanup_interval_seconds} сек.",
                chat_id=chat_id,
            )
            return True

        return False

    async def handle_update(self, update: Dict[str, Any]) -> None:
        update_type = update.get("update_type")
        logger.info("update_type=%s", update_type)

        if update_type == "bot_started":
            chat_id = update.get("chat_id")
            if chat_id:
                await self.send_user_start(chat_id)
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

        if not chat_id:
            logger.warning("Не найден chat_id в update: %s", update)
            return

        if user_id is None:
            logger.warning("Не найден user_id в update: %s", update)
            return

        if is_bot:
            logger.info("Ignoring bot message from sender=%s", user_id)
            return

        if self.bot_user_id is not None and user_id == self.bot_user_id:
            logger.info("Ignoring self message from bot_user_id=%s", user_id)
            return

        if await self.handle_admin_command(text_normalized, chat_id, user_id):
            return

        if text_normalized == "/start":
            if self.is_staff(user_id):
                await self.api.send_message(
                    "Для управления ботом используйте:\n"
                    "/pause — поставить бота на паузу\n"
                    "/resume — снять паузу\n"
                    "/status — проверить статус",
                    chat_id=chat_id,
                )
            else:
                await self.send_user_start(chat_id)
            return

        # Сотрудников игнорируем как обычных пользователей
        if self.is_staff(user_id):
            logger.info(
                "Ignoring regular message from staff user_id=%s in chat_id=%s",
                user_id,
                chat_id,
            )
            return

        if self.is_paused():
            logger.info(
                "Ignoring message because bot is paused: user_id=%s chat_id=%s",
                user_id,
                chat_id,
            )
            return

        if self.ignore_store.is_ignored(user_id):
            logger.info(
                "Ignoring message from user_id=%s due to active cooldown (%s sec left)",
                user_id,
                self.ignore_store.remaining_seconds(user_id),
            )
            return

        full_name = f"{first_name} {last_name}".strip()
        if not full_name:
            full_name = username or f"Пользователь {user_id}"

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
            user_id=user_id,
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

        self.ignore_store.set_ignore(user_id, self.ignore_ttl_seconds)

        await self.escalation_queue.put(
            {
                "user_id": user_id,
                "user_name": user_name,
                "chat_id": chat_id,
                "text": text,
            }
        )

        logger.info(
            "Escalated user_id=%s chat_id=%s and set cooldown for %s seconds",
            user_id,
            chat_id,
            self.ignore_ttl_seconds,
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

                for staff_id in self.staff_ids:
                    try:
                        # STAFF_CHAT_ID_MAX трактуем как user_id сотрудника
                        await self.api.send_message(
                            staff_notification,
                            user_id=staff_id,
                            format="markdown",
                        )
                        logger.info(
                            "Escalation sent to staff_id=%s for user_id=%s",
                            staff_id,
                            user_id,
                        )
                    except Exception:
                        logger.exception(
                            "Ошибка отправки сотруднику staff_id=%s",
                            staff_id,
                        )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Ошибка подготовки эскалации сотрудникам")
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
    token = os.getenv("MAX_BOT_TOKEN")
    if not token:
        raise RuntimeError("MAX_BOT_TOKEN environment variable not set")

    staff_id_env = os.getenv("STAFF_CHAT_ID_MAX")
    if not staff_id_env:
        raise RuntimeError("STAFF_CHAT_ID_MAX environment variable not set")

    scenarios_file = os.getenv("SCENARIOS_FILE", "scen_v5.yaml")
    if not Path(scenarios_file).exists():
        raise RuntimeError(f"Scenarios file not found: {scenarios_file}")

    ignore_ttl_seconds = int(os.getenv("IGNORE_TTL_SECONDS", "1800"))
    ignore_state_file = os.getenv("IGNORE_STATE_FILE_MAX", "ignored_users_max.json")
    bot_state_file = os.getenv("BOT_STATE_FILE_MAX", "bot_state_max.json")
    cleanup_interval_seconds = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))

    staff_ids = parse_int_list(staff_id_env, "STAFF_CHAT_ID_MAX")

    intents = load_intents(scenarios_file)
    fallback_intent = intents["fallback"]

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
            staff_ids=staff_ids,
            bot_user_id=bot_user_id,
            ignore_ttl_seconds=ignore_ttl_seconds,
            ignore_state_file=ignore_state_file,
            bot_state_file=bot_state_file,
            cleanup_interval_seconds=cleanup_interval_seconds,
        )

        worker_task = asyncio.create_task(bot.escalation_worker())
        polling_task = asyncio.create_task(bot.polling_loop())
        cleanup_task = asyncio.create_task(bot.cleanup_worker())

        try:
            await asyncio.gather(worker_task, polling_task, cleanup_task)
        finally:
            for task in (worker_task, polling_task, cleanup_task):
                task.cancel()
            await asyncio.gather(
                worker_task,
                polling_task,
                cleanup_task,
                return_exceptions=True,
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")