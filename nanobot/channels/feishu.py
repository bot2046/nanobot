"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import mimetypes
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((/[^)\s]+)\)")
_FILE_URI_RE = re.compile(r"\bfile:(/[^\s]+)")


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tenant_access_token: str | None = None
        self._token_expire_at: float = 0.0

    @staticmethod
    def _extract_explicit_attachments(text: str) -> tuple[str, list[str]]:
        attachments: list[str] = []

        def replace_md(match: re.Match[str]) -> str:
            attachments.append(match.group(1))
            return ""

        def replace_file(match: re.Match[str]) -> str:
            attachments.append(match.group(1))
            return ""

        cleaned = _MD_IMAGE_RE.sub(replace_md, text)
        cleaned = _FILE_URI_RE.sub(replace_file, cleaned)
        return cleaned.strip(), attachments

    @staticmethod
    def _normalize_attachment_paths(paths: list[str]) -> list[Path]:
        normalized: list[Path] = []
        seen: set[str] = set()
        for raw in paths:
            if not isinstance(raw, str) or raw in seen:
                continue
            seen.add(raw)
            path = Path(raw)
            if path.is_absolute() and path.is_file():
                normalized.append(path)
        return normalized

    async def _get_tenant_access_token(self) -> str | None:
        now = time.time()
        if self._tenant_access_token and now < self._token_expire_at:
            return self._tenant_access_token

        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.config.app_id,
            "app_secret": self.config.app_secret,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(token_url, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to get access token: status {response.status_code}")
                return None
            data = response.json()
            if data.get("code") != 0:
                logger.error(f"Failed to get access token: {data.get('msg')}")
                return None
            token = data.get("tenant_access_token")
            expire = data.get("expire", 0)
            if not token:
                logger.error("Failed to get access token: missing token")
                return None
            # refresh slightly early
            self._tenant_access_token = token
            self._token_expire_at = now + max(0, int(expire) - 60)
            return token

    async def _upload_image_http(self, path: Path) -> str | None:
        token = await self._get_tenant_access_token()
        if not token:
            return None

        api_url = "https://open.feishu.cn/open-apis/im/v1/images"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with path.open("rb") as f:
                files = {"image": (path.name, f)}
                data = {"image_type": "message"}
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(api_url, headers=headers, data=data, files=files)
            if response.status_code != 200:
                logger.warning(
                    f"Image upload failed: status={response.status_code}, body={response.text[:500]}"
                )
                return None
            payload = response.json()
            if payload.get("code") != 0:
                logger.warning(f"Image upload error: {payload.get('msg')}")
                return None
            return payload.get("data", {}).get("image_key")
        except Exception as e:
            logger.error(f"Image upload error: {e}")
            return None

    async def _upload_file_http(self, path: Path) -> str | None:
        token = await self._get_tenant_access_token()
        if not token:
            return None

        api_url = "https://open.feishu.cn/open-apis/im/v1/files"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with path.open("rb") as f:
                files = {"file": (path.name, f)}
                data = {"file_type": "stream", "file_name": path.name}
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(api_url, headers=headers, data=data, files=files)
            if response.status_code != 200:
                logger.warning(
                    f"File upload failed: status={response.status_code}, body={response.text[:500]}"
                )
                return None
            payload = response.json()
            if payload.get("code") != 0:
                logger.warning(f"File upload error: {payload.get('msg')}")
                return None
            return payload.get("data", {}).get("file_key")
        except Exception as e:
            logger.error(f"File upload error: {e}")
            return None
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread
        def run_ws():
            try:
                self._ws_client.start()
            except Exception as e:
                logger.error(f"Feishu WebSocket error: {e}")
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        def split_row(row: str) -> list[str]:
            return [c.strip() for c in row.strip("|").split("|")]
        headers = split_row(lines[0])
        rows = [split_row(row) for row in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})
        return elements or [{"tag": "markdown", "content": content}]

    @staticmethod
    def _guess_is_image(path: Path) -> bool:
        mime, _ = mimetypes.guess_type(path.as_posix())
        return bool(mime and mime.startswith("image/"))

    async def _send_image(self, receive_id_type: str, receive_id: str, image_key: str) -> None:
        request = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            ).build()
        response = self._client.im.v1.message.create(request)
        if not response.success():
            logger.warning(
                f"Failed to send Feishu image: code={response.code}, msg={response.msg}, "
                f"log_id={response.get_log_id()}"
            )

    async def _send_file(self, receive_id_type: str, receive_id: str, file_key: str) -> None:
        request = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("file")
                .content(json.dumps({"file_key": file_key}))
                .build()
            ).build()
        response = self._client.im.v1.message.create(request)
        if not response.success():
            logger.warning(
                f"Failed to send Feishu file: code={response.code}, msg={response.msg}, "
                f"log_id={response.get_log_id()}"
            )

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return
        
        try:
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"
            
            cleaned_text, extracted = self._extract_explicit_attachments(msg.content)
            attachments = []
            if msg.media:
                attachments.extend(msg.media)
            if extracted:
                attachments.extend(extracted)

            normalized = self._normalize_attachment_paths(attachments)

            if cleaned_text.strip():
                elements = self._build_card_elements(cleaned_text)
                card = {
                    "config": {"wide_screen_mode": True},
                    "elements": elements,
                }
                content = json.dumps(card, ensure_ascii=False)

                request = CreateMessageRequest.builder() \
                    .receive_id_type(receive_id_type) \
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(msg.chat_id)
                        .msg_type("interactive")
                        .content(content)
                        .build()
                    ).build()

                response = self._client.im.v1.message.create(request)

                if not response.success():
                    logger.error(
                        f"Failed to send Feishu message: code={response.code}, "
                        f"msg={response.msg}, log_id={response.get_log_id()}"
                    )
                else:
                    logger.debug(f"Feishu message sent to {msg.chat_id}")

            for path in normalized:
                try:
                    size = path.stat().st_size
                    if self._guess_is_image(path):
                        if size > 10 * 1024 * 1024:
                            logger.warning(f"Image too large (>10MB): {path}")
                            continue
                        image_key = await self._upload_image_http(path)
                        if image_key:
                            await self._send_image(receive_id_type, msg.chat_id, image_key)
                        else:
                            logger.warning(f"Image upload failed: {path}")
                    else:
                        if size > 30 * 1024 * 1024:
                            logger.warning(f"File too large (>30MB): {path}")
                            continue
                        file_key = await self._upload_file_http(path)
                        if file_key:
                            await self._send_file(receive_id_type, msg.chat_id, file_key)
                        else:
                            logger.warning(f"File upload failed: {path}")
                except Exception as e:
                    logger.error(f"Error sending attachment {path}: {e}")

        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "THUMBSUP")
            
            # Parse message content
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
            
            if not content:
                return
            
            # Forward to message bus
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
