import asyncio
import json
import keyword
import os
import re
import shlex
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model


DEFAULT_MCP_TIMEOUT_SECONDS = 30
DEFAULT_MCP_RESULT_MAX_CHARS = 12000
GENERIC_ARGUMENTS_FIELD = "arguments"


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    timeout_seconds: int = DEFAULT_MCP_TIMEOUT_SECONDS


def normalize_mcp_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    safe = safe.strip("_")
    return safe or "mcp"


def is_mcp_enabled() -> bool:
    raw = os.getenv("TRACEPILOT_MCP_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_args(raw_args: str) -> list[str]:
    if not raw_args.strip():
        return []
    try:
        return shlex.split(raw_args, posix=os.name != "nt")
    except ValueError:
        return raw_args.split()


def _load_env_subset(raw_env_keys: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in raw_env_keys.split(","):
        key = key.strip()
        if key and key in os.environ:
            env[key] = os.environ[key]
    return env


def load_mcp_server_config_from_env() -> Optional[MCPServerConfig]:
    if not is_mcp_enabled():
        return None

    command = os.getenv("TRACEPILOT_MCP_SERVER_COMMAND", "").strip()
    if not command:
        print(" [MCP] TRACEPILOT_MCP_ENABLED=true but TRACEPILOT_MCP_SERVER_COMMAND is empty.")
        return None

    raw_timeout = os.getenv("TRACEPILOT_MCP_TIMEOUT_SECONDS", str(DEFAULT_MCP_TIMEOUT_SECONDS))
    try:
        timeout_seconds = max(1, int(raw_timeout))
    except ValueError:
        timeout_seconds = DEFAULT_MCP_TIMEOUT_SECONDS

    return MCPServerConfig(
        name=normalize_mcp_name(os.getenv("TRACEPILOT_MCP_SERVER_NAME", "web_search")),
        command=command,
        args=_parse_args(os.getenv("TRACEPILOT_MCP_SERVER_ARGS", "")),
        env=_load_env_subset(os.getenv("TRACEPILOT_MCP_SERVER_ENV_KEYS", "")),
        timeout_seconds=timeout_seconds,
    )


def _stream_has_usable_fileno(stream: Any) -> bool:
    try:
        fileno = getattr(stream, "fileno", None)
        if not callable(fileno):
            return False
        fileno()
        return True
    except Exception:
        return False


def _select_process_errlog() -> Any:
    for stream in (getattr(sys, "__stderr__", None), getattr(sys, "stderr", None)):
        if stream is not None and _stream_has_usable_fileno(stream):
            return stream
    return None


class MCPEventLoopRuntime:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._loop and self._loop.is_running():
                return

            self._ready.clear()

            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._ready.set()
                loop.run_forever()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()

            self._thread = threading.Thread(target=run_loop, daemon=True, name="tracepilot-mcp-runtime")
            self._thread.start()
            self._ready.wait(timeout=5)

    def run(self, coro: Any, timeout_seconds: int | None = None) -> Any:
        self.start()
        if self._loop is None:
            raise RuntimeError("MCP event loop failed to start.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            raise

    def stop(self) -> None:
        with self._lock:
            if not self._loop:
                return
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None


class MCPStdioClient:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.session: Any = None
        self._stdio_context: Any = None
        self._session_context: Any = None
        self._errlog = _select_process_errlog()

    async def connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
        )
        self._stdio_context = stdio_client(server, errlog=self._errlog)
        read_stream, write_stream = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
        )
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()

    async def list_tools(self) -> Any:
        if self.session is None:
            raise RuntimeError("MCP server is not connected.")
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("MCP server is not connected.")
        return await self.session.call_tool(
            tool_name,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
        )

    async def close(self) -> None:
        try:
            if self._session_context is not None:
                await self._session_context.__aexit__(None, None, None)
        finally:
            self._session_context = None
            self.session = None
            if self._stdio_context is not None:
                await self._stdio_context.__aexit__(None, None, None)
                self._stdio_context = None

    def stderr_tail(self, max_chars: int = 4000) -> str:
        getvalue = getattr(self._errlog, "getvalue", None)
        if not callable(getvalue):
            return ""
        try:
            value = getvalue()
            return value[-max_chars:]
        except Exception:
            return ""


class MCPManager:
    def __init__(self, runtime: MCPEventLoopRuntime | None = None) -> None:
        self.runtime = runtime or MCPEventLoopRuntime()
        self.clients: dict[str, MCPStdioClient] = {}
        self._call_lock = threading.Lock()

    def connect_server(self, config: MCPServerConfig) -> list[Any]:
        client = MCPStdioClient(config)
        try:
            self.runtime.run(client.connect(), timeout_seconds=config.timeout_seconds + 10)
            result = self.runtime.run(client.list_tools(), timeout_seconds=config.timeout_seconds)
            self.clients[config.name] = client
            return list(getattr(result, "tools", []) or [])
        except Exception as exc:
            try:
                self.runtime.run(client.close(), timeout_seconds=5)
            except Exception:
                pass
            stderr = client.stderr_tail()
            if stderr:
                print(f" [MCP] Failed to load server {config.name}: {exc}\n{stderr}")
            else:
                print(f" [MCP] Failed to load server {config.name}: {exc}")
            return []

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        client = self.clients.get(server_name)
        if client is None:
            return f"MCP tool error: server '{server_name}' is not connected."

        try:
            with self._call_lock:
                result = self.runtime.run(
                    client.call_tool(tool_name, arguments),
                    timeout_seconds=client.config.timeout_seconds + 5,
                )
            return format_mcp_call_result(result)
        except Exception as exc:
            stderr = client.stderr_tail()
            detail = f" stderr: {stderr}" if stderr else ""
            return f"MCP tool error: {exc}{detail}"

    def shutdown(self) -> None:
        for client in list(self.clients.values()):
            try:
                self.runtime.run(client.close(), timeout_seconds=5)
            except Exception:
                pass
        self.clients.clear()
        self.runtime.stop()


def _json_dumps(value: Any) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def format_mcp_call_result(result: Any, max_chars: int = DEFAULT_MCP_RESULT_MAX_CHARS) -> str:
    parts: list[str] = []
    if getattr(result, "isError", False):
        parts.append("[MCP tool returned an error]")

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        parts.append(_json_dumps(structured))

    for item in getattr(result, "content", []) or []:
        item_type = getattr(item, "type", "")
        if item_type == "text":
            parts.append(getattr(item, "text", ""))
        elif item_type == "image":
            mime_type = getattr(item, "mimeType", "unknown")
            parts.append(f"[MCP image content omitted: {mime_type}]")
        elif item_type == "resource":
            resource = getattr(item, "resource", None)
            text = getattr(resource, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(f"[MCP resource content: {_json_dumps(resource)}]")
        else:
            parts.append(_json_dumps(item))

    text = "\n\n".join(part for part in parts if part)
    if not text:
        text = "(MCP tool returned no content.)"
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[MCP result truncated]..."
    return text


def _json_schema_type_to_python(prop_schema: dict[str, Any]) -> Any:
    prop_type = prop_schema.get("type")
    if isinstance(prop_type, list):
        prop_type = next((item for item in prop_type if item != "null"), prop_type[0] if prop_type else None)

    if prop_type == "string":
        return str
    if prop_type == "integer":
        return int
    if prop_type == "number":
        return float
    if prop_type == "boolean":
        return bool
    if prop_type == "array":
        return list[Any]
    return Any


def _is_simple_input_schema(input_schema: Any) -> bool:
    if not isinstance(input_schema, dict):
        return False
    if input_schema.get("type", "object") != "object":
        return False
    properties = input_schema.get("properties", {})
    if not isinstance(properties, dict):
        return False
    for prop_schema in properties.values():
        if not isinstance(prop_schema, dict):
            return False
        if any(key in prop_schema for key in ("anyOf", "allOf", "oneOf")):
            return False
        prop_type = prop_schema.get("type")
        if isinstance(prop_type, list):
            allowed = {"string", "integer", "number", "boolean", "array", "null"}
            if any(item not in allowed for item in prop_type):
                return False
        elif prop_type not in {None, "string", "integer", "number", "boolean", "array"}:
            return False
    return True


def create_args_model(server_name: str, tool_name: str, input_schema: Any) -> Type[BaseModel]:
    safe_server = normalize_mcp_name(server_name)
    safe_tool = normalize_mcp_name(tool_name)
    model_name = f"MCP_{safe_server}_{safe_tool}_Args"

    if not _is_simple_input_schema(input_schema):
        return create_model(
            model_name,
            **{
                GENERIC_ARGUMENTS_FIELD: (
                    dict[str, Any],
                    Field(default_factory=dict, description="Raw MCP tool arguments as a JSON object."),
                )
            },
        )

    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = set(input_schema.get("required", []) or []) if isinstance(input_schema, dict) else set()
    fields: dict[str, tuple[Any, Any]] = {}

    for name, prop_schema in properties.items():
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            return create_model(
                model_name,
                **{
                    GENERIC_ARGUMENTS_FIELD: (
                        dict[str, Any],
                        Field(default_factory=dict, description="Raw MCP tool arguments as a JSON object."),
                    )
                },
            )
        field_type = _json_schema_type_to_python(prop_schema if isinstance(prop_schema, dict) else {})
        description = prop_schema.get("description", "") if isinstance(prop_schema, dict) else ""
        if name in required:
            default = ...
        elif isinstance(prop_schema, dict) and "default" in prop_schema:
            default = prop_schema["default"]
        else:
            default = None
        fields[name] = (field_type, Field(default, description=description))

    return create_model(model_name, **fields)


class MCPTool(BaseTool):
    manager: Any = Field(exclude=True)
    server_name: str
    original_tool_name: str
    args_schema: Type[BaseModel]

    def _run(self, **kwargs: Any) -> str:
        arguments = kwargs
        if set(kwargs.keys()) == {GENERIC_ARGUMENTS_FIELD} and isinstance(kwargs.get(GENERIC_ARGUMENTS_FIELD), dict):
            arguments = kwargs[GENERIC_ARGUMENTS_FIELD]
        return self.manager.call_tool(self.server_name, self.original_tool_name, arguments)

    async def _arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self._run, **kwargs)


def create_mcp_tool(manager: MCPManager, server_name: str, tool: Any) -> MCPTool:
    original_name = getattr(tool, "name", "")
    safe_server = normalize_mcp_name(server_name)
    safe_tool = normalize_mcp_name(original_name)
    description = getattr(tool, "description", "") or f"MCP tool {original_name}"
    input_schema = getattr(tool, "inputSchema", None)
    args_schema = create_args_model(safe_server, safe_tool, input_schema)
    return MCPTool(
        manager=manager,
        server_name=safe_server,
        original_tool_name=original_name,
        name=f"mcp__{safe_server}__{safe_tool}",
        description=f"[MCP:{safe_server}] {description}",
        args_schema=args_schema,
    )


_GLOBAL_MANAGER: MCPManager | None = None


def load_mcp_tools() -> list[BaseTool]:
    config = load_mcp_server_config_from_env()
    if config is None:
        return []

    try:
        import mcp  # noqa: F401
    except ImportError:
        print(" [MCP] Official MCP Python SDK is not installed. 请在已确认的虚拟环境中安装 mcp>=1.0.0,<2.0.0 后重试。")
        return []

    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.shutdown()

    manager = MCPManager()
    tools = manager.connect_server(config)
    if not tools:
        manager.shutdown()
        _GLOBAL_MANAGER = None
        return []

    _GLOBAL_MANAGER = manager
    wrapped_tools = [create_mcp_tool(manager, config.name, tool) for tool in tools]
    print(f" [MCP] Loaded {len(wrapped_tools)} tool(s) from server {config.name}.")
    return wrapped_tools


def shutdown_mcp_runtime() -> None:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.shutdown()
        _GLOBAL_MANAGER = None
