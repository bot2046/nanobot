"""Factory for creating context builder instances with plugin support."""

import importlib
import sys
from pathlib import Path
from typing import Any, Type, Protocol, runtime_checkable


@runtime_checkable
class ContextBuilderProtocol(Protocol):
    """Protocol for context builder implementations.

    External plugins must implement this protocol to be compatible with nanobot.
    """

    def __init__(self, workspace: Path) -> None:
        """Initialize the context builder.

        Args:
            workspace: Path to the agent's workspace directory.
        """
        ...

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt for the agent.

        Args:
            skill_names: Optional list of skills to include.

        Returns:
            Complete system prompt string.
        """
        ...

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        ...

    def add_tool_result(
        self, messages: list[dict[str, Any]], tool_call_id: str, tool_name: str, result: str
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.

        Returns:
            Updated message list.
        """
        ...

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).

        Returns:
            Updated message list.
        """
        ...


class ExtendedContextBuilderProtocol(ContextBuilderProtocol, Protocol):
    """Extended protocol with optional methods for more advanced plugins.

    Plugins can implement these methods to provide additional functionality.
    """

    def get_available_methods(self) -> list[str]:
        """Get list of available custom methods provided by this plugin.

        Returns:
            List of method names that can be called dynamically.
        """
        ...

    def call_custom_method(self, method_name: str, *args, **kwargs) -> Any:
        """Call a custom method provided by the plugin.

        Args:
            method_name: Name of the method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Result of the method call.

        Raises:
            AttributeError: If the method is not available.
        """
        ...

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate plugin-specific configuration.

        Args:
            config: Configuration dictionary for the plugin.

        Returns:
            True if configuration is valid, False otherwise.
        """
        ...


class ContextBuilderFactory:
    """Factory for creating context builder instances with plugin support."""

    @staticmethod
    def create(
        workspace: Path,
        context_provider_package: str = "nanobot.agent.context",
        context_provider_class: str = "ContextBuilder",
        plugin_config: dict[str, Any] | None = None,
    ) -> ContextBuilderProtocol:
        """Create a context builder instance, optionally from a plugin.

        Args:
            workspace: Path to the agent's workspace directory.
            context_provider_package: Python package containing the context builder class.
            context_provider_class: Name of the context builder class.
            plugin_config: Optional configuration dictionary for the plugin.

        Returns:
            An instance of the context builder class.

        Raises:
            ImportError: If the specified package or class cannot be imported.
            TypeError: If the loaded class doesn't implement ContextBuilderProtocol.
        """
        try:
            # Import the module
            module = importlib.import_module(context_provider_package)

            # Get the class
            builder_class = getattr(module, context_provider_class)

            # Verify it implements the protocol
            if not isinstance(builder_class, type):
                raise TypeError(f"{context_provider_class} is not a class")

            # Check if it's a subclass of the default ContextBuilder (for backward compatibility)
            from nanobot.agent.context import ContextBuilder as DefaultContextBuilder

            is_subclass = issubclass(builder_class, DefaultContextBuilder)

            # For non-subclasses, verify they implement the protocol
            if not is_subclass:
                # Create a temporary instance to check protocol compliance
                temp_instance = builder_class(workspace)
                if not isinstance(temp_instance, ContextBuilderProtocol):
                    raise TypeError(
                        f"{context_provider_class} does not implement ContextBuilderProtocol. "
                        f"Required methods: __init__, build_system_prompt, build_messages, "
                        f"add_tool_result, add_assistant_message"
                    )

            # Create the instance with optional plugin config
            if plugin_config:
                # Try to pass plugin config if the class supports it
                try:
                    instance = builder_class(workspace, **plugin_config)
                except TypeError:
                    # Fall back to default initialization
                    instance = builder_class(workspace)
            else:
                instance = builder_class(workspace)

            return instance

        except ImportError as e:
            raise ImportError(
                f"Failed to import context builder from {context_provider_package}.{context_provider_class}: {e}"
            )

    @staticmethod
    def validate_plugin(
        context_provider_package: str,
        context_provider_class: str,
    ) -> tuple[bool, str]:
        """Validate that a plugin can be loaded and implements the required protocol.

        Args:
            context_provider_package: Python package containing the context builder class.
            context_provider_class: Name of the context builder class.

        Returns:
            Tuple of (is_valid, error_message). is_valid is True if plugin is valid.
        """
        try:
            # Try to create an instance with a dummy workspace
            dummy_workspace = Path("/tmp/dummy")
            instance = ContextBuilderFactory.create(
                workspace=dummy_workspace,
                context_provider_package=context_provider_package,
                context_provider_class=context_provider_class,
            )

            # Check required methods
            required_methods = [
                "build_system_prompt",
                "build_messages",
                "add_tool_result",
                "add_assistant_message",
            ]

            for method in required_methods:
                if not hasattr(instance, method):
                    return False, f"Missing required method: {method}"
                if not callable(getattr(instance, method)):
                    return False, f"Method {method} is not callable"

            return True, "Plugin is valid"

        except Exception as e:
            return False, f"Plugin validation failed: {e}"

    @staticmethod
    def get_default_builder() -> Type[ContextBuilderProtocol]:
        """Get the default context builder class.

        Returns:
            The default ContextBuilder class.
        """
        from nanobot.agent.context import ContextBuilder

        return ContextBuilder

    @staticmethod
    def list_available_plugins(search_paths: list[str] | None = None) -> list[dict[str, str]]:
        """Discover available context builder plugins in the given search paths.

        Args:
            search_paths: List of directory paths to search for plugins.
                If None, uses sys.path.

        Returns:
            List of dictionaries with plugin information.
        """
        plugins = []

        if search_paths is None:
            search_paths = sys.path

        # This is a simple implementation - in a real system you might want
        # to look for specific entry points or metadata files
        for path in search_paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                # Look for Python modules that might be context plugins
                for py_file in path_obj.glob("**/*context*.py"):
                    # Skip __pycache__ and test files
                    if "__pycache__" in str(py_file) or "test" in py_file.stem.lower():
                        continue

                    # Try to determine if this is a context plugin
                    # (This is heuristic - a real system would use entry points)
                    try:
                        # Extract module path
                        rel_path = py_file.relative_to(path_obj)
                        module_parts = list(rel_path.with_suffix("").parts)
                        module_name = ".".join(module_parts)

                        # Skip if it's our own module
                        if module_name.startswith("nanobot.agent.context"):
                            continue

                        plugins.append(
                            {
                                "module": module_name,
                                "file": str(py_file),
                                "type": "potential",  # We don't know for sure
                            }
                        )
                    except (ValueError, ImportError):
                        continue

        return plugins


