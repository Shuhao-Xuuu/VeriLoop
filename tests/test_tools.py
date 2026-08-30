from __future__ import annotations

import pytest

from veriloop.protocol import ErrorKind, ToolCall
from veriloop.tools import (
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
    contains_known_secret,
    register_completion_tool,
)


def schema(**properties: dict[str, str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def register_identity(
    registry: ToolRegistry,
    *,
    input_schema: dict[str, object] | None = None,
) -> None:
    registry.register(
        ToolSpec(
            name="identity",
            description="Return the supplied arguments",
            input_schema=input_schema or schema(value={"type": "string"}),
            handler=lambda arguments: arguments,
        )
    )


def test_known_secret_detection_covers_nested_keys_values_and_cycles() -> None:
    secret = "provider-secret-for-test"
    cyclic: list[object] = []
    cyclic.append(cyclic)
    cyclic.append({f"prefix-{secret}-suffix": ("safe",)})

    assert contains_known_secret(cyclic, ("", None, secret))  # type: ignore[arg-type]
    assert contains_known_secret({"outer": [f"prefix-{secret}-suffix"]}, (secret,))
    assert not contains_known_secret({"outer": ["unrelated"]}, (secret,))
    assert not contains_known_secret(cyclic, ("different-secret",))


def test_register_and_schemas_hide_handler() -> None:
    registry = ToolRegistry()
    register_identity(registry)

    assert registry.schemas() == [
        {
            "type": "function",
            "function": {
                "name": "identity",
                "description": "Return the supplied arguments",
                "parameters": schema(value={"type": "string"}),
            },
        }
    ]
    assert "handler" not in repr(registry.schemas())


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    register_identity(registry)

    with pytest.raises(ValueError, match="already registered"):
        register_identity(registry)


def test_normal_execution_and_success_id_are_preserved() -> None:
    registry = ToolRegistry()
    register_identity(registry)

    result = registry.execute(
        ToolCall(id="call-7", name="identity", arguments={"value": "hello"})
    )

    assert result.ok is True
    assert result.call_id == "call-7"
    assert result.tool_name == "identity"
    assert result.content == '{"value": "hello"}'
    assert result.error_kind is None


def test_unknown_tool_returns_error_and_preserves_id() -> None:
    result = ToolRegistry().execute(
        ToolCall(id="missing-1", name="missing", arguments={})
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.UNKNOWN_TOOL
    assert result.call_id == "missing-1"
    assert result.tool_name == "missing"


def test_arguments_must_be_an_object() -> None:
    registry = ToolRegistry()
    register_identity(registry)
    call = ToolCall(id="bad-object", name="identity", arguments=["not-object"])  # type: ignore[arg-type]

    result = registry.execute(call)

    assert result.error_kind is ErrorKind.INVALID_ARGUMENTS
    assert result.call_id == "bad-object"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "is required"),
        ({"value": 5}, "must be of type string"),
        ({"value": "ok", "extra": True}, "is not allowed"),
    ],
)
def test_required_type_and_extra_field_validation(
    arguments: dict[str, object], message: str
) -> None:
    registry = ToolRegistry()
    register_identity(registry)

    result = registry.execute(
        ToolCall(id="invalid-1", name="identity", arguments=arguments)
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.INVALID_ARGUMENTS
    assert message in result.content
    assert result.call_id == "invalid-1"


@pytest.mark.parametrize("json_type", ["integer", "number"])
def test_bool_is_not_an_integer_or_number(json_type: str) -> None:
    registry = ToolRegistry()
    register_identity(
        registry,
        input_schema=schema(value={"type": json_type}),
    )

    result = registry.execute(
        ToolCall(id=f"bool-{json_type}", name="identity", arguments={"value": True})
    )

    assert result.error_kind is ErrorKind.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    ("json_type", "value"),
    [
        ("integer", 4),
        ("number", 4.5),
        ("boolean", False),
        ("array", [1, 2]),
        ("object", {}),
    ],
)
def test_supported_basic_types(json_type: str, value: object) -> None:
    registry = ToolRegistry()
    register_identity(
        registry,
        input_schema=schema(value={"type": json_type}),
    )

    result = registry.execute(
        ToolCall(id="typed", name="identity", arguments={"value": value})
    )

    assert result.ok is True


def test_handler_exception_becomes_tool_error() -> None:
    registry = ToolRegistry()

    def explode(arguments: dict[str, object]) -> str:
        raise RuntimeError(f"boom: {arguments['value']}")

    registry.register(
        ToolSpec(
            name="explode",
            description="Always fail",
            input_schema=schema(value={"type": "string"}),
            handler=explode,
        )
    )

    result = registry.execute(
        ToolCall(id="failure-id", name="explode", arguments={"value": "x"})
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.TOOL_ERROR
    assert result.call_id == "failure-id"
    assert result.tool_name == "explode"
    assert "RuntimeError" in result.content


def test_additional_properties_can_be_explicitly_allowed() -> None:
    registry = ToolRegistry()
    register_identity(
        registry,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    )

    result = registry.execute(
        ToolCall(id="extra-ok", name="identity", arguments={"anything": 1})
    )

    assert result.ok is True


def test_registry_marks_only_successful_mutating_specs_as_invalidating() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="mutate",
            description="Test mutation fact",
            input_schema=schema(value={"type": "string"}),
            handler=lambda arguments: "changed",
            mutates_workspace=True,
        )
    )

    success = registry.execute(
        ToolCall(id="success", name="mutate", arguments={"value": "x"})
    )
    invalid = registry.execute(
        ToolCall(id="invalid", name="mutate", arguments={})
    )

    assert success.invalidates_verification is True
    assert invalid.invalidates_verification is False


def test_registry_preserves_host_reported_mutation_on_expected_failure() -> None:
    registry = ToolRegistry()

    def fail_after_start(arguments):
        raise ToolExecutionError(
            ErrorKind.COMMAND_NONZERO_EXIT,
            "started then failed",
            invalidates_verification=True,
        )

    registry.register(
        ToolSpec(
            name="mutate",
            description="Test mutation fact",
            input_schema=schema(value={"type": "string"}),
            handler=fail_after_start,
            mutates_workspace=True,
        )
    )

    result = registry.execute(
        ToolCall(id="failed", name="mutate", arguments={"value": "x"})
    )

    assert result.ok is False
    assert result.invalidates_verification is True


def test_complete_task_schema_rejects_model_supplied_verification_fields() -> None:
    registry = ToolRegistry()
    register_completion_tool(registry)
    schema = registry.schemas()[0]["function"]["parameters"]

    valid = registry.execute(
        ToolCall(
            id="valid",
            name="complete_task",
            arguments={"summary": "implemented", "remaining_risks": "none"},
        )
    )
    forged = registry.execute(
        ToolCall(
            id="forged",
            name="complete_task",
            arguments={"summary": "implemented", "verified": True},
        )
    )

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"summary", "remaining_risks"}
    assert valid.ok is True
    assert valid.invalidates_verification is False
    assert forged.error_kind is ErrorKind.INVALID_ARGUMENTS
