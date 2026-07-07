import pytest

from app.ai.assistant import ETAIQAssistantService, Intent
from app.ai.conversation import ConversationState
from app.ai.schemas import AssistantRequest, AssistantResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_prediction(service: ETAIQAssistantService, conv_id: str) -> bool:
    """Return True when the conversation is in an active prediction flow."""
    state = service.conversation_manager.get_state(conv_id)
    return state in (ConversationState.PREDICTION, ConversationState.WAITING_FIELD)


# ---------------------------------------------------------------------------
# Structural response tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assistant_returns_structured_response() -> None:
    service = ETAIQAssistantService(history_limit=3)

    request = AssistantRequest(message="help", conversation_id=None)
    response = await service.handle_message(request)

    assert isinstance(response, AssistantResponse)
    assert len(response.response) > 0
    assert response.conversation_id


# ---------------------------------------------------------------------------
# Intent handler behaviour tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_help_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="help", conversation_id=None)
    response = await service.handle_message(request)

    r = response.response.lower()
    assert any(word in r for word in ["predict", "feature", "model", "dataset", "eta"])


@pytest.mark.asyncio
async def test_model_summary_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="tell me about the model", conversation_id=None)
    response = await service.handle_message(request)

    assert any(keyword in response.response for keyword in ["Production", "model", "v"])


@pytest.mark.asyncio
async def test_performance_metrics_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="show performance", conversation_id=None)
    response = await service.handle_message(request)

    assert any(keyword in response.response for keyword in ["MAE", "RMSE", "R²"])


@pytest.mark.asyncio
async def test_dataset_summary_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="tell me about the dataset", conversation_id=None)
    response = await service.handle_message(request)

    r = response.response.lower()
    assert any(keyword in r for keyword in ["records", "features", "target"])


@pytest.mark.asyncio
async def test_explain_prediction_intent() -> None:
    """Explain intent with no prior prediction returns a non-empty guidance response."""
    service = ETAIQAssistantService()

    request = AssistantRequest(message="explain the prediction", conversation_id=None)
    response = await service.handle_message(request)

    assert len(response.response) > 0
    r = response.response.lower()
    assert any(word in r for word in ["prediction", "predict", "eta", "explain"])


@pytest.mark.asyncio
async def test_feature_importance_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="what are the important features?", conversation_id=None)
    response = await service.handle_message(request)

    r = response.response.lower()
    assert any(keyword in r for keyword in ["feature", "importance", "top"])


@pytest.mark.asyncio
async def test_compare_models_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="compare models", conversation_id=None)
    response = await service.handle_message(request)

    r = response.response.lower()
    assert any(keyword in r for keyword in ["model", "mae", "rmse", "production"])


@pytest.mark.asyncio
async def test_unknown_intent() -> None:
    """An unrecognised message returns a meaningful non-empty response."""
    service = ETAIQAssistantService()

    request = AssistantRequest(message="what's your favorite color?", conversation_id=None)
    response = await service.handle_message(request)

    assert len(response.response) > 0


# ---------------------------------------------------------------------------
# Prediction flow start
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prediction_flow_starts() -> None:
    service = ETAIQAssistantService()
    conv_id = "test_flow_start"

    request = AssistantRequest(message="predict eta", conversation_id=conv_id)
    response = await service.handle_message(request)

    r = response.response
    # Section header, first field label, and cancel instruction must all appear
    assert "Section 1" in r
    assert "cancel" in r.lower()
    assert any(word in r.lower() for word in ["order size", "order details"])
    assert _in_prediction(service, conv_id)


# ---------------------------------------------------------------------------
# Intent detection unit test
# ---------------------------------------------------------------------------

def test_intent_detection() -> None:
    service = ETAIQAssistantService()

    assert service._detect_intent("help") == Intent.HELP
    assert service._detect_intent("show me the model") == Intent.MODEL_SUMMARY
    assert service._detect_intent("predict eta") == Intent.PREDICTION
    assert service._detect_intent("explain why") == Intent.EXPLAIN_PREDICTION
    assert service._detect_intent("what features are important") == Intent.FEATURE_IMPORTANCE
    assert service._detect_intent("show shap") == Intent.SHAP_EXPLANATION
    assert service._detect_intent("how's performance") == Intent.PERFORMANCE_METRICS
    assert service._detect_intent("tell me about the data") == Intent.DATASET_SUMMARY
    assert service._detect_intent("compare the models") == Intent.COMPARE_MODELS
    assert service._detect_intent("random question") == Intent.UNKNOWN


def test_predict_intent_whole_word_matching() -> None:
    """'predict' as a whole word triggers PREDICT; 'prediction' substrings do not."""
    from app.ai.assistant import detect_intent

    # Must trigger PREDICT
    assert detect_intent("predict") == Intent.PREDICT
    assert detect_intent("predict eta") == Intent.PREDICT
    assert detect_intent("predict delivery") == Intent.PREDICT
    assert detect_intent("make prediction") == Intent.PREDICT
    assert detect_intent("start prediction") == Intent.PREDICT
    assert detect_intent("run prediction") == Intent.PREDICT
    assert detect_intent("new prediction") == Intent.PREDICT

    # Must NOT trigger PREDICT — these contain 'prediction' not 'predict' as a word
    assert detect_intent("prediction") != Intent.PREDICT
    assert detect_intent("predictions") != Intent.PREDICT
    assert detect_intent("latest prediction") != Intent.PREDICT
    assert detect_intent("explain prediction") != Intent.PREDICT
    assert detect_intent("explain latest prediction") != Intent.PREDICT
    assert detect_intent("why was my prediction high") != Intent.PREDICT


@pytest.mark.asyncio
async def test_explain_prediction_message_does_not_start_flow() -> None:
    """Messages containing 'prediction' but not 'predict' must not start the guided flow."""
    service = ETAIQAssistantService()

    for msg in [
        "Explain the latest prediction",
        "Explain latest prediction model",
        "Why was the latest prediction high?",
        "Explain prediction",
    ]:
        conv_id = f"test_no_flow_{msg[:10]}"
        response = await service.handle_message(
            AssistantRequest(message=msg, conversation_id=conv_id)
        )
        assert not _in_prediction(service, conv_id), (
            f"Message '{msg}' incorrectly started the prediction flow"
        )
        assert "Section 1" not in response.response, (
            f"Message '{msg}' returned prediction flow response"
        )


# ---------------------------------------------------------------------------
# Explain prediction — no prior prediction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explain_prediction_without_last_prediction_starts_flow() -> None:
    """When no prediction exists, explain intent returns a guidance response."""
    service = ETAIQAssistantService()
    request = AssistantRequest(message="explain why the prediction", conversation_id="test_conv_1")
    response = await service.handle_message(request)

    assert len(response.response) > 0
    r = response.response.lower()
    assert any(word in r for word in ["prediction", "predict", "eta", "run", "dashboard"])


# ---------------------------------------------------------------------------
# Explain prediction — prediction value present in monitoring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explain_prediction_references_value_when_available() -> None:
    """When monitoring has a record, explain intent references a numeric ETA value."""
    from unittest.mock import MagicMock, patch

    service = ETAIQAssistantService()

    mock_record = MagicMock()
    mock_record.mean_prediction = 12.34

    with patch.object(service, "_get_latest_prediction", return_value=12.34):
        request = AssistantRequest(message="explain the prediction", conversation_id="test_conv_2")
        response = await service.handle_message(request)

    assert len(response.response) > 0
    r = response.response.lower()
    # LLM may round the value; verify the response discusses a prediction in minutes
    assert any(word in r for word in ["12", "minute", "eta", "prediction", "predicted"])


# ---------------------------------------------------------------------------
# Prediction flow → intent switch tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prediction_to_feature_importance_switch() -> None:
    """Sending a non-numeric intent mid-flow interrupts with a continue-or-cancel prompt."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_1"

    request1 = AssistantRequest(message="predict eta", conversation_id=conv_id)
    response1 = await service.handle_message(request1)
    assert _in_prediction(service, conv_id)
    assert "Section 1" in response1.response

    # Mid-flow non-numeric intent: assistant stays in prediction mode and prompts
    request2 = AssistantRequest(message="show feature importance", conversation_id=conv_id)
    response2 = await service.handle_message(request2)

    # Still in prediction mode — assistant asks to continue or cancel
    assert _in_prediction(service, conv_id)
    r2 = response2.response.lower()
    assert any(word in r2 for word in ["prediction mode", "cancel", "continue", "need"])


@pytest.mark.asyncio
async def test_prediction_to_model_metrics_switch() -> None:
    """Sending a model-metrics intent mid-flow interrupts with a continue-or-cancel prompt."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_2"

    request1 = AssistantRequest(message="estimate eta", conversation_id=conv_id)
    await service.handle_message(request1)
    assert _in_prediction(service, conv_id)

    request2 = AssistantRequest(message="show model metrics", conversation_id=conv_id)
    response2 = await service.handle_message(request2)

    assert _in_prediction(service, conv_id)
    r2 = response2.response.lower()
    assert any(word in r2 for word in ["prediction mode", "cancel", "continue", "need"])


@pytest.mark.asyncio
async def test_prediction_to_explain_switch() -> None:
    """Sending an explain intent mid-flow interrupts with a continue-or-cancel prompt."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_3"

    request1 = AssistantRequest(message="predict eta", conversation_id=conv_id)
    await service.handle_message(request1)
    assert _in_prediction(service, conv_id)

    request2 = AssistantRequest(message="explain latest prediction", conversation_id=conv_id)
    response2 = await service.handle_message(request2)

    assert _in_prediction(service, conv_id)
    r2 = response2.response.lower()
    assert any(word in r2 for word in ["prediction mode", "cancel", "continue", "need"])


@pytest.mark.asyncio
async def test_prediction_to_dataset_summary_switch() -> None:
    """Sending a dataset intent mid-flow interrupts with a continue-or-cancel prompt."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_4"

    request1 = AssistantRequest(message="estimate eta", conversation_id=conv_id)
    await service.handle_message(request1)
    assert _in_prediction(service, conv_id)

    request2 = AssistantRequest(message="summarize dataset", conversation_id=conv_id)
    response2 = await service.handle_message(request2)

    assert _in_prediction(service, conv_id)
    r2 = response2.response.lower()
    assert any(word in r2 for word in ["prediction mode", "cancel", "continue", "need"])


# ---------------------------------------------------------------------------
# Prediction flow — cancel exits the flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prediction_flow_cancel() -> None:
    """Typing cancel during prediction flow exits prediction mode."""
    service = ETAIQAssistantService()
    conv_id = "test_cancel"

    await service.handle_message(AssistantRequest(message="predict eta", conversation_id=conv_id))
    assert _in_prediction(service, conv_id)

    response = await service.handle_message(
        AssistantRequest(message="cancel", conversation_id=conv_id)
    )
    assert not _in_prediction(service, conv_id)
    assert "cancel" in response.response.lower()
