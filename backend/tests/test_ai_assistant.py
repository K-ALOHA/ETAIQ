from pathlib import Path

import pytest

from app.ai.assistant import ETAIQAssistantService, Intent
from app.ai.schemas import AssistantRequest, AssistantResponse


@pytest.mark.asyncio
async def test_assistant_returns_structured_response() -> None:
    service = ETAIQAssistantService(history_limit=3)

    request = AssistantRequest(message="help", conversation_id=None)
    response = await service.handle_message(request)

    assert isinstance(response, AssistantResponse)
    assert "help" in response.response.lower()
    assert response.conversation_id


@pytest.mark.asyncio
async def test_help_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="help", conversation_id=None)
    response = await service.handle_message(request)

    assert "I can help you with" in response.response


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

    assert any(keyword in response.response for keyword in ["records", "features", "target"])


@pytest.mark.asyncio
async def test_explain_prediction_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="explain the prediction", conversation_id=None)
    response = await service.handle_message(request)

    assert any(keyword in response.response for keyword in ["ETA", "because", "driving", "provide the value"])


@pytest.mark.asyncio
async def test_feature_importance_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="what are the important features?", conversation_id=None)
    response = await service.handle_message(request)

    assert any(keyword in response.response for keyword in ["feature", "importance", "top"])


@pytest.mark.asyncio
async def test_compare_models_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="compare models", conversation_id=None)
    response = await service.handle_message(request)

    assert any(keyword in response.response for keyword in ["model", "registered"])


@pytest.mark.asyncio
async def test_unknown_intent() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="what's your favorite color?", conversation_id=None)
    response = await service.handle_message(request)

    assert "not sure" in response.response


@pytest.mark.asyncio
async def test_prediction_flow_starts() -> None:
    service = ETAIQAssistantService()

    request = AssistantRequest(message="predict eta", conversation_id=None)
    response = await service.handle_message(request)

    assert "provide the value" in response.response


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


@pytest.mark.asyncio
async def test_explain_prediction_without_last_prediction_starts_flow() -> None:
    """When no last prediction exists, explain intent should start prediction flow."""
    service = ETAIQAssistantService()
    request = AssistantRequest(message="explain why the prediction", conversation_id="test_conv_1")
    response = await service.handle_message(request)
    assert "You haven't made any predictions yet" in response.response
    assert "provide the value" in response.response


@pytest.mark.asyncio
async def test_last_prediction_is_stored() -> None:
    """After setting last prediction directly should allow explain intent to reference it."""
    service = ETAIQAssistantService()
    service._last_prediction = {"prediction": 12.34, "features": {}, "confidence": None}
    request = AssistantRequest(message="explain the prediction", conversation_id="test_conv_2")
    response = await service.handle_message(request)
    assert "Your last prediction was 12.34 minutes" in response.response


@pytest.mark.asyncio
async def test_prediction_to_feature_importance_switch() -> None:
    """Test switching from prediction flow to feature importance intent."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_1"
    
    # Start prediction flow
    request1 = AssistantRequest(message="predict eta", conversation_id=conv_id)
    response1 = await service.handle_message(request1)
    assert "provide the value" in response1.response
    assert conv_id in service._prediction_state
    
    # Now send feature importance request - should cancel prediction and handle new intent
    request2 = AssistantRequest(message="show feature importance", conversation_id=conv_id)
    response2 = await service.handle_message(request2)
    assert conv_id not in service._prediction_state
    assert any(keyword in response2.response for keyword in ["feature", "importance", "top"])


@pytest.mark.asyncio
async def test_prediction_to_model_metrics_switch() -> None:
    """Test switching from prediction flow to performance metrics intent."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_2"
    
    # Start prediction flow
    request1 = AssistantRequest(message="estimate eta", conversation_id=conv_id)
    response1 = await service.handle_message(request1)
    assert "provide the value" in response1.response
    assert conv_id in service._prediction_state
    
    # Now send performance metrics request
    request2 = AssistantRequest(message="show model metrics", conversation_id=conv_id)
    response2 = await service.handle_message(request2)
    assert conv_id not in service._prediction_state
    assert any(keyword in response2.response for keyword in ["MAE", "RMSE", "R²"])


@pytest.mark.asyncio
async def test_prediction_to_explain_switch() -> None:
    """Test switching from prediction flow to explain prediction intent."""
    service = ETAIQAssistantService()
    service._last_prediction = {"prediction": 42.0, "features": {}, "confidence": None}
    conv_id = "test_switch_3"
    
    # Start prediction flow
    request1 = AssistantRequest(message="predict eta", conversation_id=conv_id)
    response1 = await service.handle_message(request1)
    assert "provide the value" in response1.response
    assert conv_id in service._prediction_state
    
    # Now send explain prediction request
    request2 = AssistantRequest(message="explain latest prediction", conversation_id=conv_id)
    response2 = await service.handle_message(request2)
    assert conv_id not in service._prediction_state
    assert "Your last prediction was 42.00 minutes" in response2.response


@pytest.mark.asyncio
async def test_prediction_to_dataset_summary_switch() -> None:
    """Test switching from prediction flow to dataset summary intent."""
    service = ETAIQAssistantService()
    conv_id = "test_switch_4"
    
    # Start prediction flow
    request1 = AssistantRequest(message="what's the eta", conversation_id=conv_id)
    response1 = await service.handle_message(request1)
    assert "provide the value" in response1.response
    assert conv_id in service._prediction_state
    
    # Now send dataset summary request
    request2 = AssistantRequest(message="summarize dataset", conversation_id=conv_id)
    response2 = await service.handle_message(request2)
    assert conv_id not in service._prediction_state
    assert any(keyword in response2.response for keyword in ["records", "features", "target"])

