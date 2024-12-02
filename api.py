from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
from datetime import datetime

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import Config
from unseen_predictor import (
    get_bert_model,
    extract_bert_features,
    load_finetune_model,
    softmax
)
from interaction_analyzer import InteractionAnalyzer
import utils.dataset_processors as dataset_processors

app = FastAPI(title="Personality Prediction API")

class TwitterInteraction(BaseModel):
    content: str
    timestamp: datetime
    interaction_type: str  # 'post', 'reply', 'quote', 'retweet'
    reply_to_user: Optional[str] = None
    sentiment_context: Optional[str] = None

class TwitterUserData(BaseModel):
    username: str
    interactions: List[TwitterInteraction]

class PersonalityPrediction(BaseModel):
    username: str
    predictions: Dict[str, float]
    interaction_patterns: Dict
    behavioral_insights: Dict[str, str]
    confidence_score: float
    analysis_timestamp: datetime

def combine_user_interactions(interactions: List[TwitterInteraction]) -> str:
    """Combine all user interactions into a single text for analysis."""
    return " ".join([interaction.content for interaction in interactions])

def calculate_confidence_score(predictions: dict, interaction_patterns: dict) -> float:
    """Calculate confidence score based on prediction probabilities and interaction data quality."""
    prediction_confidence = np.mean([abs(prob - 0.5) * 2 for prob in predictions.values()])
    
    # Factor in interaction pattern quality
    pattern_confidence = (
        interaction_patterns["response_behavior"]["consistent_responder"] * 0.3 +
        interaction_patterns["sentiment_patterns"]["sentiment_stability"] * 0.3 +
        interaction_patterns["conversation_depth"]["sustained_conversations"] * 0.4
    )
    
    return (prediction_confidence + pattern_confidence) / 2

def generate_behavioral_insights(predictions: Dict[str, float], patterns: Dict) -> Dict[str, str]:
    """Generate human-readable insights based on predictions and interaction patterns."""
    insights = {}
    
    # Communication Style
    comm_style = []
    if patterns["interaction_style"]["inquisitiveness"] > 0.6:
        comm_style.append("highly inquisitive")
    if patterns["interaction_style"]["detail_orientation"] > 0.6:
        comm_style.append("detail-oriented")
    if patterns["interaction_style"]["supportiveness"] > 0.6:
        comm_style.append("supportive")
    insights["communication_style"] = f"Communication style is {', '.join(comm_style)}" if comm_style else "Balanced communication style"

    # Social Engagement
    engagement_level = patterns["social_engagement"]["engagement_ratio"]
    if engagement_level > 0.7:
        insights["social_engagement"] = "Highly engaged with others, frequently participates in discussions"
    elif engagement_level > 0.4:
        insights["social_engagement"] = "Moderately engaged, balanced between initiating and responding"
    else:
        insights["social_engagement"] = "More reserved, selective in engagement"

    # Emotional Expression
    sentiment_patterns = patterns["sentiment_patterns"]
    if sentiment_patterns["emotional_variability"] > 0.6:
        insights["emotional_expression"] = "Expresses emotions dynamically, showing significant emotional range"
    elif sentiment_patterns["sentiment_stability"] > 0.7:
        insights["emotional_expression"] = "Maintains consistent emotional tone in communications"
    else:
        insights["emotional_expression"] = "Moderate emotional expression with balanced stability"

    # Conversation Depth
    conv_depth = patterns["conversation_depth"]
    if conv_depth["sustained_conversations"] > 0.6:
        insights["conversation_depth"] = "Engages deeply in conversations, often maintaining extended dialogues"
    elif conv_depth["avg_conversation_depth"] > 2:
        insights["conversation_depth"] = "Maintains moderate conversation depth with balanced engagement"
    else:
        insights["conversation_depth"] = "Tends toward briefer, more focused interactions"

    # Response Patterns
    response_behavior = patterns["response_behavior"]
    if response_behavior["quick_responses_ratio"] > 0.7:
        insights["response_pattern"] = "Highly responsive, often engaging quickly in conversations"
    elif response_behavior["consistent_responder"] > 0.7:
        insights["response_pattern"] = "Consistently engaged, maintains reliable response patterns"
    else:
        insights["response_pattern"] = "Varied response patterns, more selective in engagement timing"

    return insights

@app.post("/predict/personality", response_model=PersonalityPrediction)
async def predict_personality(user_data: TwitterUserData):
    if len(user_data.interactions) < Config.MIN_INTERACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {Config.MIN_INTERACTIONS} interactions required for accurate prediction"
        )
    
    # Initialize interaction analyzer
    analyzer = InteractionAnalyzer()
    
    # Analyze interaction patterns
    interaction_patterns = analyzer.analyze_interaction_patterns(user_data.interactions)
    
    # Combine all interactions into single text
    combined_text = combine_user_interactions(user_data.interactions)
    
    # Preprocess text
    preprocessed_text = dataset_processors.preprocess_text(combined_text)
    
    # Get BERT model and tokenizer
    tokenizer, model = get_bert_model(Config.DEFAULT_EMBED)
    
    # Extract BERT features
    embeddings = extract_bert_features(
        preprocessed_text,
        tokenizer,
        model,
        Config.DEFAULT_TOKEN_LENGTH
    )
    
    # Load models and make predictions
    models = load_finetune_model(
        Config.OP_DIR,
        Config.DEFAULT_FINETUNE_MODEL,
        Config.DEFAULT_DATASET
    )
    
    base_predictions = {}
    for trait, model in models.items():
        try:
            prediction = model.predict(embeddings)
            prediction = softmax(prediction)
            base_predictions[trait] = float(prediction[0][1])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for trait {trait}: {str(e)}"
            )
    
    # Enrich predictions with interaction patterns
    enriched_predictions = analyzer.enrich_personality_prediction(
        base_predictions,
        interaction_patterns
    )
    
    # Calculate confidence score
    confidence = calculate_confidence_score(enriched_predictions, interaction_patterns)
    
    # Generate behavioral insights
    behavioral_insights = generate_behavioral_insights(enriched_predictions, interaction_patterns)
    
    return PersonalityPrediction(
        username=user_data.username,
        predictions=enriched_predictions,
        interaction_patterns=interaction_patterns,
        behavioral_insights=behavioral_insights,
        confidence_score=confidence,
        analysis_timestamp=datetime.utcnow()
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )
