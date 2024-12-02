# Enhanced Personality Prediction API

This API provides sophisticated personality prediction capabilities based on Twitter-like social media interactions. It performs deep analysis of user behavior patterns, interaction styles, and communication patterns to provide rich personality insights.

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and adjust settings if needed:
```bash
cp .env.example .env
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Start the API server:
```bash
poetry run python api.py
```

## API Endpoints

### POST /predict/personality

Predicts personality traits and provides detailed behavioral analysis based on user interactions.

#### Request Format

```json
{
  "username": "example_user",
  "interactions": [
    {
      "content": "This is a tweet about an interesting topic!",
      "timestamp": "2023-08-10T12:00:00Z",
      "interaction_type": "post",
      "reply_to_user": null,
      "sentiment_context": null
    },
    {
      "content": "I disagree with your point. Here's why...",
      "timestamp": "2023-08-10T12:05:00Z",
      "interaction_type": "reply",
      "reply_to_user": "other_user",
      "sentiment_context": "disagreement"
    }
  ]
}
```

#### Response Format

```json
{
  "username": "example_user",
  "predictions": {
    "EXT": 0.75,  // Extraversion
    "NEU": 0.45,  // Neuroticism
    "AGR": 0.82,  // Agreeableness
    "CON": 0.68,  // Conscientiousness
    "OPN": 0.90   // Openness
  },
  "interaction_patterns": {
    "response_behavior": {
      "avg_response_time": 300,
      "quick_responses_ratio": 0.7,
      "consistent_responder": 0.8
    },
    "sentiment_patterns": {
      "sentiment_stability": 0.75,
      "positive_ratio": 0.6,
      "negative_ratio": 0.2,
      "emotional_variability": 0.3,
      "sentiment_progression": "stable"
    },
    "interaction_style": {
      "inquisitiveness": 0.65,
      "agreeableness": 0.7,
      "confrontational": 0.2,
      "detail_orientation": 0.8,
      "supportiveness": 0.75
    },
    "conversation_depth": {
      "avg_conversation_depth": 3.5,
      "max_conversation_depth": 8,
      "sustained_conversations": 0.6
    },
    "social_engagement": {
      "network_breadth": 15,
      "strong_connections": 5,
      "engagement_ratio": 0.7
    }
  },
  "behavioral_insights": {
    "communication_style": "Detail-oriented and highly inquisitive",
    "social_engagement": "Highly engaged with others, frequently participates in discussions",
    "emotional_expression": "Maintains consistent emotional tone in communications",
    "conversation_depth": "Engages deeply in conversations, often maintaining extended dialogues",
    "response_pattern": "Consistently engaged, maintains reliable response patterns"
  },
  "confidence_score": 0.85,
  "analysis_timestamp": "2023-08-10T12:10:00Z"
}
```

## Sentiment Analysis

### Automatic Sentiment Analysis
The API automatically analyzes sentiment in two ways:

1. **Built-in Sentiment Analysis**
   - Uses TextBlob to analyze the sentiment of each interaction
   - Automatically detects:
     - Positive/negative sentiment
     - Emotional intensity
     - Opinion strength
   - You don't need to provide sentiment_context in your requests

2. **Optional Manual Context**
   - You can optionally provide sentiment_context in your requests
   - Useful when you have additional context about the interaction
   - Will be combined with automatic analysis (60-40 weight)

### Example Requests

1. **Basic Post with Automatic Sentiment Analysis**
```bash
curl -X POST "http://localhost:8000/predict/personality" \
     -H "Content-Type: application/json" \
     -d '{
  "username": "example_user",
  "interactions": [
    {
      "content": "Just learned about a fascinating new technology!",
      "timestamp": "2023-08-10T12:00:00Z",
      "interaction_type": "post"
    },
    {
      "content": "That's an interesting perspective. I think...",
      "timestamp": "2023-08-10T12:05:00Z",
      "interaction_type": "reply",
      "reply_to_user": "other_user"
    }
  ]
}'
```

2. **Complex Interaction Pattern with Explicit Sentiment**
```bash
curl -X POST "http://localhost:8000/predict/personality" \
     -H "Content-Type: application/json" \
     -d '{
  "username": "example_user",
  "interactions": [
    {
      "content": "This new AI development could revolutionize healthcare!",
      "timestamp": "2023-08-10T12:00:00Z",
      "interaction_type": "post",
      "sentiment_context": "excited"
    },
    {
      "content": "While I see your point, I think we need to consider the ethical implications...",
      "timestamp": "2023-08-10T12:05:00Z",
      "interaction_type": "reply",
      "reply_to_user": "tech_enthusiast",
      "sentiment_context": "concerned"
    },
    {
      "content": "You raise some valid concerns. Let's explore solutions together.",
      "timestamp": "2023-08-10T12:10:00Z",
      "interaction_type": "reply",
      "reply_to_user": "ethics_expert",
      "sentiment_context": "supportive"
    }
  ]
}'
```

3. **Extended Conversation Thread**
```bash
curl -X POST "http://localhost:8000/predict/personality" \
     -H "Content-Type: application/json" \
     -d '{
  "username": "example_user",
  "interactions": [
    {
      "content": "Just published my thoughts on sustainable technology",
      "timestamp": "2023-08-10T12:00:00Z",
      "interaction_type": "post"
    },
    {
      "content": "Interesting perspective! Have you considered the impact on developing nations?",
      "timestamp": "2023-08-10T12:30:00Z",
      "interaction_type": "reply",
      "reply_to_user": "commenter1",
      "sentiment_context": "inquisitive"
    },
    {
      "content": "That's a crucial point you're making. In my research, I found...",
      "timestamp": "2023-08-10T13:00:00Z",
      "interaction_type": "reply",
      "reply_to_user": "commenter2",
      "sentiment_context": "agreement"
    },
    {
      "content": "While the data suggests otherwise, I respect your viewpoint. Here's why...",
      "timestamp": "2023-08-10T13:30:00Z",
      "interaction_type": "reply",
      "reply_to_user": "commenter3",
      "sentiment_context": "respectful_disagreement"
    }
  ]
}'
```

### Supported Sentiment Contexts

When providing explicit sentiment_context, you can use:
- "agreement"
- "disagreement"
- "neutral"
- "supportive"
- "critical"
- "excited"
- "concerned"
- "inquisitive"
- "respectful_disagreement"

## Advanced Analysis Features

### Interaction Patterns Analysis

The API analyzes several aspects of user interactions:

1. **Response Behavior**
   - Average response time
   - Quick response ratio
   - Response consistency with different users

2. **Sentiment Patterns**
   - Emotional stability
   - Positive/negative sentiment ratios
   - Emotional variability
   - Sentiment progression over time

3. **Interaction Style**
   - Inquisitiveness (question-asking behavior)
   - Agreeableness in discussions
   - Detail orientation
   - Supportiveness
   - Confrontational tendencies

4. **Conversation Depth**
   - Average conversation length
   - Sustained conversation ratio
   - Maximum conversation depth

5. **Social Engagement**
   - Network breadth
   - Strong connection count
   - Overall engagement ratio

### Behavioral Insights

The API provides human-readable insights about:
- Communication style
- Social engagement patterns
- Emotional expression
- Conversation depth preferences
- Response patterns

### Enhanced Personality Predictions

The personality predictions are enriched by:
- Analyzing interaction patterns
- Considering response behaviors
- Evaluating emotional stability
- Assessing social engagement
- Measuring conversation depth

## Requirements

- Minimum 3 interactions required for prediction
- Each interaction must include:
  - Content text
  - Timestamp
  - Interaction type
  - Reply information (for replies)
  - Optional sentiment context

## Error Handling

The API returns appropriate HTTP status codes:
- 400: Invalid request (e.g., insufficient interactions)
- 500: Server error during prediction
- 200: Successful prediction

## Environment Variables

- `API_HOST`: Host address (default: 0.0.0.0)
- `API_PORT`: Port number (default: 8000)
- `MIN_INTERACTIONS`: Minimum required interactions (default: 3)

## Notes

- The confidence score combines prediction confidence and interaction data quality
- Timestamps should be provided in ISO 8601 format
- The API uses BERT embeddings for text analysis combined with advanced interaction pattern analysis
- Predictions are enriched using behavioral analysis and interaction patterns
- The API considers both content and meta-features of interactions for more accurate personality assessment
