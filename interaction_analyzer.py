from typing import List, Dict
from collections import defaultdict
import numpy as np
from textblob import TextBlob
from datetime import datetime
from api import TwitterInteraction

class InteractionAnalyzer:
    def __init__(self):
        # Sentiment thresholds
        self.STRONG_POS = 0.5
        self.STRONG_NEG = -0.5
        
        # Response patterns
        self.QUICK_RESPONSE = 300  # 5 minutes in seconds
        self.DELAYED_RESPONSE = 3600  # 1 hour in seconds
        
        # Sentiment context mappings
        self.SENTIMENT_CONTEXT_WEIGHTS = {
            "agreement": 0.8,
            "disagreement": -0.3,
            "neutral": 0.0,
            "supportive": 0.7,
            "critical": -0.4,
            "excited": 0.9,
            "concerned": -0.2
        }
    
    def _analyze_sentiment(self, interaction: TwitterInteraction) -> float:
        """
        Analyze sentiment using both automatic analysis and explicit context if provided.
        Returns a sentiment score between -1 and 1.
        """
        # Get automatic sentiment using TextBlob
        blob = TextBlob(interaction.content)
        auto_sentiment = blob.sentiment.polarity
        
        if interaction.sentiment_context:
            # If explicit sentiment context is provided, combine it with automatic analysis
            context_weight = self.SENTIMENT_CONTEXT_WEIGHTS.get(
                interaction.sentiment_context.lower(),
                0.0  # Default weight if context is not in our mapping
            )
            # Combine automatic and explicit sentiment (60-40 weight)
            return (auto_sentiment * 0.6) + (context_weight * 0.4)
        
        # If no explicit context, use only automatic sentiment
        return auto_sentiment
    
    def analyze_interaction_patterns(self, interactions: List[TwitterInteraction]) -> Dict:
        """Analyze patterns in user interactions to understand behavioral traits."""
        patterns = {
            "response_behavior": self._analyze_response_patterns(interactions),
            "sentiment_patterns": self._analyze_sentiment_patterns(interactions),
            "interaction_style": self._analyze_interaction_style(interactions),
            "conversation_depth": self._analyze_conversation_depth(interactions),
            "social_engagement": self._analyze_social_engagement(interactions)
        }
        return patterns
    
    def _analyze_response_patterns(self, interactions: List[TwitterInteraction]) -> Dict:
        """Analyze how quickly and consistently the user responds to others."""
        response_times = []
        response_consistency = defaultdict(list)
        
        # Sort interactions by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        for i, interaction in enumerate(sorted_interactions):
            if interaction.interaction_type == "reply" and i > 0:
                time_diff = (interaction.timestamp - sorted_interactions[i-1].timestamp).total_seconds()
                response_times.append(time_diff)
                response_consistency[interaction.reply_to_user].append(time_diff)
        
        return {
            "avg_response_time": np.mean(response_times) if response_times else None,
            "quick_responses_ratio": len([t for t in response_times if t < self.QUICK_RESPONSE]) / len(response_times) if response_times else 0,
            "consistent_responder": self._calculate_consistency(response_consistency)
        }
    
    def _analyze_sentiment_patterns(self, interactions: List[TwitterInteraction]) -> Dict:
        """Analyze emotional patterns and sentiment stability in interactions."""
        sentiments = []
        sentiment_changes = []
        
        for interaction in interactions:
            # Get combined sentiment score
            sentiment = self._analyze_sentiment(interaction)
            sentiments.append(sentiment)
            
            if len(sentiments) > 1:
                sentiment_changes.append(abs(sentiment - sentiments[-2]))
        
        return {
            "sentiment_stability": 1 - (np.std(sentiments) if sentiments else 0),
            "positive_ratio": len([s for s in sentiments if s > self.STRONG_POS]) / len(sentiments) if sentiments else 0,
            "negative_ratio": len([s for s in sentiments if s < self.STRONG_NEG]) / len(sentiments) if sentiments else 0,
            "emotional_variability": np.mean(sentiment_changes) if sentiment_changes else 0,
            "sentiment_progression": self._analyze_sentiment_progression(sentiments)
        }
    
    def _analyze_sentiment_progression(self, sentiments: List[float]) -> str:
        """Analyze how sentiment changes over the course of interactions."""
        if not sentiments:
            return "neutral"
            
        # Calculate trend using simple linear regression
        x = np.arange(len(sentiments))
        coeffs = np.polyfit(x, sentiments, 1)
        slope = coeffs[0]
        
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasingly_positive"
        else:
            return "increasingly_negative"
    
    def _analyze_interaction_style(self, interactions: List[TwitterInteraction]) -> Dict:
        """Analyze the user's interaction style and communication patterns."""
        styles = defaultdict(int)
        total = len(interactions)
        
        for interaction in interactions:
            # Analyze content characteristics
            content = interaction.content.lower()
            
            # Check for question-asking behavior
            if "?" in content:
                styles["inquisitive"] += 1
            
            # Check for agreement/disagreement patterns
            if any(word in content for word in ["agree", "yes", "exactly", "true"]):
                styles["agreeable"] += 1
            elif any(word in content for word in ["disagree", "no", "wrong", "false"]):
                styles["confrontational"] += 1
            
            # Check for detailed responses
            if len(content.split()) > 30:
                styles["detailed"] += 1
            
            # Check for supportive behavior
            if any(word in content for word in ["thanks", "appreciate", "helpful", "good point"]):
                styles["supportive"] += 1
            
            # Consider explicit sentiment context if provided
            if interaction.sentiment_context:
                if interaction.sentiment_context.lower() in ["supportive", "agreement"]:
                    styles["supportive"] += 0.5  # Add partial weight for explicit context
                elif interaction.sentiment_context.lower() in ["critical", "disagreement"]:
                    styles["confrontational"] += 0.5
        
        return {
            "inquisitiveness": styles["inquisitive"] / total,
            "agreeableness": styles["agreeable"] / total,
            "confrontational": styles["confrontational"] / total,
            "detail_orientation": styles["detailed"] / total,
            "supportiveness": styles["supportive"] / total
        }
    
    def _analyze_conversation_depth(self, interactions: List[TwitterInteraction]) -> Dict:
        """Analyze the depth and quality of conversations."""
        conversation_chains = defaultdict(list)
        
        for interaction in interactions:
            if interaction.reply_to_user:
                conversation_chains[interaction.reply_to_user].append(interaction)
        
        depths = [len(chain) for chain in conversation_chains.values()]
        
        return {
            "avg_conversation_depth": np.mean(depths) if depths else 0,
            "max_conversation_depth": max(depths) if depths else 0,
            "sustained_conversations": len([d for d in depths if d > 3]) / len(depths) if depths else 0
        }
    
    def _analyze_social_engagement(self, interactions: List[TwitterInteraction]) -> Dict:
        """Analyze patterns in social engagement and network building."""
        unique_interactions = len(set(i.reply_to_user for i in interactions if i.reply_to_user))
        repeat_interactions = defaultdict(int)
        
        for interaction in interactions:
            if interaction.reply_to_user:
                repeat_interactions[interaction.reply_to_user] += 1
        
        strong_connections = len([user for user, count in repeat_interactions.items() if count > 3])
        
        return {
            "network_breadth": unique_interactions,
            "strong_connections": strong_connections,
            "engagement_ratio": len([i for i in interactions if i.reply_to_user]) / len(interactions)
        }
    
    def _calculate_consistency(self, response_consistency: Dict) -> float:
        """Calculate consistency in response patterns with specific users."""
        if not response_consistency:
            return 0.0
            
        user_stds = []
        for user_times in response_consistency.values():
            if len(user_times) > 1:
                user_stds.append(np.std(user_times))
        
        return 1 - (np.mean(user_stds) / self.DELAYED_RESPONSE if user_stds else 0)
    
    def enrich_personality_prediction(self, base_predictions: Dict, interaction_patterns: Dict) -> Dict:
        """Enrich personality predictions with interaction pattern analysis."""
        enriched_predictions = base_predictions.copy()
        
        # Adjust Extraversion (EXT) based on social engagement
        if "EXT" in enriched_predictions:
            social_factor = (
                interaction_patterns["social_engagement"]["network_breadth"] * 0.3 +
                interaction_patterns["social_engagement"]["engagement_ratio"] * 0.7
            )
            enriched_predictions["EXT"] = (enriched_predictions["EXT"] + social_factor) / 2
        
        # Adjust Agreeableness (AGR) based on interaction style
        if "AGR" in enriched_predictions:
            agr_factor = (
                interaction_patterns["interaction_style"]["agreeableness"] * 0.4 +
                interaction_patterns["interaction_style"]["supportiveness"] * 0.4 +
                (1 - interaction_patterns["interaction_style"]["confrontational"]) * 0.2
            )
            enriched_predictions["AGR"] = (enriched_predictions["AGR"] + agr_factor) / 2
        
        # Adjust Neuroticism (NEU) based on sentiment patterns
        if "NEU" in enriched_predictions:
            neu_factor = (
                interaction_patterns["sentiment_patterns"]["emotional_variability"] * 0.5 +
                (1 - interaction_patterns["sentiment_patterns"]["sentiment_stability"]) * 0.5
            )
            enriched_predictions["NEU"] = (enriched_predictions["NEU"] + neu_factor) / 2
        
        # Adjust Conscientiousness (CON) based on response patterns
        if "CON" in enriched_predictions:
            con_factor = (
                interaction_patterns["response_behavior"]["consistent_responder"] * 0.6 +
                interaction_patterns["interaction_style"]["detail_orientation"] * 0.4
            )
            enriched_predictions["CON"] = (enriched_predictions["CON"] + con_factor) / 2
        
        # Adjust Openness (OPN) based on conversation depth
        if "OPN" in enriched_predictions:
            opn_factor = (
                interaction_patterns["conversation_depth"]["sustained_conversations"] * 0.5 +
                interaction_patterns["interaction_style"]["inquisitiveness"] * 0.5
            )
            enriched_predictions["OPN"] = (enriched_predictions["OPN"] + opn_factor) / 2
        
        return enriched_predictions
