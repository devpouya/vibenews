"""
Sentence-level bias classification using fine-tuned language models
Based on BiasScanner algorithm implementation
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .bias_types import BiasType, BIAS_DEFINITIONS

logger = logging.getLogger(__name__)


@dataclass
class BiasClassification:
    """Individual sentence bias classification result"""
    sentence: str
    sentence_index: int
    bias_type: Optional[BiasType]
    bias_strength: float  # 0.0 to 1.0
    explanation: str
    confidence: float  # 0.0 to 1.0
    is_biased: bool


class SentenceBiasClassifier:
    """
    BiasScanner implementation for sentence-level bias detection
    Uses Gemini model fine-tuned for 27 bias type classification
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize classifier with Gemini API"""
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Rate limiting (based on BiasScanner paper: 5/min, 25/day for Gemini)
        self.rate_limit_per_minute = 5
        self.rate_limit_per_day = 25
        
        # Initialize prompt template
        self.prompt_template = self._create_bias_detection_prompt()
    
    def _create_bias_detection_prompt(self) -> str:
        """
        Create comprehensive prompt for bias detection
        Based on BiasScanner's iterative prompt development
        """
        
        # Build bias type definitions for prompt
        bias_definitions_text = ""
        for bias_type, definition in BIAS_DEFINITIONS.items():
            bias_definitions_text += f"""
{definition.name} ({bias_type.value}):
Definition: {definition.description}
Examples: {'; '.join(definition.examples[:2])}
Severity Weight: {definition.severity_weight}
"""
        
        prompt = f"""You are an expert media bias detection system. Your task is to analyze news sentences and identify any media bias present.

BIAS TYPES TO DETECT (27 total):
{bias_definitions_text}

INSTRUCTIONS:
1. Analyze each sentence for any of the 27 bias types listed above
2. For each sentence, determine:
   - Is the sentence biased? (true/false)
   - If biased, which bias type(s) apply?
   - Bias strength (0.0 = no bias, 1.0 = extreme bias)
   - Confidence in classification (0.0 = uncertain, 1.0 = very confident)
   - Clear explanation of why this bias was detected

3. Return results in JSON format for each sentence:
{{
  "sentence_index": 0,
  "is_biased": true/false,
  "bias_type": "bias_type_value" or null,
  "bias_strength": 0.0-1.0,
  "confidence": 0.0-1.0,
  "explanation": "Clear explanation of bias detection"
}}

IMPORTANT GUIDELINES:
- Be precise and conservative in bias detection
- Provide specific, actionable explanations
- Consider context and intent, not just word choice
- Distinguish between legitimate criticism and bias
- Account for cultural and linguistic nuances

Analyze these sentences:
{sentences_placeholder}

Return only valid JSON array with results for each sentence."""
        
        return prompt
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using robust sentence segmentation
        Based on BiasScanner's sentence processing approach
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries (improved regex)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out very short sentences (likely not meaningful)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _format_sentences_for_prompt(self, sentences: List[str]) -> str:
        """Format sentences for the prompt"""
        formatted = ""
        for i, sentence in enumerate(sentences):
            formatted += f"Sentence {i}: {sentence}\n"
        return formatted
    
    def _parse_model_response(self, response_text: str, sentences: List[str]) -> List[BiasClassification]:
        """Parse model JSON response into BiasClassification objects"""
        try:
            # Clean response text (remove markdown formatting if present)
            response_text = re.sub(r'```json\n?', '', response_text)
            response_text = re.sub(r'\n?```', '', response_text)
            
            results = json.loads(response_text)
            classifications = []
            
            for i, result in enumerate(results):
                if i >= len(sentences):
                    logger.warning(f"Model returned more results than sentences: {i} >= {len(sentences)}")
                    break
                
                # Parse bias type
                bias_type = None
                if result.get('bias_type') and result.get('is_biased', False):
                    try:
                        bias_type = BiasType(result['bias_type'])
                    except ValueError:
                        logger.warning(f"Unknown bias type: {result.get('bias_type')}")
                
                classification = BiasClassification(
                    sentence=sentences[i],
                    sentence_index=i,
                    bias_type=bias_type,
                    bias_strength=float(result.get('bias_strength', 0.0)),
                    explanation=result.get('explanation', ''),
                    confidence=float(result.get('confidence', 0.0)),
                    is_biased=bool(result.get('is_biased', False))
                )
                classifications.append(classification)
            
            return classifications
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model response as JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            
            # Return fallback classifications
            return [
                BiasClassification(
                    sentence=sentence,
                    sentence_index=i,
                    bias_type=None,
                    bias_strength=0.0,
                    explanation="Failed to parse model response",
                    confidence=0.0,
                    is_biased=False
                )
                for i, sentence in enumerate(sentences)
            ]
        
        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            return []
    
    def classify_article(self, article_text: str) -> List[BiasClassification]:
        """
        Classify bias in an entire article at sentence level
        Main entry point for BiasScanner algorithm
        """
        
        logger.info(f"Starting bias classification for article ({len(article_text)} chars)")
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(article_text)
        logger.info(f"Split article into {len(sentences)} sentences")
        
        if not sentences:
            logger.warning("No sentences found in article")
            return []
        
        # Step 2: Prepare prompt
        sentences_text = self._format_sentences_for_prompt(sentences)
        prompt = self.prompt_template.replace("{sentences_placeholder}", sentences_text)
        
        try:
            # Step 3: Call model with safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent results
                    max_output_tokens=4000,
                    top_p=0.8
                )
            )
            
            logger.info("Received model response")
            
            # Step 4: Parse response
            classifications = self._parse_model_response(response.text, sentences)
            
            # Step 5: Log results summary
            biased_count = sum(1 for c in classifications if c.is_biased)
            logger.info(f"Classified {len(classifications)} sentences, {biased_count} found biased")
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            
            # Return fallback classifications
            return [
                BiasClassification(
                    sentence=sentence,
                    sentence_index=i,
                    bias_type=None,
                    bias_strength=0.0,
                    explanation=f"Classification failed: {str(e)}",
                    confidence=0.0,
                    is_biased=False
                )
                for i, sentence in enumerate(sentences)
            ]
    
    def classify_sentence(self, sentence: str) -> BiasClassification:
        """Classify bias in a single sentence"""
        results = self.classify_article(sentence)
        return results[0] if results else BiasClassification(
            sentence=sentence,
            sentence_index=0,
            bias_type=None,
            bias_strength=0.0,
            explanation="No classification available",
            confidence=0.0,
            is_biased=False
        )
    
    def get_bias_summary(self, classifications: List[BiasClassification]) -> Dict:
        """
        Generate bias summary for article
        Following BiasScanner's reporting approach
        """
        
        if not classifications:
            return {
                "total_sentences": 0,
                "biased_sentences": 0,
                "bias_ratio": 0.0,
                "average_bias_strength": 0.0,
                "bias_types_detected": [],
                "overall_score": 0.0,
                "confidence": 0.0
            }
        
        biased_classifications = [c for c in classifications if c.is_biased]
        
        # Count bias types
        bias_type_counts = {}
        for classification in biased_classifications:
            if classification.bias_type:
                bias_type = classification.bias_type.value
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
        
        # Calculate metrics
        total_sentences = len(classifications)
        biased_sentences = len(biased_classifications)
        bias_ratio = biased_sentences / total_sentences if total_sentences > 0 else 0.0
        
        average_bias_strength = (
            sum(c.bias_strength for c in biased_classifications) / biased_sentences
            if biased_sentences > 0 else 0.0
        )
        
        average_confidence = (
            sum(c.confidence for c in classifications) / total_sentences
            if total_sentences > 0 else 0.0
        )
        
        # Overall score (BiasScanner formula: normalized combination of ratio and strength)
        overall_score = (bias_ratio + average_bias_strength) / 2.0
        
        return {
            "total_sentences": total_sentences,
            "biased_sentences": biased_sentences,
            "bias_ratio": bias_ratio,
            "average_bias_strength": average_bias_strength,
            "bias_types_detected": list(bias_type_counts.keys()),
            "bias_type_counts": bias_type_counts,
            "overall_score": overall_score,
            "confidence": average_confidence,
            "timestamp": datetime.now().isoformat()
        }