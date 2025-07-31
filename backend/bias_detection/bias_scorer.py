"""
Bias scoring and aggregation system
Implements BiasScanner's scoring methodology with Swiss news bias spectrum integration
"""

import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .bias_types import BiasType, get_bias_severity_weight
from .sentence_classifier import BiasClassification


@dataclass
class ArticleBiasScore:
    """Complete bias assessment for an article"""
    
    # Core metrics (from BiasScanner)
    total_sentences: int
    biased_sentences: int
    bias_ratio: float  # 0.0 to 1.0
    average_bias_strength: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0 (BiasScanner normalized score)
    confidence: float  # 0.0 to 1.0
    
    # Swiss news bias spectrum integration (-1 to +1)
    bias_spectrum_score: float  # -1.0 (left bias) to +1.0 (right bias)
    political_leaning: str  # "left", "center", "right"
    
    # Detailed analysis
    bias_types_detected: List[str]
    bias_type_counts: Dict[str, int]
    severity_weighted_score: float  # Weighted by bias type severity
    
    # Metadata
    timestamp: str
    classifications: List[BiasClassification]


class BiasScorer:
    """
    Advanced bias scoring system combining BiasScanner algorithm
    with Swiss news bias spectrum (-1 to +1) for political ranking
    """
    
    def __init__(self):
        """Initialize bias scorer with political bias mappings"""
        
        # Map bias types to political spectrum (-1 = left, 0 = neutral, +1 = right)
        self.political_bias_mapping = {
            BiasType.POLITICAL: 0.0,  # Depends on content analysis
            BiasType.INTERGROUP: 0.0,  # Depends on which groups are favored
            BiasType.DISCRIMINATORY: 0.0,  # Depends on target groups
            BiasType.SOURCE_SELECTION: 0.0,  # Depends on source types
            BiasType.WORD_CHOICE: 0.0,  # Depends on specific words used
            
            # Generally neutral bias types (affect scoring but not political spectrum)
            BiasType.AD_HOMINEM: 0.0,
            BiasType.AMBIGUOUS_ATTRIBUTION: 0.0,
            BiasType.ANECDOTAL_EVIDENCE: 0.0,
            BiasType.CAUSAL_MISUNDERSTANDING: 0.0,
            BiasType.CHERRY_PICKING: 0.0,
            BiasType.CIRCULAR_REASONING: 0.0,
            BiasType.EMOTIONAL_SENSATIONALISM: 0.0,
            BiasType.EXTERNAL_VALIDATION: 0.0,
            BiasType.FALSE_BALANCE: 0.0,
            BiasType.FALSE_DICHOTOMY: 0.0,
            BiasType.FAULTY_ANALOGY: 0.0,
            BiasType.GENERALIZATION: 0.0,
            BiasType.INSINUATIVE_QUESTIONING: 0.0,
            BiasType.MUD_PRAISE: 0.0,
            BiasType.OPINIONATED: 0.0,
            BiasType.PROJECTION: 0.0,
            BiasType.SHIFTING_BENCHMARK: 0.0,
            BiasType.SPECULATION: 0.0,
            BiasType.STRAW_MAN: 0.0,
            BiasType.UNSUBSTANTIATED_CLAIMS: 0.0,
            BiasType.WHATABOUTISM: 0.0,
            BiasType.UNDER_REPORTING: 0.0
        }
        
        # Keywords that indicate political leaning (Swiss context)
        self.left_leaning_keywords = [
            'sozial', 'gerechtigkeit', 'gleichberechtigung', 'umwelt', 'klimaschutz',
            'öffentlich', 'gewerkschaft', 'solidarität', 'inklusion', 'diversität'
        ]
        
        self.right_leaning_keywords = [
            'wirtschaft', 'unternehmen', 'tradition', 'sicherheit', 'ordnung',
            'effizienz', 'leistung', 'eigenverantwortung', 'wettbewerb', 'stabilität'
        ]
    
    def calculate_article_bias_score(self, classifications: List[BiasClassification], 
                                   article_text: str = "") -> ArticleBiasScore:
        """
        Calculate comprehensive bias score for article
        Combines BiasScanner methodology with Swiss political spectrum
        """
        
        if not classifications:
            return self._create_empty_score()
        
        # Basic BiasScanner metrics
        total_sentences = len(classifications)
        biased_classifications = [c for c in classifications if c.is_biased]
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
        
        # BiasScanner overall score formula (normalized combination)
        overall_score = (bias_ratio + average_bias_strength) / 2.0
        
        # Bias type analysis
        bias_type_counts = {}
        severity_weighted_total = 0.0
        political_spectrum_signals = []
        
        for classification in biased_classifications:
            if classification.bias_type:
                bias_type = classification.bias_type.value
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
                
                # Add severity weighting
                severity_weight = get_bias_severity_weight(classification.bias_type)
                severity_weighted_total += classification.bias_strength * severity_weight
                
                # Collect political spectrum signals
                spectrum_value = self._analyze_political_bias(
                    classification, article_text
                )
                if spectrum_value != 0.0:
                    political_spectrum_signals.append(spectrum_value)
        
        severity_weighted_score = (
            severity_weighted_total / biased_sentences if biased_sentences > 0 else 0.0
        )
        
        # Calculate Swiss bias spectrum score (-1 to +1)
        bias_spectrum_score = self._calculate_spectrum_score(
            political_spectrum_signals, article_text
        )
        
        political_leaning = self._determine_political_leaning(bias_spectrum_score)
        
        return ArticleBiasScore(
            total_sentences=total_sentences,
            biased_sentences=biased_sentences,
            bias_ratio=bias_ratio,
            average_bias_strength=average_bias_strength,
            overall_score=overall_score,
            confidence=average_confidence,
            bias_spectrum_score=bias_spectrum_score,
            political_leaning=political_leaning,
            bias_types_detected=list(bias_type_counts.keys()),
            bias_type_counts=bias_type_counts,
            severity_weighted_score=severity_weighted_score,
            timestamp=datetime.now().isoformat(),
            classifications=classifications
        )
    
    def _analyze_political_bias(self, classification: BiasClassification, 
                              article_text: str) -> float:
        """
        Analyze individual classification for political spectrum signals
        Returns -1.0 to +1.0 indicating left to right bias
        """
        
        if not classification.bias_type:
            return 0.0
        
        # For context-dependent bias types, analyze the sentence content
        sentence_lower = classification.sentence.lower()
        
        if classification.bias_type == BiasType.POLITICAL:
            # Direct political bias - analyze sentence for political keywords
            left_signals = sum(1 for keyword in self.left_leaning_keywords 
                             if keyword in sentence_lower)
            right_signals = sum(1 for keyword in self.right_leaning_keywords 
                              if keyword in sentence_lower)
            
            if left_signals > right_signals:
                return -classification.bias_strength
            elif right_signals > left_signals:
                return classification.bias_strength
        
        elif classification.bias_type == BiasType.WORD_CHOICE:
            # Analyze loaded language for political lean
            return self._analyze_word_choice_bias(classification.sentence)
        
        elif classification.bias_type == BiasType.SOURCE_SELECTION:
            # Analyze if sources lean particular direction
            return self._analyze_source_bias(classification.sentence)
        
        elif classification.bias_type == BiasType.INTERGROUP:
            # Analyze which groups are favored/discriminated against
            return self._analyze_intergroup_bias(classification.sentence)
        
        return 0.0
    
    def _analyze_word_choice_bias(self, sentence: str) -> float:
        """Analyze word choice for political leaning"""
        sentence_lower = sentence.lower()
        
        # Swiss-specific political language patterns
        left_words = ['reform', 'progress', 'gerecht', 'solidarisch', 'nachhaltig']
        right_words = ['tradition', 'bewährt', 'stabilität', 'sicherheit', 'ordnung']
        
        left_count = sum(1 for word in left_words if word in sentence_lower)
        right_count = sum(1 for word in right_words if word in sentence_lower)
        
        if left_count > right_count:
            return -0.3
        elif right_count > left_count:
            return 0.3
        
        return 0.0
    
    def _analyze_source_bias(self, sentence: str) -> float:
        """Analyze source selection for political lean"""
        sentence_lower = sentence.lower()
        
        # Swiss media/source patterns
        if any(term in sentence_lower for term in ['gewerkschaft', 'sp', 'grüne']):
            return -0.2
        elif any(term in sentence_lower for term in ['economiesuisse', 'svp', 'fdp']):
            return 0.2
        
        return 0.0
    
    def _analyze_intergroup_bias(self, sentence: str) -> float:
        """Analyze intergroup bias for political implications"""
        sentence_lower = sentence.lower()
        
        # Analyze target groups and framing
        if any(term in sentence_lower for term in ['ausländer', 'immigrant', 'flüchtling']):
            # Check framing - negative framing often indicates right lean
            if any(term in sentence_lower for term in ['problem', 'gefahr', 'belastung']):
                return 0.4
            elif any(term in sentence_lower for term in ['bereicherung', 'vielfalt', 'menschlich']):
                return -0.4
        
        return 0.0
    
    def _calculate_spectrum_score(self, political_signals: List[float], 
                                article_text: str) -> float:
        """
        Calculate overall political spectrum score (-1 to +1)
        Combines individual bias signals with contextual analysis
        """
        
        if not political_signals:
            # Fallback: analyze article text for general political keywords
            return self._analyze_article_keywords(article_text)
        
        # Average the political signals
        spectrum_score = sum(political_signals) / len(political_signals)
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, spectrum_score))
    
    def _analyze_article_keywords(self, article_text: str) -> float:
        """Fallback analysis of article text for political keywords"""
        if not article_text:
            return 0.0
        
        text_lower = article_text.lower()
        
        left_count = sum(1 for keyword in self.left_leaning_keywords 
                        if keyword in text_lower)
        right_count = sum(1 for keyword in self.right_leaning_keywords 
                         if keyword in text_lower)
        
        total_political_words = left_count + right_count
        if total_political_words == 0:
            return 0.0
        
        # Calculate ratio and scale to [-1, 1]
        ratio = (right_count - left_count) / total_political_words
        return max(-1.0, min(1.0, ratio))
    
    def _determine_political_leaning(self, spectrum_score: float) -> str:
        """Determine political leaning category from spectrum score"""
        if spectrum_score < -0.3:
            return "left"
        elif spectrum_score > 0.3:
            return "right"
        else:
            return "center"
    
    def _create_empty_score(self) -> ArticleBiasScore:
        """Create empty bias score for articles with no classifications"""
        return ArticleBiasScore(
            total_sentences=0,
            biased_sentences=0,
            bias_ratio=0.0,
            average_bias_strength=0.0,
            overall_score=0.0,
            confidence=0.0,
            bias_spectrum_score=0.0,
            political_leaning="center",
            bias_types_detected=[],
            bias_type_counts={},
            severity_weighted_score=0.0,
            timestamp=datetime.now().isoformat(),
            classifications=[]
        )
    
    def compare_articles(self, scores: List[ArticleBiasScore]) -> Dict:
        """
        Compare multiple articles for bias spectrum ranking
        Core functionality for Swiss news bias aggregation
        """
        
        if not scores:
            return {"articles": [], "spectrum_distribution": {}}
        
        # Sort by bias spectrum score (left to right)
        sorted_scores = sorted(scores, key=lambda s: s.bias_spectrum_score)
        
        # Calculate distribution
        left_count = sum(1 for s in scores if s.political_leaning == "left")
        center_count = sum(1 for s in scores if s.political_leaning == "center")
        right_count = sum(1 for s in scores if s.political_leaning == "right")
        
        spectrum_distribution = {
            "left": left_count,
            "center": center_count,
            "right": right_count,
            "total": len(scores)
        }
        
        # Calculate average scores
        avg_bias_ratio = sum(s.bias_ratio for s in scores) / len(scores)
        avg_spectrum_score = sum(s.bias_spectrum_score for s in scores) / len(scores)
        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        
        return {
            "articles": [
                {
                    "spectrum_score": score.bias_spectrum_score,
                    "political_leaning": score.political_leaning,
                    "overall_bias": score.overall_score,
                    "confidence": score.confidence,
                    "bias_types": score.bias_types_detected
                }
                for score in sorted_scores
            ],
            "spectrum_distribution": spectrum_distribution,
            "averages": {
                "bias_ratio": avg_bias_ratio,
                "spectrum_score": avg_spectrum_score,
                "confidence": avg_confidence
            },
            "timestamp": datetime.now().isoformat()
        }