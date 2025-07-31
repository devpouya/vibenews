"""
BiasScanner Integration Pipeline for Swiss News
Integrates BiasScanner algorithm with existing VibeNews infrastructure
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from .sentence_classifier import SentenceBiasClassifier, BiasClassification
from .bias_scorer import BiasScorer, ArticleBiasScore

logger = logging.getLogger(__name__)


class BiasDetectionPipeline:
    """
    Main pipeline integrating BiasScanner with Swiss news processing
    Handles end-to-end bias detection and scoring
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize pipeline with Gemini API key"""
        self.classifier = SentenceBiasClassifier(gemini_api_key)
        self.scorer = BiasScorer()
        
        logger.info("BiasScanner pipeline initialized")
    
    def process_article(self, article_data: Dict) -> Dict:
        """
        Process single article through complete BiasScanner pipeline
        
        Args:
            article_data: Article dictionary with 'content', 'title', 'url', etc.
            
        Returns:
            Enhanced article data with bias analysis
        """
        
        try:
            article_text = article_data.get('content', '')
            if not article_text:
                logger.warning(f"No content found for article: {article_data.get('url', 'unknown')}")
                return self._add_empty_bias_analysis(article_data)
            
            logger.info(f"Processing article: {article_data.get('title', 'untitled')[:50]}...")
            
            # Step 1: Sentence-level bias classification
            classifications = self.classifier.classify_article(article_text)
            
            # Step 2: Calculate comprehensive bias score
            bias_score = self.scorer.calculate_article_bias_score(
                classifications, article_text
            )
            
            # Step 3: Enhance article data with bias analysis
            enhanced_article = self._enhance_article_with_bias_analysis(
                article_data, classifications, bias_score
            )
            
            logger.info(f"Completed bias analysis - Score: {bias_score.bias_spectrum_score:.2f}, "
                       f"Leaning: {bias_score.political_leaning}")
            
            return enhanced_article
            
        except Exception as e:
            logger.error(f"Error processing article {article_data.get('url', 'unknown')}: {e}")
            return self._add_error_bias_analysis(article_data, str(e))
    
    def process_article_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Process multiple articles through BiasScanner pipeline
        Includes batch-level analysis and spectrum ranking
        """
        
        logger.info(f"Processing batch of {len(articles)} articles")
        
        processed_articles = []
        bias_scores = []
        
        for i, article in enumerate(articles):
            logger.info(f"Processing article {i+1}/{len(articles)}")
            
            processed_article = self.process_article(article)
            processed_articles.append(processed_article)
            
            # Collect bias scores for batch analysis
            if 'bias_analysis' in processed_article and processed_article['bias_analysis']['success']:
                bias_scores.append(processed_article['bias_analysis']['bias_score'])
        
        # Add batch-level spectrum analysis
        if bias_scores:
            spectrum_analysis = self.scorer.compare_articles(bias_scores)
            
            # Add spectrum ranking to each article
            for i, article in enumerate(processed_articles):
                if 'bias_analysis' in article and article['bias_analysis']['success']:
                    article['bias_analysis']['spectrum_ranking'] = i + 1
                    article['bias_analysis']['batch_analysis'] = spectrum_analysis
        
        logger.info(f"Completed batch processing - {len(bias_scores)}/{len(articles)} successfully analyzed")
        
        return processed_articles
    
    def _enhance_article_with_bias_analysis(self, article_data: Dict, 
                                          classifications: List[BiasClassification],
                                          bias_score: ArticleBiasScore) -> Dict:
        """Add comprehensive bias analysis to article data"""
        
        # Create sentence-level analysis
        sentence_analysis = []
        for classification in classifications:
            sentence_analysis.append({
                "sentence": classification.sentence,
                "sentence_index": classification.sentence_index,
                "is_biased": classification.is_biased,
                "bias_type": classification.bias_type.value if classification.bias_type else None,
                "bias_strength": classification.bias_strength,
                "explanation": classification.explanation,
                "confidence": classification.confidence
            })
        
        # Create comprehensive bias analysis
        bias_analysis = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "algorithm": "BiasScanner",
            "version": "1.0.0",
            
            # BiasScanner core metrics
            "bias_score": {
                "total_sentences": bias_score.total_sentences,
                "biased_sentences": bias_score.biased_sentences,
                "bias_ratio": bias_score.bias_ratio,
                "average_bias_strength": bias_score.average_bias_strength,
                "overall_score": bias_score.overall_score,
                "confidence": bias_score.confidence,
                "severity_weighted_score": bias_score.severity_weighted_score
            },
            
            # Swiss bias spectrum (-1 to +1)
            "swiss_bias_spectrum": {
                "spectrum_score": bias_score.bias_spectrum_score,
                "political_leaning": bias_score.political_leaning,
                "leaning_confidence": abs(bias_score.bias_spectrum_score)
            },
            
            # Detailed bias analysis
            "bias_types_detected": bias_score.bias_types_detected,
            "bias_type_counts": bias_score.bias_type_counts,
            
            # Sentence-level details
            "sentence_analysis": sentence_analysis,
            
            # Summary for UI
            "summary": {
                "bias_level": self._categorize_bias_level(bias_score.overall_score),
                "primary_bias_types": self._get_primary_bias_types(bias_score.bias_type_counts),
                "political_summary": f"{bias_score.political_leaning.title()} leaning " +
                                   f"({bias_score.bias_spectrum_score:+.2f})"
            }
        }
        
        # Add bias analysis to article
        enhanced_article = article_data.copy()
        enhanced_article['bias_analysis'] = bias_analysis
        
        return enhanced_article
    
    def _add_empty_bias_analysis(self, article_data: Dict) -> Dict:
        """Add empty bias analysis for articles without content"""
        
        enhanced_article = article_data.copy()
        enhanced_article['bias_analysis'] = {
            "success": False,
            "error": "No content available for analysis",
            "timestamp": datetime.now().isoformat(),
            "bias_score": {
                "overall_score": 0.0,
                "bias_ratio": 0.0,
                "confidence": 0.0
            },
            "swiss_bias_spectrum": {
                "spectrum_score": 0.0,
                "political_leaning": "center"
            }
        }
        
        return enhanced_article
    
    def _add_error_bias_analysis(self, article_data: Dict, error_message: str) -> Dict:
        """Add error bias analysis for failed processing"""
        
        enhanced_article = article_data.copy()
        enhanced_article['bias_analysis'] = {
            "success": False,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "bias_score": {
                "overall_score": 0.0,
                "bias_ratio": 0.0,
                "confidence": 0.0
            },
            "swiss_bias_spectrum": {
                "spectrum_score": 0.0,
                "political_leaning": "center"
            }
        }
        
        return enhanced_article
    
    def _categorize_bias_level(self, overall_score: float) -> str:
        """Categorize bias level for user-friendly display"""
        if overall_score < 0.2:
            return "Low"
        elif overall_score < 0.5:
            return "Moderate"
        elif overall_score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _get_primary_bias_types(self, bias_type_counts: Dict[str, int], limit: int = 3) -> List[str]:
        """Get the most frequent bias types"""
        sorted_types = sorted(bias_type_counts.items(), key=lambda x: x[1], reverse=True)
        return [bias_type for bias_type, count in sorted_types[:limit]]
    
    def save_bias_analysis(self, articles: List[Dict], output_path: str) -> str:
        """
        Save bias analysis results to JSON Lines format
        Compatible with existing VibeNews storage system
        """
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(articles)} articles with bias analysis to {output_file}")
        return str(output_file)
    
    def create_bias_spectrum_report(self, articles: List[Dict]) -> Dict:
        """
        Create comprehensive bias spectrum report for Swiss news analysis
        Shows distribution across political spectrum (-1 to +1)
        """
        
        successful_articles = [
            article for article in articles 
            if article.get('bias_analysis', {}).get('success', False)
        ]
        
        if not successful_articles:
            return {"error": "No successful bias analyses found"}
        
        # Extract spectrum scores
        spectrum_scores = []
        for article in successful_articles:
            score = article['bias_analysis']['swiss_bias_spectrum']['spectrum_score']
            spectrum_scores.append({
                'title': article.get('title', 'Unknown'),
                'url': article.get('url', ''),
                'spectrum_score': score,
                'political_leaning': article['bias_analysis']['swiss_bias_spectrum']['political_leaning'],
                'overall_bias': article['bias_analysis']['bias_score']['overall_score']
            })
        
        # Sort by spectrum score (left to right)
        spectrum_scores.sort(key=lambda x: x['spectrum_score'])
        
        # Calculate distribution
        left_articles = [a for a in spectrum_scores if a['political_leaning'] == 'left']
        center_articles = [a for a in spectrum_scores if a['political_leaning'] == 'center']
        right_articles = [a for a in spectrum_scores if a['political_leaning'] == 'right']
        
        # Calculate averages
        avg_spectrum = sum(a['spectrum_score'] for a in spectrum_scores) / len(spectrum_scores)
        avg_bias = sum(a['overall_bias'] for a in spectrum_scores) / len(spectrum_scores)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_articles": len(successful_articles),
            "spectrum_distribution": {
                "left": len(left_articles),
                "center": len(center_articles), 
                "right": len(right_articles)
            },
            "averages": {
                "spectrum_score": avg_spectrum,
                "overall_bias": avg_bias
            },
            "articles_by_spectrum": spectrum_scores,
            "most_left_leaning": spectrum_scores[0] if spectrum_scores else None,
            "most_right_leaning": spectrum_scores[-1] if spectrum_scores else None,
            "algorithm": "BiasScanner v1.0.0"
        }
        
        return report