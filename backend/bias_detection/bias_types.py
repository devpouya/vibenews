"""
BiasScanner Implementation: 27 Bias Types Definitions
Based on the BiasScanner research paper by Menzner & Leidner (2024)
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class BiasType(Enum):
    """27 bias types detected by BiasScanner algorithm"""
    AD_HOMINEM = "ad_hominem"
    AMBIGUOUS_ATTRIBUTION = "ambiguous_attribution"
    ANECDOTAL_EVIDENCE = "anecdotal_evidence"
    CAUSAL_MISUNDERSTANDING = "causal_misunderstanding"
    CHERRY_PICKING = "cherry_picking"
    CIRCULAR_REASONING = "circular_reasoning"
    DISCRIMINATORY = "discriminatory"
    EMOTIONAL_SENSATIONALISM = "emotional_sensationalism"
    EXTERNAL_VALIDATION = "external_validation"
    FALSE_BALANCE = "false_balance"
    FALSE_DICHOTOMY = "false_dichotomy"
    FAULTY_ANALOGY = "faulty_analogy"
    GENERALIZATION = "generalization"
    INSINUATIVE_QUESTIONING = "insinuative_questioning"
    INTERGROUP = "intergroup"
    MUD_PRAISE = "mud_praise"
    OPINIONATED = "opinionated"
    POLITICAL = "political"
    PROJECTION = "projection"
    SHIFTING_BENCHMARK = "shifting_benchmark"
    SOURCE_SELECTION = "source_selection"
    SPECULATION = "speculation"
    STRAW_MAN = "straw_man"
    UNSUBSTANTIATED_CLAIMS = "unsubstantiated_claims"
    WHATABOUTISM = "whataboutism"
    WORD_CHOICE = "word_choice"
    UNDER_REPORTING = "under_reporting"  # Additional type


@dataclass
class BiasDefinition:
    """Definition and examples for each bias type"""
    name: str
    description: str
    examples: List[str]
    severity_weight: float  # 0.1 to 1.0, higher = more severe


# Comprehensive bias type definitions based on BiasScanner paper
BIAS_DEFINITIONS: Dict[BiasType, BiasDefinition] = {
    BiasType.AD_HOMINEM: BiasDefinition(
        name="Ad Hominem Bias",
        description="Attacking the person making an argument rather than addressing the argument itself",
        examples=[
            "The proposal is flawed because its author has no credibility",
            "This comes from someone who clearly doesn't understand economics"
        ],
        severity_weight=0.8
    ),
    
    BiasType.AMBIGUOUS_ATTRIBUTION: BiasDefinition(
        name="Ambiguous Attribution Bias",
        description="Using vague or unclear sources like 'some say', 'experts believe', 'critics argue'",
        examples=[
            "Some analysts suggest the policy will fail",
            "Critics argue this approach is misguided"
        ],
        severity_weight=0.6
    ),
    
    BiasType.ANECDOTAL_EVIDENCE: BiasDefinition(
        name="Anecdotal Evidence Bias",
        description="Using individual stories or isolated examples to support broader claims",
        examples=[
            "John Smith lost his job due to automation, proving technology destroys employment",
            "A single patient's recovery shows this treatment works for everyone"
        ],
        severity_weight=0.7
    ),
    
    BiasType.CAUSAL_MISUNDERSTANDING: BiasDefinition(
        name="Causal Misunderstanding Bias",
        description="Incorrectly implying causation from correlation or misrepresenting cause-effect relationships",
        examples=[
            "Crime rose after the new mayor took office, clearly due to his policies",
            "Sales increased during the advertising campaign, proving its effectiveness"
        ],
        severity_weight=0.8
    ),
    
    BiasType.CHERRY_PICKING: BiasDefinition(
        name="Cherry Picking Bias",
        description="Selectively presenting only data or evidence that supports a particular viewpoint",
        examples=[
            "Unemployment fell to 5% last month (ignoring the 6-month rising trend)",
            "The study shows 90% success rate (omitting failure conditions)"
        ],
        severity_weight=0.9
    ),
    
    BiasType.CIRCULAR_REASONING: BiasDefinition(
        name="Circular Reasoning Bias",
        description="Using the conclusion of an argument as evidence for the argument itself",
        examples=[
            "This policy is bad because it will cause harm, and we know it's harmful because it's bad",
            "He's trustworthy because he's reliable, and he's reliable because he's trustworthy"
        ],
        severity_weight=0.7
    ),
    
    BiasType.DISCRIMINATORY: BiasDefinition(
        name="Discriminatory Bias",
        description="Showing prejudice or unfair treatment based on group membership (race, gender, religion, etc.)",
        examples=[
            "Women are naturally less suited for leadership roles",
            "Young people today lack work ethic and responsibility"
        ],
        severity_weight=1.0
    ),
    
    BiasType.EMOTIONAL_SENSATIONALISM: BiasDefinition(
        name="Emotional Sensationalism Bias",
        description="Using emotionally charged language to manipulate reader feelings rather than inform",
        examples=[
            "The horrific disaster that shocked the nation",
            "Politicians' outrageous betrayal of public trust"
        ],
        severity_weight=0.8
    ),
    
    BiasType.EXTERNAL_VALIDATION: BiasDefinition(
        name="External Validation Bias",
        description="Appealing to popularity, authority, or tradition without logical justification",
        examples=[
            "Everyone knows this is the right approach",
            "Experts agree this is the only solution"
        ],
        severity_weight=0.6
    ),
    
    BiasType.FALSE_BALANCE: BiasDefinition(
        name="False Balance Bias",
        description="Presenting two sides as equally valid when evidence strongly supports one side",
        examples=[
            "Scientists debate whether climate change is real (when 97% consensus exists)",
            "Both sides have valid points about vaccine safety (ignoring scientific consensus)"
        ],
        severity_weight=0.9
    ),
    
    BiasType.FALSE_DICHOTOMY: BiasDefinition(
        name="False Dichotomy Bias",
        description="Presenting only two options when more alternatives exist",
        examples=[
            "Either we increase taxes or the economy collapses",
            "You're either with us or against us"
        ],
        severity_weight=0.7
    ),
    
    BiasType.FAULTY_ANALOGY: BiasDefinition(
        name="Faulty Analogy Bias",
        description="Drawing inappropriate comparisons between dissimilar situations",
        examples=[
            "Running a country is like running a business",
            "Regulating the internet is like censoring books"
        ],
        severity_weight=0.6
    ),
    
    BiasType.GENERALIZATION: BiasDefinition(
        name="Generalization Bias",
        description="Making broad statements about groups based on limited examples",
        examples=[
            "All politicians are corrupt",
            "Technology companies never care about privacy"
        ],
        severity_weight=0.8
    ),
    
    BiasType.INSINUATIVE_QUESTIONING: BiasDefinition(
        name="Insinuative Questioning Bias",
        description="Using leading questions to imply conclusions without stating them directly",
        examples=[
            "Why did the senator try to hide his financial records?",
            "What is the company afraid of revealing?"
        ],
        severity_weight=0.7
    ),
    
    BiasType.INTERGROUP: BiasDefinition(
        name="Intergroup Bias",
        description="Showing favoritism toward one group while discriminating against another",
        examples=[
            "Our supporters are patriotic citizens while opponents are extremists",
            "Real Americans understand this issue, unlike coastal elites"
        ],
        severity_weight=0.9
    ),
    
    BiasType.MUD_PRAISE: BiasDefinition(
        name="Mud Praise Bias",
        description="Giving backhanded compliments or praise that actually undermines the subject",
        examples=[
            "He's surprisingly articulate for someone from his background",
            "The policy is simple enough that even average citizens can understand it"
        ],
        severity_weight=0.7
    ),
    
    BiasType.OPINIONATED: BiasDefinition(
        name="Opinionated Bias",
        description="Presenting personal opinions or judgments as objective facts",
        examples=[
            "This obviously flawed strategy will certainly fail",
            "The brilliant new approach clearly demonstrates superior thinking"
        ],
        severity_weight=0.8
    ),
    
    BiasType.POLITICAL: BiasDefinition(
        name="Political Bias",
        description="Showing clear preference for particular political parties, ideologies, or candidates",
        examples=[
            "The conservative agenda threatens democracy",
            "Liberal policies are destroying traditional values"
        ],
        severity_weight=0.9
    ),
    
    BiasType.PROJECTION: BiasDefinition(
        name="Projection Bias",
        description="Attributing one's own attitudes, feelings, or motivations to others",
        examples=[
            "The opposition secretly wants this policy to fail",
            "Voters are obviously frustrated with the current situation"
        ],
        severity_weight=0.6
    ),
    
    BiasType.SHIFTING_BENCHMARK: BiasDefinition(
        name="Shifting Benchmark Bias",
        description="Changing evaluation criteria to favor preferred outcomes",
        examples=[
            "Economic growth matters more than unemployment (when growth is high)",
            "Process is more important than results (when results are poor)"
        ],
        severity_weight=0.7
    ),
    
    BiasType.SOURCE_SELECTION: BiasDefinition(
        name="Source Selection Bias",
        description="Choosing sources that predominantly support one viewpoint",
        examples=[
            "Multiple conservative analysts agree...",
            "Liberal experts unanimously support..."
        ],
        severity_weight=0.8
    ),
    
    BiasType.SPECULATION: BiasDefinition(
        name="Speculation Bias",
        description="Presenting hypothetical scenarios or unverified claims as likely facts",
        examples=[
            "This policy will likely lead to economic collapse",
            "The decision could potentially cause widespread harm"
        ],
        severity_weight=0.5
    ),
    
    BiasType.STRAW_MAN: BiasDefinition(
        name="Straw Man Bias",
        description="Misrepresenting someone's argument to make it easier to attack",
        examples=[
            "Opponents want to eliminate all environmental protection",
            "Supporters believe we should ignore all economic concerns"
        ],
        severity_weight=0.8
    ),
    
    BiasType.UNSUBSTANTIATED_CLAIMS: BiasDefinition(
        name="Unsubstantiated Claims Bias",
        description="Making assertions without providing evidence, sources, or verification",
        examples=[
            "Studies show this approach is most effective",
            "It's well-known that this method works best"
        ],
        severity_weight=0.7
    ),
    
    BiasType.WHATABOUTISM: BiasDefinition(
        name="Whataboutism Bias",
        description="Deflecting criticism by pointing to other issues or past events",
        examples=[
            "What about when the other party did something similar?",
            "Before criticizing us, look at their track record"
        ],
        severity_weight=0.6
    ),
    
    BiasType.WORD_CHOICE: BiasDefinition(
        name="Word Choice Bias",
        description="Using loaded, emotionally charged, or slanted language",
        examples=[
            "Regime vs. government", "Freedom fighters vs. terrorists",
            "Tax relief vs. tax cuts", "Death tax vs. estate tax"
        ],
        severity_weight=0.7
    ),
    
    BiasType.UNDER_REPORTING: BiasDefinition(
        name="Under-reporting Bias",
        description="Omitting important information or context that would change understanding",
        examples=[
            "Crime rates increased (without mentioning population growth)",
            "Profits rose 10% (without noting inflation or market context)"
        ],
        severity_weight=0.8
    )
}


def get_bias_definition(bias_type: BiasType) -> BiasDefinition:
    """Get definition for specific bias type"""
    return BIAS_DEFINITIONS[bias_type]


def get_all_bias_types() -> List[BiasType]:
    """Get list of all supported bias types"""
    return list(BiasType)


def get_bias_severity_weight(bias_type: BiasType) -> float:
    """Get severity weight for bias type (0.1 to 1.0)"""
    return BIAS_DEFINITIONS[bias_type].severity_weight