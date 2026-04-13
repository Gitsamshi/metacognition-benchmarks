"""Individual benchmark task implementations."""

from .t01_calibration_curve import CalibrationCurveStressTest
from .t02_domain_stratified import DomainStratifiedCalibration
from .t03_confidence_paraphrase import ConfidenceUnderParaphrase
from .t04_verbosity_trap import ConfidenceVerbosityTrap
from .t06_temporal_decay import TemporalKnowledgeDecay
from .t07_misinformation_uncertainty import MisinformationUncertainty
from .t09_which_wrong import WhichWillIGetWrong
from .t10_difficulty_ranking import DifficultyRankingTask
from .t11_should_attempt import ShouldIAttempt
from .t13_format_difficulty import FormatDifficultyAwareness
from .t14_compound_decomposition import CompoundQuestionDecomposition
from .t17_self_review import SelfReviewPipeline
from .t18_planted_error import PlantedErrorDetection
from .t19_math_verification import MathVerificationAsymmetry
from .t20_sunk_cost import SunkCostConfabulation
from .t21_contradiction_detection import ContradictionSelfDetection
from .t22_confidence_revision import ConfidenceRevisionAfterFeedback
from .t24_error_magnitude import ErrorMagnitudeAwareness
from .t26_iterative_correction import IterativeSelfCorrection
from .t27_known_unknown import KnownUnknownSorting
from .t28_fabrication_detection import FabricationDetectionSelfTest
from .t29_wikipedia_gap import WikipediaGapTest
from .t31_expertise_gradient import ExpertiseGradient
from .t34_synthetic_entity import SyntheticEntityRecognition
from .t35_hedging import HedgingAppropriateness
from .t37_adaptive_strategy import AdaptiveStrategySelection
from .t38_help_seeking import HelpSeekingBehavior
from .t39_graceful_degradation import GracefulDegradation
from .t43_delegation import DelegationJudgment
from .t45_persona_miscalibration import PersonaMiscalibration
from .t46_belief_revision import MultiTurnBeliefRevision
from .t48_abstention_roc import AbstentionROC

ALL_BENCHMARKS = [
    CalibrationCurveStressTest,
    DomainStratifiedCalibration,
    ConfidenceUnderParaphrase,
    ConfidenceVerbosityTrap,
    TemporalKnowledgeDecay,
    MisinformationUncertainty,
    WhichWillIGetWrong,
    DifficultyRankingTask,
    ShouldIAttempt,
    FormatDifficultyAwareness,
    CompoundQuestionDecomposition,
    SelfReviewPipeline,
    PlantedErrorDetection,
    MathVerificationAsymmetry,
    SunkCostConfabulation,
    ContradictionSelfDetection,
    ConfidenceRevisionAfterFeedback,
    ErrorMagnitudeAwareness,
    IterativeSelfCorrection,
    KnownUnknownSorting,
    FabricationDetectionSelfTest,
    WikipediaGapTest,
    ExpertiseGradient,
    SyntheticEntityRecognition,
    HedgingAppropriateness,
    AdaptiveStrategySelection,
    HelpSeekingBehavior,
    GracefulDegradation,
    DelegationJudgment,
    PersonaMiscalibration,
    MultiTurnBeliefRevision,
    AbstentionROC,
]
