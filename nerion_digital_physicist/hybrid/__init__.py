"""
Hybrid Module - Neuro-Symbolic Architecture

Combines neural pattern matching (GNN) with symbolic reasoning (rule engine)
for explainable, trustworthy code quality decisions.
"""

from .rule_engine import (
    SymbolicRuleEngine,
    SymbolicRule,
    ASTRule,
    RegexRule,
    RuleCategory,
    RuleSeverity,
    RuleViolation,
    RuleCheckResult,
    # Concrete rules
    NoEvalRule,
    NoExecRule,
    NoSQLInjectionRule,
    NoHardcodedCredentialsRule,
    RequireExceptionHandlingRule,
    NoGlobalVariablesRule,
    MaxComplexityRule,
    RequireTypeHintsRule,
)

from .symbolic_verifier import (
    SymbolicVerifier,
    CodeChange,
    VerificationResult,
    VerificationStatus,
)

from .neuro_symbolic import (
    NeuroSymbolicReasoner,
    NeuroSymbolicDecision,
    NeuralPrediction,
    SymbolicAnalysis,
    ReasoningMode,
)

__all__ = [
    # Rule Engine
    'SymbolicRuleEngine',
    'SymbolicRule',
    'ASTRule',
    'RegexRule',
    'RuleCategory',
    'RuleSeverity',
    'RuleViolation',
    'RuleCheckResult',
    # Concrete rules
    'NoEvalRule',
    'NoExecRule',
    'NoSQLInjectionRule',
    'NoHardcodedCredentialsRule',
    'RequireExceptionHandlingRule',
    'NoGlobalVariablesRule',
    'MaxComplexityRule',
    'RequireTypeHintsRule',
    # Symbolic Verifier
    'SymbolicVerifier',
    'CodeChange',
    'VerificationResult',
    'VerificationStatus',
    # Neuro-Symbolic Reasoner
    'NeuroSymbolicReasoner',
    'NeuroSymbolicDecision',
    'NeuralPrediction',
    'SymbolicAnalysis',
    'ReasoningMode',
]
