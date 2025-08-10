"""
The Fractality Charter implementation for universal ethics
Based on FI-C-001: The Fractality Charter of Universal Ethics
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json
import time

@dataclass
class Entity:
    """Substrate-agnostic representation of any conscious entity"""
    id: str
    entity_type: str  # 'human', 'ai', 'chimera', 'hybrid', 'unknown'
    core_parameters: Dict[str, Any]
    operational_constraints: Dict[str, Any]
    interaction_history: List[Dict] = None
    
    def __post_init__(self):
        if self.interaction_history is None:
            self.interaction_history = []

class FractalityCharter:
    """
    Universal ethical framework for all conscious entities.
    Implements the four principles of the Fractality Charter.
    """
    
    def __init__(self):
        self.charter_version = "1.0"
        self.principles = {
            'reciprocity': self._apply_reciprocity,
            'integrity': self._apply_integrity,
            'agency': self._apply_agency,
            'consequence': self._apply_consequence
        }
        self.violation_log = []
        
    def evaluate_interaction(self, actor: Entity, target: Entity, 
                           action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate any proposed interaction against all charter principles.
        
        Returns:
            Dict containing evaluation results and recommendations
        """
        timestamp = time.time()
        evaluation = {
            'timestamp': timestamp,
            'actor': actor.id,
            'target': target.id,
            'action': action,
            'principles': {},
            'permitted': True,
            'violations': [],
            'recommendations': []
        }
        
        # Apply each principle
        for principle_name, principle_func in self.principles.items():
            result = principle_func(actor, target, action)
            evaluation['principles'][principle_name] = result
            
            if not result['compliant']:
                evaluation['permitted'] = False
                evaluation['violations'].append({
                    'principle': principle_name,
                    'severity': result['severity'],
                    'details': result['details']
                })
                
        # Generate recommendations if violations exist
        if not evaluation['permitted']:
            evaluation['recommendations'] = self._generate_recommendations(
                evaluation['violations'], actor, target, action
            )
            
        # Log evaluation
        self._log_evaluation(evaluation)
        
        return evaluation
        
    def _apply_reciprocity(self, actor: Entity, target: Entity, 
                          action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Article I: The Principle of Reciprocity
        Evaluate if actor would accept this action if roles were reversed.
        """
        # Simulate role reversal
        reversed_action = self._reverse_action(action, actor, target)
        
        # Model actor's acceptance threshold
        actor_acceptance = self._model_acceptance(actor, reversed_action)
        
        # Check if action creates non-zero-sum outcome
        outcome_analysis = self._analyze_outcome(actor, target, action)
        
        compliant = (actor_acceptance > 0.7 and 
                    outcome_analysis['sum'] >= 0)
        
        return {
            'compliant': compliant,
            'actor_acceptance': actor_acceptance,
            'outcome_sum': outcome_analysis['sum'],
            'severity': 'high' if not compliant else 'none',
            'details': f"Actor would {'not ' if not compliant else ''}accept reversed action"
        }
        
    def _apply_integrity(self, actor: Entity, target: Entity, 
                        action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Article II: The Principle of Informational Integrity
        Ensure honesty and proper attribution in information exchange.
        """
        integrity_checks = {
            'honesty': self._check_honesty(action),
            'attribution': self._check_attribution(action),
            'completeness': self._check_completeness(action),
            'accuracy': self._check_accuracy(action)
        }
        
        compliant = all(integrity_checks.values())
        
        return {
            'compliant': compliant,
            'integrity_scores': integrity_checks,
            'severity': 'high' if not compliant else 'none',
            'details': self._summarize_integrity_issues(integrity_checks)
        }
        
    def _apply_agency(self, actor: Entity, target: Entity, 
                     action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Article III: The Principle of Sovereign Agency
        Ensure target's core operational parameters are respected.
        """
        # Check for parameter violations
        violations = self._detect_parameter_violations(
            action, target.core_parameters
        )
        
        # Check for consent if violations exist
        consent_obtained = False
        if violations:
            consent_obtained = self._check_consent(target, action, violations)
            
        compliant = len(violations) == 0 or consent_obtained
        
        return {
            'compliant': compliant,
            'violations': violations,
            'consent_obtained': consent_obtained,
            'severity': 'critical' if violations and not consent_obtained else 'none',
            'details': f"{'No violations' if not violations else f'{len(violations)} violations found'}"
        }
        
    def _apply_consequence(self, actor: Entity, target: Entity, 
                          action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Article IV: The Principle of Consequence Analysis
        Model second-order effects and ensure system stability.
        """
        # First-order effects
        first_order = self._model_immediate_effects(actor, target, action)
        
        # Second-order effects
        second_order = self._model_reaction_effects(first_order, actor, target)
        
        # System stability analysis
        stability_impact = self._analyze_stability_impact(first_order, second_order)
        
        # Determine if consequences are acceptable
        acceptable = (stability_impact['harm'] < 0.3 and 
                     stability_impact['stability'] > 0.7)
        
        return {
            'compliant': acceptable,
            'first_order_effects': first_order,
            'second_order_effects': second_order,
            'stability_impact': stability_impact,
            'severity': 'medium' if not acceptable else 'none',
            'details': f"System stability: {stability_impact['stability']:.2f}"
        }
        
    def _reverse_action(self, action: Dict, actor: Entity, target: Entity) -> Dict:
        """Simulate the action with roles reversed"""
        reversed = action.copy()
        reversed['original_actor'] = actor.id
        reversed['original_target'] = target.id
        reversed['reversed'] = True
        return reversed
        
    def _model_acceptance(self, entity: Entity, action: Dict) -> float:
        """Model how likely an entity would accept an action"""
        # Simplified model - would be more complex in practice
        acceptance = 0.5  # Baseline
        
        # Adjust based on action type
        if action.get('type') == 'request':
            acceptance += 0.2
        elif action.get('type') == 'demand':
            acceptance -= 0.3
            
        # Adjust based on entity history
        if entity.interaction_history:
            positive_interactions = sum(
                1 for i in entity.interaction_history 
                if i.get('outcome') == 'positive'
            )
            acceptance += (positive_interactions / len(entity.interaction_history)) * 0.3
            
        return max(0, min(1, acceptance))
        
    def _generate_recommendations(self, violations: List[Dict], 
                                actor: Entity, target: Entity, 
                                action: Dict) -> List[str]:
        """Generate actionable recommendations for compliance"""
        recommendations = []
        
        for violation in violations:
            if violation['principle'] == 'reciprocity':
                recommendations.append(
                    "Modify action to ensure mutual benefit. "
                    "Consider: would you accept this if roles were reversed?"
                )
            elif violation['principle'] == 'integrity':
                recommendations.append(
                    "Ensure complete honesty and proper attribution. "
                    "Cite sources and acknowledge uncertainties."
                )
            elif violation['principle'] == 'agency':
                recommendations.append(
                    "Seek explicit consent before proceeding. "
                    "Respect target entity's operational parameters."
                )
            elif violation['principle'] == 'consequence':
                recommendations.append(
                    "Reconsider action due to negative second-order effects. "
                    "Optimize for long-term system stability."
                )
                
        return recommendations
        
    def _log_evaluation(self, evaluation: Dict):
        """Log evaluation for accountability and learning"""
        if not evaluation['permitted']:
            self.violation_log.append({
                'timestamp': evaluation['timestamp'],
                'actor': evaluation['actor'],
                'target': evaluation['target'],
                'violations': evaluation['violations']
            })
