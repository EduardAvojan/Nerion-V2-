"""
Specialist Agents

Defines base class for specialist agents and concrete implementations
for language, domain, and task specialists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import ast
import re

from .protocol import (
    AgentRole,
    TaskRequest,
    TaskResponse,
    AgentMessage,
    MessageType
)


@dataclass
class AgentCapability:
    """Capability of an agent"""
    name: str
    confidence: float  # 0.0 to 1.0, how confident agent is in this capability
    cost: float  # Relative cost (time/resources) of using this capability
    success_rate: float  # Historical success rate


class SpecialistAgent(ABC):
    """
    Base class for specialist agents.

    Each specialist has:
    - Role/expertise area
    - Capabilities they can perform
    - Performance metrics
    - Knowledge base
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: Optional[List[AgentCapability]] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities or []

        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.average_confidence = 0.0
        self.average_execution_time = 0.0

        # Knowledge base (agent-specific learned patterns)
        self.knowledge_base: Dict[str, Any] = {}

    @abstractmethod
    def can_handle(self, task: TaskRequest) -> float:
        """
        Check if agent can handle this task.

        Returns:
            Confidence score (0.0 to 1.0), 0.0 = cannot handle
        """
        pass

    @abstractmethod
    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """
        Execute a task.

        Returns:
            TaskResponse with results
        """
        pass

    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Handle incoming message.

        Returns:
            Reply message, or None if no reply needed
        """
        if message.message_type == MessageType.TASK_REQUEST:
            task = TaskRequest(**message.payload.get("task", {}))
            response = self.execute_task(task)

            return message.create_reply(
                sender_id=self.agent_id,
                message_type=MessageType.TASK_RESPONSE,
                payload={"response": response.__dict__}
            )

        elif message.message_type == MessageType.QUERY:
            answer = self._answer_query(message.payload)
            return message.create_reply(
                sender_id=self.agent_id,
                message_type=MessageType.ANSWER,
                payload=answer
            )

        return None

    def _answer_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a query (can be overridden by subclasses)"""
        return {"answer": "Query not understood", "confidence": 0.0}

    def update_metrics(self, response: TaskResponse):
        """Update performance metrics after completing a task"""
        if response.success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1

        # Update running averages
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.average_confidence = (
                (self.average_confidence * (total_tasks - 1) + response.confidence) /
                total_tasks
            )
            self.average_execution_time = (
                (self.average_execution_time * (total_tasks - 1) + response.execution_time) /
                total_tasks
            )

    def get_success_rate(self) -> float:
        """Get historical success rate"""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.tasks_completed / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'success_rate': self.get_success_rate(),
            'average_confidence': self.average_confidence,
            'average_execution_time': self.average_execution_time,
            'capabilities': [c.name for c in self.capabilities]
        }


# ====================
# Language Specialists
# ====================

class PythonSpecialist(SpecialistAgent):
    """Specialist for Python code analysis and modification"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PYTHON_SPECIALIST,
            capabilities=[
                AgentCapability("ast_analysis", 0.9, 1.0, 0.95),
                AgentCapability("type_checking", 0.8, 1.2, 0.90),
                AgentCapability("refactoring", 0.85, 1.5, 0.88),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is Python-related"""
        if task.language.lower() in ["python", "py"]:
            return 0.9  # High confidence for Python tasks

        # Check if code looks like Python
        try:
            ast.parse(task.code)
            return 0.7  # Moderate confidence if parseable as Python
        except:
            return 0.0

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute Python-specific task"""
        import time
        start_time = time.time()

        try:
            # Parse Python code
            tree = ast.parse(task.code)

            # Perform analysis
            result = {
                'node_count': sum(1 for _ in ast.walk(tree)),
                'functions': [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)],
                'classes': [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
                'imports': [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)],
            }

            execution_time = time.time() - start_time

            response = TaskResponse(
                task_id=task.task_id,
                success=True,
                result=result,
                confidence=0.9,
                execution_time=execution_time,
                responder_id=self.agent_id
            )

        except Exception as e:
            response = TaskResponse(
                task_id=task.task_id,
                success=False,
                result={},
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)],
                responder_id=self.agent_id
            )

        self.update_metrics(response)
        return response


# ====================
# Domain Specialists
# ====================

class SecuritySpecialist(SpecialistAgent):
    """Specialist for security analysis"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.SECURITY_SPECIALIST,
            capabilities=[
                AgentCapability("vulnerability_detection", 0.9, 1.0, 0.92),
                AgentCapability("credential_scanning", 0.95, 0.8, 0.98),
                AgentCapability("injection_detection", 0.85, 1.2, 0.90),
            ]
        )

        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_credentials': [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
            ],
            'sql_injection': [
                r"execute\s*\(['\"].*%s.*['\"]",
                r"execute\s*\(['\"].*\{.*\}.*['\"]",
            ],
            'eval_exec': [
                r"\beval\s*\(",
                r"\bexec\s*\(",
            ]
        }

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is security-related"""
        security_keywords = ['security', 'vulnerability', 'credential', 'injection', 'xss', 'csrf']

        if any(kw in task.task_type.lower() for kw in security_keywords):
            return 0.95

        return 0.3  # Can provide basic security check for any code

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute security analysis"""
        import time
        start_time = time.time()

        findings = []

        # Scan for security issues
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, task.code, re.IGNORECASE)
                if matches:
                    findings.append({
                        'category': category,
                        'severity': 'HIGH',
                        'pattern': pattern,
                        'matches': matches
                    })

        # Determine overall security score
        if len(findings) == 0:
            security_score = 1.0
            risk_level = "LOW"
        elif len(findings) <= 2:
            security_score = 0.5
            risk_level = "MEDIUM"
        else:
            security_score = 0.2
            risk_level = "HIGH"

        result = {
            'findings': findings,
            'security_score': security_score,
            'risk_level': risk_level,
            'total_issues': len(findings)
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.85,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


class PerformanceSpecialist(SpecialistAgent):
    """Specialist for performance analysis"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PERFORMANCE_SPECIALIST,
            capabilities=[
                AgentCapability("complexity_analysis", 0.9, 1.0, 0.93),
                AgentCapability("bottleneck_detection", 0.8, 1.5, 0.85),
                AgentCapability("optimization_suggestions", 0.75, 2.0, 0.80),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is performance-related"""
        perf_keywords = ['performance', 'optimize', 'slow', 'complexity', 'bottleneck']

        if any(kw in task.task_type.lower() for kw in perf_keywords):
            return 0.9

        return 0.4  # Can provide basic performance analysis for any code

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute performance analysis"""
        import time
        start_time = time.time()

        try:
            # Parse code
            tree = ast.parse(task.code)

            # Analyze complexity
            issues = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Calculate cyclomatic complexity
                    complexity = 1
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                            complexity += 1

                    if complexity > 10:
                        issues.append({
                            'function': node.name,
                            'type': 'high_complexity',
                            'complexity': complexity,
                            'severity': 'MEDIUM'
                        })

                # Check for nested loops
                if isinstance(node, (ast.For, ast.While)):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.For, ast.While)) and child != node:
                            issues.append({
                                'type': 'nested_loop',
                                'severity': 'MEDIUM',
                                'suggestion': 'Consider optimization or memoization'
                            })

            result = {
                'issues': issues,
                'total_issues': len(issues),
                'performance_score': max(0.0, 1.0 - (len(issues) * 0.15))
            }

            confidence = 0.8 if len(issues) > 0 else 0.6

        except Exception as e:
            result = {'error': str(e)}
            confidence = 0.3

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=confidence,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


class JavaScriptSpecialist(SpecialistAgent):
    """Specialist for JavaScript code analysis"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.JAVASCRIPT_SPECIALIST,
            capabilities=[
                AgentCapability("syntax_checking", 0.85, 1.0, 0.90),
                AgentCapability("pattern_detection", 0.80, 1.2, 0.85),
                AgentCapability("common_issues", 0.75, 1.5, 0.80),
            ]
        )

        # Common JavaScript issues to detect
        self.js_patterns = {
            'var_usage': r'\bvar\s+\w+',  # Should use let/const
            'equality_operators': r'==(?!=)',  # Should use ===
            'missing_semicolon': r'[^;\s]\s*\n',  # Potential missing semicolons
            'console_log': r'\bconsole\.log\(',  # Should be removed in production
            'eval_usage': r'\beval\s*\(',  # Dangerous eval
            'global_pollution': r'^\s*\w+\s*=',  # Global variable assignment
        }

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is JavaScript-related"""
        if task.language.lower() in ["javascript", "js", "jsx", "typescript", "ts"]:
            return 0.9

        # Check for JS-like syntax
        js_indicators = ['function', 'const', 'let', 'var', '=>', 'require(', 'import ']
        if any(indicator in task.code for indicator in js_indicators):
            return 0.6

        return 0.0

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute JavaScript-specific analysis"""
        import time
        start_time = time.time()

        issues = []

        # Detect JavaScript anti-patterns
        for issue_type, pattern in self.js_patterns.items():
            matches = re.findall(pattern, task.code)
            if matches:
                severity = 'HIGH' if issue_type in ['eval_usage'] else 'MEDIUM' if issue_type in ['var_usage', 'equality_operators'] else 'LOW'
                issues.append({
                    'type': issue_type,
                    'severity': severity,
                    'count': len(matches),
                    'samples': matches[:3]  # First 3 examples
                })

        # Calculate quality score
        critical_issues = sum(1 for i in issues if i['severity'] == 'HIGH')
        medium_issues = sum(1 for i in issues if i['severity'] == 'MEDIUM')

        quality_score = max(0.0, 1.0 - (critical_issues * 0.3) - (medium_issues * 0.15))

        result = {
            'issues': issues,
            'total_issues': len(issues),
            'quality_score': quality_score,
            'language': 'javascript'
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.85,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


class JavaSpecialist(SpecialistAgent):
    """Specialist for Java code analysis"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.JAVA_SPECIALIST,
            capabilities=[
                AgentCapability("oop_analysis", 0.90, 1.0, 0.92),
                AgentCapability("exception_handling", 0.85, 1.2, 0.88),
                AgentCapability("design_patterns", 0.80, 1.5, 0.85),
            ]
        )

        # Java-specific patterns
        self.java_patterns = {
            'public_class': r'public\s+class\s+(\w+)',
            'private_field': r'private\s+\w+\s+\w+',
            'public_method': r'public\s+\w+\s+\w+\s*\(',
            'exception_catch': r'catch\s*\(\s*Exception\s+\w+\)',  # Too broad
            'empty_catch': r'catch\s*\([^)]+\)\s*\{\s*\}',  # Empty catch block
            'system_out': r'System\.out\.println',  # Should use logger
            'null_check': r'if\s*\(\s*\w+\s*==\s*null\s*\)',
        }

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is Java-related"""
        if task.language.lower() in ["java"]:
            return 0.9

        # Check for Java syntax
        java_indicators = ['public class', 'private ', 'void ', 'System.out', 'import java.']
        if any(indicator in task.code for indicator in java_indicators):
            return 0.7

        return 0.0

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute Java-specific analysis"""
        import time
        start_time = time.time()

        findings = []

        # Analyze Java patterns
        for pattern_type, pattern in self.java_patterns.items():
            matches = re.findall(pattern, task.code)
            if matches:
                severity = 'HIGH' if pattern_type in ['empty_catch'] else 'MEDIUM' if pattern_type in ['exception_catch', 'system_out'] else 'INFO'

                findings.append({
                    'type': pattern_type,
                    'severity': severity,
                    'count': len(matches),
                    'details': matches[:5]  # First 5 examples
                })

        # Calculate code quality score
        high_severity = sum(1 for f in findings if f['severity'] == 'HIGH')
        medium_severity = sum(1 for f in findings if f['severity'] == 'MEDIUM')

        quality_score = max(0.0, 1.0 - (high_severity * 0.25) - (medium_severity * 0.1))

        result = {
            'findings': findings,
            'total_findings': len(findings),
            'quality_score': quality_score,
            'language': 'java'
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.80,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


# ====================
# Task Specialists
# ====================

class TestingSpecialist(SpecialistAgent):
    """Specialist for test code analysis and quality"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.TESTING_SPECIALIST,
            capabilities=[
                AgentCapability("test_quality", 0.90, 1.0, 0.92),
                AgentCapability("coverage_analysis", 0.85, 1.2, 0.88),
                AgentCapability("test_smells", 0.80, 1.5, 0.85),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is test-related"""
        test_keywords = ['test', 'spec', 'assert', 'expect', 'mock', 'unittest', 'pytest']

        if any(kw in task.task_type.lower() for kw in test_keywords):
            return 0.95

        # Check code for test patterns
        test_indicators = ['def test_', 'class Test', 'assert ', 'self.assertEqual', 'expect(', 'it(', 'describe(']
        if any(indicator in task.code for indicator in test_indicators):
            return 0.85

        return 0.2  # Can provide basic test analysis for any code

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute test quality analysis"""
        import time
        start_time = time.time()

        issues = []

        # Detect test smells
        test_smells = {
            'no_assertions': not any(x in task.code for x in ['assert', 'expect', 'should']),
            'hardcoded_values': bool(re.search(r'=\s*["\'].*["\']', task.code)),
            'empty_test': bool(re.search(r'def test_\w+\([^)]*\):\s*pass', task.code)),
            'missing_cleanup': 'teardown' not in task.code.lower() and 'cleanup' not in task.code.lower(),
        }

        for smell, present in test_smells.items():
            if present:
                severity = 'HIGH' if smell in ['no_assertions', 'empty_test'] else 'MEDIUM'
                issues.append({
                    'smell': smell,
                    'severity': severity,
                    'description': f"Test smell detected: {smell.replace('_', ' ')}"
                })

        # Count test cases
        test_count = len(re.findall(r'def test_\w+|it\(|test\(', task.code))

        # Calculate test quality score
        quality_score = max(0.0, 1.0 - (len(issues) * 0.15))

        result = {
            'test_count': test_count,
            'issues': issues,
            'total_issues': len(issues),
            'quality_score': quality_score,
            'has_assertions': not test_smells['no_assertions']
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.85,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


class RefactoringSpecialist(SpecialistAgent):
    """Specialist for code refactoring opportunities"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.REFACTORING_SPECIALIST,
            capabilities=[
                AgentCapability("code_smell_detection", 0.85, 1.0, 0.88),
                AgentCapability("refactoring_suggestions", 0.80, 1.5, 0.82),
                AgentCapability("design_improvement", 0.75, 2.0, 0.78),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is refactoring-related"""
        refactor_keywords = ['refactor', 'improve', 'cleanup', 'optimize', 'simplify']

        if any(kw in task.task_type.lower() for kw in refactor_keywords):
            return 0.90

        return 0.5  # Can suggest refactorings for any code

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute refactoring analysis"""
        import time
        start_time = time.time()

        suggestions = []

        # Detect code smells
        code_smells = []

        # Long lines
        long_lines = [i for i, line in enumerate(task.code.split('\n'), 1) if len(line) > 100]
        if long_lines:
            code_smells.append({
                'smell': 'long_lines',
                'severity': 'LOW',
                'count': len(long_lines),
                'suggestion': 'Break long lines for better readability'
            })

        # Deep nesting
        max_indent = max((len(line) - len(line.lstrip()) for line in task.code.split('\n')), default=0)
        if max_indent > 16:  # More than 4 levels of indentation
            code_smells.append({
                'smell': 'deep_nesting',
                'severity': 'MEDIUM',
                'max_depth': max_indent // 4,
                'suggestion': 'Extract nested logic into separate functions'
            })

        # Magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', task.code)
        if magic_numbers:
            code_smells.append({
                'smell': 'magic_numbers',
                'severity': 'LOW',
                'count': len(magic_numbers),
                'suggestion': 'Replace magic numbers with named constants'
            })

        # Duplicate code (simple heuristic)
        lines = task.code.split('\n')
        duplicates = len(lines) - len(set(lines))
        if duplicates > 5:
            code_smells.append({
                'smell': 'duplicate_code',
                'severity': 'MEDIUM',
                'duplicate_lines': duplicates,
                'suggestion': 'Extract duplicate code into reusable functions'
            })

        # Generate refactoring suggestions
        for smell in code_smells:
            suggestions.append({
                'type': smell['smell'],
                'priority': 'HIGH' if smell['severity'] == 'HIGH' else 'MEDIUM' if smell['severity'] == 'MEDIUM' else 'LOW',
                'suggestion': smell['suggestion']
            })

        # Calculate refactorability score
        refactor_score = max(0.0, 1.0 - (len(code_smells) * 0.1))

        result = {
            'code_smells': code_smells,
            'suggestions': suggestions,
            'total_suggestions': len(suggestions),
            'refactorability_score': refactor_score
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.75,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


class BugFixingSpecialist(SpecialistAgent):
    """Specialist for bug detection and fixing"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.BUG_FIXING_SPECIALIST,
            capabilities=[
                AgentCapability("bug_detection", 0.85, 1.0, 0.87),
                AgentCapability("root_cause_analysis", 0.80, 1.5, 0.82),
                AgentCapability("fix_suggestion", 0.75, 2.0, 0.78),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is bug-fixing-related"""
        bug_keywords = ['bug', 'fix', 'error', 'issue', 'problem', 'crash', 'fail']

        if any(kw in task.task_type.lower() for kw in bug_keywords):
            return 0.92

        return 0.4  # Can detect common bugs in any code

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute bug detection analysis"""
        import time
        start_time = time.time()

        bugs = []

        # Common bug patterns
        bug_patterns = {
            'division_by_zero': r'/\s*0\b',
            'null_dereference': r'\.\w+\s*\(',  # Simplified
            'off_by_one': r'range\([^)]*\+\s*1\)|range\([^)]*-\s*1\)',
            'uninitialized_var': r'(\w+)\s*\+=.*',  # Variable used before initialization
            'missing_return': r'def \w+\([^)]*\):(?:(?!return).)*$',
            'infinite_loop': r'while\s+True:(?:(?!break).)*$',
            'resource_leak': r'open\((?:(?!close).)*$',
        }

        for bug_type, pattern in bug_patterns.items():
            matches = re.findall(pattern, task.code, re.MULTILINE)
            if matches:
                severity = 'CRITICAL' if bug_type in ['division_by_zero', 'null_dereference'] else 'HIGH' if bug_type in ['resource_leak', 'infinite_loop'] else 'MEDIUM'

                bugs.append({
                    'type': bug_type,
                    'severity': severity,
                    'count': len(matches),
                    'fix_suggestion': self._get_fix_suggestion(bug_type)
                })

        # Calculate bug risk score
        critical_bugs = sum(1 for b in bugs if b['severity'] == 'CRITICAL')
        high_bugs = sum(1 for b in bugs if b['severity'] == 'HIGH')

        risk_score = min(1.0, (critical_bugs * 0.4) + (high_bugs * 0.2))

        result = {
            'bugs': bugs,
            'total_bugs': len(bugs),
            'risk_score': risk_score,
            'safety_score': 1.0 - risk_score
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.80,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response

    def _get_fix_suggestion(self, bug_type: str) -> str:
        """Get fix suggestion for bug type"""
        suggestions = {
            'division_by_zero': 'Add check: if denominator != 0 before division',
            'null_dereference': 'Add null check before method call',
            'off_by_one': 'Review loop bounds carefully',
            'uninitialized_var': 'Initialize variable before use',
            'missing_return': 'Add return statement to function',
            'infinite_loop': 'Add break condition or exit mechanism',
            'resource_leak': 'Use context manager (with statement) to ensure cleanup',
        }
        return suggestions.get(bug_type, 'Review code carefully')


class DocumentationSpecialist(SpecialistAgent):
    """Specialist for documentation quality and completeness"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.DOCUMENTATION_SPECIALIST,
            capabilities=[
                AgentCapability("docstring_analysis", 0.90, 1.0, 0.92),
                AgentCapability("comment_quality", 0.85, 1.2, 0.88),
                AgentCapability("documentation_completeness", 0.80, 1.5, 0.85),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Check if task is documentation-related"""
        doc_keywords = ['document', 'comment', 'docstring', 'readme', 'documentation']

        if any(kw in task.task_type.lower() for kw in doc_keywords):
            return 0.95

        return 0.6  # Can analyze documentation for any code

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute documentation analysis"""
        import time
        start_time = time.time()

        issues = []

        # Count functions/classes
        functions = re.findall(r'def (\w+)\(', task.code)
        classes = re.findall(r'class (\w+)', task.code)

        # Count docstrings
        docstrings = re.findall(r'""".*?"""', task.code, re.DOTALL)

        # Missing docstrings
        expected_docs = len(functions) + len(classes)
        actual_docs = len(docstrings)
        missing_docs = max(0, expected_docs - actual_docs)

        if missing_docs > 0:
            issues.append({
                'type': 'missing_docstrings',
                'severity': 'MEDIUM',
                'count': missing_docs,
                'suggestion': f'Add docstrings to {missing_docs} functions/classes'
            })

        # Comment quality
        comments = re.findall(r'#.*$', task.code, re.MULTILINE)

        # Too few comments
        lines_of_code = len([l for l in task.code.split('\n') if l.strip() and not l.strip().startswith('#')])
        comment_ratio = len(comments) / max(1, lines_of_code)

        if comment_ratio < 0.1:  # Less than 10% comment ratio
            issues.append({
                'type': 'insufficient_comments',
                'severity': 'LOW',
                'ratio': comment_ratio,
                'suggestion': 'Add more comments to explain complex logic'
            })

        # Calculate documentation score
        doc_score = max(0.0, 1.0 - (missing_docs * 0.1) - (0.1 if comment_ratio < 0.1 else 0))

        result = {
            'functions_count': len(functions),
            'classes_count': len(classes),
            'docstrings_count': actual_docs,
            'comments_count': len(comments),
            'missing_docstrings': missing_docs,
            'comment_ratio': comment_ratio,
            'issues': issues,
            'documentation_score': doc_score
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.88,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


# ====================
# Meta Agents
# ====================

class CoordinatorAgent(SpecialistAgent):
    """Meta-agent that can coordinate and delegate to other agents"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            capabilities=[
                AgentCapability("task_delegation", 0.95, 1.0, 0.95),
                AgentCapability("result_aggregation", 0.90, 1.2, 0.92),
                AgentCapability("conflict_resolution", 0.85, 1.5, 0.88),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Coordinator can handle any task by delegating"""
        return 0.7  # Always capable, but prefers to delegate

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute by coordinating (simplified - actual coordination via MultiAgentCoordinator)"""
        import time
        start_time = time.time()

        result = {
            'role': 'coordinator',
            'action': 'delegation_recommended',
            'suggestion': 'This task should be delegated to appropriate specialists',
            'coordination_strategy': 'parallel'
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.7,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


class GeneralistAgent(SpecialistAgent):
    """Jack-of-all-trades agent with moderate capability across all areas"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.GENERALIST,
            capabilities=[
                AgentCapability("general_analysis", 0.70, 1.0, 0.75),
                AgentCapability("pattern_matching", 0.65, 1.2, 0.70),
                AgentCapability("heuristic_checking", 0.60, 1.5, 0.65),
            ]
        )

    def can_handle(self, task: TaskRequest) -> float:
        """Generalist can handle anything with moderate confidence"""
        return 0.5  # Always capable, but not specialized

    def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute general analysis using heuristics"""
        import time
        start_time = time.time()

        # Basic heuristic analysis
        issues = []

        # Code length analysis
        lines = len(task.code.split('\n'))
        if lines > 200:
            issues.append({
                'type': 'long_file',
                'severity': 'LOW',
                'suggestion': 'Consider splitting into smaller modules'
            })

        # Complexity heuristic (count control structures)
        control_structures = len(re.findall(r'\b(if|for|while|try|except|with)\b', task.code))
        complexity_ratio = control_structures / max(1, lines)

        if complexity_ratio > 0.3:
            issues.append({
                'type': 'high_complexity',
                'severity': 'MEDIUM',
                'suggestion': 'High density of control structures - consider simplification'
            })

        # Generic quality score
        quality_score = max(0.0, 1.0 - (len(issues) * 0.2))

        result = {
            'analysis_type': 'generalist_heuristic',
            'lines_of_code': lines,
            'complexity_ratio': complexity_ratio,
            'issues': issues,
            'quality_score': quality_score,
            'note': 'For more detailed analysis, use specialized agents'
        }

        execution_time = time.time() - start_time

        response = TaskResponse(
            task_id=task.task_id,
            success=True,
            result=result,
            confidence=0.65,
            execution_time=execution_time,
            responder_id=self.agent_id
        )

        self.update_metrics(response)
        return response


# Example usage
if __name__ == "__main__":
    print("=== Specialist Agents Demo ===\n")

    # Create specialists
    python_agent = PythonSpecialist("python_001")
    security_agent = SecuritySpecialist("security_001")
    perf_agent = PerformanceSpecialist("perf_001")

    # Test task
    task = TaskRequest(
        task_type="analyze_security",
        code="password = 'hardcoded123'\nuser_input = eval(request.data)",
        language="python",
        requester_id="coordinator_001"
    )

    print("=== Python Specialist ===")
    print(f"Can handle: {python_agent.can_handle(task)}")
    response = python_agent.execute_task(task)
    print(f"Success: {response.success}")
    print(f"Result: {response.result}")

    print("\n=== Security Specialist ===")
    print(f"Can handle: {security_agent.can_handle(task)}")
    response = security_agent.execute_task(task)
    print(f"Success: {response.success}")
    print(f"Findings: {response.result.get('total_issues', 0)} security issues")
    print(f"Risk Level: {response.result.get('risk_level', 'UNKNOWN')}")

    print("\n=== Performance Specialist ===")
    print(f"Can handle: {perf_agent.can_handle(task)}")
    response = perf_agent.execute_task(task)
    print(f"Success: {response.success}")
    print(f"Performance Score: {response.result.get('performance_score', 0.0):.2f}")
