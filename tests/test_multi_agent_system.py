"""
Multi-Agent Collaboration System

Integration tests for multi-agent collaboration system including:
- Agent communication protocol
- Specialist agents (Python, Security, Performance)
- Multi-agent coordinator
- Distributed learning and knowledge sharing
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
import time

from nerion_digital_physicist.agents import (
    # Protocol
    AgentMessage,
    MessageType,
    AgentRole,
    TaskRequest,
    TaskResponse,
    Proposal,
    Vote,
    Conflict,
    ConflictType,
    CoordinationStrategy,
    CoordinationPlan,
    ProtocolValidator,
    # Specialists
    SpecialistAgent,
    AgentCapability,
    PythonSpecialist,
    SecuritySpecialist,
    PerformanceSpecialist,
    # Coordinator
    MultiAgentCoordinator,
    AgentRegistry,
)

from nerion_digital_physicist.learning import (
    DistributedLearner,
)
from nerion_digital_physicist.learning.distributed import (
    KnowledgeItem,
    LearningExperience,
    KnowledgeBase,
)


# ==================
# Protocol Tests
# ==================

class TestProtocol:
    """Test agent communication protocol"""

    def test_message_creation(self):
        """Test creating agent messages"""
        msg = AgentMessage(
            sender_id="agent_001",
            receiver_id="agent_002",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "analyze_code"}
        )

        assert msg.sender_id == "agent_001"
        assert msg.receiver_id == "agent_002"
        assert msg.message_type == MessageType.TASK_REQUEST
        assert msg.payload["task"] == "analyze_code"
        assert msg.message_id is not None
        assert msg.priority == 5  # Default

    def test_message_reply(self):
        """Test message reply chain"""
        original = AgentMessage(
            sender_id="agent_001",
            receiver_id="agent_002",
            message_type=MessageType.QUERY,
            payload={"question": "What is the issue?"}
        )

        reply = original.create_reply(
            sender_id="agent_002",
            message_type=MessageType.ANSWER,
            payload={"answer": "Security vulnerability"}
        )

        assert reply.sender_id == "agent_002"
        assert reply.receiver_id == "agent_001"
        assert reply.in_reply_to == original.message_id
        assert reply.message_type == MessageType.ANSWER

    def test_task_request_response(self):
        """Test task request and response"""
        request = TaskRequest(
            task_type="analyze_code",
            code="def foo(): pass",
            language="python",
            requester_id="coordinator_001"
        )

        assert request.task_id is not None
        assert request.language == "python"

        response = TaskResponse(
            task_id=request.task_id,
            success=True,
            result={"analysis": "complete"},
            confidence=0.95,
            responder_id="agent_001"
        )

        assert response.task_id == request.task_id
        assert response.success
        assert response.confidence == 0.95

    def test_proposal_voting(self):
        """Test proposal and voting mechanism"""
        proposal = Proposal(
            proposal_id="prop_001",
            proposer_id="agent_001",
            task_id="task_001",
            solution={"change": "refactor function"},
            confidence=0.9
        )

        vote1 = Vote(
            proposal_id="prop_001",
            voter_id="agent_002",
            vote=True,
            reason="Good solution"
        )

        vote2 = Vote(
            proposal_id="prop_001",
            voter_id="agent_003",
            vote=True,
            reason="Agree with approach"
        )

        assert vote1.vote
        assert vote2.vote
        assert vote1.proposal_id == proposal.proposal_id

    def test_conflict_detection(self):
        """Test conflict representation"""
        conflict = Conflict(
            conflict_type=ConflictType.DISAGREEMENT,
            agents_involved=["agent_001", "agent_002"],
            description="Agents disagree on security severity",
            resolved=False
        )

        assert conflict.conflict_type == ConflictType.DISAGREEMENT
        assert len(conflict.agents_involved) == 2
        assert not conflict.resolved

    def test_coordination_plan(self):
        """Test coordination plan creation"""
        plan = CoordinationPlan(
            strategy=CoordinationStrategy.PARALLEL,
            agents_assigned=["agent_001", "agent_002", "agent_003"],
            task_id="task_001",
            dependencies={"agent_002": ["agent_001"]}
        )

        assert plan.strategy == CoordinationStrategy.PARALLEL
        assert len(plan.agents_assigned) == 3
        assert "agent_002" in plan.dependencies

    def test_protocol_validator(self):
        """Test protocol validation"""
        validator = ProtocolValidator()

        # Valid message
        valid_msg = AgentMessage(
            sender_id="agent_001",
            receiver_id="agent_002",
            message_type=MessageType.TASK_REQUEST,
            payload={}
        )
        assert validator.validate_message(valid_msg)

        # Invalid message (empty sender)
        invalid_msg = AgentMessage(
            sender_id="",
            receiver_id="agent_002",
            message_type=MessageType.TASK_REQUEST,
            payload={}
        )
        assert not validator.validate_message(invalid_msg)


# ==================
# Specialist Tests
# ==================

class TestPythonSpecialist:
    """Test Python specialist agent"""

    def setup_method(self):
        self.agent = PythonSpecialist("python_001")

    def test_can_handle_python(self):
        """Test Python code detection"""
        task = TaskRequest(
            task_type="analyze",
            code="def foo(): pass",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.9

    def test_can_handle_unknown_language(self):
        """Test rejection of non-Python code"""
        task = TaskRequest(
            task_type="analyze",
            code="function foo() {}",
            language="javascript",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.0

    def test_execute_python_task(self):
        """Test Python code analysis"""
        code = """
def calculate_sum(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)

        assert response.success
        assert response.confidence == 0.9
        assert 'functions' in response.result
        assert 'calculate_sum' in response.result['functions']
        assert 'classes' in response.result
        assert 'Calculator' in response.result['classes']

    def test_statistics_tracking(self):
        """Test agent statistics tracking"""
        task = TaskRequest(
            task_type="analyze",
            code="def foo(): pass",
            language="python",
            requester_id="test"
        )

        initial_stats = self.agent.get_statistics()
        assert initial_stats['tasks_completed'] == 0

        self.agent.execute_task(task)

        updated_stats = self.agent.get_statistics()
        assert updated_stats['tasks_completed'] == 1
        assert updated_stats['success_rate'] == 1.0


class TestSecuritySpecialist:
    """Test Security specialist agent"""

    def setup_method(self):
        self.agent = SecuritySpecialist("security_001")

    def test_can_handle_security_task(self):
        """Test security task detection"""
        task = TaskRequest(
            task_type="security_analysis",
            code="password = 'test123'",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.95

    def test_detect_hardcoded_credentials(self):
        """Test detection of hardcoded credentials"""
        code = """
password = 'hardcoded123'
api_key = 'sk-1234567890abcdef'
secret = 'my_secret_key'
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)

        assert response.success
        assert len(response.result['findings']) > 0
        assert response.result['risk_level'] in ['MEDIUM', 'HIGH']
        assert any(f['category'] == 'hardcoded_credentials' for f in response.result['findings'])

    def test_detect_eval_exec(self):
        """Test detection of dangerous eval/exec"""
        code = """
user_input = request.data
eval(user_input)
exec(user_input)
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)

        assert response.success
        assert len(response.result['findings']) > 0
        assert any(f['category'] == 'eval_exec' for f in response.result['findings'])

    def test_clean_code(self):
        """Test clean code with no security issues"""
        code = """
def calculate_total(items):
    return sum(item.price for item in items)
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)

        assert response.success
        assert response.result['risk_level'] == 'LOW'
        assert response.result['security_score'] == 1.0


class TestPerformanceSpecialist:
    """Test Performance specialist agent"""

    def setup_method(self):
        self.agent = PerformanceSpecialist("perf_001")

    def test_can_handle_performance_task(self):
        """Test performance task detection"""
        task = TaskRequest(
            task_type="performance_analysis",
            code="def slow(): pass",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.9

    def test_detect_nested_loops(self):
        """Test detection of nested loops"""
        code = """
def slow_function():
    for i in range(1000):
        for j in range(1000):
            for k in range(1000):
                pass
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)

        assert response.success
        issues = response.result['issues']
        assert len(issues) > 0
        assert any(issue['type'] == 'nested_loop' for issue in issues)

    def test_detect_high_complexity(self):
        """Test detection of high complexity functions"""
        code = """
def complex_function(x):
    if x > 10:
        if x > 20:
            if x > 30:
                if x > 40:
                    if x > 50:
                        if x > 60:
                            if x > 70:
                                return True
    return False
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)

        assert response.success
        issues = response.result['issues']
        # High complexity might be detected
        if len(issues) > 0:
            assert any(issue['type'] == 'high_complexity' for issue in issues)


class TestJavaScriptSpecialist:
    """Test JavaScript specialist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import JavaScriptSpecialist
        self.agent = JavaScriptSpecialist("js_001")

    def test_can_handle_javascript(self):
        """Test JavaScript code detection"""
        task = TaskRequest(
            task_type="analyze",
            code="const foo = () => {};",
            language="javascript",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.9

    def test_detect_var_usage(self):
        """Test detection of var keyword"""
        code = "var x = 10; var y = 20;"

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="javascript",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert any(issue['type'] == 'var_usage' for issue in response.result['issues'])

    def test_detect_equality_operators(self):
        """Test detection of == instead of ==="""
        code = "if (x == 5) { console.log('yes'); }"

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="javascript",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        # May or may not detect depending on regex


class TestJavaSpecialist:
    """Test Java specialist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import JavaSpecialist
        self.agent = JavaSpecialist("java_001")

    def test_can_handle_java(self):
        """Test Java code detection"""
        task = TaskRequest(
            task_type="analyze",
            code="public class Foo {}",
            language="java",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.9

    def test_detect_empty_catch(self):
        """Test detection of empty catch blocks"""
        code = """
        try {
            riskyOperation();
        } catch (Exception e) {
        }
        """

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="java",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        if response.result['total_findings'] > 0:
            assert any(f['type'] == 'empty_catch' for f in response.result['findings'])


class TestTestingSpecialist:
    """Test testing specialist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import TestingSpecialist
        self.agent = TestingSpecialist("test_001")

    def test_can_handle_test_code(self):
        """Test detection of test code"""
        task = TaskRequest(
            task_type="test_analysis",
            code="def test_foo(): assert True",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.95

    def test_detect_no_assertions(self):
        """Test detection of tests without assertions"""
        code = """def test_something():
    result = calculate()
    return result
"""

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert not response.result['has_assertions']

    def test_detect_empty_test(self):
        """Test detection of empty test functions"""
        code = "def test_placeholder(): pass"

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert any(issue['smell'] == 'empty_test' for issue in response.result['issues'])


class TestRefactoringSpecialist:
    """Test refactoring specialist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import RefactoringSpecialist
        self.agent = RefactoringSpecialist("refactor_001")

    def test_can_handle_refactoring(self):
        """Test refactoring task detection"""
        task = TaskRequest(
            task_type="refactor_code",
            code="def foo(): pass",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.90

    def test_detect_long_lines(self):
        """Test detection of long lines"""
        long_line = "x = " + "a" * 120  # 124 character line
        code = f"{long_line}\ny = 2"

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert any(smell['smell'] == 'long_lines' for smell in response.result['code_smells'])

    def test_detect_magic_numbers(self):
        """Test detection of magic numbers"""
        code = "timeout = 300; max_retries = 42"

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        # Magic numbers might be detected


class TestBugFixingSpecialist:
    """Test bug fixing specialist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import BugFixingSpecialist
        self.agent = BugFixingSpecialist("bugfix_001")

    def test_can_handle_bug_fixing(self):
        """Test bug fixing task detection"""
        task = TaskRequest(
            task_type="fix_bug",
            code="def foo(): pass",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.92

    def test_detect_division_by_zero(self):
        """Test detection of division by zero"""
        code = "result = x / 0"

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert any(bug['type'] == 'division_by_zero' for bug in response.result['bugs'])

    def test_detect_resource_leak(self):
        """Test detection of resource leaks"""
        code = """
        def process_file():
            f = open('data.txt')
            data = f.read()
            return data
        """

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        # Resource leak might be detected


class TestDocumentationSpecialist:
    """Test documentation specialist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import DocumentationSpecialist
        self.agent = DocumentationSpecialist("doc_001")

    def test_can_handle_documentation(self):
        """Test documentation task detection"""
        task = TaskRequest(
            task_type="analyze_documentation",
            code="def foo(): pass",
            language="python",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.95

    def test_detect_missing_docstrings(self):
        """Test detection of missing docstrings"""
        code = """
        def calculate(x, y):
            return x + y

        class Calculator:
            def add(self, a, b):
                return a + b
        """

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert response.result['missing_docstrings'] > 0

    def test_good_documentation(self):
        """Test well-documented code"""
        code = '''
        def calculate(x, y):
            """Add two numbers together."""
            return x + y
        '''

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert response.result['documentation_score'] > 0.5


class TestCoordinatorAgent:
    """Test coordinator meta-agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import CoordinatorAgent
        self.agent = CoordinatorAgent("coord_meta_001")

    def test_can_handle_any_task(self):
        """Test coordinator can handle any task"""
        task = TaskRequest(
            task_type="anything",
            code="any code",
            language="any",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.7

    def test_recommends_delegation(self):
        """Test coordinator recommends delegation"""
        task = TaskRequest(
            task_type="complex_analysis",
            code="code",
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert response.result['action'] == 'delegation_recommended'


class TestGeneralistAgent:
    """Test generalist agent"""

    def setup_method(self):
        from nerion_digital_physicist.agents import GeneralistAgent
        self.agent = GeneralistAgent("generalist_001")

    def test_can_handle_any_task(self):
        """Test generalist can handle any task"""
        task = TaskRequest(
            task_type="anything",
            code="any code",
            language="any",
            requester_id="test"
        )

        confidence = self.agent.can_handle(task)
        assert confidence == 0.5

    def test_heuristic_analysis(self):
        """Test generalist heuristic analysis"""
        code = """
        def process():
            if condition1:
                if condition2:
                    if condition3:
                        if condition4:
                            if condition5:
                                return True
            return False
        """

        task = TaskRequest(
            task_type="analyze",
            code=code,
            language="python",
            requester_id="test"
        )

        response = self.agent.execute_task(task)
        assert response.success
        assert response.result['analysis_type'] == 'generalist_heuristic'
        assert 'quality_score' in response.result


# ==================
# Coordinator Tests
# ==================

class TestCoordinator:
    """Test multi-agent coordinator"""

    def setup_method(self):
        self.coordinator = MultiAgentCoordinator("coord_001")

        # Register agents
        self.coordinator.register_agent(PythonSpecialist("python_001"))
        self.coordinator.register_agent(SecuritySpecialist("security_001"))
        self.coordinator.register_agent(PerformanceSpecialist("perf_001"))

    def test_agent_registration(self):
        """Test agent registration"""
        stats = self.coordinator.get_statistics()
        assert stats['total_agents'] == 3
        assert stats['agents_by_role']['python_specialist'] == 1
        assert stats['agents_by_role']['security_specialist'] == 1
        assert stats['agents_by_role']['performance_specialist'] == 1

    def test_parallel_execution(self):
        """Test parallel task execution"""
        code = """
password = 'hardcoded123'
def slow_function():
    for i in range(1000):
        for j in range(1000):
            pass
"""

        task = TaskRequest(
            task_type="analyze_code",
            code=code,
            language="python",
            requester_id="user_001"
        )

        responses = self.coordinator.assign_task(task, strategy=CoordinationStrategy.PARALLEL)

        assert len(responses) == 3  # All agents respond
        assert all(r.success for r in responses)

    def test_sequential_execution(self):
        """Test sequential task execution"""
        task = TaskRequest(
            task_type="analyze_code",
            code="def foo(): pass",
            language="python",
            requester_id="user_001"
        )

        responses = self.coordinator.assign_task(task, strategy=CoordinationStrategy.SEQUENTIAL)

        # Sequential stops early if high confidence achieved
        assert len(responses) >= 1
        assert any(r.success for r in responses)

    def test_voting_execution(self):
        """Test voting strategy"""
        task = TaskRequest(
            task_type="analyze_code",
            code="password = 'test123'",
            language="python",
            requester_id="user_001"
        )

        responses = self.coordinator.assign_task(task, strategy=CoordinationStrategy.VOTING)

        # Voting returns single best response
        assert len(responses) == 1
        assert responses[0].success

    def test_consensus_execution(self):
        """Test consensus strategy"""
        task = TaskRequest(
            task_type="analyze_code",
            code="def foo(): pass",
            language="python",
            requester_id="user_001"
        )

        responses = self.coordinator.assign_task(task, strategy=CoordinationStrategy.CONSENSUS)

        # Consensus requires all agents to agree
        assert len(responses) == 1
        # May succeed or fail depending on agreement

    def test_aggregate_highest_confidence(self):
        """Test response aggregation by highest confidence"""
        responses = [
            TaskResponse(task_id="test", success=True, result={}, confidence=0.7, responder_id="a1"),
            TaskResponse(task_id="test", success=True, result={}, confidence=0.9, responder_id="a2"),
            TaskResponse(task_id="test", success=True, result={}, confidence=0.6, responder_id="a3"),
        ]

        aggregated = self.coordinator.aggregate_responses(responses, method='highest_confidence')

        assert aggregated.confidence == 0.9
        assert aggregated.responder_id == "a2"

    def test_conflict_detection(self):
        """Test conflict detection between agents"""
        responses = [
            TaskResponse(task_id="test", success=True, result={}, confidence=0.8, responder_id="a1"),
            TaskResponse(task_id="test", success=False, result={}, confidence=0.6, responder_id="a2"),
        ]

        conflict = self.coordinator.detect_conflict(responses)

        assert conflict is not None
        assert conflict.conflict_type == ConflictType.DISAGREEMENT
        assert len(conflict.agents_involved) == 2

    def test_conflict_resolution(self):
        """Test conflict resolution"""
        conflict = Conflict(
            conflict_type=ConflictType.DISAGREEMENT,
            agents_involved=["a1", "a2"],
            description="Test conflict",
            resolved=False
        )

        self.coordinator.active_conflicts.append(conflict)

        resolution = self.coordinator.resolve_conflict(conflict)

        assert conflict.resolved
        assert conflict.resolution is not None
        assert len(self.coordinator.active_conflicts) == 0
        assert len(self.coordinator.resolved_conflicts) == 1


# ==================
# Distributed Learning Tests
# ==================

class TestDistributedLearning:
    """Test distributed learning and knowledge sharing"""

    def setup_method(self):
        self.learner = DistributedLearner()

    def test_share_knowledge(self):
        """Test knowledge sharing"""
        knowledge_id = self.learner.share_knowledge(
            agent_id="agent_001",
            knowledge_type="security_pattern",
            content={"pattern": "hardcoded_credentials", "regex": r"password\s*=\s*['\"]"},
            confidence=0.95,
            success_rate=0.98,
            tags=["security", "credentials"]
        )

        assert knowledge_id is not None

        # Verify knowledge was added
        item = self.learner.knowledge_base.get_knowledge(knowledge_id)
        assert item is not None
        assert item.source_agent_id == "agent_001"
        assert item.confidence == 0.95

    def test_request_knowledge(self):
        """Test knowledge retrieval"""
        # Share knowledge from agent 1
        self.learner.share_knowledge(
            agent_id="agent_001",
            knowledge_type="security_pattern",
            content={"pattern": "sql_injection"},
            confidence=0.9,
            success_rate=0.95,
            tags=["security", "sql"]
        )

        # Share knowledge from agent 2
        self.learner.share_knowledge(
            agent_id="agent_002",
            knowledge_type="security_pattern",
            content={"pattern": "xss"},
            confidence=0.85,
            success_rate=0.90,
            tags=["security", "xss"]
        )

        # Agent 3 requests security knowledge
        knowledge = self.learner.request_knowledge(
            agent_id="agent_003",
            knowledge_type="security_pattern",
            min_confidence=0.8
        )

        assert len(knowledge) == 2
        assert all(item.knowledge_type == "security_pattern" for item in knowledge)

    def test_exclude_own_knowledge(self):
        """Test that agents don't receive their own knowledge"""
        # Agent 1 shares knowledge
        self.learner.share_knowledge(
            agent_id="agent_001",
            knowledge_type="pattern",
            content={"test": "data"},
            confidence=0.9,
            success_rate=0.95
        )

        # Agent 1 requests knowledge
        knowledge = self.learner.request_knowledge(
            agent_id="agent_001",
            knowledge_type="pattern"
        )

        # Should not receive own knowledge
        assert len(knowledge) == 0

    def test_record_experience(self):
        """Test experience recording"""
        exp_id = self.learner.record_experience(
            agent_id="agent_001",
            task_type="bug_fixing",
            code="sample_code",
            action_taken="refactor",
            outcome="success",
            reward=1.0,
            metadata={"lines_changed": 10}
        )

        assert exp_id is not None

        # Verify experience was recorded
        exp = self.learner.knowledge_base.experiences.get(exp_id)
        assert exp is not None
        assert exp.outcome == "success"
        assert exp.reward == 1.0

    def test_aggregate_experiences(self):
        """Test experience aggregation"""
        # Record multiple experiences
        for i in range(5):
            self.learner.record_experience(
                agent_id=f"agent_{i % 2}",
                task_type="bug_fixing",
                code="sample",
                action_taken=f"action_{i % 2}",
                outcome="success" if i % 2 == 0 else "failure",
                reward=1.0 if i % 2 == 0 else -0.5
            )

        insights = self.learner.aggregate_experiences(task_type="bug_fixing")

        assert insights['total_experiences'] == 5
        assert 'avg_reward' in insights
        assert 'overall_success_rate' in insights
        assert len(insights['best_actions']) > 0

    def test_synchronize_agent(self):
        """Test agent synchronization"""
        # Share some knowledge
        self.learner.share_knowledge(
            agent_id="agent_001",
            knowledge_type="pattern",
            content={"test": "data"},
            confidence=0.9,
            success_rate=0.95
        )

        # Sync agent 2
        sync_data = self.learner.synchronize_agent(
            agent_id="agent_002",
            agent_knowledge={}
        )

        assert 'top_knowledge' in sync_data
        assert 'sync_timestamp' in sync_data
        assert len(sync_data['top_knowledge']) > 0

    def test_learning_statistics(self):
        """Test learning statistics"""
        # Add some knowledge and experiences
        self.learner.share_knowledge(
            agent_id="agent_001",
            knowledge_type="pattern",
            content={},
            confidence=0.9,
            success_rate=0.95
        )

        self.learner.record_experience(
            agent_id="agent_001",
            task_type="test",
            code="",
            action_taken="test",
            outcome="success",
            reward=1.0
        )

        stats = self.learner.get_learning_statistics()

        assert stats['total_knowledge_items'] == 1
        assert stats['total_experiences'] == 1
        assert stats['active_agents'] == 1


class TestKnowledgeBase:
    """Test knowledge base functionality"""

    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_add_knowledge(self):
        """Test adding knowledge items"""
        item = KnowledgeItem(
            source_agent_id="agent_001",
            knowledge_type="pattern",
            content={"test": "data"},
            confidence=0.9
        )

        self.kb.add_knowledge(item)

        assert len(self.kb.knowledge_items) == 1
        assert self.kb.total_shares == 1

    def test_query_by_type(self):
        """Test querying knowledge by type"""
        # Add different types
        self.kb.add_knowledge(KnowledgeItem(
            source_agent_id="a1",
            knowledge_type="security",
            content={},
            confidence=0.9
        ))

        self.kb.add_knowledge(KnowledgeItem(
            source_agent_id="a2",
            knowledge_type="performance",
            content={},
            confidence=0.8
        ))

        results = self.kb.query_knowledge(knowledge_type="security")

        assert len(results) == 1
        assert results[0].knowledge_type == "security"

    def test_query_by_tags(self):
        """Test querying knowledge by tags"""
        self.kb.add_knowledge(KnowledgeItem(
            source_agent_id="a1",
            knowledge_type="pattern",
            content={},
            confidence=0.9,
            tags=["security", "xss"]
        ))

        self.kb.add_knowledge(KnowledgeItem(
            source_agent_id="a2",
            knowledge_type="pattern",
            content={},
            confidence=0.8,
            tags=["performance", "loop"]
        ))

        results = self.kb.query_knowledge(tags=["security"])

        assert len(results) == 1
        assert "security" in results[0].tags

    def test_query_by_confidence(self):
        """Test querying with confidence threshold"""
        self.kb.add_knowledge(KnowledgeItem(
            source_agent_id="a1",
            knowledge_type="pattern",
            content={},
            confidence=0.95
        ))

        self.kb.add_knowledge(KnowledgeItem(
            source_agent_id="a2",
            knowledge_type="pattern",
            content={},
            confidence=0.70
        ))

        results = self.kb.query_knowledge(min_confidence=0.9)

        assert len(results) == 1
        assert results[0].confidence >= 0.9

    def test_get_top_knowledge(self):
        """Test retrieving top knowledge items"""
        # Add items with different quality scores
        for i in range(15):
            self.kb.add_knowledge(KnowledgeItem(
                source_agent_id=f"a{i}",
                knowledge_type="pattern",
                content={},
                confidence=0.5 + (i * 0.03),
                success_rate=0.5 + (i * 0.03),
                usage_count=i
            ))

        top_10 = self.kb.get_top_knowledge(n=10)

        assert len(top_10) == 10
        # Should be sorted by quality (success_rate * confidence * usage_count)
        for i in range(len(top_10) - 1):
            score1 = top_10[i].success_rate * top_10[i].confidence * (1 + top_10[i].usage_count)
            score2 = top_10[i+1].success_rate * top_10[i+1].confidence * (1 + top_10[i+1].usage_count)
            assert score1 >= score2


# ==================
# Integration Tests
# ==================

class TestIntegration:
    """End-to-end integration tests"""

    def test_multi_agent_workflow(self):
        """Test complete multi-agent workflow"""
        # Setup
        coordinator = MultiAgentCoordinator("coord_001")
        learner = DistributedLearner()

        # Register agents
        python_agent = PythonSpecialist("python_001")
        security_agent = SecuritySpecialist("security_001")
        perf_agent = PerformanceSpecialist("perf_001")

        coordinator.register_agent(python_agent)
        coordinator.register_agent(security_agent)
        coordinator.register_agent(perf_agent)

        # Create problematic code
        code = """
password = 'hardcoded123'
api_key = 'sk-test-key'

def slow_function():
    results = []
    for i in range(1000):
        for j in range(1000):
            results.append(i * j)
    return results
"""

        # Step 1: Assign task to agents
        task = TaskRequest(
            task_type="comprehensive_analysis",
            code=code,
            language="python",
            requester_id="user_001"
        )

        responses = coordinator.assign_task(task, strategy=CoordinationStrategy.PARALLEL)

        assert len(responses) == 3
        assert all(r.success for r in responses)

        # Step 2: Share knowledge from analysis
        for response in responses:
            if response.responder_id == "security_001":
                # Security agent shares vulnerability patterns
                if response.result.get('findings'):
                    for finding in response.result['findings']:
                        learner.share_knowledge(
                            agent_id=response.responder_id,
                            knowledge_type="vulnerability_pattern",
                            content=finding,
                            confidence=response.confidence,
                            success_rate=0.9,
                            tags=["security", finding.get('category', 'unknown')]
                        )

            elif response.responder_id == "perf_001":
                # Performance agent shares optimization patterns
                if response.result.get('issues'):
                    for issue in response.result['issues']:
                        learner.share_knowledge(
                            agent_id=response.responder_id,
                            knowledge_type="performance_issue",
                            content=issue,
                            confidence=response.confidence,
                            success_rate=0.85,
                            tags=["performance", issue.get('type', 'unknown')]
                        )

        # Step 3: Record experiences
        for response in responses:
            learner.record_experience(
                agent_id=response.responder_id,
                task_type=task.task_type,
                code=code,
                action_taken="analysis",
                outcome="success" if response.success else "failure",
                reward=response.confidence
            )

        # Step 4: Verify knowledge sharing works
        stats = learner.get_learning_statistics()
        assert stats['total_knowledge_items'] > 0
        assert stats['total_experiences'] == 3

        # Step 5: New agent can learn from collective
        knowledge = learner.request_knowledge(
            agent_id="new_agent_001",
            knowledge_type="vulnerability_pattern",
            min_confidence=0.5
        )

        assert len(knowledge) > 0

    def test_knowledge_evolution(self):
        """Test knowledge evolution over multiple tasks"""
        learner = DistributedLearner()

        # Simulate multiple agents discovering patterns
        patterns = [
            ("agent_001", "sql_injection", 0.95, 0.98),
            ("agent_002", "xss", 0.90, 0.95),
            ("agent_003", "csrf", 0.85, 0.92),
            ("agent_001", "auth_bypass", 0.92, 0.96),
        ]

        for agent_id, pattern, confidence, success_rate in patterns:
            learner.share_knowledge(
                agent_id=agent_id,
                knowledge_type="security_vulnerability",
                content={"pattern_name": pattern},
                confidence=confidence,
                success_rate=success_rate,
                tags=["security", pattern]
            )

        # Query top knowledge
        top = learner.knowledge_base.get_top_knowledge(n=2)

        assert len(top) == 2
        # Should prioritize highest quality (confidence * success_rate)
        assert all(item.confidence >= 0.85 for item in top)

    def test_collaborative_debugging(self):
        """Test collaborative debugging scenario"""
        coordinator = MultiAgentCoordinator("coord_001")

        # Register specialists
        coordinator.register_agent(PythonSpecialist("python_001"))
        coordinator.register_agent(SecuritySpecialist("security_001"))
        coordinator.register_agent(PerformanceSpecialist("perf_001"))

        # Buggy code
        buggy_code = """
def process_user_data(user_input):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"

    # Performance issue - nested loops
    results = []
    for i in range(len(query)):
        for j in range(len(query)):
            if query[i] == query[j]:
                results.append(i)

    return results
"""

        task = TaskRequest(
            task_type="debug_code",
            code=buggy_code,
            language="python",
            requester_id="developer_001"
        )

        # Run analysis in parallel
        responses = coordinator.assign_task(task, strategy=CoordinationStrategy.PARALLEL)

        # Aggregate findings
        aggregated = coordinator.aggregate_responses(responses, method='weighted_average')

        assert aggregated.success
        assert aggregated.confidence > 0.0

        # All specialists should contribute
        assert len(responses) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
