"""
Terminal Output Parser for Nerion Mission Control
================================================

Parses terminal output to extract structured events for dashboard updates.
"""
import re
from typing import Optional, Dict, Any, List


class OutputParser:
    """Parse terminal output and generate dashboard events."""

    def __init__(self):
        # Patterns for different command outputs
        self.patterns = {
            # Health check patterns
            'voice_ready': re.compile(r'Voice Stack:\s*(Ready|Online|Active)', re.IGNORECASE),
            'voice_offline': re.compile(r'Voice Stack:\s*(Offline|Inactive|Disabled)', re.IGNORECASE),
            'network_online': re.compile(r'Network Gate:\s*(Online|Active|Connected)', re.IGNORECASE),
            'network_offline': re.compile(r'Network Gate:\s*(Offline|Inactive|Disconnected)', re.IGNORECASE),
            'coverage': re.compile(r'Coverage:\s*(\d+)%'),
            'errors': re.compile(r'Errors:\s*(\d+)'),
            'memory_count': re.compile(r'Memory:\s*(\d+)\s*entries', re.IGNORECASE),

            # Autonomous actions
            'autonomous_fix': re.compile(r'\[AUTONOMOUS\].*(?:Fixed|Patched|Repaired)\s+(?:bug|issue|error)\s+in\s+(\S+)', re.IGNORECASE),
            'autonomous_deploy': re.compile(r'\[AUTONOMOUS\].*(?:Deployed|Applied)\s+(?:patch|fix)\s+to\s+(\S+)', re.IGNORECASE),
            'autonomous_detect': re.compile(r'\[AUTONOMOUS\].*(?:Detected|Found)\s+(\d+)\s+(?:threat|issue|bug)', re.IGNORECASE),

            # Memory operations
            'memory_pin': re.compile(r'Pinned:\s*(.+)'),
            'memory_learn': re.compile(r'Learned:\s*(.+)'),

            # Artifacts
            'artifact_created': re.compile(r'Created artifact:\s*(.+\.(?:md|json|txt))'),
            'artifact_saved': re.compile(r'Saved (?:report|analysis|plan):\s*(.+)'),

            # Upgrades
            'upgrade_ready': re.compile(r'Upgrade ready:\s*(.+)'),
            'upgrade_applied': re.compile(r'Applied upgrade:\s*(.+)'),

            # Learning
            'learned_preference': re.compile(r'Learned:\s*(.+)'),
            'tool_adjusted': re.compile(r'Tool adjustment:\s*(.+)'),
        }

    def parse_line(self, line: str) -> List[Dict[str, Any]]:
        """Parse a single line of output and return events.

        Args:
            line: Terminal output line

        Returns:
            List of event dictionaries
        """
        events = []

        # Health/Signal updates
        if match := self.patterns['voice_ready'].search(line):
            events.append({
                'type': 'signal_update',
                'data': {'voice': 'online'}
            })
        elif match := self.patterns['voice_offline'].search(line):
            events.append({
                'type': 'signal_update',
                'data': {'voice': 'offline'}
            })

        if match := self.patterns['network_online'].search(line):
            events.append({
                'type': 'signal_update',
                'data': {'network': 'online'}
            })
        elif match := self.patterns['network_offline'].search(line):
            events.append({
                'type': 'signal_update',
                'data': {'network': 'offline'}
            })

        if match := self.patterns['coverage'].search(line):
            coverage = int(match.group(1))
            events.append({
                'type': 'health_update',
                'data': {'coverage': coverage}
            })

        if match := self.patterns['errors'].search(line):
            errors = int(match.group(1))
            events.append({
                'type': 'signal_update',
                'data': {'errors': errors}
            })

        if match := self.patterns['memory_count'].search(line):
            count = int(match.group(1))
            events.append({
                'type': 'memory_update',
                'data': {'count': count}
            })

        # Autonomous actions
        if match := self.patterns['autonomous_fix'].search(line):
            file = match.group(1)
            events.append({
                'type': 'autonomous_action',
                'data': {
                    'action': 'bug_fixed',
                    'file': file,
                    'description': f'Fixed bug in {file}'
                }
            })
            # Also increment auto-fixes counter
            events.append({
                'type': 'immune_update',
                'data': {'increment_fixes': True}
            })

        if match := self.patterns['autonomous_deploy'].search(line):
            target = match.group(1)
            events.append({
                'type': 'autonomous_action',
                'data': {
                    'action': 'deployed',
                    'target': target,
                    'description': f'Deployed to {target}'
                }
            })

        if match := self.patterns['autonomous_detect'].search(line):
            count = int(match.group(1))
            events.append({
                'type': 'immune_update',
                'data': {'threats': count}
            })

        # Memory operations
        if match := self.patterns['memory_pin'].search(line):
            fact = match.group(1).strip()
            events.append({
                'type': 'memory_update',
                'data': {'pinned': [fact]}
            })

        if match := self.patterns['memory_learn'].search(line):
            fact = match.group(1).strip()
            events.append({
                'type': 'memory_update',
                'data': {'recent': [fact]}
            })

        # Artifacts
        if match := self.patterns['artifact_created'].search(line):
            name = match.group(1)
            events.append({
                'type': 'artifact_created',
                'data': {'name': name}
            })

        if match := self.patterns['artifact_saved'].search(line):
            name = match.group(1)
            events.append({
                'type': 'artifact_created',
                'data': {'name': name}
            })

        # Upgrades
        if match := self.patterns['upgrade_ready'].search(line):
            title = match.group(1).strip()
            events.append({
                'type': 'upgrade_ready',
                'data': {'title': title}
            })

        if match := self.patterns['upgrade_applied'].search(line):
            title = match.group(1).strip()
            events.append({
                'type': 'upgrade_applied',
                'data': {'title': title}
            })

        # Learning
        if match := self.patterns['learned_preference'].search(line):
            event_text = match.group(1).strip()
            events.append({
                'type': 'learning_event',
                'data': {'event': event_text}
            })

        if match := self.patterns['tool_adjusted'].search(line):
            event_text = match.group(1).strip()
            events.append({
                'type': 'learning_event',
                'data': {'event': event_text}
            })

        return events

    def parse_buffer(self, buffer: str) -> List[Dict[str, Any]]:
        """Parse multiple lines of output.

        Args:
            buffer: Terminal output buffer

        Returns:
            List of event dictionaries
        """
        events = []
        for line in buffer.split('\n'):
            line = line.strip()
            if line:
                events.extend(self.parse_line(line))
        return events


# Convenience function
def parse_output(text: str) -> List[Dict[str, Any]]:
    """Parse terminal output and return events.

    Args:
        text: Terminal output text

    Returns:
        List of event dictionaries
    """
    parser = OutputParser()
    return parser.parse_buffer(text)
