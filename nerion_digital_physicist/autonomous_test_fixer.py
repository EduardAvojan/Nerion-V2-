"""
Universal Autonomous Fixer 🔧
Real autonomous system that:
1. Runs python scripts OR pytest → detects actual failures
2. Reasons about the fix using Chain-of-Thought
3. Uses Gemini/Claude to generate the fix
4. Modifies the real file
5. Verifies the fix works
6. Learns from the result
"""
import subprocess
import re
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from selfcoder.planner.explainable_planner import ExplainablePlanner
from nerion_digital_physicist.training.online_learner import OnlineLearner
from nerion_digital_physicist.agent.data import create_graph_data_object
from nerion_digital_physicist.agent.semantics import get_global_embedder
import torch

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UniversalFixer")

class ExecutionError:
    def __init__(self, file_path: str, line_num: int, error_msg: str, traceback: str, error_type: str = "Unknown"):
        self.file_path = file_path
        self.line_num = line_num
        self.error_msg = error_msg
        self.traceback = traceback
        self.error_type = error_type

class UniversalFixer:
    def __init__(self, enable_learning: bool = True, dry_run: bool = False):
        self.planner = ExplainablePlanner(min_confidence_for_execution=0.7)
        self.fixes_applied = []
        self.enable_learning = enable_learning
        self.dry_run = dry_run
        self.embedder = get_global_embedder()
        
        # Initialize the learning system
        if enable_learning:
            try:
                self.learner = OnlineLearner()
                self.learning_examples = []  # Store (graph, error, fix) tuples
                self.model = self._load_or_create_model()
                self.error_to_label = {  # Map error types to class labels
                    'attribute_error': 0,
                    'type_error': 1,
                    'assertion_error': 2,
                    'import_error': 3,
                    'other': 4
                }
                logger.info("🧠 Learning mode ENABLED - Will learn from successful fixes")
            except Exception as e:
                logger.warning(f"Could not initialize OnlineLearner: {e}")
                self.enable_learning = False
        
    def detect_errors(self, target_path: str) -> List[ExecutionError]:
        """Step 1: Run the target (script or test) and detect failures"""
        target_path_obj = Path(target_path)
        is_test = target_path.endswith(".py") and ("test_" in target_path or "_test" in target_path)
        
        if is_test:
            return self._run_pytest(target_path)
        else:
            return self._run_python_script(target_path)

    def _run_pytest(self, test_path: str) -> List[ExecutionError]:
        logger.info(f"🔍 Running pytest on {test_path}")
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return self._parse_pytest_output(result.stdout + result.stderr)

    def _run_python_script(self, script_path: str) -> List[ExecutionError]:
        logger.info(f"🔍 Running python script: {script_path}")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return []
            
        return self._parse_python_traceback(result.stderr, script_path)

    def _parse_python_traceback(self, stderr: str, script_path: str) -> List[ExecutionError]:
        errors = []
        lines = stderr.split('\n')
        
        # Basic traceback parsing
        traceback_lines = []
        error_msg = "Unknown Error"
        error_type = "Unknown"
        file_path = script_path
        line_num = 0
        
        in_traceback = False
        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                traceback_lines = [line]
                continue
            
            if in_traceback:
                traceback_lines.append(line)
                # Look for file path and line number in "File "...", line X, in ..."
                if line.strip().startswith("File "):
                    parts = line.split('"')
                    if len(parts) >= 2:
                        fpath = parts[1]
                        # Only care about the target script or files in the project, not system libs
                        if fpath.endswith(Path(script_path).name) or os.getcwd() in fpath:
                            file_path = fpath
                            # Extract line number
                            line_match = re.search(r"line (\d+)", line)
                            if line_match:
                                line_num = int(line_match.group(1))
                
                # The last line usually contains the Error Type and Message
                if not line.startswith(" ") and ":" in line:
                    error_type, error_msg = line.split(":", 1)
                    error_type = error_type.strip()
                    error_msg = error_msg.strip()
        
        if in_traceback:
            errors.append(ExecutionError(
                file_path=file_path,
                line_num=line_num,
                error_msg=f"{error_type}: {error_msg}",
                traceback="\n".join(traceback_lines),
                error_type=error_type
            ))
            
        return errors

    def _parse_pytest_output(self, output: str) -> List[ExecutionError]:
        failures = []
        summary_start = output.find("short test summary")
        if summary_start < 0:
            return failures
            
        summary_section = output[summary_start:]
        for line in summary_section.split("\n"):
            if line.startswith("FAILED") or line.startswith("ERROR"):
                # Line format: FAILED file::test - error msg
                parts = line.split()
                if len(parts) < 2: continue
                
                nodeid = parts[1]
                if "::" not in nodeid: continue
                
                test_file, test_name = nodeid.split("::", 1)
                
                # Find failure section using just the test name (without parameters or error msg)
                # Test name might have parameters like test_add[1-2], so we search for the base name
                # But the header usually contains the full parameterized name.
                # Let's try to find the test name surrounded by spaces in the header lines
                
                # Search for "____ test_name ____" pattern
                # Pytest headers are like ________________________________ test_name _________________________________
                
                # Simple search for the test name might match other things, but usually it's unique enough
                failure_start = output.find(f" {test_name} ")
                
                if failure_start > 0:
                    next_failure = output.find("\n____", failure_start + 10)
                    if next_failure < 0: next_failure = output.find("\n====", failure_start)
                    
                    # If end not found, take the rest (or up to summary)
                    if next_failure < 0: next_failure = summary_start
                    
                    failure_section = output[failure_start:next_failure]
                    
                    error_lines = []
                    line_num = 0
                    for fline in failure_section.split("\n"):
                        if test_file in fline and ":" in fline:
                            parts = fline.split(":")
                            # Look for line number after file path
                            # path/to/file.py:123: ...
                            for i, part in enumerate(parts):
                                if part.strip().endswith(test_file) and i + 1 < len(parts):
                                    possible_line = parts[i+1].strip()
                                    if possible_line.isdigit():
                                        line_num = int(possible_line)
                                        break
                                        
                        if fline.strip().startswith("E "):
                            error_lines.append(fline.strip()[2:])
                    
                    error_msg = " ".join(error_lines) if error_lines else "Unknown error"
                    
                    failures.append(ExecutionError(
                        file_path=test_file,
                        line_num=line_num,
                        error_msg=error_msg,
                        traceback=failure_section,
                        error_type="AssertionError"
                    ))
        return failures
    
    def reason_about_fix(self, error: ExecutionError) -> Optional[Dict[str, Any]]:
        """Step 2: Use Chain-of-Thought to reason about the fix"""
        logger.info(f"🧠 Reasoning about error in: {error.file_path}")
        
        file_path = Path(error.file_path)
        if not file_path.is_absolute():
            file_path = Path(os.getcwd()) / file_path
            
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        code = file_path.read_text()
        
        context = {
            "file": str(file_path),
            "error": error.error_msg,
            "traceback": error.traceback,
            "code_snippet": code[max(0, error.line_num-10):error.line_num+10] if error.line_num > 0 else "",
        }
        
        task_description = f"""Python execution failure in {error.file_path}:
Error: {error.error_msg}

The script is failing. Analyze the error and propose a fix."""
        
        plan = self.planner.create_plan(task_description, context)
        
        logger.info(f"Plan confidence: {plan.reasoning.overall_confidence:.2f}")
        
        if not plan.reasoning.execution_approved:
            logger.warning("⚠️ Fix requires human review")
            return None
            
        return {
            "plan": plan,
            "context": context
        }
    
    def apply_fix(self, error: ExecutionError, fix_plan: Dict[str, Any]) -> bool:
        """Step 3: Use Claude to generate and apply the actual fix"""
        if self.dry_run:
            logger.info(f"🛑 DRY RUN: Would apply fix to {error.file_path}")
            return True

        logger.info(f"🔧 Using Claude to generate fix for {error.file_path}")
        
        try:
            from anthropic import Anthropic
            import os
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found")
                return False
            
            client = Anthropic(api_key=api_key)
            
            file_path = Path(error.file_path)
            if not file_path.is_absolute():
                file_path = Path(os.getcwd()) / file_path
                
            code = file_path.read_text()
            
            prompt = f"""You are an expert Python developer fixing a bug.

File: {error.file_path}
Error: {error.error_msg}

Full Traceback:
{error.traceback}

Current Code:
```python
{code}
```

Task: Fix the code to resolve the error.
Analyze the traceback to find the root cause.
Return ONLY the corrected Python code for the ENTIRE file. No explanations, no markdown."""

            logger.info("🤖 Asking Claude to generate fix...")
            
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            fixed_code = message.content[0].text
            
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
            
            file_path.write_text(fixed_code)
            logger.info(f"✅ Applied Claude's fix to {error.file_path}")
            
            self._store_fix_example(file_path, code, fixed_code, error)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating fix: {e}")
            return False
    
    def verify_fix(self, target_path: str) -> bool:
        """Step 4: Re-run to verify"""
        logger.info(f"✅ Verifying fix...")
        errors = self.detect_errors(target_path)
        return len(errors) == 0
    
    def _store_fix_example(self, file_path: Path, original_code: str, fixed_code: str, error: ExecutionError):
        if not self.enable_learning: return
        try:
            graph_data = create_graph_data_object(file_path, embedder=self.embedder)
            example = {
                'graph': graph_data,
                'error_type': self._categorize_error(error.error_msg),
                'original_code': original_code,
                'fixed_code': fixed_code,
                'file_path': str(file_path)
            }
            self.learning_examples.append(example)
            logger.info(f"💾 Stored learning example (total: {len(self.learning_examples)})")
        except Exception as e:
            logger.warning(f"Could not create learning example: {e}")

    def _categorize_error(self, error_msg: str) -> str:
        if "AttributeError" in error_msg: return "attribute_error"
        elif "TypeError" in error_msg: return "type_error"
        elif "AssertionError" in error_msg: return "assertion_error"
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg: return "import_error"
        else: return "other"

    def learn_from_fixes(self):
        """Step 5: Learn from all successful fixes"""
        if not self.enable_learning or not self.learning_examples: return
        
        logger.info(f"♾️  Learning from {len(self.learning_examples)} successful fixes")
        try:
            training_data = []
            for example in self.learning_examples:
                graph = example['graph']
                error_type = example['error_type']
                label = self.error_to_label.get(error_type, 4)
                training_data.append((graph, label))
            
            updated_model, update_info = self.learner.incremental_update(
                current_model=self.model,
                new_data=training_data
            )
            self.model = updated_model
            self._save_model()
            logger.info(f"✅ GNN updated: accuracy={update_info.new_accuracy:.2%}")
        except Exception as e:
            logger.error(f"Error during learning: {e}")

    def _load_or_create_model(self):
        from nerion_digital_physicist.agent.brain import CodeGraphGCN
        import os
        model_path = "/Users/ed/Nerion-V2/models/autonomous_fixer_model.pt"
        if os.path.exists(model_path):
            try:
                model = CodeGraphGCN(num_node_features=800, hidden_channels=256, num_classes=5, num_layers=4)
                model.load_state_dict(torch.load(model_path))
                return model
            except: pass
        return CodeGraphGCN(num_node_features=800, hidden_channels=256, num_classes=5, num_layers=4)

    def _save_model(self):
        import os
        model_path = "/Users/ed/Nerion-V2/models/autonomous_fixer_model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def run(self, target_path: str):
        logger.info("=" * 60)
        logger.info(f"🤖 UNIVERSAL FIXER - Target: {target_path}")
        logger.info("=" * 60)
        
        max_iterations = 3
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'🔄' if iteration > 1 else '▶️'} Iteration {iteration}/{max_iterations}")
            
            errors = self.detect_errors(target_path)
            
            if not errors:
                logger.info("✅ Execution successful / No errors found!")
                if iteration > 1: self.learn_from_fixes()
                return True
            
            logger.info(f"\n📊 Found {len(errors)} errors")
            
            fixes_applied = 0
            for error in errors:
                logger.info(f"\n{'─'*60}")
                logger.info(f"Processing Error: {error.error_msg}")
                logger.info(f"Location: {error.file_path}:{error.line_num}")
                
                fix_plan = self.reason_about_fix(error)
                if fix_plan:
                    if self.apply_fix(error, fix_plan):
                        fixes_applied += 1
            
            if fixes_applied > 0:
                if self.dry_run:
                    logger.info("🛑 Dry run complete. Exiting.")
                    return True
                
                if self.verify_fix(target_path):
                    logger.info("\n🎉 SUCCESS! Fixed all errors.")
                    self.learn_from_fixes()
                    return True
            else:
                logger.warning("⚠️  No fixes applied. Stopping.")
                return False
        
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Universal Autonomous Fixer")
    parser.add_argument("target", help="Path to script or test file to fix")
    parser.add_argument("--dry-run", action="store_true", help="Propose fixes without applying them")
    args = parser.parse_args()
    
    fixer = UniversalFixer(dry_run=args.dry_run)
    fixer.run(args.target)

if __name__ == "__main__":
    main()
