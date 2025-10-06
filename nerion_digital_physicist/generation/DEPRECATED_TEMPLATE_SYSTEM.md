# DEPRECATED: Template-Based Generation System

## Status: DEPRECATED
This template-based generation system has been deprecated in favor of the LLM-based generators.

## Files Being Deprecated
- `service.py` - CLI harness for Phase 3 tasks
- `builder.py` - TaskBuilder with template factories  
- `worker.py` - Worker harness for queue processing
- `orchestrator.py` - Orchestrates generation workers
- `queue_manager.py` - Queue management
- `sampler.py` - Template sampling
- `templates/` - Jinja2 templates directory

## Reason for Deprecation
The template-based system was designed for Phase 3 tasks but has been superseded by the LLM-based generation system which:

1. **Integrates with Vertex AI** - Dynamic content generation
2. **Supports validation** - GNN-based structural analysis
3. **Has repair mechanisms** - Automatic lesson repair on failure
4. **Aligns with architecture** - Better integration with Digital Physicist

## Migration Path
All generation should now use the LLM-based generators:
- `curriculum_generator.py` - Main generator
- `bug_fix_generator.py` - Bug-fixing lessons
- `feature_generator.py` - Feature implementation lessons
- `performance_generator.py` - Performance optimization lessons
- `explanation_generator.py` - Code explanation lessons

## Timeline
- **Immediate**: Mark as deprecated
- **Next Release**: Remove from active codebase
- **Future**: Archive in separate branch if needed for reference

## Contact
For questions about this deprecation, contact the Digital Physicist team.
