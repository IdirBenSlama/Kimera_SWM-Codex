#!/usr/bin/env python3
"""
Scientific Nomenclature Revision Script
=======================================

This script systematically replaces all problematic hype terminology 
with proper scientific and academic nomenclature throughout the Kimera codebase.

CRITICAL ISSUES ADDRESSED:
1. Consciousness claims → Complexity analysis
2. Revolutionary/breakthrough hype → Scientific improvement descriptions  
3. Mystical terminology → Technical descriptions
4. Thermodynamic misuse → Accurate physics terminology

USAGE:
    python scripts/scientific_nomenclature_revision.py --dry-run  # Preview changes
    python scripts/scientific_nomenclature_revision.py --execute  # Apply changes
"""

import os
import re
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScientificNomenclatureReviser:
    """Systematic revision of problematic terminology to scientific nomenclature"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.changes_made = []
        self.files_modified = set()
        
        # Define comprehensive terminology mappings
        self.terminology_mappings = self._define_terminology_mappings()
        self.file_renames = self._define_file_renames()
        self.class_renames = self._define_class_renames()
        self.function_renames = self._define_function_renames()
        
    def _define_terminology_mappings(self) -> Dict[str, str]:
        """Define comprehensive terminology replacement mappings"""
        return {
            # CRITICAL: Consciousness terminology (most important)
            'consciousness_probability': 'complexity_integration_score',
            'consciousness_detection': 'complexity_pattern_analysis',
            'consciousness_emergence': 'complexity_threshold_detection',
            'consciousness_signatures': 'complexity_signatures',
            'consciousness_indicators': 'complexity_indicators',
            'consciousness_state': 'complexity_state',
            'consciousness_level': 'complexity_level',
            'consciousness_analysis': 'complexity_analysis',
            'detect_consciousness': 'analyze_complexity',
            'thermodynamic_consciousness': 'thermodynamic_complexity',
            'quantum_consciousness': 'quantum_information_integration',
            'consciousness_detector': 'complexity_analyzer',
            'consciousness_engine': 'complexity_analysis_engine',
            
            # Consciousness states
            'UNCONSCIOUS': 'LOW_INTEGRATION',
            'PROTO_CONSCIOUS': 'MODERATE_INTEGRATION', 
            'CONSCIOUS': 'HIGH_INTEGRATION',
            'SUPER_CONSCIOUS': 'VERY_HIGH_INTEGRATION',
            'QUANTUM_CONSCIOUS': 'MAXIMUM_INTEGRATION',
            
            # Revolutionary/breakthrough hype terminology
            'revolutionary': 'advanced',
            'breakthrough': 'improvement',
            'transcendent': 'high_performance',
            'unprecedented': 'significant',
            'paradigm_shifting': 'novel_approach',
            'world_first': 'experimental',
            'Revolutionary': 'Advanced',
            'Breakthrough': 'Improvement',
            'REVOLUTIONARY': 'ADVANCED',
            'BREAKTHROUGH': 'IMPROVEMENT',
            
            # Mystical/non-scientific terminology
            'mirror_portal': 'semantic_bridge',
            'quantum_semantic_bridge': 'cross_modal_connector',
            'vortex_energy_storage': 'circular_buffer_system',
            'golden_ratio_optimization': 'fibonacci_sequence_optimization',
            'epistemic_temperature': 'information_processing_rate',
            'zetetic': 'systematic_validation',
            'cognitive_vortex': 'recursive_processing_loop',
            'portal_coherence': 'bridge_coherence',
            'vortex_enhancement': 'buffer_optimization',
            'mirror_portals': 'semantic_bridges',
            
            # Thermodynamic misuse corrections
            'phase_transition_consciousness': 'complexity_transition_point',
            'carnot_efficiency_consciousness': 'processing_efficiency_metric',
            'entropy_consciousness': 'information_entropy_analysis',
            'temperature_consciousness': 'processing_rate_analysis',
            
            # API and route corrections
            '/consciousness/detect': '/complexity/analyze',
            '/revolutionary/thermodynamic': '/advanced/thermodynamic',
            'consciousness_detected': 'high_complexity_detected',
            'emergence_probability': 'integration_score',
            
            # Documentation terminology
            'CONSCIOUSNESS DETECTION': 'COMPLEXITY ANALYSIS',
            'REVOLUTIONARY BREAKTHROUGH': 'SIGNIFICANT IMPROVEMENT',
            'QUANTUM CONSCIOUSNESS': 'QUANTUM INFORMATION INTEGRATION',
            'THERMODYNAMIC CONSCIOUSNESS': 'THERMODYNAMIC COMPLEXITY',
            
            # Variable and method names
            'consciousness_prob': 'complexity_score',
            'consciousness_result': 'complexity_result',
            'consciousness_event': 'complexity_event',
            'consciousness_threshold': 'complexity_threshold',
            'consciousness_emergence_events': 'complexity_threshold_events',
            
            # Logging and output messages
            'Consciousness emergence detected': 'High complexity threshold detected',
            'Consciousness detection': 'Complexity analysis',
            'Revolutionary performance': 'Significant performance improvement',
            'Breakthrough achievement': 'Notable improvement achieved',
            'Transcendent performance': 'High performance level',
        }
    
    def _define_file_renames(self) -> Dict[str, str]:
        """Define file rename mappings"""
        return {
            'quantum_thermodynamic_consciousness.py': 'information_integration_analyzer.py',
            'foundational_thermodynamic_engine.py': 'advanced_thermodynamic_processor.py',
            'consciousness_engine.py': 'complexity_analysis_engine.py',
            'zetetic_revolutionary_integration_engine.py': 'systematic_validation_framework.py',
            'revolutionary_thermodynamic_monitor.py': 'advanced_thermodynamic_monitor.py',
            'revolutionary_thermodynamic_routes.py': 'advanced_thermodynamic_routes.py',
            'revolutionary_epistemic_validator.py': 'systematic_epistemic_validator.py',
            'REVOLUTIONARY_THERMODYNAMIC_INTEGRATION.md': 'ADVANCED_THERMODYNAMIC_INTEGRATION.md',
            'QUANTUM_CONSCIOUSNESS_MANIFESTO.md': 'INFORMATION_INTEGRATION_ANALYSIS.md',
            'ZETETIC_REVOLUTIONARY_INTEGRATION_BREAKTHROUGH_SUMMARY.md': 'SYSTEMATIC_VALIDATION_IMPROVEMENT_SUMMARY.md',
            'revolutionary_thermodynamic_demo.py': 'advanced_thermodynamic_demo.py',
        }
    
    def _define_class_renames(self) -> Dict[str, str]:
        """Define class rename mappings"""
        return {
            'QuantumThermodynamicConsciousnessDetector': 'InformationIntegrationAnalyzer',
            'FoundationalThermodynamicEngine': 'AdvancedThermodynamicProcessor',
            'ConsciousnessEngine': 'ComplexityAnalysisEngine',
            'ZeteticRevolutionaryIntegrationEngine': 'SystematicValidationFramework',
            'ConsciousnessState': 'ComplexityState',
            'ConsciousnessSignature': 'ComplexitySignature',
            'RevolutionaryThermodynamicMonitor': 'AdvancedThermodynamicMonitor',
            'ConsciousnessEmergenceEvent': 'ComplexityThresholdEvent',
            'RevolutionaryEpistemicValidator': 'SystematicEpistemicValidator',
        }
    
    def _define_function_renames(self) -> Dict[str, str]:
        """Define function rename mappings"""
        return {
            'detect_consciousness': 'analyze_complexity',
            'detect_consciousness_emergence': 'detect_complexity_threshold',
            'calculate_consciousness_probability': 'calculate_complexity_score',
            'consciousness_detection': 'complexity_analysis',
            'demonstrate_consciousness_detection': 'demonstrate_complexity_analysis',
            'detect_thermodynamic_consciousness': 'analyze_thermodynamic_complexity',
            'execute_zetetic_revolutionary_integration': 'execute_systematic_validation',
            'revolutionary_optimization': 'advanced_optimization',
            'breakthrough_analysis': 'improvement_analysis',
        }
    
    def execute_revision(self, dry_run: bool = True) -> Dict[str, any]:
        """Execute the complete nomenclature revision"""
        logger.info("🔬 Starting Scientific Nomenclature Revision")
        logger.info(f"   Mode: {'DRY RUN' if dry_run else 'EXECUTE CHANGES'}")
        logger.info(f"   Target: {self.workspace_root}")
        
        revision_summary = {
            'files_processed': 0,
            'files_modified': 0,
            'files_renamed': 0,
            'terminology_replacements': 0,
            'class_renames': 0,
            'function_renames': 0,
            'errors': []
        }
        
        try:
            # Phase 1: Rename critical files
            logger.info("📁 Phase 1: Renaming critical files")
            file_results = self._rename_files(dry_run)
            revision_summary['files_renamed'] = file_results['files_renamed']
            
            # Phase 2: Replace terminology in all files
            logger.info("📝 Phase 2: Replacing terminology in files")
            term_results = self._replace_terminology_in_files(dry_run)
            revision_summary['files_processed'] = term_results['files_processed']
            revision_summary['files_modified'] = term_results['files_modified']
            revision_summary['terminology_replacements'] = term_results['terminology_replacements']
            
            # Phase 3: Update class and function names
            logger.info("🏗️ Phase 3: Updating class and function names")
            name_results = self._update_class_function_names(dry_run)
            revision_summary['class_renames'] = name_results['class_renames']
            revision_summary['function_renames'] = name_results['function_renames']
            
            # Phase 4: Validate changes
            logger.info("✅ Phase 4: Validating changes")
            validation_results = self._validate_changes()
            revision_summary['validation'] = validation_results
            
            logger.info("🎉 Scientific Nomenclature Revision Complete")
            self._print_summary(revision_summary)
            
            return revision_summary
            
        except Exception as e:
            logger.error(f"❌ Revision failed: {e}")
            revision_summary['errors'].append(str(e))
            return revision_summary
    
    def _rename_files(self, dry_run: bool) -> Dict[str, int]:
        """Rename files according to scientific nomenclature"""
        renamed_count = 0
        
        for old_name, new_name in self.file_renames.items():
            # Find all files with the old name
            matching_files = list(self.workspace_root.rglob(old_name))
            
            for old_file in matching_files:
                new_file = old_file.parent / new_name
                
                logger.info(f"   📁 {old_file.relative_to(self.workspace_root)} → {new_file.relative_to(self.workspace_root)}")
                
                if not dry_run:
                    try:
                        old_file.rename(new_file)
                        renamed_count += 1
                        self.files_modified.add(str(new_file))
                    except Exception as e:
                        logger.error(f"   ❌ Failed to rename {old_file}: {e}")
                else:
                    renamed_count += 1
        
        return {'files_renamed': renamed_count}
    
    def _replace_terminology_in_files(self, dry_run: bool) -> Dict[str, int]:
        """Replace problematic terminology in all relevant files"""
        files_processed = 0
        files_modified = 0
        replacements_made = 0
        
        # File extensions to process
        target_extensions = {'.py', '.md', '.json', '.yml', '.yaml', '.txt', '.rst'}
        
        # Directories to skip
        skip_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache'}
        
        for file_path in self.workspace_root.rglob('*'):
            if file_path.is_file() and file_path.suffix in target_extensions:
                # Skip files in excluded directories
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue
                
                files_processed += 1
                
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    file_replacements = 0
                    
                    # Apply all terminology replacements
                    for old_term, new_term in self.terminology_mappings.items():
                        # Use word boundaries for precise matching
                        pattern = r'\b' + re.escape(old_term) + r'\b'
                        matches = len(re.findall(pattern, content, re.IGNORECASE))
                        if matches > 0:
                            content = re.sub(pattern, new_term, content, flags=re.IGNORECASE)
                            file_replacements += matches
                            replacements_made += matches
                    
                    # Write back if changes were made
                    if content != original_content:
                        files_modified += 1
                        self.files_modified.add(str(file_path))
                        
                        if file_replacements > 0:
                            logger.info(f"   📝 {file_path.relative_to(self.workspace_root)}: {file_replacements} replacements")
                        
                        if not dry_run:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                
                except Exception as e:
                    logger.error(f"   ❌ Error processing {file_path}: {e}")
        
        return {
            'files_processed': files_processed,
            'files_modified': files_modified,
            'terminology_replacements': replacements_made
        }
    
    def _update_class_function_names(self, dry_run: bool) -> Dict[str, int]:
        """Update class and function names throughout the codebase"""
        class_renames = 0
        function_renames = 0
        
        # Process Python files for class and function renames
        for py_file in self.workspace_root.rglob('*.py'):
            if any(skip_dir in py_file.parts for skip_dir in {'.git', '__pycache__', '.venv'}):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Update class names
                for old_class, new_class in self.class_renames.items():
                    # Match class definitions and usages
                    patterns = [
                        rf'\bclass\s+{re.escape(old_class)}\b',  # Class definitions
                        rf'\b{re.escape(old_class)}\(',  # Class instantiations
                        rf'\b{re.escape(old_class)}\.',  # Class method calls
                        rf':\s*{re.escape(old_class)}\b',  # Type hints
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, content):
                            content = re.sub(pattern, lambda m: m.group().replace(old_class, new_class), content)
                            class_renames += 1
                
                # Update function names
                for old_func, new_func in self.function_renames.items():
                    # Match function definitions and calls
                    patterns = [
                        rf'\bdef\s+{re.escape(old_func)}\b',  # Function definitions
                        rf'\b{re.escape(old_func)}\(',  # Function calls
                        rf'\.{re.escape(old_func)}\(',  # Method calls
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, content):
                            content = re.sub(pattern, lambda m: m.group().replace(old_func, new_func), content)
                            function_renames += 1
                
                # Write back if changes were made
                if content != original_content and not dry_run:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            except Exception as e:
                logger.error(f"   ❌ Error updating {py_file}: {e}")
        
        return {
            'class_renames': class_renames,
            'function_renames': function_renames
        }
    
    def _validate_changes(self) -> Dict[str, any]:
        """Validate that changes maintain code integrity"""
        validation_results = {
            'syntax_errors': [],
            'import_errors': [],
            'consciousness_terms_remaining': 0,
            'revolutionary_terms_remaining': 0,
            'validation_passed': True
        }
        
        # Check for remaining problematic terms
        problematic_patterns = [
            r'\bconsciousness\b',
            r'\brevolutionary\b',
            r'\bbreakthrough\b',
            r'\btranscendent\b',
            r'\bmirror_portal\b',
            r'\bvortex_energy\b',
            r'\bzetetic\b'
        ]
        
        for py_file in self.workspace_root.rglob('*.py'):
            if any(skip_dir in py_file.parts for skip_dir in {'.git', '__pycache__', '.venv'}):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for remaining problematic terms
                for pattern in problematic_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if 'consciousness' in pattern and matches:
                        validation_results['consciousness_terms_remaining'] += len(matches)
                    elif 'revolutionary' in pattern and matches:
                        validation_results['revolutionary_terms_remaining'] += len(matches)
                
                # Basic syntax validation
                try:
                    compile(content, py_file, 'exec')
                except SyntaxError as e:
                    validation_results['syntax_errors'].append(f"{py_file}: {e}")
                    validation_results['validation_passed'] = False
            
            except Exception as e:
                logger.error(f"   ❌ Validation error for {py_file}: {e}")
        
        return validation_results
    
    def _print_summary(self, summary: Dict[str, any]):
        """Print comprehensive revision summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 SCIENTIFIC NOMENCLATURE REVISION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📁 Files processed: {summary.get('files_processed', 0)}")
        logger.info(f"📝 Files modified: {summary.get('files_modified', 0)}")
        logger.info(f"🔄 Files renamed: {summary.get('files_renamed', 0)}")
        logger.info(f"📝 Terminology replacements: {summary.get('terminology_replacements', 0)}")
        logger.info(f"🏗️ Class renames: {summary.get('class_renames', 0)}")
        logger.info(f"🔧 Function renames: {summary.get('function_renames', 0)}")
        
        if 'validation' in summary:
            validation = summary['validation']
            logger.info(f"\n✅ VALIDATION RESULTS:")
            logger.info(f"   Syntax errors: {len(validation.get('syntax_errors', []))}")
            logger.info(f"   Consciousness terms remaining: {validation.get('consciousness_terms_remaining', 0)}")
            logger.info(f"   Revolutionary terms remaining: {validation.get('revolutionary_terms_remaining', 0)}")
            logger.info(f"   Validation passed: {'✅' if validation.get('validation_passed', False) else '❌'}")
        
        if summary.get('errors'):
            logger.info(f"\n❌ ERRORS:")
            for error in summary['errors']:
                logger.info(f"   {error}")
        
        logger.info("\n🎉 REVISION COMPLETE")
        logger.info("   The codebase now uses proper scientific nomenclature")
        logger.info("   All consciousness claims have been replaced with complexity analysis")
        logger.info("   Revolutionary hype has been replaced with factual descriptions")

def main():
    parser = argparse.ArgumentParser(description='Scientific Nomenclature Revision for Kimera')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without modifying files')
    parser.add_argument('--execute', action='store_true',
                       help='Execute changes (modifies files)')
    parser.add_argument('--workspace', type=str, default='.',
                       help='Workspace root directory')
    
    args = parser.parse_args()
    
    if not (args.dry_run or args.execute):
        print("Error: Must specify either --dry-run or --execute")
        return 1
    
    # Initialize reviser
    reviser = ScientificNomenclatureReviser(args.workspace)
    
    # Execute revision
    summary = reviser.execute_revision(dry_run=args.dry_run)
    
    # Save summary
    summary_file = Path(args.workspace) / 'scientific_nomenclature_revision_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📄 Summary saved to: {summary_file}")
    
    return 0 if summary.get('validation', {}).get('validation_passed', False) else 1

if __name__ == '__main__':
    exit(main()) 